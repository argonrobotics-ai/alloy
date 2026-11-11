# SPDX-License-Identifier: Apache-2.0
"""Optional TensorRT fp16 inference for the denoiser (same ``denoise`` API as :class:`Denoiser`).

    from alloy.camera.denoiser.trt import load_trt
    dn = load_trt("default")                                # needs: pip install tensorrt-cu12 (and a CUDA GPU)
    out = dn.denoise(rgb, albedo, normal, spp=1, tta=2)     # numpy or torch in, same type out

Split of the network: the engine runs the U-Net trunk and the KPN head convolutions and returns the raw head
tensors (full-res kernels + alpha, 1/2-res, 1/4-res and, for "kpnd", the direct RGB + blend logit). The kernel
application and coarse-to-fine combine stay in torch float32, exactly as in the torch path. One engine is built
per input shape (N, 10, Hpad, Wpad) on first use and cached on disk, keyed by weights sha256, shape and
TensorRT version.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ._torch import F, nn, torch
from .infer import Denoiser
from .model import KPN_MODES, KPNHeads, UNet, make_input, to_linear
from .weights import default_cache_dir, resolve_weights, sha256_file

log = logging.getLogger(__name__)

INSTALL_HINT = ("TensorRT is not installed. Run `pip install tensorrt-cu12`, or use "
                "`alloy.camera.denoiser.load` (torch fp16) instead.")
_TRT_TO_TORCH = {"HALF": torch.float16, "FLOAT": torch.float32, "BF16": torch.bfloat16}


def _import_tensorrt() -> Any:
    """Import tensorrt or raise an ImportError that says how to install it or what to use instead.

    Example: ``trt = _import_tensorrt(); trt.__version__``.
    """
    try:
        import tensorrt
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    return tensorrt


class Trunk(nn.Module):
    """UNet.forward up to the KPN head convolutions: raw head tensors, no pixel shuffle, no filtering.

    Example: ``Trunk(net)(torch.zeros(1, 10, 32, 32))`` is ``(a0, a1, a2[, ad])``.
    """

    def __init__(self, net: UNet) -> None:
        super().__init__()
        if net.cfg.out_mode not in KPN_MODES:
            raise ValueError(f"TensorRT path supports out_mode 'kpn' or 'kpnd', got {net.cfg.out_mode!r}; "
                             f"use alloy.camera.denoiser.load instead.")
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h, h1, _ = self.net.trunk(x)
        kp = self.net.kpn
        outs = [kp.h0(h), kp.h1(h), kp.h2(h1)]
        if kp.hd is not None:
            outs.append(kp.hd(h))
        return tuple(outs)


def kpn_filter(heads: tuple[torch.Tensor, ...], x: torch.Tensor, demod: bool) -> torch.Tensor:
    """Raw head tensors + the padded float32 network input -> log1p-space output (N, 3, H, W), float32.

    The math of ``KPNHeads.forward`` after the head convolutions (same ``_combine``, float32).
    Example: ``out = kpn_filter(trunk(x.half()), x, demod=False)``.
    """
    a0, a1, a2 = heads[:3]
    ad = heads[3] if len(heads) > 3 else None
    with torch.autocast(x.device.type, enabled=False):
        k0 = F.pixel_shuffle(a0.float(), 2)
        rgb = torch.expm1(x[:, :3]).float().clamp_min(0)  # as UNet.forward: the filtered image is read from x
        mod = x[:, 3:6].float().clamp_min(0) + 0.02 if demod else None
        c0 = KPNHeads._combine(k0, a1.float(), a2.float(), rgb / mod if demod else rgb)
        if mod is not None:
            c0 = c0 * mod
        if ad is not None:
            d = F.pixel_shuffle(ad.float(), 2)
            w = torch.sigmoid(d[:, 3:4] + 3.0)
            c0 = w * c0 + (1 - w) * torch.expm1(d[:, :3].clamp(0, 12))
        return torch.log1p(c0.clamp_min(0))


def state_sha256(net: UNet) -> str:
    """sha256 over the config and all weights of a network (cache key when no weights file is known).

    Example: ``state_sha256(net)`` is a 64-character hex string.
    """
    h = hashlib.sha256(repr(net.cfg).encode())
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def export_onnx(net: UNet, path: Path, shape: tuple[int, ...], opset: int = 17) -> None:
    """Export the fp16 trunk to ONNX at a fixed input shape (strongly typed: ONNX types set layer precision).

    Example: ``export_onnx(net, Path("trunk.onnx"), (1, 10, 64, 64))``.
    """
    tr = Trunk(copy.deepcopy(net)).cuda().eval().half()
    names = ["a0", "a1", "a2"] + (["ad"] if net.cfg.out_mode == "kpnd" else [])
    x = torch.zeros(shape, device="cuda", dtype=torch.float16)
    torch.onnx.export(tr, (x,), str(path), input_names=["x"], output_names=names, opset_version=opset,
                      dynamo=False)


def build_plan(onnx_path: Path, shape: tuple[int, ...]) -> bytes:
    """Build a serialized fp16 TensorRT engine from an ONNX file for exactly ``shape``.

    Example: ``plan = build_plan(Path("trunk.onnx"), (1, 10, 64, 64))``.
    """
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("TensorRT could not parse the ONNX export: "
                           + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.builder_optimization_level = 3
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise RuntimeError(f"TensorRT engine build failed for input shape {shape}.")
    return bytes(plan)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class _Engine:
    """A deserialized engine plus its persistent input/output buffers for one shape."""

    def __init__(self, trt: Any, plan: bytes, shape: tuple[int, ...]) -> None:
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.ctx.set_input_shape("x", shape)
        self.x = torch.empty(shape, dtype=torch.float16, device="cuda")
        self.ctx.set_tensor_address("x", self.x.data_ptr())
        self.outs: list[torch.Tensor] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            dt = _TRT_TO_TORCH[self.engine.get_tensor_dtype(name).name]
            self.outs.append(torch.empty(tuple(self.ctx.get_tensor_shape(name)), dtype=dt, device="cuda"))
            self.ctx.set_tensor_address(name, self.outs[-1].data_ptr())

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Copy x in (cast to fp16), execute on the current stream, return the persistent output buffers."""
        self.x.copy_(x)
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed.")
        return tuple(self.outs)


class TrtDenoiser(Denoiser):
    """:class:`Denoiser` whose network forward is a TensorRT fp16 engine (kpn / kpnd networks only).

    Example: ``dn = TrtDenoiser.from_denoiser(load("default")); out = dn.denoise(rgb, albedo, normal)``.
    """

    def __init__(self, net: UNet, weights_sha256: str | None = None, cache_dir: str | Path | None = None) -> None:
        self._trt = _import_tensorrt()
        if not torch.cuda.is_available():
            raise RuntimeError("TrtDenoiser needs a CUDA GPU; use alloy.camera.denoiser.load(..., device='cpu').")
        Trunk(net)  # validates out_mode
        super().__init__(net, device="cuda", dtype="fp32")
        self.sha256 = weights_sha256 or state_sha256(net)
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir() / "trt"
        self._engines: dict[tuple[int, ...], _Engine] = {}

    @classmethod
    def from_denoiser(cls, dn: Denoiser, cache_dir: str | Path | None = None,
                      weights_sha256: str | None = None) -> TrtDenoiser:
        """Build from an already loaded :class:`Denoiser` (its network is copied to the GPU).

        Example: ``TrtDenoiser.from_denoiser(Denoiser.load("dn.pt"), cache_dir="trt_cache")``.
        """
        return cls(copy.deepcopy(dn.net), weights_sha256, cache_dir)

    def plan_path(self, shape: tuple[int, ...]) -> Path:
        """Cache file of the engine for an input shape (N, 10, Hpad, Wpad).

        Example: ``dn.plan_path((1, 10, 64, 64)).name``.
        """
        n, c, h, w = shape
        return self.cache_dir / f"{self.sha256[:20]}_trt{self._trt.__version__}_{n}x{c}x{h}x{w}.plan"

    def _get_plan(self, shape: tuple[int, ...]) -> bytes:
        path = self.plan_path(shape)
        if path.is_file():
            return path.read_bytes()
        log.info("building TensorRT engine for shape %s (cached afterwards as %s)", shape, path)
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "trunk.onnx"
            export_onnx(self.net, onnx, shape)
            plan = build_plan(onnx, shape)
        _atomic_write(path, plan)
        log.info("built engine %s in %.1f s", path.name, time.perf_counter() - t0)
        return plan

    def _engine(self, shape: tuple[int, ...]) -> _Engine:
        if shape not in self._engines:
            self._engines[shape] = _Engine(self._trt, self._get_plan(shape), shape)
        return self._engines[shape]

    def _forward(self, rgb: torch.Tensor, albedo: torch.Tensor, normal: torch.Tensor,
                 spp: float | torch.Tensor) -> torch.Tensor:
        """One pass on (N,H,W,3) float32 tensors: pad, engine (raw heads), torch fp32 kernels, crop."""
        x = make_input(rgb, albedo, normal, spp)
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.stride, (-w) % self.stride
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        x = x.contiguous()
        heads = self._engine(tuple(x.shape)).run(x)
        out = kpn_filter(heads, x, self.net.cfg.demod)
        return to_linear(out[..., :h, :w])


def load_trt(name_or_path: str | Path, cache_dir: str | Path | None = None) -> TrtDenoiser:
    """Resolve weights ("default", a manifest name or a .pt path) and load them as a :class:`TrtDenoiser`.

    Example: ``dn = load_trt("default", cache_dir="trt_cache")``.
    """
    _import_tensorrt()
    path = resolve_weights(name_or_path)
    dn = Denoiser.load(path, device="cuda", dtype="fp32")
    if not isinstance(dn, Denoiser):  # pragma: no cover - Denoiser.load returns an Ensemble only for lists
        raise TypeError("load_trt takes a single weights file.")
    return TrtDenoiser.from_denoiser(dn, cache_dir, sha256_file(path))
