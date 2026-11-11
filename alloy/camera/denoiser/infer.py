# SPDX-License-Identifier: Apache-2.0
"""Inference API: :class:`Denoiser` (one network) and :class:`Ensemble` (averaged networks).

    from alloy.camera.denoiser import load
    dn = load("default")                                   # cuda fp16 if available, else cpu fp32
    out = dn.denoise(rgb, albedo, normal, spp=1)           # (..., H, W, 3) linear RGB, torch or numpy in = same out

Inputs follow Cycles' denoising passes: rgb = noisy linear radiance, albedo = "Denoising Albedo", normal =
"Denoising Normal" in CAMERA space (x right, y up, z forward; a face towards the camera has z = -1). Pass
``world_to_cam`` (..., 3, 3) to rotate world-space normals first.
"""
from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ._torch import F, torch
from .model import KPN_MODES, NetConfig, UNet, make_input, to_linear
from .weights import resolve_weights, sha256_file, sha256_in_manifest

log = logging.getLogger(__name__)

_DTYPES = {"fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
           "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp32": torch.float32, "float32": torch.float32, "float": torch.float32}


def default_device() -> torch.device:
    """cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    """Accept "fp16"/"bf16"/"fp32" strings or torch dtypes; default fp16 on CUDA, fp32 on CPU.

    Example: ``parse_dtype("bf16", torch.device("cpu"))`` is ``torch.bfloat16``.
    """
    if dtype is None:
        return torch.float16 if device.type == "cuda" else torch.float32
    if isinstance(dtype, torch.dtype):
        if dtype not in _DTYPES.values():
            raise ValueError(f"dtype {dtype} is not supported; use torch.float16, torch.bfloat16 or torch.float32.")
        return dtype
    if str(dtype).lower() not in _DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}; use one of 'fp16', 'bf16', 'fp32'.")
    return _DTYPES[str(dtype).lower()]


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint with weights_only=True; fall back to full unpickling only for manifest-listed files.

    Example: ``read_checkpoint("dn.pt")["cfg"]``.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as e:
        digest = sha256_file(path)
        if not sha256_in_manifest(digest):
            raise ValueError(
                f"{path} could not be loaded with weights_only=True ({e}) and its sha256 is not in the weights "
                f"manifest, so it is not unpickled. Re-save it with Denoiser.save(net, path) "
                f"(plain tensors and config only).") from e
        log.warning("%s needs full unpickling; sha256 matches the manifest, loading with weights_only=False", path)
        return torch.load(path, map_location="cpu", weights_only=False)


def _to_tensor(x: Any) -> torch.Tensor:
    return x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))


def _check_images(rgb: torch.Tensor, albedo: torch.Tensor, normal: torch.Tensor) -> None:
    for name, t in (("rgb", rgb), ("albedo", albedo), ("normal", normal)):
        if t.ndim < 3 or t.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (..., H, W, 3) with 3 channels last, got {tuple(t.shape)}.")
    if not (rgb.shape == albedo.shape == normal.shape):
        raise ValueError(f"rgb, albedo and normal must have the same shape, got {tuple(rgb.shape)}, "
                         f"{tuple(albedo.shape)}, {tuple(normal.shape)}.")


def _prepare_spp(spp: Any, lead: tuple[int, ...], hw: tuple[int, int], device: torch.device) -> float | torch.Tensor:
    """Validate spp; return a float, an (N,) tensor or an (N,H,W) per-pixel tensor."""
    t = _to_tensor(spp).to(device=device, dtype=torch.float32)
    if t.numel() and bool((t < 1).any()):
        raise ValueError(f"spp must be >= 1 everywhere (log2 of it is an input), got min {float(t.min())}.")
    n = int(np.prod(lead, dtype=np.int64))
    if t.ndim == 0:
        return float(t)
    if tuple(t.shape) == lead or t.numel() == n and t.ndim == 1:
        return t.reshape(n)
    if tuple(t.shape) == (*lead, *hw):
        return t.reshape(n, *hw)
    raise ValueError(f"spp tensor has shape {tuple(t.shape)}; pass a number, one value per image {lead}, "
                     f"or one per pixel {(*lead, *hw)}.")


def _rotate_normals(normal: torch.Tensor, w2c: Any, lead: tuple[int, ...]) -> torch.Tensor:
    """normal (N,H,W,3) rotated by world_to_cam (...,3,3) (or a single (3,3) for all images)."""
    r = _to_tensor(w2c).to(device=normal.device, dtype=normal.dtype)
    if r.shape[-2:] != (3, 3) or r.ndim not in (2, len(lead) + 2):
        raise ValueError(f"world_to_cam must have shape {(*lead, 3, 3)} or (3, 3), got {tuple(r.shape)}.")
    r = r.reshape(-1, 3, 3).expand(normal.shape[0], 3, 3)
    return torch.einsum("nij,nhwj->nhwi", r, normal)


def _transpose(rgb: torch.Tensor, albedo: torch.Tensor, normal: torch.Tensor, spp: float | torch.Tensor):
    """Swap the image axes of (N,H,W,3) tensors (and a per-pixel spp map); normals (x, y, z) -> (-y, -x, z)."""
    t = lambda x: x.transpose(-3, -2)  # noqa: E731
    n = t(normal)
    n = torch.stack([-n[..., 1], -n[..., 0], n[..., 2]], -1)
    s = spp.transpose(-2, -1) if torch.is_tensor(spp) and spp.ndim == 3 else spp
    return t(rgb), t(albedo), n, s


def _flip(t: torch.Tensor, fx: bool, fy: bool, spatial_last: bool = False) -> torch.Tensor:
    """Flip horizontally (fx) / vertically (fy); spatial_last: H, W are the last two dims instead of (-3, -2)."""
    hd, wd = (-2, -1) if spatial_last else (-3, -2)
    dims = [d for d, f in ((wd, fx), (hd, fy)) if f]
    return t.flip(dims) if dims else t


class Denoiser:
    """Inference wrapper around a :class:`UNet`: pads to the network stride, channels_last, optional TTA.

    Example: ``dn = Denoiser.load("dn.pt"); out = dn.denoise(rgb, albedo, normal, spp=1)``.
    """

    def __init__(self, net: UNet, device: str | torch.device | None = None,
                 dtype: str | torch.dtype | None = None, compile: bool = False) -> None:
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = parse_dtype(dtype, self.device)
        self.net = net.to(self.device).eval().to(memory_format=torch.channels_last)
        self.stride = net.cfg.unshuffle * 2 ** (len(net.cfg.enc) - 1)
        self.fwd = torch.compile(self.net, mode="max-autotune-no-cudagraphs") if compile else self.net

    @classmethod
    def load(cls, path_or_name: str | Path | Sequence[str | Path], device: str | torch.device | None = None,
             dtype: str | torch.dtype | None = None, compile: bool = False) -> Denoiser | Ensemble:
        """Load a .pt checkpoint (or a list of them as an averaging :class:`Ensemble`).

        Names and "default" must go through :func:`alloy.camera.denoiser.load`, which resolves them first.
        Example: ``Denoiser.load("dn.pt", device="cpu", dtype="fp32")``.
        """
        if isinstance(path_or_name, (list, tuple)):
            return Ensemble([cls.load(p, device, dtype, compile) for p in path_or_name])
        ck = read_checkpoint(path_or_name)
        if not isinstance(ck, dict) or "cfg" not in ck or "state" not in ck:
            raise ValueError(f"{path_or_name} is not a denoiser checkpoint (needs 'cfg' and 'state' keys); "
                             f"create one with Denoiser.save(net, path).")
        net = UNet(NetConfig(**ck["cfg"]))
        net.load_state_dict(ck["state"])
        return cls(net, device, dtype, compile)

    @staticmethod
    def save(net: UNet, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """Save {"cfg", "state", **extra} (plain tensors and config only). Example: ``Denoiser.save(net, "dn.pt")``."""
        torch.save({"cfg": asdict(net.cfg), "state": net.state_dict(), **(extra or {})}, path)

    @torch.inference_mode()
    def denoise(self, rgb: Any, albedo: Any, normal: Any, spp: Any = 1, world_to_cam: Any = None,
                tta: int = 1) -> Any:
        """Denoise (..., H, W, 3) linear frames; returns float32 of the same shape, same type and device as rgb.

        spp: number, one value per image, or one per pixel. world_to_cam (..., 3, 3) rotates world-space normals.
        tta 2 / 4 / 8: also run the horizontally (and vertically) flipped (and transposed, 8) input and average
        (2x / 4x / 8x the cost).
        Example: ``out = dn.denoise(rgb, albedo, normal, spp=1, tta=2)``.
        """
        if tta not in (1, 2, 4, 8):
            raise ValueError(f"tta must be 1, 2, 4 or 8, got {tta}.")
        is_np = not torch.is_tensor(rgb)
        src_device = None if is_np else rgb.device
        rgb, albedo, normal = (_to_tensor(t) for t in (rgb, albedo, normal))
        _check_images(rgb, albedo, normal)
        lead = tuple(rgb.shape[:-3])
        hw = (rgb.shape[-3], rgb.shape[-2])
        rgb, albedo, normal = (t.to(self.device, torch.float32).reshape(-1, *hw, 3) for t in (rgb, albedo, normal))
        spp_ = _prepare_spp(spp, lead, hw, self.device)
        if world_to_cam is not None:
            normal = _rotate_normals(normal, world_to_cam, lead)
        out = self._run_tta(rgb, albedo, normal, spp_, tta).reshape(*lead, *hw, 3)
        return out.cpu().numpy() if is_np else out.to(src_device)

    def _run_tta(self, rgb: torch.Tensor, albedo: torch.Tensor, normal: torch.Tensor,
                 spp: float | torch.Tensor, tta: int) -> torch.Tensor:
        flips = [(False, False), (True, False), (False, True), (True, True)][:min(tta, 4)]
        acc = None
        for tr in ((False, True) if tta == 8 else (False,)):
            imgs = (rgb, albedo, normal, spp)
            if tr:  # transpose: image (x, y) -> (y, x), camera-space normal (x, y) -> (-y, -x)
                imgs = _transpose(*imgs)
            for fx, fy in flips:
                out = self._flipped(*imgs, fx, fy)
                out = out.transpose(-3, -2) if tr else out
                acc = out if acc is None else acc + out
        return acc / (len(flips) * (2 if tta == 8 else 1))

    def _flipped(self, rgb: torch.Tensor, albedo: torch.Tensor, normal: torch.Tensor, spp: float | torch.Tensor,
                 fx: bool, fy: bool) -> torch.Tensor:
        """One pass on the flipped input, flipped back."""
        sgn = normal.new_tensor([-1.0 if fx else 1.0, -1.0 if fy else 1.0, 1.0])
        s = _flip(spp, fx, fy, True) if torch.is_tensor(spp) and spp.ndim == 3 else spp
        out = self._forward(_flip(rgb, fx, fy), _flip(albedo, fx, fy), _flip(normal, fx, fy) * sgn, s)
        return _flip(out, fx, fy)

    def _forward(self, rgb: torch.Tensor, albedo: torch.Tensor, normal: torch.Tensor,
                 spp: float | torch.Tensor) -> torch.Tensor:
        """One network pass on (N,H,W,3) float32 tensors: pad to the stride, run, crop, convert to linear."""
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=6.0e4, neginf=0.0)  # half-float fireflies overflow to inf
        x = make_input(rgb, albedo, normal, spp)
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.stride, (-w) % self.stride
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        half = self.dtype != torch.float32
        if half and self.net.cfg.out_mode not in KPN_MODES:  # kpn filters the input itself: keep it float32
            x = x.to(self.dtype)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=half):
            out = self.fwd(x)
        return to_linear(out[..., :h, :w], albedo)


class Ensemble:
    """Mean of several Denoisers' linear outputs (same ``denoise`` signature; each member's cost adds up).

    Example: ``Denoiser.load(["a.pt", "b.pt"]).denoise(rgb, albedo, normal)``.
    """

    def __init__(self, members: Sequence[Denoiser]) -> None:
        if not members:
            raise ValueError("Ensemble needs at least one member; pass a non-empty list of weights.")
        self.members = list(members)
        self.net = self.members[0].net

    def denoise(self, rgb: Any, albedo: Any, normal: Any, spp: Any = 1, world_to_cam: Any = None,
                tta: int = 1) -> Any:
        """Average of the members' ``denoise`` outputs. Example: ``ens.denoise(rgb, albedo, normal, spp=2)``."""
        outs = [m.denoise(rgb, albedo, normal, spp, world_to_cam, tta) for m in self.members]
        return sum(outs[1:], outs[0]) / len(outs)


def load(name_or_path: str | Path | Sequence[str | Path], **kw: Any) -> Denoiser | Ensemble:
    """Resolve weights (path, "default", manifest name, or a list of them) and load them.

    ``kw`` goes to :meth:`Denoiser.load` (device, dtype, compile). Example: ``load("default", device="cpu")``.
    """
    if isinstance(name_or_path, (list, tuple)):
        return Ensemble([load(p, **kw) for p in name_or_path])
    return Denoiser.load(resolve_weights(name_or_path), **kw)
