# SPDX-License-Identifier: Apache-2.0
"""Training core for the denoiser: display-space loss, augmentation, the train loop and per-scene fine-tuning.

    from alloy.camera.denoiser.training import PairSet, finetune
    pairs = PairSet(rgb, albedo, normal, ref, spp)        # (N, K, H, W, 3) inputs at K spp, (N, H, W, 3) targets
    path = finetune("default", pairs, out="my_scene.pt", steps=2000)   # minutes on one GPU
    dn = load(path)

The loss is L1 + ``w_lpips`` * LPIPS (if the ``lpips`` package is installed) in display space, plus ``w_log`` * L1
on log1p(linear). The display transform is a :class:`DisplayLUT` (baked from Blender for teacher sets) or the
plain sRGB curve when no LUT is given. Needs torch; lpips is optional.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._torch import F, torch
from .model import UNet, make_input, to_linear

log = logging.getLogger(__name__)


@dataclass
class Batch:
    """One training batch: (B, H, W, 3) float32 tensors, spp (B,) and display-LUT index (B,)."""

    rgb: torch.Tensor
    albedo: torch.Tensor
    normal: torch.Tensor
    ref: torch.Tensor
    spp: torch.Tensor
    lut_idx: torch.Tensor


class DisplayLUT:
    """Stack of display LUTs (L, 3, n, n, n), axes [c, b, g, r], on a log2 shaper grid [lo, hi].

    Example: ``DisplayLUT(torch.from_numpy(lut)[None], -12, 10).apply(x, torch.zeros(len(x), dtype=torch.long))``.
    ``DisplayLUT.srgb()`` is a stand-in (clamped sRGB curve) for scenes without a baked LUT."""

    def __init__(self, luts: torch.Tensor | None, lo: float = -12.0, hi: float = 10.0) -> None:
        self.luts, self.lo, self.hi = luts, lo, hi

    @classmethod
    def srgb(cls) -> DisplayLUT:
        """The sRGB display curve (no tone mapping) instead of a baked LUT."""
        return cls(None)

    def to(self, device: torch.device | str) -> DisplayLUT:
        """Move the LUT stack to ``device``."""
        return DisplayLUT(None if self.luts is None else self.luts.to(device).float(), self.lo, self.hi)

    def apply(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Linear (B, H, W, 3) -> display-encoded (B, H, W, 3) in [0, 1] (before 8-bit rounding)."""
        if self.luts is None:
            v = x.float().clamp(0, 1)
            return torch.where(v <= 0.0031308, 12.92 * v, 1.055 * v.clamp_min(1e-8) ** (1 / 2.4) - 0.055)
        b, h, w, _ = x.shape
        u = (torch.log2(x.float().clamp_min(0) + 2.0 ** self.lo) - self.lo) / (self.hi - self.lo)
        grid = (u.clamp(0, 1) * 2 - 1).reshape(b, 1, h * w, 1, 3)
        out = F.grid_sample(self.luts[idx], grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out.reshape(b, 3, h, w).permute(0, 2, 3, 1)


class Loss:
    """Display-space L1 + w_lpips * LPIPS-alex + w_log * L1(log1p linear). Example: ``Loss(DisplayLUT.srgb())``."""

    def __init__(self, display: DisplayLUT, w_lpips: float = 0.2, w_log: float = 0.1,
                 device: torch.device | str = "cuda") -> None:
        self.display, self.w_lpips, self.w_log = display.to(device), w_lpips, w_log
        self.lp = _lpips(device) if w_lpips > 0 else None

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        dp, dr = self.display.apply(pred, idx), self.display.apply(ref, idx)
        l1 = (dp - dr).abs().mean()
        llog = (torch.log1p(pred.clamp_min(0)) - torch.log1p(ref.clamp_min(0))).abs().mean()
        tot = l1 + self.w_log * llog
        parts = {"l1": float(l1.detach()), "log": float(llog.detach())}
        if self.lp is not None:
            lp = self.lp(dp.permute(0, 3, 1, 2) * 2 - 1, dr.permute(0, 3, 1, 2) * 2 - 1).mean()
            tot = tot + self.w_lpips * lp
            parts["lpips"] = float(lp.detach())
        return tot, parts


def _lpips(device: torch.device | str) -> torch.nn.Module | None:
    try:
        import lpips
    except ImportError:
        log.warning("lpips is not installed: training uses L1 only (pip install lpips for the full loss)")
        return None
    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def augment(b: Batch, g: torch.Generator, stops: float = 1.5) -> Batch:
    """Random flips / transpose (camera-space normals follow), RGB permutation, exposure x 2^U(-stops, stops).

    Example: ``augment(batch, torch.Generator("cuda").manual_seed(0))``."""
    dev = b.rgb.device
    r = torch.rand(4, generator=g, device=dev).tolist()
    rgb, alb, nrm, ref = b.rgb, b.albedo, b.normal, b.ref
    if r[0] < 0.5:  # horizontal flip: camera x -> -x
        rgb, alb, nrm, ref = (t.flip(2) for t in (rgb, alb, nrm, ref))
        nrm = nrm * nrm.new_tensor([-1.0, 1.0, 1.0])
    if r[1] < 0.5:  # vertical flip: camera y -> -y
        rgb, alb, nrm, ref = (t.flip(1) for t in (rgb, alb, nrm, ref))
        nrm = nrm * nrm.new_tensor([1.0, -1.0, 1.0])
    if r[2] < 0.5:  # transpose: (x, y) -> (-y, -x)
        rgb, alb, nrm, ref = (t.transpose(1, 2) for t in (rgb, alb, nrm, ref))
        nrm = torch.stack([-nrm[..., 1], -nrm[..., 0], nrm[..., 2]], -1)
    if r[3] < 0.5:
        p = torch.randperm(3, generator=g, device=dev)
        rgb, alb, ref = rgb[..., p], alb[..., p], ref[..., p]
    e = torch.exp2((torch.rand(rgb.shape[0], 1, 1, 1, device=dev, generator=g) * 2 - 1) * stops)
    return Batch(rgb * e, alb, nrm, ref * e, b.spp, b.lut_idx)


@dataclass
class TrainConfig:
    """Optimiser and loss settings. Example: ``TrainConfig(steps=2000, lr=1e-4)``."""

    steps: int = 40000
    batch: int = 64
    lr: float = 3e-4
    warmup: int = 1000
    final_lr_frac: float = 0.03
    w_lpips: float = 0.2
    w_log: float = 0.1
    exp_stops: float = 1.5
    clip: float = 1.0
    log_every: int = 250
    ckpt_every: int = 0
    ckpt: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def lr_lambda(cfg: TrainConfig) -> Callable[[int], float]:
    """Linear warm-up then cosine decay to ``final_lr_frac``."""
    def f(s: int) -> float:
        cos = 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / max(cfg.steps, 1)))
        return min(1.0, (s + 1) / max(cfg.warmup, 1)) * (cfg.final_lr_frac + (1 - cfg.final_lr_frac) * cos)
    return f


Sampler = Callable[[int, torch.Generator], Batch]


def train_step(net: UNet, b: Batch, loss: Loss, opt: torch.optim.Optimizer, clip: float) -> dict:
    """One optimiser step on an (already augmented) batch; returns the loss parts."""
    x = make_input(b.rgb, b.albedo, b.normal, b.spp).contiguous(memory_format=torch.channels_last)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = net(x)
    tot, parts = loss(to_linear(out, b.albedo), b.ref, b.lut_idx)
    opt.zero_grad(set_to_none=True)
    tot.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    return parts


def save_checkpoint(net: UNet, path: str | Path, step: int, opt: torch.optim.Optimizer | None = None,
                    extra: dict | None = None) -> None:
    """Weights to ``path`` (loadable by ``Denoiser.load``); optimiser state to ``path + ".train"`` for resuming."""
    from .infer import Denoiser
    tmp = Path(f"{path}.part")
    Denoiser.save(net, tmp, {"step": step, **(extra or {})})
    tmp.replace(path)
    if opt is not None:
        torch.save({"opt": opt.state_dict(), "step": step}, f"{path}.part.train")
        Path(f"{path}.part.train").replace(f"{path}.train")


def train(net: UNet, sampler: Sampler, loss: Loss, cfg: TrainConfig, g: torch.Generator,
          on_log: Callable[[dict], None] | None = None, resume: bool = False) -> int:
    """Train ``net`` for ``cfg.steps`` steps on batches from ``sampler(batch_size, g)``; returns the last step.

    ``on_log`` gets one row per ``cfg.log_every`` steps (and may run validation). With ``resume`` and
    ``cfg.ckpt``, the optimiser and step continue from ``cfg.ckpt + ".train"``; load ``net`` from ``cfg.ckpt``
    first. Example::

        train(net, pairs.sampler(crop=128), Loss(DisplayLUT.srgb()), TrainConfig(steps=500), g)
    """
    net.train().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    start = _resume(opt, cfg) if resume else 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    for _ in range(start):  # replay the schedule up to the resumed step (cheap: lambda evaluations only)
        sched.step()
    run, t0 = {}, time.perf_counter()
    for step in range(start + 1, cfg.steps + 1):
        parts = train_step(net, augment(sampler(cfg.batch, g), g, cfg.exp_stops), loss, opt, cfg.clip)
        sched.step()
        for k, v in parts.items():
            run[k] = run.get(k, 0.0) + v
        if step % cfg.log_every == 0 or step == cfg.steps:
            n = cfg.log_every if step % cfg.log_every == 0 else step % cfg.log_every
            row = {"step": step, "lr": round(sched.get_last_lr()[0], 7), "s": round(time.perf_counter() - t0, 1),
                   **{k: round(v / n, 5) for k, v in run.items()}}
            run = {}
            log.info("train %s", row)
            if on_log is not None:
                on_log(row)
        if cfg.ckpt and cfg.ckpt_every and step % cfg.ckpt_every == 0:
            save_checkpoint(net, cfg.ckpt, step, opt, cfg.extra)
    if cfg.ckpt:
        save_checkpoint(net, cfg.ckpt, cfg.steps, opt, cfg.extra)
    return cfg.steps


def _resume(opt: torch.optim.Optimizer, cfg: TrainConfig) -> int:
    p = Path(f"{cfg.ckpt}.train")
    if not cfg.ckpt or not p.exists():
        return 0
    st = torch.load(p, map_location="cpu", weights_only=True)
    opt.load_state_dict(st["opt"])
    log.info("resumed optimiser state from %s at step %d", p, st["step"])
    return int(st["step"])
