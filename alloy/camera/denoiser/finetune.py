# SPDX-License-Identifier: Apache-2.0
"""Per-scene fine-tuning: adapt a denoiser to a new scene from a few dozen noisy / reference pairs in minutes.

    import numpy as np
    from alloy.camera.denoiser import load
    from alloy.camera.denoiser.finetune import PairSet, finetune

    # 40 frames of your scene: inputs at 1-16 spp and one high-spp reference each (any renderer that follows
    # Cycles' denoising-pass conventions: noisy linear rgb, albedo, camera-space normal)
    pairs = PairSet.from_numpy(rgb, albedo, normal, ref, spp=[1, 2, 4, 8, 16])   # (N, K, H, W, 3), (N, H, W, 3)
    path = finetune("default", pairs, out="my_scene.pt", steps=2000)
    dn = load(path)

The recipe (measured on new scenes in the research round: about 2-3 minutes on one RTX-class GPU) starts from the
base weights, uses a lower learning rate, and can mix in a share of the base teacher set (``base_pairs``) against
forgetting. Check the result against the base model on held-out frames of the same scene before using it.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._torch import torch
from .infer import Denoiser, read_checkpoint
from .model import NetConfig, UNet
from .training import Batch, DisplayLUT, Loss, Sampler, TrainConfig, train
from .weights import resolve_weights

log = logging.getLogger(__name__)


@dataclass
class PairSet:
    """Teacher pairs held on one device: inputs (N, K, H, W, 3), references (N, H, W, 3), spp (K,), LUT index (N,).

    Example: ``PairSet.from_numpy(rgb, albedo, normal, ref, spp=[1, 4, 16])``."""

    rgb: torch.Tensor
    albedo: torch.Tensor
    normal: torch.Tensor
    ref: torch.Tensor
    spp: torch.Tensor
    lut_idx: torch.Tensor

    @classmethod
    def from_numpy(cls, rgb: Any, albedo: Any, normal: Any, ref: Any, spp: Sequence[float],
                   lut_idx: Any = None, device: str | torch.device | None = None) -> PairSet:
        """Build from arrays; inputs are stored as float16. Shapes are checked with errors that say what to fix."""
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        t = [torch.as_tensor(np.asarray(a)).to(dev, torch.float16) for a in (rgb, albedo, normal)]
        r = torch.as_tensor(np.asarray(ref)).to(dev, torch.float16)
        _check_pairs(t, r, spp)
        n = r.shape[0]
        idx = torch.zeros(n, dtype=torch.long) if lut_idx is None else torch.as_tensor(np.asarray(lut_idx)).long()
        return cls(*t, r, torch.as_tensor(list(spp), dtype=torch.float32, device=dev), idx.to(dev))

    def sampler(self, crop: int = 128) -> Sampler:
        """A batch sampler: random frame and input per item, one random crop per batch (``crop`` <= H, W)."""
        n, k, h, w, _ = self.rgb.shape
        c = min(crop, h, w)

        def draw(b: int, g: torch.Generator) -> Batch:
            dev = self.rgb.device
            i = torch.randint(0, n, (b,), device=dev, generator=g)
            j = torch.randint(0, k, (b,), device=dev, generator=g)
            y0 = int(torch.randint(0, h - c + 1, (1,), device=dev, generator=g))
            x0 = int(torch.randint(0, w - c + 1, (1,), device=dev, generator=g))
            sl = (slice(y0, y0 + c), slice(x0, x0 + c))
            pick = lambda a: a[i, j, sl[0], sl[1]].float()
            return Batch(pick(self.rgb), pick(self.albedo), pick(self.normal), self.ref[i, sl[0], sl[1]].float(),
                         self.spp[j], self.lut_idx[i])
        return draw


def _check_pairs(t: list[torch.Tensor], ref: torch.Tensor, spp: Sequence[float]) -> None:
    if t[0].ndim != 5 or t[0].shape[-1] != 3:
        raise ValueError(f"rgb must be (N, K, H, W, 3) (K inputs per frame), got {tuple(t[0].shape)}; add a K axis "
                         f"with rgb[:, None] for one input per frame.")
    if any(x.shape != t[0].shape for x in t[1:]):
        raise ValueError(f"albedo / normal must match rgb's shape {tuple(t[0].shape)}.")
    if tuple(ref.shape) != (t[0].shape[0], *t[0].shape[2:]):
        raise ValueError(f"ref must be (N, H, W, 3) = {(t[0].shape[0], *t[0].shape[2:])}, got {tuple(ref.shape)}.")
    if len(spp) != t[0].shape[1] or min(spp) < 1:
        raise ValueError(f"spp needs one value >= 1 per input (K = {t[0].shape[1]}), got {list(spp)}.")


def mixed_sampler(main: Sampler, base: Sampler | None, base_frac: float) -> Sampler:
    """Draw ``base_frac`` of each batch from ``base`` (anti-forgetting) and the rest from ``main``."""
    if base is None or base_frac <= 0:
        return main

    def draw(b: int, g: torch.Generator) -> Batch:
        nb = round(b * base_frac)
        parts = [main(b - nb, g)] + ([base(nb, g)] if nb else [])
        return Batch(*(torch.cat([getattr(p, f) for p in parts]) for f in Batch.__dataclass_fields__))
    return draw


def finetune(base: str | Path, pairs: PairSet, out: str | Path, steps: int = 2000, lr: float = 1e-4,
             crop: int = 128, base_pairs: PairSet | None = None, base_frac: float = 0.3,
             display: DisplayLUT | None = None, w_lpips: float = 0.4, seed: int = 0) -> Path:
    """Fine-tune ``base`` weights (path, manifest name or "default") on ``pairs``; writes and returns ``out``.

    ``display`` is the scene's display LUT (default: the sRGB curve). Example: see the module docstring."""
    dev = pairs.rgb.device
    ck = read_checkpoint(resolve_weights(base))
    net = UNet(NetConfig(**ck["cfg"]))
    net.load_state_dict(ck["state"])
    net.to(dev)
    batch = 32 if dev.type == "cuda" else 4
    cfg = TrainConfig(steps=steps, batch=batch, lr=lr, warmup=min(200, steps // 10), w_lpips=w_lpips,
                      log_every=max(steps // 10, 1), ckpt=str(out), extra={"finetuned_from": str(base)})
    loss = Loss(display or DisplayLUT.srgb(), w_lpips=w_lpips, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    sampler = mixed_sampler(pairs.sampler(crop), base_pairs.sampler(crop) if base_pairs else None, base_frac)
    train(net, sampler, loss, cfg, g)
    Path(f"{out}.train").unlink(missing_ok=True)
    log.info("fine-tuned %s on %d frames -> %s", base, pairs.rgb.shape[0], out)
    return Path(out)


def evaluate(dn: Denoiser, pairs: PairSet, display: DisplayLUT | None = None, tta: int = 1) -> dict[int, float]:
    """Mean display-space L1 per input spp on ``pairs`` (lower is better), e.g. to compare base vs fine-tuned.

    Example: ``evaluate(load("default"), heldout)`` -> ``{1: 0.031, 4: 0.018}``."""
    disp = (display or DisplayLUT.srgb()).to(pairs.rgb.device)
    out = {}
    for j, s in enumerate(pairs.spp.tolist()):
        errs = []
        for i in range(pairs.rgb.shape[0]):
            x = [a[i:i + 1, j].float() for a in (pairs.rgb, pairs.albedo, pairs.normal)]
            pred = torch.as_tensor(dn.denoise(*x, spp=s, tta=tta)).to(pairs.rgb.device)
            idx = pairs.lut_idx[i:i + 1]
            errs.append(float((disp.apply(pred, idx) - disp.apply(pairs.ref[i:i + 1].float(), idx)).abs().mean()))
        out[int(s)] = float(np.mean(errs))
    return out
