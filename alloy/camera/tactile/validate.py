# SPDX-License-Identifier: Apache-2.0
"""Rendered ball presses vs a sensor's real calibration presses: L1 and SSIM per image (JSON out).

Taxim's data packs (``calibs/dataPack.npz``, ``calibs/digit/dataPack.npz``; MIT) hold real images of a 4 mm ball
pressed into the gel, with the contact centre and radius marked by hand. Each press is rendered as a 2 mm-radius
sphere whose contact circle has that radius, at the calibration resolution, and compared with the real image::

    python -m alloy.camera.tactile.validate --pack <Taxim>/calibs/dataPack.npz --preset gelsight --out val.json

Prints one JSON summary line: mean L1 (0..255) and SSIM over the whole image and over the contact disk (1.6 x the
radius), and the same for the unpressed gel as a baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

BALL_MM = 2.0
DISK = 1.6


def sphere_press(h: int, w: int, mm_px: float, cx: float, cy: float, contact_mm: float) -> np.ndarray:
    """Press [mm] of a 2 mm-radius ball with contact radius ``contact_mm`` centred at pixel (cx, cy)."""
    a = min(contact_mm, BALL_MM * 0.999)
    depth = BALL_MM - math.sqrt(BALL_MM ** 2 - a * a)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) * mm_px ** 2
    return np.clip(np.sqrt(np.clip(BALL_MM ** 2 - r2, 0, None)) - (BALL_MM - depth), 0, None).astype(np.float32)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM of two (H, W, 3) 0..255 images (Gaussian window sigma 1.5, the usual constants), channel mean."""
    from scipy.ndimage import gaussian_filter
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    out = []
    for ch in range(3):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
        vx = gaussian_filter(x * x, 1.5) - mx * mx
        vy = gaussian_filter(y * y, 1.5) - my * my
        cxy = gaussian_filter(x * y, 1.5) - mx * my
        out.append(((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2)))
    return np.mean(out, axis=0)


def _scores(img: np.ndarray, real: np.ndarray, disk: np.ndarray) -> dict:
    l1 = np.abs(img.astype(np.float64) - real.astype(np.float64)).mean(-1)
    s = ssim_map(img, real)
    return {"l1": float(l1.mean()), "ssim": float(s.mean()), "l1_disk": float(l1[disk].mean()),
            "ssim_disk": float(s[disk].mean())}


def evaluate(pack: str | Path, preset: str = "gelsight", device: str = "cpu", batch: int = 16) -> dict:
    """Score every press of a Taxim data pack; returns {"per_image": [...], "sim": {...}, "blank": {...}}."""
    from .renderer import TactileRenderer
    from .augment import Augment
    with np.load(pack, allow_pickle=False) as z:
        imgs, cen, rad = z["imgs"][..., ::-1], z["touch_center"], z["touch_radius"]
    n, h, w = len(rad), imgs.shape[1], imgs.shape[2]
    tr = TactileRenderer(preset, w, h, batch, device, markers=False, augment=Augment(noise_std=0.0))
    if (tr.height, tr.width) != (h, w):
        raise ValueError(f"pack images are {h}x{w}, preset {preset} renders {tr.height}x{tr.width}")
    blank = np.clip(np.round(tr._bg.numpy()), 0, 255).astype(np.uint8)
    rows = []
    for k0 in range(0, n, batch):
        ids = list(range(k0, min(n, k0 + batch)))
        press = np.zeros((batch, h, w), np.float32)
        for j, i in enumerate(ids):
            press[j] = sphere_press(h, w, tr.mm_per_px, cen[i, 0], cen[i, 1], rad[i] * tr.mm_per_px)
        out = tr.render(press, seed=0).numpy()
        yy, xx = np.mgrid[0:h, 0:w]
        for j, i in enumerate(ids):
            disk = (xx - cen[i, 0]) ** 2 + (yy - cen[i, 1]) ** 2 < (DISK * rad[i]) ** 2
            rows.append({"index": i, "sim": _scores(out[j], imgs[i], disk), "blank": _scores(blank, imgs[i], disk)})
    mean = {kind: {m: float(np.mean([r[kind][m] for r in rows])) for m in rows[0][kind]} for kind in ("sim", "blank")}
    return {"preset": preset, "images": n, "size": [h, w], **mean, "per_image": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="rendered vs real ball presses (L1, SSIM)")
    ap.add_argument("--pack", required=True, help="a Taxim dataPack.npz")
    ap.add_argument("--preset", default="gelsight")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the full result (per image) as JSON here")
    args = ap.parse_args(argv)
    res = evaluate(args.pack, args.preset, args.device)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_image"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
