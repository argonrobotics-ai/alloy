# SPDX-License-Identifier: Apache-2.0
"""Convert a Taxim calibration (``polycalib.npz`` + ``dataPack.npz``) into Alloy's tactile calibration file.

Taxim (Si and Yuan 2022, MIT licence) fits, for each bin of gel slope (magnitude 0..90 deg, direction -180..180 deg),
a quadratic polynomial over the image position that gives the colour change from the unpressed gel. Alloy stores the
same table with the polynomial in normalised image coordinates (u, v), as float16, in RGB order, plus the
unpressed-gel background, so one file holds a sensor's whole optical model.

Run once to rebuild the shipped presets from a Taxim checkout (``git clone https://github.com/CMURoboTouch/Taxim``)::

    python -m alloy.camera.tactile.calibrate --taxim <Taxim>/calibs --name gelsight --mm-per-px 0.0295
    python -m alloy.camera.tactile.calibrate --taxim <Taxim>/calibs/digit --name digit --mm-per-px 0.0266

Your own sensor: calibrate it with Taxim's tools, then ``save_calibration(from_taxim(dir, mm_per_px), "my.npz")``
and ``load_preset("gelsight", data_file="my.npz")``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..errors import CameraError

log = logging.getLogger(__name__)
TAXIM_BG_SIGMA = 50.0  # Taxim params.kscale: blur of the unpressed frame
TAXIM_BG_DIFF = 5.0  # Taxim params.diffThreshold
TAXIM_BG_MIX = 0.15  # Taxim params.frameMixingPercentage


def poly_to_normalized(table_xy: np.ndarray, height: int, width: int) -> np.ndarray:
    """Re-express polynomial coefficients over pixel (x, y) as coefficients over normalised (u, v).

    Taxim's terms are [x^2, y^2, xy, x, y, 1] with x the column and y the row in pixels. Alloy uses
    u = (x - cx) / s, v = (y - cy) / s with cx = (W - 1) / 2, cy = (H - 1) / 2, s = (W - 1) / 2, so the table is
    resolution free and every coefficient is a colour amount (float16 keeps it to 0.2 intensity levels).
    Example: ``poly_to_normalized(t, 480, 640)`` for a (..., 6) table.
    """
    a, b, c, d, e, f = (table_xy[..., i].astype(np.float64) for i in range(6))
    cx, cy, s = (width - 1) / 2.0, (height - 1) / 2.0, (width - 1) / 2.0
    out = np.stack([a * s * s, b * s * s, c * s * s, (2 * a * cx + c * cy + d) * s, (2 * b * cy + c * cx + e) * s,
                    a * cx * cx + b * cy * cy + c * cx * cy + d * cx + e * cy + f], axis=-1)
    return out


def taxim_background(f0: np.ndarray) -> np.ndarray:
    """Taxim's processed unpressed frame: a strong blur where the raw frame has dark specks, else mostly raw.

    Same steps as Taxim's ``processInitialFrame`` (uint8 rounding included), so the table's colour offsets add to
    the background they were fitted against. Example: ``taxim_background(f0_uint8)`` -> (H, W, 3) uint8.
    """
    from scipy.ndimage import gaussian_filter
    raw = f0.astype(np.float64)
    blur = np.stack([gaussian_filter(raw[..., ch], TAXIM_BG_SIGMA) for ch in range(raw.shape[2])], -1)
    blur = blur.astype(np.uint8)
    keep = (blur.astype(np.float64) - raw).mean(axis=2) < TAXIM_BG_DIFF
    mixed = (TAXIM_BG_MIX * blur.astype(np.float64) + (1.0 - TAXIM_BG_MIX) * raw).astype(np.uint8)
    out = blur.copy()
    out[keep] = mixed[keep]
    return out


def from_taxim(folder: str | Path, mm_per_px: float, bgr: bool = True) -> dict:
    """Read Taxim's ``polycalib.npz`` and ``dataPack.npz`` from ``folder`` into an Alloy calibration dict.

    ``bgr``: Taxim's images and tables are in OpenCV's BGR order (true for Taxim's own files).
    Example: ``cal = from_taxim("Taxim/calibs/digit", 0.0266)``.
    """
    folder = Path(folder)
    poly, pack = folder / "polycalib.npz", folder / "dataPack.npz"
    if not poly.is_file() or not pack.is_file():
        raise CameraError(f"{folder} has no polycalib.npz and dataPack.npz",
                          "pass a Taxim calibration folder (e.g. Taxim/calibs or Taxim/calibs/digit)")
    with np.load(poly, allow_pickle=False) as z:
        table = np.stack([z["grad_r"], z["grad_g"], z["grad_b"]])  # Taxim's names; channel order of its images
    with np.load(pack, allow_pickle=False) as z:
        f0 = np.asarray(z["f0"])
        box = fit_box(z["touch_center"], z["touch_radius"], f0.shape[:2])
    h, w = f0.shape[:2]
    bg = taxim_background(f0)
    if bgr:
        table, bg = table[::-1], bg[..., ::-1]
    return {"table": poly_to_normalized(table, h, w), "background": np.ascontiguousarray(bg),
            "meta": {"calib_size": [int(h), int(w)], "mm_per_px": float(mm_per_px), "bins": int(table.shape[1]),
                     "fit_box": box,
                     "coords": "u=(x-(W-1)/2)/((W-1)/2), v=(y-(H-1)/2)/((W-1)/2); terms u2 v2 uv u v 1; RGB"}}


def fit_box(centers: np.ndarray, radii: np.ndarray, size: tuple[int, int]) -> list[float]:
    """[x0, y0, x1, y1] (pixels) that the calibration presses covered; the table is not extrapolated beyond it.

    Example: ``fit_box(np.array([[320, 240]]), np.array([60]), (480, 640)) == [260.0, 180.0, 380.0, 300.0]``.
    """
    c, r = np.asarray(centers, np.float64).reshape(-1, 2), np.asarray(radii, np.float64).reshape(-1)
    h, w = size
    lo = np.maximum((c - r[:, None]).min(0), 0.0)
    hi = np.minimum((c + r[:, None]).max(0), [w - 1.0, h - 1.0])
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def save_calibration(cal: dict, path: str | Path) -> Path:
    """Write a calibration dict (``table``, ``background``, ``meta``) as a compressed .npz; returns the path.

    Example: ``save_calibration(from_taxim("calibs", 0.0295), "gelsight.npz")``.
    """
    path = Path(path)
    np.savez_compressed(path, table=np.asarray(cal["table"], np.float16), background=cal["background"],
                        meta=np.array(json.dumps(cal["meta"])))
    log.info("tactile calibration written to %s (%d bytes)", path, path.stat().st_size)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI: convert one Taxim calibration folder. Prints a JSON line with the output path and size."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--taxim", required=True, help="Taxim calibration folder (polycalib.npz + dataPack.npz)")
    ap.add_argument("--name", required=True, help="output name, e.g. gelsight")
    ap.add_argument("--mm-per-px", type=float, required=True, help="gel millimetres per calibration pixel")
    ap.add_argument("--out-dir", default=str(Path(__file__).with_name("data")))
    args = ap.parse_args(argv)
    out = save_calibration(from_taxim(args.taxim, args.mm_per_px), Path(args.out_dir) / f"{args.name}.npz")
    print(json.dumps({"path": str(out), "bytes": out.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
