# SPDX-License-Identifier: Apache-2.0
"""Tactile sensor presets: the gel geometry, the optical calibration (a Taxim-style polynomial table) and defaults.

Two presets ship with Alloy, both converted from the MIT-licensed Taxim calibrations (``data/LICENSE-Taxim.txt``):

- ``"gelsight"``: Taxim's GelSight sensor (640 x 480 camera, 0.0295 mm per pixel, 18.9 x 14.2 mm of gel, 4 mm
  calibration ball), with a marker grid (GelSight Mini style, 1.5 mm pitch).
- ``"digit"``: Taxim's DIGIT calibration (640 x 480, 0.0266 mm per pixel, 17.0 x 12.8 mm), no markers (DIGIT's stock
  gel has none).

Example::

    from alloy.camera.tactile.presets import load_preset, list_presets
    p = load_preset("gelsight")
    assert p.table.shape == (3, p.bins, p.bins, 6) and p.background.shape == (480, 640, 3)
    w_mm, h_mm = p.field_mm                     # gel area seen by the camera, in millimetres
    sorted(list_presets())                      # ['digit', 'gelsight']
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..errors import CameraError

DATA_DIR = Path(__file__).with_name("data")


@dataclass(frozen=True)
class MarkerSpec:
    """Printed marker dots on the gel: grid pitch and dot radius [mm], darkness 0..1 (1 = black)."""

    pitch_mm: float = 1.5
    radius_mm: float = 0.2
    darkness: float = 0.85


@dataclass(frozen=True)
class TactilePreset:
    """One vision-based tactile sensor model. Physical units are millimetres; images are (rows, cols)."""

    name: str
    calib_size: tuple[int, int]  # (H, W) of the calibration camera image
    mm_per_px: float  # gel millimetres per calibration pixel
    default_size: tuple[int, int]  # (H, W) rendered when the user gives no size
    smooth_sigmas_mm: tuple[float, ...]  # Taxim's pyramid of Gaussian blurs (gel spreading the press)
    final_sigma_mm: float  # last light blur
    contact_scale: float = 0.4  # pixels deeper than this x the max press keep the object's shape
    gel_radius_mm: float = 0.0  # dome radius of the gel surface (0 = flat)
    max_depth_mm: float = 4.0  # deepest press measured (rays start this far behind the gel surface)
    markers: MarkerSpec | None = None
    noise_std: float = 2.0  # camera noise, 0..255 intensity units
    source: str = ""
    table: np.ndarray = field(default=None, repr=False)  # (3, bins, bins, 6) float32, RGB
    background: np.ndarray = field(default=None, repr=False)  # (H, W, 3) uint8, RGB, the unpressed gel
    fit_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # calibration px covered by the presses
    flat_fade_deg: float = 1.5  # slopes below this fade to the unpressed gel (the table is not fitted near 0)

    @property
    def bins(self) -> int:
        """Number of bins of the slope magnitude and of the slope direction."""
        return int(self.table.shape[1])

    @property
    def field_mm(self) -> tuple[float, float]:
        """(width, height) of the gel area the camera sees, in mm."""
        return self.calib_size[1] * self.mm_per_px, self.calib_size[0] * self.mm_per_px

    def with_options(self, **changes) -> TactilePreset:
        """A copy with some fields changed. Example: ``load_preset("digit").with_options(markers=MarkerSpec())``."""
        unknown = sorted(set(changes) - set(self.__dataclass_fields__))
        if unknown:
            raise CameraError(f"unknown preset fields {unknown}",
                              f"use some of {sorted(self.__dataclass_fields__)}")
        return replace(self, **changes)


def _cv_sigma_mm(kernel: int, mm_per_px: float) -> float:
    """OpenCV's sigma for a Gaussian kernel of ``kernel`` pixels with sigma=0 (Taxim's blurs), in mm."""
    return (0.3 * ((kernel - 1) * 0.5 - 1.0) + 0.8) * mm_per_px


def _builtin(name: str) -> dict:
    """Constants of the built-in presets (see the module docstring for their sources)."""
    if name == "gelsight":
        mm = 0.0295
        return dict(name=name, calib_size=(480, 640), mm_per_px=mm, default_size=(240, 320),
                    smooth_sigmas_mm=tuple(_cv_sigma_mm(k, mm) for k in (101, 51, 21, 11, 5)),
                    final_sigma_mm=_cv_sigma_mm(5, mm), gel_radius_mm=0.0, max_depth_mm=4.0,
                    markers=MarkerSpec(1.5, 0.2, 0.85), noise_std=2.0,
                    source="Taxim GelSight calibration (Si and Yuan 2022, github.com/CMURoboTouch/Taxim, MIT)")
    if name == "digit":
        mm = 0.0266
        return dict(name=name, calib_size=(480, 640), mm_per_px=mm, default_size=(240, 320),
                    smooth_sigmas_mm=tuple(_cv_sigma_mm(k, mm) for k in (101, 51, 21, 11, 5)),
                    final_sigma_mm=_cv_sigma_mm(5, mm), gel_radius_mm=0.0, max_depth_mm=4.0,
                    markers=None, noise_std=2.0,
                    source="Taxim DIGIT calibration (github.com/CMURoboTouch/Taxim calibs/digit, MIT)")
    raise CameraError(f"no tactile preset {name!r}", f"use one of {sorted(list_presets())}")


def list_presets() -> dict[str, str]:
    """Built-in preset names and where their calibration comes from. Example: ``"digit" in list_presets()``."""
    return {n: _builtin(n)["source"] for n in ("gelsight", "digit")}


@lru_cache(maxsize=8)
def _load_data(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return z["table"].astype(np.float32), np.ascontiguousarray(z["background"]), meta


def load_preset(name_or_preset: str | TactilePreset = "gelsight", data_file: str | Path | None = None) -> TactilePreset:
    """A preset by name ("gelsight" or "digit"), with its calibration loaded; a TactilePreset passes through.

    ``data_file``: a calibration ``.npz`` written by :func:`alloy.camera.tactile.calibrate.save_calibration`
    (``table``, ``background``, ``meta``), to use your own sensor's calibration with a built-in preset's gel.
    Example: ``load_preset("digit").field_mm`` is about (17.0, 12.8).
    """
    if isinstance(name_or_preset, TactilePreset):
        return name_or_preset
    consts = _builtin(str(name_or_preset))
    path = Path(data_file) if data_file else DATA_DIR / f"{consts['name']}.npz"
    if not path.is_file():
        raise CameraError(f"tactile calibration file {path} is missing",
                          "reinstall alloy, or pass data_file= to a calibration .npz")
    table, bg, meta = _load_data(str(path))
    if tuple(meta["calib_size"]) != tuple(bg.shape[:2]) or table.shape[0] != 3 or table.shape[-1] != 6:
        raise CameraError(f"calibration {path} is malformed (table {table.shape}, background {bg.shape})",
                          "rebuild it with alloy.camera.tactile.calibrate")
    consts["calib_size"] = tuple(int(v) for v in meta["calib_size"])
    consts["mm_per_px"] = float(meta.get("mm_per_px", consts["mm_per_px"]))
    h, w = consts["calib_size"]
    consts["fit_box"] = tuple(float(v) for v in meta.get("fit_box", (0.0, 0.0, w - 1.0, h - 1.0)))
    return TactilePreset(table=table, background=bg, **consts)


def output_size(preset: TactilePreset, width: int | None, height: int | None) -> tuple[int, int]:
    """(H, W) of the rendered image: the preset default, or the given width/height keeping the gel's aspect.

    Example: ``output_size(load_preset("gelsight"), 160, None) == (120, 160)``.
    """
    ch, cw = preset.calib_size
    if width is None and height is None:
        return preset.default_size
    if width is not None and height is not None:
        if abs(width * ch - height * cw) > max(cw, ch):
            raise CameraError(f"tactile image {width}x{height} does not keep the gel's {cw}:{ch} aspect",
                              f"give only width= (height follows) or a {cw}:{ch} size")
        return int(height), int(width)
    if width is not None:
        return max(1, int(round(width * ch / cw))), int(width)
    return int(height), max(1, int(round(height * cw / ch)))


def gel_sag_mm(preset: TactilePreset, x_mm: float, y_mm: float) -> float:
    """How far a domed gel's surface lies behind the flat plane at (x, y) [mm] (0 for a flat gel)."""
    r = preset.gel_radius_mm
    if r <= 0.0:
        return 0.0
    d2 = x_mm * x_mm + y_mm * y_mm
    return r - math.sqrt(max(r * r - d2, 0.0))
