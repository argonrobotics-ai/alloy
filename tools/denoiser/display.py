# SPDX-License-Identifier: Apache-2.0
"""Exact display transforms for scoring: linear radiance -> the 8-bit image Blender would save.

    from display import Display
    disp = Display.from_job("packed/c2gate_v000_wide")       # reads meta.json: view settings + white balance
    u8 = disp(linear)                                         # (H, W, 3) or (1, H, W, 3) float -> (H, W, 3) uint8

Standard view without a look or white balance is the sRGB curve (bit-exact with Blender's PNGs). Other views (AgX
and its looks) go through OpenColorIO with Blender's own ``config.ocio`` after the white-balance matrix measured at
render time (``pack.py`` stores it): within 1 code value of Blender's PNGs. The baked LUT used for training is
within a few codes, which matters only for the strictest gate bars. Needs ``opencolorio`` for non-Standard views and
Blender's colour-management folder (``BLENDER_OCIO``, default: the Blender install's ``config.ocio``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OCIO_CANDIDATES = ("/opt/blender/4.5/datafiles/colormanagement/config.ocio",)


def srgb_u8(lin: np.ndarray) -> np.ndarray:
    """Linear -> 8-bit sRGB, clipped, round half up (Blender's Standard view, dither off)."""
    v = np.clip(np.asarray(lin, np.float32), 0.0, 1.0)
    f = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    return np.floor(f * 255.0 + 0.5).astype(np.uint8)


def ocio_config_path() -> str:
    """Blender's OCIO config: $BLENDER_OCIO, else a known install path; raises with what to set otherwise."""
    for p in (os.environ.get("BLENDER_OCIO"), *OCIO_CANDIDATES):
        if p and Path(p).exists():
            return p
    raise FileNotFoundError("Blender's config.ocio not found: set BLENDER_OCIO to "
                            "<blender>/<version>/datafiles/colormanagement/config.ocio")


@dataclass
class Display:
    """One scene's display transform; call it on linear RGB. Example: ``Display.from_job(job_dir)(img)``."""

    view: dict
    wb: np.ndarray

    @classmethod
    def from_job(cls, job_dir: str | Path) -> Display:
        meta = json.loads((Path(job_dir) / "meta.json").read_text())
        return cls(meta["view"], np.asarray(meta["wb"], np.float64))

    @property
    def is_standard(self) -> bool:
        look = self.view.get("look") or "None"
        return (self.view["view_transform"] == "Standard" and look == "None" and abs(self.view["exposure"]) < 1e-9
                and np.allclose(self.wb, np.eye(3), atol=1e-4))

    def __call__(self, lin: np.ndarray) -> np.ndarray:
        x = np.asarray(lin, np.float32)
        x = x[0] if x.ndim == 4 else x
        if self.is_standard:
            return srgb_u8(x)
        x = (x.reshape(-1, 3) @ self.wb.T.astype(np.float32)) * np.float32(2.0 ** self.view["exposure"])
        y = np.ascontiguousarray(x, np.float32)
        self._processor().applyRGB(y)
        return np.floor(np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(lin.shape[-3:])

    def _processor(self):
        if getattr(self, "_proc", None) is None:
            import PyOpenColorIO as OCIO
            cfg = OCIO.Config.CreateFromFile(ocio_config_path())
            t = OCIO.DisplayViewTransform()
            t.setSrc("Linear Rec.709")
            t.setDisplay(self.view["display_device"])
            t.setView(self.view["view_transform"])
            pipe = OCIO.LegacyViewingPipeline()
            pipe.setDisplayViewTransform(t)
            if (self.view.get("look") or "None") != "None":
                pipe.setLooksOverrideEnabled(True)
                pipe.setLooksOverride(self.view["look"])
            self._proc = pipe.getProcessor(cfg).getDefaultCPUProcessor()
        return self._proc
