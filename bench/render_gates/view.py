# SPDX-License-Identifier: Apache-2.0
"""Linear scene-referred RGB -> the 8-bit sRGB image Blender 4.5 saves (view transform, look, exposure, gamma, white
balance; dither 0). The gate converts every scored image with this one function, Cycles references and engines alike.

The view transforms (AgX and its looks, Filmic) come from the OpenColorIO config of an installed Blender, read at run
time: no Blender data ships here. Each (view, look) is baked once into a 257^3 LUT over a log2 shaper and cached
under ``~/.cache/alloy_render_gates/luts``. "Standard" with no look is the plain sRGB curve and needs no Blender.

Set ``BLENDER_OCIO`` to ``<blender>/<version>/datafiles/colormanagement/config.ocio`` when Blender is not under
``/opt/blender``. Needs ``PyOpenColorIO`` (BSD-3-Clause) for any view other than Standard.

Example:
    >>> import numpy as np
    >>> to_png(np.full((2, 2, 3), 0.18, np.float32), {"view_transform": "Standard", "look": "None"})[0, 0].tolist()
    [118, 118, 118]
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .whitebalance import temp_tint_from_whitepoint_rgb, white_point_matrix

N = 257
LUT_CACHE = Path.home() / ".cache" / "alloy_render_gates" / "luts"
OCIO_GLOBS = ("/opt/blender/*/datafiles/colormanagement/config.ocio",
              "/opt/blender*/*/datafiles/colormanagement/config.ocio")


class ViewError(RuntimeError):
    """The view cannot be built (no Blender OCIO config, no PyOpenColorIO); the message says what to install or set."""


@dataclass(frozen=True)
class ViewSpec:
    """Display settings of one scene, as Blender's ``view_settings`` (see :func:`from_view_json`)."""

    view: str = "Standard"
    look: str = "None"
    exposure: float = 0.0
    gamma: float = 1.0
    whitepoint: tuple[float, float, float] | None = None


def from_view_json(v: dict) -> ViewSpec:
    """ViewSpec from a reference folder's ``view.json`` (keys as Blender's view_settings).

    Example:
        >>> from_view_json({"view_transform": "AgX", "look": "AgX - Medium High Contrast"}).view
        'AgX'
    """
    wp = tuple(float(x) for x in v["white_balance_whitepoint"]) if v.get("use_white_balance") else None
    return ViewSpec(v.get("view_transform", "Standard"), v.get("look", "None") or "None",
                    float(v.get("exposure", 0.0)), float(v.get("gamma", 1.0)), wp)  # type: ignore[arg-type]


def ocio_config() -> Path:
    """Path of Blender's config.ocio: ``$BLENDER_OCIO`` or the newest ``/opt/blender*`` install."""
    env = os.environ.get("BLENDER_OCIO")
    if env:
        if not Path(env).exists():
            raise ViewError(f"BLENDER_OCIO={env} does not exist; point it at <blender>/<ver>/datafiles/colormanagement/config.ocio")
        return Path(env)
    import glob

    found = sorted({p for g in OCIO_GLOBS for p in glob.glob(g)})
    if not found:
        raise ViewError("no Blender OCIO config found; install Blender 4.5 under /opt/blender or set BLENDER_OCIO to "
                        "<blender>/<ver>/datafiles/colormanagement/config.ocio")
    return Path(found[-1])


def shaper(view: str) -> tuple[float, float, float]:
    """(log2 min, log2 max, linear-ramp fraction c0) of the LUT axis. AgX: its own 25-stop log2 allocation."""
    return (-12.47393, 12.5260688117, 0.0) if view == "AgX" else (-12.0, 8.0, 1.0 / (N - 1))


def _ocio():
    try:
        import PyOpenColorIO as ocio
    except ImportError as e:
        raise ViewError("views other than Standard need `pip install opencolorio`") from e
    return ocio


def input_matrix(view: str, cfg_path: Path) -> np.ndarray:
    """The 3x3 the view applies before its per-channel shaper (AgX: Rec.709 -> E-Gamut; others identity)."""
    if view != "AgX":
        return np.eye(3)
    ocio = _ocio()
    cfg = ocio.Config.CreateFromFile(str(cfg_path))
    cols = np.eye(3, dtype=np.float32)
    cfg.getProcessor("Linear Rec.709", "Linear FilmLight E-Gamut").getDefaultCPUProcessor().applyRGB(cols)
    return cols.T.astype(np.float64)


def _grid(view: str) -> np.ndarray:
    lo, hi, c0 = shaper(view)
    g = np.arange(N) / (N - 1)
    if c0 == 0.0:
        return 2.0 ** (lo + g * (hi - lo))
    x = np.where(g < c0, g / c0 * 2.0**lo, 2.0 ** (lo + (g - c0) / (1 - c0) * (hi - lo)))
    x[0] = 0.0
    return x


def bake(view: str, look: str, cfg_path: Path) -> np.ndarray:
    """LUT[r, g, b] = display-encoded RGB of A^-1 @ grid point, through Blender's display "sRGB" / view (+ look)."""
    ocio = _ocio()
    cfg = ocio.Config.CreateFromFile(str(cfg_path))
    group = ocio.GroupTransform()
    if look and look != "None":
        group.appendTransform(ocio.LookTransform(src="Linear Rec.709", dst="Linear Rec.709", looks=look))
    group.appendTransform(ocio.DisplayViewTransform(src="Linear Rec.709", display="sRGB", view=view))
    ax = _grid(view)
    r, g, b = np.meshgrid(ax, ax, ax, indexing="ij")
    p = np.stack([r, g, b], -1).reshape(-1, 3) @ np.linalg.inv(input_matrix(view, cfg_path)).T
    px = np.ascontiguousarray(p, np.float32)
    cfg.getProcessor(group).getDefaultCPUProcessor().applyRGB(px)
    return px.reshape(N, N, N, 3)


@lru_cache(maxsize=4)
def lut(view: str, look: str) -> tuple[np.ndarray, np.ndarray]:
    """(LUT, input matrix) for a view and look, baked once per Blender config and cached on disk."""
    cfg = ocio_config()
    key = hashlib.sha1(cfg.read_bytes() + f"|{view}|{look}|{N}".encode()).hexdigest()[:16]
    f = LUT_CACHE / f"{key}.npz"
    if not f.exists():
        LUT_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(f, lut=bake(view, look, cfg), a=input_matrix(view, cfg))
    d = np.load(f)
    return d["lut"], d["a"]


def colour_matrix(spec: ViewSpec) -> np.ndarray:
    """Exposure scale times white-point adaptation (scene linear, before the view).

    Example:
        >>> colour_matrix(ViewSpec(exposure=1.0))[0, 0]
        2.0
    """
    m = np.eye(3) * 2.0**spec.exposure
    if spec.whitepoint is None:
        return m
    return m @ white_point_matrix(*temp_tint_from_whitepoint_rgb(spec.whitepoint))


def srgb_encode(c: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 transfer function on [0, 1]."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _coords(v: np.ndarray, view: str) -> np.ndarray:
    lo, hi, c0 = shaper(view)
    xmin = 2.0**lo
    with np.errstate(divide="ignore", invalid="ignore"):
        s = c0 + (np.log2(np.maximum(v, 1e-30)) - lo) / (hi - lo) * (1.0 - c0)
    s = np.where(v < xmin, v / xmin * c0, np.minimum(s, 1.0))
    return np.where(v <= 0.0, 0.0, s) * (N - 1)


def trilinear(table: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Trilinear lookup of ``table`` (N, N, N, 3) at fractional node coordinates ``p`` (..., 3)."""
    i0 = np.minimum(p.astype(np.int64), N - 2)
    f = p - i0
    out = np.zeros(p.shape, np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1]) * \
                    (f[..., 2] if dz else 1 - f[..., 2])
                out += w[..., None] * table[i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz]
    return out


def to_png(linear: np.ndarray, view: dict | ViewSpec) -> np.ndarray:
    """Linear RGB ``(H, W, 3)`` float -> ``(H, W, 3) uint8`` as Blender saves it with these view settings, dither 0.

    ``view`` is a reference folder's ``view.json`` dict or a :class:`ViewSpec`.

    Example:
        >>> import numpy as np
        >>> to_png(np.zeros((1, 1, 3), np.float32), {}).tolist()
        [[[0, 0, 0]]]
    """
    spec = view if isinstance(view, ViewSpec) else from_view_json(view)
    rgb = np.asarray(linear, np.float64)[..., :3]
    plain = spec.view == "Standard" and spec.look in ("None", "")
    table, a = (None, np.eye(3)) if plain else lut(spec.view, spec.look)
    v = rgb @ (a @ colour_matrix(spec)).T
    d = srgb_encode(v) if plain else trilinear(table, _coords(v, spec.view))
    if spec.gamma != 1.0:
        d = np.power(np.maximum(d, 0.0), 1.0 / spec.gamma)
    return np.clip(np.floor(d * 255.0 + 0.5), 0, 255).astype(np.uint8)


def load_exr(path: Path) -> np.ndarray:
    """Linear RGB of an EXR as ``(H, W, 3) float32`` (needs ``pip install OpenEXR``)."""
    try:
        import OpenEXR
    except ImportError as e:
        raise ViewError("reading EXR needs `pip install OpenEXR`") from e
    ch = OpenEXR.File(str(path)).channels()
    if "RGB" in ch:
        return np.ascontiguousarray(ch["RGB"].pixels[..., :3], dtype=np.float32)
    return np.stack([ch[k].pixels for k in "RGB"], -1).astype(np.float32)


def exr_folder_to_png(root: Path, engines: list[str] | None = None) -> int:
    """Rebuild every gate PNG under ``root/<scene>/<frame>/<engine>/`` from the EXR beside it, with that folder's
    view.json (Standard when there is none). Returns the number of PNGs written.

    Example:
        >>> exr_folder_to_png(Path("refs"))  # doctest: +SKIP
        2050
    """
    import json

    from .layout import parse_name
    from .pngio import write_png

    n = 0
    for eng in sorted(p for p in root.glob("*/*/*") if p.is_dir() and (not engines or p.name in engines)):
        vj = eng / "view.json"
        spec = from_view_json(json.loads(vj.read_text())) if vj.exists() else ViewSpec()
        for exr in sorted(eng.glob("*.exr")):
            if parse_name(exr.with_suffix(".png").name):
                write_png(exr.with_suffix(".png"), to_png(load_exr(exr), spec))
                n += 1
    return n
