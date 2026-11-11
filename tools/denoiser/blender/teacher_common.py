# SPDX-License-Identifier: Apache-2.0
"""Blender-side helpers for denoiser teacher renders (run inside Blender 4.5's Python).

- ``configure``: teacher settings on top of the look's own: exact spp (adaptive sampling off), motion blur off,
  OIDN on (its output is the baseline), denoising passes stored;
- ``render_exr``: one render -> a multilayer half EXR with ``Noisy Image`` (the input), ``Combined`` (Cycles' OIDN
  output of the same samples), ``Denoising Albedo``, ``Denoising Normal`` (camera space) and ``Denoising Depth``;
- ``view_settings``: the scene's colour management as a dict (saved next to the frames);
- ``bake_lut``: Blender's own display transform (view, look, exposure, white balance; dither off) sampled on a 3D
  grid into a 16-bit PNG, so training can apply the exact display transform on the GPU;
- ``bake_wb_matrix``: the scene-linear 3x3 white-balance matrix Blender applies before the view transform.

Example (inside Blender, after the scene is built)::

    import teacher_common as T
    T.configure(bpy.context.scene)
    T.render_exr(bpy.context.scene, Path("out/00042/in_s11_4"))
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import bpy
import numpy as np

log = logging.getLogger("denoiser.teacher")
LUT_N = 65  # grid points per axis
LUT_LO, LUT_HI = -12.0, 10.0  # shaper: u = (log2(x + 2^LO) - LO) / (HI - LO), x linear >= 0


def configure(scene: bpy.types.Scene) -> None:
    """Teacher settings: exact spp, no motion blur, OIDN on, denoising passes and depth stored.

    Example: ``configure(bpy.context.scene)`` right after alloy's ``build_scene``; render through
    :class:`FreshSessions` so every input keeps Cycles' blue-noise sampling."""
    c = scene.cycles
    c.use_adaptive_sampling = False
    c.use_denoising = True
    scene.render.use_motion_blur = False
    vl = bpy.context.view_layer
    vl.use_pass_z = True
    vl.cycles.denoising_store_passes = True


class FreshSessions:
    """A new Cycles session whenever the image size or the seed changes.

    Persistent data (alloy's builder turns it on) keeps one Cycles session across renders; after a size or seed change
    a reused session loses Cycles' blue-noise sampling and renders white noise, unlike any fresh render. Renders of
    one (size, seed) in ascending spp may share a session. Example::

        fresh = FreshSessions(scene)
        fresh.before(w, h, seed); render_exr(scene, stem); fresh.after()
    """

    def __init__(self, scene: bpy.types.Scene) -> None:
        self.scene, self.key, self.restore = scene, None, False

    def before(self, w: int, h: int, seed: int) -> None:
        key = (w, h, seed)
        self.restore = key != self.key and bool(self.scene.render.use_persistent_data)
        if key != self.key:
            self.scene.render.use_persistent_data = False   # this render: a fresh session, freed afterwards
        self.key = key

    def after(self) -> None:
        if self.restore:
            self.scene.render.use_persistent_data = True    # the next render of this key opens and keeps a new one
        self.restore = False


def set_format(scene: bpy.types.Scene, fmt: str, bits: int, mode: str) -> None:
    """Set the output image format, e.g. ``set_format(scene, "PNG", 8, "RGB")``."""
    s = scene.render.image_settings
    s.file_format, s.color_depth, s.color_mode = fmt, str(bits), mode
    if fmt == "PNG":
        s.compression = 15
    else:
        s.exr_codec = "ZIP"


def render_exr(scene: bpy.types.Scene, stem: Path, png: bool = False) -> tuple[float, float]:
    """Render the current frame to ``stem.exr`` (multilayer half; atomic rename) and, if ``png``, ``stem.png`` (the
    Combined pass through the scene's view transform). The EXR is renamed last, so an existing EXR means the render
    is complete. Returns (render_s, write_s).

    Example: ``render_exr(scene, Path("out/00042/ref"), png=True)``."""
    exr = stem.with_suffix(".exr")
    tmp_exr = stem.with_name(f".{stem.name}.part.exr")
    set_format(scene, "OPEN_EXR_MULTILAYER", 16, "RGBA")
    scene.render.filepath = str(tmp_exr)
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    t1 = time.perf_counter()
    if png:
        tmp_png = stem.with_name(f".{stem.name}.part.png")
        set_format(scene, "PNG", 8, "RGB")
        bpy.data.images["Render Result"].save_render(str(tmp_png), scene=scene)
        os.replace(tmp_png, stem.with_suffix(".png"))
    os.replace(tmp_exr, exr)
    return t1 - t0, time.perf_counter() - t1


def view_settings(scene: bpy.types.Scene) -> dict:
    """The scene's colour-management settings as a JSON-able dict.

    Example: ``Path("view.json").write_text(json.dumps(view_settings(scene)))``."""
    vs = scene.view_settings
    d = {"display_device": scene.display_settings.display_device, "view_transform": vs.view_transform,
         "look": vs.look, "exposure": vs.exposure, "gamma": vs.gamma, "use_curve_mapping": vs.use_curve_mapping,
         "dither_intensity": scene.render.dither_intensity}
    for k in ("use_white_balance", "white_balance_temperature", "white_balance_tint"):
        if hasattr(vs, k):
            d[k] = getattr(vs, k)
    if hasattr(vs, "white_balance_whitepoint"):
        d["white_balance_whitepoint"] = list(vs.white_balance_whitepoint)
    return d


def lut_grid(n: int = LUT_N) -> np.ndarray:
    """(n^3, 3) linear colours on the log shaper grid, index = r + n*g + n*n*b.

    Example: ``lut_grid(5).shape == (125, 3)``."""
    u = np.linspace(0.0, 1.0, n)
    x = np.exp2(u * (LUT_HI - LUT_LO) + LUT_LO) - np.exp2(LUT_LO)
    b, g, r = np.meshgrid(x, x, x, indexing="ij")
    return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1).astype(np.float32)


def _save_linear_image(scene: bpy.types.Scene, px: np.ndarray, w: int, h: int, out_png: Path) -> None:
    """Push linear RGBA pixels (h*w, 4) through save-as-render (16-bit PNG, dither off)."""
    img = bpy.data.images.new("dn_tmp", width=w, height=h, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Linear Rec.709"
    img.pixels.foreach_set(px.ravel())
    dither = scene.render.dither_intensity
    scene.render.dither_intensity = 0.0
    set_format(scene, "PNG", 16, "RGB")
    tmp = out_png.with_name(f".{out_png.name}.part.png")
    img.save_render(str(tmp), scene=scene)
    os.replace(tmp, out_png)
    scene.render.dither_intensity = dither
    bpy.data.images.remove(img)


def bake_lut(scene: bpy.types.Scene, out_png: Path, n: int = LUT_N) -> None:
    """Write the display LUT: pixel k (row-major from the top-left once Blender's bottom-up order is undone by the
    reader) holds display(grid[k]) as 16-bit display-encoded values; ``out_png.with_suffix(".json")`` holds the grid.

    Example: ``bake_lut(scene, Path("out/lut.png"))``."""
    grid = lut_grid(n)
    w = 1024
    h = (len(grid) + w - 1) // w
    px = np.zeros((h * w, 4), np.float32)
    px[: len(grid), :3] = grid
    px[:, 3] = 1.0
    _save_linear_image(scene, px, w, h, out_png)
    meta = {"n": n, "lo": LUT_LO, "hi": LUT_HI, "width": w, "height": h}
    out_png.with_suffix(".json").write_text(json.dumps(meta))


def emit(path: Path, row: dict) -> None:
    """Append one JSON row to ``path`` and log it. Example: ``emit(Path("timing.jsonl"), {"frame": 3})``."""
    line = json.dumps(row)
    log.info(line)
    with path.open("a") as f:
        f.write(line + "\n")


def _srgb_decode(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _read_back(path: Path) -> np.ndarray:
    back = bpy.data.images.load(str(path))
    back.colorspace_settings.name = "Non-Color"
    got = np.array(back.pixels[:], np.float64).reshape(-1, 4)
    bpy.data.images.remove(back)
    return got


def bake_wb_matrix(scene: bpy.types.Scene, out_json: Path, work: Path) -> list[list[float]]:
    """Measure the scene-linear white-balance matrix: grey 0.2 and grey + 0.1 * e_i through save-as-render with the
    Standard view (invertible sRGB), dither off, 16-bit. Writes ``{"matrix", "grey_residual"}`` to ``out_json``.

    Example: ``bake_wb_matrix(scene, Path("out/wb.json"), Path("out"))``."""
    vs = scene.view_settings
    saved = (vs.view_transform, vs.look, vs.exposure, vs.gamma)
    vs.view_transform, vs.look, vs.exposure, vs.gamma = "Standard", "None", 0.0, 1.0
    cols = np.array([[0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.2, 0.3, 0.2], [0.2, 0.2, 0.3]], np.float32)
    px = np.ones((4, 4), np.float32)
    px[:, :3] = cols
    tmp = work / ".dn_wb.png"
    _save_linear_image(scene, px, 4, 1, tmp)
    lin = _srgb_decode(_read_back(tmp)[:4, :3])
    tmp.unlink()
    m = np.stack([(lin[i + 1] - lin[0]) / 0.1 for i in range(3)], axis=1)  # columns = M e_i
    resid = float(np.abs(m @ cols[0] - lin[0]).max())
    vs.view_transform, vs.look, vs.exposure, vs.gamma = saved
    out_json.write_text(json.dumps({"matrix": m.tolist(), "grey_residual": resid}))
    return m.tolist()
