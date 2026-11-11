# SPDX-License-Identifier: Apache-2.0
"""Tactile images from the engine's gel fields: the outputs of the physics side's ``GelSensor`` plug-in.

The plug-in (interface: ``docs/notes/wp13-engine10-plans.md`` section 6) writes, per world W and sensor S:
``depth`` [W, S, Hg, Wg] (m, gel indentation at cell centres, row 0 at +y) and ``marker_disp`` [W, S, M] vec3 (m,
displacement of each marker from ``marker_rest`` [M, 2] m). ``EngineTactile`` resamples the depth grid to the image
(bilinear, row order handled), takes the markers' tangential displacement, and renders W x S images with the
preset's optics and augmentation; image ``w * S + s`` is sensor s of world w.

Example (``gel`` an installed ``alloy.plugins.tactile.GelSensor``, after ``sim.step_period()``)::

    from alloy.camera.tactile.engine import EngineTactile
    view = EngineTactile.from_gel_sensor(gel, width=320)   # preset picked by the gel's mm per pixel
    img = view.render(gel.outputs())                        # wp.array uint8 (worlds * sensors, 240, 320, 3)
"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..errors import CameraError
from .renderer import TactileRenderer


@wp.kernel
def resample_depth(depth: wp.array4d(dtype=float), sensors: int, rows_up: int, out: wp.array3d(dtype=float)):
    """Image press [mm] from a cell grid of depth [m] (bilinear at pixel centres; rows_up: grid row 0 is +y)."""
    i, r, c = wp.tid()
    w = i // sensors
    s = i - w * sensors
    hg = depth.shape[2]
    wg = depth.shape[3]
    gy = (float(r) + 0.5) * float(hg) / float(out.shape[1]) - 0.5
    gx = (float(c) + 0.5) * float(wg) / float(out.shape[2]) - 0.5
    if rows_up != 0:
        gy = float(hg - 1) - gy
    gy = wp.clamp(gy, 0.0, float(hg - 1))
    gx = wp.clamp(gx, 0.0, float(wg - 1))
    r0 = int(gy)
    c0 = int(gx)
    r1 = wp.min(r0 + 1, hg - 1)
    c1 = wp.min(c0 + 1, wg - 1)
    fy = gy - float(r0)
    fx = gx - float(c0)
    top = depth[w, s, r0, c0] * (1.0 - fx) + depth[w, s, r0, c1] * fx
    bot = depth[w, s, r1, c0] * (1.0 - fx) + depth[w, s, r1, c1] * fx
    out[i, r, c] = wp.max(top * (1.0 - fy) + bot * fy, 0.0) * 1000.0


@wp.kernel
def marker_xy_mm(disp: wp.array3d(dtype=wp.vec3), sensors: int, order: wp.array(dtype=wp.int32),
                 out: wp.array2d(dtype=wp.vec2)):
    """Tangential displacement [mm] of grid marker j = engine marker order[j]."""
    i, j = wp.tid()
    w = i // sensors
    s = i - w * sensors
    d = disp[w, s, order[j]]
    out[i, j] = wp.vec2(d[0], d[1]) * 1000.0


class EngineTactile:
    """Renderer for a ``GelSensor``'s outputs: W worlds x S sensors of one preset."""

    def __init__(self, preset="gelsight", num_worlds: int = 1, sensors: int = 1, width: int | None = None,
                 marker_rest_m: np.ndarray | None = None, rows_up: bool = True, device: str | None = None, **kw):
        """``marker_rest_m``: the engine's marker rest positions [M, 2] (m); None renders without markers.
        ``rows_up``: the depth grid's row 0 is at +y (section 6 as written). ``kw``: TactileRenderer options
        (augment, motion)."""
        self.num_worlds, self.sensors, self.rows_up = int(num_worlds), int(sensors), bool(rows_up)
        rest = None if marker_rest_m is None else np.asarray(marker_rest_m, np.float64).reshape(-1, 2) * 1000.0
        self.renderer = TactileRenderer(preset, width, None, self.num_worlds * self.sensors, device,
                                        markers=rest is not None, marker_rest_mm=rest, **kw)
        r, dev = self.renderer, self.renderer.device
        self.press = wp.zeros((r.num_worlds, r.height, r.width), dtype=float, device=dev)
        self._disp = wp.zeros((r.num_worlds, r.grid.count if r.grid else 1), dtype=wp.vec2, device=dev)
        order = r.marker_order if r.marker_order is not None else np.zeros(1, np.int32)
        self._order = wp.array(order, dtype=wp.int32, device=dev)

    @classmethod
    def from_gel_sensor(cls, gel, preset: str | None = None, width: int | None = None, **kw) -> EngineTactile:
        """A view of an installed ``GelSensor`` (its worlds, sensors, marker layout and field of view). ``preset``:
        the optics; None picks the built-in preset whose gel matches the sensor's field of view."""
        from .presets import list_presets, load_preset
        fov = (gel.cols * gel.config.stride * gel.pixel * 1e3, gel.rows * gel.config.stride * gel.pixel * 1e3)
        names = [preset] if preset else list(list_presets())
        match = [n for n in names if np.allclose(load_preset(n).field_mm, fov, rtol=0.02)]
        if not match:
            raise CameraError(f"no tactile preset has a {fov[0]:.1f} x {fov[1]:.1f} mm gel (asked: {names})",
                              "pass preset= with a matching gel, or set the GelSensor's preset to gelsight / digit size")
        rest = gel.marker_xy if len(gel.marker_xy) else None
        return cls(match[0], int(gel.ctx.world_count), int(gel.S), width, marker_rest_m=rest,
                   device=str(gel.ctx.device), **kw)

    def render(self, outputs, output: str = "uint8", seed: int | None = None):
        """Images of ``outputs`` (``GelSensor.outputs()``: a dict or an object with ``depth`` and, for markers,
        ``marker_disp``). Returns wp.array uint8 (W*S, H, W, 3) or a TactileOutput (``output="all"``)."""
        dev = self.renderer.device
        depth = _as_device(_field(outputs, "depth"), float, dev)
        disp = _field(outputs, "marker_disp", required=False)
        disp = None if disp is None else _as_device(disp, wp.vec3, dev)
        want = (self.num_worlds, self.sensors)
        if depth.ndim != 4 or tuple(depth.shape[:2]) != want:
            raise CameraError(f"depth has shape {tuple(depth.shape)}", f"pass [W, S, Hg, Wg] with (W, S) = {want}")
        r = self.renderer
        wp.launch(resample_depth, dim=self.press.shape, inputs=[depth, self.sensors, int(self.rows_up)],
                  outputs=[self.press], device=r.device)
        marker_disp = None
        if r.grid is not None and disp is not None:
            if tuple(disp.shape) != (*want, len(r.marker_order)):
                raise CameraError(f"marker_disp has shape {tuple(disp.shape)}",
                                  f"pass [W, S, M] vec3 with M = {len(r.marker_order)} (the marker_rest count)")
            wp.launch(marker_xy_mm, dim=self._disp.shape, inputs=[disp, self.sensors, self._order],
                      outputs=[self._disp], device=r.device)
            marker_disp = self._disp
        return r.render(self.press, output=output, seed=seed, marker_disp=marker_disp)


def _field(outputs, name: str, required: bool = True):
    val = outputs.get(name) if isinstance(outputs, dict) else getattr(outputs, name, None)
    if val is None and required:
        raise CameraError(f"gel sensor outputs have no {name!r}", "pass GelSensor.outputs() (depth, marker_disp)")
    return val


def _as_device(arr, dtype, device: str) -> wp.array:
    """A warp array on ``device`` (numpy input: vec3 fields have a trailing axis of 3)."""
    if isinstance(arr, wp.array):
        return arr if str(arr.device) == device else arr.to(device)
    return wp.array(np.ascontiguousarray(arr, np.float32), dtype=dtype, device=device)
