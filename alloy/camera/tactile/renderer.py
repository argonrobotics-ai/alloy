# SPDX-License-Identifier: Apache-2.0
"""Press maps -> tactile images, batched over worlds on the GPU (no simulation needed).

Use it directly to turn any press field into GelSight / DIGIT images, e.g. an engine's gel penetration field::

    import numpy as np
    from alloy.camera.tactile import TactileRenderer
    tr = TactileRenderer("gelsight", width=160, num_worlds=2, device="cpu")
    yy, xx = np.mgrid[0:tr.height, 0:tr.width]
    r2 = ((xx - 80.0) ** 2 + (yy - 60.0) ** 2) * tr.mm_per_px ** 2
    press = np.clip(np.sqrt(np.clip(4.0 - r2, 0, None)) - 1.2, 0, None)     # 4 mm ball, 0.8 mm deep [mm]
    img = tr.render(np.stack([press, 0 * press]).astype(np.float32))        # wp.array uint8 (2, 120, 160, 3)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..errors import CameraError
from .augment import Augment
from .compose import AugArrays, MarkerDraw, compose_kernel
from .markers import MarkerGrid, MarkerMotion, move_markers, place_markers
from .optics import GelOptics
from .presets import MarkerSpec, TactilePreset, load_preset, output_size

OUTPUTS = ("uint8", "all")


@dataclass
class TactileOutput:
    """Everything one render computed, on the device: the image and the fields behind it."""

    rgb: wp.array  # (worlds, H, W, 3) uint8, row 0 = top of the sensor image
    press: wp.array  # (worlds, H, W) float32 [mm]: how deep objects press into the undeformed gel
    gel: wp.array  # (worlds, H, W) float32 [mm]: the gel surface after Taxim's spreading
    markers: wp.array | None  # (worlds, markers) vec2 [mm]: displaced marker centres, sensor frame
    motion: wp.array | None  # (worlds,) vec4: stick-slip (dx_mm, dy_mm, twist_rad, active)


def resize_area(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """(H, W, C) float image resized to (height, width): block mean for integer factors, else bilinear."""
    h, w = img.shape[:2]
    if h % height == 0 and w % width == 0:
        return img.reshape(height, h // height, width, w // width, -1).mean(axis=(1, 3))
    ys = (np.arange(height) + 0.5) * h / height - 0.5
    xs = (np.arange(width) + 0.5) * w / width - 0.5
    rows = np.stack([np.interp(ys, np.arange(h), img[:, x, ch]) for x in range(w) for ch in range(img.shape[2])], -1)
    rows = rows.reshape(height, w, img.shape[2])
    return np.stack([[np.interp(xs, np.arange(w), rows[y, :, ch]) for ch in range(img.shape[2])]
                     for y in range(height)]).transpose(0, 2, 1)


def _marker_spec(preset: TactilePreset, markers) -> MarkerSpec | None:
    if markers is None:
        return preset.markers
    if markers is True:
        return preset.markers or MarkerSpec()
    if markers is False:
        return None
    if isinstance(markers, MarkerSpec):
        return markers
    raise CameraError(f"markers={markers!r}", "pass None (preset default), True, False or a MarkerSpec")


class TactileRenderer:
    """Optics + markers + augmentation for ``num_worlds`` images of one preset at one size."""

    def __init__(self, preset: str | TactilePreset = "gelsight", width: int | None = None, height: int | None = None,
                 num_worlds: int = 1, device: str | None = None, markers=None, augment: Augment | None = None,
                 motion: MarkerMotion | None = None, marker_rest_mm: np.ndarray | None = None):
        """``marker_rest_mm``: (M, 2) rest centres of a regular marker grid (e.g. the engine's), any order; the grid
        is rebuilt row-major (``grid.rest_mm``) and ``marker_order[j]`` is the input row of grid marker j.
        ``render(marker_disp=...)`` takes displacements in ``grid.rest_mm`` order. Default: the preset's grid."""
        self.preset = load_preset(preset)
        self.height, self.width = output_size(self.preset, width, height)
        self.num_worlds = int(num_worlds)
        self.device = str(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self.optics = GelOptics(self.preset, self.height, self.width, self.num_worlds, self.device)
        self.mm_per_px = self.optics.mm_per_px
        self.motion_model = motion or MarkerMotion()
        bg = resize_area(self.preset.background.astype(np.float64), self.height, self.width)
        self._bg = wp.array(bg.astype(np.float32), dtype=wp.vec3, device=self.device)
        self._spec = _marker_spec(self.preset, markers)
        self._setup_markers(marker_rest_mm)
        self.augment = augment or Augment()
        self.resample(self.augment.seed)
        self.rgb = wp.zeros((self.num_worlds, self.height, self.width, 3), dtype=wp.uint8, device=self.device)
        self._idle_motion = wp.zeros(self.num_worlds, dtype=wp.vec4, device=self.device)
        self._idle_center = wp.zeros(self.num_worlds, dtype=wp.vec2, device=self.device)
        self._no_field = wp.zeros((1, 1, 1), dtype=wp.vec2, device=self.device)
        self._frame = 0

    def _setup_markers(self, rest_mm: np.ndarray | None) -> None:
        spec, dev = self._spec, self.device
        self.grid, self.marker_order = None, None
        if spec and rest_mm is not None:
            self.grid, self.marker_order = MarkerGrid.from_points(rest_mm)
        elif spec:
            self.grid = MarkerGrid(spec.pitch_mm, self.preset.field_mm)
        m = self.grid.count if self.grid else 1
        self._rest = wp.array(self.grid.rest_mm if self.grid else np.zeros((1, 2), np.float32), dtype=wp.vec2,
                              device=dev)
        self.marker_pos = wp.zeros((self.num_worlds, m), dtype=wp.vec2, device=dev)
        d = MarkerDraw()
        d.pos = self.marker_pos
        d.enabled = 1 if spec else 0
        if spec:
            g = self.grid
            d.x0, d.y0, d.pitch_x, d.pitch_y, d.rows, d.cols = g.x0, g.y0, g.pitch_x, g.pitch_y, g.rows, g.cols
            d.radius, d.darkness = spec.radius_mm, spec.darkness
        else:
            d.x0 = d.y0 = d.radius = d.darkness = 0.0
            d.pitch_x, d.pitch_y, d.rows, d.cols = 1.0, 1.0, 0, 0
        self._draw = d

    def resample(self, seed: int | None = None) -> None:
        """Draw new per-world augmentation values (``seed`` None keeps the current one)."""
        if seed is not None:
            self.augment = self.augment.with_seed(seed)
        m = self.grid.count if self.grid else 1
        vals = self.augment.sample(self.num_worlds, self.preset.field_mm, m, self.preset.noise_std)
        a, dev = AugArrays(), self.device
        a.light = wp.array(vals["light"], dtype=wp.vec3, device=dev)
        a.bg_gain = wp.array(vals["bg_gain"], dtype=wp.vec3, device=dev)
        a.tone = wp.array(vals["tone"], dtype=wp.vec4, device=dev)
        a.shift = wp.array(vals["shift"], dtype=wp.vec2, device=dev)
        a.spots = wp.array(vals["spots"], dtype=wp.vec4, device=dev)
        a.marker_alpha = wp.array(vals["marker_alpha"], dtype=float, device=dev)
        self._aug = a

    def render(self, press, motion: wp.array | None = None, center: wp.array | None = None,
               field: wp.array | None = None, output: str = "uint8", seed: int | None = None,
               marker_disp=None):
        """Images for press maps ``press`` (worlds, H, W) [mm] (wp.array or numpy).

        Markers move by, in order of precedence: ``marker_disp`` (worlds, markers, 2) [mm], the engine's displacement
        of each marker (index = ``grid.rest_mm`` row); ``field``, an engine tangential displacement field
        (worlds, H, W) vec2 [mm]; else the built-in model, dilation from the press plus the stick-slip patch
        ``motion`` (worlds,) vec4 and ``center`` (worlds,) vec2 [mm] when given. Returns the uint8
        (worlds, H, W, 3) wp.array, or a :class:`TactileOutput` for ``output="all"``. The arrays are reused by the
        next render: copy what you keep.
        """
        self.prepare(press)
        return self.finish(motion, center, field, output, seed, marker_disp)

    def prepare(self, press, touched: wp.array | None = None) -> None:
        """First half of :meth:`render`: gel spreading and colour of ``press`` (worlds, H, W) [mm]; fills
        ``optics.stats`` (``touched``: the shape id per pixel, for the touched-shape summary)."""
        self._press = self._as_press(press)
        self._gel = self.optics.spread(self._press, touched)
        self.optics.shade(self._gel)

    def finish(self, motion: wp.array | None = None, center: wp.array | None = None, field: wp.array | None = None,
               output: str = "uint8", seed: int | None = None, marker_disp=None):
        """Second half of :meth:`render`: markers, augmentation and the uint8 image of the prepared press maps."""
        if output not in OUTPUTS:
            raise CameraError(f"output {output!r}", f"use one of {OUTPUTS}")
        if self.grid is not None and marker_disp is not None:
            self._place_markers(marker_disp)
        elif self.grid is not None:
            self._move_markers(self._press, motion, center, field)
        s = self._frame if seed is None else int(seed)
        self._frame += 1
        wp.launch(compose_kernel, dim=(self.num_worlds, self.height, self.width),
                  inputs=[self.optics.delta, self._bg, self._aug, self._draw, self.mm_per_px, s], outputs=[self.rgb],
                  device=self.device)
        if output == "uint8":
            return self.rgb
        return TactileOutput(self.rgb, self._press, self._gel, self.marker_pos if self.grid else None, motion)

    def _as_press(self, press) -> wp.array:
        shape = (self.num_worlds, self.height, self.width)
        if not isinstance(press, wp.array):
            press = wp.array(np.ascontiguousarray(press, np.float32), dtype=float, device=self.device)
        if tuple(press.shape) != shape:
            raise CameraError(f"press map has shape {tuple(press.shape)}", f"pass (worlds, H, W) = {shape} in mm")
        return press

    def _move_markers(self, press: wp.array, motion, center, field) -> None:
        use_field = field is not None
        if use_field and tuple(field.shape) != tuple(press.shape):
            raise CameraError(f"shear field has shape {tuple(field.shape)}",
                              f"pass a (worlds, H, W) vec2 field [mm], {tuple(press.shape)}")
        mp = self.motion_model
        max_disp = mp.max_disp_frac * self.grid.pitch
        wp.launch(move_markers, dim=(self.num_worlds, self.grid.count),
                  inputs=[self._rest, press, motion if motion is not None else self._idle_motion,
                          center if center is not None else self._idle_center,
                          field if use_field else self._no_field, 1 if use_field else 0, self.mm_per_px,
                          mp.as_vec4(), max_disp], outputs=[self.marker_pos], device=self.device)

    def _place_markers(self, disp) -> None:
        shape = (self.num_worlds, self.grid.count)
        if not isinstance(disp, wp.array):
            disp = np.ascontiguousarray(disp, np.float32)
            if disp.shape != (*shape, 2):
                raise CameraError(f"marker_disp has shape {disp.shape}", f"pass (worlds, markers, 2) = {(*shape, 2)} mm")
            disp = wp.array(disp, dtype=wp.vec2, device=self.device)
        elif tuple(disp.shape) != shape:
            raise CameraError(f"marker_disp has shape {tuple(disp.shape)}", f"pass a (worlds, markers) vec2 array {shape}")
        max_disp = self.motion_model.max_disp_frac * self.grid.pitch
        wp.launch(place_markers, dim=shape, inputs=[self._rest, disp, max_disp], outputs=[self.marker_pos],
                  device=self.device)
