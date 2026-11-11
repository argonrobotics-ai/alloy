# SPDX-License-Identifier: Apache-2.0
"""A handheld / presentation barcode imager on ``alloy.camera``: the 1D/2D area imager of a Zebra DS2208-class scanner
(640 x 480 mono at 30 Hz, 32.8 x 24.8 deg), as Gym-Anything's barcode sim models it (their ``params.py`` and
``optics.py``, MIT; each constant below names its source there).

A frame: the camera renders linear RGB (path traced, every world at once) -> sensor response ``L = 0.35 R + 0.45 G +
0.20 B`` -> lens blur (diffraction, sigma = 0.42 lambda / A, plus thin-lens defocus at the subject distance) ->
exposure (photoelectrons, auto-exposed so the centre 30 % window averages 110 DN) -> Poisson shot noise and 8 e- read
noise -> 8-bit DN.

Approximations (disclosed): the defocus is one Gaussian per world at the given subject distance (labels are flat
and near one depth; ``alloy.camera`` renders a pinhole), its sigma a quarter of the blur-circle diameter (the
standard deviation of a uniform disc); auto-exposure is ideal (set from the frame itself, gain 1); a sun from the
scanner's side with no sky (``barcode_scene.scanner_lighting``) stands in for the scanner's red LEDs, which
outshine the room 10:1 in their model (a sky-lit label at 8 samples per pixel is 40 % Monte Carlo noise, sun-lit
4 %); no aim line, no motion blur. zxing alone reads a 100 % EAN-13 to about 0.18 m here (0.22 m in some runs);
their own sub-pixel 1-D front end reaches 0.38 m.

Poses are of the scanner's window; the camera sits at the entrance pupil, 50 mm behind it on the optical axis.

``crop`` renders only the centre (w, h) pixels of the sensor at the same pixel scale (the field of view shrinks with
it): CPU tests render 320 x 240 of the 640 x 480 sensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImagerSpec:
    """DS2208-class imager constants (Gym-Anything barcode_scanning/params.py)."""

    width: int = 640                 # WIDTH_PX
    height: int = 480                # HEIGHT_PX
    hfov_deg: float = 32.8           # HFOV_DEG (DS2208 PRG)
    pupil_m: float = 0.41e-3         # PUPIL_DIAMETER_M, fitted to the PRG decode-range table
    focus_m: float = 0.232           # FOCUS_DISTANCE_M, from the entrance pupil
    pupil_behind_window_m: float = 0.050
    wavelength_m: float = 645e-9     # LED wavelength
    weights: tuple = (0.35, 0.45, 0.20)   # SENSOR_WEIGHTS (mono sensor's response to R, G, B)
    full_well_e: float = 6000.0
    read_noise_e: float = 8.0
    target_dn: float = 110.0         # AE_TARGET_DN on the centre 30 % window
    frame_hz: float = 30.0

    def ifov(self) -> float:
        """Angle per pixel (rad) at the image centre."""
        return 2.0 * math.tan(math.radians(self.hfov_deg) / 2.0) / self.width

    def blur_sigma_px(self, distance_m: float) -> float:
        """Gaussian sigma (px) of diffraction and defocus for a subject ``distance_m`` from the scanner window."""
        d = max(distance_m + self.pupil_behind_window_m, 1e-3)
        diffraction = 0.42 * self.wavelength_m / self.pupil_m / self.ifov()
        defocus = self.pupil_m * abs(1.0 / d - 1.0 / self.focus_m) / self.ifov() / 4.0
        return math.hypot(diffraction, defocus)

    def crop_fov_deg(self, crop_w: int) -> float:
        """Horizontal field of view of the centre ``crop_w`` pixels."""
        return math.degrees(2.0 * math.atan(math.tan(math.radians(self.hfov_deg) / 2.0) * crop_w / self.width))


DS2208 = ImagerSpec()


def _sensor_render():
    """Path-tracer settings of a sensor frame: a 1-pixel box filter (a pixel's aperture; Cycles' 1.5-pixel
    Blackman-Harris lost a 100 % EAN-13 beyond 0.19 m where the box still read it at 0.22 m, in CPU runs) and their
    imager's 3 bounces."""
    from alloy.camera.settings import RenderSettings

    return RenderSettings(max_bounces=3, filter_type="BOX", filter_width=1.0)


def window_pose(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """(7,) pose of a window at ``eye`` looking at ``target`` (camera convention: looks along -z, +y up)."""
    from scipy.spatial.transform import Rotation

    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(np.asarray(up, np.float64), back)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross((0.0, 1.0, 0.0), back)
    right /= np.linalg.norm(right)
    rot = np.stack([right, np.cross(back, right), back], axis=1)
    return np.concatenate([eye, Rotation.from_matrix(rot).as_quat()]).astype(np.float32)


def pupil_pose(pose, behind_m: float):
    """Window pose(s) (7,) or (worlds, 7) moved ``behind_m`` back along the optical axis (+z): the pupil."""
    if pose is None:
        return None
    from scipy.spatial.transform import Rotation

    p = np.array(pose, np.float64, ndmin=2)
    p[:, :3] += Rotation.from_quat(p[:, 3:]).apply((0.0, 0.0, behind_m))
    return p.astype(np.float32).reshape(np.shape(pose))


class Imager:
    """The scanner's imager over every world. ``attach_to``: a body index per world (the scanner's body), with
    ``pose`` (7,) the window's pose in that body's frame, looking along its -z (camera convention); or a fixed world
    ``pose`` / ``look_at`` (a presentation scanner). ``crop``: render the centre (w, h) of the sensor."""

    def __init__(self, spec: ImagerSpec = DS2208, *, pose=None, look_at=None, attach_to=None,
                 crop: tuple[int, int] | None = None, spp: int = 8, seed: int = 0):
        from alloy.camera import Camera

        self.spec = spec
        w, h = crop or (spec.width, spec.height)
        if w > spec.width or h > spec.height:
            raise ValueError(f"crop {crop} is larger than the {spec.width} x {spec.height} sensor")
        self.size = (int(w), int(h))
        if look_at is not None:
            pose, look_at = window_pose(*look_at), None
        self.camera = Camera("scanner", pose=pupil_pose(pose, spec.pupil_behind_window_m), attach_to=attach_to,
                             width=w, height=h, fov=spec.crop_fov_deg(w), fit="HORIZONTAL", spp=spp,
                             clip=(0.005, 50.0), settings=_sensor_render())
        self.rng = np.random.default_rng(seed)
        self.seed, self.frames, self.renders = int(seed), 0, 0

    def linear(self, scene, state) -> np.ndarray:
        """(worlds, h, w) linear mono radiance (the sensor's response), lens blur not applied. Every render draws new
        Monte Carlo noise (a new tracer seed), as successive frames of a real sensor do."""
        rgb = self.camera.render(scene, state, output="linear", seed=self.seed * 100003 + self.renders).numpy()
        self.renders += 1
        return np.tensordot(rgb[..., :3], np.asarray(self.spec.weights, np.float32), axes=([-1], [0]))

    def expose(self, mono: np.ndarray, distance_m, noise: bool = True) -> np.ndarray:
        """(worlds, h, w) uint8 frames from linear mono images, subjects ``distance_m`` (scalar or per world) away."""
        from scipy.ndimage import gaussian_filter

        s = self.spec
        dist = np.broadcast_to(np.asarray(distance_m, np.float64), (mono.shape[0],))
        out = np.empty(mono.shape, np.uint8)
        for w in range(mono.shape[0]):
            img = gaussian_filter(np.maximum(mono[w].astype(np.float64), 0.0), s.blur_sigma_px(float(dist[w])),
                                  mode="nearest")
            hh, ww = img.shape
            roi = img[int(0.35 * hh):int(0.65 * hh), int(0.35 * ww):int(0.65 * ww)]
            k = s.target_dn / 255.0 * s.full_well_e / max(float(roi.mean()), 1e-12)
            electrons = k * img
            if noise:
                electrons = self.rng.poisson(electrons).astype(np.float64)
                electrons += self.rng.normal(0.0, s.read_noise_e, img.shape)
            out[w] = np.clip(np.round(255.0 * electrons / s.full_well_e), 0, 255).astype(np.uint8)
        self.frames += 1
        return out

    def frame(self, scene, state, distance_m, noise: bool = True) -> np.ndarray:
        """One frame per world: (worlds, h, w) uint8."""
        return self.expose(self.linear(scene, state), distance_m, noise)


__all__ = ["DS2208", "Imager", "ImagerSpec", "pupil_pose", "window_pose"]
