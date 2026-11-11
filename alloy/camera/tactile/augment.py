# SPDX-License-Identifier: Apache-2.0
"""Per-world randomisation of tactile images: lighting, gel colour, tone, sensor offset, noise and gel wear.

TacSL (Akinola et al. 2024) found colour augmentation decisive for sim-to-real with vision-based tactile images
(87.7 % vs 27.2 % zero-shot placement). Every world draws its own values from ``seed`` once (``resample`` draws new
ones, e.g. per episode); camera noise is drawn fresh every frame.

Example::

    from alloy.camera.tactile.augment import Augment
    aug = Augment.strong(seed=3)           # the TacSL-style ranges below
    params = aug.sample(num_worlds=8, field_mm=(18.9, 14.2), num_markers=108)
    params["light"].shape                  # (8, 3) gain on the colour change of the pressed gel
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..errors import CameraError

MAX_SPOTS = 8


@dataclass(frozen=True)
class Augment:
    """Ranges of the per-world randomisation (each value is drawn uniformly in +-range unless noted)."""

    light: float = 0.0  # relative gain per channel on the colour change (LED strength and colour)
    background: float = 0.0  # relative gain per channel on the unpressed gel colour
    brightness: float = 0.0  # intensity offset, 0..255 units
    contrast: float = 0.0  # relative contrast around mid grey
    shift_px: float = 0.0  # background offset in output pixels (sensor-to-sensor variation)
    noise_std: float | None = None  # camera noise per pixel, 0..255 units; None = the preset's value
    wear: float = 0.0  # 0..1: worn dark spots on the gel (count and strength grow with wear)
    marker_loss: float = 0.0  # 0..1: fraction of markers faded or missing
    seed: int = 0

    @staticmethod
    def strong(seed: int = 0) -> Augment:
        """Ranges for sim-to-real training (TacSL-style colour augmentation, plus mild wear)."""
        return Augment(light=0.25, background=0.1, brightness=10.0, contrast=0.15, shift_px=3.0, noise_std=3.0,
                       wear=0.3, marker_loss=0.05, seed=seed)

    def with_seed(self, seed: int) -> Augment:
        """The same ranges with another seed. Example: ``aug.with_seed(episode)``."""
        return replace(self, seed=int(seed))

    def check(self) -> None:
        """Raise CameraError when a range is out of bounds."""
        if not 0.0 <= self.wear <= 1.0 or not 0.0 <= self.marker_loss <= 1.0:
            raise CameraError(f"wear={self.wear}, marker_loss={self.marker_loss}", "use values between 0 and 1")
        if min(self.light, self.background, self.brightness, self.contrast, self.shift_px) < 0.0:
            raise CameraError("a negative augmentation range", "give ranges as non-negative half-widths")
        if self.noise_std is not None and self.noise_std < 0.0:
            raise CameraError(f"noise_std={self.noise_std}", "use a noise std >= 0 (0..255 units)")

    def sample(self, num_worlds: int, field_mm: tuple[float, float], num_markers: int,
               default_noise: float = 2.0) -> dict[str, np.ndarray]:
        """Per-world values as host arrays (see the module docstring for their meaning)."""
        self.check()
        rng = np.random.default_rng(self.seed)
        n = int(num_worlds)

        def sym(scale: float, *shape: int) -> np.ndarray:
            return (rng.uniform(-1.0, 1.0, (n, *shape)) * scale).astype(np.float32)

        noise = default_noise if self.noise_std is None else self.noise_std
        tone = np.stack([sym(self.contrast), sym(self.brightness), np.full(n, noise, np.float32),
                         np.zeros(n, np.float32)], -1)
        return {"light": 1.0 + sym(self.light, 3), "bg_gain": 1.0 + sym(self.background, 3), "tone": tone,
                "shift": sym(self.shift_px, 2), "spots": self._spots(rng, n, field_mm),
                "marker_alpha": (rng.uniform(0, 1, (n, max(num_markers, 1))) >= self.marker_loss).astype(np.float32)}

    def _spots(self, rng: np.random.Generator, n: int, field_mm: tuple[float, float]) -> np.ndarray:
        """(worlds, MAX_SPOTS, 4): centre x, y [mm], radius [mm], darkening 0..1 (0 = no spot)."""
        out = np.zeros((n, MAX_SPOTS, 4), np.float32)
        if self.wear <= 0.0:
            return out
        w, h = field_mm
        count = rng.binomial(MAX_SPOTS, self.wear, n)
        for i in range(n):
            k = int(count[i])
            out[i, :k, 0] = rng.uniform(-0.5 * w, 0.5 * w, k)
            out[i, :k, 1] = rng.uniform(-0.5 * h, 0.5 * h, k)
            out[i, :k, 2] = rng.uniform(0.2, 1.2, k) * (0.5 + self.wear)
            out[i, :k, 3] = rng.uniform(0.1, 0.45, k) * self.wear
        return out
