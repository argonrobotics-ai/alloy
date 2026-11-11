# SPDX-License-Identifier: Apache-2.0
"""Paint optics for the painting tasks: Kubelka-Munk reflectance of a dry film over a substrate, and the hide test.

Each linear-RGB channel is a two-flux Kubelka-Munk medium (Kubelka & Munk 1931; Kubelka 1948). The paint's
absorption-to-scattering ratio comes from the colour of an infinitely thick film, K/S = (1 - R∞)²/(2R∞). Its
scattering per micrometre of dry film, S, is the paint's hiding power. A layer X µm thick over a background of
reflectance R_g reflects

    R = (1 - R_g·(a - b·coth(bSX))) / (a - R_g + b·coth(bSX)),   a = 1 + K/S,   b = sqrt(a² - 1).

A cell is **hidden** when its reflectance is within ΔE*ab 1.0 (CIE 1976) of the paint's own R∞, over whatever is
under it. This is the criterion of the Gym-Anything painting tasks (ASTM D2805's contrast ratio of 0.98 is about one
just-noticeable difference). A dark paint hides a light wall at a fraction of the DFT a white base needs, so no single
DFT threshold fits every colour.

Example::

    from alloy.tasks.paint_optics import Optics, srgb8_to_linear
    naval = Optics(srgb8_to_linear((47, 61, 76)), s_per_um=0.0037)
    hidden = naval.hidden(dft_um, srgb8_to_linear((110, 112, 114)))     # bool per cell
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIDE_DE = 1.0               # ΔE*ab: hidden when the stack is this close to the paint's own colour
_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],      # IEC 61966-2-1 sRGB primaries, D65 white
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])  # fmt: skip
_WHITE = _SRGB_TO_XYZ @ np.ones(3)


def srgb8_to_linear(rgb) -> np.ndarray:
    """8-bit sRGB to linear reflectance per channel (IEC 61966-2-1 transfer curve)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lab(linear) -> np.ndarray:
    """CIELAB (D65) of linear sRGB reflectance [..., 3]."""
    xyz = np.asarray(linear, dtype=np.float64) @ _SRGB_TO_XYZ.T / _WHITE
    eps = (6.0 / 29.0) ** 3
    f = np.where(xyz > eps, np.cbrt(xyz), xyz / (3.0 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1)


def delta_e(a, b) -> np.ndarray:
    """CIE 1976 colour difference ΔE*ab between linear reflectances."""
    return np.linalg.norm(lab(a) - lab(b), axis=-1)


def k_over_s(r_inf) -> np.ndarray:
    """Kubelka-Munk K/S of an opaque film of reflectance ``r_inf`` (clipped off 0 and 1)."""
    r = np.clip(np.asarray(r_inf, dtype=np.float64), 1e-4, 0.9999)
    return (1.0 - r) ** 2 / (2.0 * r)


def km_reflectance(ks, s_per_um: float, x_um, r_g) -> np.ndarray:
    """Reflectance [..., 3] of a layer ``x_um`` µm thick (K/S ``ks`` per channel, scattering ``s_per_um``) over a
    background ``r_g`` [..., 3]. A layer of 1e-6 µm or less leaves the background."""
    ks = np.asarray(ks, dtype=np.float64)
    x = np.asarray(x_um, dtype=np.float64)[..., None]
    r_g = np.asarray(r_g, dtype=np.float64)
    a = 1.0 + ks
    b = np.sqrt(np.maximum(a * a - 1.0, 1e-12))
    coth = 1.0 / np.tanh(np.clip(b * s_per_um * x, 1e-9, 50.0))
    r = (1.0 - r_g * (a - b * coth)) / (a - r_g + b * coth)
    return np.where(x <= 1e-6, r_g, r)


@dataclass(frozen=True)
class Optics:
    """A paint's colour as an opaque film (linear RGB) and its scattering per µm of dry film."""

    r_inf: np.ndarray
    s_per_um: float

    @property
    def ks(self) -> np.ndarray:
        return k_over_s(self.r_inf)

    def reflectance(self, dft_um, r_g) -> np.ndarray:
        return km_reflectance(self.ks, self.s_per_um, dft_um, r_g)

    def delta_e(self, dft_um, r_g) -> np.ndarray:
        """ΔE*ab of the film over ``r_g`` from the paint's own colour, per cell."""
        return delta_e(self.reflectance(dft_um, r_g), self.r_inf)

    def hidden(self, dft_um, r_g, limit: float = HIDE_DE) -> np.ndarray:
        return self.delta_e(dft_um, r_g) <= limit

    def hide_dft_um(self, r_g, limit: float = HIDE_DE, hi: float = 2000.0) -> float:
        """The least DFT [µm] that hides a uniform background ``r_g`` (bisection; ``hi`` if even that does not)."""
        lo, top = 0.0, hi
        if not bool(self.hidden(np.array(top), np.asarray(r_g, dtype=np.float64))):
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + top)
            lo, top = (lo, mid) if bool(self.hidden(np.array(mid), np.asarray(r_g, dtype=np.float64))) else (mid, top)
        return top


__all__ = ["HIDE_DE", "Optics", "delta_e", "k_over_s", "km_reflectance", "lab", "srgb8_to_linear"]
