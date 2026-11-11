# SPDX-License-Identifier: Apache-2.0
"""Mode tables for ``ModalSound``: analytic beams and plates, and ``modes.npz`` files (engine item 10, sound).

A ``ModeSet`` describes one sounding object in its body frame:
- ``freq_hz [K]``, ``zeta [K]`` (damping ratio, ζ = tan δ / 2 of the material), ``mass [K]`` (modal mass, kg);
- ``points [P, 3]``: sample points on the object (m, body frame);
- ``gains [K, P, 3]``: each mode's shape at each point as a vector (body frame). A contact force F at the point
  nearest to the contact drives mode k with the modal force ``gains[k, p] · F``. Shapes are normalised to unit
  mean square over the object, so the modal mass is the object's mass (ρAL for a beam, ρhab for a plate);
- ``radiation [K]``: each mode's weight in the monopole proxy (1 = unit mean-square surface motion);
- ``area``: the radiating area (m²).

Closed forms (Euler-Bernoulli; Blevins, *Formulas for natural frequency and mode shape*, 1979, tables 8-1 and
11-4, from memory, UNVERIFIED page numbers):
- beam: f_k = (β_k L)² / (2π L²) · sqrt(E I / (ρ A)), I = w t³ / 12 for bending across t;
  free-free: cos βL · cosh βL = 1; clamped-free: cos βL · cosh βL = -1;
- simply supported plate a x b x h: f_mn = (π/2) · ((m/a)² + (n/b)²) · sqrt(D / (ρ h)), D = E h³ / (12 (1 - ν²)),
  shape 2 sin(mπx/a) sin(nπy/b).
Shear and rotary inertia are ignored (slender beams and thin plates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Material:
    """Elastic constants and the loss factor tan δ, constant or a table over frequency (log interpolation, held
    beyond the ends)."""

    name: str
    youngs: float                                  # Pa
    poisson: float
    density: float                                 # kg/m³
    loss_hz: tuple[float, ...] = (1000.0,)
    loss_tan_delta: tuple[float, ...] = (1e-3,)
    source: str = ""

    def zeta(self, freq_hz: np.ndarray | float) -> np.ndarray:
        """Damping ratio ζ = tan δ / 2 at each frequency."""
        f = np.log(np.maximum(np.asarray(freq_hz, float), 1e-9))
        return 0.5 * np.interp(f, np.log(np.asarray(self.loss_hz)), np.asarray(self.loss_tan_delta))


@dataclass(frozen=True)
class ModeSet:
    """One sounding object's modes in its body frame (see the module docstring)."""

    freq_hz: np.ndarray
    zeta: np.ndarray
    mass: np.ndarray
    radiation: np.ndarray
    points: np.ndarray
    gains: np.ndarray
    area: float
    name: str = ""

    def __post_init__(self) -> None:
        k, p = len(self.freq_hz), len(self.points)
        if self.gains.shape != (k, p, 3):
            raise ValueError(f"{self.name}: gains {self.gains.shape}, expected ({k}, {p}, 3)")
        for field in ("zeta", "mass", "radiation"):
            if len(getattr(self, field)) != k:
                raise ValueError(f"{self.name}: {field} needs {k} entries")
        if k and (np.any(self.freq_hz <= 0) or np.any(self.mass <= 0) or np.any((self.zeta < 0) | (self.zeta >= 1))):
            raise ValueError(f"{self.name}: frequencies and masses must be > 0 and 0 <= zeta < 1")

    @property
    def count(self) -> int:
        return len(self.freq_hz)

    def below(self, freq_hz: float, limit: int | None = None) -> ModeSet:
        """The modes under ``freq_hz`` (drop those that would alias at the audio rate), lowest first, at most
        ``limit``."""
        order = np.argsort(self.freq_hz, kind="stable")
        keep = order[self.freq_hz[order] < freq_hz][:limit]
        return replace(self, freq_hz=self.freq_hz[keep], zeta=self.zeta[keep], mass=self.mass[keep],
                       radiation=self.radiation[keep], gains=self.gains[keep])

    def save(self, path: str | Path) -> None:
        np.savez(path, freq_hz=self.freq_hz, zeta=self.zeta, mass=self.mass, radiation=self.radiation,
                 points=self.points, gains=self.gains, area=np.float64(self.area), name=np.str_(self.name))

    @classmethod
    def load(cls, path: str | Path) -> ModeSet:
        """A ``modes.npz`` (the keys of ``save``; ``radiation`` defaults to 1, ``name`` to the file stem)."""
        with np.load(path, allow_pickle=False) as z:
            k = len(z["freq_hz"])
            return cls(np.asarray(z["freq_hz"], float), np.asarray(z["zeta"], float), np.asarray(z["mass"], float),
                       np.asarray(z["radiation"], float) if "radiation" in z else np.ones(k),
                       np.asarray(z["points"], float), np.asarray(z["gains"], float), float(z["area"]),
                       str(z["name"]) if "name" in z else Path(path).stem)


def concat(sets: list[ModeSet], name: str = "") -> ModeSet:
    """Modes of several parts of one body (same points), e.g. a beam's two bending planes."""
    pts = sets[0].points
    if any(s.points.shape != pts.shape or not np.allclose(s.points, pts) for s in sets):
        raise ValueError("concat: every set needs the same sample points")
    cat = lambda f: np.concatenate([getattr(s, f) for s in sets])  # noqa: E731
    return ModeSet(cat("freq_hz"), cat("zeta"), cat("mass"), cat("radiation"), pts, cat("gains"),
                   max(s.area for s in sets), name or sets[0].name)


# --------------------------------------------------------------------------------------------- beams

def beam_roots(n: int, clamped: bool) -> np.ndarray:
    """The first ``n`` nonzero β_k L of a free-free (cos·cosh = 1) or clamped-free (cos·cosh = -1) beam."""
    sign = -1.0 if clamped else 1.0
    f = lambda x: math.cos(x) - sign / math.cosh(x)  # noqa: E731  (cos x cosh x = sign, divided by cosh x)
    out = []
    for k in range(n):
        centre = (2 * k + (1 if clamped else 3)) * math.pi / 2
        out.append(brentq(f, centre - 0.6, centre + 0.6, xtol=1e-14))
    return np.asarray(out)


def beam_shape(bl: float, xi: np.ndarray, clamped: bool) -> np.ndarray:
    """Mode shape at ξ = x / L in [0, 1], numerically stable for large βL (hyperbolic terms as decaying
    exponentials). Free-free, or clamped at ξ = 0."""
    x = bl * xi
    s, c, e = math.sin(bl), math.cos(bl), math.exp(-bl)
    if clamped:   # cosh - cos - σ (sinh - sin), σ = (cosh βL + cos βL) / (sinh βL + sin βL)
        den = 1.0 - e * e + 2.0 * s * e                          # (sinh + sin)·2e^{-βL}
        sigma = (1.0 + e * e + 2.0 * c * e) / den
        one_m = (-2.0 * e * e + 2.0 * (s - c) * e) / den         # 1 - σ
        hyp = 0.5 * (one_m * np.exp(x) + (1.0 + sigma) * np.exp(-x))   # cosh x - σ sinh x
        return hyp - np.cos(x) + sigma * np.sin(x)
    den = 1.0 - e * e - 2.0 * s * e                              # (sinh - sin)·2e^{-βL}
    sigma = (1.0 + e * e - 2.0 * c * e) / den                    # (cosh - cos) / (sinh - sin)
    one_m = (-2.0 * e * e + 2.0 * (c - s) * e) / den             # 1 - σ
    hyp = 0.5 * (one_m * np.exp(x) + (1.0 + sigma) * np.exp(-x))
    return hyp + np.cos(x) - sigma * np.sin(x)


def _unit_ms(shape_fn, xi_fine: np.ndarray) -> float:
    v = shape_fn(xi_fine)
    return 1.0 / math.sqrt(float(np.mean(v * v)))


def beam_modes(length: float, width: float, thickness: float, mat: Material, n: int = 8, *,
               clamped: bool = False, axis: str = "x", bend: str = "z", samples: int = 65,
               centre: tuple[float, float, float] = (0.0, 0.0, 0.0), name: str = "beam") -> ModeSet:
    """Euler-Bernoulli bending modes of a rectangular beam along body ``axis``, bending along ``bend`` (the
    ``thickness`` direction). Free-free spans ``centre`` ± L/2; a cantilever is clamped at ``centre`` - L/2·axis."""
    a, d = AXES[axis], AXES[bend]
    if a == d:
        raise ValueError("beam: bend must differ from axis")
    area_xs = width * thickness
    c = math.sqrt(mat.youngs * thickness ** 2 / (12.0 * mat.density))       # sqrt(EI/ρA)
    bls = beam_roots(n, clamped)
    freq = bls ** 2 / (2.0 * math.pi * length ** 2) * c
    xi = np.linspace(0.0, 1.0, samples)
    pts = np.tile(np.asarray(centre, float), (samples, 1))
    pts[:, a] += (xi - 0.5) * length
    fine = np.linspace(0.0, 1.0, 4001)
    gains = np.zeros((n, samples, 3))
    for k, bl in enumerate(bls):
        fn = lambda u, bl=bl: beam_shape(bl, u, clamped)  # noqa: E731
        gains[k, :, d] = fn(xi) * _unit_ms(fn, fine)
    mass = np.full(n, mat.density * area_xs * length)
    return ModeSet(freq, mat.zeta(freq), mass, np.ones(n), pts, gains, length * width, name)


def plate_modes(a: float, b: float, h: float, mat: Material, n: int = 16, *, samples: tuple[int, int] = (17, 17),
                centre: tuple[float, float, float] = (0.0, 0.0, 0.0), name: str = "plate") -> ModeSet:
    """Simply supported rectangular plate in the body x-y plane (a along x, b along y), moving along body z: the
    ``n`` lowest (m, n) modes."""
    dflex = mat.youngs * h ** 3 / (12.0 * (1.0 - mat.poisson ** 2))
    c = math.sqrt(dflex / (mat.density * h))
    span = int(math.ceil(math.sqrt(n))) + 4
    pairs = sorted(((i / a) ** 2 + (j / b) ** 2, i, j) for i in range(1, span + 1) for j in range(1, span + 1))[:n]
    freq = np.array([0.5 * math.pi * s * c for s, _, _ in pairs])
    u, v = np.meshgrid(np.linspace(0.0, 1.0, samples[0]), np.linspace(0.0, 1.0, samples[1]), indexing="ij")
    pts = np.zeros((u.size, 3)) + np.asarray(centre, float)
    pts[:, 0] += (u.ravel() - 0.5) * a
    pts[:, 1] += (v.ravel() - 0.5) * b
    gains = np.zeros((n, u.size, 3))
    for k, (_, i, j) in enumerate(pairs):
        gains[k, :, 2] = 2.0 * np.sin(i * math.pi * u.ravel()) * np.sin(j * math.pi * v.ravel())
    mass = np.full(n, mat.density * h * a * b)
    return ModeSet(freq, mat.zeta(freq), mass, np.ones(n), pts, gains, a * b, name)


__all__ = ["AXES", "Material", "ModeSet", "beam_modes", "beam_roots", "beam_shape", "concat", "plate_modes"]
