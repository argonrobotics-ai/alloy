# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the species tasks (mix_colours.py, titration_colour.py): a stirrer, mixing and spill measures.

- **Stirrer.** A paddle on a vertical hinge, a position servo turned by a ramp target (constant speed): a hub disc
  and two crossed blades reaching into the liquid. It stands for a spoon or a magnetic stir bar, driven, not held.
- **Stir bar.** The same servo on a capsule lying on the floor (a magnetic stirrer; titration_colour.py).
- **Mixing (CV over portions).** The container's cavity is cut into ``nx`` x ``ny`` x ``nz`` portions; each species'
  concentration per portion is its amount over the volume of the particles in it. CV = standard deviation / mean
  over the portions holding liquid (Gym-Anything's mixing criterion: CV <= 0.10).
- **Particle CV.** The same over single particles (intensity of segregation; streaks finer than a portion
  cannot lower it).
- **Spill.** The fraction of active particles outside the cavity's footprint or below its floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HUB_Z = 0.050  # hub height above the cavity floor [m]


def stirrer_mjcf(radius: float, reach: float, blade: float = 0.003) -> str:
    """A hub disc on a vertical hinge ``spin`` with two crossed blades of half-width ``radius`` reaching ``reach``
    below the hub (capsules of radius ``blade``). Servo kp 0.5 N m/rad, kv 0.02 N m s/rad on 1e-5 kg m^2."""
    r, d, b = radius, reach, blade
    blades = "".join(f'<geom type="capsule" fromto="{x0} {y0} {-d} {x1} {y1} {-d}" size="{b}" mass="0.005"/>'
                     f'<geom type="capsule" fromto="{x0} {y0} 0 {x0} {y0} {-d}" size="{b}" mass="0.005"/>'
                     for x0, y0, x1, y1 in ((-r, 0, r, 0), (0, -r, 0, r)))
    return ('<mujoco><compiler angle="radian"/><worldbody><body name="hub" pos="0 0 0">'
            '<joint name="spin" type="hinge" axis="0 0 1" damping="0.001"/>'
            '<geom type="cylinder" fromto="0 0 0 0 0 0.004" size="0.010" mass="0.2"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 {-d}" size="{b}" mass="0.005"/>{blades}'
            '</body></worldbody><actuator><position joint="spin" kp="0.5" kv="0.02"/></actuator></mujoco>')


def stir_bar_mjcf(half_length: float = 0.010, radius: float = 0.0008, mass: float = 1.0) -> str:
    """A magnetic stir bar: a capsule of half-length ``half_length`` lying flat on a vertical hinge ``spin``, its
    position servo standing for the stirrer's rotating magnet, scaled to the bar's inertia I (kp = 5e4 I, kv = 2e3 I:
    the damping ratio of ``stirrer_mjcf``). Thin, so that set 4 mm above the floor (between the first two layers of an
    API liquid's 4 mm lattice) no particle starts inside it. Its base is welded.

    ``mass`` stands for the drive (magnet and rotor), not the bar: the magnet sets the bar's motion and the liquid's
    torque barely moves it. The gains scale with I, so the turning is the same at any mass. Margin: SolverDFSPH
    integrates an immersed body at its own mass inside each substep (sph/bodies.py); on main fb69a08b9 a 10 g bar in
    the titration flask went INVALID (SPEED_FAIL) within 0.02 s, on bbbb384c2 it stays healthy."""
    h, r = half_length, radius
    inertia = mass * (2.0 * h) ** 2 / 12.0
    return ('<mujoco><compiler angle="radian"/><worldbody><body name="bar" pos="0 0 0">'
            '<joint name="spin" type="hinge" axis="0 0 1" damping="0"/>'
            f'<geom type="capsule" fromto="{-h} 0 0 {h} 0 0" size="{r}" mass="{mass}"/>'
            f'</body></worldbody><actuator><position joint="spin" kp="{5e4 * inertia:.6g}" kv="{2e3 * inertia:.6g}"/>'
            '</actuator></mujoco>')


@dataclass(frozen=True)
class Cavity:
    """An axis-aligned container cavity: centre of its floor ``floor`` (x, y, z) and inner size (x, y, depth)."""

    floor: tuple
    size: tuple

    def lo(self) -> np.ndarray:
        return np.asarray(self.floor, float) - np.array([self.size[0] / 2, self.size[1] / 2, 0.0])

    def hi(self) -> np.ndarray:
        return np.asarray(self.floor, float) + np.array([self.size[0] / 2, self.size[1] / 2, self.size[2]])

    def inside(self, x: np.ndarray) -> np.ndarray:
        return np.all((x >= self.lo()) & (x <= self.hi()), axis=-1)


def portion_cv(x: np.ndarray, amount: np.ndarray, volume: np.ndarray, active: np.ndarray, cavity: Cavity,
               bins: tuple = (2, 2, 2), top: float | None = None) -> np.ndarray:
    """CV per species over the portions holding liquid. ``x`` [N, 3], ``amount`` [N, S], ``volume`` [N], ``active``
    [N] of one world; portions split the cavity up to ``top`` (default: the highest active particle)."""
    on = (active != 0) & cavity.inside(x)
    lo, hi = cavity.lo(), cavity.hi().copy()
    hi[2] = (x[on, 2].max() if top is None else top) + 1.0e-9
    cell = (x[on] - lo) / (hi - lo) * np.asarray(bins)
    k = np.clip(cell.astype(int), 0, np.asarray(bins) - 1)
    key = (k[:, 0] * bins[1] + k[:, 1]) * bins[2] + k[:, 2]
    n = bins[0] * bins[1] * bins[2]
    vol = np.bincount(key, weights=volume[on], minlength=n)
    full = vol > 0.0
    out = []
    for s in range(amount.shape[1]):
        c = np.bincount(key, weights=amount[on, s], minlength=n)[full] / vol[full]
        out.append(float(c.std() / c.mean()) if c.mean() > 0.0 else 0.0)
    return np.asarray(out)


def particle_cv(x: np.ndarray, amount: np.ndarray, volume: np.ndarray, active: np.ndarray,
                cavity: Cavity) -> np.ndarray:
    """Volume-weighted CV of each species' concentration over the active particles in the cavity (one world): the
    intensity of segregation at the particle scale, 1 for two unmixed halves, 0 when uniform."""
    on = (active != 0) & cavity.inside(x)
    v = volume[on]
    c = amount[on] / v[:, None]
    mean = (c * v[:, None]).sum(axis=0) / v.sum()
    std = np.sqrt(((c - mean) ** 2 * v[:, None]).sum(axis=0) / v.sum())
    return np.where(mean > 0.0, std / np.maximum(mean, 1e-300), 0.0)


def spill_fraction(x: np.ndarray, active: np.ndarray, cavity: Cavity) -> float:
    """Fraction of active particles outside the cavity's footprint or below its floor (one world); liquid above the
    rim (a pour in flight) is not spilled."""
    on = active != 0
    lo, hi = cavity.lo(), cavity.hi()
    held = np.all((x[:, :2] >= lo[:2]) & (x[:, :2] <= hi[:2]), axis=-1) & (x[:, 2] >= lo[2])
    return float((on & ~held).sum() / max(int(on.sum()), 1))
