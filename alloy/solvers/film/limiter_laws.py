# SPDX-License-Identifier: Apache-2.0
"""Laws of the film's stretch limit as a force carrier (docs/known_gaps/film-projection-force.md).

A strip of HDPE film (40 x 80 mm, 15 um, 8 x 16 cells of 5 mm) hangs in the x-z plane from a pinned top row, a load
on its bottom row. A hung load must be carried by the pins, steadily, with the film at its cap plus its real elastic
stretch. Every measurement takes ``limiter``: ``None`` builds ``FilmConfig`` without the keyword (the default
projection) and reads the pins from ``projection_reaction``; a name passes ``limiter=`` and reads
``SolverFilm.limiter_force``. The pins' force is the pull they feel (-weight in z).

- L1 hang (0.002, 0.050, 0.500 kg on the bottom row, let go at the film's rest length; 1.0 s, then 0.05 s measured):
  - the largest (s1 - 1)/cap of any limited triangle is at most 1.01 + 2 e/cap, e = m g / (E h b), a real strip's
    elastic strain under the load (the stretch is not uniform: the pinned row cannot narrow; the limit itself is
    27 % stiff in a one-way pull, stress.py: the strip's mean stretch past the cap is 0.79-0.82 e);
  - the pins' mean force z is the hanging weight, less at most what VBD's soft membrane carries below the cap
    (``SOFT_SHARE`` x its pull at the cap, 2.6 mN here) and 0.5 %, and at most 1 % over it;
  - the load's centre of mass rises or falls at most 1 mm/s (the 0.5 kg load bounces on the film's own stiffness for
    0.7 s first), and no bottom particle moves faster than 30 mm/s (the strip sways as a pendulum at 2-3 mm/s, and
    the nine loaded particles move against each other across the pull at 2-20 mm/s: this model, not a real film,
    has next to no shear stiffness with the cross direction slack, stress.py; the projection left 80-480 mm/s);
  - the 0.002 kg hang passes the same bounds 1 m from the origin (float32 positions: a particle's force is
    quantised at 1.8 mN there, the pins' sum still reads 0.738 of the weight against 0.742 at the origin).
- L2 moment (0.050 kg on the 3 rightmost bottom particles; 0.1 s, then 0.4 s measured): the pins' moment about the
  top row's centre (y component, each pin's own force at its own place) is what the free film asks of them, the
  moment of its weight less its mass times acceleration (the off-centre load sways), within 2 % of W times half the
  strip's width on the mean of the steps.
- L3 pendulum (0.050 kg, strip rotated 20 degrees about the pin line, released at rest at the film's rest length,
  1.8 s): the period (mean of the first two full periods, from the bob row's zero crossings) is within 1 % of
  2 pi sqrt(L/g)(1 + theta^2/16), L the bob row's measured depth below the pins at its lowest (the rest length
  times 1 + cap, and the elastic stretch) and theta the measured swing after the catch (19.4 degrees). The model is
  soft below its cap (a real film is elastic there), so the load first falls the slack (cap x L, 3 mm) and the film
  catches it with one small bounce, which costs 55-60 % of m g L (1 - cos theta0) by 0.2 s; from 0.3 s on the
  kinetic plus potential energy never rises by more than 0.5 % of that swing energy and falls at most 15 % of it
  per period (the implicit step's own damping is 8 %).
- L4 release (0.050 kg, settled 0.5 s, then an upward force of the load's weight on the bottom row every step): after
  10 steps the pins' force z plus the film's own weight is within 2 % of the load's weight; over the next 50 steps
  the bottom row moves no faster than the film's elastic recoil, 1.2 g sqrt(m L / (E h b)) (the stretch's stored
  energy as the load's speed: 25 mm/s here, 30 with the margin).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .config import FilmConfig
from .emit import add_film_mesh
from .materials import HDPE_15UM
from .rigs import FilmRig, grid_mesh, pin

G = 9.81
DT = 1.0e-3
TOP_Z = 0.1
HALF_WIDTH = 0.02
LOADS = (0.002, 0.050, 0.500)
WIDTH, LENGTH = 0.04, 0.08
STRETCH_OVER, FORCE_OVER, FORCE_UNDER, SOFT_SHARE = 0.01, 0.01, 0.005, 2.5
SPEED_MAX, PARTICLE_SPEED_MAX = 1.0e-3, 30.0e-3
MOMENT_TOL = 0.02
PERIOD_TOL, GAIN_TOL, LOSS_MAX = 0.01, 0.005, 0.15
CATCH_S = 0.3                               # the pendulum's energy is read from here on
RELEASE_FORCE_TOL, RECOIL_MARGIN = 0.02, 1.2


def elastic_strain(load_kg: float) -> float:
    """The strip's mean elastic strain under the load: m g / (E h b)."""
    return load_kg * G / (HDPE_15UM.youngs * HDPE_15UM.thickness * WIDTH)


def stretch_bound(load_kg: float) -> float:
    return 1.0 + STRETCH_OVER + 2.0 * elastic_strain(load_kg) / HDPE_15UM.strain_cap


def soft_pull() -> float:
    """What VBD's soft membrane (membrane_scale of the film's stiffness) pulls with at the cap [N]."""
    f = HDPE_15UM
    return f.extras.membrane_scale * f.youngs * f.thickness / (1.0 - f.poisson ** 2) * f.strain_cap * WIDTH


def recoil_speed(load_kg: float) -> float:
    """The load's speed when the film's elastic stretch under it is let go: g sqrt(m L / (E h b))."""
    return G * math.sqrt(load_kg * LENGTH / (HDPE_15UM.youngs * HDPE_15UM.thickness * WIDTH))


@dataclass(frozen=True)
class Strip:
    rig: FilmRig
    top: np.ndarray
    bottom: np.ndarray
    free: np.ndarray
    limiter: str | None

    def pin_forces(self) -> np.ndarray:
        """Force on each pinned particle (N), from the limiter's own array."""
        s = self.rig.solver
        arr = s.projection_reaction if self.limiter is None else s.limiter_force
        return arr.numpy()[self.top]

    def pin_force(self) -> np.ndarray:
        return self.pin_forces().sum(axis=0)

    def free_weight(self) -> float:
        return G * float(self.rig.model.particle_mass.numpy()[self.free].sum())


def make_strip(limiter: str | None, load_kg: float, loaded: slice | None = None, angle_deg: float = 0.0,
               iterations: int = 10, x0: float = 0.0) -> Strip:
    """The strip; ``loaded`` picks bottom-row particles (by x order) carrying ``load_kg``; ``angle_deg`` rotates it
    rigidly about the pin line, into the y-z plane; ``x0`` moves it along x (float32 positions far from the origin)."""
    v, t = grid_mesh(8, 16, 5e-3, plane="xz")
    th = math.radians(angle_deg)
    d = v[:, 2].copy()
    v[:, 1], v[:, 2] = -d * math.sin(th), d * math.cos(th)
    top = np.flatnonzero(np.abs(d - d.max()) < 1e-9)
    bottom = np.flatnonzero(np.abs(d - d.min()) < 1e-9)
    bottom = bottom[np.argsort(v[bottom, 0])]
    carry = bottom if loaded is None else bottom[loaded]

    def build(b):
        p0, _ = add_film_mesh(b, v, t, HDPE_15UM, pos=(x0, 0.0, TOP_Z))
        pin(b, p0 + top)
        for i in p0 + carry:
            b.particle_mass[int(i)] += load_kg / len(carry)
        return {}

    cfg = FilmConfig(air_density=0.0) if limiter is None else FilmConfig(air_density=0.0, limiter=limiter)
    rig = FilmRig(build, dt=DT, iterations=iterations, film=cfg)
    m = rig.model.particle_mass.numpy()
    return Strip(rig, top, bottom, np.flatnonzero(m > 0.0), limiter)


def bottom_speed(s: Strip) -> float:
    return float(np.linalg.norm(s.rig.qd()[s.bottom], axis=1).max())


def load_speed(s: Strip) -> float:
    """Vertical speed of the bottom row's centre of mass."""
    return float(abs(s.rig.qd()[s.bottom].mean(axis=0)[2]))


@dataclass(frozen=True)
class HangResult:
    load_kg: float
    stretch_ratio: float     # largest (s1 - 1)/cap over limited triangles
    force_ratio: float       # pins' mean force z over minus the hanging weight
    weight: float            # the hanging weight, N
    speed: float             # the load's centre of mass, largest speed, m/s
    particle_speed: float    # bottom row's largest particle speed, m/s
    wall_s: float

    @property
    def force_low(self) -> float:
        """The least force ratio the law accepts: the soft membrane's share and FORCE_UNDER off."""
        return 1.0 - FORCE_UNDER - SOFT_SHARE * soft_pull() / self.weight

    @property
    def ok(self) -> bool:
        return (self.stretch_ratio <= stretch_bound(self.load_kg) and self.speed <= SPEED_MAX
                and self.force_low <= self.force_ratio <= 1.0 + FORCE_OVER
                and self.particle_speed <= PARTICLE_SPEED_MAX)


def measure_hang(limiter: str | None, load_kg: float, seconds: float = 1.05, x0: float = 0.0) -> HangResult:
    t0 = time.perf_counter()
    s = make_strip(limiter, load_kg, x0=x0)
    rig = s.rig
    rig.run(seconds - 0.05)
    stretch, fz, speed, fastest = [], [], [], []
    for _ in range(50):
        rig.step()
        stretch.append(float(rig.solver.strain_ratio(rig.s0).max()))
        fz.append(s.pin_force()[2])
        speed.append(load_speed(s))
        fastest.append(bottom_speed(s))
    ratio = float(np.mean(fz)) / -s.free_weight()
    return HangResult(load_kg, max(stretch), ratio, s.free_weight(), max(speed), max(fastest),
                      time.perf_counter() - t0)


@dataclass(frozen=True)
class MomentResult:
    residual: float          # |mean pin moment - mean moment the free film asks for| over W * half width
    step_rms: float          # the same per step, rms
    wall_s: float

    @property
    def ok(self) -> bool:
        return self.residual <= MOMENT_TOL


def measure_moment(limiter: str | None, load_kg: float = 0.050, settle: float = 0.1,
                   seconds: float = 0.4) -> MomentResult:
    """The pins' moment about the top row's centre against what the free film asks of them each step: the moment of
    its weight less its mass times acceleration (the off-centre load sways, so the weight alone is not it)."""
    t0 = time.perf_counter()
    s = make_strip(limiter, load_kg, loaded=slice(-3, None))
    rig = s.rig
    rig.run(settle)
    m = rig.model.particle_mass.numpy()[s.free][:, None]
    centre, g = np.array([HALF_WIDTH, 0.0, TOP_Z]), np.array([0.0, 0.0, -G])
    pins, asked, v0 = [], [], rig.qd()[s.free]
    for _ in range(int(round(seconds / DT))):
        rig.step()
        q, v1 = rig.q(), rig.qd()[s.free]
        asked.append(np.cross(q[s.free] - centre, m * (g - (v1 - v0) / DT)).sum(axis=0)[1])
        pins.append(np.cross(q[s.top] - centre, s.pin_forces()).sum(axis=0)[1])
        v0 = v1
    d, scale = np.asarray(pins) - np.asarray(asked), s.free_weight() * HALF_WIDTH
    return MomentResult(abs(float(d.mean())) / scale, float(np.sqrt((d ** 2).mean())) / scale,
                        time.perf_counter() - t0)


def _crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = np.flatnonzero(y[:-1] * y[1:] < 0.0)
    return t[k] + (t[k + 1] - t[k]) * y[k] / (y[k] - y[k + 1])


@dataclass(frozen=True)
class PendulumResult:
    period: float
    period_ref: float
    swing_deg: float         # the measured swing after the catch
    energy_gain: float         # largest later rise of the energy after the catch, over m g L (1 - cos theta0)
    loss_per_period: float   # largest energy lost over one period after the catch, over m g L (1 - cos theta0)
    wall_s: float

    @property
    def ok(self) -> bool:
        return (abs(self.period / self.period_ref - 1.0) <= PERIOD_TOL and self.energy_gain <= GAIN_TOL
                and self.loss_per_period <= LOSS_MAX)


def _swing_series(s: Strip, steps: int) -> dict[str, np.ndarray]:
    rig, m = s.rig, s.rig.model.particle_mass.numpy()[s.free]

    def sample() -> list[float]:
        q, qd = rig.q(), rig.qd()
        e = 0.5 * (m * (qd[s.free] ** 2).sum(1)).sum() + G * (m * q[s.free, 2]).sum()
        return [rig.t, q[s.bottom, 1].mean(), q[s.bottom, 2].mean(), qd[s.bottom, 1].mean(), e]

    rows = [sample()]
    for _ in range(steps):
        rig.step()
        rows.append(sample())
    return dict(zip(("t", "y", "z", "vy", "e"), np.asarray(rows).T))


def measure_pendulum(limiter: str | None, load_kg: float = 0.050, angle_deg: float = 20.0,
                     seconds: float = 1.8) -> PendulumResult:
    t0 = time.perf_counter()
    s = make_strip(limiter, load_kg, angle_deg=angle_deg)
    r = _swing_series(s, int(round(seconds / DT)))
    th0, nan = math.radians(angle_deg), float("nan")
    cross = _crossings(r["t"], r["y"])
    k0 = int(np.flatnonzero(r["y"][:-1] * r["y"][1:] < 0.0)[0]) if len(cross) else 0
    length = TOP_Z - r["z"][k0]
    swing = math.asin(min(1.0, float(np.abs(r["y"][r["t"] >= CATCH_S]).max()) / length))
    ref = 2.0 * math.pi * math.sqrt(length / G) * (1.0 + swing ** 2 / 16.0)
    period = float(np.mean([cross[2] - cross[0], cross[3] - cross[1]])) if len(cross) >= 5 else nan
    scale = load_kg * G * length * (1.0 - math.cos(th0))
    e = r["e"][r["t"] >= CATCH_S]                      # kinetic + potential energy once the film has caught the load
    gain = float((e - np.minimum.accumulate(e)).max() / scale)
    n = int(round(ref / DT))
    loss = float((e[:-n] - e[n:]).max() / scale) if len(e) > n else nan
    return PendulumResult(period, ref, math.degrees(swing), float(gain), loss, time.perf_counter() - t0)


@dataclass(frozen=True)
class ReleaseResult:
    load_kg: float
    force_error: float       # |pins' force z + film weight| over the load's weight, 10 steps after the release
    speed: float             # bottom row's largest speed over the next 50 steps, m/s
    wall_s: float

    @property
    def speed_bound(self) -> float:
        return RECOIL_MARGIN * recoil_speed(self.load_kg)

    @property
    def ok(self) -> bool:
        return self.force_error <= RELEASE_FORCE_TOL and self.speed <= self.speed_bound


def measure_release(limiter: str | None, load_kg: float = 0.050, settle: float = 0.5) -> ReleaseResult:
    t0 = time.perf_counter()
    s = make_strip(limiter, load_kg)
    rig = s.rig
    film_weight = s.free_weight() - load_kg * G
    rig.run(settle)

    def lift(r: FilmRig) -> None:
        f = r.s0.particle_f.numpy()
        f[s.bottom, 2] += load_kg * G / len(s.bottom)
        r.s0.particle_f.assign(f)

    rig.step(10, before=lift)
    err = abs(s.pin_force()[2] + film_weight) / (load_kg * G)
    speeds = []
    for _ in range(50):
        rig.step(1, before=lift)
        speeds.append(bottom_speed(s))
    return ReleaseResult(load_kg, float(err), max(speeds), time.perf_counter() - t0)
