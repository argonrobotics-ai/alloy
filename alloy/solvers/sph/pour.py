# SPDX-License-Identifier: Apache-2.0
"""R7 pour twin (bench §4 R7, SPEC §5.2 LIQ-7): a kinematic cup tilted about its rim at a constant rate pours into a
receiver on a scale.

Geometry comes from the rig card (``RigCard.objects``: ``cup``, ``liquid``, ``receiver``, optional ``sph``), see
``PourGeometry``. The cup body's origin is the pivot, the inner top edge of its +x wall, so the tilt is a pure
rotation about +y (the opening turns towards the receiver at +x). The twin writes the exact pose and twist of the cup
each step (device kernel, no host sync); SolverDFSPH advances it as a kinematic body.

Recorded per world (probes.region_mass_kernel): liquid mass in the receiver (``m_poured``, the scale), in the cup
and in total. Scalars: onset angle (tilt when 0.5 % of the liquid has left the cup), poured mass, spill (neither in
the cup nor in the receiver). ``quasi_static_pour`` is the geometric model behind the synthetic rig data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.types import Condition, RigCard, RigTrial

from . import probes as pb
from .build import add_box_tank, add_fluid_points, lattice_points
from .config import BOUNDARY_SDF, particle_mass
from .twins import G, SPHTwin

POUR_DT = 0.02 / 64  # 0.3125 ms: CFL 0.2 for a 1.7 m/s stream at 2r = 4 mm
ONSET_FRACTION = 0.005  # of the initial liquid mass outside the cup


@dataclass(frozen=True)
class PourGeometry:
    """Cup, liquid and receiver of the pour rig [SI]. Cup inner box x in [-cup_x, 0], z in [-cup_h, 0] around the
    pivot; the receiver's inner box starts ``gap`` beyond the pivot and its rim sits ``drop`` below it."""

    cup_x: float = 0.06
    cup_y: float = 0.06
    cup_h: float = 0.08
    wall: float = 0.008
    pivot_z: float = 0.2
    liquid_mass: float = 0.2
    density: float = 1000.0
    viscosity: float = 1.0e-3
    gap: float = 0.03
    drop: float = 0.06
    recv_x: float = 0.12
    recv_y: float = 0.12
    recv_h: float = 0.08
    particle_radius: float = 2.0e-3

    @staticmethod
    def from_card(card: RigCard) -> PourGeometry:
        o = card.objects
        g = PourGeometry()
        keys = {"cup": ("cup_x", "cup_y", "cup_h", "wall", "pivot_z"), "liquid": ("liquid_mass", "density",
                "viscosity"), "receiver": ("gap", "drop", "recv_x", "recv_y", "recv_h"), "sph": ("particle_radius",)}
        vals = {k: float(o[obj][k]) for obj, ks in keys.items() for k in ks if k in o.get(obj, {})}
        return PourGeometry(**{**asdict(g), **vals})

    def to_objects(self) -> dict[str, dict[str, float]]:
        d = asdict(self)
        groups = {"cup": ("cup_x", "cup_y", "cup_h", "wall", "pivot_z"), "liquid": ("liquid_mass", "density",
                  "viscosity"), "receiver": ("gap", "drop", "recv_x", "recv_y", "recv_h"), "sph": ("particle_radius",)}
        return {name: {k: d[k] for k in ks} for name, ks in groups.items()}

    @property
    def receiver_lo(self) -> tuple[float, float, float]:
        return (self.gap, -0.5 * self.recv_y, self.pivot_z - self.drop - self.recv_h)

    @property
    def receiver_hi(self) -> tuple[float, float, float]:
        return (self.gap + self.recv_x, 0.5 * self.recv_y, self.pivot_z - self.drop)


def retained_volume(g: PourGeometry, theta: np.ndarray) -> np.ndarray:
    """Largest liquid volume the box cup holds at tilt ``theta`` [rad] (below the level plane through the pivot)."""
    th = np.asarray(theta, np.float64)
    tan = np.tan(np.clip(th, 1.0e-9, 0.5 * math.pi - 1.0e-9))
    full = g.cup_y * g.cup_x * (g.cup_h - 0.5 * g.cup_x * tan)
    wedge = g.cup_y * g.cup_h ** 2 / (2.0 * tan)
    v = np.where(g.cup_x * tan <= g.cup_h, full, wedge)
    return np.where(th >= 0.5 * math.pi, 0.0, np.where(th <= 0.0, g.cup_x * g.cup_y * g.cup_h, v))


def onset_angle_static(g: PourGeometry) -> float:
    """Tilt [rad] at which the resting surface reaches the pivot edge (quasi-static)."""
    fill = g.liquid_mass / (g.density * g.cup_x * g.cup_y)
    if fill >= g.cup_h:
        return 0.0
    tan = 2.0 * (g.cup_h - fill) / g.cup_x
    if g.cup_x * tan > g.cup_h:  # the level plane reaches the floor first: V = y h^2 / (2 tan)
        tan = g.cup_y * g.cup_h ** 2 / (2.0 * g.liquid_mass / g.density)
    return math.atan(tan)


def tilt(cond: Condition, t: np.ndarray) -> np.ndarray:
    """Tilt angle [rad] at times ``t``: constant rate up to theta_max, then held."""
    rate, top = math.radians(cond.params["tilt_rate_deg_s"]), math.radians(cond.params["theta_max_deg"])
    return np.minimum(rate * np.maximum(np.asarray(t, np.float64), 0.0), top)


def quasi_static_pour(g: PourGeometry, cond: Condition, t: np.ndarray) -> np.ndarray:
    """Scale reading [kg]: liquid above the retained volume, delayed by the free-fall time to the receiver floor."""
    delay = math.sqrt(2.0 * (g.drop + g.recv_h) / G)
    v0 = g.liquid_mass / g.density
    return g.density * np.maximum(v0 - retained_volume(g, tilt(cond, np.asarray(t) - delay)), 0.0)


@wp.kernel(enable_backward=False)
def drive_cup_kernel(cup: wp.array[wp.int32], pivot: wp.array[wp.vec3], rate: wp.array[wp.float32],
                     top: wp.array[wp.float32], t: float, body_q: wp.array[wp.transform],
                     body_qd: wp.array[wp.spatial_vector]):
    """Exact cup pose at time t and its twist (rotation about +y through the pivot = the body origin)."""
    w = wp.tid()
    b = cup[w]
    th = wp.min(rate[w] * t, top[w])
    omega = float(0.0)
    if rate[w] * t < top[w]:
        omega = rate[w]
    body_q[b] = wp.transform(pivot[w], wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), th))
    body_qd[b] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, omega, 0.0))


def add_cup_and_liquid(b: newton.ModelBuilder, cfg, g: PourGeometry, body: int, wall_density: float = 0.0) -> dict:
    """Cup walls on ``body`` (origin at the pivot, untilted at build time) and its liquid on the 2r lattice."""
    s = cfg.spacing
    nx, ny = max(1, round(g.cup_x / s)), max(1, round(g.cup_y / s))
    lx, ly = nx * s, ny * s
    add_box_tank(b, (-lx, -0.5 * ly, -g.cup_h), (0.0, 0.5 * ly, 0.0), wall=g.wall, body=body, density=wall_density)
    m_p = particle_mass(cfg, g.density)
    nz = max(1, round(g.liquid_mass / (m_p * nx * ny)))
    if nz * s > g.cup_h:
        raise ValueError(f"pour: {g.liquid_mass} kg needs {nz} layers > cup height {g.cup_h} m")
    pts = lattice_points((-lx, -0.5 * ly, g.pivot_z - g.cup_h), (nx, ny, nz), s)
    add_fluid_points(b, cfg, pts, g.density, g.viscosity)
    return {"cup": body, "cup_lo": (-lx, -0.5 * ly, -g.cup_h), "cup_hi": (0.0, 0.5 * ly, 0.5 * s),
            "mass": nx * ny * nz * m_p, "nx": nx, "ny": ny, "nz": nz}


TRAY_HEIGHT = 0.5  # [m] tray walls reach this far above the pivot


def _tray(g: PourGeometry, h: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Inner box of the tray that encloses a pour world: wide enough for every spill, walls TRAY_HEIGHT above the
    pivot, floor top one h below the receiver's bottom."""
    span = 2.0 * (g.gap + g.recv_x + g.cup_x + g.cup_h)
    cx = 0.5 * (g.gap + g.recv_x)
    floor_top = g.receiver_lo[2] - h
    return (cx - span, -span, floor_top), (cx + span, span, g.pivot_z + TRAY_HEIGHT)


def add_receiver_and_floor(b: newton.ModelBuilder, cfg, g: PourGeometry) -> dict[str, int]:
    """Static receiver (open box tank) standing in a closed tray (floor + four walls): spilled liquid stays in the
    scene, so a particle that leaves the tray went through a wall or was flung over it (LIQ-8)."""
    ids = add_box_tank(b, g.receiver_lo, g.receiver_hi, wall=cfg.h)
    lo, hi = _tray(g, cfg.h)
    tray = add_box_tank(b, lo, hi, wall=cfg.h)
    ids.update({f"tray_{k}": v for k, v in tray.items()})
    ids["ground"] = tray["floor"]
    return ids


def scene_bounds(g: PourGeometry, h: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Escape box of a pour world: the tray interior, h/2 beyond its walls and floor, 0.1 m above its rim.
    Leaving it means through the tray (bottom / side) or flung over the rim (top); the leak monitor classifies it."""
    lo, hi = _tray(g, h)
    m = 0.5 * h
    return (lo[0] - m, lo[1] - m, lo[2] - m), (hi[0] + m, hi[1] + m, hi[2] + 0.1)


def add_pour_world(b: newton.ModelBuilder, cfg, g: PourGeometry) -> dict:
    """Twin scene: kinematic cup (zero-density walls, origin at the pivot) with its liquid, receiver and floor."""
    cup = b.add_body(xform=wp.transform((0.0, 0.0, g.pivot_z), wp.quat_identity()), label="cup")
    info = add_cup_and_liquid(b, cfg, g, cup)
    add_receiver_and_floor(b, cfg, g)
    return info


class PourTwin(SPHTwin):
    """R7 SimTwin (rig suite): conditions from the dataset (``tilt_rate_deg_s``, ``theta_max_deg``)."""

    law_id = "R7.pour"
    units = {"onset_angle": "deg", "m_poured": "kg", "poured_mass": "kg", "spill": "kg"}

    def __init__(self, dt: float) -> None:
        super().__init__(dt)
        self.geometry = PourGeometry()
        self._info: list[dict] = []

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        """Geometry and liquid are measured rig quantities (card); nothing is fitted from the calib trials."""
        self.geometry = PourGeometry.from_card(card)
        self.monitors["geometry"] = asdict(self.geometry)

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"tilt{r:g}", {"tilt_rate_deg_s": float(r), "theta_max_deg": 120.0})
                     for r in (10, 20, 40))

    def build_world(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        params = {**cond.params, "r": self.geometry.particle_radius}
        super().build_world(b, Condition(cond.id, params), sample)

    def build(self, b: newton.ModelBuilder, cond: Condition, cfg) -> None:
        self._info.append(add_pour_world(b, cfg, self.geometry))

    def on_model(self, model: newton.Model) -> None:
        d = model.device
        g = self.geometry
        self._cup = wp.array([i["cup"] for i in self._info], dtype=wp.int32, device=d)
        self._pivot = wp.array([(0.0, 0.0, g.pivot_z)] * len(self._info), dtype=wp.vec3, device=d)
        self._rate = wp.array(np.radians(self.param("tilt_rate_deg_s")), dtype=wp.float32, device=d)
        self._top = wp.array(np.radians(self.param("theta_max_deg")), dtype=wp.float32, device=d)
        self._masses = wp.zeros((len(self._info), 3), dtype=wp.int64, device=d)
        self.monitors["liquid_mass_kg"] = [i["mass"] for i in self._info]

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        self._launch(drive_cup_kernel, len(self._info), [self._cup, self._pivot, self._rate, self._top, float(t),
                                                         state.body_q, state.body_qd])

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: Any) -> None:
        col = log_.column(step)
        if col < 0:
            return
        g, info = self.geometry, self._info[0]
        self._masses.zero_()
        self._launch(pb.region_mass_kernel, self.solver.layout.n,
                     [self.solver._particles(state), wp.vec3(*g.receiver_lo), wp.vec3(*g.receiver_hi), state.body_q,
                      self._cup, wp.vec3(*info["cup_lo"]), wp.vec3(*info["cup_hi"]), self._masses])
        W = len(self._info)
        for slot, name in enumerate(("m_poured", "m_cup", "m_total")):
            self._launch(pb.store_fixed_kernel, W, [self._masses, slot, pb.MASS_SCALE_HOST, col, log_.channel(name)])

    def scalars(self, log_: Any) -> dict[str, np.ndarray]:
        t = log_.times()
        poured, cup, total = (log_.series(n).astype(np.float64) for n in ("m_poured", "m_cup", "m_total"))
        m0 = total[:, 0]
        onset = np.full(len(self.worlds), np.nan)
        for w, cond in enumerate(self.worlds):
            out = np.nonzero(m0[w] - cup[w] >= ONSET_FRACTION * m0[w])[0]
            if len(out):
                onset[w] = math.degrees(float(tilt(cond, t[out[0]])))
        spill = m0 - cup[:, -1] - poured[:, -1]
        self.monitors.update(residual_in_cup_kg=cup[:, -1].tolist(), onset_static_deg=math.degrees(
            onset_angle_static(self.geometry)))
        return {"onset_angle": onset, "poured_mass": poured[:, -1], "spill": spill}


__all__ = ["POUR_DT", "PourGeometry", "PourTwin", "scene_bounds", "add_cup_and_liquid", "add_pour_world",
           "add_receiver_and_floor", "onset_angle_static", "quasi_static_pour",
           "retained_volume", "tilt"]
