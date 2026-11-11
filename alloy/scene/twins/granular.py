# SPDX-License-Identifier: Apache-2.0
"""R8 granular twin (bench §4.4 R8), fine media on implicit MPM: repose angle (± 2°) and column-collapse runout (10 %).

Covered: sand (fine medium, MPM route) for the lifting-cylinder repose angle and the aspect-2 column collapse.
Not covered, and why:
- blade force at depth {10, 20, 40} mm and the exponent n in F ∝ dⁿ: resolving a 10 mm cut needs voxels <= 2.5 mm;
  a 250 × 60 mm plane-strain bed is then ~38k MPM particles, and this VM measured ~60-70 µs per particle-step
  (1580 particles: 97-120 ms/step), i.e. ~2.5 s/step × 400+ steps per depth: not a CPU quick run;
- rice and lentils (coarse media): their route is rigid spheres with a fitted rolling lever (CON-75), not MPM; that
  twin is not built here.

Twin (one condition = one medium; `expand` makes one repose world and one collapse world per condition):
- both scenes are quarter-symmetric: the domain is the x, y >= 0 quadrant behind two frictionless walls (mu = 0
  colliders act as symmetry planes), on a floor with the medium's friction;
- repose: a Ø50 × 100 mm column inside a kinematic tube (8 box staves) lifted at SPEEDUP × the rig's 5 mm/s, then
  held clear for SETTLE_S; the angle is the slope of the deposit's radial height profile between 20 % and 80 % of
  its peak height;
- collapse: a free-standing column of radius R0 and height 2·R0 released at t = 0; runout = (r_99.5 − R0)/R0, r the
  radial particle distance after settling;
- material: rigid-plastic Drucker-Prager (mpm:young_modulus 1e15 as Newton's granular example), friction = tan φ with
  φ = mean calib repose angle (the `emit_mpm` mapping, alloy/scene/emitters/deformable.py `mpm_values`), 2 particles
  per voxel per axis, mass ρ_bulk·spacing³ (as `emit_mpm`).
Quick runs lift 20× faster than the rig (0.1 m/s) at 5 mm voxels; full runs lift 2× (10 mm/s) at 2.5 mm voxels.
The 20× lift is not quasi-static: measured on this VM, a 5 mm-voxel tube lift deposits a 17.3° heap at 20× and
24.7° at 5× (atan μ = 33.0°), and PIC transfer instead of APIC gives 27.5° at 20×. Runout at a = 2: 1.88 (5 mm
voxels, μ 0.6), 2.04 (3.5 mm voxels), 2.81 (μ 0.45) against Lube's 2.26.

Synthetic truth (`dataset`): repose = the medium's angle of repose; runout = Lube et al. (JFM 508, 2004)
axisymmetric law (R∞ − R0)/R0 = 1.24·a for a < 1.7, 1.6·a^½ above (material-independent).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp
from alloy._vendor.newton.solvers import SolverImplicitMPM

from alloy.bench.analytic.common import BaseTwin
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, RigCard, RigDataset, RigTrial, Tolerance
from alloy.core.errors import LawError
from alloy.integration.mpm_solve import NEWTON_TOLERANCE, solve_kwargs
from alloy.testing.fakes import synthetic_rig

log = logging.getLogger("alloy.scene")

ROLE_REPOSE, ROLE_COLLAPSE = 0, 1
RIG_LIFT_SPEED = 5.0e-3          # [m/s] bench §4.4 R8
SPEEDUP = {True: 20.0, False: 2.0}       # quick -> 20x, full -> 2x the rig lift speed
VOXEL = {True: 5.0e-3, False: 2.5e-3}    # [m]
DT = 2.5e-3                      # [s]
SETTLE_S = 0.3                   # [s] hold after the tube clears the deposit
LIFT_CLEAR = 0.12                # [m] tube travel that clears a 100 mm column
TUBE_STAVES, TUBE_WALL = 8, 3.0e-3
DOMAIN = 0.2                     # [m] quarter-domain side
PITCH = 0.5                      # [m] world spacing along x on the shared MPM grid
_SRC = "bench §4.4 R8"


def lube_runout(aspect: float) -> float:
    """Lube et al. 2004 axisymmetric collapse: (R∞ − R0)/R0 = 1.24 a (a < 1.7), 1.6 a^½ (a > 1.7)."""
    return 1.24 * aspect if aspect < 1.7 else 1.6 * math.sqrt(aspect)


def _duration(quick: bool) -> float:
    return LIFT_CLEAR / (SPEEDUP[quick] * RIG_LIFT_SPEED) + SETTLE_S


LAW = LawSpec(
    id="R8.granular_repose", suite="rig",
    metrics={"repose_angle": Tolerance(abs=math.radians(2.0), source=f"{_SRC}: repose angle +/- 2 deg"),
             "runout": Tolerance(rel=0.10, source=f"{_SRC}: column-collapse runout within 10 %")},
    stat={"repose_angle": "mean", "runout": "mean"}, dt=DT, substeps=1, duration_s=_duration(False),
    worlds_per_condition=2, solvers=("implicit_mpm",), producer="WP2",
    quick={"worlds": 2, "duration_s": _duration(True)},
)
CONDITIONS = (Condition("sand", {"density": 1600.0, "tube_radius": 0.025, "tube_height": 0.10,
                                 "column_radius": 0.025, "aspect": 2.0}),)


# --- geometry -----------------------------------------------------------------------------------------------------

def quarter_column(radius: float, height: float, spacing: float) -> np.ndarray:
    """Particle centres of a quarter column (x, y >= 0) on a regular grid of `spacing`."""
    n, nz = int(round(radius / spacing)), int(round(height / spacing))
    c = (np.arange(n) + 0.5) * spacing
    X, Y = np.meshgrid(c, c, indexing="ij")
    keep = X**2 + Y**2 <= (radius - 0.25 * spacing) ** 2
    xy = np.stack([X[keep], Y[keep]], axis=1)
    z = (np.arange(nz) + 0.5) * spacing
    return np.concatenate([np.column_stack([xy, np.full(len(xy), zk)]) for zk in z])


def _add_quarter_box(b: newton.ModelBuilder, floor_mu: float, x0: float) -> None:
    """Two frictionless symmetry walls (x = x0, y = 0 planes, as thick boxes) and a floor with the medium's mu."""
    wall = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0)
    half, t = 0.5 * DOMAIN, 0.02
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x0 - t, half, half), wp.quat_identity()), hx=t,
                    hy=half + t, hz=half, cfg=wall)
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x0 + half, -t, half), wp.quat_identity()), hx=half + t,
                    hy=t, hz=half, cfg=wall)
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x0 + half, half, -t), wp.quat_identity()), hx=half + t,
                    hy=half + t, hz=t, cfg=newton.ModelBuilder.ShapeConfig(mu=floor_mu, density=0.0))


def _add_tube(b: newton.ModelBuilder, radius: float, height: float, x0: float) -> int:
    """Kinematic quarter tube of TUBE_STAVES box staves just outside `radius`; returns its body index."""
    body = b.add_body(xform=wp.transform(wp.vec3(x0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True,
                      label="lift_tube")
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, density=0.0)
    r = radius + TUBE_WALL
    half_w = r * math.tan(0.25 * math.pi / TUBE_STAVES) + 5e-4
    for k in range(TUBE_STAVES):
        a = (k + 0.5) * 0.5 * math.pi / TUBE_STAVES
        xf = wp.transform(wp.vec3(r * math.cos(a), r * math.sin(a), 0.5 * height + 0.005),
                          wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), a))
        b.add_shape_box(body, xform=xf, hx=TUBE_WALL, hy=half_w, hz=0.5 * height + 0.01, cfg=cfg)
    return body


# --- measurements -------------------------------------------------------------------------------------------------

def repose_from_deposit(q: np.ndarray, bin_w: float) -> float:
    """Slope angle [rad] of the radial max-height profile between 20 % and 80 % of its peak (quarter heap at 0)."""
    r, z = np.hypot(q[:, 0], q[:, 1]), q[:, 2]
    idx = np.floor(r / bin_w).astype(int)
    h = np.full(idx.max() + 1, -np.inf)
    np.maximum.at(h, idx, z)
    rc = (np.arange(h.size) + 0.5) * bin_w
    ok = np.isfinite(h)
    rc, h = rc[ok], h[ok]
    sel = (h > 0.2 * h.max()) & (h < 0.8 * h.max())
    if sel.sum() < 3:
        return math.nan
    return math.atan(-float(np.polyfit(rc[sel], h[sel], 1)[0]))


def runout_from_deposit(q: np.ndarray, r0: float) -> float:
    """(r_99.5 − R0)/R0 of the settled particles' radial distance."""
    return (float(np.percentile(np.hypot(q[:, 0], q[:, 1]), 99.5)) - r0) / r0


@wp.kernel
def _drive_tubes(bodies: wp.array(dtype=wp.int32), speed: wp.array(dtype=float), x0: wp.array(dtype=float),
                 t_next: float, clear: float, body_q: wp.array(dtype=wp.transform),
                 body_qd: wp.array(dtype=wp.spatial_vector)):
    k = wp.tid()
    v = speed[k]
    z = wp.min(v * t_next, clear)
    if v * t_next >= clear:
        v = 0.0
    body_q[bodies[k]] = wp.transform(wp.vec3(x0[k], 0.0, z), wp.quat_identity())
    body_qd[bodies[k]] = wp.spatial_vector(wp.vec3(0.0, 0.0, v), wp.vec3(0.0))


@wp.kernel
def _record_vmax(qd: wp.array(dtype=wp.vec3), start: wp.array(dtype=wp.int32), end: wp.array(dtype=wp.int32),
                 col: int, out: wp.array2d(dtype=float)):
    w = wp.tid()
    m = float(0.0)
    for i in range(start[w], end[w]):
        m = wp.max(m, wp.length(qd[i]))
    out[w, col] = m


# --- twin --------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _World:
    cond: str
    role: int
    start: int
    end: int
    tube: int          # kinematic tube body, -1 for the collapse world
    x0: float          # world origin along x on the shared grid


class GranularTwin(BaseTwin):
    """R8 twin on SolverImplicitMPM: friction = tan(mean calib repose angle) per medium."""

    law_id = LAW.id
    units = {"repose_angle": "rad", "runout": "1"}
    needs_contacts = False

    def __init__(self, max_iterations: int = 250, transfer_scheme: str = "apic") -> None:
        super().__init__()
        self.max_iterations, self.transfer_scheme = int(max_iterations), transfer_scheme
        self.quick = True
        self.phi: dict[str, float] = {}
        self._w: list[_World] = []
        self._final_q: np.ndarray | None = None

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        for cid in dict.fromkeys(t.condition_id for t in calib):
            self.phi[cid] = float(np.mean([t.scalars["repose_angle"] for t in calib if t.condition_id == cid]))
        self.monitors["friction_angle_calib_deg"] = {c: math.degrees(p) for c, p in self.phi.items()}

    def expand(self, cond: Condition, quick: bool, per: int) -> list[dict[str, float]]:
        self.quick = quick
        return [{"role": ROLE_REPOSE}, {"role": ROLE_COLLAPSE}]

    def prepare_builder(self, b: newton.ModelBuilder, solver: str) -> None:
        SolverImplicitMPM.register_custom_attributes(b)

    def settings_record(self, solver: str) -> dict[str, Any]:
        return {"solver": "implicit_mpm", "voxel_size": VOXEL[self.quick], "max_iterations": self.max_iterations,
                "tolerance": solve_kwargs().get("tolerance", NEWTON_TOLERANCE), "transfer_scheme": self.transfer_scheme,
                "lift_speedup": SPEEDUP[self.quick]}

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        if cond.id not in self.phi:
            raise LawError(f"R8: no calib friction angle for condition {cond.id!r}")
        p, spacing, x0 = cond.params, 0.5 * VOXEL[self.quick], PITCH * len(self._w)
        _add_quarter_box(b, math.tan(self.phi[cond.id]), x0)
        role, tube = int(sample["role"]), -1
        if role == ROLE_REPOSE:
            pts = quarter_column(p["tube_radius"], p["tube_height"], spacing)
            tube = _add_tube(b, p["tube_radius"], p["tube_height"], x0)
        else:
            pts = quarter_column(p["column_radius"], p["aspect"] * p["column_radius"], spacing)
        start, mass = b.particle_count, p["density"] * spacing**3
        for x in pts + np.array([x0, 0.0, 0.0]):
            b.add_particle(wp.vec3(*x), wp.vec3(0.0), mass=mass, radius=0.5 * spacing)
        self._w.append(_World(cond.id, role, start, b.particle_count, tube, x0))

    def make_solver(self, model: newton.Model, solver: str) -> SolverImplicitMPM:
        self.model, self.solver_name = model, solver
        mu = np.zeros(model.particle_count, np.float32)
        for w in self._w:
            mu[w.start:w.end] = math.tan(self.phi[w.cond])
        model.mpm.friction.assign(mu)
        model.mpm.young_modulus.fill_(1.0e15)  # rigid-plastic limit (Newton's granular example default)
        cfg = SolverImplicitMPM.Config(**solve_kwargs())  # the production rheology tolerance (WP4's review)
        # shared grid, worlds PITCH apart: isolated per-world grids are CUDA-only (warp@1.18:warp/_src/fem/geometry/
        # nanogrid.py:1216 raises for NanoVDB environment volumes on CPU)
        cfg.voxel_size, cfg.separate_worlds = VOXEL[self.quick], False
        cfg.max_iterations, cfg.transfer_scheme = self.max_iterations, self.transfer_scheme
        self.solver = SolverImplicitMPM(model, config=cfg)
        dev = model.device
        tubes = [w for w in self._w if w.tube >= 0]
        self._tubes = wp.array(np.asarray([w.tube for w in tubes], np.int32), dtype=wp.int32, device=dev)
        self._tube_x0 = wp.array(np.asarray([w.x0 for w in tubes], np.float32), dtype=float, device=dev)
        speed = SPEEDUP[self.quick] * RIG_LIFT_SPEED
        self._speed = wp.array(np.full(len(tubes), speed, np.float32), dtype=float, device=dev)
        self._start = wp.array(np.array([w.start for w in self._w], np.int32), dtype=wp.int32, device=dev)
        self._end = wp.array(np.array([w.end for w in self._w], np.int32), dtype=wp.int32, device=dev)
        return self.solver

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        if self._tubes.shape[0]:
            wp.launch(_drive_tubes, dim=self._tubes.shape[0], inputs=[self._tubes, self._speed, self._tube_x0, t + DT, LIFT_CLEAR],
                      outputs=[state.body_q, state.body_qd], device=self.model.device)
        if step > 0:
            self.solver.project_outside(state, state, DT)  # as Newton's MPM examples do after every step

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: DeviceTwinLog) -> None:
        col = log_.column(step)
        if col >= 0:
            wp.launch(_record_vmax, dim=len(self._w), inputs=[state.particle_qd, self._start, self._end, col],
                      outputs=[log_.channel("vmax")], device=self.model.device)
        if step == log_.n_steps:
            self._final_q = state.particle_q.numpy()  # one host copy after the run

    def _per_condition(self, cid: str) -> dict[str, float]:
        rep = next(w for w in self._w if w.cond == cid and w.role == ROLE_REPOSE)
        col = next(w for w in self._w if w.cond == cid and w.role == ROLE_COLLAPSE)
        r0 = next(c for c in self.worlds if c.id == cid).params["column_radius"]
        return {"repose_angle": repose_from_deposit(self._local(rep), VOXEL[self.quick]),
                "runout": runout_from_deposit(self._local(col), r0)}

    def _local(self, w: _World) -> np.ndarray:
        return self._final_q[w.start:w.end] - np.array([w.x0, 0.0, 0.0], np.float32)

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        if self._final_q is None:
            raise LawError("R8: the run ended without a final state")
        per = {cid: self._per_condition(cid) for cid in dict.fromkeys(w.cond for w in self._w)}
        vmax = log_.series("vmax")
        self.monitors.update({"per_condition": per, "vmax_end": vmax[:, -1].tolist(),
                              "repose_deg": {c: math.degrees(v["repose_angle"]) for c, v in per.items()}})
        log.info("R8 twin: %s", per)
        return {m: np.array([per[w.cond][m] for w in self._w]) for m in LAW.metrics}


# --- synthetic dataset ---------------------------------------------------------------------------------------------

TRUTH = {"sand": {"repose_deg": 33.0}}
NOISE = {"repose_angle": math.radians(0.5), "runout": 0.07}
INSTRUMENTS = {"repose_angle": {"u95": math.radians(0.3), "what": "4 photos, edge fit"},
               "runout": {"u95": 0.04, "what": "ruler 1 mm on R0 = 25 mm"}}
TRIALS = 20  # per medium: the rig protocol has n = 5; 10 per split keeps TUR >= 3 at this scatter


def _truth(params: Mapping[str, float], cond: Condition) -> dict[str, float]:
    return {"repose_angle": math.radians(params[f"{cond.id}.repose_deg"]),
            "runout": lube_runout(cond.params["aspect"])}


def dataset(seed: int = 0) -> RigDataset:
    """Synthetic sand trials: repose angle of the medium and Lube's runout, plus Gaussian scatter."""
    params = {f"{c}.{k}": v for c, m in TRUTH.items() for k, v in m.items()}
    data = synthetic_rig(LAW.id, params, NOISE, conditions=CONDITIONS, trials_per_condition=TRIALS,
                         calib_per_condition=TRIALS // 2, seed=seed, model=_truth)
    card = dataclasses.replace(data.card, dataset_id="synthetic-v1", protocol_version="synthetic/1",
                               instruments=dict(INSTRUMENTS), objects={c.id: dict(c.params) for c in CONDITIONS})
    return dataclasses.replace(data, card=card)


__all__ = ["LAW", "GranularTwin", "dataset", "lube_runout", "repose_from_deposit", "runout_from_deposit",
           "quarter_column"]
