# SPDX-License-Identifier: Apache-2.0
"""R1 SimTwin: peg insertion with 0.2 mm diametral clearance (bench §4.4 R1; SPEC CT-3, SB-2), and its LawSpec.

Scene, one world per (condition, setting sample), every world in the same template order (the contact layer maps
template id ``t`` to ``shape_world_start[w] + t``): shape 0 the static meshed blind hole (chamfered or sharp per
condition), shape 1 the meshed Ø9.80 × 40 mm steel peg on a free body (body 0 of the world). The pair is an sdf-mode
precision pair ``PrecisionPairSpec(0, 1, mode="sdf")`` (voxel 50 µm, per-patch K = 2e6 N/m, speculative CCD) filtered
from native collision with flat solimp, and MuJoCo collides natively with the precision rows appended
(``AlloyContactLayer(native="mujoco")``, ``bind_mujoco(solver).install()``). sdf rather than hydroelastic: the R1
metrics are forces and depths, sdf mode is validated on the same peg-in-hole (CT-1, CT-8) and ~8x faster there
(38 s vs 317 s for 3,520 steps × 16 worlds), and Newton has no speculative hydroelastic contacts.

Holder and stage (``alloy.contact.rig_holder``): spring-damper between the stage target and the peg's compliance
centre with the card's k_x, k_theta, l_g, k_z; the stage moves at 1 mm/s (5 mm/s in quick mode, inside CT-3's
1-5 mm/s; the metrics are quasi-static) and latches on the force limit. dt = 0.3125 ms is the §3.5 L2 step of the
pair for the 23.5 g peg (``size_l2``: design force 50 N, contact speed <= 0.05 m/s). Insertion force and tip depth are
recorded on device and reduced with ``alloy.contact.rig_r1.insertion_metrics`` (the code that reduces the rig data).
Worlds per condition: sample 0 nominal, the rest jittered by the card's setting repeatability. mu is the card's
separately measured pair friction; nothing is fitted on calib (``bind_calib`` reads the card only).
Importing this module registers ``R1_LAW`` with ``R1Twin`` as the factory (texture SDFs need CUDA: the factory's
twin bakes on cuda:0, so on a CPU-only host the law ends as ``error`` with the reason, never as a pass).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import BaseTwin
from alloy.bench.twin import DeviceTwinLog, WorldRanges
from alloy.bench.types import Condition, LawSpec, RigCard, RigTrial, Tolerance
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.rig_holder import HolderDrive, HolderSetup
from alloy.contact.rig_r1 import (
    DEFAULT_MU, LAW_ID, METRICS, UNITS, R1Geometry, R1Holder, R1Protocol, chamfered_hole_tris, insertion_metrics,
    mean_profile, r1_conditions,
)
from alloy.contact.sdf_cache import build_feature_sdf
from alloy.contact.shadow import set_flat_solimp
from alloy.contact.types import PrecisionPairSpec
from alloy.core.defaults import GRAVITY
from alloy.core.errors import LawError
from alloy.core.log import get_logger
from alloy.testing.contact import cylinder_tris, to_newton_mesh

log = get_logger("contact.rig")

_SRC = "bench §4.4 R1"
PROTO = R1Protocol()
DT = 3.125e-4
PRECISION_ROWS_PER_WORLD = 192
SOLVER_KW: Mapping[str, Any] = {"use_mujoco_contacts": True, "use_speculative_contacts": False,
                                "integrator": "euler", "cone": "elliptic", "nconmax": 256, "njmax": 1024,
                                "iterations": 100, "ls_iterations": 50}

R1_LAW = LawSpec(
    id=LAW_ID, suite="rig",
    metrics={"sliding_force": Tolerance(abs=0.3, rel=0.10, source=f"{_SRC}: sliding force 10 % or 0.3 N "
                                                                   "(Tolerance sums abs + rel)"),
             "peak_force": Tolerance(rel=0.10, source=f"{_SRC}: two-point peak force 10 %"),
             "peak_depth": Tolerance(abs=1.0e-4, source=f"{_SRC}: two-point peak depth 0.1 mm"),
             "jam": Tolerance(abs=0.1, source=f"{_SRC}: jam rate inside the rig's Wilson interval +/- 0.1")},
    stat={"sliding_force": "mean", "peak_force": "mean", "peak_depth": "mean", "jam": "rate"},
    dt=DT, substeps=1, duration_s=PROTO.duration(PROTO.speed), worlds_per_condition=10, solvers=("mujoco",),
    producer="WP3",
    quick={"worlds": 8, "duration_s": PROTO.duration(PROTO.quick_speed), "stage_speed": PROTO.quick_speed},
)


def r1_pair(geometry: R1Geometry) -> PrecisionPairSpec:
    """The sdf precision pair (template shapes: hole 0, peg 1)."""
    return PrecisionPairSpec(0, 1, mode="sdf", voxel_size=5e-5, pair_stiffness=2e6, clearance=geometry.clearance,
                             ccd="speculative", gap=5e-4, narrow_band=(-5e-4, 5e-4))


class LayerPipeline:
    """The ``run_law`` contact source: it calls ``collide(state, contacts)`` without dt; the layer needs the step."""

    def __init__(self, layer: AlloyContactLayer, dt: float) -> None:
        self.layer, self.dt = layer, dt

    def contacts(self) -> newton.Contacts:
        return self.layer.contacts()

    def collide(self, state: newton.State, contacts: newton.Contacts) -> None:
        self.layer.collide(state, contacts, dt=self.dt)


class R1Twin(BaseTwin):
    """Peg-insertion twin; ``device`` must be CUDA (texture SDFs)."""

    law_id = LAW_ID
    units = UNITS

    def __init__(self, device: str = "cuda:0", seed: int = 0, geometry: R1Geometry | None = None) -> None:
        super().__init__()
        self.device, self.seed = device, seed
        self.geometry = geometry or R1Geometry()
        self.holder, self.mu, self.proto = R1Holder(), DEFAULT_MU, PROTO
        self.speed = self.proto.speed
        self.pair = r1_pair(self.geometry)
        self.layer: AlloyContactLayer | None = None
        self.drive_state: HolderDrive | None = None
        self.profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.world_values: dict[str, np.ndarray] = {}
        self.series: dict[str, np.ndarray] = {}           # host [W, T] channels of the last run (diagnostics)
        self._solver: Any = None
        self._meshes: dict[str, Any] = {}

    # ---------------------------------------------------------------------------------------- data, worlds ---
    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return r1_conditions(quick, self.geometry)

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        """Geometry, holder and mu from the card; the stage profile must be this protocol's."""
        want = {"stage_speed": self.proto.speed, "f_stop": self.proto.f_stop, "z_end": self.proto.z_end,
                "approach": self.proto.approach}
        for key, val in want.items():
            if key in card.holder and not math.isclose(float(card.holder[key]), val, rel_tol=1e-9):
                raise LawError(f"R1 card holder {key}={card.holder[key]} differs from protocol {val}")
        self.geometry = R1Geometry.from_objects(card.objects)
        self.holder = R1Holder.from_card(card.holder)
        self.mu = float(card.objects["pair"]["mu"])
        self.pair = r1_pair(self.geometry)
        self.monitors.update(card_mu=self.mu, calib_trials=len(calib), holder=dict(card.holder))

    def expand(self, cond: Condition, quick: bool, per: int) -> list[dict[str, float]]:
        """Sample 0 nominal, the rest jittered by the card's setting repeatability (deterministic per condition)."""
        self.speed = self.proto.quick_speed if quick else self.proto.speed
        rng = np.random.default_rng([self.seed, zlib.crc32(cond.id.encode())])
        tilt, off, half = cond.params["tilt"], cond.params["offset"], 0.5 * self.geometry.clearance
        out = []
        for k in range(per):
            dt_, do = (0.0, 0.0) if k == 0 else (rng.normal(0.0, self.holder.tilt_sigma),
                                                  rng.normal(0.0, self.holder.offset_sigma))
            out.append({"tilt": abs(tilt + dt_), "offset": float(np.clip(off + do, 0.0, half))})
        return out

    def _mesh(self, key: str, verts: np.ndarray, tris: np.ndarray, inertia: bool) -> Any:
        """One mesh object per shape kind, SDF baked once on the twin's CUDA device."""
        if key not in self._meshes:
            mesh = to_newton_mesh(verts, tris, compute_inertia=inertia)
            build_feature_sdf(mesh, self.pair, device=self.device)
            self._meshes[key] = mesh
        return self._meshes[key]

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        g, chamfer = self.geometry, float(cond.params["chamfer"])
        hole_mesh = self._mesh(f"hole{chamfer:.6g}", *chamfered_hole_tris(g.hole_spec(), chamfer), inertia=False)
        peg_mesh = self._mesh("peg", *cylinder_tris(0.5 * g.peg_d, 0.5 * g.peg_len, g.segments), inertia=True)
        cfg = newton.ModelBuilder.ShapeConfig(density=g.density, mu=self.mu, gap=self.pair.gap, margin=0.0)
        hole = b.add_shape_mesh(-1, mesh=hole_mesh, cfg=cfg, label="hole")
        tilt, off = float(sample["tilt"]), float(sample["offset"])
        axis = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
        centre = np.array([off, 0.0, g.height + self.proto.approach]) + 0.5 * g.peg_len * axis
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)      # lean +x
        body = b.add_body(xform=wp.transform(wp.vec3(*centre.tolist()), q), label="peg")
        peg = b.add_shape_mesh(body, mesh=peg_mesh, cfg=cfg, label="peg")
        b.add_shape_collision_filter_pair(hole, peg)
        for shape in (hole, peg):
            set_flat_solimp(b, shape, self.pair.mj_imp)

    # ------------------------------------------------------------------------------ solver, contacts, drive ---
    def settings_record(self, solver: str) -> dict[str, Any]:
        return {**SOLVER_KW, "impratio": self.pair.mj_impratio, "pair_mode": self.pair.mode,
                "pair_stiffness": self.pair.pair_stiffness, "voxel": self.pair.voxel_size, "stage_speed": self.speed}

    def make_solver(self, model: newton.Model, solver: str) -> newton.solvers.SolverBase:
        if solver != "mujoco":
            raise LawError(f"R1 twin runs on mujoco only (native contacts + precision append), got {solver!r}")
        self.model, self.solver_name = model, solver
        self.ranges = WorldRanges.bodies(model)
        self._first_body = self.ranges.start
        self.on_model(model)
        self._solver = newton.solvers.SolverMuJoCo(model, impratio=self.pair.mj_impratio, **SOLVER_KW)
        return self._solver

    def make_pipeline(self, model: newton.Model, solver: str) -> LayerPipeline:
        if self._solver is None:
            raise LawError("R1 twin: make_solver must run before make_pipeline")
        self.layer = AlloyContactLayer(model, [self.pair], native="mujoco",
                                       precision_contacts_per_world=PRECISION_ROWS_PER_WORLD, deterministic=True)
        self.layer.bind_mujoco(self._solver).install()
        return LayerPipeline(self.layer, R1_LAW.dt / R1_LAW.substeps)

    def on_model(self, model: newton.Model) -> None:
        g = self.geometry
        setup = HolderSetup(self.holder, self.proto, self.speed, R1_LAW.dt / R1_LAW.substeps, g.peg_len, g.height,
                            GRAVITY)
        self.drive_state = HolderDrive(model, self.ranges.start.numpy(), setup)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        self.drive_state.apply(state, t)

    def record_stride(self, n_steps: int) -> int:
        """One column per 10 µm of stage travel (box-averaged force inside it)."""
        return max(1, int(round(1.0e-5 / (self.speed * R1_LAW.dt))))

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: DeviceTwinLog) -> None:
        self.drive_state.record(state, step * R1_LAW.dt, log_.column(step), log_)

    # ----------------------------------------------------------------------------------------------- reduce ---
    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        depth, force = log_.series("depth"), log_.series("force")
        per = [insertion_metrics(depth[w], force[w], self.proto) for w in range(len(self.worlds))]
        values = {m: np.array([getattr(x, m) for x in per], np.float64) for m in METRICS}
        self.world_values = {m: v.copy() for m, v in values.items()}
        self.series = {k: log_.series(k).copy() for k in ("depth", "force", "tilt", "fx", "my")}
        self._profiles(depth, force)
        self._summarise(values, log_)
        return values

    def _profiles(self, depth: np.ndarray, force: np.ndarray) -> None:
        """Mean force profile per condition (for CT-3's curve error against the rig's valid trials)."""
        ids = np.array([c.id for c in self.worlds])
        for cid in dict.fromkeys(ids):
            self.profiles[cid] = mean_profile([(depth[w], force[w]) for w in np.flatnonzero(ids == cid)], self.proto)

    def _summarise(self, values: Mapping[str, np.ndarray], log_: DeviceTwinLog) -> None:
        ids = np.array([c.id for c in self.worlds])
        per_cond = {}
        for cid in dict.fromkeys(ids):
            m = ids == cid
            per_cond[cid] = {k: float(np.nanmean(v[m])) if np.isfinite(v[m]).any() else None
                             for k, v in values.items()}
        tilt = log_.series("tilt")
        self.monitors.update(per_condition=per_cond, stage_speed=self.speed,
                             stage_stopped=int(self.drive_state.stopped().sum()),
                             max_tilt_deg=float(np.degrees(np.nanmax(tilt))) if np.isfinite(tilt).any() else None,
                             overflow=self.overflow())

    def overflow(self) -> dict[str, int]:
        """Dropped contact rows per cause (host readback after the run; CT-12)."""
        rep = self.layer.overflow_report() if self.layer is not None else {}
        if self._solver is not None:
            rep["mjwarp_overflow"] = int(np.asarray(self._solver.mjw_data.overflow.numpy()).max())
        binding = getattr(self.layer, "_binding", None)
        rep["append"] = 0 if binding is None else int(binding.overflow.numpy()[0])
        return rep


__all__ = ["R1_LAW", "R1Twin", "LayerPipeline", "r1_pair", "PROTO", "DT"]
