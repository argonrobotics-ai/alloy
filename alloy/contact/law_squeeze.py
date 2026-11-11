# SPDX-License-Identifier: Apache-2.0
"""A8.squeeze_wedge / A8.squeeze_face: a light block squeezed by a fingertip against a sheet edge or face stays within
1 mm of the sheet (coordinator 2026-11-03, BUL-0014; scenes and reference by WP13).

Each world imports WP13's scene (``alloy.tasks.squeeze_ref``: library/robots/task_squeeze_{wedge,face}, the contact
parameters T12 gets on the route) and starts from its keyframe ``init``: T12's 12.7 g block_3 and a 0.3 kg fingertip
(T12's measured effective tip mass, WP13 a0eb9cf6; the scenes' source.json) on three world-axis slides,
position-servoed (kp 20000 N/m, kv 300 N s/m, 30 N cap per axis) along a straight target path. MuJoCo runs as Alloy
routes it (``BaseTwin.settings_for``: implicitfast for the actuated slides, the scene's MJCF options through Newton,
box-box on the primitive collider). The metric is the deepest block-sheet contact over the run and over every
world (MuJoCo Warp's own contacts, a device reduction each step); pass at <= 1 mm.

- wedge (case B, BUL-0014): the block flicked into the sheet's top edge with the tip following at 1.27 m/s. A single
  start is chaotic (MuJoCo C 6.9 mm at dt and 0.4 mm at dt/2; MuJoCo Warp 0.8 mm), so the law runs a seeded
  ensemble: block velocity x U(0.8, 1.2) per component, position +- 0.5 mm, tip speed x U(0.8, 1.2) and its
  direction tilted by up to 5 deg (``[internal note]`` uses the same perturbation).
- face (case A, control): the block flat against the sheet's side face, pushed face-on until the 30 N cap holds.

The wedge's detail also gives the depth a rigid tip would reach in the declared pair stiffness, delta_exp =
sqrt(2E / K_pair) with E = m_tip v^2 / 2 (the scene's tip mass, each world's drive speed), and measured / delta_exp
(coordinator ruling 2026-11-05 21:04Z, notes/wp0-a8-bound.md). K_pair is marked "default" while it is D22's (the
cards carry no ``contact_stiffness``). No verdict effect: 1 mm stays the bound.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import mujoco
from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench import default_route as DR
from alloy.bench import solvers as SV
from alloy.bench.analytic.common import BaseTwin
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.contact.light_body import K_PAIR
from alloy.tasks.squeeze_ref import DRIVES, scene_path

MAX_PEN_M = 1.0e-3
SPREAD = 0.2           # block velocity and tip speed scale U(1 - s, 1 + s)
POS_JITTER_M = 5.0e-4  # block start position +- per axis
TILT_DEG = 5.0         # tip drive direction tilt
SEED = 20261103
DT = 0.000625014305    # the scenes' timestep (T12's 20 ms period / 32)
CAPACITIES = {"nconmax": 48, "njmax": 400}  # per world, as the assembler would size these scenes (the bench route
                                            # has no capacities: MuJoCo Warp's 64-row default overflows)
_SRC = "coordinator 2026-11-03 (BUL-0014, WP13 scenes): squeeze penetration <= 1 mm on the default route"
K_PAIR_SOURCE = "default: D22's two 2e6 N/m materials in series (the cards carry no contact_stiffness)"


def _law(case: str, worlds: int, quick_worlds: int) -> LawSpec:
    drive = DRIVES[case]
    return LawSpec(
        id=f"A8.squeeze_{case}", suite="analytic",
        metrics={"peak_pen_m": Tolerance(abs=MAX_PEN_M, source=f"{_SRC}; deepest over the run and every world")},
        stat={"peak_pen_m": "mean"}, dt=DT, substeps=1, duration_s=drive.push_s + drive.hold_s,
        worlds_per_condition=1, solvers=("mujoco",), producer="WP3",
        quick={"worlds": quick_worlds, "duration_s": drive.push_s + drive.hold_s})


ENSEMBLE = {"wedge": (64, 16), "face": (4, 2)}  # (worlds, quick worlds)
LAW_WEDGE = _law("wedge", *ENSEMBLE["wedge"])
LAW_FACE = _law("face", *ENSEMBLE["face"])


def tilt(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """``v`` rotated by ``deg`` about the part of ``axis`` normal to it."""
    a = axis - axis @ v / (v @ v) * v
    a /= np.linalg.norm(a)
    t = math.radians(deg)
    return v * math.cos(t) + np.cross(a, v) * math.sin(t)


def perturbations(n: int, perturb: bool) -> list[dict[str, float]]:
    """Seeded per-world start perturbations (all zero when ``perturb`` is False)."""
    rng = np.random.default_rng(SEED)
    out = []
    for k in range(n):
        p = {"k": float(k)}
        if perturb and k > 0:  # world 0 is the recorded start
            p.update({f"vel{i}": float(rng.uniform(1 - SPREAD, 1 + SPREAD)) for i in range(6)})
            p.update({f"pos{i}": float(rng.uniform(-POS_JITTER_M, POS_JITTER_M)) for i in range(3)})
            p["speed"] = float(rng.uniform(1 - SPREAD, 1 + SPREAD))
            ax = rng.standard_normal(3)
            p.update({"ax0": float(ax[0]), "ax1": float(ax[1]), "ax2": float(ax[2]),
                      "tilt": float(rng.uniform(0.0, TILT_DEG))})
        out.append(p)
    return out


@wp.kernel
def _drive(coord: wp.array(dtype=wp.int32), drive: wp.array(dtype=wp.vec3), s: float,
           target_q: wp.array(dtype=float)):
    w = wp.tid()
    i = coord[w]
    v = drive[w] * s
    target_q[i] = v[0]
    target_q[i + 1] = v[1]
    target_q[i + 2] = v[2]


@wp.kernel
def _deepest(nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), dist: wp.array(dtype=float),
             worldid: wp.array(dtype=int), is_block: wp.array(dtype=wp.int32), sheet: int,
             step_min: wp.array(dtype=float)):
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    hit = (g[0] == sheet and is_block[g[1]] != 0) or (g[1] == sheet and is_block[g[0]] != 0)
    if hit:
        wp.atomic_min(step_min, worldid[i], dist[i])


@wp.kernel
def _record(step_min: wp.array(dtype=float), col: int, peak: wp.array(dtype=wp.float64),
            depth: wp.array2d(dtype=float)):
    w = wp.tid()
    d = wp.min(step_min[w], 0.0)
    peak[w] = wp.min(peak[w], wp.float64(d))
    if col >= 0:
        depth[w, col] = d


class SqueezeTwin(BaseTwin):
    """See module docstring; ``case`` picks the scene ("wedge" or "face")."""

    case = "wedge"
    law_id = LAW_WEDGE.id
    units: Mapping[str, str] = {"peak_pen_m": "m"}

    def __init__(self) -> None:
        super().__init__()
        self.solver: Any = None

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        n = ENSEMBLE[self.case][1 if quick else 0]
        return tuple(Condition(f"{self.case}_{int(p['k']):03d}", p)
                     for p in perturbations(n, perturb=self.case == "wedge"))

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        b.add_mjcf(str(scene_path(self.case)))

    def on_model(self, model: newton.Model) -> None:
        jstart = model.joint_q_start.numpy()
        if model.joint_count != 2 * len(self.worlds):
            raise ValueError(f"{self.law_id}: expected a free joint + a tip D6 per world, got {model.joint_count} joints")
        self._free = jstart[0:-1:2].astype(np.int32)
        self._tip = jstart[1:-1:2].astype(np.int32)
        self._tip_mass = model.body_mass.numpy()[model.joint_child.numpy()[1::2]].astype(float)
        self._start = [self._start_state(c.params) for c in self.worlds]
        dev = model.device
        self._tip_dev = wp.array(self._tip, dtype=wp.int32, device=dev)
        self._drive_dev = wp.array(np.array([s["drive"] for s in self._start], np.float32), dtype=wp.vec3, device=dev)
        self._step_min = wp.zeros(len(self.worlds), dtype=float, device=dev)

    def _start_state(self, p: Mapping[str, float]) -> dict[str, np.ndarray]:
        """The keyframe ``init`` with this world's perturbation, as MuJoCo C state (qpos, block world twist, tip)."""
        mjm = mujoco.MjModel.from_xml_path(str(scene_path(self.case)))
        d = mujoco.MjData(mjm)
        mujoco.mj_resetDataKeyframe(mjm, d, mjm.key("init").id)
        drive = np.asarray(DRIVES[self.case].velocity, np.float64)
        if "speed" in p:
            d.qvel[0:6] *= [p[f"vel{i}"] for i in range(6)]
            d.qpos[0:3] += [p[f"pos{i}"] for i in range(3)]
            drive = tilt(drive * p["speed"], np.array([p["ax0"], p["ax1"], p["ax2"]]), p["tilt"])
            d.qvel[6:9] = drive
        mujoco.mj_forward(mjm, d)
        twist = np.zeros(6)
        mujoco.mj_objectVelocity(mjm, d, mujoco.mjtObj.mjOBJ_BODY, mjm.body("block").id, twist, 0)
        return {"qpos": d.qpos.copy(), "ang": twist[:3], "lin": twist[3:], "tip_v": d.qvel[6:9].copy(),
                "drive": drive}

    def init_state(self, model: newton.Model, state: Any) -> None:
        q = state.joint_q.numpy()
        for w, s in enumerate(self._start):
            f, t, qp = self._free[w], self._tip[w], s["qpos"]
            q[f:f + 3], q[f + 3:f + 7], q[t:t + 3] = qp[0:3], np.r_[qp[4:7], qp[3]], qp[7:10]
        state.joint_q.assign(q)
        state.joint_qd.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        bqd = state.body_qd.numpy()
        first = self.ranges.start.numpy()
        for w, s in enumerate(self._start):  # block then tip in each world (scene order)
            bqd[first[w], :3], bqd[first[w], 3:] = s["lin"], s["ang"]
            bqd[first[w] + 1, :3], bqd[first[w] + 1, 3:] = s["tip_v"], 0.0
        state.body_qd.assign(bqd)
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)

    def settings_for(self, solver: str) -> SV.SolverSettings:
        """Alloy's route (builtin kwargs, routed integrator) plus per-world capacities and the box-box line contacts
        T12 runs with (``alloy.contact.mjw_boxbox_line``, BUL-0014)."""
        from alloy.contact.mjw_boxbox_line import LINE_KEY  # noqa: PLC0415

        routed = DR.routed_mujoco_settings(self.model)
        return SV.SolverSettings("mujoco", {**routed.kwargs, **CAPACITIES, LINE_KEY: True})

    def make_solver(self, model: newton.Model, solver: str) -> newton.solvers.SolverBase:
        """The routed solver, with the noslip the scene file asks for (SCR-SC6-3: as the route installs it for T12;
        Newton's MJCF importer drops it)."""
        from alloy.scene.noslip_route import install_file_noslip

        self.solver = super().make_solver(model, solver)
        install_file_noslip(self.solver, [str(scene_path(self.case))])
        self._bind_geoms(self.solver)
        return self.solver

    def _bind_geoms(self, solver: Any) -> None:
        """MuJoCo geom ids of the sheet and of the block's boxes (shared by every world of the Warp model)."""
        mjm = solver.mj_model
        names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(mjm.ngeom)]
        sheet = [g for g, n in enumerate(names) if n.rsplit("/", 1)[-1].startswith("sheet")]   # ".../sheet_5"
        block = np.array([int("/block/" in n) for n in names], np.int32)                    # ".../block/block_box0_0"
        if len(sheet) != 1 or block.sum() != 3:
            raise ValueError(f"{self.law_id}: sheet geoms {sheet}, block geoms {int(block.sum())} in {names}")
        self._sheet = int(sheet[0])
        self._is_block = wp.array(block, dtype=wp.int32, device=self.model.device)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        s = min((step + 1) * DT, DRIVES[self.case].push_s)
        wp.launch(_drive, dim=len(self.worlds), device=self.model.device,
                  inputs=[self._tip_dev, self._drive_dev, float(s)], outputs=[control.joint_target_q])

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        d = self.solver.mjw_data
        self._step_min.fill_(1.0e9)
        wp.launch(_deepest, dim=d.naconmax, device=self.model.device,
                  inputs=[d.nacon, d.contact.geom, d.contact.dist, d.contact.worldid, self._is_block, self._sheet],
                  outputs=[self._step_min])
        wp.launch(_record, dim=len(self.worlds), device=self.model.device,
                  inputs=[self._step_min, log.column(step)],
                  outputs=[log.accumulator("peak"), log.channel("depth")])

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        peak_mm = -log.series("acc:peak") * 1e3
        end_mm = -log.series("depth")[:, -1] * 1e3
        ids = [c.id for c in self.worlds]
        self.monitors.update({
            "peak_mm": dict(zip(ids, np.round(peak_mm, 4).tolist())), "end_mm": dict(zip(ids, np.round(end_mm, 4).tolist())),
            "over_1mm_fraction": float(np.mean(peak_mm > MAX_PEN_M * 1e3)), "p50_mm": float(np.median(peak_mm)),
            "p95_mm": float(np.percentile(peak_mm, 95)), "max_mm": float(peak_mm.max())})
        if self.case == "wedge" and hasattr(self, "_tip_mass"):
            self.monitors.update(expected_depth(peak_mm, self._tip_mass, [s["drive"] for s in self._start], ids))
        return {"peak_pen_m": np.array([float(peak_mm.max()) * 1e-3])}


def expected_depth(peak_mm: np.ndarray, tip_mass: np.ndarray, drive: list, ids: list[str],
                   k_pair: float = K_PAIR) -> dict:
    """delta_exp = sqrt(2E / K_pair) = sqrt(m v^2 / K_pair) per world, and measured / delta_exp (module docstring)."""
    v = np.linalg.norm(np.asarray(drive, float), axis=1)
    delta_mm = np.sqrt(np.asarray(tip_mass, float) * v**2 / k_pair) * 1e3
    ratio = np.asarray(peak_mm, float) / delta_mm
    worst = int(np.argmax(peak_mm))
    return {"k_pair_n_per_m": k_pair, "k_pair_source": K_PAIR_SOURCE,
            "tip_mass_kg": float(tip_mass[worst]), "delta_exp_mm": dict(zip(ids, np.round(delta_mm, 4).tolist())),
            "measured_over_delta_exp": dict(zip(ids, np.round(ratio, 3).tolist())),
            "delta_exp_worst_mm": float(delta_mm[worst]), "ratio_worst": float(ratio[worst]),
            "ratio_max": float(ratio.max())}


class SqueezeFaceTwin(SqueezeTwin):
    case = "face"
    law_id = LAW_FACE.id


__all__ = ["K_PAIR_SOURCE", "LAW_FACE", "LAW_WEDGE", "MAX_PEN_M", "SqueezeFaceTwin", "SqueezeTwin", "expected_depth",
           "perturbations", "tilt"]
