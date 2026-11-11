# SPDX-License-Identifier: Apache-2.0
"""Ladder L3 (parallel §4.9, SPEC §3.9): WP13 task scenes through Alloy's own assembly and entry stepper.

- ``t12`` (``task_t12``, converted tier): bimanual block stacking, replayed from its 20 s recording (bench/traces/t12):
  ``mujoco.ctrl``, the arm ``joint_f`` columns and the gripper ``jnt_actfrcrange`` row by row, all worlds alike.
- ``t09`` (``task_t09``): USB-A plug and 1 m cable. WP13 has no recording for it, so it runs at its card controls.
- ``emb_<key>`` (P3-2b embodiments, WP13's ``task_emb_<key>`` cards): stepped as the SB-7 bully target ``emb_<key>``
  steps (alloy/tasks/emb_targets.py): its 20 ms action period with the stepper's clock for it (``choose_clock``,
  0.625 ms x 32), its MuJoCo Warp capacities (nconmax 256, njmax 1,024 per world), WP5's ordered route, its initial
  state and its B0 random-legal ctrl traces (the SB-7 / rival workload, P3-2); floor gate below.

Per pack: ``SceneAssembler(lib).assemble(card, device, world_count=N)`` -> ``Simulation(built)`` for the entry stepper
(SolverMuJoCo with the router's options and capacities) and its state/control, then a WP8 period loop of
``stepper.step`` + swap. T12 and T09 run native "mujoco" with no contact layer, topology, health or gap monitor, so
that loop is what ``Simulation._exchange`` does for them minus the host clock. ``Simulation``'s own graph period is
not used: on main it builds ``PeriodGraph(body)`` without the device argument and replays an odd number of state
swaps when a period is one step (reported to WP2 in status/WP8.md). A period that is one step is captured two at a
time; trace rows are written on the device (``TraceReplayer``), so the captured call is graph-safe.

Gate (t12): from a reset after the timed run, every world reproduces the recorded object poses within 1 mm (WP13's
point bound ``|Δp| + r·θ``) over the first 0.2 s, plus finite state and no MJW overflow during the timed run. t09:
finite, no overflow, nothing below the bench (z > -0.01 m, WP13's 20-period check). emb: finite, no overflow, no
body 1 cm below the lower of its start height and the scene's ground plane (falls and reaches off a table edge are
legal B0 outcomes; sinking through the floor is not).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.cards.library import CardLibrary
from alloy.parallel.counters import ProbeSample, ReplayLayout, TraceReplayer, mujoco_counters
from alloy.parallel.probe import StepTarget
from alloy.parallel.types import PackSpec
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.tasks.attribution import pose_errors
from alloy.tasks.newton_map import body_index, initial_to_newton, joint_index, lookup, runs
from alloy.tasks.trace import ACTFRC, CTRL, JOINT_F, QPOS, QVEL, WARM, TaskTrace, load_trace

log = logging.getLogger("alloy.parallel")
TRACES = Path(__file__).resolve().parents[2] / "bench" / "traces"  # = alloy.tasks.cli.TRACES
GATE_M = 1e-3  # WP13 replay gate (SPEC §7 WP13)
GATE_S = 0.2  # replay window of the gate (WP13's assembled-model acceptance window)
FLOOR_M = -0.01


@dataclass(frozen=True)
class TaskSpec:
    scene: str  # CardLibrary scene name
    trace: str | None  # bench/traces/<name>, None = no recording
    target: str | None = None  # bully target: its initial state and B0 random-legal actions (P3-2b embodiments)


B0_SEED = 100_000  # P3-2 SB-7 campaign's first seed (bench/xbench/ours/b0.py FIRST_SEED)
B0_ROWS = 2048     # traces per seed; world w plays row w % 2048


def _embodiment_tasks() -> dict[str, TaskSpec]:
    """``emb_<key>`` -> scene card + bully target of each P3-2b embodiment (WP13), driven by B0 actions."""
    try:
        from alloy.tasks.emb_specs import EMBODIMENTS
        from alloy.tasks.emb_targets import card_name
    except ImportError as exc:  # WP13's package absent: the ladder keeps its task scenes
        log.warning("no embodiment rungs (%s)", exc)
        return {}
    return {e.name: TaskSpec(card_name(e), None, e.name) for e in EMBODIMENTS.values()}


TASKS: dict[str, TaskSpec] = {"t12": TaskSpec("task_t12", "t12"), "t09": TaskSpec("task_t09", None),
                              **_embodiment_tasks()}


class TaskRung:
    """One pack of a task scene: assembled model, Alloy entry stepper, device trace replay, counters and gate."""

    def __init__(self, task: str, n: int, device: str, lib: CardLibrary | None = None,
                 capacities: PackSpec | None = None) -> None:
        if task not in TASKS:
            raise ValueError(f"no task rung {task!r} (tasks: {sorted(TASKS)})")
        spec = TASKS[task]
        self.task, self.scene, self.param, self.n, self.device = task, task, 0, n, device
        lib = lib or CardLibrary()
        if capacities is None and spec.target:   # the SB-7 target's MuJoCo Warp capacities
            capacities = _target_capacities(spec.target, n, device)
        self.capacities = capacities  # None = the router's defaults; a PackSpec = WP8-sized (njmax, nconmax)
        self.built = SceneAssembler(lib).assemble(lib.scene(spec.scene), device, world_count=n, capacities=capacities)
        self.model = self.built.model
        _check_layout(self.model, n)
        with wp.ScopedDevice(device):
            self.sim = Simulation(self.built)
        self.stepper, self.contacts = self.sim.stepper, self.sim.contacts
        self.solver = self.stepper.solvers()[0]
        self.target_name = spec.target
        self.control_dt, self.k, self.dt = _clock(self)
        self.steps_per_period = self.k
        self.route = _route(self) if spec.target else None
        self.periods_per_call = 1 if self.k % 2 == 0 else 2  # even swaps per captured call (SCR-WP8-3)
        self.sim_per_call = self.control_dt * self.periods_per_call
        self.s0, self.s1, self.control = self.sim.state, self.model.state(), self.sim.control
        self.tt: TaskTrace | None = load_trace(TRACES / spec.trace) if spec.trace else None
        self.replayer = _bind_trace(self) if self.tt is not None else _bind_b0(self) if spec.target else None
        self._q0, self._qd0 = _initial_joint_state(self)
        self._read = mujoco_counters(self.solver, self.model)
        self.overflow = 0
        self.iter_limit = 0  # worlds at the solver's iteration cap (information, not in the gate)
        self._ground = _ground_z(self.model)
        self.reset()
        log.info("task rung %s: %d worlds, dt %.4g s, %d steps per period, %d periods per call, route %s", task, n,
                 self.dt, self.k, self.periods_per_call, self.route)

    def reset(self) -> None:
        """Initial joint state (the recording's for t12) in every world, warm start seeded, trace row 0."""
        self.s0.joint_q.assign(self._q0.astype(np.float32))
        self.s0.joint_qd.assign(self._qd0.astype(np.float32))
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        self._z0 = self.s0.body_q.numpy()[:, 2].astype(np.float64).copy()
        if self.tt is not None:
            _seed_warmstart(self)
        if self.replayer is not None:
            self.replayer.reset()

    def step(self) -> None:
        """``periods_per_call`` control periods: trace row, then ``k`` stepper steps with swaps."""
        for _ in range(self.periods_per_call):
            if self.replayer is not None:
                self.replayer.apply()
            for _ in range(self.k):
                self.stepper.step(self.s0, self.s1, self.control, self.contacts, self.dt)
                self.s0, self.s1 = self.s1, self.s0

    def counters(self) -> ProbeSample:
        sample = self._read()
        self.overflow = max(self.overflow, sample.overflow)
        self.iter_limit = max(self.iter_limit, sample.iter_limit)
        return sample

    def target(self) -> StepTarget:
        return StepTarget(step=self.step, nworld=self.n, sim_dt=self.sim_per_call, device=self.device,
                          counters=self.counters, buffers=lambda: (self.s0.body_q, self.s1.body_q), model=self.model)

    def gate(self) -> dict[str, Any]:
        """Quality gate: finite + no overflow over the timed run, then the replay check (t12) or the floor (t09)."""
        finite = all(bool(np.all(np.isfinite(a.numpy()))) for a in (self.s0.body_q, self.s0.body_qd))
        out: dict[str, Any] = {"finite": finite, "overflow_worlds": self.overflow, "iter_limit_worlds": self.iter_limit}
        if self.tt is None and self.target_name:   # falls are legal; nothing sinks 1 cm below min(start, ground)
            z = self.s0.body_q.numpy()[:, 2].astype(np.float64)
            floor = np.minimum(self._z0, self._ground) + FLOOR_M
            out.update(min_z_margin_m=float(np.min(z - floor)), ok=bool(np.all(z > floor)))
        elif self.tt is None:
            z = float(self.s0.body_q.numpy()[:, 2].min())
            out.update(min_z_m=z, ok=z > FLOOR_M)
        else:
            err = self.replay_error(GATE_S)
            out.update(replay_point_err_max_m=err, replay_window_s=GATE_S, ok=err <= GATE_M)
        ok = out.pop("ok")
        out["pass"] = bool(finite and self.overflow == 0 and ok)
        return out

    def replay_error(self, seconds: float) -> float:
        """Max over worlds, objects and reference rows of the point error vs the recording, from a reset."""
        tt = self.tt
        every = int(tt.meta.get("ref_every", 50))
        if every % self.periods_per_call:
            raise ValueError(f"ref_every {every} is not a multiple of {self.periods_per_call} periods per call")
        rows = min(int(round(seconds / tt.trace.control_dt)), len(tt.ref.times) * every)
        bodies = _world_bodies(self.model, lookup(body_index(self.model), tt.ref.names, "body"), self.n)
        self.reset()
        worst = 0.0
        for k in range(rows // every):
            for _ in range(every // self.periods_per_call):
                self.step()
            q = self.s0.body_q.numpy().astype(np.float64)[bodies]  # [W, n_obj, 7]
            point = pose_errors(q[..., :3], q[..., 3:], tt.ref.pos[k], tt.ref.quat[k], tt.ref.radius, True)[2]
            worst = max(worst, float(np.max(point)))
        return worst


# ---- bindings -----------------------------------------------------------------------------------------------------
def _check_layout(model: newton.Model, n: int) -> None:
    """Worlds are replicas laid out world-major (the trace binds world 0 and is tiled)."""
    for name, world in (("body", model.body_world.numpy()), ("joint", model.joint_world.numpy())):
        if len(world) % n or not np.array_equal(world, np.repeat(np.arange(n), len(world) // n)):
            raise ValueError(f"task rung needs world-major {name}s without global entities")


def _world_bodies(model: newton.Model, world0: np.ndarray, n: int) -> np.ndarray:
    per = model.body_count // n
    return world0[None, :] + per * np.arange(n)[:, None]


def _mj_joints(solver: Any, newton_joints: np.ndarray) -> np.ndarray:
    """MuJoCo-Warp joint index (world 0) of each Newton joint."""
    to_newton = solver.mjc_jnt_to_newton_jnt.numpy()[0]
    inverse = {int(j): k for k, j in enumerate(to_newton) if j >= 0}
    return np.array([inverse[int(j)] for j in newton_joints], dtype=np.int64)


def _bind_trace(r: TaskRung) -> TraceReplayer:
    tr, model, jindex = r.tt.trace, r.model, joint_index(r.model)
    inputs, targets, layouts = {}, {}, {}
    if tr.inputs[CTRL].shape[1]:
        inputs[CTRL], targets[CTRL] = tr.inputs[CTRL], r.control.mujoco.ctrl
    if tr.inputs[JOINT_F].shape[1]:
        dofs = model.joint_qd_start.numpy()[lookup(jindex, tr.columns[JOINT_F], "joint")]
        inputs[JOINT_F], targets[JOINT_F] = tr.inputs[JOINT_F], r.control.joint_f
        layouts[JOINT_F] = ReplayLayout(dofs, model.joint_dof_count // r.n)
    if tr.inputs[ACTFRC].shape[1]:
        arr = r.solver.mjw_model.jnt_actfrcrange  # [W, njnt] vec2 (batched per world)
        if arr.ndim != 2 or arr.shape[0] != r.n:
            raise ValueError(f"jnt_actfrcrange is {arr.shape}, expected per-world [{r.n}, njnt]")
        mj = _mj_joints(r.solver, lookup(jindex, tr.columns[ACTFRC], "joint"))
        inputs[ACTFRC] = tr.inputs[ACTFRC].reshape(tr.inputs[ACTFRC].shape[0], -1)
        targets[ACTFRC] = arr.view(wp.float32)
        layouts[ACTFRC] = ReplayLayout(np.stack([2 * mj, 2 * mj + 1], axis=1).ravel(), 2 * arr.shape[1])
    return TraceReplayer(inputs, targets, r.n, r.device, layouts)


def _bind_b0(r: TaskRung) -> TraceReplayer:
    """The target's B0 random-legal ctrl traces, world w playing row w % 2048 of seed 100000, as the SB-7 campaign
    and the P3-2 rival runs play them (alloy.bully.stress.random_legal_traces, legalised again as EpisodeRunner
    does); past the 2 s horizon the last row is held."""
    from alloy.bully.actions import MUJOCO_CTRL, legalize
    from alloy.bully.stress import random_legal_traces
    from alloy.bully.targets import get_target

    tg = get_target(r.target_name)
    if tg.action_kind != MUJOCO_CTRL:
        raise ValueError(f"target {r.target_name}: B0 rung needs mujoco.ctrl actions, has {tg.action_kind}")
    sp = tg.actions
    rows = random_legal_traces(tg, min(r.n, B0_ROWS), 2.0, seed=B0_SEED)
    rows = legalize(rows, sp.low, sp.high, sp.max_delta_per_tick, sp.initial).trace  # [W, T, A], one row per tick
    hold = max(1, int(round(1.0 / (sp.rate_hz * r.control_dt))))   # periods per B0 tick (20 ms / control_dt)
    per_period = np.repeat(rows.transpose(1, 0, 2), hold, axis=0)
    return TraceReplayer({CTRL: np.ascontiguousarray(per_period)}, {CTRL: r.control.mujoco.ctrl}, r.n, r.device)


def _target_initial(name: str) -> tuple[np.ndarray, np.ndarray]:
    """One world's joint_q / joint_qd as the bully target sets them (its make_sim, CPU, one world)."""
    from alloy.bully.targets import get_target

    model = get_target(name).make_sim(1, "cpu").model
    return model.joint_q.numpy().astype(np.float64), model.joint_qd.numpy().astype(np.float64)


def _initial_joint_state(r: TaskRung) -> tuple[np.ndarray, np.ndarray]:
    q = r.model.joint_q.numpy().astype(np.float64)
    qd = np.zeros(r.model.joint_dof_count)
    if r.tt is None and r.target_name:
        q1, qd1 = _target_initial(r.target_name)
        return np.tile(q1, r.n), np.tile(qd1, r.n)
    if r.tt is None:
        return q, qd
    tr = r.tt.trace
    q, qd = initial_to_newton(r.model, tr.columns, tr.initial[QPOS], tr.initial[QVEL])  # fills world 0
    nq, nd = len(q) // r.n, len(qd) // r.n
    return np.tile(q[:nq], r.n), np.tile(qd[:nd], r.n)


def _seed_warmstart(r: TaskRung) -> None:
    """Recorded ``qacc_warmstart`` (MuJoCo dof order) into every world's MuJoCo-Warp data."""
    tr = r.tt.trace
    if WARM not in tr.columns or WARM not in tr.initial:
        return
    mjm, jindex = r.solver.mj_model, joint_index(r.model)
    warm = r.solver.mjw_data.qacc_warmstart.numpy()
    for name, start, width in runs(tr.columns[WARM]):
        k = int(_mj_joints(r.solver, lookup(jindex, (name,), "joint"))[0])
        adr = int(mjm.jnt_dofadr[k])
        warm[:, adr:adr + width] = tr.initial[WARM][start:start + width]
    r.solver.mjw_data.qacc_warmstart.assign(warm)


# ---- embodiment rungs: the SB-7 target's clock, capacities and route -------------------------------------------------
def _target_capacities(name: str, n: int, device: str) -> PackSpec:
    """The bully target's per-world MuJoCo Warp capacities (``EmbTargetSpec``: nconmax, njmax) for ``n`` worlds."""
    from alloy.tasks.emb_specs import EMBODIMENTS
    from alloy.tasks.emb_targets import EmbTargetSpec

    key = next(e.key for e in EMBODIMENTS.values() if e.name == name)
    ts = EmbTargetSpec(key)
    return pack_spec(n, device, {"rigid_contact_max": ts.nconmax * n, "njmax": ts.njmax, "nconmax": ts.nconmax,
                                 "shape_pairs_max": 0})


def _clock(r: TaskRung) -> tuple[float, int, float]:
    """(control_dt, steps per period, dt): the card's clock, or for a target rung the target's action period with
    the stepper's dt for it (as ``emb_targets._build``: ``choose_clock(control_dt, None)``)."""
    if not r.target_name:
        k = r.sim.clock.steps_per_period
        return r.built.control_dt, k, r.built.control_dt / k
    from alloy.bully.targets import get_target

    control_dt = 1.0 / get_target(r.target_name).actions.rate_hz
    dt, _ = r.stepper.choose_clock(control_dt, None)
    k = int(round(control_dt / dt))
    if k < 1 or abs(k * dt - control_dt) > 1e-9 * control_dt:
        raise ValueError(f"{r.task}: control period {control_dt} s is not a whole number of steps of {dt} s")
    return control_dt, k, control_dt / k


def _route(r: TaskRung) -> str:
    """WP5's ordered MuJoCo Warp route, on as for the SB-7 target (``ALLOY_MJW_ROUTE=off`` keeps it off)."""
    from alloy.precision.mjw_route import enable_route

    why = enable_route(r.solver)
    return "on" if why is None else f"off ({why})"


def _ground_z(model: newton.Model) -> float:
    """Height of the lowest static ground plane (0 if the scene has none)."""
    typ, body = model.shape_type.numpy(), model.shape_body.numpy()
    planes = np.nonzero((typ == int(newton.GeoType.PLANE)) & (body < 0))[0]
    if planes.size == 0:
        return 0.0
    return float(np.min(model.shape_transform.numpy()[planes, 2]))


def pack_spec(n: int, device: str, caps: dict[str, int]) -> PackSpec:
    """A throughput PackSpec of ``n`` consecutive worlds carrying ``costmodel.capacities`` output."""
    return PackSpec(device=device, world_ids=tuple(range(n)), n=n, stream_priority=0, kind="throughput",
                    rigid_contact_max=caps["rigid_contact_max"], njmax=caps["njmax"], nconmax=caps["nconmax"],
                    shape_pairs_max=caps["shape_pairs_max"], predicted_bytes=0)


__all__ = ["GATE_M", "GATE_S", "TASKS", "TaskRung", "TaskSpec", "pack_spec"]
