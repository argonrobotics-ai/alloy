# SPDX-License-Identifier: Apache-2.0
"""Read named state out of an engine simulation (host copies), and write joint targets into it."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from alloy.api import _anchor, _heat, _joint_yield, _liquid, _rods, _targets
from alloy.api._actuators import Actuators
from alloy.api.errors import SimulationError, UnknownNameError
from alloy.api.names import NameIndex


class StateIO:
    """Gathers per-world arrays for names; one host copy of each engine array per call."""

    def __init__(self, engine: Any, index: NameIndex) -> None:
        self.engine, self.index = engine, index
        m = engine.model
        self.q_start = m.joint_q_start.numpy() if m.joint_count else np.zeros(1, np.int64)
        self.qd_start = m.joint_qd_start.numpy() if m.joint_count else np.zeros(1, np.int64)
        self.target_start = getattr(m, "joint_target_q_start", None)
        self.target_start = self.target_start.numpy() if self.target_start is not None else self.q_start
        self.acts = Actuators.from_model(m, index.robot_joints, self.qd_start)
        self.limits = _targets.Limits.from_model(m, self.qd_start)       # position targets stop at the joint's range
        from alloy._vendor import newton  # noqa: PLC0415 - the engine is loaded once a Sim exists

        self.active_flag = int(newton.ParticleFlags.ACTIVE)
        mj = getattr(m, "mujoco", None)
        self.dof_ref = mj.dof_ref.numpy() if mj is not None and hasattr(mj, "dof_ref") else None
        built = getattr(engine, "built", None)
        self.rods = _rods.rod_lines(built) if built is not None and hasattr(built, "handles") else {}
        self.wrench_rows: set[int] = set()
        self.j2_sheets = has_plastic_strain(engine)        # "<sheet>.plastic_strain" read only when some film has them
        # objects placed by their anchor (Scene.add_object): .pos reads that point, .body_pos the body frame
        self.anchors = {n: np.asarray(c, np.float64) for n, c in getattr(engine, "_alloy_anchors", {}).items()
                        if n in index.bodies}

    def point(self, name: str, body_q: np.ndarray) -> np.ndarray:
        """(W, 3) ``"<name>.pos"`` for the body's rows of ``body_q``: its anchor point in the world for an object
        placed by one, else the body origin. Example: ``io.point("can", body_q[rows])``."""
        c = self.anchors.get(name)
        return body_q[:, :3].copy() if c is None else _anchor.world_points(body_q, c)

    def expand(self, names: Iterable[str] | str | None) -> list[str]:
        """Names to read: None = every body, joint and particle object. Example: ``io.expand(["cube"])``."""
        if names is None:
            idx = self.index
            every = list(idx.bodies) + [j for j in idx.joints if "/" in j] + list(idx.particles)
            return list(dict.fromkeys(every))        # a robot joint named like its body: once, read as both
        names = [names] if isinstance(names, str) else list(names)
        for n in names:
            if n not in self.index.bodies and n not in self.index.joints and n not in self.index.particles:
                raise self.index._unknown(n, "body, joint or particle object")
        return names

    def read(self, names: list[str]) -> dict[str, np.ndarray]:
        """``{"<name>.pos": (W, 3), ...}`` for the given names (see ``alloy.Trajectory`` for the keys)."""
        st, idx, out = self.engine.state, self.index, {}
        bq = st.body_q.numpy() if any(n in idx.bodies for n in names) else None
        bqd = st.body_qd.numpy() if bq is not None else None
        jq = st.joint_q.numpy() if any(n in idx.joints for n in names) else None
        pq = st.particle_q.numpy() if any(n in idx.particles for n in names) else None
        pf = _liquid.particle_flags(self.engine) if pq is not None else None
        ps = plastic_strain(self.engine) if pq is not None and self.j2_sheets else None
        for n in names:
            if n in idx.bodies:
                rows = idx.per_world(idx.bodies[n], "body")
                q = bq[rows]
                out[f"{n}.pos"] = self.point(n, q)
                if n in self.anchors:
                    out[f"{n}.body_pos"] = q[:, :3].copy()
                out[f"{n}.quat_xyzw"] = q[:, 3:7].copy()
                out[f"{n}.quat_wxyz"] = q[:, [6, 3, 4, 5]].copy()
                out[f"{n}.vel"] = bqd[rows, :3].copy()
                if n in self.rods:                     # a rod's name: its first segment, plus the centre line
                    line = self.rods[n]
                    seg_rows = np.stack([idx.per_world(int(b), "body") for b in line.bodies])
                    out[f"{n}.points"] = _rods.points(line, bq, seg_rows)
                    out[f"{n}.twist"] = _rods.twist(line, bq, seg_rows)
                    wire = _rods.plastic_strain(self.engine, seg_rows)     # not `ps`: that is the sheets' read, below
                    if wire is not None:               # a wire that yields (J2): per joint, its fibres' largest
                        out[f"{n}.plastic_strain"] = wire
            if n in idx.joints:                       # also a robot joint named like its body (AUDIT-STACK PHYSB-2)
                out[f"{n}.q"] = self._joint_values(jq, idx.joints[n])
            if n in idx.particles:
                r = idx.particles[n]
                rows = idx.per_world(r.start, "particle")[:, None] + np.arange(len(r))[None, :]
                out[f"{n}.points"] = pq[rows].copy()
                out[f"{n}.active"] = (pf[rows] & self.active_flag).astype(bool)   # False: not poured yet / retired
                if _heat.is_heated(self.engine, n):
                    out[f"{n}.temperature"] = _heat.temperature(self.engine, n)
                if ps is not None and ps[1][rows].any():               # a J2 metal sheet: its hinges' yield
                    out[f"{n}.plastic_strain"] = ps[0][rows]
        out.update(_joint_yield.readout(self.engine, idx, [n for n in names if n in idx.joints]))
        return out

    def _joint_values(self, jq: np.ndarray, joint: int) -> np.ndarray:
        rows = self.index.per_world(joint, "joint")
        lo, hi = self.q_start[rows], self.q_start[rows + 1]
        width = int(hi[0] - lo[0])
        vals = np.stack([jq[a:a + width] for a in lo])
        return vals[:, 0].copy() if width == 1 else vals

    def set_targets(self, targets: Mapping[str, Any], kind: str = "position") -> None:
        """Write joint targets: ``{"joint": value or (worlds,) values}``. ``kind``: "position" or "velocity".

        Joints driven from joint targets (MJCF ``<position>`` / ``<velocity>``) get ``joint_target_q`` / ``_qd``;
        joints with a ``<general>`` servo get MuJoCo ``ctrl`` (see ``alloy.api._actuators``). A position target stops
        at the joint's range (``_targets.Limits``); a joint that nothing drives raises."""
        ctrl = self.engine.control
        arr = ctrl.joint_target_q if kind == "position" else ctrl.joint_target_qd
        host = arr.numpy() if arr is not None else None
        start = self.target_start if kind == "position" else self.qd_start
        direct: dict[int, np.ndarray] = {}
        for name, value in targets.items():
            j = self.index.joint(name)
            vals = _per_world(value, self.index.worlds, name)
            if kind == "position":
                vals = self.limits.clamp(name, j, vals, self._ctrl_span(j))
            servo = (self.acts.position if kind == "position" else self.acts.velocity).get(j)
            if servo is not None:
                direct[servo] = vals + (self._ref(servo) if kind == "position" else 0.0)
            elif host is not None and (j in self.acts.joint_target or self.limits.driven(j, kind, self.acts.stride)):
                host[start[self.index.per_world(j, "joint")]] = vals       # no actuator table: Newton's own targets
            else:
                raise _not_driven(name, kind, self.acts.actuated(j))
        if direct:
            self._write_ctrl(direct)
        if host is not None:
            arr.assign(host)

    def set_controls(self, controls: Mapping[str, Any]) -> None:
        """Raw MuJoCo ``ctrl`` per actuator name ("<robot>/<actuator>" or a unique "<actuator>"). The ctrl of an MJCF
        ``<position>`` / ``<velocity>`` actuator is the joint position (MuJoCo's qpos) / velocity it servos to, and the
        engine drives those from joint targets, so it is written there."""
        rows, ctrl = {}, self.engine.control
        for name, value in controls.items():
            if name not in self.acts.names:
                raise UnknownNameError(f"no actuator named {name!r}", f"use one of {sorted(self.acts.names)[:20]}",
                                       field=name)
            row, vals = self.acts.names[name], _per_world(value, self.index.worlds, name)
            joint, kind = self.acts.target_rows.get(row, (None, None))
            if joint is None:
                rows[row] = vals
            elif kind == "position":
                _targets.write(ctrl.joint_target_q, self.target_start[self.index.per_world(joint, "joint")],
                               vals - self._ref(row))
            elif kind == "velocity":
                _targets.write(ctrl.joint_target_qd, self.qd_start[self.index.per_world(joint, "joint")], vals)
            else:
                raise SimulationError(f"actuator {name!r} is driven from joint targets, and its gains are neither a "
                                      "position nor a velocity servo's", "drive its joint with sim.set_targets",
                                      field=name)
        if rows:
            self._write_ctrl(rows)

    def apply_wrench(self, name: str, force: Any, torque: Any, at: str | None) -> None:
        """Hold a world-frame force and torque about a body's COM in ``body_f`` of both of the engine's states (it
        swaps them every exchange and never clears ``body_f``). ``at``: a rod's "start" or "end" segment. Health
        books the wrench's work as outside work, as a force plug-in's (``health_work.WrenchWork``)."""
        states = [getattr(self.engine, "state", None), getattr(self.engine, "_next", None)]
        if any(st is None or getattr(st, "body_f", None) is None for st in states):
            raise SimulationError("apply_wrench needs an in-process engine with body forces",
                                  "build without processes= to apply wrenches", field=name)
        rows = self.index.per_world(self._wrench_body(name, at), "body")
        w = self.index.worlds
        f, t = _per_world_vec(force, w, "force"), _per_world_vec(torque, w, "torque")
        self.wrench_rows.update(int(r) for r in rows)
        for st in states:
            host = st.body_f.numpy()
            host[rows, :3], host[rows, 3:6] = f, t
            st.body_f.assign(host)
        self._book_wrench(rows, f, t)

    def _book_wrench(self, rows: np.ndarray, f: np.ndarray, t: np.ndarray) -> None:
        """The held wrench's work goes into health's energy check (EASE-TEST red 2026-11-09: unbooked, a 2 N push
        read as ENERGY_JUMP and froze the world). The first wrench changes the step, so captured graphs are dropped."""
        from alloy.api import _drive  # noqa: PLC0415
        from alloy.precision import health_work  # noqa: PLC0415

        ww, made = health_work.wrench_work(self.engine)
        if ww is None:
            return
        ww.hold(rows, f, t)
        if made:
            _drive.invalidate_graphs(self.engine)

    def clear_wrenches(self) -> None:
        """Zero every body_f row apply_wrench wrote (sim.reset). Example: ``io.clear_wrenches()``."""
        if not self.wrench_rows:
            return
        rows = np.fromiter(self.wrench_rows, np.int64)
        from alloy.precision.health_work import WRENCH_ATTR  # noqa: PLC0415

        ww = getattr(self.engine, WRENCH_ATTR, None)
        if ww is not None:
            ww.clear()
        for st in (getattr(self.engine, "state", None), getattr(self.engine, "_next", None)):
            if st is not None and getattr(st, "body_f", None) is not None:
                host = st.body_f.numpy()
                host[rows] = 0.0
                st.body_f.assign(host)
        self.wrench_rows.clear()

    def match_states(self) -> None:
        """After ``sim.reset``: the engine's other state (``_next``) takes every array of ``state``. The engine swaps
        the two every exchange and the reset restores ``state`` alone, so a row no solver writes (a kinematic body in
        an MPM-only scene) came back from before the reset every second exchange. Example: ``io.match_states()``."""
        nxt = getattr(self.engine, "_next", None)
        if nxt is not None:
            nxt.assign(self.engine.state)

    def _wrench_body(self, name: str, at: str | None) -> int:
        if at is None:
            return self.index.body(name)
        if at not in ("start", "end"):
            raise SimulationError(f"at must be 'start', 'end' or None, got {at!r}", "pass at='end' for a rod's free "
                                  "end, or no at= for a body", field="at")
        if name not in self.rods:
            raise UnknownNameError(f"{name!r} is not a rod, so at= does not apply", f"rods: {sorted(self.rods)}; "
                                   "drop at= for a body", field=name)
        return int(self.rods[name].bodies[0 if at == "start" else -1])

    def _ctrl_span(self, joint: int) -> tuple[float, float] | None:
        """The ctrl range of the joint's position servo as joint targets (ctrl is MuJoCo's qpos: minus the joint's
        ref), or None when the robot file gives the servo none."""
        row, lo, hi = self.acts.ctrl_span.get(joint, (None, 0.0, 0.0))
        if row is None:
            return None
        ref = float(self._ref(row)[0])
        return lo - ref, hi - ref

    def _ref(self, row: int) -> np.ndarray:
        dof = self.acts.dof_of_row[row]
        if self.dof_ref is None:
            return np.zeros(self.index.worlds)
        per_world = len(self.dof_ref) // max(self.index.worlds, 1)
        return self.dof_ref[dof + np.arange(self.index.worlds) * per_world]

    def _write_ctrl(self, rows: Mapping[int, np.ndarray]) -> None:
        mj = getattr(self.engine.control, "mujoco", None)
        if mj is None or getattr(mj, "ctrl", None) is None:
            raise SimulationError("this scene has no MuJoCo actuator controls", "add actuators to the robot file")
        host = mj.ctrl.numpy()
        for row, vals in rows.items():
            host[self.acts.rows(row, self.index.worlds)] = vals
        mj.ctrl.assign(host)


def has_plastic_strain(engine: Any) -> bool:
    """Whether any solver of ``engine`` carries J2 sheet hinges. Example: ``has_plastic_strain(sim.engine)``."""
    return any(getattr(s, "has_plastic_strain", False) for s in _liquid.all_solvers(engine))


def plastic_strain(engine: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Per model particle: (the largest layer ε̄p of the J2 sheet hinges along it, whether it has any), or None
    when no solver carries J2 sheets. Example: ``plastic_strain(sim.engine)[0].max()``."""
    found = None
    for solver in _liquid.all_solvers(engine):
        if not getattr(solver, "has_plastic_strain", False):
            continue
        got = solver.plastic_strain()
        if found is None:
            n = int(engine.model.particle_count)
            found = (np.zeros(n), np.zeros(n, dtype=bool))
        ids = _liquid._local_to_global(engine, solver)
        ids = np.arange(len(got[0])) if ids is None else ids
        k = len(ids)
        found[0][ids] = np.maximum(found[0][ids], got[0][:k])
        found[1][ids] |= got[1][:k]
    return found


def _not_driven(name: str, kind: str, actuated: bool) -> SimulationError:
    if actuated:
        return SimulationError(f"joint {name!r} has an actuator that does not take {kind} targets (a motor, or a "
                               "<general> actuator that is not a servo)", "drive it with sim.set_controls({actuator: "
                               "value}) (names: sim.names()['actuators'])", field=name)
    return SimulationError(f"joint {name!r} has no actuator, so a {kind} target cannot move it",
                           f"add <{kind} joint=\"...\" kp=\"...\"/> for it to the robot's MJCF <actuator> block",
                           field=name)


def _per_world_vec(value: Any, worlds: int, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape == (3,):
        return np.broadcast_to(v, (worlds, 3))
    if v.shape == (worlds, 3) and np.isfinite(v).all():
        return v
    raise SimulationError(f"{name} has shape {v.shape}, expected (3,) or ({worlds}, 3) finite values",
                          f"pass {name}=(x, y, z) in world coordinates (N or N m)", field=name)


def _per_world(value: Any, worlds: int, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape == ():
        return np.full(worlds, float(v))
    if v.shape == (worlds,):
        return v
    raise UnknownNameError(f"target for {name!r} has shape {v.shape}, expected () or ({worlds},)",
                           "pass one number for every world or one per world", field=name)
