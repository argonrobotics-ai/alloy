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
        self.acts = Actuators.from_model(m, index.robot_joints, self.qd_start, index.joint_short or None)
        self.controls0: dict[str, np.ndarray] | None = None    # targets and ctrl before the first command (reset)
        self.limits = _targets.Limits.from_model(m, self.qd_start)       # position targets stop at the joint's range
        from alloy._vendor import newton  # noqa: PLC0415 - the engine is loaded once a Sim exists

        self.active_flag = int(newton.ParticleFlags.ACTIVE)
        mj = getattr(m, "mujoco", None)
        self.dof_ref = mj.dof_ref.numpy() if mj is not None and hasattr(mj, "dof_ref") else None
        self.joint_type = m.joint_type.numpy() if m.joint_count else np.zeros(0, np.int32)
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
        at the joint's range (``_targets.Limits``); a joint that nothing drives, or whose servo takes the other kind,
        raises. A ``<general>`` servo's ctrl is MuJoCo's: gear x (the target, plus the joint's ref for a position).
        A body with several MJCF joints loads as one joint ("<a>_<b>") of one coordinate per MJCF joint: it takes one
        value per coordinate, in the file's order (``(n,)`` or ``(worlds, n)``), each to its own servo. A fixed
        joint has no coordinate and raises."""
        self._keep_controls()
        arr = self._target_array(kind)
        host = arr.numpy().copy() if arr is not None else None   # a copy: a call that raises writes nothing (on the
        start = self.target_start if kind == "position" else self.qd_start      # CPU numpy() is the array itself)
        direct: dict[int, np.ndarray] = {}
        for name, value in targets.items():
            j = self.index.joint(name)
            per_axis = self._axis_values(name, j, value)
            for k in range(per_axis.shape[1]):
                self._target_axis((name, j, k), per_axis[:, k], kind, (host, start), direct)
        if direct:
            self._write_ctrl(direct)
        if host is not None:
            arr.assign(host)

    def _target_array(self, kind: str) -> Any:
        ctrl = self.engine.control
        return ctrl.joint_target_q if kind == "position" else ctrl.joint_target_qd

    def _route(self, name: str, j: int, kind: str, has_targets: bool, axis: int = 0) -> int | None:
        """The servo (actuator row) that takes coordinate ``axis`` of joint ``j``'s ``kind`` target, None for Newton's
        own joint target; raises when nothing drives it that way, and for a fixed joint (no coordinate)."""
        if int(self.qd_start[j + 1] - self.qd_start[j]) == 0:
            raise _fixed(name)
        dof = int(self.qd_start[j]) + axis
        servo = (self.acts.position_dof if kind == "position" else self.acts.velocity_dof).get(dof)
        if servo is not None:
            return servo
        if kind not in self._kinds(dof, kind):
            raise _other_kind(name, kind, self._kinds(dof, kind))
        if has_targets and (dof in self.acts.target_dofs or self.limits.driven(j, kind, self.acts.stride, axis)):
            return None
        raise _not_driven(name, kind, self.acts.actuated(j))

    def ramps(self, targets: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``Sim.run``'s position targets as ``{joint: (start, end)}``, (worlds,) values each: a value holds, a
        (start, end) pair ramps. Checked as ``set_targets`` checks a target (name, shape, finite, driven) before the
        run steps, so a run too short for one period refuses a bad target too. Example:
        ``io.ramps({"lift": (0.0, 0.12)})``."""
        w, out, has_targets = self.index.worlds, {}, self._target_array("position") is not None
        for name, value in targets.items():
            a, b = value if isinstance(value, tuple) and len(value) == 2 else (value, value)
            out[name] = (_per_world(a, w, name), _per_world(b, w, name))
            self._route(name, self.index.joint(name), "position", has_targets)
        return out

    def _axis_values(self, name: str, joint: int, value: Any) -> np.ndarray:
        """``value`` as (worlds, axes): one axis for a joint of one coordinate (and for a ball or free joint, as
        before), one per coordinate for a body's several MJCF joints; a fixed joint raises."""
        n = int(self.qd_start[joint + 1] - self.qd_start[joint])
        if n == 0:
            raise _fixed(name)
        worlds = self.index.worlds
        if n == 1 or int(self.joint_type[joint]) != D6_JOINT:
            return _per_world(value, worlds, name)[:, None]
        try:
            v = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            v = np.full(1, np.nan)
        if (v.shape == (n,) or v.shape == (worlds, n)) and np.isfinite(v).all():
            return np.broadcast_to(v, (worlds, n)).copy()
        raise SimulationError(f"joint {name!r} is {n} MJCF joints on one body, one coordinate each (named in the "
                              f"file's order), so its target is {n} finite values; got {value!r}",
                              f"pass {n} values, ({n},) or ({worlds}, {n}), or give each MJCF joint its own nested "
                              "body to drive it by its own name", field=name)

    def _target_axis(self, where: tuple[str, int, int], vals: np.ndarray, kind: str, out: tuple[Any, Any],
                     direct: dict[int, np.ndarray]) -> None:
        """One coordinate's target: its own servo's ctrl, or the joint target array at that coordinate."""
        name, j, k = where
        host, start = out
        if kind == "position":
            vals = self.limits.clamp(name, j, vals, self._ctrl_span(j, k), axis=k)
        servo = self._route(name, j, kind, host is not None, k)
        if servo is not None:
            direct[servo] = self.acts.gear_of(servo) * (vals + (self._ref(servo) if kind == "position" else 0.0))
        else:
            host[start[self.index.per_world(j, "joint")] + k] = vals    # no actuator table: Newton's own targets

    def set_controls(self, controls: Mapping[str, Any]) -> None:
        """Raw MuJoCo ``ctrl`` per actuator name ("<robot>/<actuator>" or a unique "<actuator>"). The ctrl of an MJCF
        ``<position>`` / ``<velocity>`` actuator is the joint position (MuJoCo's qpos) / velocity it servos to, and the
        engine drives those from joint targets, so it is written there (divided by the actuator's gear, as MuJoCo's
        actuator length is gear x the joint value)."""
        self._keep_controls()
        rows, ctrl = {}, self.engine.control
        for name, value in controls.items():
            row, vals = self._actuator(name), _per_world(value, self.index.worlds, name)
            joint, kind = self.acts.target_rows.get(row, (None, None))
            if joint is None:
                rows[row] = vals
            elif kind in ("position", "velocity"):          # its own coordinate of the joint (several on one body)
                axis = self.acts.dof_of_row[row] - int(self.qd_start[joint])
                arr, start = ((ctrl.joint_target_q, self.target_start) if kind == "position"
                              else (ctrl.joint_target_qd, self.qd_start))
                ref = self._ref(row) if kind == "position" else 0.0
                _targets.write(arr, start[self.index.per_world(joint, "joint")] + axis,
                               vals / self.acts.gear_of(row) - ref)
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

    def after_reset(self) -> None:
        """``sim.reset``: loads off, joint targets and ctrl back to the start, and both of the engine's states alike.
        Example: ``io.after_reset()``."""
        self.clear_wrenches()
        self._restore_controls()
        self.match_states()

    def save_controls(self) -> None:
        """Take the joint targets and ctrl now as those ``sim.reset`` returns to (``sim.settle``, and before the first
        command). Example: ``io.save_controls()``."""
        self.controls0 = {k: a.numpy().copy() for k, a in self._control_arrays().items()}

    def _keep_controls(self) -> None:
        if self.controls0 is None:
            self.save_controls()

    def _restore_controls(self) -> None:
        """The targets and ctrl of the start: a reset no longer steps toward the last command (a robot snapped back
        to its last targets after ``sim.reset()``)."""
        for key, arr in self._control_arrays().items():
            if self.controls0 is not None and key in self.controls0:
                arr.assign(self.controls0[key])

    def _control_arrays(self) -> dict[str, Any]:
        ctrl = self.engine.control
        mj = getattr(ctrl, "mujoco", None)
        arrays = {"q": getattr(ctrl, "joint_target_q", None), "qd": getattr(ctrl, "joint_target_qd", None),
                  "ctrl": getattr(mj, "ctrl", None) if mj is not None else None}
        return {k: a for k, a in arrays.items() if a is not None}

    def _kinds(self, dof: int, kind: str) -> set[str]:
        """The target kinds the coordinate's servos take (``{kind}`` when none is known: a motor or Newton's own
        gains)."""
        kinds = set(self.acts.target_dofs.get(dof, set()))
        kinds |= {k for k in ("position", "velocity") if dof in getattr(self.acts, f"{k}_dof")}
        return kinds or {kind}

    def _actuator(self, name: str) -> int:
        if name in self.acts.names:
            return self.acts.names[name]
        if name in self.acts.ambiguous:
            raise UnknownNameError(f"actuator name {name!r} is in more than one robot",
                                   f"name the robot: one of {self.acts.ambiguous[name]}", field=name)
        raise UnknownNameError(f"no actuator named {name!r}", f"use one of {sorted(self.acts.names)[:20]}", field=name)

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

    def _ctrl_span(self, joint: int, axis: int = 0) -> tuple[float, float] | None:
        """The ctrl range of the joint's position servo as joint targets (ctrl is MuJoCo's qpos: minus the joint's
        ref), or None when the robot file gives the servo none."""
        row, lo, hi = self.acts.span_dof.get(int(self.qd_start[joint]) + axis, (None, 0.0, 0.0))
        if row is None:
            return None
        ref, g = float(self._ref(row)[0]), self.acts.gear_of(row)
        return tuple(sorted((lo / g - ref, hi / g - ref)))

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


D6_JOINT = 6        # newton.JointType.D6: a body's several MJCF joints, one coordinate each


def _fixed(name: str) -> SimulationError:
    return SimulationError(f"joint {name!r} is fixed: it has no coordinate, so a target cannot move it (its slot "
                           "would be the next joint's)", "drive a moving joint; a fixed joint's .q is empty",
                           field=name)


def _other_kind(name: str, kind: str, kinds: set[str]) -> SimulationError:
    other = " and ".join(sorted(kinds))
    return SimulationError(f"joint {name!r} is driven by a {other} servo, so a {kind} target does nothing",
                           f"use sim.set_targets(..., kind={sorted(kinds)[0]!r}), or add a <{kind}> actuator for it to "
                           "the robot's MJCF", field=name)


def _not_driven(name: str, kind: str, actuated: bool) -> SimulationError:
    if actuated:
        return SimulationError(f"joint {name!r} has an actuator that does not take {kind} targets (a motor, or a "
                               "<general> actuator that is not a servo)", "drive it with sim.set_controls({actuator: "
                               "value}) (names: sim.names()['actuators'])", field=name)
    return SimulationError(f"joint {name!r} has no actuator, so a {kind} target cannot move it",
                           f"give the joint a servo in the robot file. MJCF: add <{kind} joint=\"...\" kp=\"...\"/> "
                           "to its <actuator> block. USD: a PhysicsDriveAPI stiffness (damping for a velocity "
                           "target) on the joint. A URDF has no servo gains (a <transmission> names an interface "
                           "only): convert it to MJCF and add the actuator there", field=name)


def _per_world_vec(value: Any, worlds: int, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape == (3,):
        return np.broadcast_to(v, (worlds, 3))
    if v.shape == (worlds, 3) and np.isfinite(v).all():
        return v
    raise SimulationError(f"{name} has shape {v.shape}, expected (3,) or ({worlds}, 3) finite values",
                          f"pass {name}=(x, y, z) in world coordinates (N or N m)", field=name)


def _per_world(value: Any, worlds: int, name: str) -> np.ndarray:
    """A target or control: one number for every world or one per world, finite, as (worlds,) float64. A NaN
    passed the range clamp and reached MuJoCo's ctrl, reported as "held at the limit"."""
    try:
        v = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise SimulationError(f"the value for {name!r} is not a number: {value!r}",
                              "pass a number (m or rad) for every world, or a list with one per world",
                              field=name) from None
    if v.shape not in ((), (worlds,)):
        raise SimulationError(f"the value for {name!r} has shape {v.shape}, expected () or ({worlds},)",
                              f"pass one number for every world or one per world ({worlds} values)", field=name)
    if not np.isfinite(v).all():
        raise SimulationError(f"the value for {name!r} is not finite: {v.tolist()}",
                              "pass finite numbers: NaN or inf would reach the actuator as it is", field=name)
    return np.full(worlds, float(v)) if v.shape == () else v
