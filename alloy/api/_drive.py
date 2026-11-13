# SPDX-License-Identifier: Apache-2.0
"""Move kinematic objects: teleport (``set_pose``) or drive with a constant twist for a time (``drive``).

A kinematic object (``Scene.add_box(..., kinematic=True)`` and friends) is a body the solvers do not integrate; it
moves only as told, and everything that touches it (rigid bodies, liquids, cloth) feels its walls and their speed.
``drive`` runs a small kernel after every physics step (a step observer, captured in CUDA graphs) that writes the
body's pose and its velocity, so contacts see a moving wall rather than a jump each control period.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.api")


@wp.kernel(enable_backward=False)
def _drive_kernel(clock: wp.array(dtype=wp.float64), dt: float, q_start: wp.array(dtype=wp.int32),
                  qd_start: wp.array(dtype=wp.int32), body: wp.array(dtype=wp.int32), p0: wp.array(dtype=wp.vec3),
                  r0: wp.array(dtype=wp.quat), pivot: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3),
                  vs: wp.array(dtype=wp.vec3), ws: wp.array(dtype=wp.vec3), t0: wp.float64, t1: wp.float64, pose: int,
                  joint_q: wp.array(dtype=float), joint_qd: wp.array(dtype=float), body_q: wp.array(dtype=wp.transform),
                  body_qd: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    v = vs[i]                                        # one twist per world (a (worlds, 3) drive)
    w = ws[i]
    now = clock[0] + wp.float64(dt)                  # float64: a float32 ``now`` fell just short of t1 (API-17)
    tau = float(wp.min(wp.max(now - t0, wp.float64(0.0)), t1 - t0))
    rate = wp.length(w)
    rot = wp.quat_identity()
    if rate > 0.0:
        rot = wp.quat_from_axis_angle(w / rate, rate * tau)
    piv = pivot[i] + v * tau
    pos = piv + wp.quat_rotate(rot, p0[i] - pivot[i])
    q = rot * r0[i]
    lin = wp.vec3(0.0)
    ang = wp.vec3(0.0)
    if now >= t0 and now < t1 - wp.float64(0.5 * dt):    # the step ending nearest t1 ends at rest (API-17)
        ang = w
        lin = v + wp.cross(w, pos + wp.quat_rotate(q, com[i]) - piv)       # COM velocity (Newton's convention)
    qs = q_start[i]
    ds = qd_start[i]
    b = body[i]
    for k in range(3):
        joint_qd[ds + k] = lin[k]
        joint_qd[ds + 3 + k] = ang[k]
    body_qd[b] = wp.spatial_vector(lin, ang)
    if pose != 0:                                    # 0: ``Drive.prime`` writes the start velocity only
        for k in range(3):
            joint_q[qs + k] = pos[k]
        joint_q[qs + 3] = q[0]
        joint_q[qs + 4] = q[1]
        joint_q[qs + 5] = q[2]
        joint_q[qs + 6] = q[3]
        body_q[b] = wp.transform(pos, q)


@wp.kernel(enable_backward=False)
def _tick(clock: wp.array(dtype=wp.float64), dt: float):
    clock[0] = clock[0] + wp.float64(dt)


@dataclass
class Drive:
    """One constant-twist motion of one kinematic object in every world, each world's own twist and pivot (rows
    of ``arrays["v"]``, ``["w"]``, ``["pivot"]``; a step observer)."""

    arrays: dict[str, Any]
    t0: float
    t1: float
    name: str = ""                       # the object it moves: a new drive of the same object replaces this one

    def __call__(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        self._write(state_out, float(dt), 1)
        wp.launch(_tick, dim=1, inputs=[self.arrays["clock"], float(dt)], device=state_out.joint_q.device)

    def prime(self, engine: Any) -> None:
        """Write the drive's start velocity (pose kept) into both of the engine's states, so the first physics step
        sees the wall moving; the kernel writes only the state each step ends in, which had held the old velocity
        for one step (API-17: a new drive lagged one step). Example: ``drive.prime(sim.engine)``."""
        for st in (engine.state, getattr(engine, "_next", None)):
            if st is not None:
                self._write(st, 0.0, 0)

    def _write(self, st: Any, dt: float, pose: int) -> None:
        a = self.arrays
        wp.launch(_drive_kernel, dim=len(a["body"]),
                  inputs=[a["clock"], dt, a["q_start"], a["qd_start"], a["body"], a["p0"], a["r0"], a["pivot"],
                          a["com"], a["v"], a["w"], self.t0, self.t1, pose, st.joint_q, st.joint_qd, st.body_q,
                          st.body_qd], device=st.joint_q.device)

    def save_period(self) -> None:
        """Keep the clock at a control period's start (``IntegratedSimulation.step_period``, outside graphs)."""
        wp.copy(self.arrays["saved"], self.arrays["clock"])

    def restore_period(self) -> None:
        """Back to the period start's clock when the period re-runs (a coupled scene's rollback restores the state and
        the simulation's clock, not this one); else the object jumps ahead by the discarded attempts' time."""
        wp.copy(self.arrays["clock"], self.arrays["saved"])


def make_drive(sim: Any, name: str, velocity: Any, angular_velocity: Any, pivot: Any, duration: float) -> Drive:
    """A Drive of ``name`` starting now, from its current pose in each world. ``velocity``, ``angular_velocity`` and
    ``pivot``: one (x, y, z) for every world or one row per world, (worlds, 3); ``pivot`` None: each world's
    ``.pos``."""
    idx, m, st = sim.index, sim.engine.model, sim.engine.state
    bodies = idx.per_world(idx.body(name), "body")
    joints = idx.per_world(free_joint(sim, name), "joint")
    n, t = len(bodies), _duration(duration)
    v, w = rows(velocity, n, "velocity"), rows(angular_velocity, n, "angular_velocity")
    bq = st.body_q.numpy()[bodies]
    piv = sim._io.point(name, bq) if pivot is None else rows(pivot, n, "pivot")
    dev = m.device
    arrays = {
        "clock": wp.array([sim.t], dtype=wp.float64, device=dev),
        "saved": wp.array([sim.t], dtype=wp.float64, device=dev),       # the clock at the period start (rollbacks)
        "q_start": wp.array(m.joint_q_start.numpy()[joints].astype(np.int32), dtype=wp.int32, device=dev),
        "qd_start": wp.array(m.joint_qd_start.numpy()[joints].astype(np.int32), dtype=wp.int32, device=dev),
        "body": wp.array(bodies.astype(np.int32), dtype=wp.int32, device=dev),
        "p0": wp.array(bq[:, :3].astype(np.float32), dtype=wp.vec3, device=dev),
        "r0": wp.array(bq[:, 3:7].astype(np.float32), dtype=wp.quat, device=dev),
        "pivot": wp.array(np.asarray(piv, np.float32), dtype=wp.vec3, device=dev),
        "com": wp.array(m.body_com.numpy()[bodies].astype(np.float32), dtype=wp.vec3, device=dev),
        "v": wp.array(v, dtype=wp.vec3, device=dev),
        "w": wp.array(w, dtype=wp.vec3, device=dev),
    }
    t0 = float(sim.t)
    return Drive(arrays, t0, t0 + t, name)


def rows(value: Any, worlds: int, what: str) -> np.ndarray:
    """``value`` as (worlds, 3) float32 rows: one (x, y, z) for every world, or one row per world; a clear error for
    any other shape or a value that is not finite. Example: ``rows((0, 0, 1), 4, "velocity").shape`` -> (4, 3)."""
    from alloy.api.errors import SimulationError  # noqa: PLC0415

    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        a = None
    if a is None or a.shape not in ((3,), (worlds, 3)):
        got = "a value of no numeric shape" if a is None else f"shape {a.shape}"
        raise SimulationError(f"sim.drive {what}: got {got}; it takes one (x, y, z) for every world or one row per "
                              f"world, shape ({worlds}, 3)", f"pass {what}=(x, y, z) or a ({worlds}, 3) array",
                              field=what)
    if not np.isfinite(a).all():
        raise SimulationError(f"sim.drive {what} is not finite: {a.tolist()}", f"pass finite numbers as {what}",
                              field=what)
    return np.ascontiguousarray(np.broadcast_to(a, (worlds, 3)), dtype=np.float32)


def _duration(duration: Any) -> float:
    """``sim.drive``'s duration in s: a number >= 0 (inf: until replaced or stopped)."""
    from alloy.api.errors import SimulationError  # noqa: PLC0415

    try:
        t = float(duration)
    except (TypeError, ValueError):
        t = math.nan
    if not t >= 0.0:
        raise SimulationError(f"sim.drive duration must be a number of seconds >= 0, got {duration!r}",
                              "pass duration=seconds (float('inf') drives until the next drive or stop_drives())",
                              field="duration")
    return t


def replace_drives(engine: Any, drives: list[Drive]) -> None:
    """Add ``drives`` as step observers, removing each object's earlier drive first: the newest drive of an object
    writes its pose last in every step anyway, so the result is the same, but drives no longer pile up (one drive per
    control period for 4 s was 1,600 observers and 3,200 kernel launches per step). Example: see ``Sim.drive``."""
    from alloy.coupling import world_clock_api  # noqa: PLC0415

    moved = {d.name for d in drives}
    engine._observers[:] = [o for o in engine._observers if not (isinstance(o, Drive) and o.name in moved)]
    world_clock_api.evict(engine)                # clock_worlds: the start velocity lands where it is read
    for d in drives:
        engine.add_step_observer(d)
        d.prime(engine)
    follow_drives(engine, drives)
    invalidate_graphs(engine)


def follow_drives(engine: Any, drives: list[Drive]) -> None:
    """Widen each liquid's escape box to every place ``drives`` can take their bodies' shapes, so liquid a driven
    container carries is not retired where it crosses the build-time box (``escape_box.widen``).
    Example: ``follow_drives(sim.engine, drives)``."""
    from alloy.api._anchor import world_points  # noqa: PLC0415
    from alloy.api._liquid import liquid_solvers  # noqa: PLC0415
    from alloy.solvers.sph import escape_box  # noqa: PLC0415

    solvers, m = liquid_solvers(engine), engine.model
    if not solvers or not int(getattr(m, "shape_count", 0) or 0) or m.shape_collision_radius is None:
        return
    shape_body, radii = m.shape_body.numpy(), m.shape_collision_radius.numpy().astype(np.float64)
    local = m.shape_transform.numpy()[:, :3].astype(np.float64)
    for d in drives:
        a = d.arrays
        row = np.full(int(m.body_count), -1, np.int64)
        row[a["body"].numpy()] = np.arange(len(a["body"]))
        s = np.flatnonzero((shape_body >= 0) & (row[np.maximum(shape_body, 0)] >= 0) & (radii < 1.0e5))
        if not len(s):
            continue
        r = row[shape_body[s]]
        bq = np.concatenate([a["p0"].numpy(), a["r0"].numpy()], axis=1)[r].astype(np.float64)
        centres = world_points(bq, local[s])                     # each shape's centre at the drive's start
        box = escape_box.drive_reach(centres, radii[s], a["pivot"].numpy()[r].astype(np.float64),
                                     a["v"].numpy()[r].astype(np.float64), a["w"].numpy()[r].astype(np.float64),
                                     d.t1 - d.t0)
        for solver in solvers:
            escape_box.widen(solver, *box)


def stop_all(sim: Any) -> None:
    """``Sim.stop_drives``: remove every drive and bring the bodies they moved to rest where they are. No solver
    integrates a kinematic body, so it keeps the last velocity its drive wrote, and contacts, MPM colliders and liquid
    walls read that as a moving wall (WP7: a stopped tool kept pushing sand at 0.8-0.9x its old speed).
    Example: ``stop_all(sim)``."""
    from alloy.coupling import world_clock_api  # noqa: PLC0415

    eng = sim.engine
    drives = [o for o in eng._observers if isinstance(o, Drive)]
    if not drives:
        return
    world_clock_api.evict(eng)                   # clock_worlds: the write lands where it is read
    eng._observers[:] = [o for o in eng._observers if not isinstance(o, Drive)]
    hold(eng, *(np.concatenate([d.arrays[k].numpy() for d in drives]) for k in ("body", "q_start", "qd_start")))
    invalidate_graphs(eng)


def hold(eng: Any, bodies: np.ndarray, q_starts: np.ndarray, qd_starts: np.ndarray) -> None:
    """Hold free / kinematic ``bodies`` at rest at their pose in ``eng.state``, in both of the engine's states: it
    swaps them every exchange, and no solver may write a kinematic body (an MPM-only scene), so a row left in the
    other state comes back every second exchange (WP9; a stopped tool's old speed and its pose one step back).
    ``q_starts`` / ``qd_starts``: the bodies' free joints' first ``joint_q`` / ``joint_qd`` rows.
    Example: ``hold(sim.engine, bodies, q_starts, qd_starts)``."""
    bq, jq = eng.state.body_q.numpy(), eng.state.joint_q.numpy()
    for st in (eng.state, getattr(eng, "_next", None)):
        if st is None:
            continue
        sbq, sbqd, sjq, sjqd = st.body_q.numpy(), st.body_qd.numpy(), st.joint_q.numpy(), st.joint_qd.numpy()
        sbq[bodies], sbqd[bodies] = bq[bodies], 0.0
        for a, d in zip(q_starts, qd_starts):
            sjq[a:a + 7], sjqd[d:d + 6] = jq[a:a + 7], 0.0
        for arr, host in ((st.body_q, sbq), (st.body_qd, sbqd), (st.joint_q, sjq), (st.joint_qd, sjqd)):
            arr.assign(host)
    rebase_history(eng, bodies, bq)


POSE_HISTORY = ("body_q_prev", "_coupling_body_q_prev_snapshot", "body_q_prev_collision_detection")
"""Solver arrays that keep each body's pose from the last step (VBD, its rods and films): a body found elsewhere
next step is read as having moved there in one step."""


def rebase_history(eng: Any, bodies: np.ndarray, body_q: np.ndarray) -> None:
    """Put ``bodies``' rows of every solver's pose history (``POSE_HISTORY``) at ``body_q`` (their pose now), so a
    teleported or stopped body starts the next step at rest. A VBD scene read a 0.25 m ``set_pose`` as 200 m/s and
    froze the world (SPEED_FAIL), and an 18.75 mm one kicked the cloth under it at 15 m/s. Only solvers on
    the engine's own model: a coupled entry's view re-reads its bodies every exchange.
    Example: ``rebase_history(sim.engine, bodies, sim.engine.state.body_q.numpy())``."""
    from alloy.api._liquid import all_solvers  # noqa: PLC0415

    for solver in all_solvers(eng):
        if getattr(solver, "model", None) is not eng.model:
            continue
        for name in POSE_HISTORY:
            arr = getattr(solver, name, None)
            if arr is None or not hasattr(arr, "numpy") or arr.shape[0] != body_q.shape[0]:
                continue
            host = arr.numpy()
            host[bodies] = body_q[bodies]
            arr.assign(host)


def free_joint(sim: Any, name: str) -> int:
    """The template free joint of a single-body object; a clear error otherwise."""
    from alloy.api.errors import SimulationError  # noqa: PLC0415

    j = sim.index.free_joint.get(name)
    if j is None:
        raise SimulationError(f"{name!r} is not a free or kinematic body", "drive / set_pose objects added with "
                              "kinematic=True (or free dynamic bodies for set_pose); static objects never move",
                              field=name)
    return j


def parts_of(sim: Any, name: str) -> list[str]:
    """``name`` itself when it is a body (a kinematic fixture is one, ``alloy.api.fixtures``), else the parts of the
    fixture called ``name``."""
    if name in sim.index.bodies:
        return [name]
    items = getattr(sim.scene, "items", {}) if sim.scene is not None else {}
    parts = [n for n, it in items.items() if it.get("fixture") == name]
    if parts:
        return parts
    sim.index.body(name)
    return [name]


def fixture_centre(sim: Any, name: str, parts: list[str]) -> np.ndarray | None:
    """``sim.drive``'s default pivot for a fixture, (worlds, 3): in each world, the mean of its parts' centres there
    (each part's own body position, or for a kinematic fixture, one body, each part's shape centre on it); None for
    any other object (it turns about its own ``.pos``). World 0's centre for every world swung the fixtures of the
    other worlds round world 0's when they started apart. Example: ``fixture_centre(sim, "jug", ["jug"])``."""
    if len(parts) > 1:
        st = sim.state(parts)
        return np.mean([st[f"{p}.pos"] for p in parts], axis=0)
    from alloy.api.fixtures import kinematic_groups  # noqa: PLC0415

    groups = kinematic_groups(getattr(sim.scene, "items", {}) if sim.scene is not None else {})
    members = groups.get(name) or next((p for p in groups.values() if p[0] == name), None)   # "<name>_floor" too
    if not members:
        return None
    m, body = sim.engine.model, sim.index.body(members[0])
    labels, shape_body = [str(x) for x in m.shape_label], m.shape_body.numpy()
    local = m.shape_transform.numpy()[[s for s, lab in enumerate(labels) if lab in members and shape_body[s] == body]]
    rows = sim.engine.state.body_q.numpy()[sim.index.per_world(body, "body")]
    return np.stack([_shape_centre(bq, local) for bq in rows])         # world by world, as world 0 always was


def _shape_centre(bq: np.ndarray, local: np.ndarray) -> np.ndarray:
    """The mean world centre of shapes at ``local`` (rows of shape transforms) on a body at pose ``bq``."""
    x, y, z, w = (float(c) for c in bq[3:7])
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return (bq[:3] + local[:, :3] @ rot.T).mean(axis=0)


def invalidate_graphs(engine: Any) -> None:
    """A captured CUDA graph does not see an observer added or removed later: drop the graphs (recaptured next step)."""
    graphs = getattr(engine, "_graphs", None)
    if graphs is not None:
        graphs.invalidate()
        engine.exchange_nodes.clear()


def release(sim: Any, name: str, world: int | None) -> None:
    """Before ``set_pose(name)``: remove the drives of its body, which write its pose every step and so would undo the
    teleport at the next one. A drive that has ended only holds its last pose, which a kinematic body does on its own,
    so it goes; a running one stops when every world is teleported, and a one-world teleport under it raises.
    Example: ``release(sim, "tray", None)``."""
    from alloy.api.errors import SimulationError  # noqa: PLC0415

    eng = sim.engine
    bodies = set(sim.index.per_world(sim.index.body(name), "body").tolist())
    mine = [o for o in eng._observers if isinstance(o, Drive) and bodies & set(o.arrays["body"].numpy().tolist())]
    if not mine:
        return
    if world is not None and any(d.t1 > float(sim.t) + 1e-9 for d in mine):
        raise SimulationError(f"{name!r} is being driven (sim.drive) in every world, and its drive writes its pose "
                              f"every step: set_pose(world={world}) would be undone at the next step",
                              "stop the drive first (sim.stop_drives()), or teleport every world (world=None), "
                              "which stops its drive", field=name)
    eng._observers[:] = [o for o in eng._observers if not any(o is d for d in mine)]
    invalidate_graphs(eng)
    log.info("set_pose(%r): its drive removed (a drive writes the pose every step)", name)


def set_pose(sim: Any, name: str, pos: Any, quat_wxyz: Any, world: int | None) -> None:
    """Write a free / kinematic body's joint coordinates and body pose now, velocity zero, in both of the engine's
    states (``hold``)."""
    eng = sim.engine
    idx, st, m = sim.index, eng.state, eng.model
    worlds = list(range(sim.worlds)) if world is None else [int(world)]
    bodies = idx.per_world(idx.body(name), "body")[worlds]
    joints = idx.per_world(free_joint(sim, name), "joint")[worlds]
    jq, bq = st.joint_q.numpy(), st.body_q.numpy()
    qs, ds = m.joint_q_start.numpy()[joints], m.joint_qd_start.numpy()[joints]
    p = None if pos is None else np.broadcast_to(np.asarray(pos, np.float32), (len(worlds), 3))
    q = None if quat_wxyz is None else np.broadcast_to(np.asarray(quat_wxyz, np.float32)[..., [1, 2, 3, 0]],
                                                       (len(worlds), 4))
    p = _origins(sim, name, bq[bodies], p, q)
    for k, (b, a) in enumerate(zip(bodies, qs)):
        if p is not None:
            jq[a:a + 3] = p[k]
            bq[b, :3] = p[k]
        if q is not None:
            jq[a + 3:a + 7] = q[k]
            bq[b, 3:7] = q[k]
    st.joint_q.assign(jq)
    st.body_q.assign(bq)
    hold(eng, bodies, qs, ds)                        # velocity 0, and the other state too (WP7: a hold drifted)


def _origins(sim: Any, name: str, bq: np.ndarray, p: np.ndarray | None, q: np.ndarray | None) -> np.ndarray | None:
    """The body origins for ``set_pose``: ``p`` itself, or for an object placed by its anchor the origins that put
    the anchor at ``p`` (its current anchor point when only ``q`` is given, so it turns about it)."""
    c = sim._io.anchors.get(name)
    if c is None or (p is None and q is None):
        return p
    from alloy.api import _anchor  # noqa: PLC0415

    point = _anchor.world_points(bq, c) if p is None else p
    quat = bq[:, 3:7] if q is None else q
    return _anchor.body_origins(point, quat, c).astype(np.float32)


def rebase_energy(sim: Any, world: int | None) -> None:
    """A teleport changes a world's energy on purpose: take the health check's energy reference (the ENERGY_JUMP
    baseline) from the state now, for ``world`` (all if None). Example: ``rebase_energy(sim, None)``."""
    health = sim.engine.health
    if health is None or not hasattr(health, "reset_reference"):
        return
    mask = np.zeros(sim.worlds, np.int32)
    mask[slice(None) if world is None else int(world)] = 1
    health.reset_reference(sim.engine.state, wp.array(mask, dtype=wp.int32, device=sim.engine.model.device))
