# SPDX-License-Identifier: Apache-2.0
"""Move kinematic objects: teleport (``set_pose``) or drive with a constant twist for a time (``drive``).

A kinematic object (``Scene.add_box(..., kinematic=True)`` and friends) is a body the solvers do not integrate; it
moves only as told, and everything that touches it (rigid bodies, liquids, cloth) feels its walls and their speed.
``drive`` runs a small kernel after every physics step (a step observer, captured in CUDA graphs) that writes the
body's pose and its velocity, so contacts see a moving wall rather than a jump each control period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.api")


@wp.kernel(enable_backward=False)
def _drive_kernel(clock: wp.array(dtype=wp.float64), dt: float, q_start: wp.array(dtype=wp.int32),
                  qd_start: wp.array(dtype=wp.int32), body: wp.array(dtype=wp.int32), p0: wp.array(dtype=wp.vec3),
                  r0: wp.array(dtype=wp.quat), pivot: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3),
                  v: wp.vec3, w: wp.vec3, t0: float, t1: float, joint_q: wp.array(dtype=float),
                  joint_qd: wp.array(dtype=float), body_q: wp.array(dtype=wp.transform),
                  body_qd: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    now = float(clock[0]) + dt
    tau = wp.clamp(now - t0, 0.0, t1 - t0)
    rate = wp.length(w)
    rot = wp.quat_identity()
    if rate > 0.0:
        rot = wp.quat_from_axis_angle(w / rate, rate * tau)
    piv = pivot[i] + v * tau
    pos = piv + wp.quat_rotate(rot, p0[i] - pivot[i])
    q = rot * r0[i]
    lin = wp.vec3(0.0)
    ang = wp.vec3(0.0)
    if now >= t0 and now < t1:
        ang = w
        lin = v + wp.cross(w, pos + wp.quat_rotate(q, com[i]) - piv)       # COM velocity (Newton's convention)
    qs = q_start[i]
    ds = qd_start[i]
    for k in range(3):
        joint_q[qs + k] = pos[k]
        joint_qd[ds + k] = lin[k]
        joint_qd[ds + 3 + k] = ang[k]
    joint_q[qs + 3] = q[0]
    joint_q[qs + 4] = q[1]
    joint_q[qs + 5] = q[2]
    joint_q[qs + 6] = q[3]
    b = body[i]
    body_q[b] = wp.transform(pos, q)
    body_qd[b] = wp.spatial_vector(lin, ang)


@wp.kernel(enable_backward=False)
def _tick(clock: wp.array(dtype=wp.float64), dt: float):
    clock[0] = clock[0] + wp.float64(dt)


@dataclass
class Drive:
    """One constant-twist motion of one kinematic object in every world (a step observer)."""

    arrays: dict[str, Any]
    v: tuple[float, float, float]
    w: tuple[float, float, float]
    t0: float
    t1: float
    name: str = ""                       # the object it moves: a new drive of the same object replaces this one

    def __call__(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        a = self.arrays
        n = len(a["body"])
        wp.launch(_drive_kernel, dim=n, inputs=[a["clock"], float(dt), a["q_start"], a["qd_start"], a["body"],
                                                a["p0"], a["r0"], a["pivot"], a["com"], wp.vec3(*self.v),
                                                wp.vec3(*self.w), self.t0, self.t1, state_out.joint_q,
                                                state_out.joint_qd, state_out.body_q, state_out.body_qd],
                  device=state_out.joint_q.device)
        wp.launch(_tick, dim=1, inputs=[a["clock"], float(dt)], device=state_out.joint_q.device)

    def save_period(self) -> None:
        """Keep the clock at a control period's start (``IntegratedSimulation.step_period``, outside graphs)."""
        wp.copy(self.arrays["saved"], self.arrays["clock"])

    def restore_period(self) -> None:
        """Back to the period start's clock when the period re-runs (a coupled scene's rollback restores the state and
        the simulation's clock, not this one); else the object jumps ahead by the discarded attempts' time."""
        wp.copy(self.arrays["clock"], self.arrays["saved"])


def make_drive(sim: Any, name: str, velocity: Any, angular_velocity: Any, pivot: Any, duration: float) -> Drive:
    """A Drive of ``name`` starting now, from its current pose in each world."""
    idx, m, st = sim.index, sim.engine.model, sim.engine.state
    bodies = idx.per_world(idx.body(name), "body")
    joints = idx.per_world(free_joint(sim, name), "joint")
    bq = st.body_q.numpy()[bodies]
    piv = sim._io.point(name, bq) if pivot is None else np.broadcast_to(np.asarray(pivot, np.float32),
                                                                         (len(bodies), 3))
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
    }
    t0 = float(sim.t)
    return Drive(arrays, tuple(map(float, velocity)), tuple(map(float, angular_velocity)), t0, t0 + float(duration),
                 name)


def replace_drives(engine: Any, drives: list[Drive]) -> None:
    """Add ``drives`` as step observers, removing each object's earlier drive first: the newest drive of an object
    writes its pose last in every step anyway, so the result is the same, but drives no longer pile up (one drive per
    control period for 4 s was 1,600 observers and 3,200 kernel launches per step). Example: see ``Sim.drive``."""
    moved = {d.name for d in drives}
    engine._observers[:] = [o for o in engine._observers if not (isinstance(o, Drive) and o.name in moved)]
    for d in drives:
        engine.add_step_observer(d)
    invalidate_graphs(engine)


def stop_all(sim: Any) -> None:
    """``Sim.stop_drives``: remove every drive and bring the bodies they moved to rest where they are. No solver
    integrates a kinematic body, so it keeps the last velocity its drive wrote, and contacts, MPM colliders and liquid
    walls read that as a moving wall (WP7: a stopped tool kept pushing sand at 0.8-0.9x its old speed).
    Example: ``stop_all(sim)``."""
    eng = sim.engine
    drives = [o for o in eng._observers if isinstance(o, Drive)]
    if not drives:
        return
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
    """``sim.drive``'s default pivot for a fixture: the mean of its parts' centres in world 0 (each part's own body
    position, or for a kinematic fixture, one body, each part's shape centre on it); None for any other object (it
    turns about its own ``.pos``). Example: ``fixture_centre(sim, "jug", ["jug"])``."""
    if len(parts) > 1:
        st = sim.state(parts, world=0)
        return np.mean([st[f"{p}.pos"] for p in parts], axis=0)
    from alloy.api.fixtures import kinematic_groups  # noqa: PLC0415

    groups = kinematic_groups(getattr(sim.scene, "items", {}) if sim.scene is not None else {})
    members = groups.get(name) or next((p for p in groups.values() if p[0] == name), None)   # "<name>_floor" too
    if not members:
        return None
    m, body = sim.engine.model, sim.index.body(members[0])
    labels, shape_body = [str(x) for x in m.shape_label], m.shape_body.numpy()
    local = m.shape_transform.numpy()[[s for s, lab in enumerate(labels) if lab in members and shape_body[s] == body]]
    bq = sim.engine.state.body_q.numpy()[body]
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
