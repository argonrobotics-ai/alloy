# SPDX-License-Identifier: Apache-2.0
"""The scripted flush-floor policy (:mod:`alloy.tasks.rinse_flush_floor`): Gym-Anything's demo A6 reduced to a script,
plus the chase that moves the food scraps (Alloy's addition) onto the drain.

1. ``passes`` inward spirals of the jets' aim over the soil, the jets leaning 30° toward the drain. The first loosens
   the soil, the next rinses the dirty water down the drain, its turns half a pitch in so they run between the
   first's: beside the can, where the jets lean less, cells midway between one pass's turns stayed soiled.
2. Each scrap in turn: the jets aim at its centre from 0.15 m, leaning 45° toward the drain, until it lies on the
   drain. Steeper jets pin a scrap: at 30° each ray presses it down twice as hard as it pushes it on. Lower is
   faster: the 6° half-angle cone spreads about 2 cm from 0.21 m along its axis, about the scrap's size (on a 0.13 m
   run, 4.0 s from 0.15 m, 5.9 s from 0.20 m, 10.2 s from 0.30 m). This is
   closed-loop on the scrap's position, per world, read exactly from the engine: privileged state, a stand-in for
   perception (Gym-Anything ships no scripted policy and has no scraps).
3. The valve shuts and docks in its holster.

Between these the valve moves with its jets off, at most ``TRANSIT`` m/s (:func:`transit`): out of the holster, back
to the spiral's start, to each scrap, and home. Gym-Anything's garbage can stands in the bay: near it the jets lean
less, so neither the valve nor its jets meet it, and where no lean clears it the spiral passes with the jets off
(:func:`aim_pose`, :func:`sweep`).

The valve is a kinematic body; :func:`place` writes its pose through its free joint (MuJoCo keeps joint coordinates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.tasks.paint_supply import bodies

CONTROL_DT = 0.02               # s
TRANSIT = 1.0                   # m/s: the valve's speed between sweeps, jets off (a hand carrying it)
CLEAR = 0.035                   # m: the valve (27 mm radius) keeps this far off the standing can
JET_CLEAR = 0.005               # m: the jets' axis keeps this far off it
JUMP = 0.05                     # m: a sweep step longer than this is carried with the jets off
NOZZLE = 0.104                  # m: the nozzle sits this far along the valve's axis from its body origin (theirs)


@dataclass(frozen=True)
class Flush:
    """``passes`` inward spirals of the jets' aim point, each from ``r0`` (default: past the soil's farthest corner)
    to ``r_end`` round the drain, ``pitch`` per turn at ``speed``; the valve stays open between them. The valve's head
    is ``height`` above the floor and its jets lean ``lean`` toward the drain. Then the chase (``chase_*``): a scrap is
    done within ``chase_done`` of the drain's centre, or after ``chase_cap`` s."""

    height: float = 0.55                     # m
    lean: float = math.radians(30.0)
    speed: float = 0.20                      # m/s along the spiral
    pitch: float = 0.10                      # m per turn
    r0: float | None = None
    r_end: float = 0.02                      # m
    passes: int = 2
    chase_height: float = 0.15               # m
    chase_lean: float = math.radians(45.0)
    chase_done: float = 0.03                 # m from the drain's centre
    chase_cap: float = 30.0                  # s per scrap
    settle: float = 30.0                     # s: the settle's cap (Gym-Anything's settle_s; it stops when quiet)


def spiral(bay: Any, fl: Flush, shift: float = 0.0) -> np.ndarray:
    """One pass's aim points [P, 2] (bay frame), one per control period; ``shift`` [m] moves the turns inward (the
    second pass runs half a pitch in, between the first's turns)."""
    (x0, x1), (y0, y1) = bay.region
    dx, dy = bay.drain
    r0 = fl.r0 if fl.r0 is not None else max(math.hypot(x - dx, y - dy) for x in (x0, x1) for y in (y0, y1)) + 0.05
    r0 -= shift
    b, phi, r, out = fl.pitch / (2.0 * math.pi), 0.0, r0, []
    while r > fl.r_end:
        out.append((dx + r * math.cos(phi), dy + r * math.sin(phi)))
        phi += fl.speed * CONTROL_DT / math.hypot(r, b)
        r = r0 - b * phi
    return np.asarray(out)


def quat_down_to(j: np.ndarray) -> tuple[float, float, float, float]:
    """The rotation (x, y, z, w) taking the body's -z onto unit ``j``."""
    a = np.array([0.0, 0.0, -1.0])
    ax, c = np.cross(a, j), float(np.dot(a, j))
    s = float(np.linalg.norm(ax))
    if s < 1e-12:
        return (0.0, 0.0, 0.0, 1.0) if c > 0 else (1.0, 0.0, 0.0, 0.0)
    half = 0.5 * math.atan2(s, c)
    ax = ax / s * math.sin(half)
    return float(ax[0]), float(ax[1]), float(ax[2]), math.cos(half)


def _pose(aim: tuple[float, float], height: float, lean: float, toward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The nozzle's tip and the jets' axis for ``aim`` from ``height``, leaning ``lean`` along unit ``toward`` (2-D)."""
    j = np.array([toward[0] * math.sin(lean), toward[1] * math.sin(lean), -math.cos(lean)])
    return np.array([aim[0], aim[1], 0.0]) - (height / math.cos(lean)) * j, j


def _off_can(can: Any, p: np.ndarray, q: np.ndarray) -> float:
    """How far the segment p-q (2-D) passes from the can's surface."""
    c = np.asarray(can[:2])
    u = float(np.clip(np.dot(c - p, q - p) / max(float(np.dot(q - p, q - p)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p + u * (q - p) - c)) - can[2]


def _clears(can: Any, tip: np.ndarray, j: np.ndarray, aim: tuple[float, float]) -> bool:
    """The valve (its body reaches ``2 NOZZLE`` back from the tip) clears the standing can by ``CLEAR`` and its jets'
    line by ``JET_CLEAR`` (all of them lie below its top)."""
    if can is None:
        return True
    back = (tip - 2.0 * NOZZLE * j)[:2]
    return _off_can(can, back, tip[:2]) >= CLEAR and _off_can(can, tip[:2], np.asarray(aim)) >= JET_CLEAR


def aim_pose(bay: Any, aim: tuple[float, float], height: float, lean: float) -> tuple[tuple[float, ...], bool]:
    """The valve body's pose (x, y, z, qx, qy, qz, qw) whose jets' axis meets the floor at ``aim`` from ``height``:
    leaning toward the drain, ``lean`` or less where the valve or its jets would meet the standing can (down to
    straight down, in 1° steps; straight down over the drain itself); else leaning ``lean`` toward the can from its
    far side (an aim at its foot). ``False`` when nothing clears it: sweep past with the jets off."""
    can = getattr(bay, "can", None)
    to = np.array([bay.drain[0] - aim[0], bay.drain[1] - aim[1]])
    n = float(np.linalg.norm(to))
    drain = to / n if n > 1e-9 else np.zeros(2)
    tries = [(max(lean - math.radians(k), 0.0) if n > 1e-9 else 0.0, drain) for k in range(int(math.degrees(lean)) + 1)]
    if can is not None:
        at = np.array([can[0] - aim[0], can[1] - aim[1]])
        if np.linalg.norm(at) < can[2] + JET_CLEAR:
            tries = []
        else:
            tries.append((lean, at / np.linalg.norm(at)))
    for tilt, toward in tries:
        tip, j = _pose(aim, height, tilt, toward)
        if _clears(can, tip, j, aim):
            return (*(tip - NOZZLE * j), *quat_down_to(j)), True
    tip, j = _pose(aim, height, 0.0, drain)
    return (*(tip - NOZZLE * j), *quat_down_to(j)), False


def valve_pose(bay: Any, aim: tuple[float, float], height: float, lean: float) -> tuple[float, ...]:
    """:func:`aim_pose`'s pose."""
    return aim_pose(bay, aim, height, lean)[0]


def sweep(sim: Any, flow: Any, bay: Any, fl: Flush, starts: np.ndarray, edge: str, pose: np.ndarray,
          shift: float = 0.0) -> np.ndarray:
    """One pass of the spiral (turns moved ``shift`` in): the jets on along each run of aims the valve can reach,
    carried with the jets off (:func:`transit`) to the start of each run. Returns where the valve ends [W, 7]."""
    W = len(starts)
    poses = [aim_pose(bay, tuple(a), fl.height, fl.lean) for a in spiral(bay, fl, shift)]
    on = False
    for p, ok in poses:
        if not ok:
            on = False
            continue
        if on and np.linalg.norm(np.asarray(p[:3]) - pose[0, :3]) > JUMP:      # the lean changed fast: re-place it
            on = False
        if not on:
            pose = transit(sim, flow, starts, pose, np.asarray(p), edge)
            flow.set_input(edge, np.ones(W, np.float32))
            on = True
        place(sim, starts, p)
        sim.step_period()
        pose = np.broadcast_to(np.asarray(p, dtype=float), (W, 7)).copy()
    flow.set_input(edge, np.zeros(W, np.float32))
    return pose


def parked(bay: Any) -> tuple[float, ...]:
    """The valve docked head down in its holster: its body's origin at the holster's docked centre."""
    return (*bay.holster, 0.0, 0.0, 0.0, 1.0)


def q_starts(model: Any, idx: np.ndarray) -> np.ndarray:
    """Where each body's free joint starts in ``joint_q``."""
    child, start = model.joint_child.numpy(), model.joint_q_start.numpy()
    return np.array([start[int(np.flatnonzero(child == b)[0])] for b in idx], dtype=np.int64)


def place(sim: Any, starts: np.ndarray, poses) -> None:  # noqa: ANN001 - one pose or one per world
    """The kinematic valve's pose in every world (``poses``: one (7,) or (W, 7)), through its free joint."""
    from alloy._vendor import newton  # noqa: PLC0415

    q = sim.state.joint_q.numpy().copy()
    p = np.broadcast_to(np.asarray(poses, dtype=np.float64), (len(starts), 7))
    for s, pw in zip(starts, p, strict=True):
        q[s:s + 7] = pw
    sim.state.joint_q.assign(q)
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)


def transit(sim: Any, flow: Any, starts: np.ndarray, frm: np.ndarray, to: np.ndarray, edge: str) -> np.ndarray:
    """Carry the valve from poses ``frm`` to ``to`` ([W, 7] or [7]) with its jets off, at most ``TRANSIT`` m/s
    (positions straight, attitudes normalised-lerped). Returns ``to`` as [W, 7]."""
    a, b = np.broadcast_to(frm, (len(starts), 7)).astype(float), np.broadcast_to(to, (len(starts), 7)).astype(float)
    n = int(np.ceil(np.linalg.norm(b[:, :3] - a[:, :3], axis=1).max() / (TRANSIT * CONTROL_DT) - 1e-9))
    flow.set_input(edge, np.zeros(len(starts), np.float32))
    for k in range(1, n + 1):
        t = k / n
        q = (1.0 - t) * a[:, 3:] + t * np.where((a[:, 3:] * b[:, 3:]).sum(1, keepdims=True) < 0.0, -b[:, 3:], b[:, 3:])
        place(sim, starts, np.concatenate([(1.0 - t) * a[:, :3] + t * b[:, :3], q / np.linalg.norm(q, axis=1,
                                                                                                    keepdims=True)], 1))
        sim.step_period()
    return b.copy()


def chase(sim: Any, flow: Any, bay: Any, fl: Flush, starts: np.ndarray, scraps: list[np.ndarray], edge: str,
          pose: np.ndarray) -> tuple[list[float], np.ndarray]:
    """Push each scrap onto the drain in turn (every world at once, each on its own scrap), carrying the valve to it
    with the jets off first; a world whose scrap is done shuts its valve. ``pose`` is where the valve starts [W, 7].
    Returns the seconds each scrap took (its slowest world) and where the valve ends."""
    W, drain, took = len(starts), np.asarray(bay.drain), []
    for idx in scraps:
        p = sim.state.body_q.numpy()[idx, :2]
        pose = transit(sim, flow, starts, pose, np.array([valve_pose(bay, tuple(p[w]), fl.chase_height,
                                                                     fl.chase_lean) for w in range(W)]), edge)
        k = 0
        for k in range(int(round(fl.chase_cap / CONTROL_DT))):
            p = sim.state.body_q.numpy()[idx, :2]
            todo = np.linalg.norm(p - drain, axis=1) > fl.chase_done
            if not todo.any():
                break
            pose = np.array([valve_pose(bay, tuple(p[w]), fl.chase_height, fl.chase_lean) for w in range(W)])
            place(sim, starts, pose)
            flow.set_input(edge, todo.astype(np.float32))
            sim.step_period()
        took.append(round(k * CONTROL_DT, 2))
    flow.set_input(edge, np.zeros(W, np.float32))
    return took, pose


__all__ = ["CLEAR", "CONTROL_DT", "JET_CLEAR", "JUMP", "NOZZLE", "TRANSIT", "Flush", "aim_pose", "bodies", "chase",
           "parked", "place", "q_starts", "quat_down_to", "spiral", "sweep", "transit", "valve_pose"]
