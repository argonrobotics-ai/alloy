# SPDX-License-Identifier: Apache-2.0
"""Rest drift (stability): how far each dynamic object moves when nothing pushes it, from the run's pose samples.

Per object and world: the drop while settling (it started above its support), the total fall, the tilt, the drift
over the second half of the run (sliding or creeping) and the speed at the end. Findings name the worst world, and a
fix that is a pose carries the settled pose as a ``suggestion``. With ``expect_rest=False`` (scenes that move by
design: drops, pours) the gross-motion findings are info only.

Example::

    >>> found, metrics = motion_findings(res, built, index)      # doctest: +SKIP
    >>> found[0].message
    'mug dropped 4.2 mm while settling (worst world 0): it starts above its support'
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.check.advice import mm
from alloy.check.objects import ObjectIndex
from alloy.check.run import RunResult
from alloy.check.report import Finding

log = logging.getLogger("alloy.check")

SETTLE_INFO_M = 2e-4
SETTLE_WARN_M = 2e-3
FALL_M = 5e-2
TIP_WARN_DEG = 10.0
TIP_ERROR_DEG = 45.0
DRIFT_WARN_M = 5e-4       # over the second half of the run
DRIFT_ERROR_M = 1e-2
SPEED_WARN_MPS = 5e-2
MIN_JUDGE_S = 0.5         # late drift and end speed need at least this much simulated time


@dataclass
class Motion:
    """[W] per-world worst values of one object (bodies: over its bodies; particles: their centroid).

    Example::

        >>> z = np.zeros(2)
        >>> Motion(z, z, z, z, np.zeros((2, 3)), z, z).late.shape
        (2,)
    """

    settle: np.ndarray        # |x(settle) - x(0)|
    drop: np.ndarray          # z(0) - z(end), > 0 = fell
    disp: np.ndarray          # |x(end) - x(0)|
    late: np.ndarray          # |x(end) - x(mid)|
    late_dir: np.ndarray      # [W, 3] unit direction of the late drift
    tilt_deg: np.ndarray      # rotation between start and end
    speed_end: np.ndarray     # linear speed at the end
    flexible: bool = False    # several bodies per world (a cable, a chain, an articulated object): centroid values


def motion_findings(res: RunResult, built: Any, index: ObjectIndex,
                    expect_rest: bool = True) -> tuple[list[Finding], dict[str, Any]]:
    """Rest-drift findings and per-object metrics from ``res.samples`` (start, settle, mid, end).

    Example::

        >>> motion_findings(RunResult(), built, index)      # a run with no samples
        ([], {})
    """
    if len(res.samples) < 2:
        return [], {}
    found: list[Finding] = []
    metrics: dict[str, Any] = {}
    s = res.samples
    mid = s[len(s) // 2] if len(s) > 2 else s[0]
    for name in index.dynamic_names():
        mo = object_motion(name, built, index, s[0], s[1], mid, s[-1])
        if mo is None:
            continue
        metrics[name] = {"settle_m": float(mo.settle.max()), "drop_m": float(mo.drop.max()),
                         "drift_m": float(mo.late.max()), "tilt_deg": float(mo.tilt_deg.max()),
                         "speed_end_mps": float(mo.speed_end.max())}
        found += _object_findings(name, mo, (s[-1].t, s[-1].t - mid.t), expect_rest,
                                  _settled_pose(name, built, index, s[0], s[-1]))
    log.info("check rest drift: %d objects, %d findings", len(metrics), len(found))
    return found, metrics


def object_motion(name: str, built: Any, index: ObjectIndex, s0: Any, s1: Any, smid: Any, send: Any) -> Motion | None:
    """``Motion`` of ``name`` from four samples; None when it owns neither bodies nor particles.

    Example::

        >>> s0, s1, smid, send = res.samples  # doctest: +SKIP
        >>> object_motion("mug", built, index, s0, s1, smid, send).disp.shape
        (4,)
    """
    w = index.world_count
    bodies = index.bodies_of(name)
    if bodies.size and bodies.size > w:          # a cable or chain sags as it settles: judge its centroid
        bw = built.model.body_world.numpy()[bodies]
        cent = lambda smp: _centroids(smp.body_q[bodies, :3], bw, w)            # noqa: E731
        vel = lambda smp: _centroids(smp.body_qd[bodies, :3], bw, w)            # noqa: E731
        return dataclasses.replace(_motion_of(cent, np.arange(w), w, s0, s1, smid, send, speed=vel), flexible=True)
    if bodies.size:
        bw = built.model.body_world.numpy()[bodies]
        return _motion_of(lambda smp: smp.body_q[bodies, :3], bw, w, s0, s1, smid, send,
                          quats=lambda smp: smp.body_q[bodies, 3:7], speed=lambda smp: smp.body_qd[bodies, :3])
    parts = np.flatnonzero(index.particle_obj == index.names.index(name))
    if not parts.size:
        return None
    pw = built.model.particle_world.numpy()[parts]
    cent = lambda smp: _centroids(smp.particle_q[parts], pw, w)   # noqa: E731
    return _motion_of(cent, np.arange(w), w, s0, s1, smid, send)


def _motion_of(pos, world: np.ndarray, w: int, s0, s1, smid, send, quats=None, speed=None) -> Motion:
    x0, x1, xm, xe = pos(s0), pos(s1), pos(smid), pos(send)
    late_v = xe - xm
    late = np.linalg.norm(late_v, axis=1)
    settle, drop, disp = _wmax(np.linalg.norm(x1 - x0, axis=1), world, w), \
        _wmax(x0[:, 2] - xe[:, 2], world, w), _wmax(np.linalg.norm(xe - x0, axis=1), world, w)
    late_w, late_dir = _wmax(late, world, w), np.zeros((w, 3))
    for k in range(w):
        sel = np.flatnonzero(world == k)
        if sel.size:
            j = sel[np.argmax(late[sel])]
            late_dir[k] = late_v[j] / max(late[j], 1e-12)
    tilt = _wmax(_angle_deg(quats(s0), quats(send)), world, w) if quats is not None else np.zeros(w)
    spd = _wmax(np.linalg.norm(speed(send), axis=1), world, w) if speed is not None else \
        late_w / max(send.t - smid.t, 1e-9)
    return Motion(settle, drop, disp, late_w, late_dir, tilt, spd)


def _object_findings(name: str, mo: Motion, times: tuple[float, float], expect_rest: bool,
                     settled: dict | None) -> list[Finding]:
    """Gross motion first (fall, tip); settling only when neither; then late drift and end speed, which a run
    shorter than ``MIN_JUDGE_S`` (cut by the wall budget) cannot judge, so they are info there."""
    total_s, late_s = times
    gross = "error" if expect_rest else "info"
    out = [f for f in (_fall(name, mo, gross, settled), _tip(name, mo, gross, settled)) if f is not None]
    if not out:
        out += [f for f in (_settle(name, mo, settled, expect_rest and not mo.flexible),) if f is not None]
    judge = expect_rest and total_s >= MIN_JUDGE_S
    late = [f for f in (_drift(name, mo, late_s, judge), _moving(name, mo, judge)) if f is not None]
    if expect_rest and not judge:
        late = [dataclasses.replace(f, message=f"{f.message} (the run lasted only {total_s:.2g} s: too short to "
                                    f"judge; check again with a larger budget_s)") for f in late]
    return out + late


def _fall(name: str, mo: Motion, level: str, settled: dict | None) -> Finding | None:
    w = int(np.argmax(mo.drop))
    if mo.drop[w] <= FALL_M:
        return None
    rest = mo.speed_end[w] < SPEED_WARN_MPS
    fix = (f"place {name} on its support: it came to rest {mm(mo.drop[w])} lower; use the settled pose"
           if rest and settled else f"put a support under {name} (a table or the floor) or start it lower")
    return Finding("rest_drift", level, f"{name} fell {mm(mo.drop[w])} (worst world {w}): nothing holds it at its "
                   f"start pose", fix, objects=(name,), value=float(mo.drop[w]), limit=FALL_M, unit="m",
                   suggestion=settled if rest else None)


def _tip(name: str, mo: Motion, level: str, settled: dict | None) -> Finding | None:
    w = int(np.argmax(mo.tilt_deg))
    a = float(mo.tilt_deg[w])
    if a <= TIP_WARN_DEG:
        return None
    lvl = level if a > TIP_ERROR_DEG else "warn" if level == "error" else "info"
    return Finding("rest_drift", lvl, f"{name} tipped {a:.0f} deg (worst world {w}): it is not stable as placed",
                   f"place {name} upright on a flat support, or start it in its settled orientation"
                   + (" (suggested pose)" if settled else ""), objects=(name,), value=a, limit=TIP_WARN_DEG,
                   unit="deg", suggestion=settled)


def _settle(name: str, mo: Motion, settled: dict | None, expect_rest: bool = True) -> Finding | None:
    w = int(np.argmax(mo.settle))
    d = float(mo.settle[w])
    if d <= SETTLE_INFO_M:
        return None
    level = "warn" if d > SETTLE_WARN_M and expect_rest else "info"
    dropped = float(mo.drop[w]) > 0.5 * d
    why = "it starts above its support" if dropped else "it was pushed while settling (neighbour, overlap)"
    if dropped:
        fix = f"lower {name} by {mm(math.floor(d / 1e-4) * 1e-4)}"
        fix += " (or use the suggested settled pose)" if settled else ""
    else:
        fix = f"give {name} room or check its overlaps"
    return Finding("rest_drift", level, f"{name} moved {mm(d)} while settling (worst world {w}): {why}", fix,
                   objects=(name,), value=d, limit=SETTLE_WARN_M, unit="m", suggestion=settled if dropped else None)


def _drift(name: str, mo: Motion, late_s: float, expect_rest: bool) -> Finding | None:
    w = int(np.argmax(mo.late))
    d = float(mo.late[w])
    if d <= DRIFT_WARN_M:
        return None
    level = ("error" if d > DRIFT_ERROR_M else "warn") if expect_rest else "info"
    vertical = abs(float(mo.late_dir[w, 2])) > 0.7
    how = "sinks" if vertical and mo.late_dir[w, 2] < 0 else "rises" if vertical else "slides"
    fix = (f"{name} {how} into or off its support: check the penetration findings and the contact_stiffness of "
           f"its material" if vertical else
           f"{name} slides: level its support, raise mu_static of its material or of the pair card with its "
           f"support, or move it away from whatever pushes it")
    return Finding("rest_drift", level, f"{name} still {how} {mm(d)} over the last {late_s:.2g} s (worst world "
                   f"{w}); resting objects should move < {mm(DRIFT_WARN_M)}", fix, objects=(name,), value=d,
                   limit=DRIFT_WARN_M, unit="m")


def _moving(name: str, mo: Motion, expect_rest: bool) -> Finding | None:
    w = int(np.argmax(mo.speed_end))
    v = float(mo.speed_end[w])
    if v <= SPEED_WARN_MPS:
        return None
    return Finding("rest_drift", "warn" if expect_rest else "info",
                   f"{name} is still moving at {v:.3g} m/s at the end (worst world {w})",
                   f"if {name} should rest, give it a flat, level support (round objects roll); run longer "
                   f"(seconds=...) if it is still settling", objects=(name,), value=v, limit=SPEED_WARN_MPS,
                   unit="m/s")


def _settled_pose(name: str, built: Any, index: ObjectIndex, s0: Any, send: Any) -> dict | None:
    """The card pose moved by world 0's rigid motion of a one-body object (None otherwise)."""
    bodies = index.bodies_of(name)
    pose = index.poses.get(name)
    if bodies.size != index.world_count or pose is None or name not in index.placement:
        return None
    b = int(bodies[0])
    q0, qe = s0.body_q[b], send.body_q[b]
    rel = _qmul(qe[3:7], _qconj(q0[3:7]))
    x = np.asarray(pose[:3]) + (qe[:3] - q0[:3])
    q = _qmul(rel, np.asarray(pose[3:7], np.float64))
    q = q / np.linalg.norm(q)
    new = [round(float(v), 6) for v in (*x, *q)]
    return {"object": name, "field": f"objects.{index.placement[name]}.pose", "pose": new}


def _centroids(x: np.ndarray, world: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros((w, 3))
    for k in range(w):
        sel = world == k
        if sel.any():
            out[k] = x[sel].mean(axis=0)
    return out


def _wmax(v: np.ndarray, world: np.ndarray, w: int) -> np.ndarray:
    out = np.full(w, -np.inf)
    np.maximum.at(out, world, v)
    return np.where(np.isfinite(out), out, 0.0)


def _angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    dot = np.abs(np.einsum("ij,ij->i", q0, q1)) / (np.linalg.norm(q0, axis=1) * np.linalg.norm(q1, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


__all__ = ["Motion", "motion_findings", "object_motion"]
