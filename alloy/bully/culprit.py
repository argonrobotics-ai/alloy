# SPDX-License-Identifier: Apache-2.0
"""Which body a bully finding is about (``alloy.bully.audit``).

The episode monitors keep one worst value per world, not the body behind it. ``BodyProbe`` is a host-side step
observer for a one-world replay of the finding's trace up to its first tick: per body it keeps the peak speed, the
largest one-step jump, the largest rise of kinetic + potential energy, the largest distance from the start, the first
non-finite step, and the deepest contact (with both shapes' bodies). ``culprit`` picks the body that matches the
finding's cause. Host reads every step make the probe slow; it runs on one world, for a few findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

SPEED = frozenset({"max_speed", "impossible_speed", "impossible_speed_d12", "contact_free_accel", "SPEED_FAIL"})
JUMP = frozenset({"teleport"})
PEN = frozenset({"max_pen", "pen_s1_ratio", "PEN_FAIL", "PEN_WARN", "PAIR_PEN_FAIL", "PAIR_PEN_WARN"})
NONFINITE = frozenset({"nonfinite", "NONFINITE_Q", "NONFINITE_QD", "NONFINITE_PARTICLE"})
ENERGY = frozenset({"e_gain", "e_gain_cum", "ENERGY_JUMP"})
DISTANCE = frozenset({"nontarget_disp", "FAR_FROM_ORIGIN"})
SCENE_LEVEL = frozenset({"capacity_overflow", "CONTACT_OVERFLOW", "ROW_OVERFLOW"})
BODY_CAUSES = SPEED | JUMP | PEN | NONFINITE | ENERGY | DISTANCE


@dataclass(frozen=True)
class Culprit:
    body: int | None  # template body index (None: scene-level cause, or nothing to point at)
    label: str | None  # the placed object's name or the body's label
    detail: str  # how it was picked, e.g. "fastest body: 23.1 m/s"


class BodyProbe:
    """Host step observer for a one-world simulation (module doc)."""

    def __init__(self, model: newton.Model, state: newton.State) -> None:
        nb = int(model.body_count)
        self.model = model
        self.mass = model.body_mass.numpy()[:nb].astype(np.float64)
        self.g = np.asarray(model.gravity.numpy(), dtype=np.float64).reshape(-1, 3)[0]
        self.shape_body = model.shape_body.numpy()
        q0 = state.body_q.numpy().astype(np.float64)
        self.x0 = q0[:, :3].copy()
        self.e0 = self._energy(q0, state.body_qd.numpy().astype(np.float64))
        self.speed, self.jump, self.gain, self.dist, self.work = (np.zeros(nb) for _ in range(5))
        self.nonfinite_step = np.full(nb, -1, dtype=np.int64)
        self.pen, self.pen_bodies, self.pen_shapes = 0.0, (-1, -1), (-1, -1)
        self.steps = 0

    def _energy(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        return 0.5 * self.mass * np.sum(qd[:, :3] ** 2, axis=1) - self.mass * (q[:, :3] @ self.g)

    def observer(self, s0: newton.State, s1: newton.State, control: Any, contacts: Any, dt: float) -> None:
        if self.mass.size:
            qd0, qd1 = s0.body_qd.numpy().astype(np.float64), s1.body_qd.numpy().astype(np.float64)
            f = getattr(s0, "body_f", None)
            if f is not None:  # work of applied wrenches (audit pushes), as the bench energy ledger books it
                self.work += dt * np.nan_to_num(np.sum(f.numpy().astype(np.float64) * 0.5 * (qd0 + qd1), axis=1))
            self._bodies(s0.body_q.numpy().astype(np.float64), s1.body_q.numpy().astype(np.float64), qd1)
        if contacts is not None and getattr(contacts, "rigid_contact_max", 0):
            self._contacts(contacts, s1.body_q.numpy().astype(np.float64))
        self.steps += 1

    def _bodies(self, q0: np.ndarray, q1: np.ndarray, qd1: np.ndarray) -> None:
        bad = ~np.isfinite(q1).all(axis=1) | ~np.isfinite(qd1).all(axis=1)
        self.nonfinite_step[bad & (self.nonfinite_step < 0)] = self.steps
        with np.errstate(invalid="ignore", over="ignore"):
            pairs = ((self.speed, np.linalg.norm(qd1[:, :3], axis=1)),
                     (self.jump, np.linalg.norm(q1[:, :3] - q0[:, :3], axis=1)),
                     (self.gain, self._energy(q1, qd1) - self.e0 - self.work),
                     (self.dist, np.linalg.norm(q1[:, :3] - self.x0, axis=1)))
            for peak, now in pairs:
                np.fmax(peak, np.where(np.isfinite(now), now, 0.0), out=peak)

    def _contacts(self, c: Any, body_q: np.ndarray) -> None:
        n = int(c.rigid_contact_count.numpy()[0])
        if n == 0:
            return
        s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
        ok = (s0 >= 0) & (s1 >= 0)
        if not ok.any():
            return
        b0, b1 = self.shape_body[np.maximum(s0, 0)], self.shape_body[np.maximum(s1, 0)]
        x0 = _to_world(body_q, b0, c.rigid_contact_point0.numpy()[:n])
        x1 = _to_world(body_q, b1, c.rigid_contact_point1.numpy()[:n])
        margin = c.rigid_contact_margin0.numpy()[:n] + c.rigid_contact_margin1.numpy()[:n]
        pen = -(np.sum(c.rigid_contact_normal.numpy()[:n] * (x1 - x0), axis=1) - margin)
        pen[~ok] = -np.inf
        k = int(np.argmax(pen))
        if pen[k] > self.pen:
            self.pen, self.pen_bodies, self.pen_shapes = float(pen[k]), (int(b0[k]), int(b1[k])), (int(s0[k]), int(s1[k]))


def culprit(probe: BodyProbe, cause: str, names: Any = None) -> Culprit:
    """The body behind ``cause`` (a monitor row or health flag name) from a probe run; ``names(kind, index)`` names
    a "body" or "shape" (None: unknown)."""
    names = names or (lambda kind, i: None)

    def name(b: int) -> str | None:
        return names("body", b)
    if cause in SCENE_LEVEL:
        return Culprit(None, None, "scene-level: a contact or constraint buffer overflowed (raise the capacities)")
    if cause in PEN:
        (a, b), (sa, sb) = probe.pen_bodies, probe.pen_shapes
        if probe.pen <= 0.0:
            return Culprit(None, None, "no penetrating contact seen in the contact buffer")
        first = a >= 0 and (b < 0 or probe.mass[a] <= probe.mass[b])  # the lighter, moving one
        body, other, other_shape = (a, b, sb) if first else (b, a, sa)
        against = (name(other) if other >= 0 else names("shape", other_shape)) or "a static shape"
        return Culprit(body, name(body), f"deepest contact: {1e3 * probe.pen:.3g} mm against {against}")
    if cause in NONFINITE and (probe.nonfinite_step >= 0).any():
        bad = np.where(probe.nonfinite_step >= 0, probe.nonfinite_step, np.iinfo(np.int64).max)
        body = int(np.argmin(bad))
        return Culprit(body, name(body), f"first non-finite body (step {int(bad[body])})")
    table = ((SPEED, probe.speed, "fastest body: {:.3g} m/s"), (JUMP, probe.jump, "largest one-step jump: {:.3g} m"),
             (ENERGY, probe.gain, "largest energy rise net of applied work: {:.3g} J"), (DISTANCE, probe.dist, "farthest moved: {:.3g} m"))
    for causes, peak, text in table:
        if cause in causes and peak.size:
            body = int(np.argmax(peak))
            return Culprit(body, name(body), text.format(float(peak[body])))
    return Culprit(None, None, f"no body attribution for {cause}")


def _to_world(body_q: np.ndarray, bodies: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Body-frame contact points to world (body -1: already world)."""
    out = points.astype(np.float64).copy()
    moving = bodies >= 0
    if moving.any():
        q = body_q[bodies[moving]]
        out[moving] = q[:, :3] + _rotate(q[:, 3:7], out[moving])
    return out


def _rotate(quat_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = quat_xyzw[:, :3], quat_xyzw[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)
