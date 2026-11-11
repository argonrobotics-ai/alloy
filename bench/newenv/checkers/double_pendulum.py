# SPDX-License-Identifier: Apache-2.0
"""E07 double pendulum: energy conservation at rest, tip vs forward kinematics, work done by the elbow torque.

Model (from the request): two uniform rods, L = 0.3 m, m = 0.5 kg each, frictionless hinges, pivot fixed, swinging in
a vertical plane; torque 2 N m x action at the elbow. With theta1 = the upper rod's angle from straight down and
theta2 = the lower rod's angle from straight down (absolute; if the probe's elbow is relative, theta2 = shoulder + elbow):
  tip - pivot = L (sin t1 + sin t2, -(cos t1 + cos t2)) in the plane
  KE = (1/2)(m L^2 / 3) w1^2 + (1/2) m (L^2 w1^2 + (L/2)^2 w2^2 + L^2 w1 w2 cos(t1 - t2)) + (1/2)(m L^2 / 12) w2^2
  PE = m g (-(L/2) cos t1 - L cos t1 - (L/2) cos t2)       (z up, pivot at 0)
The probe convention is not stated in the request (relative or absolute elbow; either sign): the one whose forward
kinematics matches tip_pos best is used. Both signs of rotation give the same energy and the same |horizontal tip
motion|, so no sign is assumed. The pivot is not probed: tip displacement from the first row is compared, the vertical
part signed and the horizontal part as a length (its direction in the world is not specified).
Energy drift at rest: max |E(t) - E(0)| over 2 s against max(E(0) - E_hang, 0.2 J) x 2 % (E_hang = -2 m g L).
Work: the elbow torque is 2 N m x (+1); its power is torque x the elbow's relative rate, so E(t) - E(0) = +-2 (q_rel(t) -
q_rel(0)) (sign by the convention of the torque, taken as the better fit); max over rows of the deviation must be
< 20 % of the largest |work| (floor 0.05 J).
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

L, M, G, TORQUE = 0.3, 0.5, 9.81, 2.0
FK_TOL, DRIFT_TOL, WORK_TOL = 0.005, 0.02, 0.20
DRIFT_FLOOR_J, WORK_FLOOR_J = 0.2, 0.05
MIN_WORLDS = 15


def absolute_angles(q: np.ndarray, qd: np.ndarray, relative: bool) -> tuple[np.ndarray, ...]:
    """(t1, t2, w1, w2) absolute rod angles from down and rates, from probe joint_pos / joint_vel (T, N, 2)."""
    if relative:
        return q[..., 0], q[..., 0] + q[..., 1], qd[..., 0], qd[..., 0] + qd[..., 1]
    return q[..., 0], q[..., 1], qd[..., 0], qd[..., 1]


def energy(t1: np.ndarray, t2: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    ke = (0.5 * M * L ** 2 / 3 * w1 ** 2
          + 0.5 * M * (L ** 2 * w1 ** 2 + (L / 2) ** 2 * w2 ** 2 + L ** 2 * w1 * w2 * np.cos(t1 - t2))
          + 0.5 * M * L ** 2 / 12 * w2 ** 2)
    pe = M * G * (-(L / 2) * np.cos(t1) - L * np.cos(t1) - (L / 2) * np.cos(t2))
    return ke + pe


def fk_error(q: np.ndarray, tip: np.ndarray, relative: bool) -> np.ndarray:
    """(N,) worst tip-displacement mismatch (m) over rows: vertical signed, horizontal as a length."""
    t1, t2, _, _ = absolute_angles(q, np.zeros_like(q), relative)
    height = -L * (np.cos(t1) + np.cos(t2))
    horiz = L * (np.sin(t1) + np.sin(t2))
    err_v = np.abs((tip[..., 2] - tip[0:1, :, 2]) - (height - height[0:1]))
    moved = np.linalg.norm(tip[..., :2] - tip[0:1, :, :2], axis=-1)
    err_h = np.abs(moved - np.abs(horiz - horiz[0:1]))
    return np.maximum(err_v, err_h).max(axis=0)


def best_convention(q: np.ndarray, tip: np.ndarray) -> tuple[bool, np.ndarray]:
    """(relative?, per-world error) for the elbow convention whose median FK error is smaller."""
    rel, ab = fk_error(q, tip, True), fk_error(q, tip, False)
    return (True, rel) if np.median(rel) <= np.median(ab) else (False, ab)


def _energy_rows(phase: c.Phase, relative: bool) -> np.ndarray:
    a = absolute_angles(phase.probe["joint_pos"], phase.probe["joint_vel"], relative)
    return energy(*a)


def _rest_checks(r: c.Rollout, relative: bool) -> list[c.Check]:
    rest = r.phase("rest")
    e = _energy_rows(rest, relative)
    scale = np.maximum(e[0] + 2 * M * G * L, DRIFT_FLOOR_J)
    drift = np.abs(e - e[0]).max(axis=0) / scale
    return [c.within("rest_energy_drift", drift, 0.0, DRIFT_TOL, 2, min_worlds=MIN_WORLDS)]


def _work_check(r: c.Rollout, relative: bool) -> c.Check:
    ph = r.phase("act_torque")
    e = _energy_rows(ph, relative)
    q = ph.probe["joint_pos"][..., 1]
    q_rel = q if relative else q - ph.probe["joint_pos"][..., 0]
    de, work = e - e[0], TORQUE * (q_rel - q_rel[0])
    ok = np.zeros(e.shape[1], bool)
    for sign in (1.0, -1.0):
        dev = np.abs(de - sign * work).max(axis=0)
        ok |= dev <= WORK_TOL * np.maximum(np.abs(work).max(axis=0), WORK_FLOOR_J / WORK_TOL)
    done_mid = ph.done[:-1].any(axis=0)                  # a reset inside the 1 s window spoils the integral
    good = ok & ~done_mid
    return c.at_least("act_work_equals_torque_x_elbow_angle", float(good.sum()), MIN_WORLDS, 2)


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    rest = rollout.phase("rest")
    relative, err = best_convention(rest.probe["joint_pos"], rest.probe["tip_pos"])
    act = rollout.phase("act_torque")
    err_act = fk_error(act.probe["joint_pos"], act.probe["tip_pos"], relative)
    checks.append(c.within("tip_matches_fk", np.maximum(err, err_act), 0.0, FK_TOL, 2, min_worlds=MIN_WORLDS))
    checks += _rest_checks(rollout, relative) + [_work_check(rollout, relative)]
    return c.finish(checks, {"elbow_relative": relative, "fk_err_median_m": float(np.median(err))})
