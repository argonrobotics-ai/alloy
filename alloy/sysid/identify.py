# SPDX-License-Identifier: Apache-2.0
"""Which fitted parameters this recording really determines.

Around the fit, each variable is nudged up and down (in every joint at once, one pair of worlds per variable) and
the change of each joint's simulated motion is its sensitivity. With the size of the remaining error this gives an
approximate 1-sigma interval per parameter (Gauss-Newton: ``sigma^2 (J^T J)^-1``, in the search's unit box). A
parameter whose interval is wider than ``WIDE`` of its bounds, or whose joint never moved, is reported as poorly
determined: its fitted value is kept, and the report says not to trust it.

Limits of the estimate: joints are nudged together, so the cross-coupling of joints is folded into each joint's
own numbers; errors of neighbouring samples are treated as independent only every ``CORRELATION`` seconds; a model
error that is not noise (backlash, a flexible link) makes the intervals too narrow.
"""

from __future__ import annotations

import numpy as np

NUDGE = 0.04            # unit-box step each way
WIDE = 0.08             # a 1-sigma interval wider than this share of the bounds is "poorly determined"
CORRELATION = 0.1       # s: tracking errors closer in time than this count as one sample
MOVED_HINGE = 0.02      # rad: a joint whose command and position both span less never moved
MOVED_SLIDE = 0.002     # m


def moved(command: np.ndarray, position: np.ndarray, hinge: np.ndarray) -> np.ndarray:
    """(J,) bool: the joint's command or measured position spans more than 0.02 rad (2 mm for a slide)."""
    span = np.maximum(np.ptp(command, axis=0), np.ptp(position, axis=0))
    return span > np.where(hinge, MOVED_HINGE, MOVED_SLIDE)


def nudged(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Worlds for the sensitivity run around ``u`` (V, J): ``(points (2V, V, J), steps (V, J))`` where world 2v
    has variable v raised and world 2v + 1 lowered, by ``steps`` each way on average (less at a bound)."""
    count = u.shape[0]
    points = np.repeat(u[None], 2 * count, axis=0)
    for v in range(count):
        points[2 * v, v] = np.clip(u[v] + NUDGE, 0.0, 1.0)
        points[2 * v + 1, v] = np.clip(u[v] - NUDGE, 0.0, 1.0)
    steps = np.stack([(points[2 * v, v] - points[2 * v + 1, v]) / 2.0 for v in range(count)])
    return points, steps


def unit_sigma(residual: np.ndarray, response: np.ndarray, steps: np.ndarray, dt: float) -> np.ndarray:
    """1-sigma of each variable in the unit box, (V, J). ``residual`` (N, J): simulated minus measured at the
    fit; ``response`` (2V, N, J): simulated positions of the nudged worlds; ``steps`` (V, J) from ``nudged``."""
    count, joints = steps.shape
    samples = max(1.0, residual.shape[0] * dt / CORRELATION)
    out = np.full((count, joints), np.inf)
    for j in range(joints):
        slope = np.stack([(response[2 * v, :, j] - response[2 * v + 1, :, j]) / (2.0 * max(steps[v, j], 1e-9))
                          for v in range(count)], axis=1)                    # (N, V)
        if not np.all(np.isfinite(slope)):
            continue
        noise = float(np.mean(residual[:, j] ** 2))
        normal = slope.T @ slope / residual.shape[0] * samples               # information of `samples` points
        scale = np.sqrt(np.maximum(np.diag(normal), 1e-300))
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):      # a flat direction reads inf
                cov = np.linalg.inv(normal / np.outer(scale, scale) + 1e-9 * np.eye(count)) / np.outer(scale, scale)
        except np.linalg.LinAlgError:
            continue
        out[:, j] = np.sqrt(np.maximum(noise * np.diag(cov), 0.0))
        out[np.diag(normal) < 1e-24, j] = np.inf
    return out


def determined(sigma: np.ndarray, has_moved: np.ndarray) -> np.ndarray:
    """(V, J) bool: the recording pins the variable down (interval under ``WIDE`` of its bounds, joint moved)."""
    return (sigma < WIDE) & has_moved[None, :]


__all__ = ["WIDE", "determined", "moved", "nudged", "unit_sigma"]
