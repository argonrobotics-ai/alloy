# SPDX-License-Identifier: Apache-2.0
"""Newton runs behind the rod self-test: a clamped CHAIN on SolverMuJoCo (CPU), tip drop history."""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.route import rodref
from alloy.route.types import RodCard

log = logging.getLogger("alloy.route")

ROOT_Z = 1.0  # m, clamp height of the self-test cantilever


def build_cantilever(rod: RodCard, dt: float, damping_tau: float, device: str = "cpu"):
    """(model, solver, tip body index, tip offset along the last link) of the clamped chain of `rod`."""
    from alloy.scene.emitters.chain import ChainOptions, build_chain, first_mode

    w1 = first_mode(rod)
    rod = replace(rod, damping_ratio=0.5 * damping_tau * w1)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -rodref.G))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    pts = [(0.0, 0.0, ROOT_Z), (rod.length, 0.0, ROOT_Z)]
    h = build_chain(b, rod, pts, ChainOptions(capability="mujoco", dt=dt, clamp=True, collide=False),
                    label="selftest")
    model = b.finalize(device=device)
    from alloy.scene.inertia import restore_authored_inertia

    restore_authored_inertia(b, model)          # stubby links: Newton's 1e-10 floor added 1e-6 kg·m² (CARD-1)
    solver = newton.solvers.SolverMuJoCo(model, integrator=rod.integrator, solver="newton", iterations=50,
                                         disable_contacts=True)
    return model, solver, h.bodies[-1], 0.5 * h.lengths[-1]


def run_clamped(rod: RodCard, dt: float, t_end: float, damping_tau: float,
                device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Step the clamped chain from straight; returns (t, tip drop) sampled every few steps [s, m]."""
    model, solver, tip, off = build_cantilever(rod, dt, damping_tau, device)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    steps = int(math.ceil(t_end / dt))
    every = max(1, int(round(2.5e-3 / dt)))
    ts, drops = [0.0], [0.0]
    for k in range(1, steps + 1):
        solver.step(s0, s1, ctrl, None, dt)
        s0, s1 = s1, s0
        if k % every == 0 or k == steps:
            q = s0.body_q.numpy()[tip]
            p = wp.transform_point(wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:])), wp.vec3(off, 0.0, 0.0))
            ts.append(k * dt)
            drops.append(ROOT_Z - float(p[2]))
    return np.asarray(ts), np.asarray(drops)


def period_from_signal(t: np.ndarray, x: np.ndarray) -> float:
    """Mean period from upward zero crossings (linear interpolation); NaN with fewer than two crossings."""
    s = np.sign(x)
    idx = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(idx) < 2:
        return math.nan
    tc = t[idx] - x[idx] * (t[idx + 1] - t[idx]) / (x[idx + 1] - x[idx])
    return float(np.mean(np.diff(tc)))
