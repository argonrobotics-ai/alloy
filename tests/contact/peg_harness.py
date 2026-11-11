# SPDX-License-Identifier: Apache-2.0
"""Shared GPU harness for the peg-in-hole CT tests: build, step with external forces, measure, record rows."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.precision import ladder_rungs, size_l2
from alloy.precision.types import StepHealth
from alloy.testing.contact import PegHoleScene

RESULTS = Path(os.environ.get("ALLOY_RESULTS_DIR", "bench/results/wp3"))
CONTROL_DT = 0.02


@dataclass
class PegRun:
    scene: PegHoleScene
    native: str
    dt: float
    layer: AlloyContactLayer
    solver: object
    monitor: object
    health: wp.array
    s0: newton.State
    s1: newton.State
    ctrl: newton.Control
    integrator: str
    rungs: tuple[str, ...]
    steps: int = 0
    wall_s: float = 0.0
    extra: dict = field(default_factory=dict)


def l2_dt(scene: PegHoleScene, contact_speed: float = 0.05, design_force: float = 50.0) -> float:
    """The §3.5 L2 step for the pair (grid value)."""
    return size_l2(scene.spec, m_eff=scene.peg_mass, design_force=design_force, contact_speed=contact_speed,
                   control_dt=CONTROL_DT).dt


def _health(n: int, device) -> wp.array:
    h = StepHealth()
    h.min_pair_gap_m = np.inf
    return wp.array([h] * n, dtype=StepHealth, device=device)


def make_run(scene: PegHoleScene, native: str, dt: float, *, integrator: str = "euler",
             deterministic: bool = True, spacing: float = 5e-4) -> PegRun:
    model, spec = scene.model, scene.spec
    layer = AlloyContactLayer(model, [spec], native=native, precision_contacts_per_world=192,
                              deterministic=deterministic, measure_unsupported=True)  # CT rows measure every combo
    l1 = spec.ccd != "off" and integrator in ("euler", "implicitfast")
    solver = newton.solvers.SolverMuJoCo(
        model, use_mujoco_contacts=(native == "mujoco"), use_speculative_contacts=(native == "newton" and l1),
        integrator=integrator, cone="elliptic", impratio=spec.mj_impratio, nconmax=256, njmax=1024,
        iterations=100, ls_iterations=50)
    if native == "mujoco":
        layer.bind_mujoco(solver).install()
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    return PegRun(scene, native, dt, layer, solver, layer.monitor(sample_spacing=spacing),
                  _health(model.world_count, model.device), s0, s1, model.control(), integrator,
                  ladder_rungs(spec, integrator, native))


def perturb(run: PegRun, rng: np.random.Generator, lateral: float = 3e-5, tilt_deg: float = 0.2) -> None:
    """Per-world random lateral offset and tilt of the peg (16 worlds per condition, bench §4.9)."""
    q = run.s0.body_q.numpy()
    for b in run.scene.peg_bodies:
        ax = rng.normal(size=3)
        ax[2] = 0.0
        ax /= np.linalg.norm(ax)
        ang = math.radians(rng.uniform(0.0, tilt_deg))
        quat = np.r_[ax * math.sin(ang / 2), math.cos(ang / 2)]
        q[b, 0:2] += rng.uniform(-lateral, lateral, 2)
        q[b, 3:7] = quat
    run.s0.body_q.assign(q)
    _sync_joints(run)


def _sync_joints(run: PegRun) -> None:
    """Free-joint coordinates from body_q (the solver reads joint_q)."""
    newton.eval_ik(run.scene.model, run.s0, run.s0.joint_q, run.s0.joint_qd)


def set_wrench(run: PegRun, force: np.ndarray, at_top: bool = False) -> None:
    """World-frame force [W,3] on each peg at its COM (or at the top end centre, which adds r x F)."""
    f = np.zeros((run.scene.model.body_count, 6), np.float32)
    q = run.s0.body_q.numpy()
    for w, b in enumerate(run.scene.peg_bodies):
        f[b, :3] = force[w]
        if at_top:
            axis = _axis(q[b, 3:7])
            f[b, 3:] = np.cross(0.5 * run.scene.geometry.peg_len * axis, force[w])
    for st in (run.s0, run.s1):
        st.body_f.assign(f)


def _axis(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def advance(run: PegRun, steps: int) -> None:
    t0 = time.time()
    for _ in range(steps):
        run.layer.collide(run.s0, run.layer.contacts(), dt=run.dt)
        run.solver.step(run.s0, run.s1, run.ctrl, run.layer.contacts(), run.dt)
        run.monitor.update(run.s0, run.s1, run.health)
        bf = run.s0.body_f
        run.s0, run.s1 = run.s1, run.s0
        run.s0.body_f.assign(bf)
    wp.synchronize_device(run.scene.model.device)
    run.steps += steps
    run.wall_s += time.time() - t0


def ramp(run: PegRun, direction: np.ndarray, f0: float, f1: float, duration: float, n_seg: int = 20,
         at_top: bool = False) -> None:
    steps = max(1, int(round(duration / run.dt / n_seg)))
    for k in range(n_seg):
        f = f0 + (f1 - f0) * (k + 1) / n_seg
        set_wrench(run, np.tile(direction * f, (run.scene.model.world_count, 1)), at_top)
        advance(run, steps)


def tilt_deg(run: PegRun) -> np.ndarray:
    q = run.s0.body_q.numpy()
    return np.array([math.degrees(math.acos(min(1.0, abs(_axis(q[b, 3:7])[2])))) for b in run.scene.peg_bodies])


def overflow_total(run: PegRun) -> dict[str, int]:
    rep = run.layer.overflow_report()
    rep["mjwarp_overflow"] = int(np.asarray(run.solver.mjw_data.overflow.numpy()).max())
    binding = getattr(run.layer, "_binding", None)
    rep["append"] = 0 if binding is None else int(binding.overflow.numpy()[0])
    return rep


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=False).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def record(name: str, row: dict) -> None:
    """Append one JSON row to bench/results/wp3/<name>.jsonl (copied back by scripts/gpu_run.sh)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = dict(row, commit=commit(), date=time.strftime("%Y-%m-%d %H:%M:%S"))
    with (RESULTS / f"{name}.jsonl").open("a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
