# SPDX-License-Identifier: Apache-2.0
"""Harness smoke test: one free-falling sphere per world, compared with the exact parabola.

    python -m bench.run --family _example --device cpu --ladder 1,16

Law: z(t) = z0 - |g| t^2 / 2 (exact). A first-order (semi-implicit Euler) integrator is off by |g| h t / 2 at
substep h, so Newton's verification test allows 2 * |g| h t / 2 (newton@ac7f7830:
newton/tests/test_physics_verification.py:85); metrics.law_bound_mm / law_pass report that bound.
"""
from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import warp as wp

from bench import metrics as M
from bench.harness import Built, Config, Scenario

Z0 = 2.0  # [m], no ground: the horizon (0.5 s) ends before the sphere would reach z = 0.77 m


def make_rigid_solver(cfg: Config, model: newton.Model):
    """Rigid solver factory shared by simple scenes (families may copy it)."""
    S, p = newton.solvers, cfg.params
    if cfg.solver == "xpbd":
        return S.SolverXPBD(model, iterations=p.get("iterations", 4))
    if cfg.solver == "semi_implicit":
        return S.SolverSemiImplicit(model)
    if cfg.solver == "featherstone":
        return S.SolverFeatherstone(model)
    if cfg.solver == "mujoco":
        return S.SolverMuJoCo(model, use_mujoco_cpu=p.get("use_mujoco_cpu", False))
    raise ValueError(f"unknown solver {cfg.solver!r}")


class FreeFall(Scenario):
    id = "X01_free_fall"
    family = "_example"
    title = "Free-falling sphere per world vs z0 - g t^2/2"
    horizon_s = 0.5
    accuracy_unit = "mm"
    accuracy_ref = "max |z - (z0 - g t^2/2)| over worlds and samples; exact parabola"
    configs = (
        Config("xpbd_s10", "xpbd", dt=1 / 60, substeps=10),
        Config("semi_implicit_s10", "semi_implicit", dt=1 / 60, substeps=10),
        Config("featherstone_s10", "featherstone", dt=1 / 60, substeps=10),
        Config("mujoco_warp_s1", "mujoco", dt=1 / 600, substeps=1),
    )

    def build(self, nworld: int, device: str, config: Config) -> Built:
        scene = newton.ModelBuilder()
        body = scene.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, Z0), q=wp.quat_identity()), label="ball")
        scene.add_shape_sphere(body, radius=0.05)
        main = newton.ModelBuilder()
        main.replicate(scene, copy_count=nworld)
        model = main.finalize(device=device)
        solver = make_rigid_solver(config, model)
        s0, s1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        g = float(np.linalg.norm(np.asarray(model.gravity.numpy()).reshape(-1, 3)[0]))
        return Built(model, solver, s0, s1, control, nworld=nworld,
                     extra={"g": g, "h": config.dt / config.substeps, "err_mm": 0.0, "bound_mm": 0.0})

    def measure(self, b: Built, t: float, rec: dict) -> None:
        z = b.state_0.body_q.numpy()[:, 2].astype(np.float64)
        g, h = b.extra["g"], b.extra["h"]
        err = float(np.max(np.abs(z - (Z0 - 0.5 * g * t * t)))) * 1e3
        b.extra["err_mm"] = max(b.extra["err_mm"], err)
        b.extra["bound_mm"] = max(b.extra["bound_mm"], 2.0 * 0.5 * g * h * t * 1e3)

    def finish(self, b: Built, rec: dict) -> None:
        rec["accuracy_error"] = b.extra["err_mm"]
        rec["metrics"].update(law_bound_mm=b.extra["bound_mm"], law_pass=b.extra["err_mm"] <= b.extra["bound_mm"],
                              gravity=b.extra["g"])

    def energy(self, b: Built) -> float:
        return M.rigid_energy(b.model, b.state_0)


SCENARIOS = [FreeFall()]
