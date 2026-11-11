# SPDX-License-Identifier: Apache-2.0
"""Opt-in L2 force norm of the pass gate (``InterfaceTolerance.force_norm``; WP11's norm for SPH-VBD P2 pairs).

Per world: ||F_raw - F_used||_2 / max(||F_raw||_2, force_floor·sqrt(n)), the endpoint vec3 residuals flattened into one
vector ("l2"), or the same over free endpoints only (meff > 0, "l2_free"). Sums are float64, one thread per world over
a world-sorted CSR, so they do not depend on the batch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.coupling import residuals as rs
from alloy.coupling.aitken import build_world_csr
from alloy.coupling.proxy_loop import IterationConfig
from alloy.coupling.types import InterfaceTolerance, SchedulerPolicy
from alloy.testing.coupling import ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth

DEV = "cpu"
FLOOR = 1.0e-3
# two worlds; world 1 is listed first to check the CSR grouping; endpoint 2 is pinned (meff 0)
WORLD = [1, 0, 0, 1]
MEFF = [0.1, 0.1, 0.0, 0.1]
RAW = np.array([[0, 0, 2.0], [0, 0, 3.0], [0, 0, 40.0], [0, 0, 0.0]], np.float32)
USED = np.array([[0, 0, 1.0], [0, 0, 3.0], [0, 0, 0.0], [0, 0, 0.0]], np.float32)


def _expected(world: int, free_only: bool) -> float:
    keep = [i for i, w in enumerate(WORLD) if w == world and (not free_only or MEFF[i] > 0.0)]
    d, r = (RAW - USED)[keep].astype(np.float64), RAW[keep].astype(np.float64)
    return float(np.linalg.norm(d) / max(np.linalg.norm(r), FLOOR * np.sqrt(len(keep))))


def _gate(norm: str, body: bool) -> np.ndarray:
    """``res_force`` per world after the L2 sums and the convergence kernel."""
    worlds = 2
    csr = build_world_csr(np.array(WORLD), worlds, DEV)
    if body:
        dtype = wp.spatial_vector
        pad = lambda a: np.concatenate([a, np.full_like(a, 7.0)], axis=1)   # noqa: E731  torque part is ignored
        raw, used = wp.array(pad(RAW), dtype=dtype, device=DEV), wp.array(pad(USED), dtype=dtype, device=DEV)
    else:
        raw, used = wp.array(RAW, dtype=wp.vec3, device=DEV), wp.array(USED, dtype=wp.vec3, device=DEV)
    converged = wp.zeros(worlds, dtype=wp.int32, device=DEV)
    sums = wp.zeros((worlds, 3), dtype=wp.float64, device=DEV)
    wp.launch(rs.force_l2_sums_kernel, dim=worlds, device=DEV,
              inputs=[int(norm == "l2_free"), csr.start, csr.index, wp.array(MEFF, dtype=float, device=DEV),
                      converged, raw, used, sums])
    p = rs.ConvergenceParams()
    p.k_min, p.k_max, p.tol_vel, p.tol_pos, p.tol_force, p.check_force = 1, 4, 1.0, 1.0, 1.0e-2, 1
    p.force_norm, p.force_floor = rs.FORCE_NORMS[norm], FLOOR
    res_force = wp.full(worlds, 99.0, dtype=float, device=DEV)
    zeros = wp.zeros(worlds, dtype=float, device=DEV)
    wp.launch(rs.convergence_kernel, dim=worlds, device=DEV,
              inputs=[p, wp.zeros(1, dtype=wp.int32, device=DEV), zeros, zeros, res_force, sums, converged,
                      wp.zeros(worlds, dtype=wp.int32, device=DEV), wp.zeros(1, dtype=wp.int32, device=DEV)])
    return res_force.numpy()


@pytest.mark.parametrize("body", [False, True], ids=["particle", "body"])
@pytest.mark.parametrize("norm", ["l2", "l2_free"])
def test_l2_norm_matches_the_formula(norm: str, body: bool) -> None:
    got = _gate(norm, body)
    want = [_expected(w, norm == "l2_free") for w in (0, 1)]
    np.testing.assert_allclose(got, want, rtol=1.0e-6)
    if norm == "l2_free":
        assert got[0] == 0.0          # world 0's only free endpoint has no residual; its pinned one is left out


def test_max_norm_leaves_res_force_alone() -> None:
    assert list(_gate("max", body=False)) == [99.0, 99.0]


def test_unknown_norm_is_refused() -> None:
    cfg = IterationConfig(k_min=1, k_max=4, tol_vel=1.0, tol_pos=1.0, tol_force=1.0e-2, force_floor=FLOOR,
                          gravity=9.81, omega0=0.5, omega0_strong=0.1, omega_min=0.05, omega_max=1.0)
    assert cfg.force_norm == "max"
    with pytest.raises(ValueError, match="force_norm"):
        dataclasses.replace(cfg, force_norm="l1")


def test_staggered_scheduler_gates_on_the_l2_norm() -> None:
    """Box on cloth, staggered, CPU: a world that stops before k_max passed the L2 gate."""
    spec = ClothSceneSpec(box_mass=0.2, cells=8)
    model = coupling_box_on_cloth(2, spec, device=DEV)
    policy = SchedulerPolicy(tol=InterfaceTolerance(force_norm="l2"))
    sched = cloth_scheduler(model, spec, policy=policy, mode="staggered", vbd_adaptive=False)
    loop = sched.iteration_loop
    assert loop.cfg.force_norm == "l2"
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    early = 0
    for _ in range(40):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        sched.step(s0, s1, control, contacts, 1.0 / 480.0)
        s0, s1 = s1, s0
        iters, res = loop.iters_used.numpy(), loop.res_force.numpy()
        assert np.all(np.isfinite(res))
        done = iters < loop.cfg.k_max
        assert np.all(res[done] <= loop.cfg.tol_force)
        early += int(done.sum())
    assert early > 0
