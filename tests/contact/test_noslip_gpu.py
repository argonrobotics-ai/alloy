# SPDX-License-Identifier: Apache-2.0
"""Scoped noslip on CUDA: the block solve (32 lanes per world, alloy.contact.noslip_block) against MuJoCo Warp's own
noslip, and inside a CUDA graph.

- scope "all" on the incline boxes (4 worlds): one step from states equal up to GPU run-to-run noise gives MuJoCo
  Warp's forces, qacc and sweep count; 800 more steps stay within float32 summation-order noise;
- scope "robot" on the T12 trace start (sparse Jacobian, nv 64, N = 1): same poses as MuJoCo Warp's full noslip;
- scope "all" on a hinged box snake (one 10-dof tree: the warp-wide factor and qacc restart) against MuJoCo Warp;
- a captured graph of the step replays the eager result (MuJoCo Warp's own GPU step is not bit-reproducible run to
  run: its collision and solver use atomics, so states match to float noise, not bit for bit).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy.contact.noslip import ALL, ROBOT, NoslipSpec, install_noslip
from alloy.core import newton_compat as nc
from alloy.testing.scenes import incline
from noslip_scenes import snake_on_slope

pytestmark = pytest.mark.gpu
DEVICE = os.environ.get("ALLOY_DEVICE", "cuda:0")
T12_XML = Path(__file__).resolve().parents[2] / "library/robots/task_t12/scene.xml"


def _incline_pair(worlds: int = 4, steps: int = 150):
    runners = [incline(worlds, 0.8, 0.4, device=DEVICE).runner("mujoco", cone="elliptic") for _ in range(2)]
    for r in runners:
        r.step(steps)
    return runners


@pytest.mark.parametrize(("iterations", "tolerance"), [(10, 1e-30), (50, 1e-10)])
def test_block_solve_matches_mujoco_warp(iterations: int, tolerance: float) -> None:
    ref, alloy = _incline_pair()
    assert np.abs(ref.state.body_q.numpy() - alloy.state.body_q.numpy()).max() < 1e-7  # GPU run-to-run noise
    nc.enable_mujoco_noslip(ref.solver, iterations, tolerance)
    inst = install_noslip(alloy.solver, NoslipSpec(iterations, tolerance, scope=ALL))
    ref.step(1)
    alloy.step(1)
    a, b = ref.solver.mjw_data, alloy.solver.mjw_data
    n = int(a.nefc.numpy().max())
    assert n > 0 and np.array_equal(a.solver_niter.numpy(), b.solver_niter.numpy())
    assert np.allclose(a.efc.force.numpy()[:, :n], b.efc.force.numpy()[:, :n], rtol=0, atol=1e-5)
    assert np.allclose(a.qacc.numpy(), b.qacc.numpy(), rtol=0, atol=1e-4)
    assert np.allclose(a.efc.Ma.numpy(), b.efc.Ma.numpy(), rtol=0, atol=1e-4)
    ref.step(800)
    alloy.step(800)
    assert np.abs(ref.state.body_q.numpy() - alloy.state.body_q.numpy()).max() < 1e-6  # float32 summation order
    assert inst.overflow_worlds() == 0 and int(inst.count.numpy().min()) == 4  # four box corners per world


def test_large_tree_matches_mujoco_warp() -> None:
    """Trees over MAXT8 dofs factor and restart warp-wide (32 lanes split each column and dot product)."""
    ref, alloy = (snake_on_slope(worlds=4, device=DEVICE).runner("mujoco", cone="elliptic") for _ in range(2))
    for r in (ref, alloy):
        r.step(150)
    assert np.abs(ref.state.body_q.numpy() - alloy.state.body_q.numpy()).max() < 1e-6  # GPU run-to-run noise
    nc.enable_mujoco_noslip(ref.solver, 50, 1e-10)
    inst = install_noslip(alloy.solver, NoslipSpec(50, 1e-10, scope=ALL))
    ref.step(1)
    alloy.step(1)
    a, b = ref.solver.mjw_data, alloy.solver.mjw_data
    n = int(a.nefc.numpy().max())
    assert int(alloy.solver.mj_model.tree_dofnum.max()) == 10 and int(inst.count.numpy().min()) >= 5
    # MuJoCo Warp orders contact rows by atomics on GPU, so compare the per-world sorted forces, not row by row
    fa, fb = np.sort(a.efc.force.numpy()[:, :n], axis=1), np.sort(b.efc.force.numpy()[:, :n], axis=1)
    assert np.allclose(fa, fb, rtol=0, atol=1e-4)
    assert np.allclose(a.qacc.numpy(), b.qacc.numpy(), rtol=0, atol=1e-3)
    ref.step(400)
    alloy.step(400)
    assert np.abs(ref.state.body_q.numpy() - alloy.state.body_q.numpy()).max() < 1e-5
    assert inst.overflow_worlds() == 0


def test_graph_replays_eager() -> None:
    eager, graph = _incline_pair()
    for r in (eager, graph):
        install_noslip(r.solver, NoslipSpec(50, 1e-10, scope=ALL))
        r.step(2)  # compile and warm up outside the capture
    with wp.ScopedDevice(DEVICE):
        wp.capture_begin()
        try:
            graph.step(2)
        finally:
            g = wp.capture_end()
        for _ in range(100):
            wp.capture_launch(g)
    eager.step(200)
    assert np.abs(eager.state.body_q.numpy() - graph.state.body_q.numpy()).max() < 1e-6


def test_t12_robot_scope_matches_full_noslip() -> None:
    """Sparse path on the recorded scene, first 0.2 s (arm moving, blocks at rest): the scoped rows give MuJoCo
    Warp's noslip over every row, because the resting blocks' friction rows carry no slip to remove."""
    from alloy.tasks.cli import TRACES
    from alloy.tasks.replay import ReplayConfig, build_rig, import_scene, run_replay
    from alloy.tasks.trace import load_trace

    tt = load_trace(TRACES / "t12")
    cfg = ReplayConfig(seconds=0.2, device=DEVICE)
    out = []
    for scoped in (False, True):
        with wp.ScopedDevice(DEVICE):
            rig = build_rig(import_scene(T12_XML, DEVICE), tt, cfg)
            if scoped:
                inst = install_noslip(rig.solver, NoslipSpec(50, 1e-10, scope=ROBOT))
            else:
                nc.enable_mujoco_noslip(rig.solver, 50, 1e-10)
            run_replay(rig, cfg)
            out.append(rig.states[0].body_q.numpy())
    assert inst.overflow_worlds() == 0
    assert int(inst.count.numpy()[0]) >= 16  # the two arms' dof friction rows
    assert np.abs(out[0][:, :3] - out[1][:, :3]).max() < 1e-5  # GPU run-to-run noise: 2-3 µm over 0.2 s
