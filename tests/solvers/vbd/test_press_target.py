# SPDX-License-Identifier: Apache-2.0
"""SB-7 cloth bully target ``cloth_box_press`` (alloy.solvers.vbd.press_target): registration, actions, S1 rows."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.bully.targets import get_target
from alloy.solvers.vbd.press_target import TARGET_NAME, PressSpec, count_crossings, sheet_pairs

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def sim():
    """Two worlds of the target on CPU, settled for two control periods."""
    s = get_target(TARGET_NAME).make_sim(2, "cpu")
    s.step_period()
    s.step_period()
    return s


def test_registered_with_the_agreed_action_space():
    t = get_target(TARGET_NAME)
    spec = PressSpec()
    assert t.name == TARGET_NAME and t.action_kind == "joint_target_q" and t.action_index == (0, 1, 2)
    np.testing.assert_allclose(t.actions.low, [-0.12, -0.12, -0.12])
    np.testing.assert_allclose(t.actions.high, [0.12, 0.12, 0.0])
    np.testing.assert_allclose(t.actions.max_delta_per_tick, np.full(3, spec.speed * spec.control_dt))


def test_rows_are_zero_at_rest_and_the_box_waits_above_the_sheet(sim):
    rows = sim.bully_monitors()
    assert set(rows) == {"cloth_pass_through", "cloth_crossings"}
    for counts in rows.values():
        assert counts.shape == (2,) and not counts.any()
    box_bottom = sim.state.body_q.numpy()[sim.box_of_world.numpy(), 2] - PressSpec().half
    sheet_top = sim.state.particle_q.numpy()[:, 2].max()
    assert np.all(box_bottom > sheet_top)


def test_rows_count_a_particle_in_the_box_and_a_crossing_per_world(sim):
    q0 = sim.state.particle_q.numpy().copy()
    n = q0.shape[0] // 2
    try:
        q = q0.copy()
        q[n + 40] = sim.state.body_q.numpy()[sim.box_of_world.numpy()[1], :3]      # world 1: a particle in its box
        tris = sim.model.tri_indices.numpy()
        far = tris[100]                                                            # world 0: an edge through a far
        c = q0[far].mean(axis=0)                                                   # triangle (cell 1,1's first edge)
        q[0], q[1] = c + np.array([0.0, 0.0, 0.02]), c - np.array([0.0, 0.0, 0.02])
        sim.state.particle_q.assign(q.astype(np.float32))
        through, crossings = sim.counts()
        assert through.tolist() == [0, 1]
        assert crossings[0] > 0 and crossings[1] == 0
    finally:
        sim.state.particle_q.assign(q0)


def test_reset_clears_the_peaks_of_the_reset_worlds_only(sim):
    sim.peak_pass[:] = [3, 4]
    sim.peak_cross[:] = [5, 6]
    sim.reset(wp.array(np.array([1, 0], np.int32), dtype=wp.int32))
    assert sim.peak_pass.tolist() == [0, 4] and sim.peak_cross.tolist() == [0, 6]
    sim.reset()
    assert not sim.peak_pass.any() and not sim.peak_cross.any()


def test_a_full_press_by_a_pinned_edge_stays_out_of_the_box():
    """The calibration's hardest legal path (box at x = -reach, pressed to the bottom of its range at the rate limit,
    here on the diagonal, then held): the 150 N carriage cap keeps every cloth particle out of the box."""
    t = get_target(TARGET_NAME)
    s = t.make_sim(1, "cpu")
    start, step = int(s.model.joint_q_start.numpy()[0]), t.actions.max_delta_per_tick
    goal, cmd = np.array([t.actions.low[0], 0.0, t.actions.low[2]]), np.zeros(3)
    for _ in range(20):                                    # 12 periods to the corner of the range, 8 held there
        cmd += np.clip(goal - cmd, -step, step)
        tq = s.control.joint_target_q.numpy()
        tq[start:start + 3] = cmd
        s.control.joint_target_q.assign(tq)
        s.step_period()
    np.testing.assert_allclose(cmd, goal)
    rows = s.bully_monitors()
    assert not rows["cloth_pass_through"].any() and not rows["cloth_crossings"].any()


def _strip_sheets(sim, worlds: int) -> np.ndarray:
    """Sheets whose free particles sag by x only (+-1e-7 m jitter): the strips by each pinned edge are coplanar, an edge
    (free, pinned) touching a triangle (pinned, free, pinned)'s plane at its pinned end only. World 0 also gets a
    real crossing: an edge pushed through a far triangle (as in the crossing test above)."""
    rest = sim.model.particle_q.numpy().astype(np.float64)
    n = rest.shape[0] // 2
    rest, free = rest[:n], sim.model.particle_mass.numpy()[:n] > 0
    rng = np.random.default_rng(0)
    q = np.repeat(rest[None], worlds, axis=0)
    q[:, free, 2] -= rng.uniform(0.005, 0.05, (worlds, 1)) * np.cos(np.pi * rest[free, 0] / PressSpec().size)[None]
    q[:, free, 2] += rng.uniform(-1.0e-7, 1.0e-7, (worlds, int(free.sum())))
    c = q[0][sim.model.tri_indices.numpy()[100]].mean(axis=0)
    q[0, 0], q[0, 1] = c + np.array([0.0, 0.0, 0.02]), c - np.array([0.0, 0.0, 0.02])
    return q.astype(np.float32)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_coplanar_strips_by_a_pinned_edge_are_no_crossing(sim, device):
    """WP7's SB-7 trips (11 in 300k on an L4, CPU and float64 counting 0): the float32 test with CUDA's fused
    multiply-adds counted these strips as crossings (59 of 1,024 such sheets); the float64 test counts only the real
    one."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    worlds = 256
    q = _strip_sheets(sim, worlds)
    n = q.shape[1]
    _, edges, tris, pe, pt = sheet_pairs(sim.model)
    args = [wp.array((np.arange(worlds) * n).astype(np.int32), dtype=wp.int32, device=device)]
    args += [wp.array(a, dtype=wp.int32, device=device) for a in (edges, tris, pe, pt)]
    count = wp.zeros(worlds, dtype=wp.int32, device=device)
    wp.launch(count_crossings, dim=(worlds, pe.shape[0]), device=device,
              inputs=[wp.array(q.reshape(-1, 3), dtype=wp.vec3, device=device), *args], outputs=[count])
    c = count.numpy()
    assert c[0] > 0 and not c[1:].any()


@pytest.mark.gpu
def test_many_worlds_on_cuda():
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    s = get_target(TARGET_NAME).make_sim(64, "cuda:0")
    for _ in range(3):
        s.step_period()
    rows = s.bully_monitors()
    assert rows["cloth_pass_through"].shape == (64,) and not rows["cloth_pass_through"].any()
    assert not rows["cloth_crossings"].any()
