# SPDX-License-Identifier: Apache-2.0
"""JOINT_FAIL (SPEC §7 WP5 acceptance): quiet on a default-solref weld under gravity (SolverMuJoCo), and raised for
a joint pulled apart; the weld limit follows the solref steady-state rule of §4.2."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import mujoco as mj
from alloy.precision.health_setup import steady_state_violation
from alloy.precision.types import CheckPolicy, HealthFlags, Severity
from alloy.testing.precision import HealthLoop, set_body, world_bodies
from alloy.testing.scenes import RIGID_DT, RefScene, pendulum


def test_steady_state_rule() -> None:
    # k = 1/(dmax² τ² ζ²) = 1/(0.95² · 0.02² · 1) = 2770 s⁻²; r = (1 - 0.9)·9.81/(0.9·2770) = 0.393 mm
    r = steady_state_violation((0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0), 9.81)
    assert r == pytest.approx(0.1 * 9.81 / (0.9 / (0.95**2 * 0.02**2)), rel=1e-12)
    assert r == pytest.approx(3.93e-4, rel=1e-2)
    assert steady_state_violation((-1000.0, -10.0), (0.9, 0.95), 9.81) == pytest.approx(0.1 * 9.81 / (0.9 * 1000.0))


def _welded_box(n: int, device: str, lever: float = 0.2, at_body: bool = True) -> RefScene:
    """A 1 kg box welded in place with default solref/solimp. ``at_body``: the weld point is the box origin (world
    -> box, anchor in the box frame), so gravity exerts no torque about it; else the weld point is the world origin
    (box -> world), ``|(lever, 0, 0.5)|`` m from the box COM."""
    template = newton.ModelBuilder()
    box = template.add_body(xform=wp.transform(wp.vec3(lever, 0.0, 0.5), wp.quat_identity()), label="box")
    template.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05,
                           cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, has_shape_collision=False))
    if at_body:
        mj.add_equality_weld(template, -1, box)  # default solref (0.02, 1), solimp (0.9, 0.95, ...), relpose = qpos0
    else:
        mj.add_equality_weld(template, box)
    root = newton.ModelBuilder()
    root.replicate(template, n)
    return RefScene("welded_box", root.finalize(device=device), RIGID_DT, "mujoco", {"box": box}, {"lever": lever})


@pytest.mark.compile_heavy
def test_default_weld_under_gravity_is_quiet(device: str) -> None:
    scene = _welded_box(2, device)
    loop = HealthLoop(scene, checks=CheckPolicy(level="full", full_every=1), kind="mujoco",
                      solver_kwargs={"use_mujoco_contacts": True})
    loop.step(400, record=True)  # 0.5 s: several solref time constants, the weld settles at its sag
    h = np.stack(loop.history)
    lim_m = float(loop.monitor._welds.lim_m.numpy()[0])
    sag = float(h["max_joint_err_m"][-1, 0])
    assert 0.0 < sag < lim_m, (sag, lim_m)
    assert not (np.bitwise_or.reduce(h["flags"].ravel()) & int(HealthFlags.JOINT_FAIL))
    assert h["severity"].max() == Severity.OK
    assert lim_m >= 1e-3
    # a policy tighter than the realised sag must fire (negative control)
    tight = HealthLoop(_welded_box(2, device), checks=CheckPolicy(level="full", full_every=1, joint_limit_m=1e-6),
                       kind="mujoco", solver_kwargs={"use_mujoco_contacts": True})
    tight.step(50)
    rep = tight.monitor.poll()
    assert rep.flags[0] & int(HealthFlags.JOINT_FAIL) and rep.severity[0] == Severity.DEGRADED


def test_pulled_joint_fires(device: str) -> None:
    scene = pendulum(2, 0.3, device=device)
    loop = HealthLoop(scene)
    loop.step(5)
    rep = loop.monitor.poll()
    assert not (rep.flags & int(HealthFlags.JOINT_FAIL)).any()
    bob = int(world_bodies(scene.model, 1)[0])

    def pull(step: int, state) -> None:
        q = state.body_q.numpy()[bob].copy()
        q[0] += 5e-3  # 5 mm off the hinge: > the 1 mm floor
        set_body(state.body_q, bob, q)

    loop.step(1, inject=pull)
    h = loop.monitor.record.numpy()
    assert h["flags"][1] & int(HealthFlags.JOINT_FAIL) and h["severity"][1] == Severity.DEGRADED
    assert h["max_joint_err_m"][1] == pytest.approx(5e-3, rel=0.05)
    assert not h["flags"][0] & int(HealthFlags.JOINT_FAIL)


@pytest.mark.compile_heavy
def test_lever_arm_weld_droops_and_fires(device: str) -> None:
    """Finding (docs/notes/wp5-results.md): with the weld point 0.54 m from the COM, MuJoCo's coupled soft rows let
    the box droop ~0.18 rad in 0.5 s, far beyond the per-row solref bound; JOINT_FAIL is then a true positive."""
    loop = HealthLoop(_welded_box(1, device, lever=0.2, at_body=False), checks=CheckPolicy(level="full", full_every=1),
                      kind="mujoco", solver_kwargs={"use_mujoco_contacts": True})
    loop.step(400, record=True)
    h = np.stack(loop.history)
    droop = float(h["max_joint_err_rad"][-1, 0])
    assert droop > 0.05
    assert h["flags"][-1, 0] & int(HealthFlags.JOINT_FAIL) and h["severity"][-1, 0] == Severity.DEGRADED
