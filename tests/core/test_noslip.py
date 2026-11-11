# SPDX-License-Identifier: Apache-2.0
"""MuJoCo Warp noslip through Newton ``SolverMuJoCo`` (coordinator task 2026-11-01; docs/notes/wp0-noslip.md).

Pinned mujoco-warp 86e2fae0 has the noslip solver that 3.16.0 lacked; Newton ac7f7830 exposes no option for it, so
``newton_compat.enable_mujoco_noslip`` sets it on the built solver. Measured effect (CPU, elliptic cone, box on an
incline with mu = 0.8 > tan 0.4): creep over 1 s after settling 990 um without noslip, 0.69 um with 10 iterations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.core import newton_compat as nc
from alloy.core.errors import SolverOptionError
from alloy.testing.scenes import box_on_plane, incline


def test_mujoco_warp_pin() -> None:
    assert nc.installed_mujoco_warp_commit() == nc.MJWARP_PIN
    assert nc.mujoco_warp_has_noslip()


def _creep(noslip: int, *, mu: float = 0.8, theta: float = 0.4, settle: int = 200, steps: int = 800) -> float:
    """Down-slope displacement [m] of the incline box between ``settle`` and ``settle + steps`` steps of 1/800 s."""
    scene = incline(1, mu, theta, device="cpu")
    runner = scene.runner("mujoco", cone="elliptic")
    nc.enable_mujoco_noslip(runner.solver, noslip, 1e-8)
    runner.step(settle)
    x0 = runner.state.body_q.numpy()[0, :3].copy()
    runner.step(steps)
    d = runner.state.body_q.numpy()[0, :3] - x0
    return float(abs(d @ np.array([math.cos(theta), 0.0, -math.sin(theta)])))


@pytest.mark.compile_heavy
def test_noslip_stops_incline_creep() -> None:
    off, on = _creep(0), _creep(10)
    assert off > 1e-4, f"no creep to remove without noslip: {off:.3g} m"
    assert on < 1e-5, f"noslip left {on:.3g} m of creep (off: {off:.3g} m)"


def test_noslip_sets_both_models() -> None:
    solver = box_on_plane(2, device="cpu").make_solver("mujoco")
    nc.enable_mujoco_noslip(solver, 7, 1e-9)
    assert solver.mj_model.opt.noslip_iterations == 7
    assert solver.mjw_model.opt.noslip_iterations == 7
    assert np.allclose(solver.mjw_model.opt.noslip_tolerance.numpy(), 1e-9)
    nc.enable_mujoco_noslip(solver, 0)
    assert solver.mjw_model.opt.noslip_iterations == 0


def test_noslip_rejects_bad_options() -> None:
    scene = box_on_plane(1, device="cpu")
    with pytest.raises(SolverOptionError):
        nc.enable_mujoco_noslip(scene.make_solver("mujoco"), -1)
    with pytest.raises(SolverOptionError):
        nc.enable_mujoco_noslip(scene.make_solver("mujoco"), 5, 0.0)
    with pytest.raises(SolverOptionError):
        nc.enable_mujoco_noslip(scene.make_solver("mujoco", integrator="discrete"), 5)
