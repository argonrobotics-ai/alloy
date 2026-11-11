# SPDX-License-Identifier: Apache-2.0
"""Implicit MPM: world 0 does not feel the other worlds (SCR-WP13-MPMISO-1; alloy/integration/mpm_worlds.py).

Newton's rheology loop stops only when every world has converged, and its dense grid takes its bounds from every
world's particles. Softening worlds 1.. (a melt: E 1e4, mu 1.5 Pa s, no yield or friction) moved world 0 by 1.66e-7 m in
melt_chocolate's scene (CPU, 2 worlds, 6 periods). With mpm_worlds each world's solve stops at its own convergence and
a multi-world dense grid holds its bounds, so world 0's trajectory is bitwise the same whatever the others do.

The scene: a 16 x 16 x 6 particle block (1.25 mm spacing, 2.5 mm voxels) on a plate, CPU, 3 periods (~4 s a run).
"""

from __future__ import annotations

import contextlib
import inspect

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp
from alloy._vendor.newton._src.solvers.implicit_mpm import solve_rheology as SR
from alloy._vendor.newton._src.solvers.implicit_mpm import solver_implicit_mpm as SIM

from alloy.integration import mpm_worlds
from alloy.plugins.harness.scene import TestScene, build_simulation

SPACING, COUNTS = 1.25e-3, (16, 16, 6)
VOXEL = 2.0 * SPACING
SOFT = (("viscosity", 1.5), ("yield_stress", 0.0), ("friction", 0.0), ("young_modulus", 1.0e4))
PERIODS = 3


def _build(b: newton.ModelBuilder) -> None:
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0025), wp.quat_identity()), hx=0.04, hy=0.04,
                    hz=0.0025, label="plate")
    newton.solvers.SolverImplicitMPM.register_custom_attributes(b)
    mass, lo = 1000.0 * SPACING ** 3, -0.5 * COUNTS[0] * SPACING
    for i in range(COUNTS[0]):
        for j in range(COUNTS[1]):
            for k in range(COUNTS[2]):
                pos = wp.vec3(lo + SPACING * (i + 0.5), lo + SPACING * (j + 0.5), 0.005 + SPACING * (k + 0.5))
                b.add_particle(pos, wp.vec3(0.0, 0.0, 0.0), mass, 0.5 * SPACING)


def _run(worlds: int, grid: str, soften: bool, device: str = "cpu") -> np.ndarray:
    """Particle positions [worlds, n, 3] after PERIODS; ``soften`` melts worlds 1.."""
    scene = TestScene(build=_build, worlds=worlds, periods=PERIODS, control_dt=0.02, dt=2.5e-3, solver="mpm",
                      solver_kwargs={"voxel_size": VOXEL, "grid_type": grid})
    sim = build_simulation(scene, [], device, worlds)
    if soften:
        mpm, w = sim.model.mpm, sim.model.particle_world.numpy()
        for name, value in SOFT:
            a = getattr(mpm, name).numpy()
            a[w >= 1] = value
            getattr(mpm, name).assign(a)
    for _ in range(PERIODS):
        sim.step_period()
    return sim.state.particle_q.numpy().reshape(worlds, -1, 3).copy()


@contextlib.contextmanager
def _newtons_own() -> None:
    """Newton's loop and grid, as before mpm_worlds."""
    mpm_worlds.ENABLED = False
    mpm_worlds.uninstall()
    try:
        yield
    finally:
        mpm_worlds.ENABLED = True
        mpm_worlds.install()


@pytest.mark.parametrize("worlds", [2, 4])
def test_world_0_does_not_feel_the_others(worlds: int, caplog: pytest.LogCaptureFixture) -> None:
    """CPU: Newton builds per-world sparse grids on CUDA only, so separate worlds take the dense grid. The melt stays
    inside the held box (a box regrown for every world would couple them by an ulp: the known gap)."""
    grid = "dense"
    a, b = _run(worlds, grid, False), _run(worlds, grid, True)
    assert "left the held box" not in caplog.text
    assert np.abs(a[1:] - b[1:]).max() > 1.0e-4                    # the others did melt
    np.testing.assert_array_equal(a[0], b[0])


def test_without_mpm_worlds_world_0_feels_them() -> None:
    """The canary: Newton's own loop and dense grid let the melt next door move world 0."""
    with _newtons_own():
        a, b = _run(2, "dense", False), _run(2, "dense", True)
    assert np.abs(a[0] - b[0]).max() > 0.0


def test_one_world_is_newtons_own_run() -> None:
    with _newtons_own():
        ref = _run(1, "dense", False)
    np.testing.assert_array_equal(_run(1, "dense", False), ref)


def test_the_held_dense_box_keeps_within_the_dense_budget() -> None:
    """The margin shrinks so the box (all worlds) fits mpm_grid's dense budget, which raises on several CPU worlds."""
    lo, hi = np.array([-4.0, -4.0, 2.0]), np.array([4.0, 4.0, 5.0])       # the test block: 8 x 8 x 3 voxels
    assert np.array_equal(mpm_worlds._with_margin(lo, hi, 2, 100_000)[1], hi + mpm_worlds.MARGIN)
    blo, bhi = mpm_worlds._with_margin(lo, hi, 64, 4 * 64 * 1536)
    assert 0 < (bhi - hi)[0] < mpm_worlds.MARGIN and np.prod(bhi - blo) * 64 <= 4 * 64 * 1536


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def test_the_newton_internals_it_relies_on() -> None:
    """A Newton bump that moves any of these fails here, not silently (WP12's condition for the wrap)."""
    assert _params(mpm_worlds._PREV.get("loop", SR._run_solver_loop)) == [
        "rheology_solver", "contact_solver", "max_iterations", "tolerance", "l2_tolerance_scale", "use_graph",
        "verbose", "temporary_store"]
    assert _params(mpm_worlds._PREV.get("solve", SIM.SolverImplicitMPM._solve_rheology)) == [
        "self", "scratch", "rigidity_operator"]
    grid = _params(mpm_worlds._PREV.get("grid", SIM.SolverImplicitMPM._allocate_grid))
    assert grid[:6] == ["self", "positions", "particle_flags", "voxel_size", "temporary_store", "padding_voxels"]
    loop = inspect.getsource(mpm_worlds._PREV.get("loop", SR._run_solver_loop))
    assert "solve_granularity = 5" in loop and "contact_solver.solve()" in loop and "rheology_solver.solve()" in loop
    assert "residual[0, batch] < residual_threshold * scale * scale" in inspect.getsource(SR.update_batched_condition)
    assert "self.delta_stress" in inspect.getsource(SR._RheologySolver.__init__)
    assert {"stress", "strain_environment_offsets"} <= set(SR.RheologyData.__dataclass_fields__)
    assert {"collider_impulse", "collider_velocities", "has_colliders"} <= set(SR.CollisionData.__dataclass_fields__)
    pad = inspect.getsource(SIM.ImplicitMPMScratchpad.__init__)
    assert "self.velocity_environment_offsets" in pad and "self.collider_environment_offsets" in pad
    alloc = inspect.getsource(mpm_worlds._PREV.get("grid", SIM.SolverImplicitMPM._allocate_grid))
    assert "np.floor(bbox_min / voxel_size) - padding_voxels" in alloc and "env_count=self._environment_count" in alloc
    assert hasattr(SR, "_ScopedDisableGC")
