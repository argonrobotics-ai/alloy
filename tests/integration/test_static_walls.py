# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB2-5: in a coupled scene a static shape is a wall for every particle entry (CPU).

JOB2's spill fell through the yam_bench plywood: the plywood sits on the robot's base body (welded to the world,
owned by the MuJoCo entry) and the SPH entry's proxies are the parts only, so Newton's entry view hid the plywood.
``static_wall_like``: a welded bench with a plywood sheet, a jar + lid proxied to the SPH entry, a free cup that is not
proxied, and liquid dropped onto the plywood.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton

import alloy.coupling.scheduler  # noqa: F401 - imported before the empty registry, as in a full run
from alloy.core.registry import isolated_registry
from alloy.coupling.estimators import register_builtin_estimators
from alloy.core.newton_compat import SolverCoupled
from alloy.coupling.kinematic import world_welded_bodies
from alloy.coupling.static_walls import static_wall_shapes
from alloy.integration import build_simulation
from alloy.testing import integration as I

pytestmark = [pytest.mark.compile_heavy, pytest.mark.integration, pytest.mark.timeout(600)]

F = newton.ShapeFlags
R = 2.0e-3                                        # liquid particle radius [m]


def _sim(worlds: int = 2):
    with isolated_registry(empty=True):
        from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

        register_builtin_solvers()
        register_builtin_estimators()
        built = I.static_wall_like(worlds, "cpu")
        return built, build_simulation(built, "standard", graph="off")


def _shapes(model, name: str) -> list[int]:
    return [i for i, x in enumerate(model.shape_label) if str(x).rsplit("/", 1)[-1] == name]


@pytest.fixture(scope="module")
def run():
    built, sim = _sim()
    view = sim.stepper._entries["water"].view
    flags = {n: view.shape_flags.numpy()[_shapes(built.model, n)].tolist() for n in ("plywood", "jar", "cup")}
    mj = sim.stepper._entries["mujoco"].view.shape_flags.numpy()[_shapes(built.model, "plywood")].tolist()
    for _ in range(20):                               # 0.2 s: the liquid falls 1 cm onto the plywood
        sim.step_period()
    return built, sim, flags, mj


def test_newton_still_has_the_view_hooks():
    """The Scheduler overrides these private Newton methods; a rename upstream would silently drop the walls."""
    for name in ("_entry_visible_shapes", "_customize_compact_view"):
        assert callable(getattr(SolverCoupled, name, None)), name


def test_static_bodies_are_world_welded():
    b = newton.ModelBuilder()
    base = b.add_link(label="base")
    arm = b.add_link(label="arm")
    tip = b.add_link(label="tip")
    free = b.add_body(label="free")
    b.add_articulation([b.add_joint_fixed(-1, base), b.add_joint_fixed(base, arm), b.add_joint_revolute(arm, tip)])
    for body in (base, arm, tip, free):
        b.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
    m = b.finalize(device="cpu")
    assert set(map(int, world_welded_bodies(m).nonzero()[0])) == {base, arm}
    assert static_wall_shapes(m) == [0, 1]                      # default shapes collide with particles


def test_static_wall_reaches_the_particle_entry_as_a_wall_only(run):
    built, sim, flags, mj = run
    wall = int(F.COLLIDE_PARTICLES)
    assert all(f & wall and not f & int(F.COLLIDE_SHAPES) for f in flags["plywood"]), flags   # a wall, not a collider
    assert all(f & wall for f in flags["jar"]), flags                                       # proxied: as before
    assert not any(f & wall for f in flags["cup"]), flags                                   # dynamic, not proxied
    assert all(f & wall and f & int(F.COLLIDE_SHAPES) for f in mj), mj                     # the owner: unchanged
    assert sorted(static_wall_shapes(built.model)) == _shapes(built.model, "plywood")


def test_liquid_rests_on_the_welded_plywood(run):
    built, sim, _, _ = run
    z = sim.state.particle_q.numpy()[:, 2]
    assert np.isfinite(z).all()
    I.record("WP12-JOB2-5 wall", float(z.min()), "pass" if z.min() > -R else "fail",
             "PYTHONPATH=. python -m pytest tests/integration/test_static_walls.py -q", min_z=float(z.min()),
             mean_z=float(z.mean()), particles=int(z.size), worlds=2)
    assert z.min() > -R, z.min()                     # on the plywood (top z = 0), not through it
    assert z.max() < 0.03                            # and settled, not bounced off


def test_no_static_wall_in_a_scene_without_welded_bodies():
    built = I.job2_like(1, "cpu")                    # jar and lid free, no welded body
    assert static_wall_shapes(built.model) == []
