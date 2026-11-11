# SPDX-License-Identifier: Apache-2.0
"""``build_simulation(built, profile, plugins=[...])``: plug-ins run in the wired stepper (JOB3's ask, 2026-11-06).

Before, an engine-layer caller had to patch ``wiring.make_stepper`` to install a ``PluginHost`` (species/build.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import warp as wp

from alloy.core.registry import isolated_registry
from alloy.integration import build_simulation
from alloy.plugins import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench
from alloy.testing import integration as I
from alloy.testing import scene as S

pytestmark = [pytest.mark.integration]


@wp.kernel
def _push_up(body: wp.array[wp.int32], f: wp.float32, body_f: wp.array[wp.spatial_vector]):
    add_wrench(body_f, body[wp.tid()], wp.spatial_vector(0.0, 0.0, f, 0.0, 0.0, 0.0))


class _Lift(ForcePlugin):
    """A constant upward force on one body [N]."""

    name = "test_lift"
    summary = "constant upward force (test)"

    @dataclass
    class Config:
        body: str = "box"
        force: float = 0.0

    def setup(self, ctx):
        self.body = ctx.bodies(self.config.body)

    def apply(self, state, forces, dt):
        wp.launch(_push_up, dim=self.ctx.world_count, inputs=[self.body, float(self.config.force)],
                  outputs=[forces.body_f], device=self.ctx.device)


def _box_z(plugins=()) -> tuple[float, object]:
    card, lib = S.box_on_table(worlds=1)
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 1)
        weight = float(built.model.body_mass.numpy().max()) * 9.81
        made = [p(weight) for p in plugins]
        sim = build_simulation(built, "standard", graph="off", health=False, plugins=made)   # the push adds energy
        z0 = float(sim.state.body_q.numpy()[:, 2].max())
        for _ in range(30):
            sim.step_period()
        return float(sim.state.body_q.numpy()[:, 2].max()) - z0, sim


def test_a_plugin_runs_in_the_wired_stepper() -> None:
    rest, plain = _box_z()
    lifted, sim = _box_z([lambda w: (lambda built: _Lift(body="box", force=3.0 * w))])   # a function of the scene
    assert abs(rest) < 3e-3 and lifted > 0.01                       # 2 g net upward: it leaves the table
    assert sorted(sim.plugins.plugins) == ["test_lift"] and not hasattr(plain, "plugins")


def test_an_entry_that_is_not_a_plugin_raises() -> None:
    card, lib = S.box_on_table(worlds=1)
    with isolated_registry(empty=True), pytest.raises(PluginError, match="plugins entry"):
        build_simulation(I.assemble(card, lib, "cpu", 1), "standard", graph="off", plugins=[lambda built: 3])
