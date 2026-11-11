# SPDX-License-Identifier: Apache-2.0
"""The CPU route's per-world models when worlds differ in mass (WP3's F5: a per-world density draw kept world 0's
``body_subtreemass``, which MuJoCo C's ``mj_comPos`` divides by, so spinning released bodies diverged)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy._vendor import newton

from alloy.bully.direct import DirectSim
from alloy.contact.light_body import LIGHT_KEY
from alloy.route.mjc_core import MjcCoreSolver
from alloy.route.mjc_core_config import CPU_ROUTE_KEY
from alloy.route.mjc_core_models import subtree_mass
from alloy.scene.builtin_plugins import builtin_capabilities

pytestmark = pytest.mark.cpu_route

TREE = """<mujoco><worldbody>
 <body pos="0 0 1"><freejoint/><geom type="box" size=".1 .05 .02" mass="1.3"/>
  <body pos=".2 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" size=".02 .1" mass=".7"/>
   <body pos=".2 0 0"><joint type="hinge" axis="0 0 1"/><geom type="sphere" size=".03" mass=".25"/></body></body>
  <body pos="-.2 0 0"><joint type="hinge" axis="1 0 0"/><geom type="sphere" size=".04" mass=".4"/></body></body>
 <body pos="1 0 1"><freejoint/><geom type="cylinder" size=".05 .1" mass=".9"/></body>
</worldbody></mujoco>"""

SPIN = """<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="2 2 .1"/>
 <body pos="0 0 .3"><freejoint/><geom type="cylinder" size=".04 .12" mass=".5"/></body>
</worldbody></mujoco>"""


def test_subtree_mass_is_mujocos_sum_bit_for_bit():
    """A branched tree and a second free body: the children-first sum equals mj_setConst's, bit for bit."""
    m = mujoco.MjModel.from_xml_string(TREE)
    assert np.array_equal(subtree_mass(m), m.body_subtreemass)
    np.testing.assert_allclose(subtree_mass(m)[1], 1.3 + 0.7 + 0.25 + 0.4, rtol=1e-15)


def _route(worlds: int, scale: list[float], light: bool = True) -> DirectSim:
    """The spinning cylinder in ``worlds`` worlds, world w's mass and inertia x ``scale[w]`` (a density draw); spun
    at 20 rad/s about a tilted axis and released 0.3 m over the floor. ``light=False`` turns the light-body rule off
    (its rows take world 0's masses in every world, on both routes: ``alloy.contact.light_body``)."""
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(SPIN, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=worlds)
    model = b.finalize(device="cpu")
    k = np.repeat(np.asarray(scale, np.float32), model.body_count // worlds)
    for name, power in (("body_mass", 1), ("body_inertia", 1), ("body_inv_mass", -1), ("body_inv_inertia", -1)):
        arr = getattr(model, name)
        v = arr.numpy()
        arr.assign((v * (k if v.ndim == 1 else k[:, None, None]) ** power).astype(v.dtype))
    kw = {CPU_ROUTE_KEY: "mjc"} if light else {CPU_ROUTE_KEY: "mjc", LIGHT_KEY: False}
    solver = next(c for c in builtin_capabilities() if c.name == "mujoco").factory(model, kw)
    assert isinstance(solver, MjcCoreSolver)
    sim = DirectSim(model, solver, control_dt=0.016, substeps=8, collide=False, export_contacts=False)
    qd = sim.state.joint_qd.numpy().reshape(worlds, 6)
    qd[:, 3:] = np.array([20.0, 5.0, 3.0], np.float32)                       # the free joint's angular velocity
    sim.state.joint_qd.assign(qd.reshape(-1))
    return sim


def test_each_world_carries_its_own_subtree_mass():
    """Worlds 1 and 2 three and half times heavier: each world model's subtree mass is its own body masses' sum;
    world 0's stays the template's bit for bit."""
    sim = _route(3, [1.0, 3.0, 0.5])
    worlds = sim.solver.worlds
    for m in worlds:
        assert np.array_equal(m.body_subtreemass, subtree_mass(m))
    assert np.array_equal(worlds[0].body_subtreemass, sim.solver.mj_model.body_subtreemass)
    np.testing.assert_allclose(worlds[1].body_subtreemass[1], 3.0 * worlds[0].body_subtreemass[1], rtol=1e-6)


def test_a_heavier_world_spins_as_its_own_scene():
    """World 1 (3x the mass) of a two-world scene follows a one-world scene built at that mass bit for bit, in the air
    and after it lands and tumbles (0.8 s): its subtree mass, contact weights and solver scale are its own
    (``mjc_core_consts``). The light-body rule is off here: its rows take world 0's mass in every world. With world 0's
    subtree mass, world 1 spun up to 64 rad/s in the air and went non-finite (WP3's F5); with world 0's contact weights
    it agreed only to 1 mm and 1 % of the spin after landing."""
    two, one = _route(2, [1.0, 3.0], light=False), _route(1, [3.0], light=False)
    for _ in range(50):
        two.step_period()
        one.step_period()
        assert np.array_equal(two.state.body_q.numpy()[1], one.state.body_q.numpy()[0])
        assert np.array_equal(two.state.body_qd.numpy()[1], one.state.body_qd.numpy()[0])
    assert np.isfinite(two.state.body_q.numpy()).all()
