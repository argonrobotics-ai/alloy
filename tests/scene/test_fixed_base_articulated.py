# SPDX-License-Identifier: Apache-2.0
"""SCR-DC4-2: a `motion: static` MJCF object with nested joints (rack rails, DIMM clips, bay latches) routes STATIC
with the MuJoCo entry. The entry owns and steps its articulated bodies, and a scene without a MuJoCo capability
fails naming the card. The breakaway case follows DC4's tests/tasks/test_dc_rack_checks.py (branch dc4 4b4919d):
a nested slide with frictionloss 2 N breaks away within 1.6x of it under a slow ramp (DC4: 2.88 N)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.errors import RouteError
from alloy.core.registry import isolated_registry
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler

FRICTIONLOSS = 2.0
RAIL = "wp2_test_rail"
RAIL_XML = f"""<mujoco model="rail">
  <worldbody>
    <body name="outer">
      <geom name="{RAIL}_outer" type="box" size="0.3 0.01 0.02" pos="0 -0.25 0"/>
      <body name="slider">
        <inertial pos="0 0 0" mass="4" diaginertia="0.05387 0.12053 0.17333"/>
        <joint name="slide" type="slide" axis="1 0 0" range="0 0.5" frictionloss="{FRICTIONLOSS}"/>
        <geom name="{RAIL}_chassis" type="box" size="0.3 0.2 0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _library(tmp_path):
    try:
        from alloy.cards.library import CardLibrary, default_root
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    (tmp_path / "mjcf").mkdir(exist_ok=True)
    (tmp_path / "mjcf" / "rail.xml").write_text(RAIL_XML)
    (tmp_path / "objects").mkdir(exist_ok=True)
    (tmp_path / "objects" / f"{RAIL}.yaml").write_text(
        f"schema: alloy.object/1\nname: {RAIL}\nshape:\n  mjcf:\n    path: mjcf/rail.xml\n"
        "material: steel_zinc_plated\nmotion: static\n")
    return CardLibrary([tmp_path, default_root()])


def _scene(*extra):
    from alloy.cards.types import Placement, SceneCard

    objs = (Placement(RAIL, (0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0)), *extra)
    return SceneCard(robots=(), objects=objs, control_dt=0.02)


def _registered():
    from alloy.scene.builtin_plugins import register_builtin_solvers

    register_builtin_solvers()


def test_routes_static_with_the_mujoco_entry(tmp_path):
    with isolated_registry(empty=True):
        _registered()
        plan = SceneAssembler(_library(tmp_path)).plan(_scene())
    d = plan.report.decisions[RAIL]
    assert d.route == Route.STATIC and d.entry == "mujoco"
    assert d.numbers["articulated_joints"] == 1.0
    assert any("SCR-DC4-2" in r and "slide" in r for r in d.reasons)
    assert [c.name for c in plan.caps] == ["mujoco"]


def test_plain_static_object_keeps_no_entry(tmp_path):
    from alloy.cards.types import Placement

    with isolated_registry(empty=True):
        _registered()
        plan = SceneAssembler(_library(tmp_path)).plan(_scene(Placement("cube_beech_30", (0.0, 0.5, 0.0155,
                                                                                         0.0, 0.0, 0.0, 1.0))))
    static = [n for n, t in plan.resolved.items() if t.card.motion == "static" and n != RAIL]
    for name in static:
        d = plan.report.decisions[name]
        assert d.entry == "" and d.reasons == ("motion=static: shapes in the shared pipeline, no entry",)


def test_no_mujoco_capability_names_the_card(tmp_path):
    from alloy.route.router import Router
    from alloy.testing.route import context

    asm = SceneAssembler(_library(tmp_path))
    obj = asm.plan(_scene()).resolved[RAIL]
    with pytest.raises(RouteError, match=RAIL):                 # no capability registered
        Router().route(obj, context(frozenset()))
    with pytest.raises(RouteError, match=RAIL):                 # the rigid route served by another solver
        SceneAssembler(_library(tmp_path), Router(entries={Route.RIGID: "featherstone"})).plan(_scene())


def _ramp_breakaway(sim, body: int, joint: int, f_max: float, seconds: float, travel: float) -> float:
    """Force along +x on `body` ramped 0 -> f_max over `seconds`; the force at which the joint moved > `travel`."""
    q_start = sim.model.joint_q_start.numpy()[joint]
    q0 = float(sim.state.joint_q.numpy()[q_start])
    periods = round(seconds / sim.clock.control_dt)
    for k in range(periods):
        f = f_max * (k + 1) / periods
        for st in (sim.state, sim._next):
            w = st.body_f.numpy()
            w[body] = (f, 0.0, 0.0, 0.0, 0.0, 0.0)
            st.body_f.assign(w)
        sim.step_period()
        if float(sim.state.joint_q.numpy()[q_start]) - q0 > travel:
            return f
    return float("nan")


def test_fixed_base_slide_steps_and_breaks_away_at_frictionloss(tmp_path):
    from alloy.scene.articulated import moving_mass
    from alloy.sim import Simulation

    with isolated_registry(empty=True):
        _registered()
        built = SceneAssembler(_library(tmp_path)).assemble(_scene(), "cpu", world_count=1)
        sim = Simulation(built)
        m = built.model
        body = next(i for i, s in enumerate(m.body_label) if s.endswith("slider"))
        joint = next(i for i, s in enumerate(m.joint_label) if s.endswith("slide"))
        (entry,) = built.entries
        assert entry.name == "mujoco" and body in entry.bodies and joint in entry.joints
        rng = built.handles[RAIL].template.body
        assert moving_mass(rng, m.body_mass.numpy(), m.joint_parent.numpy(), m.joint_child.numpy(),
                           m.joint_type.numpy()) == pytest.approx(4.0)       # the slider, not the fixed root
        f = _ramp_breakaway(sim, body, joint, 6.0, 1.0, 1e-3)
        assert np.isfinite(sim.state.body_q.numpy()).all()
    assert FRICTIONLOSS <= f <= 1.6 * FRICTIONLOSS, f

