# SPDX-License-Identifier: Apache-2.0
"""§7 WP2 acceptance: an arm + box + cloth scene card (the arms come from WP13's vendored bench MJCF) assembles and
steps 20 control periods on CPU with 4 worlds; everything stays finite and on the bench."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.cards.types import Placement, RobotRef, SceneCard
from alloy.core.registry import isolated_registry
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.testing import scene as S

UP = (0.0, 0.0, 0.0, 1.0)
PERIODS = 20
WORLDS = 4


def _library():
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    lib = CardLibrary()
    if not lib.asset_path("robots/yam_bench/robot.xml").exists():
        pytest.skip("WP13 bench MJCF not vendored")
    return lib


def arm_box_cloth_card() -> SceneCard:
    robot = RobotRef(path="robots/yam_bench/robot.xml", kind="mjcf", base_pose=(0.0, 0.0, 0.0, *UP), name="yam_bench")
    objs = (Placement("cube_beech_30", (0.35, 0.15, 0.0155, *UP)),
            Placement("towel_cotton", (0.35, -0.15, 0.002, *UP)))
    return SceneCard(robots=(robot,), objects=objs, worlds=WORLDS)


@pytest.mark.compile_heavy
@pytest.mark.timeout(1500)
def test_arm_box_cloth_steps_20_periods_on_cpu():
    lib = _library()
    with isolated_registry(empty=True):
        built = SceneAssembler(lib).assemble(arm_box_cloth_card(), "cpu", world_count=WORLDS)
        routes = {n: h.decision.route for n, h in built.handles.items()}
        assert routes["cube_beech_30"] == Route.RIGID and routes["towel_cotton"] == Route.CLOTH_VBD
        assert built.world_count == WORLDS and {e.capability for e in built.entries} == {"mujoco", "vbd"}
        sim = Simulation(built)
        for _ in range(PERIODS):
            sim.step_period()
        q, p = sim.state.body_q.numpy(), sim.state.particle_q.numpy()
    finite = bool(np.isfinite(q).all() and np.isfinite(p).all())
    on_bench = bool(finite and q[:, 2].min() > -0.01 and p[:, 2].min() > -0.01)
    S.record("WP2.arm_box_cloth", float(finite and on_bench), "pass" if finite and on_bench else "fail",
             worlds=WORLDS, periods=PERIODS, coupler=built.coupler.kind, device="cpu",
             min_body_z=float(np.nanmin(q[:, 2])), min_particle_z=float(np.nanmin(p[:, 2])))
    assert finite and on_bench
