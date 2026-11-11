# SPDX-License-Identifier: Apache-2.0
"""Library load checks for three objects XB4-7 failed (REPORT.md §3, rubric row 8; coordinator 2026-11-06). CPU only.

- hinged_box.xml: without ``<compiler angle="radian"/>`` MJCF read the lid range "0 2" as degrees, and the hinge axis
  swung the lid down into the box (parent and child do not collide). The lid now opens upward, 0 to 2 rad.
- dc_show_cutter and the hinged box: a placement sets the MJCF's frame; the root body keeps its own pose inside it
  (the cutter's lever B has a 7.2-degree yaw, the box sits 30 mm up). XB4-7's check placed the root body instead, so
  it read the cutter's yaw as 4.5 mm of drift and dropped the box 30 mm (its 1.1 mm penetration). Both rest still.
- job3_pot: a static fixture by design (JOB3's bain-marie). A scene with nothing dynamic has nothing to step, so it is
  checked by setting a nut on its floor.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.cards.library import default_root
from alloy.cards.roots import assets_root

needs_assets = pytest.mark.skipif(assets_root() is None, reason="needs the alloy-assets checkout")


def _lid_far_edge_z(q: float) -> float:
    """World z of the lid's free edge (local y = -0.08 from the hinge) at hinge angle ``q``, by forward kinematics."""
    from alloy._vendor import newton
    import warp as wp

    b = newton.ModelBuilder()
    b.add_mjcf(str(default_root() / "mjcf/hinged_box.xml"))
    m = b.finalize(device="cpu")
    joint_q = m.joint_q.numpy()
    joint_q[-1] = q
    state = m.state()
    newton.eval_fk(m, wp.array(joint_q, dtype=wp.float32, device="cpu"), m.joint_qd, state)
    lid = wp.transform(*state.body_q.numpy()[-1].tolist())
    return float(wp.transform_point(lid, wp.vec3(0.0, -0.08, 0.006))[2])


def test_hinged_box_lid_opens_upward_to_two_radians() -> None:
    from alloy._vendor import newton

    b = newton.ModelBuilder()
    b.add_mjcf(str(default_root() / "mjcf/hinged_box.xml"))
    m = b.finalize(device="cpu")
    assert m.joint_limit_lower.numpy()[-1] == pytest.approx(0.0)
    assert m.joint_limit_upper.numpy()[-1] == pytest.approx(2.0)            # 0.0349 (2 degrees) without the compiler
    assert _lid_far_edge_z(1.0) > _lid_far_edge_z(0.0) + 0.05                # opening lifts the free edge


def _settle(scene: alloy.Scene, key: str, periods: int = 25) -> tuple[np.ndarray, np.ndarray, alloy.Simulation]:
    """Build on the CPU; ``key``'s value after the first control period and after ``periods`` more."""
    sim = alloy.build(scene, device="cpu")
    sim.step(1)
    first = np.asarray(sim.state(world=0)[key], dtype=np.float64)
    sim.step(periods)
    return first, np.asarray(sim.state(world=0)[key], dtype=np.float64), sim


def _on_ground(name: str, card: str, **pose: object) -> alloy.Scene:
    scene = alloy.Scene(dt=0.02)
    scene.add_ground()
    scene.add_object(name, card, **pose)
    return scene


def test_hinged_box_rests_closed() -> None:
    first, last, sim = _settle(_on_ground("box", "hinged_box_plywood"), "box.pos")
    assert np.linalg.norm(last - first) < 1.0e-4
    assert abs(float(sim.state(world=0)["box/hinge.q"])) < 0.01              # the closed lid rests on its limit
    assert sim.check()["ok"]


@needs_assets
def test_cutter_rests_without_drift() -> None:
    scene = _on_ground("cutter", "dc_show_cutter", pos=(0.0, 0.0, 0.0025), quat_xyzw=(1.0, 0.0, 0.0, 0.0))
    first, last, sim = _settle(scene, "cutter.pos")                          # XB4-7's stable pose, upside down
    assert np.linalg.norm(last - first) < 5.0e-4                             # 0.03 mm over 0.5 s measured
    assert sim.check()["ok"]


@needs_assets
def test_static_pot_holds_a_nut_on_its_floor() -> None:
    scene = _on_ground("pot", "job3_pot")
    scene.add_object("nut", "nut_m10", pos=(0.0, 0.0, 0.03))
    _, last, sim = _settle(scene, "nut.pos")
    assert sim.routes()["pot"]["route"] == "static"
    assert last[2] == pytest.approx(0.001 + 0.004, abs=5.0e-4)               # on the 1 mm floor, not the ground (4 mm)
    assert sim.check()["ok"]


def test_round_bottom_tube_weighs_its_wall_and_rests_on_its_side() -> None:
    """SCR OSS-9-5: the real 13 x 75 mm round-bottom tube (0.8 mm wall, 15 hulls) replaces the solid cylinder."""
    scene = _on_ground("tube", "sample_tube_13x75", pos=(0.0, 0.0, 0.02), rpy=(1.5707963, 0.0, 0.0))
    first, last, sim = _settle(scene, "tube.pos", periods=100)
    mass = float(sim.engine.model.body_mass.numpy().sum())
    assert sim.routes()["tube"]["route"] == "rigid"
    assert mass == pytest.approx(905.0 * 2.278782e-6, rel=1e-3)               # 2.06 g of PP wall (the cylinder: 9.0 g)
    assert last[2] == pytest.approx(0.00649, abs=3.0e-4)                     # on its side: 6.5 mm, the 13 mm diameter / 2
    assert sim.check()["ok"]


def test_a4_paper_takes_the_film_route_with_its_strain_cap_and_fold_memory() -> None:
    """Sweep finding F9 (combo matrix 11-07): paper_a4, compose's "a sheet of paper", had no route override and went to
    VBD cloth (31 s of CPU for 0.2 s). It now takes SOFT-1's paper route: cloth_film at ~15 mm cells (stiff paper
    freezes on fine film cells), a 0.9 % strain cap (35 MPa / 4 GPa) and printing paper's 36° fold memory."""
    from alloy.cards.library import CardLibrary
    from alloy.core.registry import isolated_registry, registered_routes
    from alloy.route.router import Router
    from alloy.route.types import SceneContext
    from alloy.scene.builtin_plugins import register_builtin_solvers
    from alloy.solvers.film.emit import spec_from_card

    lib = CardLibrary(assets=None)
    obj = lib.resolve(lib.object("paper_a4"))
    with isolated_registry(empty=True):
        register_builtin_solvers()
        ctx = SceneContext(rigid_partners={"paper_a4": ()}, free_bodies_per_world=1, registered_routes=registered_routes())
        assert str(Router().route(obj, ctx).route) == "cloth_film"
    spec = spec_from_card(obj)
    assert spec.strain_cap == pytest.approx(35e6 / 4e9) and spec.thickness == pytest.approx(1e-4)
    assert spec.extras.yield_angle == pytest.approx(0.63)
    assert 0.010 < np.sqrt(0.21 * 0.297 / obj.card.budget.particles) < 0.016        # cell size on the A4 grid
