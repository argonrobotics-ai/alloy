# SPDX-License-Identifier: Apache-2.0
"""``Scene.add_rod`` (SCR-WP9-ROD R2), on CPU.

- The card: a curve with a radius; `segments` becomes `features.chain_links`.
- The rod routes as the router says: CHAIN today. `_rods.DEFAULT_ROUTE = "rod_vbd"` (SCR-WP9-ROD's flip) writes a
  ROD_VBD override, and the `vbd` entry then steps with SolverVBDRod mixed in.
- A clamp holds the end segments exactly, and needs ROD_VBD.
- `"<rod>.points"` is the centre line on both routes.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api import _rods
from alloy.api.errors import SceneError
from alloy.solvers.rod import SolverVBDRod


@pytest.fixture
def rod_vbd(monkeypatch):
    monkeypatch.setattr(_rods, "DEFAULT_ROUTE", "rod_vbd")


def _scene(**kw) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_rod("cable", length=0.5, radius=0.002, pos=(0.0, 0.0, 0.6), material="nylon_pa6", **kw)
    return sc


def test_the_item_is_a_curve_card_and_round_trips():
    sc = alloy.Scene()
    sc.add_rod("wire", [(0, 0, 0), (0.2, 0, 0), (0.2, 0.1, 0)], radius=0.001, segments=12, clamp="end")
    it = sc.items["wire"]
    assert it["kind"] == "rod" and it["shape"]["curve"]["radius"] == 0.001
    assert it["card_extra"]["features"] == {"chain_links": 12}
    assert it["card_extra"]["route"]["route"] == "rod_vbd"                        # a clamp asks for the elastic rod
    free = alloy.Scene()
    free.add_rod("free", length=0.1, radius=0.001)
    assert "route" not in free.items["free"]["card_extra"]                       # unclamped: the router decides
    assert it["volume"] == pytest.approx(np.pi * 1e-6 * 0.3, rel=1e-12, abs=0)
    assert alloy.Scene.from_dict(sc.to_dict()).items["wire"]["rod"] == {"clamp": "end"}


@pytest.mark.parametrize(("kw", "field"), [({"points": [(0, 0, 0), (1, 0, 0)], "length": 1.0}, "length"),
                                          ({}, "length"), ({"length": 0.5, "clamp": "middle"}, "clamp"),
                                          ({"length": 0.5, "segments": 2}, "segments"),
                                          ({"points": [(0, 0, 0), (0, 0, 0)]}, "points")])
def test_bad_arguments_name_their_fix(kw, field):
    with pytest.raises(SceneError) as err:
        alloy.Scene().add_rod("r", radius=0.002, **kw)
    assert err.value.field == field and err.value.fix


def test_the_default_route_is_the_routers_chain_and_points_read_on_it():
    sim = alloy.build(_scene(), device="cpu")
    assert sim.routes()["cable"]["route"] == "chain"
    p = sim.state(["cable"])["cable.points"]
    assert p.shape[0] == 1 and p.shape[2] == 3
    assert p[0, 0] == pytest.approx([0.0, 0.0, 0.6], abs=1e-6) and p[0, -1] == pytest.approx([0.5, 0.0, 0.6], abs=1e-6)
    sim.close()


def test_a_clamp_asks_for_the_elastic_rod_and_says_when_it_cannot_have_it():
    """OSS-3's review: a clamped rod routes to rod_vbd (only the elastic rod holds an end); with a card override that
    sends it elsewhere, the build names the fix in plain words."""
    sc = _scene(clamp="start")
    assert sc.items["cable"]["card_extra"]["route"]["route"] == "rod_vbd"
    sc.items["cable"]["card_extra"]["route"] = {"route": "chain", "reason": "test"}
    with pytest.raises(SceneError) as err:
        alloy.build(sc, device="cpu")
    assert err.value.field == "clamp" and "SCR" not in err.value.fix


def test_rods_refuse_per_world_starts_and_saving_warns_about_clamps(tmp_path, caplog):
    sc = alloy.Scene(worlds=2)
    with pytest.raises(SceneError) as err:
        sc.add_rod("r", length=0.2, radius=0.002, pos=[(0, 0, 0.1), (0, 0, 0.2)])
    assert err.value.field == "pos"
    sc.add_rod("r", length=0.2, radius=0.002, pos=(0, 0, 0.1), clamp="start")
    with pytest.raises(SceneError):
        sc.randomize("r", "x", -0.01, 0.01)
    caplog.set_level("WARNING", logger="alloy.api")
    sc.save(tmp_path / "cards")
    assert any("do not keep clamp= for r;" in r.getMessage() for r in caplog.records)


@pytest.mark.law
def test_rod_vbd_steps_on_the_rod_solve_and_holds_its_clamp(rod_vbd):
    sim = alloy.build(_scene(clamp="start"), device="cpu")
    assert sim.routes()["cable"]["route"] == "rod_vbd"
    assert isinstance(sim.engine.stepper.solver, SolverVBDRod)
    p0 = sim.state(["cable"], world=0)["cable.points"]
    sim.step(50)
    p1 = sim.state(["cable"], world=0)["cable.points"]
    assert p1.shape == p0.shape and np.isfinite(p1).all()
    assert np.abs(p1[0] - p0[0]).max() < 1e-6                      # the clamped end does not move
    assert p1[-1, 2] < p0[-1, 2] - 0.01                            # the free end falls under its weight
    lengths = np.linalg.norm(np.diff(p1, axis=0), axis=1).sum()
    assert lengths == pytest.approx(0.5, rel=1e-3, abs=0)          # inextensible to 0.1 %
    assert sim.check()["ok"]
    sim.close()


@pytest.mark.law
def test_a_held_end_torque_twists_the_rod_t_l_over_gj(rod_vbd):
    """sim.apply_wrench at the free end of a clamped rod, gravity off: the summed joint twist settles at T L / (G J)
    within 1 % (G J of nylon PA6 from E and nu); the wrench stays until changed, and zeros release it."""
    sc = alloy.Scene(dt=0.01)
    sc.add_rod("cable", length=0.4, radius=0.002, pos=(0.0, 0.0, 0.5), material="nylon_pa6", segments=12,
               clamp="start")
    sim = alloy.build(sc, device="cpu")
    sim.engine.model.gravity.assign(np.zeros((1, 3), np.float32))
    d = sim.built.report.decisions["cable"].numbers
    T = 2e-3
    sim.apply_wrench("cable", torque=(T, 0.0, 0.0), at="end")
    sim.step(300)
    tw = sim.state(["cable"], world=0)["cable.twist"]
    assert tw.shape == (11,)
    assert tw.sum() == pytest.approx(T * d["rod_length"] / d["GJ"], rel=0.01, abs=0)
    sim.apply_wrench("cable", at="end")
    sim.step(300)
    assert abs(sim.state(["cable"], world=0)["cable.twist"].sum()) < 1e-3 * T * d["rod_length"] / d["GJ"]
    assert sim.check()["ok"]
    with pytest.raises(alloy.UnknownNameError):
        sim.apply_wrench("nothing", torque=(T, 0, 0))
    sim.close()


def test_a_cli_recipe_adds_a_rod():
    """OSS-15's recipe adder: {"add": "rod", ...} calls Scene.add_rod with the object's keys."""
    from alloy.cli.scenes import to_scene

    sc = to_scene({"objects": [{"add": "rod", "name": "cord", "length": 0.3, "radius": 0.002, "clamp": "start"}]})
    assert sc.items["cord"]["kind"] == "rod" and sc.items["cord"]["rod"] == {"clamp": "start"}


def test_a_held_push_is_healthy_and_reset_removes_it(rod_vbd):
    """OSS-3's notes: a held force on a free rod trips no health flag; sim.reset() removes every wrench."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_rod("cable", length=0.3, radius=0.003, pos=(-0.15, 0.0, 0.003), material="nylon_pa6", segments=8)
    sim = alloy.build(sc, device="cpu")
    x0 = sim.state(["cable"], world=0)["cable.points"][:, 0].mean()
    sim.apply_wrench("cable", force=(0.05, 0.0, 0.0), at="start")
    sim.step(100)
    assert sim.state(["cable"], world=0)["cable.points"][:, 0].mean() > x0 + 0.005
    assert sim.check()["ok"]
    sim.reset()
    assert not sim.engine.state.body_f.numpy().any() and not sim.engine._next.body_f.numpy().any()
    sim.close()


@pytest.mark.timeout(300)
def test_a_wrench_reaches_a_rigid_body_in_a_coupled_scene():
    """OSS-3's note: a cloth beside a MuJoCo box makes the scene coupled; the box still feels its wrench."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_box("box", size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05), mass=1.0, friction=0.01)
    sc.add_cloth("sheet", size=(0.2, 0.2), pos=(1.0, 0.0, 0.3), resolution=6, mass=0.02, self_contact=False)
    sim = alloy.build(sc, device="cpu")
    assert sim.routes()["box"]["solver"] == "mujoco" and sim.routes()["sheet"]["solver"] == "vbd"
    p0 = sim.state(["box"], world=0)["box.pos"].copy()
    sim.apply_wrench("box", force=(2.0, 0.0, 0.0))
    sim.step(5)
    assert sim.state(["box"], world=0)["box.pos"][0] - p0[0] > 0.8 * 0.5 * 2.0 * 0.05 ** 2    # F/m t^2/2, low friction
    sim.close()
