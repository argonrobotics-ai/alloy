# SPDX-License-Identifier: Apache-2.0
"""API tests of per-world robot / gravity randomization: Scene-side errors and round trip, build-side errors, the
reset hook contract, draw determinism and the process split. Physics laws are in test_robot_random_laws.py."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import alloy.api as alloy
from alloy.cards.library import default_root

RIG = """<mujoco><compiler angle="radian"/><worldbody>
 <body name="mass" pos="0 0 0.5"><joint name="lift" type="slide" axis="0 0 1" range="-0.4 0.4" damping="5" armature="0.1"/>
  <geom type="box" size="0.05 0.05 0.05" mass="2" contype="0" conaffinity="0"/></body></worldbody>
 <actuator><position joint="lift" kp="400" kv="40"/></actuator></mujoco>"""
PASSIVE = RIG.split("<actuator>")[0] + "</mujoco>"
TWO_LINK = """<mujoco><compiler angle="radian"/><worldbody>
 <body name="a" pos="0 0 0.5"><joint name="ja" type="slide" axis="0 0 1" range="-0.4 0.4" damping="5"/>
  <geom type="box" size="0.05 0.05 0.05" mass="2" contype="0" conaffinity="0"/>
  <body name="b" pos="0.2 0 0"><joint name="jb" type="slide" axis="1 0 0" range="-0.1 0.1" damping="5"/>
   <geom type="box" size="0.05 0.05 0.05" mass="1" contype="0" conaffinity="0"/></body></body></worldbody>
 <actuator><position joint="ja" kp="400" kv="40"/><position joint="jb" kp="400" kv="40"/></actuator></mujoco>"""


def rig_scene(worlds: int = 1, mjcf: str = RIG) -> alloy.Scene:
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_robot("rig", mjcf=mjcf, pos=(0, 0, 0))
    return scene


def refused(call, match: str, *fix_words: str) -> alloy.SceneError:
    with pytest.raises(alloy.SceneError, match=match) as err:
        call()
    assert err.value.fix and all(w in err.value.fix for w in fix_words), err.value.fix
    return err.value


# --- Scene side -----------------------------------------------------------------------------------------------


def test_unknown_robot_fields_name_the_field_to_use():
    scene = rig_scene()
    for bad, good in (("mass", "mass_scale"), ("mass", "mass_add"), ("kp", "kp_scale"), ("damping", "kd_scale"),
                      ("mu", "friction")):
        err = refused(lambda bad=bad: scene.randomize("rig", bad, 1, 2), "cannot randomize")
        assert good in err.message and err.field == "field"
    refused(lambda: scene.randomize("rig", "nonsense", 1, 2), "cannot randomize", "mass_scale", "kp_scale")
    assert scene.robot_random == []


def test_factor_range_must_stay_positive_and_friction_not_negative():
    scene = rig_scene()
    for field in ("mass_scale", "kp_scale", "kd_scale", "armature_scale"):
        err = refused(lambda field=field: scene.randomize("rig", field, 0.0, 2.0), "above 0", "(0.8, 1.2)")
        assert err.field == "lo"
    refused(lambda: scene.randomize("rig", "friction", -0.1, 1.0), "cannot be negative", "lo >= 0")
    refused(lambda: scene.randomize("rig", "armature", -0.1, 1.0), "cannot be negative")
    refused(lambda: scene.randomize("rig", "mass_scale", 2.0, 0.5), "empty", "lo <= hi")
    assert scene.robot_random == []


def test_when_must_be_build_or_reset_and_reset_is_for_robot_fields():
    scene = rig_scene()
    scene.add_box("box", size=(0.1, 0.1, 0.1), pos=(1, 0, 0.1), mass=1.0)
    err = refused(lambda: scene.randomize("rig", "mass_scale", 0.5, 2, when="sometimes"), "not known", "reset")
    assert err.field == "when"
    err = refused(lambda: scene.randomize("box", "density", 500, 900, when="reset"), "robot fields and gravity",
                  "drop when=")
    assert err.field == "when" and scene.robot_random == []
    scene.randomize("rig", "mass_scale", 0.5, 2, when="reset")
    assert scene.robot_random[0]["when"] == "reset"


def test_gravity_axis_must_be_x_y_or_z():
    scene = rig_scene()
    err = refused(lambda: scene.randomize("gravity", "w", -10, -9), "cannot randomize", "'z'")
    assert err.field == "field" and scene.robot_random == []
    scene.randomize("gravity", "z", -10, -9)
    assert scene.robot_random == [{"target": "gravity", "field": "z", "lo": -10.0, "hi": -9.0, "when": "build"}]


def test_object_and_free_base_rows_still_go_the_old_way():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("box", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.1), mass=1.0)
    scene.add_robot("free", mjcf='<mujoco><worldbody><body name="b"><freejoint/><geom type="box" size="0.05 0.05 '
                    '0.05" mass="1"/></body></worldbody></mujoco>', pos=(1, 0, 0.1))
    scene.randomize("box", "density", 500, 900)
    scene.randomize("box", "friction", 0.4, 0.8)
    scene.randomize("free", "x", -0.02, 0.02)
    assert scene.robot_random == []
    assert [r["field"] for r in scene.items["box"]["randomize"]] == ["material.density", "material.surface.mu_static"]
    assert scene.items["free"]["randomize"] and scene.items["free"]["randomize"][0]["hi"] == 0.02


def test_an_object_named_gravity_keeps_being_that_object():
    scene = alloy.Scene(dt=0.01)
    scene.add_box("gravity", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5), mass=1.0)
    scene.randomize("gravity", "x", -0.01, 0.01)
    scene.randomize("gravity", "density", 500, 900)
    assert scene.robot_random == [] and len(scene.items["gravity"]["randomize"]) == 2


# --- round trip ----------------------------------------------------------------------------------------------


def test_robot_rows_survive_json_and_scenes_without_them_have_no_key():
    scene = rig_scene(2)
    assert "robot_random" not in scene.to_dict() and '"robot_random"' not in scene.to_json()
    scene.randomize("rig", "mass_scale", 0.5, 2.0)
    scene.randomize("rig/lift", "kp_scale", 0.8, 1.2, when="reset")
    scene.randomize("gravity", "z", -10.3, -9.3)
    back = alloy.Scene.from_dict(json.loads(scene.to_json()))
    assert back.robot_random == scene.robot_random and len(back.robot_random) == 3
    assert back.to_json() == scene.to_json()
    assert alloy.Scene.from_dict(json.loads(rig_scene().to_json())).robot_random == []


# --- build-side errors ---------------------------------------------------------------------------------------


@pytest.mark.compile_heavy
def test_unknown_link_or_joint_lists_the_valid_names():
    for field, valid in (("mass_add", "rig/mass"), ("kp_scale", "rig/lift"), ("armature_scale", "rig/lift")):
        scene = rig_scene()
        scene.randomize("rig/nope", field, 0.5 if field != "mass_add" else 0.1, 1.5)
        with pytest.raises(alloy.UnknownNameError, match="nope") as err:
            alloy.build(scene, device="cpu")
        assert valid in err.value.fix


@pytest.mark.compile_heavy
def test_build_refuses_unservoed_joint_nonpositive_mass_and_unknown_robot():
    scene = rig_scene(mjcf=PASSIVE)
    scene.randomize("rig", "kp_scale", 0.5, 2.0)
    with pytest.raises(alloy.BuildError, match="no joint servo") as err:
        alloy.build(scene, device="cpu")
    assert "<position" in err.value.fix
    scene = rig_scene()
    scene.randomize("rig/mass", "mass_add", -2.5, 5.0)     # the link weighs 2 kg: refused at the low end, any draw
    with pytest.raises(alloy.BuildError, match="zero or less") as err:
        alloy.build(scene, device="cpu")
    assert "mass_add" in err.value.fix
    scene = rig_scene()
    scene.randomize("rig", "mass_scale", 0.5, 2.0)
    scene.robot_random[0]["target"] = "ghost"
    with pytest.raises(alloy.BuildError, match="not a robot"):
        alloy.build(scene, device="cpu")


@pytest.mark.compile_heavy
def test_gravity_with_cloth_is_refused_at_build():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("sheet", size=(0.1, 0.1), pos=(0, 0, 0.1), resolution=4, mass=0.01)
    scene.add_box("weight", size=(0.1, 0.1, 0.1), pos=(1, 0, 0.1), mass=1.0)
    scene.randomize("gravity", "z", -10.0, -9.0)
    with pytest.raises(alloy.BuildError, match="one gravity for all worlds") as err:
        alloy.build(scene, device="cpu")
    assert "rigid" in err.value.fix


# --- feature off ---------------------------------------------------------------------------------------------


def plain_scene() -> alloy.Scene:
    scene = rig_scene(2)
    scene.add_ground()
    scene.add_box("box", size=(0.1, 0.1, 0.1), pos=(1, 0, 0.5), mass=1.0)
    return scene


@pytest.mark.compile_heavy
def test_no_rows_means_no_hooks_no_draws_and_an_unchanged_reset():
    sim = alloy.build(plain_scene(), device="cpu")
    assert sim.engine.reset_hooks == [] and sim.randomized() == {}
    assert getattr(sim.built, "robot_random", None) is None
    sim.step(30)
    first = sim.state(["rig", "box", "rig/lift"])
    sim.reset().step(30)
    again = sim.state(["rig", "box", "rig/lift"])
    assert all(np.array_equal(first[k], again[k]) for k in first if k != "t")      # "t" keeps counting over a reset
    sim.close()


@pytest.mark.compile_heavy
def test_build_time_rows_install_no_reset_hook():
    scene = plain_scene()
    scene.randomize("rig", "mass_scale", 0.5, 2.0)
    scene.randomize("gravity", "z", -10.0, -9.0)
    sim = alloy.build(scene, device="cpu")
    assert sim.engine.reset_hooks == [] and sorted(sim.randomized()) == ["gravity.z", "rig.mass_scale"]
    before = {k: v.copy() for k, v in sim.randomized().items()}
    sim.reset()
    assert all(np.array_equal(before[k], v) for k, v in sim.randomized().items())      # not redrawn
    sim.close()


# --- hook contract -------------------------------------------------------------------------------------------


def reset_scene(worlds: int = 4) -> alloy.Scene:
    scene = rig_scene(worlds)
    scene.randomize("rig", "mass_scale", 0.5, 2.0, when="reset")
    scene.randomize("rig", "kp_scale", 0.5, 2.0)
    return scene


def mask_of(bits) -> object:
    import warp as wp

    return wp.array(np.asarray(bits, np.int32), dtype=wp.int32, device="cpu")


@pytest.mark.compile_heavy
def test_hooks_get_none_from_sim_reset_and_the_mask_from_engine_reset():
    sim = alloy.build(reset_scene(), device="cpu", seed=1)
    assert len(sim.engine.reset_hooks) == 1                                     # the redraw
    seen: list = []
    sim.engine.reset_hooks.append(seen.append)
    sim.reset()
    assert seen == [None]
    bits = [1, 0, 1, 0]
    before = sim.randomized()["rig.mass_scale"].copy()
    mask = mask_of(bits)
    sim.engine.reset(mask)
    assert len(seen) == 2 and seen[1] is mask
    after = sim.randomized()["rig.mass_scale"]
    assert np.array_equal(after[[1, 3]], before[[1, 3]]) and np.all(after[[0, 2]] != before[[0, 2]])
    sim.close()


@pytest.mark.compile_heavy
def test_a_failed_redraw_puts_the_previous_draws_back(monkeypatch):
    from alloy.api import _robot_random_apply as A

    sim = alloy.build(reset_scene(), device="cpu", seed=1)
    pre = {k: v.copy() for k, v in sim.randomized().items()}
    real, calls = A.WRITERS["inertial"], []

    def boom(model, st, worlds):
        calls.append(len(worlds))
        if len(calls) == 1:
            raise RuntimeError("writer failed")
        return real(model, st, worlds)

    monkeypatch.setitem(A.WRITERS, "inertial", boom)
    with pytest.raises(RuntimeError, match="writer failed"):
        sim.reset()
    assert len(calls) == 2                                                      # the failed write, then the restore
    assert all(np.array_equal(pre[k], v) for k, v in sim.randomized().items())
    assert np.array_equal(sim.built.robot_random.counts, np.zeros(4, np.int64))
    sim.reset()                                                                 # works again, redraws
    assert not np.array_equal(pre["rig.mass_scale"], sim.randomized()["rig.mass_scale"])
    sim.close()


# --- determinism of draws ------------------------------------------------------------------------------------


def draws(worlds: int, seed: int, scene_fn=reset_scene) -> dict[str, np.ndarray]:
    sim = alloy.build(scene_fn(worlds), device="cpu", seed=seed)
    out = sim.randomized()
    sim.close()
    return out


@pytest.mark.compile_heavy
def test_draws_follow_the_seed_and_the_global_world_id_inside_the_range():
    a, b, c = draws(4, 5), draws(4, 5), draws(4, 6)
    assert all(np.array_equal(a[k], b[k]) for k in a) and not np.array_equal(a["rig.mass_scale"], c["rig.mass_scale"])
    big = draws(8, 5)
    for k in a:
        assert np.array_equal(a[k], big[k][:4])                                 # world w is world w at any count
        assert np.all((a[k] >= 0.5) & (a[k] <= 2.0)) and len(set(a[k])) == 4


def link_masses(worlds: int, rows) -> tuple[np.ndarray, dict]:
    scene = rig_scene(worlds, TWO_LINK)
    for target, field, lo, hi in rows:
        scene.randomize(target, field, lo, hi)
    sim = alloy.build(scene, device="cpu", seed=2)
    mass, drawn = sim.engine.model.body_mass.numpy().reshape(worlds, -1), sim.randomized()
    sim.close()
    return mass, drawn


@pytest.mark.compile_heavy
def test_a_robot_wide_row_gives_every_link_one_factor_and_two_rows_two_draws():
    plain, _ = link_masses(3, [])
    wide, drawn = link_masses(3, [("rig", "mass_scale", 0.5, 2.0)])
    assert np.allclose(wide / plain, drawn["rig.mass_scale"][:, None])
    split, drawn = link_masses(3, [("rig/a", "mass_scale", 0.5, 2.0), ("rig/b", "mass_scale", 0.5, 2.0)])
    ratio = split / plain
    assert not np.allclose(drawn["rig/a.mass_scale"], drawn["rig/b.mass_scale"])
    assert np.allclose(ratio[:, 0], drawn["rig/a.mass_scale"]) and np.allclose(ratio[:, 1], drawn["rig/b.mass_scale"])


# --- process split -------------------------------------------------------------------------------------------


def rest_scene() -> alloy.Scene:
    scene = rig_scene(4)
    scene.randomize("rig", "mass_scale", 0.5, 2.0)
    scene.randomize("rig", "kp_scale", 0.5, 2.0)
    return scene


@pytest.mark.compile_heavy
def test_a_process_split_steps_the_randomized_worlds_like_one_process():
    one = alloy.build(rest_scene(), device="cpu", seed=3)
    ref = one.run(2.0, record=["rig/lift"])["rig/lift.q"][-1]
    drawn = one.randomized()
    one.close()
    assert len(set(np.round(ref, 6))) == 4                                      # the draws moved the rest position
    with alloy.build(rest_scene(), device="cpu", seed=3, processes=2) as split:
        got = split.run(2.0, record=["rig/lift"])["rig/lift.q"][-1]
        assert np.array_equal(ref, got), (ref, got)
        assert np.all((drawn["rig.mass_scale"] >= 0.5) & (drawn["rig.mass_scale"] <= 2.0))
        try:
            split.randomized()
        except alloy.FeatureUnavailableError as err:                            # report: no per-process draws
            pytest.skip(f"sim.randomized() on a processes= sim raises: {err.message}")


@pytest.mark.timeout(600)
def test_the_guide_pages_printed_payloads_and_heights_are_what_a_run_gives():
    """docs/guide/randomize_robots.md prints the payload and base height of 8 Go2 worlds: a run gives them again
    (payload to the printed digit, height within 3 mm), so the page cannot drift from the shipped robot."""
    page = (Path(__file__).resolve().parents[2] / "docs" / "guide" / "randomize_robots.md").read_text()
    block = page.split("height falls as payload grows:")[1].split("```text")[1]
    rows = [np.array(row.split(), float) for row in re.findall(r"\[([^\]]+)\]", block)[:2]]
    scene = alloy.Scene(worlds=8, dt=0.01)
    scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")
    scene.randomize("go2/base", "mass_add", 0.0, 6.0)
    sim = alloy.build(scene, seed=0, device="cpu")
    sim.run(1.0)
    payload = sim.randomized()["go2/base.mass_add"]
    height = sim.state(["go2/base"])["go2/base.pos"][:, 2]
    assert sim.check()["ok"]
    np.testing.assert_allclose(payload, rows[0], rtol=0, atol=0.006)
    np.testing.assert_allclose(height, rows[1], rtol=0, atol=0.003)
    assert np.corrcoef(payload, height)[0, 1] < 0
