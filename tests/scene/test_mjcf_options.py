# SPDX-License-Identifier: Apache-2.0
"""MuJoCo entry options follow the scene's MJCF <option> (coordinator ruling at the end of [internal note]):
capability defaults < MJCF values < route overrides, with every taken value and every override listed in the route
report (`RouteReport.entry_options`)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alloy.scene import mjcf_options as MO
from alloy.scene.fast_profile import NEWTON_EFFECT
from alloy.scene.entries import (FILL_IMPRATIO, PRECISION_IMPRATIO, chain_on_newton, mujoco_overrides,
                                 sliding_warning)

ROBOT = '<mujoco><option impratio="100" cone="pyramidal" solver="Newton" iterations="50" tolerance="1e-8" ' \
        'integrator="Euler" noslip_iterations="50"/><worldbody/></mujoco>'
OBJ = '<mujoco><option impratio="1" iterations="7"/><option tolerance="1e-6"/><worldbody/></mujoco>'


def _files(tmp_path):
    (tmp_path / "robot").mkdir()
    (tmp_path / "obj").mkdir()
    (tmp_path / "robot" / "r.xml").write_text(ROBOT)
    (tmp_path / "obj" / "o.xml").write_text(OBJ)
    return str(tmp_path / "robot" / "r.xml"), str(tmp_path / "obj" / "o.xml")


def test_first_file_wins_and_conflicts_are_listed(tmp_path):
    robot, obj = _files(tmp_path)
    opts = MO.scene_options([robot, obj])
    assert opts.values == {"impratio": 100.0, "cone": "pyramidal", "solver": "newton", "iterations": 50,
                           "tolerance": 1e-8}
    assert opts.asked == {"integrator": ("euler", "robot/r.xml")}       # reported, not honoured
    assert {c.split(":")[0] for c in opts.conflicts} == {"impratio", "iterations", "tolerance"}
    only_obj = MO.scene_options([obj])
    assert only_obj.values == {"impratio": 1.0, "iterations": 7, "tolerance": 1e-6}   # later <option> merges in


def test_report_names_each_value_and_override_reason(tmp_path):
    robot, _ = _files(tmp_path)
    opts = MO.scene_options([robot])
    over = mujoco_overrides({"mujoco:solver=cg"}, has_precision=True)
    kw = {"integrator": "implicitfast", **opts.values, **{k: v for k, (v, _) in over.items()}}
    lines = MO.report_lines(opts, kw, over)
    assert f"mujoco impratio={PRECISION_IMPRATIO}: robot/r.xml sets 100.0; route override: precision pairs" in lines[0]
    assert any(line.startswith("mujoco cone=elliptic: robot/r.xml sets pyramidal; route override") for line in lines)
    assert any(line.startswith("mujoco tolerance=1e-08 from robot/r.xml") for line in lines)
    assert any("integrator=implicitfast: robot/r.xml asks euler" in line for line in lines)


def test_no_mjcf_keeps_the_capability_defaults():
    from alloy.scene.builtin_plugins import register_builtin_solvers
    from alloy.scene.entries import solver_kwargs

    register_builtin_solvers()
    from alloy.route.types import Route

    obj = SimpleNamespace(decision=SimpleNamespace(scene_effects=(), route=Route.RIGID), obj=None)
    plain = solver_kwargs("mujoco", [obj], None, False, 10)
    assert plain["cone"] == "elliptic" and plain["impratio"] == FILL_IMPRATIO        # SCR-WP3-10 fill-in
    opts = MO.MjcfOptions({"impratio": 100.0, "cone": "pyramidal"}, {"impratio": "r", "cone": "r"})
    honoured = solver_kwargs("mujoco", [obj], None, False, 10, mjcf=opts)
    assert honoured["cone"] == "elliptic" and honoured["impratio"] == 100.0     # pyramidal: route override
    elliptic = MO.MjcfOptions({"cone": "elliptic", "iterations": 30}, {"cone": "r", "iterations": "r"})
    assert solver_kwargs("mujoco", [obj], None, False, 10, mjcf=elliptic)["iterations"] == 30
    precise = solver_kwargs("mujoco", [obj], None, True, 10, mjcf=opts)
    assert precise["cone"] == "elliptic" and precise["impratio"] == PRECISION_IMPRATIO


def test_fill_in_only_where_no_file_sets_impratio():
    """SCR-WP3-10: 300 fills in; a file's value stands (with a warning below 50); precision pairs still set 300."""
    none = mujoco_overrides(set(), False, None)
    assert none["impratio"] == (FILL_IMPRATIO, "Coulomb sliding force independent of speed (A5.speed, SCR-WP3-10)")
    low = MO.MjcfOptions({"impratio": 1.0}, {"impratio": "emb/scene.xml"})
    assert "impratio" not in mujoco_overrides(set(), False, low)
    assert "A5.speed needs >= 50" in sliding_warning(low, {"impratio": 1.0})
    assert sliding_warning(MO.MjcfOptions({"impratio": 100.0}, {"impratio": "r"}), {"impratio": 100.0}) is None
    assert mujoco_overrides(set(), True, low)["impratio"][0] == PRECISION_IMPRATIO
    assert sliding_warning(low, {"impratio": PRECISION_IMPRATIO}) is None              # overridden: no warning
    lines = MO.report_lines(MO.MjcfOptions(), {"impratio": FILL_IMPRATIO}, none)
    assert lines == [f"mujoco impratio={FILL_IMPRATIO}: no file sets it; route: {none['impratio'][1]}"]


@pytest.mark.parametrize("scene,want", [("task_t12", {"impratio": 100.0, "cone": "elliptic", "iterations": 50}),
                                        ("task_emb_e1_fr3", {"impratio": 1.0, "cone": "elliptic", "iterations": 100})])
def test_library_scene_takes_its_robot_options(scene, want):
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    from alloy.scene.assembler import SceneAssembler

    lib = CardLibrary()
    plan = SceneAssembler(lib).assemble(lib.scene(scene), "cpu", world_count=1)
    kw = next(e for e in plan.entries if e.name == "mujoco").solver_kwargs
    assert {k: kw[k] for k in want} == want
    assert any(line.startswith("mujoco impratio=") for line in plan.report.entry_options)
    if scene == "task_emb_e1_fr3":                      # the file's pyramidal cone, overridden with its reason
        assert any("A5.speed needs >= 50" in w for w in plan.report.warnings)    # its file's impratio 1 stands
        assert any(line.startswith("mujoco cone=elliptic: emb_e1_fr3/scene.xml sets pyramidal; route override: "
                                   "the ordered MuJoCo Warp route") for line in plan.report.entry_options)


def test_chain_on_newton_holds_the_fill_in():
    """A CHAIN on the Newton solver went non-finite at impratio 300 (two-entry box/cloth/cable scenes): no fill-in;
    a chain on CG (equality attachment, T09) keeps it."""
    from alloy.route.types import Route

    chain = [SimpleNamespace(decision=SimpleNamespace(route=Route.CHAIN, scene_effects=()))]
    assert chain_on_newton(chain, set()) and "impratio" not in mujoco_overrides(set(), False, None, True)
    assert not chain_on_newton(chain, {"mujoco:solver=cg"})
    assert chain_on_newton(chain, {"mujoco:solver=cg", NEWTON_EFFECT})          # the fast profile's Newton
    assert not chain_on_newton([SimpleNamespace(decision=SimpleNamespace(route=Route.RIGID))], set())
