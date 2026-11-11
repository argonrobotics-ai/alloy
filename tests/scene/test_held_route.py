# SPDX-License-Identifier: Apache-2.0
"""HELD as the route default (alloy.scene.held_route, docs/notes/held-grip-route.md): grippers read from robot files,
a scene with a gripper and a free object gets HELD's noslip, other scenes and the library packs keep theirs, and
CoupledSim's build options."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from alloy.cards.types import RobotRef
from alloy.scene import held_route as HR
from alloy.scene import noslip_route as NR

UP = (0.0, 0.0, 0.0, 1.0)
ARM = ('<mujoco><worldbody><body name="link1"><joint name="shoulder" type="hinge"/><geom size=".02"/></body>'
       '</worldbody><actuator><position joint="shoulder" kp="200"/></actuator></mujoco>')
URDF = ('<robot name="r"><link name="base"/><link name="left_finger"/>'
        '<joint name="finger_joint" type="{kind}"><parent link="base"/><child link="left_finger"/></joint></robot>')


def _file(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _gripper(tmp_path, noslip: int = 0) -> str:
    from alloy.envs.tasks import GRIPPER_MJCF

    opt = f'<option noslip_iterations="{noslip}" noslip_tolerance="1e-10"/>' if noslip else ""
    return _file(tmp_path, "gripper.xml", GRIPPER_MJCF.replace("<mujoco>", f"<mujoco>{opt}", 1))


def _obj(name: str, motion: str = "dynamic", entry: str = "mujoco"):
    return NS(name=name, obj=NS(card=NS(motion=motion)), decision=NS(entry=entry))


def test_robot_grip_reads_mjcf_and_urdf(tmp_path):
    assert HR.robot_grip(_gripper(tmp_path), "mjcf") == "palm"
    assert HR.robot_grip(_file(tmp_path, "arm.xml", ARM), "mjcf") is None                  # no gripper part named
    passive = '<mujoco><worldbody><body name="finger"><joint name="j"/><geom size=".01"/></body></worldbody></mujoco>'
    assert HR.robot_grip(_file(tmp_path, "passive.xml", passive), "mjcf") is None          # no actuator
    _gripper(tmp_path)
    scene = _file(tmp_path, "scene.xml", '<mujoco><include file="gripper.xml"/></mujoco>')
    assert HR.robot_grip(scene, "mjcf") == "palm"                                          # through <include>
    assert HR.robot_grip(_file(tmp_path, "r.urdf", URDF.format(kind="prismatic")), "urdf") == "finger_joint"
    assert HR.robot_grip(_file(tmp_path, "f.urdf", URDF.format(kind="fixed")), "urdf") is None
    assert HR.robot_grip(scene, "usd") is None
    assert HR.GRIP_NAME.search("left_hand_link") and not HR.GRIP_NAME.search("door_handle")


@pytest.mark.parametrize("robot, part", [("emb_e1_fr3", "finger_joint1"), ("emb_e3_ur10e_2f85",
                                         "gripper_right_driver_joint"), ("emb_e9_leap", "palm"), ("emb_e8_go2", None)])
def test_library_robots(robot, part):
    from alloy.cards.library import default_root

    assert HR.robot_grip(str(default_root() / "robots" / robot / "scene.xml"), "mjcf") == part


def test_held_noslip_needs_a_gripper_and_a_free_object(tmp_path):
    grip = RobotRef(path="g.xml", kind="mjcf", base_pose=(0.0, 0.0, 0.0, *UP), name="gripper")
    arm = RobotRef(path="a.xml", kind="mjcf", base_pose=(0.0, 0.0, 0.0, *UP), name="arm")
    files = {"g.xml": _gripper(tmp_path), "a.xml": _file(tmp_path, "arm.xml", ARM)}
    req = NR.scene_noslip((grip,), files.__getitem__, [_obj("cube"), _obj("mug")])
    assert (req.iterations, req.tolerance, req.source) == (100, 1e-6, NR.HELD_SOURCE)
    assert req.reason == "robot gripper grips (palm), free object cube (+1)"
    assert "HELD default, robot gripper grips (palm)" in NR.report_line(req)
    assert NR.scene_noslip((grip,), files.__getitem__, []) is None                          # nothing to hold
    assert NR.scene_noslip((grip,), files.__getitem__, [_obj("table", motion="static")]) is None
    assert NR.scene_noslip((grip,), files.__getitem__, [_obj("towel", entry="vbd")]) is None
    assert NR.scene_noslip((arm,), files.__getitem__, [_obj("cube")]) is None               # no gripper
    files["g.xml"] = _gripper(tmp_path, noslip=50)                                         # a file's request stands
    assert NR.scene_noslip((grip,), files.__getitem__, [_obj("cube")]).iterations == 50


def test_library_packs_keep_their_settings():
    """Every library scene with a gripper robot either asks for noslip in its files (T and DC packs: 50) or is a
    whole-file scene without object cards (XB4 embodiments), so HELD never changes a pack's MuJoCo entry."""
    from alloy.cards.library import CardLibrary

    lib = CardLibrary()
    for name in lib.names("scenes"):
        card = lib.scene(name)
        paths = [(str(lib.asset_path(r.path)), r.kind) for r in card.robots]
        if not any(HR.robot_grip(p, k) for p, k in paths):
            continue
        asks = NR.files_noslip([p for p, k in paths if k == "mjcf"]) is not None
        assert asks or not card.objects, f"{name}: a gripper scene with object cards would take HELD's noslip"


def test_gripper_lift_takes_held_noslip():
    import alloy
    from alloy.contact.noslip import SOLVER_ATTR
    from alloy.envs.tasks import gripper_lift_scene

    sim = alloy.build(gripper_lift_scene(1), worlds=1, device="cpu")
    (mj,) = [e for e in sim.built.entries if e.capability == "mujoco"]
    assert mj.solver_kwargs[NR.NOSLIP_KEY] == (100, 1e-6)
    assert mj.solver_kwargs["cone"] == "elliptic" and mj.solver_kwargs["impratio"] == 300.0   # the route's own
    assert "mujoco:noslip=100" in sim.built.report.scene_effects and HR.HELD_EFFECT in sim.built.report.scene_effects
    lines = sim.built.report.entry_options
    assert any("HELD default, robot gripper grips (palm), free object cube" in x for x in lines)
    solver = sim.engine.stepper.solvers()[0]       # MuJoCo Warp (ScopedNoslip) or the CPU C route (its noslip spec)
    scoped = getattr(solver, SOLVER_ATTR, None)
    spec = scoped.spec if scoped is not None else solver.route.noslip
    assert (spec.iterations, spec.tolerance, spec.scope) == (100, 1e-6, "robot")


def test_held_raises_a_files_low_impratio():
    from alloy.scene.entries import FILL_IMPRATIO, PRECISION_IMPRATIO, mujoco_overrides
    from alloy.scene.mjcf_options import MjcfOptions

    held = {HR.HELD_EFFECT}
    fr3 = MjcfOptions({"impratio": 1.0, "cone": "pyramidal"}, {"impratio": "fr3/scene.xml", "cone": "fr3/scene.xml"})
    assert mujoco_overrides(held, False, fr3)["impratio"] == (3.0, HR.HELD_REASON)
    assert mujoco_overrides(held, False, fr3)["cone"][0] == "elliptic"
    assert "impratio" not in mujoco_overrides(set(), False, fr3)                           # no HELD: the file's 1
    assert "impratio" not in mujoco_overrides(held, False, MjcfOptions({"impratio": 100.0}, {"impratio": "t12"}))
    assert mujoco_overrides(held, False, MjcfOptions())["impratio"][0] == FILL_IMPRATIO     # no file: the route's
    assert mujoco_overrides(held, True, fr3)["impratio"][0] == PRECISION_IMPRATIO
    assert "impratio" not in mujoco_overrides(held, False, fr3, chain_newton=True)         # the CHAIN hold-back


def test_held_options_raise_impratio_and_set_the_cone():
    import warp as wp

    from alloy._vendor import newton
    from alloy._vendor.newton.solvers import SolverMuJoCo

    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    b.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
    model = b.finalize(device="cpu")
    opts = HR.held_options(model)
    assert opts == {"cone": "elliptic", "impratio": 3.0}
    solver = SolverMuJoCo(model=model, use_mujoco_contacts=True, **opts)
    assert int(solver.mj_model.opt.cone) == 1 and float(solver.mj_model.opt.impratio) == pytest.approx(3.0)
    model.mujoco.impratio.fill_(100.0)                                                     # a higher one stays
    assert HR.held_options(model)["impratio"] == pytest.approx(100.0)
    assert HR.held_options(NS(), HR.HeldGrip(impratio=5.0)) == {"cone": "elliptic", "impratio": 5.0}
