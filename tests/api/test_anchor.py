# SPDX-License-Identifier: Apache-2.0
"""``Scene.add_object(..., anchor=)``: a library object's own geometry, not its file origin, goes to ``pos`` (CPU).

Scanned objects (YCB, GSO, RoboCasa) have origins 19-85 mm off the mesh, so objects placed by origin landed inside
walls (OSS-8, WP13's sweep). A fake scanned library in the fetch cache holds a box mesh whose origin is off its
centre, as a scan's is. Default "auto" puts an imported object's bottom centre at ``pos`` (where it stands) and keeps
Alloy's own / your own cards on their authored origin; "center" puts the box centre there; "origin" is the old
behaviour. What you set is what you read (coordinator ruling, WP13's mug): ``"<name>.pos"`` reports that anchor point
and ``set_pose`` puts it there; ``"<name>.body_pos"`` is the body frame.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

import alloy.api as alloy
from alloy.api import _anchor, catalog, meshes
from alloy.api import _args as A

OFF = np.array([0.03, -0.02, 0.05])             # the scan's origin is this far from its box centre
FAR = np.array([0.4, 0.3, -0.2])                # a far-origin mesh (Objaverse before OSS-10's recentre)
SIZE = (0.06, 0.04, 0.10)
BASE = OFF - np.array([0.0, 0.0, 0.05])        # its bottom centre in the card's frame


def _card(root, name: str, off: np.ndarray = OFF) -> None:
    v, f = meshes.box(SIZE)                     # bottom at z = 0, centred in x and y
    v = v + off - np.array([0.0, 0.0, 0.5 * SIZE[2]])           # box centre at off in the card's frame
    obj = "".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in v.tolist()) + "".join(f"f {a + 1} {b + 1} {c + 1}\n"
                                                                             for a, b, c in f.tolist())
    (root / "meshes").mkdir(parents=True, exist_ok=True)
    (root / "objects").mkdir(parents=True, exist_ok=True)
    (root / "meshes" / f"{name}.obj").write_text(obj)
    sha = hashlib.sha256(obj.encode()).hexdigest()
    (root / "objects" / f"{name}.yaml").write_text(
        f"schema: alloy.object/1\nname: {name}\nshape:\n  mesh:\n    path: meshes/{name}.obj\n    sha256: {sha}\n"
        "    collision: convex\nmaterial: abs\n")


@pytest.fixture
def scan(tmp_path, monkeypatch):
    """A fetched (imported) library 'scans' with card 'scan_box', and the user root 'mine' with 'my_box'."""
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    _card(tmp_path / "cache" / "imports" / "scans", "scan_box")
    _card(tmp_path / "cache" / "imports" / "scans", "far_box", FAR)
    _card(tmp_path / "mine", "my_box")
    catalog.refresh()
    yield tmp_path / "mine"
    catalog.refresh()


def placed(scene: alloy.Scene, name: str) -> np.ndarray:
    """Where the body origin goes (the scene the build writes)."""
    return np.asarray(_anchor.resolve(scene).items[name]["pose"][:3])


def test_auto_stands_an_imported_scan_on_pos_and_keeps_own_cards_on_their_origin(scan):
    scene = alloy.Scene(roots=[str(scan)])
    scene.add_object("a", "scan_box", pos=(0.0, 0.0, 0.3))
    scene.add_object("b", "my_box", pos=(0.5, 0.0, 0.3))
    scene.add_object("c", "scan_box", pos=(1.0, 0.0, 0.3), anchor="center")
    scene.add_object("d", "scan_box", pos=(1.5, 0.0, 0.3), anchor="origin")
    scene.add_object("e", "my_box", pos=(2.0, 0.0, 0.3), anchor="center", rpy=(0.0, 0.0, 0.5 * math.pi))
    assert np.allclose(placed(scene, "a"), [0.0, 0.0, 0.3] - BASE)                         # bottom centre on pos
    assert np.allclose(placed(scene, "b"), [0.5, 0.0, 0.3])                                # own card: origin
    assert np.allclose(placed(scene, "c"), [1.0, 0.0, 0.3] - OFF)                          # box centre on pos
    assert np.allclose(placed(scene, "d"), [1.5, 0.0, 0.3])
    assert np.allclose(placed(scene, "e"), [2.0 - 0.02, -0.03, 0.3 - OFF[2]])              # turned 90 deg: R·c
    assert scene.items["a"]["pose"][:3] == [0.0, 0.0, 0.3]                                # the scene keeps pos
    with pytest.raises(alloy.SceneError) as err:
        scene.add_object("x", "scan_box", anchor="middle")
    assert err.value.field == "anchor"


def test_per_world_rows_and_randomized_ranges_move_with_the_anchor(scan):
    scene = alloy.Scene(worlds=2, roots=[str(scan)])
    scene.add_object("a", "scan_box", pos=[[0.0, 0.0, 0.3], [0.2, 0.0, 0.3]])
    scene.randomize("a", "y", -0.01, 0.01)
    out = _anchor.resolve(scene).items["a"]
    assert np.allclose(out["per_world_pos"], np.array([[0.0, 0.0, 0.3], [0.2, 0.0, 0.3]]) - BASE)
    assert out["randomize"] == [{"field": "pose.1", "lo": pytest.approx(-0.01 - OFF[1]),
                                 "hi": pytest.approx(0.01 - OFF[1])}]
    old = scene.to_dict()
    del old["items"]["a"]["anchor"]                                                      # saved before anchors
    assert _anchor.resolve(alloy.Scene.from_dict(old)).items["a"]["pose"][:3] == [0.0, 0.0, 0.3]


def _xyzw(rpy) -> np.ndarray:
    """(x, y, z, w) of roll-pitch-yaw, as the Scene reads ``rpy=``."""
    return np.asarray(A.rpy_to_xyzw(rpy))


def quat_wxyz(rpy) -> np.ndarray:
    return _xyzw(rpy)[[3, 0, 1, 2]]


def test_anchor_points_and_body_origins_invert_each_other():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    pts, c = rng.normal(size=(5, 3)), np.array([0.03, -0.02, 0.05])
    origins = _anchor.body_origins(pts, q, c)
    assert np.allclose(_anchor.world_points(np.concatenate([origins, q], axis=1), c), pts, atol=1e-12)
    yaw = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])     # 90 deg about z
    assert np.allclose(_anchor.world_points(yaw, (0.1, 0.0, 0.0)), [[1.0, 2.1, 3.0]])


@pytest.mark.compile_heavy
def test_what_you_set_is_what_you_read(scan):
    """pos, set_pose and record= agree at the anchor for yawed and tilted scans; primitives and own cards keep their
    body origin as .pos and have no .body_pos."""
    yaw, tilt = (0.0, 0.0, 0.7), (0.4, -0.3, 0.2)
    scene = alloy.Scene(dt=0.01, worlds=2, roots=[str(scan)])
    scene.add_ground()
    scene.add_object("can", "scan_box", pos=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], rpy=yaw)     # stands on the ground
    scene.add_object("far", "far_box", pos=(0.5, 0.0, 0.3), rpy=tilt, anchor="center")
    scene.add_box("prim", size=(0.05,) * 3, pos=(1.0, 0.0, 0.3), rpy=tilt, mass=0.1)
    scene.add_object("own", "my_box", pos=(1.5, 0.0, 0.3), rpy=tilt)
    sim = alloy.build(scene, device="cpu")
    s = sim.state()
    assert np.allclose(s["can.pos"], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], atol=1e-6)
    assert np.allclose(s["far.pos"], [[0.5, 0.0, 0.3]] * 2, atol=1e-6)
    assert np.allclose(s["far.body_pos"][0], [0.5, 0.0, 0.3] - _anchor._rotate(_xyzw(tilt), FAR), atol=1e-6)
    assert np.allclose(s["prim.pos"], [[1.0, 0.0, 0.3]] * 2) and np.allclose(s["own.pos"], [[1.5, 0.0, 0.3]] * 2)
    assert "prim.body_pos" not in s and "own.body_pos" not in s
    q = quat_wxyz((0.0, 0.5, 0.0))
    sim.set_pose("far", pos=(0.6, 0.1, 0.4), quat_wxyz=q)
    assert np.allclose(sim.state(["far"])["far.pos"], [[0.6, 0.1, 0.4]] * 2, atol=1e-6)
    sim.set_pose("far", quat_wxyz=(1.0, 0.0, 0.0, 0.0))                               # turns about its anchor
    s = sim.state(["far"], world=0)
    assert np.allclose(s["far.pos"], [0.6, 0.1, 0.4], atol=1e-6)
    assert np.allclose(s["far.body_pos"], [0.6, 0.1, 0.4] - FAR, atol=1e-6)
    sim.set_pose("far", pos=(0.5, 0.0, 0.3), quat_wxyz=quat_wxyz(tilt))
    traj = sim.run(0.3, record=["can", "far"]).world(0)
    assert np.allclose(traj["far.pos"][0], [0.5, 0.0, 0.3], atol=1e-6)                 # frame 0: what was set
    assert traj["can.pos"][-1, 2] == pytest.approx(0.0, abs=1e-3) and sim.check()["ok"]  # base rests on the ground
    sim.close()


@pytest.mark.compile_heavy
def test_ycb_and_objaverse_read_back_where_they_were_put():
    """Real scans, yawed and tilted, beside a tilted primitive: .pos at t = 0 within 1 mm of pos (skipped where the
    meshes cannot be fetched)."""
    cards = {"soup": ("ycb_005_tomato_soup_can", (0.0, 0.0, 0.9)), "mug": ("ycb_025_mug", (0.3, -0.2, 0.4)),
             "spray": ("objv_aerosol_can_0adc4e0d", (-0.4, 0.5, 0.1))}
    for card, _ in cards.values():
        try:
            _anchor.local_box(card, ())
        except (alloy.SceneError, alloy.BuildError, KeyError, FileNotFoundError) as exc:
            pytest.skip(f"no {card} mesh here: {exc}")
    scene = alloy.Scene(dt=0.01)
    for i, (name, (card, rpy)) in enumerate(cards.items()):
        scene.add_object(name, card, pos=(0.3 * i, 0.0, 0.2), rpy=rpy)
    scene.add_box("block", size=(0.04, 0.04, 0.04), pos=(1.0, 0.0, 0.2), rpy=(0.3, 0.2, 0.1), mass=0.1)
    sim = alloy.build(scene, device="cpu")
    s = sim.state(world=0)
    for i, name in enumerate(cards):
        assert np.allclose(s[f"{name}.pos"], [0.3 * i, 0.0, 0.2], atol=1e-3), name
        assert f"{name}.body_pos" in s
    assert np.allclose(s["block.pos"], [1.0, 0.0, 0.2], atol=1e-6) and "block.body_pos" not in s
    sim.close()


def test_ycb_soup_can_mesh_centre_lands_on_pos():
    """The case that started it: ycb_005's origin is 85 mm off the can (skipped where its mesh cannot be fetched).
    By default it stands on pos; with anchor="center" its box centre is there."""
    try:
        lo, hi = _anchor.local_box("ycb_005_tomato_soup_can", ())
    except (alloy.SceneError, alloy.BuildError, KeyError, FileNotFoundError) as exc:
        pytest.skip(f"no YCB mesh here: {exc}")
    scene = alloy.Scene()
    scene.add_object("can", "ycb_005_tomato_soup_can", pos=(0.1, 0.2, 0.3))
    scene.add_object("mid", "ycb_005_tomato_soup_can", pos=(0.1, 0.2, 0.3), anchor="center")
    base = placed(scene, "can") + np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
    assert np.allclose(base, [0.1, 0.2, 0.3], atol=1e-3)
    assert np.allclose(placed(scene, "mid") + 0.5 * (lo + hi), [0.1, 0.2, 0.3], atol=1e-3)
