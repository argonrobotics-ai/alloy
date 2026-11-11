# SPDX-License-Identifier: Apache-2.0
"""Instance ids, semantic classes and optical flow (alloy.camera.aov): flow against the pinhole model for a moving
body, a turning body and a moving camera; what flow does not track; ids and classes by name (tiny CPU images)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.camera import Camera, CameraError, camera_scene, image_extent
from alloy.camera.aov import instance_labels, semantic_labels, set_semantics
from alloy.camera.scene import IDENTITY_XF, SceneData
from tests.camera.helpers import FakeState, body_scene, cube, plane_scene, quad

DEV = "cpu"
DOWN_UP = (0.0, 1.0, 0.0)
W, FOV, HEIGHT = 16, 30.0, 1.0


def _down(eye=(0.0, 0.0, HEIGHT)):
    """Top-down camera: world +x is image right, world +y is image up."""
    return Camera("down", look_at=(eye, (eye[0], eye[1], 0.0), DOWN_UP), width=W, height=W, fov=FOV, spp=1)


def _pixels_per_metre(depth: float) -> float:
    return 0.5 * W / (depth * image_extent(FOV, W, W, "VERTICAL")[0])


def _flow(cam, sc, q=None):
    state = None if q is None else FakeState(np.asarray(q, np.float32).reshape(-1, 7), DEV)
    return cam.render(sc, state, output="flow").numpy().copy()


@pytest.mark.law
def test_still_scene_and_camera_read_zero_and_misses_nan():
    """Nothing moved between two flow renders: 0 on every surface (float32 rounding), NaN where nothing is hit."""
    sc, cam = camera_scene(body_scene(1), device=DEV), _down()
    first, again = _flow(cam, sc, IDENTITY_XF), _flow(cam, sc, IDENTITY_XF)
    hit = np.isfinite(cam.render(sc, output="depth").numpy())
    assert first.shape == (1, W, W, 2) and first.dtype == np.float32
    assert hit[0, 8, 8] and not hit.all()
    assert np.abs(first[hit]).max() < 1e-3 and np.abs(again[hit]).max() < 1e-3
    assert np.isnan(first[~hit]).all() and np.isnan(again[~hit]).all()


@pytest.mark.law
def test_translating_body_matches_the_pinhole_model():
    """A cube moved by (dx, dy) under a top-down camera: flow = (dx, -dy) * W / (2 * depth * extent) on its top."""
    sc, cam = camera_scene(body_scene(1), device=DEV), _down()
    q = np.array(IDENTITY_XF, np.float32)
    _flow(cam, sc, q)
    q[:2] = (0.02, -0.01)
    flow = _flow(cam, sc, q)[0]
    k = _pixels_per_metre(HEIGHT - 0.1)  # the cube's top face is 0.1 above its centre
    top = np.isclose(cam.render(sc, output="depth").numpy()[0], HEIGHT - 0.1, atol=1e-4)
    assert top.sum() >= 4
    assert np.allclose(flow[top], (0.02 * k, 0.01 * k), atol=2e-3)
    assert np.abs(_flow(cam, sc, q)[0][top]).max() < 1e-3  # the reference is the previous flow render


@pytest.mark.law
def test_turning_body_flow_points_back_to_where_each_point_was():
    """A cube turned by 0.2 rad about z: the point at a pixel was at R(-0.2) of its position, in pixels."""
    sc, cam, angle = camera_scene(body_scene(1), device=DEV), _down(), 0.2
    q = np.array(IDENTITY_XF, np.float32)
    _flow(cam, sc, q)
    q[3:] = (0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2))
    flow = _flow(cam, sc, q)[0]
    k = _pixels_per_metre(HEIGHT - 0.1)
    top = np.isclose(cam.render(sc, output="depth").numpy()[0], HEIGHT - 0.1, atol=1e-4)
    row, col = np.nonzero(top)
    x, y = (col + 0.5 - W / 2) / k, -(row + 0.5 - W / 2) / k  # world position of each top-face pixel
    c, s = np.cos(-angle), np.sin(-angle)
    before = np.stack([c * x - s * y, s * x + c * y], axis=1)
    expect = np.stack([(x - before[:, 0]) * k, -(y - before[:, 1]) * k], axis=1)
    assert len(row) >= 4 and np.abs(expect).max() > 0.1
    assert np.allclose(flow[top], expect, atol=2e-3)


@pytest.mark.law
def test_moving_camera_over_a_still_plane():
    """The camera moved +x by d over a plane at depth h: every pixel reads (-d * W / (2 h extent), 0)."""
    sc, cam = camera_scene(plane_scene(), device=DEV), _down()
    _flow(cam, sc)
    cam.set_pose(look_at=((0.05, 0.0, HEIGHT), (0.05, 0.0, 0.0), DOWN_UP))
    flow = _flow(cam, sc)
    assert np.allclose(flow, (-0.05 * _pixels_per_metre(HEIGHT), 0.0), atol=2e-3)


def test_flow_is_per_world_and_leaves_other_outputs_alone():
    """Only the world whose cube moved reads flow; depth and ids are the same with and without flow asked."""
    sc, cam = camera_scene(body_scene(2), device=DEV), _down()
    q = np.tile(IDENTITY_XF, (2, 1)).astype(np.float32)
    _flow(cam, sc, q)
    q[1, 0] = 0.03
    state = FakeState(q, DEV)
    out = cam.render(sc, state, output=("depth", "segmentation", "flow"))
    flow, depth, seg = out.flow.numpy().copy(), out.depth.numpy().copy(), out.segmentation.numpy().copy()
    plain = cam.render(sc, state, output=("depth", "segmentation"))
    assert np.array_equal(depth, plain.depth.numpy()) and np.array_equal(seg, plain.segmentation.numpy())
    hit = np.isfinite(depth)
    assert np.abs(flow[0][hit[0]]).max() < 1e-3
    assert np.allclose(flow[1][hit[1]][:, 0], 0.03 * _pixels_per_metre(HEIGHT - 0.1), atol=2e-3)


def test_colour_is_bit_identical_beside_the_new_outputs():
    sc, cam = camera_scene(body_scene(1), device=DEV), _down()
    set_semantics(sc.sd, {"*": "thing"})
    rgb = cam.render(sc, seed=3).numpy().copy()
    out = cam.render(sc, seed=3, output=("rgb", "flow", "instance_id", "semantic"))
    assert np.array_equal(out.rgb.numpy(), rgb) and out.depth is None and out.flow is not None


def _two_shape_scene():
    """World w: a fixed plate (not on a body) and a cube on body w, side by side."""
    n = 2
    return SceneData([quad(0.2), cube(0.05)], [0, 1] * n, [0, 0, 1, 1], [-1, 0, -1, 1],
                     [[-0.1, 0, 0, 0, 0, 0, 1], IDENTITY_XF] * n,
                     np.full((2 * n, 3), 0.5), n, ["plate", "cube", "plate", "cube"])


def test_instance_ids_name_each_shape_the_same_in_every_world():
    sd = _two_shape_scene()
    sc = camera_scene(sd, device=DEV)
    q = np.array([[0.12, 0, 0, 0, 0, 0, 1]] * 2, np.float32)
    inst = _down().render(sc, FakeState(q, DEV), output="instance_id").numpy()
    assert instance_labels(sd) == ["background", "plate", "cube"]
    assert inst.dtype == np.int32 and np.array_equal(inst[0], inst[1])
    assert set(np.unique(inst)) == {0, 1, 2}
    assert inst[0, 8, 5] == 1 and inst[0, 8, 10] == 2  # plate on the left, cube on the right


def test_semantic_classes_by_pattern_and_replacement():
    sd = _two_shape_scene()
    sc, cam = camera_scene(sd, device=DEV), _down()
    q = np.array([[0.12, 0, 0, 0, 0, 0, 1]] * 2, np.float32)
    with pytest.raises(CameraError, match="set_semantics"):
        cam.render(sc, FakeState(q, DEV), output="semantic")
    assert set_semantics(sd, {"cu*": "toy"}) == ["background", "unlabeled", "toy"]
    sem = cam.render(sc, FakeState(q, DEV), output="semantic").numpy().copy()
    assert sem[0, 8, 5] == 1 and sem[0, 8, 10] == 2 and (sem[:, 0, 0] == 0).all() and np.array_equal(sem[0], sem[1])
    assert set_semantics(sd, {"plate": "floor", "cube": "toy", "c*": "never"}) == [
        "background", "unlabeled", "floor", "toy", "never"]
    assert semantic_labels(sd)[3] == "toy"
    sem = cam.render(sc, FakeState(q, DEV), output="semantic").numpy()
    assert sem[0, 8, 5] == 2 and sem[0, 8, 10] == 3  # the first key that matches wins
    with pytest.raises(CameraError):
        set_semantics(sd, {"cube": 3})


ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.05">
  <joint name="shoulder" type="hinge" axis="0 0 1" damping="1"/>
  <geom type="box" size="0.03 0.03 0.03" mass="1"/></body></worldbody></mujoco>"""


@pytest.mark.integration
@pytest.mark.timeout(300)  # a cold kernel cache compiles the sim's kernels inside this test
def test_sim_instances_semantics_and_cloth_flow():
    """In an alloy.Sim: static objects get their own instance ids, classes match body names (``arm/*``), a cloth
    reads NaN flow while the ground under the still camera reads 0, and rendering does not change the state."""
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.1, 0.1, 0.02), pos=(-0.15, 0.15, 0.01), static=True)
    scene.add_robot("arm", mjcf=ARM, pos=(0.15, 0.15, 0.0))
    scene.add_cloth("sheet", size=(0.12, 0.12), pos=(0.0, -0.15, 0.2), resolution=6, mass=0.02)
    sim = alloy.build(scene, device=DEV)
    cam = Camera("top", look_at=((0.0, 0.0, 1.2), (0.0, 0.0, 0.0), DOWN_UP), width=32, height=32, fov=30, spp=1)
    names = instance_labels(sim)
    assert names[0] == "background" and {"ground", "table"} <= set(names)
    classes = set_semantics(sim, {"arm/*": "robot", "table": "furniture", "ground": "floor"})
    before = {k: np.array(v, copy=True) for k, v in sim.state().items()}
    out = sim.render(cam, output=("instance_id", "semantic", "flow", "segmentation"))
    after = sim.state()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    inst, sem, flow = out["instance_id"], out["semantic"], out["flow"]
    assert flow.shape == (2, 32, 32, 2) and np.array_equal(inst[0], inst[1])
    seen = {names[i] for i in np.unique(inst)}
    assert {"ground", "table"} <= seen and len(seen) >= 4
    assert {classes[i] for i in np.unique(sem)} == {"robot", "furniture", "floor", "unlabeled"}
    cloth = sem == 1
    assert cloth.any() and np.isnan(flow[cloth]).all()
    assert np.abs(flow[inst == names.index("ground")]).max() < 1e-3
    arm = sem == classes.index("robot")
    assert arm.any() and (out["segmentation"][arm] >= 2).all()


@pytest.mark.gpu
def test_cpu_and_gpu_agree_on_the_new_outputs():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    res = {}
    for dev in ("cpu", "cuda:0"):
        sd = _two_shape_scene()
        set_semantics(sd, {"cube": "toy"})
        sc, cam = camera_scene(sd, device=dev), _down()
        q = np.array([[0.12, 0, 0, 0, 0, 0, 1]] * 2, np.float32)
        cam.render(sc, FakeState(q, dev), output="flow")
        q[:, 1] = 0.02
        out = cam.render(sc, FakeState(q, dev), output=("instance_id", "semantic", "flow"))
        res[dev] = {k: getattr(out, k).numpy().copy() for k in ("instance_id", "semantic", "flow")}
    assert np.array_equal(res["cpu"]["instance_id"], res["cuda:0"]["instance_id"])
    assert np.array_equal(res["cpu"]["semantic"], res["cuda:0"]["semantic"])
    assert np.allclose(res["cpu"]["flow"], res["cuda:0"]["flow"], atol=1e-3, equal_nan=True)
    assert np.nanmax(np.abs(res["cpu"]["flow"])) > 0.1


def test_render_torch_returns_the_new_outputs():
    torch = pytest.importorskip("torch")
    sd = _two_shape_scene()
    sc, cam = camera_scene(sd, device=DEV), _down()
    q = np.array([[0.12, 0, 0, 0, 0, 0, 1]] * 2, np.float32)
    out = cam.render_torch(sc, FakeState(q, DEV), output=("flow", "instance_id"))
    assert set(out) == {"flow", "instance_id"}
    assert tuple(out["flow"].shape) == (2, W, W, 2) and out["flow"].dtype == torch.float32
    assert tuple(out["instance_id"].shape) == (2, W, W) and out["instance_id"].dtype == torch.int32
