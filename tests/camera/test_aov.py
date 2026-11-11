# SPDX-License-Identifier: Apache-2.0
"""Depth and segmentation images (alloy.camera.aov): analytic plane/cube depth, ids per world and body, the miss and
clip conventions, colour untouched, independence of spp and seed, and the Sim entry point (tiny CPU images)."""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.camera import Camera, CameraError, camera_scene, image_extent
from alloy.camera.aov import segment_labels
from alloy.camera.camera import CameraOutput
from alloy.camera.scene import IDENTITY_XF, SceneData
from tests.camera.helpers import FakeState, body_scene, cube, plane_scene

DEV = "cpu"
DOWN_UP = (0.0, 1.0, 0.0)


def _down(h=1.0, w=8, hh=8, **kw):
    return Camera("down", look_at=((0.0, 0.0, h), (0.0, 0.0, 0.0), DOWN_UP), width=w, height=hh, fov=40, spp=kw.pop("spp", 1), **kw)


def _np(a):
    return a.numpy().copy()


@pytest.mark.law
def test_top_down_plane_depth_is_constant_along_the_axis():
    """Depth is distance along the optical axis, so a plane facing the camera reads h at every pixel."""
    d = _np(_down(1.7).render(camera_scene(plane_scene(), device=DEV), output="depth"))
    assert d.shape == (1, 8, 8) and np.allclose(d, 1.7, rtol=1e-5)


@pytest.mark.law
def test_tilted_plane_depth_matches_the_pinhole_model():
    """Depth of pixel (row, col) is the ray parameter to z=0 for the ray through the pixel centre (z-depth = t)."""
    w, h, fov = 9, 7, 50.0
    eye, target = np.array([0.3, -0.8, 1.2]), np.array([0.0, 0.1, 0.0])
    cam = Camera("t", look_at=(tuple(eye), tuple(target), (0, 0, 1)), width=w, height=h, fov=fov, spp=1)
    d = _np(cam.render(camera_scene(plane_scene(), device=DEV), output="depth"))[0]
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    ex, ey = image_extent(fov, w, h, "VERTICAL")
    col, row = np.meshgrid(np.arange(w), np.arange(h))
    u, v = 2 * (col + 0.5) / w - 1, 1 - 2 * (row + 0.5) / h
    dirs = fwd + right * (u * ex)[..., None] + up * (v * ey)[..., None]
    assert np.allclose(d, -eye[2] / dirs[..., 2], rtol=1e-4)


@pytest.mark.law
def test_cube_depth_at_centre_and_plane_or_inf_elsewhere():
    """Under a top-down camera the cube's top is h - half at the centre; off the cube: the plane (h) or inf."""
    cam = _down(1.0, 16, 16, clip=(0.01, 100.0))
    sd = SceneData([cube(0.1)], [0], [0], [-1], [[0, 0, 0.2, 0, 0, 0, 1]], [(0.8, 0.2, 0.2)], 1)
    d = _np(cam.render(camera_scene(sd, device=DEV), output="depth"))[0]
    assert np.isclose(d[8, 8], 1.0 - 0.3, rtol=1e-5) and np.isinf(d[0, 0])
    sd2 = SceneData([cube(0.1), plane_scene().meshes[0]], [0, 1], [0, 0], [-1, -1],
                    [[0, 0, 0.2, 0, 0, 0, 1], IDENTITY_XF], [(0.8, 0.2, 0.2), (0.7, 0.7, 0.7)], 1)
    d2 = _np(_down(1.0, 16, 16).render(camera_scene(sd2, device=DEV), output="depth"))[0]
    assert np.isclose(d2[8, 8], 0.7, rtol=1e-5) and np.isclose(d2[0, 0], 1.0, rtol=1e-5)


def test_nothing_hit_and_clip_range():
    """Misses read depth inf, segmentation 0, body_id -2, world_id -2; a surface past the clip end is a miss too."""
    sc = camera_scene(plane_scene(), device=DEV)
    up = Camera("up", look_at=((0, 0, 1), (0, 0, 2), (0, 1, 0)), width=6, height=6, spp=1)
    out = up.render(sc, output=("depth", "segmentation", "body_id", "world_id"))
    assert np.isinf(_np(out.depth)).all() and (_np(out.segmentation) == 0).all()
    assert (_np(out.body_id) == -2).all() and (_np(out.world_id) == -2).all()
    far = _down(1.0, 6, 6, clip=(0.01, 0.5))
    assert np.isinf(_np(far.render(sc, output="depth"))).all()
    assert (_np(far.render(sc, output="segmentation")) == 0).all()


def _worlds_cam():
    return Camera("w", look_at=((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), DOWN_UP), width=8, height=8, fov=20, spp=1)


def test_ids_per_world_and_body():
    """Cube w of world w: body_id w, world_id w, segmentation 2 (the same id for the same body in every world)."""
    sc = camera_scene(body_scene(3), device=DEV)
    out = _worlds_cam().render(sc, output=("depth", "segmentation", "body_id", "world_id"))
    seg, body, world, depth = (_np(getattr(out, k)) for k in ("segmentation", "body_id", "world_id", "depth"))
    for w in range(3):
        hit = np.isfinite(depth[w])
        assert hit[4, 4] and hit.sum() > 4
        assert (seg[w][hit] == 2).all() and (body[w][hit] == w).all() and (world[w][hit] == w).all()
        assert (seg[w][~hit] == 0).all() and (body[w][~hit] == -2).all()


def test_fixed_instance_and_shared_group_ids():
    """inst_body -1 reads body_id -1 and segmentation 1; inst_group -1 (every world) reads world_id -1."""
    sd = SceneData([cube()], [0], [-1], [-1], [IDENTITY_XF], [(0.5, 0.5, 0.5)], 2)
    out = _worlds_cam().render(camera_scene(sd, device=DEV), output=("depth", "segmentation", "body_id", "world_id"))
    hit = np.isfinite(_np(out.depth))
    assert hit.any(axis=(1, 2)).all()
    assert (_np(out.body_id)[hit] == -1).all() and (_np(out.segmentation)[hit] == 1).all()
    assert (_np(out.world_id)[hit] == -1).all()


def test_depth_and_ids_follow_the_body():
    """Moving a cube's body toward the camera shrinks its depth, and the ids follow it."""
    sc = camera_scene(body_scene(1), device=DEV)
    cam = _worlds_cam()
    q = np.tile(IDENTITY_XF, (1, 1)).astype(np.float32)
    d0 = _np(cam.render(sc, FakeState(q, DEV), output="depth"))[0, 4, 4]
    q[0, :3] = (0.0, 0.0, 0.3)
    out = cam.render(sc, FakeState(q, DEV), output=("depth", "body_id"))
    assert np.isclose(_np(out.depth)[0, 4, 4], d0 - 0.3, rtol=1e-4) and _np(out.body_id)[0, 4, 4] == 0
    q[0, :3] = (5.0, 0.0, 0.0)
    assert np.isinf(_np(cam.render(sc, FakeState(q, DEV), output="depth"))).all()


def _mixed_scene():
    return SceneData([plane_scene().meshes[0], cube()], [0, 1], [0, 0], [-1, 0],
                     [IDENTITY_XF, [0, 0, 0.1, 0, 0, 0, 1]], [(0.7, 0.7, 0.7), (0.8, 0.2, 0.2)], 1)


def _colour_set(cam, sc):
    """uint8, linear and every field of output='all', as host copies."""
    full = cam.render(sc, output="all")
    out = {"uint8": _np(cam.render(sc, output="uint8")), "linear": _np(cam.render(sc, output="linear"))}
    out.update({f"all.{k}": _np(getattr(full, k)) for k in ("rgb", "linear", "albedo", "normal", "noisy")})
    return out


def test_colour_is_untouched_by_aov_renders():
    """uint8, linear and every field of 'all' are byte-identical with an AOV render in between; rgb tuple matches."""
    sc = camera_scene(_mixed_scene(), device=DEV)
    cam = Camera("c", look_at=((0.6, -0.6, 0.5), (0, 0, 0.1)), width=12, height=12, spp=2)
    before = _colour_set(cam, sc)
    cam.render(sc, output="depth")
    out = cam.render(sc, output=("rgb", "linear", "depth"))
    assert np.array_equal(_np(out.rgb), before["uint8"]) and np.array_equal(_np(out.linear), before["linear"])
    assert out.albedo is None and out.normal is None and out.noisy is None  # only the fields asked for
    assert out.depth is not None and out.segmentation is None
    after = _colour_set(cam, sc)
    assert all(np.array_equal(before[k], after[k]) for k in before)
    full = cam.render(sc, output="all")
    assert isinstance(full, CameraOutput)
    assert all(getattr(full, k) is None for k in ("depth", "segmentation", "body_id", "world_id"))


def test_the_aov_pass_is_freed_with_its_camera():
    """After an AOV render, deleting the camera and collecting frees its renderer (the pass holds no strong ref)."""
    sc = camera_scene(plane_scene(), device=DEV)
    cam = _down()
    cam.render(sc, output="depth")
    ref = weakref.ref(cam._bound[1])
    del cam
    gc.collect()
    assert ref() is None


class _Dyn:
    """Stands in for a LiquidSet or an OptiX scene: never stale, nothing to update or refit."""

    def stale(self):
        return False

    def update(self, state):
        pass

    def refit(self):
        pass


def test_live_liquid_on_optix_raises():
    """A scene with a live liquid and an OptiX scene has no AOV surface for the liquid: CameraError."""
    sc = camera_scene(plane_scene(), device=DEV)
    cam = _down()
    try:
        sc.liquid, sc.optix = _Dyn(), _Dyn()
        with pytest.raises(CameraError, match="OptiX"):
            cam.render(sc, output="depth")
    finally:
        sc.liquid = sc.optix = None


def _newton_model(device=DEV):
    from alloy._vendor import newton
    b = newton.ModelBuilder()
    b.add_ground_plane()
    return b.finalize(device=device)


def test_still_render_with_aov_output_needs_a_camera():
    """alloy.camera.render(source, output='depth') without a Camera raises CameraError."""
    from alloy.camera import render

    class Engine:
        model = _newton_model()
        state = model.state()

    class FakeSim:
        engine = Engine()

    with pytest.raises(CameraError, match="Camera"):
        render(FakeSim(), output="depth")


@pytest.mark.gpu
def test_aov_render_inside_a_graph_capture_raises():
    """An AOV render while the device captures a CUDA graph raises CameraError, as colour renders do."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    model = _newton_model("cuda:0")
    cam = Camera("c", look_at=((0, 0, 1), (0, 0, 0), DOWN_UP), width=4, height=4, spp=1)
    with wp.ScopedCapture(device="cuda:0"):
        with pytest.raises(CameraError):
            cam.render(model, model.state(), output="depth")


@pytest.mark.gpu
def test_optix_matches_warp_and_leaves_colour_alone():
    """On OptiX, depth/ids equal the Warp backend's exactly, and each backend's uint8 is unchanged by an AOV render."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from alloy.camera.optix import available
    ok, why = available()
    if not ok:
        pytest.skip(f"OptiX backend not available: {why}")
    sc = camera_scene(_mixed_scene(), device="cuda:0")
    kw = dict(look_at=((0.6, -0.6, 0.5), (0, 0, 0.1)), width=16, height=16, spp=2)
    names = ("depth", "segmentation", "body_id", "world_id")
    res = {}
    for backend in ("warp", "optix"):
        cam = Camera(backend, backend=backend, **kw)
        before = _np(cam.render(sc, output="uint8"))
        res[backend] = {k: _np(getattr(cam.render(sc, output=names), k)) for k in names}
        assert np.array_equal(before, _np(cam.render(sc, output="uint8")))
    for k in names:
        assert np.array_equal(res["warp"][k], res["optix"][k])


def test_aovs_do_not_depend_on_spp_or_seed():
    """Depth and segmentation come from the pixel-centre ray: identical at spp 1 and 8 and with another seed."""
    sc = camera_scene(_mixed_scene(), device=DEV)
    kw = dict(look_at=((0.6, -0.6, 0.5), (0, 0, 0.1)), width=12, height=12)
    names = ("depth", "segmentation")
    a = Camera("a", spp=1, **kw).render(sc, output=names)
    b = Camera("b", spp=8, **kw).render(sc, output=names)
    c = Camera("c", spp=1, **kw).render(sc, output=names, seed=7)
    for k in names:
        assert np.array_equal(_np(getattr(a, k)), _np(getattr(b, k)))
        assert np.array_equal(_np(getattr(a, k)), _np(getattr(c, k)))


@pytest.mark.parametrize("bad", ["nope", ("rgb", "nope"), ()])
def test_unknown_or_empty_output_raises(bad):
    """An unknown name, or an empty tuple, raises CameraError."""
    sc = camera_scene(plane_scene(), device=DEV)
    with pytest.raises(CameraError):
        _down().render(sc, output=bad)


def test_segment_labels_of_scene_data():
    """A SceneData has anonymous bodies: ['background', 'no_body', 'body_0', ...]."""
    assert segment_labels(body_scene(3)) == ["background", "no_body", "body_0"]


def test_render_torch_gives_a_dict_of_tensors():
    """render_torch with a tuple output returns {name: torch tensor}."""
    torch = pytest.importorskip("torch")
    out = _down().render_torch(camera_scene(plane_scene(), device=DEV), output=("rgb", "depth"))
    assert set(out) == {"rgb", "depth"} and isinstance(out["depth"], torch.Tensor)
    assert torch.allclose(out["depth"], torch.ones_like(out["depth"]), rtol=1e-5)


def _tiny_sim():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.2, 0.2, 0.1), pos=(0.5, 0.5, 0.05), static=True)
    scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05), mass=0.2)
    scene.add_sphere("ball", 0.04, pos=(-0.4, 0, 0.04), mass=0.1)
    return alloy.build(scene, device=DEV)


def _state_arrays(sim):
    return {k: np.array(v, copy=True) for k, v in sim.state().items()}


@pytest.mark.integration
@pytest.mark.timeout(300)  # a cold kernel cache compiles the sim's kernels inside this test
def test_sim_labels_images_and_unchanged_physics():
    """sim.render gives numpy AOVs with Sim body names, and rendering never changes the state or the stepping."""
    sim, twin = _tiny_sim(), _tiny_sim()
    assert segment_labels(sim) == ["background", "no_body", "block", "ball"]
    cam = Camera("top", look_at=((0.0, 0.0, 0.6), (0.0, 0.0, 0.0), DOWN_UP), width=16, height=16, fov=30, spp=1)
    before = _state_arrays(sim)
    out = sim.render(cam, output=("rgb", "depth", "segmentation", "body_id", "world_id"))
    assert all(isinstance(v, np.ndarray) for v in out.values())
    after = _state_arrays(sim)
    assert all(np.array_equal(before[k], after[k]) for k in before)
    names = segment_labels(sim)
    assert names[int(out["segmentation"][0, 8, 8])] == "block"
    assert abs(float(out["depth"][0, 8, 8]) - (0.6 - 0.1)) < 1e-3
    for _ in range(5):
        sim.step(2)
        sim.render(cam, output=("depth", "segmentation"))
        twin.step(2)
    a, b = _state_arrays(sim), _state_arrays(twin)
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.integration
def test_api_ground_and_static_read_their_own_world():
    """A scene built with alloy.Scene gives each world its own ground and static objects: world_id is the world
    they are rendered in (never -1), and they are picked with segmentation 1 / body_id -1."""
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.1, 0.1, 0.1), pos=(0.1, 0.1, 0.05), static=True)
    scene.add_sphere("ball", 0.03, pos=(-0.08, -0.08, 0.03), mass=0.1)
    sim = alloy.build(scene, device=DEV)
    cam = Camera("top", look_at=((0.0, 0.0, 0.6), (0.0, 0.0, 0.0), DOWN_UP), width=16, height=16, fov=40, spp=1)
    out = sim.render(cam, output=("depth", "segmentation", "body_id", "world_id"))
    depth, fixed = out["depth"], out["segmentation"] == 1
    for w in range(2):
        assert fixed[w].any() and (out["world_id"][w] == w).all()
        assert (np.abs(depth[w][fixed[w]] - 0.6) < 1e-3).any() and (np.abs(depth[w][fixed[w]] - 0.5) < 1e-3).any()
    assert (out["body_id"][fixed] == -1).all() and (out["segmentation"] == 2).any()


@pytest.mark.integration
def test_cloth_sheet_reads_segmentation_one():
    """A cloth over the ground is a surface not on a body: segmentation 1 where the sheet is, in a Sim."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("sheet", size=(0.3, 0.3), pos=(0, 0, 0.3), resolution=8, mass=0.05)
    sim = alloy.build(scene, device=DEV)
    cam = Camera("top", look_at=((0.0, 0.0, 0.8), (0.0, 0.0, 0.0), DOWN_UP), width=16, height=16, fov=30, spp=1)
    out = sim.render(cam, output=("depth", "segmentation"))
    assert out["segmentation"][0, 8, 8] == 1 and abs(float(out["depth"][0, 8, 8]) - 0.5) < 5e-3


@pytest.mark.gpu
def test_cpu_and_gpu_agree():
    """The same scene on cuda:0 and cpu gives equal ids and depth within 1e-5."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    res = {}
    for dev in ("cuda:0", "cpu"):
        sc = camera_scene(body_scene(2), device=dev)
        out = _worlds_cam().render(sc, output=("depth", "segmentation", "body_id", "world_id"))
        res[dev] = {k: _np(getattr(out, k)) for k in ("depth", "segmentation", "body_id", "world_id")}
    for k in ("segmentation", "body_id", "world_id"):
        assert np.array_equal(res["cuda:0"][k], res["cpu"][k])
    assert np.allclose(res["cuda:0"]["depth"], res["cpu"]["depth"], rtol=1e-5, atol=1e-5)
