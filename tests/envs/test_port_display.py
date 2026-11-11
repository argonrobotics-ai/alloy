# SPDX-License-Identifier: Apache-2.0
"""The port envs' cameras draw the gift bags' paper and the porridge (RENDER-GA-4), CPU.

The drawn mask is where the first hit changes (albedo or normal) when the deformables are added, through the env's
own path (the paper's ``DeformSet``, the porridge's smooth ``LiquidSet`` surface, both from the state), to the model's
scene with its rigid shapes parked away (so none hides what the projection counts); the projected mask is the sim's
own geometry (the paper's triangles, the porridge particles' spheres) projected through the same top-down camera.
The open-flaps carton is rigid hinged panels (drawn as shapes already), so the paper is the bag-candle's kraft bag
(film route).
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import make
from alloy.envs.errors import EnvError
from alloy.envs.ports import PORTS
from alloy.envs.ports.env import _not_capturing

SIZE, SPP, FOV = 192, 4, 40.0


def _expert(name: str, steps: int, camera=None):  # noqa: ANN001, ANN202
    env = make(name, device="cpu", camera=camera, spp=1)
    obs, info = env.reset(seed=0)
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(info["expert_action"])
    return env


def _points(sim, layout) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:  # noqa: ANN001
    """World 0's deformable now: (points, triangles or None, sphere radii or None)."""
    from alloy.envs.ports.display import Liquid  # noqa: PLC0415

    o = (layout.layout if isinstance(layout, Liquid) else layout).objects[0]
    q = sim.state.particle_q.numpy()[o.start:o.start + o.count].astype(np.float64)
    if getattr(o, "kind", None) == "surface":
        return q, o.tris, None
    return q, None, sim.model.particle_radius.numpy()[o.start:o.start + o.count].astype(np.float64)


def _project(p: np.ndarray, eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raster x (from the left), raster y (from the bottom) and depth, as the tracer's camera_ray inverts them."""
    from alloy.camera.pose import image_extent, look_at_matrix  # noqa: PLC0415

    c = (p - eye) @ look_at_matrix(eye, target)
    depth = -c[:, 2]
    hx, hy = image_extent(FOV, SIZE, SIZE)
    return (c[:, 0] / depth / hx + 1.0) * 0.5 * SIZE, (c[:, 1] / depth / hy + 1.0) * 0.5 * SIZE, depth


def _projected(p, tris, radii, eye, target) -> np.ndarray:  # noqa: ANN001
    rx, ry, depth = _project(p, eye, target)
    cx, cy = np.meshgrid(np.arange(SIZE) + 0.5, np.arange(SIZE) + 0.5)        # pixel centres, y from the bottom
    mask = np.zeros((SIZE, SIZE), bool)
    if tris is not None:
        for a, b, c in tris:
            xs, ys = rx[[a, b, c]], ry[[a, b, c]]
            d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(d) < 1e-12:
                continue
            l1 = ((ys[1] - ys[2]) * (cx - xs[2]) + (xs[2] - xs[1]) * (cy - ys[2])) / d
            l2 = ((ys[2] - ys[0]) * (cx - xs[2]) + (xs[0] - xs[2]) * (cy - ys[2])) / d
            mask |= (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
    else:
        from alloy.camera.pose import image_extent  # noqa: PLC0415

        rpx = radii / depth / image_extent(FOV, SIZE, SIZE)[1] * 0.5 * SIZE
        for x, y, r in zip(rx, ry, rpx, strict=True):
            mask |= (cx - x) ** 2 + (cy - y) ** 2 <= r * r
    return mask[::-1]                                                          # rows from the top, as the image


def _scenes(sim, layout) -> tuple:  # noqa: ANN001
    """(the deformables alone, nothing): the model's shapes parked 10 km away, with and without the deformables."""
    from alloy.camera.from_newton import scene_from_model  # noqa: PLC0415
    from alloy.camera.tracer.gpu_scene import GpuScene  # noqa: PLC0415

    from alloy.envs.ports.display import deform_scene  # noqa: PLC0415

    sd = scene_from_model(sim.model)
    sd.inst_body[:] = -1
    sd.inst_local[:, :3] = 1.0e4
    return deform_scene(sim, layout, rigid=False), GpuScene(sd, device="cpu")


def _drawn(sim, layout, eye, target) -> np.ndarray:  # noqa: ANN001
    from alloy.camera import Camera  # noqa: PLC0415

    bufs = []
    for scene in _scenes(sim, layout):
        cam = Camera("iou", look_at=(eye, target), width=SIZE, height=SIZE, spp=SPP, fov=FOV)
        out = cam.render(scene, sim.state, output="all")
        bufs.append(np.concatenate([out.albedo.numpy()[0], out.normal.numpy()[0]], axis=-1))
    return np.abs(bufs[0] - bufs[1]).max(axis=-1) > 0.25


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
@pytest.mark.parametrize(("name", "steps"), [("GiftBagCandle-v0", 1), ("CookReheatPorridge-v0", 3)])
def test_the_camera_draws_the_paper_and_the_porridge(name: str, steps: int) -> None:
    env = _expert(name, steps)
    sim = env._slots[0].run.sim
    layout = PORTS[name].display(sim)
    p, tris, radii = _points(sim, layout)
    target = 0.5 * (p.min(0) + p.max(0))
    eye = target + np.array([1e-3, 0.0, 1.6 * float(np.linalg.norm(p.max(0) - p.min(0)))])  # straight down
    drawn, projected = _drawn(sim, layout, eye, target), _projected(p, tris, radii, eye, target)
    iou = (drawn & projected).sum() / (drawn | projected).sum()
    assert projected.sum() > 500, "the deformable must cover a fair share of the image"
    assert iou >= 0.9, f"{name}: drawn-mask IoU {iou:.3f}"
    env.close()


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["GiftBagCandle-v0", "CookReheatPorridge-v0"])
def test_rendering_leaves_the_sim_bitwise(name: str) -> None:
    states = []
    for camera in (None, (32, 32)):
        env = _expert(name, 4, camera)
        st = env._slots[0].run.sim.state
        states.append([st.joint_q.numpy().copy(), st.body_q.numpy().copy(), st.particle_q.numpy().copy()])
        env.close()
    for a, b in zip(*states, strict=True):
        assert np.array_equal(a, b)


def test_envs_without_deformables_render_their_model_as_before() -> None:
    """A port without deformables renders its engine sim; with no surface film that is byte for byte its model."""
    from tests.envs.test_ports import SLIDE  # noqa: PLC0415

    from alloy.camera import Camera  # noqa: PLC0415
    from alloy.envs.ports import PortEnv  # noqa: PLC0415

    env = PortEnv(SLIDE, num_envs=1, device="cpu", camera=(24, 16), spp=1)
    obs, info = env.reset(seed=0)
    slot = env._slots[0]
    sim = slot.run.sim
    assert slot.scene is sim
    view = slot.camera._pose
    fresh = Camera("fresh", pose=view[0] if view.ndim == 2 else view, width=24, height=16, spp=1)
    assert np.array_equal(fresh.render(sim.model, sim.state).numpy(), obs["image"])
    env.close()


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_a_painting_env_draws_its_film() -> None:
    """PaintRollStrip-v0 (small wall): the wall starts with wet cut-in bands and the strip rolled before, so the env's
    image (its engine sim, films drawn) differs from its model's alone."""
    from alloy.camera import Camera  # noqa: PLC0415

    env = make("PaintRollStrip-v0", device="cpu", size="small", camera=(48, 48), spp=1)
    obs, info = env.reset(seed=0)
    env.step(info["expert_action"])
    slot = env._slots[0]
    sim = slot.run.sim
    view = slot.camera._pose
    plain = Camera("plain", pose=view[0] if view.ndim == 2 else view, width=48, height=48, spp=1)
    with_film = Camera("film", pose=view[0] if view.ndim == 2 else view, width=48, height=48, spp=1)
    assert not np.array_equal(plain.render(sim.model, sim.state).numpy(), with_film.render(sim).numpy())
    env.close()


def test_no_render_inside_a_graph_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    import warp as wp  # noqa: PLC0415

    class Capturing:
        is_cuda, is_capturing = True, True

    class Sim:
        class model:  # noqa: N801
            device = "cuda:0"

    monkeypatch.setattr(wp, "get_device", lambda *a: Capturing())
    with pytest.raises(EnvError, match="graph capture"):
        _not_capturing(Sim())
