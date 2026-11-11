# SPDX-License-Identifier: Apache-2.0
"""SPH liquids in the batched camera (CPU): per-world surfaces built on the device (closed, outward, volume-true,
streams joined), drawn live exactly as a scene built fresh at the same state, with water / wine looks and in packs."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.camera import Camera
from alloy.camera.camera import _forget, camera_scene
from alloy.camera.liquid_gpu import LiquidObject, LiquidSurface
from alloy.camera.liquid_looks import LN10_PER_CM, fast_color, liquid_key, liquid_material
from alloy.camera.liquid_scene import has_liquids, liquid_objects

DX = 0.004
LOOK_AT = ((0.16, -0.16, 0.5), (0.0, 0.0, 0.36))


def _surfaces(sets: list, vel: list | None = None):
    """One world per particle set; returns the LiquidSurface after one build."""
    n = max(len(p) for p in sets) + 3
    pq = np.zeros((len(sets), n, 3), np.float32)
    qd = np.zeros_like(pq)
    fl = np.zeros((len(sets), n), np.int32)
    for w, p in enumerate(sets):
        pq[w, : len(p)], fl[w, : len(p)] = p, 1
        if vel is not None:
            qd[w, : len(p)] = vel[w]
    s = LiquidSurface(LiquidObject("water", 0, n, DX), len(sets), n, "cpu")
    arr = lambda a, t: wp.array(a.reshape(-1, *a.shape[2:]), dtype=t, device="cpu")  # noqa: E731
    s.build(arr(pq, wp.vec3), arr(fl, wp.int32), arr(qd, wp.vec3) if vel is not None else None)
    return s


def _components(v: np.ndarray, f: np.ndarray) -> int:
    parent = np.arange(len(v))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in np.concatenate([f[:, [0, 1]], f[:, [1, 2]]]):
        parent[root(a)] = root(b)
    return len({root(i) for i in range(len(v))})


def _ball():
    g = np.arange(-0.03, 0.0301, DX)
    x = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return x[np.linalg.norm(x, axis=1) <= 0.03]


def _plates():
    """A falling stream as a pour leaves it: 6 x 6 layers drifting apart (5 to 16 mm)."""
    g = np.arange(6) * DX
    pl = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    zs = np.cumsum([0, 5, 6.3, 7.4, 9.9, 10.1, 11.3, 12.6, 15.5, 15.2, 16.4]) * 1e-3
    return np.concatenate([np.c_[pl, np.full(len(pl), z)] for z in zs])


def test_a_ball_of_particles_is_one_closed_outward_surface_of_its_volume():
    s = _surfaces([_ball(), np.zeros((0, 3))])
    v, f, vn = s.mesh(0)
    p = v[f]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    edges = {tuple(e) for e in np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])}
    assert all((b, a) in edges for a, b in edges)  # closed: every edge once each way
    assert (np.einsum("ij,ij->i", fn, p.mean(1) - v.mean(0)) > 0).all()  # outward
    assert (np.einsum("ij,ij->i", vn[f[:, 0]], fn) > 0).all()  # vertex normals on the outside
    volume = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0
    assert volume == pytest.approx(len(_ball()) * DX ** 3, rel=0.06)
    assert s.mesh(1)[1].shape == (0, 3) and s.counts[1][1] == 1  # an empty world keeps one degenerate triangle


def test_a_falling_stream_is_one_surface_not_a_stack_of_plates():
    plates = _plates()
    still = _surfaces([plates])
    moving = _surfaces([plates], vel=[np.array([0.0, 0.0, -1.5], np.float32)])
    assert _components(*still.mesh(0)[:2]) > 5
    assert _components(*moving.mesh(0)[:2]) == 1


def test_the_field_near_the_particles_only_keeps_every_bit(monkeypatch):
    import alloy.camera.liquid_gpu as lg
    sets, vel = [_ball(), _plates()], [np.zeros(3, np.float32), np.array([0.0, 0.0, -1.5], np.float32)]
    sparse = _surfaces(sets, vel)
    assert sparse.g.blocks.numpy().mean() < 0.9  # most blocks are skipped
    monkeypatch.setattr(lg, "DENSE_FIELD", True)
    dense = _surfaces(sets, vel)
    n = len(sets) * dense.g.nn
    assert np.array_equal(sparse.g.field.numpy()[:n], dense.g.field.numpy()[:n])
    for w in range(len(sets)):
        assert all(np.array_equal(a, b) for a, b in zip(sparse.mesh(w), dense.mesh(w)))


def test_a_surface_over_the_capacity_warns_with_the_knob(monkeypatch, caplog):
    monkeypatch.setenv("ALLOY_CAMERA_LIQUID_TRIS", "1024")
    with caplog.at_level(logging.WARNING):
        s = _surfaces([_ball()])
    assert s.counts[1][0] == 1024 and "ALLOY_CAMERA_LIQUID_TRIS" in caplog.text


def test_wine_absorbs_by_its_glories_indices_and_water_stays_clear():
    assert [liquid_key(m) for m in ("water", "red_wine", "wine_rose", "white wine", "olive_oil")] == [
        "water", "wine_red", "wine_rose", "wine_white", "water"]
    mat = liquid_material("wine_red")
    (vol,) = [n for n in mat["tree"]["nodes"] if n["type"] == "ShaderNodeVolumeAbsorption"]
    ins = {i["id"]: i["value"] for i in vol["inputs"]}
    sigma = [(1.0 - c) * ins["Density"] for c in ins["Color"][:3]]
    assert sigma == pytest.approx([0.8 * LN10_PER_CM, 3.6 * LN10_PER_CM, 3.3 * LN10_PER_CM], rel=1e-6)
    assert any(lk["to_id"] == "Volume" for lk in mat["tree"]["links"])
    assert mat["principled"]["inputs"]["Transmission Weight"]["value"] == 1.0
    assert fast_color("water")[2] > fast_color("water")[0]  # the fast tracer's water reads blue


@pytest.fixture(scope="module")
def pour_sim():
    s = alloy.Scene(dt=0.01, worlds=2)
    s.add_ground()
    s.add_box("table", size=(0.3, 0.3, 0.3), pos=(0, 0, 0.15), static=True)
    s.add_open_cylinder("cup", inner_radius=0.035, depth=0.06, wall=0.004, pos=(0, 0, 0.3))
    s.add_pour("water", 3.0e-5, pos=(0.0, 0.0, 0.45), rate=2e-4, speed=0.3)
    sim = alloy.build(s, device="cpu")
    sim.step(12)
    return sim


def test_the_pour_is_drawn_live_exactly_as_a_fresh_scene(pour_sim):
    lay = liquid_objects(pour_sim)
    (o,) = lay.objects
    assert has_liquids(pour_sim) and (o.name, o.material, lay.worlds) == ("water", "water", 2)
    assert o.spacing == pytest.approx(DX, rel=1e-6)
    cam = Camera("c", look_at=LOOK_AT, width=32, height=32, spp=2)
    cam.render(pour_sim)
    pour_sim.step(4)
    live = cam.render(pour_sim).numpy().copy()
    gs = camera_scene(pour_sim, device="cpu")
    assert gs.liquid.surfaces[0].counts[1].min() > 100  # a surface in every world
    _forget(gs)
    fresh = Camera("c", look_at=LOOK_AT, width=32, height=32, spp=2).render(pour_sim).numpy()
    assert np.array_equal(live, fresh)


def test_a_sim_pack_carries_the_liquid_surface_with_its_look(pour_sim, tmp_path):
    from alloy.camera.pack_write import write_sim_pack
    root = write_sim_pack(pour_sim, tmp_path / "pack", LOOK_AT)
    frame = json.loads((root / "frames/0/frame.json").read_text())
    (water,) = [m for m in frame["meshes"] if m["object_name"] == "water"]
    assert water["materials"] == ["liquid_water"] and water["tris"] > 100
    static = json.loads((root / "pack.json").read_text())
    nodes = static["materials"]["liquid_water"]["tree"]["nodes"]
    assert any(n["type"] == "ShaderNodeVolumeAbsorption" for n in nodes)
    m = np.load(root / water["file"])
    assert m["n"].shape == (water["tris"], 3, 3) and np.allclose(np.linalg.norm(m["n"], axis=-1), 1.0, atol=1e-5)


def _pour(device: str, worlds: int = 2):
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_open_cylinder("cup", inner_radius=0.035, depth=0.06, wall=0.004, pos=(0.0, 0.0, 0.0))
    s.add_pour("water", 3.0e-5, pos=(0.0, 0.0, 0.15), rate=2e-4, speed=0.3)
    return alloy.build(s, device=device, seed=0)


def _state_bits(sim) -> list[np.ndarray]:
    st = sim.engine.state
    return [a.numpy().copy() for a in (st.particle_q, st.particle_qd) if a is not None]


def _render_on_off(device: str, cameras: list) -> None:
    """Gate law: the sim is bitwise the same whether the camera renders between its steps or not. The two sims run
    one after the other: two SPH sims stepping in turn in one process do not reproduce each other on the GPU (their
    captured graphs share Warp's hash grid descriptor copy; see liquid_kernels.to_f64)."""
    on = _pour(device)
    for _ in range(3):
        on.step(4)
        for cam, quality in cameras:
            cam.render(camera_scene(on, quality=quality, device=device), on.engine.state)
    bits = _state_bits(on)
    del on
    off = _pour(device)
    for _ in range(3):
        off.step(4)
    for a, b in zip(bits, _state_bits(off)):
        assert np.array_equal(a, b)


def test_rendering_leaves_the_sim_bitwise_unchanged():
    _render_on_off("cpu", [(Camera("c", look_at=((0.16, -0.16, 0.2), (0, 0, 0.06)), width=16, height=16, spp=1),
                            "fast")])


@pytest.mark.gpu
def test_optix_rebuilds_the_liquid_every_render_as_a_fresh_scene():
    from alloy.camera.optix import available
    if not wp.is_cuda_available() or not available("cuda:0")[0]:
        pytest.skip("needs a CUDA device with the OptiX helper built")
    s = alloy.Scene(dt=0.01, worlds=2)
    s.add_ground()
    s.add_open_cylinder("cup", inner_radius=0.035, depth=0.06, wall=0.004, pos=(0, 0, 0.0))
    s.add_pour("water", 3.0e-5, pos=(0.0, 0.0, 0.15), rate=2e-4, speed=0.3)
    sim = alloy.build(s, device="cuda:0")
    sim.step(12)
    la = ((0.16, -0.16, 0.2), (0.0, 0.0, 0.06))
    for quality in ("fast", "full"):
        cam = Camera("c", look_at=la, width=48, height=48, spp=4, backend="optix")
        sc = camera_scene(sim, quality=quality)
        cam.render(sc, sim.engine.state)
        sim.step(3)
        live = cam.render(sc, sim.engine.state).numpy().copy()
        _forget(sc)
        fresh = Camera("c", look_at=la, width=48, height=48, spp=4, backend="optix").render(
            camera_scene(sim, quality=quality), sim.engine.state).numpy()
        assert np.array_equal(live, fresh), quality
        sim.step(3)  # the solver's own hash grid still works after the camera's (liquid_kernels.to_f64)
        assert sim.check()["ok"]


@pytest.mark.gpu
def test_on_the_gpu_rendering_leaves_the_sim_bitwise_unchanged_and_refuses_inside_a_capture():
    from alloy.camera.errors import CameraError
    from alloy.camera.optix import available
    if not wp.is_cuda_available() or not available("cuda:0")[0]:
        pytest.skip("needs a CUDA device with the OptiX helper built")
    la = ((0.16, -0.16, 0.2), (0.0, 0.0, 0.06))
    _render_on_off("cuda:0", [(Camera("f", look_at=la, width=32, height=32, spp=2, backend=b), q)
                              for b in ("optix", "warp") for q in ("fast", "full")])
    sim = _pour("cuda:0", worlds=1)
    sim.step(4)
    cam = Camera("c", look_at=la, width=16, height=16, spp=1)
    with pytest.raises(CameraError, match="captures a CUDA graph"):
        with wp.ScopedCapture(device="cuda:0"):
            cam.render(sim)
