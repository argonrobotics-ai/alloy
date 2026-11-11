# SPDX-License-Identifier: Apache-2.0
"""The OptiX backend: build helper (licence gating, cache paths, availability reasons; no compiling), wavefront helpers,
and GPU checks against the Warp backend (skipped when OptiX is not built / not available)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.camera import Camera, RenderSettings
from alloy.camera.errors import CameraError
from alloy.camera.optix import available
from alloy.camera.optix import build as build_mod
from alloy.camera.optix.__main__ import main as optix_main
from alloy.camera.optix.scene import ias_slots
from alloy.camera.wavefront.knobs import WavefrontOptions
from alloy.camera.wavefront.sort import key_shift, spec_launches
from alloy.camera.wavefront.tracers import resolve_backend
from tests.camera.packs import principled_material, write_pack_v1, write_pack_v2


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty ALLOY_CACHE, no licence acceptance and no OPTIX_SDK."""
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv(build_mod.LICENCE_ENV, raising=False)
    monkeypatch.delenv("OPTIX_SDK", raising=False)
    return tmp_path / "cache"


def _no_subprocess(monkeypatch):
    def boom(*a, **k):
        raise AssertionError(f"a command ran without the licence being accepted: {a}")
    monkeypatch.setattr(build_mod.subprocess, "run", boom)


# ------------------------------------------------------------------------------------------------ build helper (CPU)
def test_build_refuses_without_licence(cache, monkeypatch):
    _no_subprocess(monkeypatch)
    with pytest.raises(CameraError) as exc:
        build_mod.build()
    msg = str(exc.value)
    assert "github.com/NVIDIA/optix-dev" in msg and "v9.0.0" in msg and "Apache-2.0" in msg
    assert "--accept-nvidia-optix-license" in msg
    assert not cache.exists()  # nothing was cloned or created


def test_cli_refuses_without_licence_and_prints_why(cache, monkeypatch, capsys):
    _no_subprocess(monkeypatch)
    assert optix_main(["build"]) == 2
    err = capsys.readouterr().err
    assert "SDK licence" in err and "not Apache-2.0" in err and "fetch" in err


def test_licence_flag_and_environment(monkeypatch):
    monkeypatch.delenv(build_mod.LICENCE_ENV, raising=False)
    assert not build_mod.licence_accepted() and build_mod.licence_accepted(True)
    monkeypatch.setenv(build_mod.LICENCE_ENV, "1")
    assert build_mod.licence_accepted()
    monkeypatch.setenv(build_mod.LICENCE_ENV, "0")
    assert not build_mod.licence_accepted()


def test_cache_paths(cache, monkeypatch):
    d = build_mod.build_dir()
    assert d.parent == cache / "camera" / "optix" and len(d.name) == 12
    assert build_mod.built_files() == (d / "libowf.so", d / "owf_programs.optixir")
    assert build_mod.sdk_dir() == cache / "camera" / "optix-dev-9.0.0"
    monkeypatch.setenv("OPTIX_SDK", str(cache / "mine"))
    assert build_mod.sdk_dir() == cache / "mine"


def test_cache_dir_follows_the_sources(tmp_path, monkeypatch, cache):
    src = tmp_path / "csrc"
    src.mkdir()
    (src / "a.cpp").write_text("one")
    monkeypatch.setattr(build_mod, "CSRC", src)
    first = build_mod.csrc_hash()
    assert build_mod.csrc_hash() == first
    (src / "a.cpp").write_text("two")
    assert build_mod.csrc_hash() != first


def test_accepted_build_fetches_compiles_and_publishes_atomically(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(build_mod, "_fetch_headers", lambda sdk: calls.append(("fetch", sdk)))
    monkeypatch.setattr(build_mod, "find_compilers", lambda: ("g++", build_mod.Path("/cuda/bin/nvcc")))

    def fake_compile(out, sdk, cxx, nvcc):
        calls.append(("compile", sdk))
        (out / build_mod.LIB_NAME).write_bytes(b"lib")
        (out / build_mod.PROGRAMS_NAME).write_bytes(b"ir")
    monkeypatch.setattr(build_mod, "compile_all", fake_compile)
    out = build_mod.build(accept_licence=True)
    assert out == build_mod.build_dir() and build_mod.is_built()
    assert [c[0] for c in calls] == ["fetch", "compile"]
    assert [p.name for p in out.parent.iterdir()] == [out.name]  # no half-built temp directories left
    assert build_mod.build(accept_licence=True) == out and len(calls) == 2  # cached: no second build


def test_existing_headers_need_no_acceptance(cache, monkeypatch):
    sdk = cache / "mine"
    (sdk / "include").mkdir(parents=True)
    (sdk / "include" / "optix.h").write_text("//")
    monkeypatch.setenv("OPTIX_SDK", str(sdk))
    monkeypatch.setattr(build_mod, "find_compilers", lambda: (_ for _ in ()).throw(CameraError("no g++", "install it")))
    with pytest.raises(CameraError, match="no g\\+\\+"):  # got past the licence gate to the compilers
        build_mod.build()


def test_not_available_reason_says_how_to_build(cache):
    ok, why = available()
    assert not ok and "python -m alloy.camera.optix build --accept-nvidia-optix-license" in why


def test_missing_driver_library_names_the_package(cache, monkeypatch):
    import alloy.camera.optix as ox
    lib, ir = build_mod.built_files()
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"x")
    ir.write_bytes(b"x")
    monkeypatch.setattr(ox, "_driver_has_optix", lambda: False)
    ok, why = available()
    assert not ok and "libnvoptix.so.1" in why and "libnvidia-gl-<driver version>-server" in why


def test_gpu_scene_without_build_says_what_to_run(cache):
    from alloy.camera.tracer.gpu_scene import GpuScene
    from tests.camera.helpers import plane_scene
    sc = GpuScene(plane_scene(), device="cpu")
    with pytest.raises(CameraError, match="optix build"):
        sc.enable_optix()


def test_backend_names(cache):
    assert resolve_backend("warp") == "warp" and resolve_backend("wavefront-warp") == "wavefront-warp"
    assert resolve_backend("auto", "cpu") == "warp"
    with pytest.raises(CameraError, match="unknown render backend"):
        resolve_backend("vulkan")


# ------------------------------------------------------------------------------------------------ wavefront helpers (CPU)
def test_ias_slots_list_shared_instances_first_and_give_each_world_a_mask_bit():
    group = np.array([0, 0, 1, 2, 2, 3, 3, 3])  # worlds 0..2 own instances, 3 = shared by all
    slots, masks, off, cnt = ias_slots(group, 3, 8)
    assert slots.tolist() == [5, 6, 7, 0, 1, 2, 3, 4] and off == [0] and cnt == [8]
    assert masks.tolist() == [0xFF] * 3 + [1, 1, 2, 4, 4]
    slots, masks, off, cnt = ias_slots(group, 3, 1)  # one IAS per world: every mask is 0xFF
    assert off == [0, 5, 9] and cnt == [5, 4, 5] and set(masks.tolist()) == {0xFF}
    slots, masks, off, cnt = ias_slots(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 10, 8)  # 10 worlds: two IAS
    assert off == [0, 9] and cnt == [9, 3] and slots.tolist()[9:] == [10, 8, 9] and masks.tolist()[9:] == [0xFF, 1, 2]


def test_paths_per_wave_follows_gpu_memory(monkeypatch):
    import types

    import alloy.camera.wavefront.knobs as K
    gb = 1 << 30
    for mem, large, want in ((24 * gb, True, 1 << 22), (48 * gb, True, 1 << 24), (96 * gb, True, 1 << 25),
                             (96 * gb, False, 1 << 24)):
        monkeypatch.setattr(K.wp, "get_device", lambda d, m=mem: types.SimpleNamespace(total_memory=m))
        assert WavefrontOptions().paths_per_wave("cuda:0", large) == want


def test_wavefront_options_from_environment(monkeypatch):
    assert WavefrontOptions.from_env() == WavefrontOptions()
    monkeypatch.setenv("ALLOY_CAMERA_WF_SPEC", "0")
    monkeypatch.setenv("ALLOY_CAMERA_WF_BEVEL", "trace")
    monkeypatch.setenv("ALLOY_CAMERA_WF_MAX_PATHS", "4096")
    o = WavefrontOptions.from_env()
    assert not o.spec and o.bevel == "trace" and o.max_paths == 4096 and o.shade_warps == 12 and o.sort
    assert o.paths_per_wave("cpu", True) == 4096
    monkeypatch.setenv("ALLOY_CAMERA_WF_SORT", "maybe")
    with pytest.raises(CameraError, match="ALLOY_CAMERA_WF_SORT"):
        WavefrontOptions.from_env()
    with pytest.raises(CameraError, match="Bevel mode"):
        WavefrontOptions(bevel="inline")


def test_material_ranges_of_the_sorted_queue():
    assert key_shift(4) == 29 and key_shift(34) == 25
    hist = [5000, 10, 20, 0, 6000]  # 3 materials, then misses (bin 3: empty) and terminated paths (bin 4)
    assert spec_launches(hist, 3, 4096) == [(0, 0, 5000), (-2, 5000, 30), (-1, 5030, 6000)]
    assert spec_launches([0, 0, 0], 1, 10) == []


# ------------------------------------------------------------------------------------------------ GPU
@pytest.fixture(scope="module")
def optix_device():
    ok, why = available()
    if not ok:
        pytest.skip(f"OptiX backend not available: {why}")
    return "cuda:0"


def _with_backend(st: RenderSettings, backend: str) -> RenderSettings:
    return dataclasses.replace(st, backend=backend)


@pytest.mark.gpu
def test_diffuse_pack_means_agree(optix_device, tmp_path):
    from alloy.camera.pack_v1 import scene_from_pack_v1
    from alloy.camera.tracer.gpu_scene import GpuScene
    from alloy.camera.tracer.renderer import Renderer
    root = write_pack_v1(tmp_path / "p1")
    means = {}
    for backend in ("warp", "optix", "wavefront-warp"):
        sc = GpuScene(scene_from_pack_v1(root, num_worlds=2), device=optix_device)
        r = Renderer(sc, 32, 24, RenderSettings(seed=1, backend=backend))
        r.set_cameras(*_pack_view(sc, 2, 32, 24))
        rgb, alb, nrm = (a.numpy() for a in r.render(spp=32, seed=1))
        assert np.isfinite(rgb).all()
        means[backend] = (rgb.mean(), alb.mean())
    for b in ("optix", "wavefront-warp"):
        assert means[b] == pytest.approx(means["warp"], rel=1e-3)


def _pack_view(sc, n, w, h):
    """eye, rot, extent, clip arrays of the v1 pack's camera for ``n`` images."""
    cam = sc.sd.camera
    from alloy.camera.scene import quat_to_matrix
    rot = quat_to_matrix(cam["pose"][3:])
    ext = np.tan(0.5 * np.radians(cam["fov_deg"]))
    extent = (ext * w / h, ext) if cam["fit"] == "VERTICAL" else (ext, ext * h / w)
    return (np.tile(cam["pose"][:3], (n, 1)), np.tile(rot, (n, 1, 1)), np.tile(extent, (n, 1)),
            np.tile(cam["clip"], (n, 1)))


@pytest.mark.gpu
@pytest.mark.parametrize("light", [0.0, 50.0])
def test_pbr_pack_means_agree(optix_device, tmp_path, light):
    from alloy.camera.pbr import PbrRenderer, PbrScene
    mat = principled_material((0.5, 0.5, 0.5), specular=0.5, roughness=0.4)
    root = write_pack_v2(tmp_path / "p2", mat, sky=0.4, light_energy=light)
    out = {}
    for backend in ("warp", "optix"):
        ps = PbrScene.from_pack(root, device=optix_device, num_worlds=2)
        r = PbrRenderer(ps, 24, 24, _with_backend(ps.settings, backend))
        out[backend] = [a.numpy().reshape(-1, 3).astype(np.float64) for a in r.render(spp=128, seed=3)]
    for k in range(3):  # radiance, albedo, normal
        assert np.allclose(out["optix"][k].mean(0), out["warp"][k].mean(0), rtol=0.03, atol=2e-3)


@pytest.mark.gpu
def test_newton_model_with_physics_matches_warp(optix_device):
    from alloy._vendor import newton
    import warp as wp
    b = newton.ModelBuilder()
    b.add_ground_plane()
    box = b.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_rpy(0.3, 0.2, 0.1)), label="box")
    b.add_shape_box(box, hx=0.08, hy=0.06, hz=0.04)
    s = newton.ModelBuilder()
    s.replicate(b, 16)
    model = s.finalize(device=optix_device)
    solver = newton.solvers.SolverXPBD(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    kw = dict(look_at=((0.7, -0.7, 0.5), (0.0, 0.0, 0.1)), width=32, height=32, fov=45, spp=32)
    cams = {b: Camera("front", settings=RenderSettings(backend=b), **kw) for b in ("warp", "optix")}
    first = cams["optix"].render(model, s0).numpy().astype(float)
    for _ in range(20):
        solver.step(s0, s1, ctrl, model.collide(s0), 1.0 / 60.0)
        s0, s1 = s1, s0
    later = {k: c.render(model, s0).numpy().astype(float) for k, c in cams.items()}
    assert first.shape == (16, 32, 32, 3) and not np.array_equal(first, later["optix"])  # the IAS followed the box
    assert abs(later["optix"].mean() - later["warp"].mean()) < 0.03 * later["warp"].mean()
    assert abs(first.mean() - later["optix"].mean()) > 0.0


@pytest.mark.gpu
def test_wavefront_buffers_grow_with_spp_and_images_do_not_depend_on_earlier_renders(tmp_path):
    import warp as wp

    from alloy.camera.pack_v1 import scene_from_pack_v1
    from alloy.camera.tracer.gpu_scene import GpuScene
    from alloy.camera.tracer.renderer import Renderer
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device")
    sc = GpuScene(scene_from_pack_v1(write_pack_v1(tmp_path / "p1"), num_worlds=1), device="cuda:0")
    rs = []
    for _ in range(2):
        r = Renderer(sc, 32, 24, RenderSettings(seed=1, backend="wavefront-warp"))
        r.set_cameras(*_pack_view(sc, 1, 32, 24))
        rs.append(r)
    fresh = [a.numpy().copy() for a in rs[0].render(spp=8, seed=1)]
    rs[1].render(spp=64, seed=2)
    used = [a.numpy().copy() for a in rs[1].render(spp=8, seed=1)]
    assert rs[0].wf.wf.n_paths == 32 * 24 * 8 and rs[1].wf.wf.n_paths == 32 * 24 * 64
    for a, b in zip(fresh, used):
        assert np.array_equal(a, b)


@pytest.mark.gpu
def test_a_dropped_wavefront_renderer_frees_at_once(tmp_path):
    import gc
    import weakref

    import warp as wp

    from alloy.camera import Camera, load_pack
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device")
    packs = (write_pack_v1(tmp_path / "p1"), write_pack_v2(tmp_path / "p2", principled_material()))
    for root in packs:  # first use imports kernel modules (Warp's decorators hold that call stack until gc runs)
        Camera("c", width=16, height=12, spp=2, backend="wavefront-warp").render(load_pack(root, device="cuda:0"))
    gc.collect()
    gc.disable()
    try:
        for root in packs:
            cam = Camera("c", width=16, height=12, spp=2, backend="wavefront-warp")
            cam.render(load_pack(root, device="cuda:0"))
            ref = weakref.ref(cam._bound[1])
            del cam
            assert ref() is None  # freed by reference counting, without waiting for the cycle collector
    finally:
        gc.enable()


@pytest.mark.gpu
def test_optix_breaks_distance_ties_like_warp(optix_device):
    # the any-hit traces over the tie window: glass 0.2 um above a table (float64 plane distances) or a real gap is
    # in front, 0.2 um below it is not; an exact tie is the same per-ray coin on both backends
    from tests.camera.glass_packs import tie_hits
    for base in (0.0, 0.002):
        for gap, want in ((1e-3, 1), (2e-7, 1), (-2e-7, 0)):
            assert (tie_hits("optix", gap, optix_device, base=base) == want).all(), (base, gap)
            assert (tie_hits("wavefront-warp", gap, optix_device, base=base) == want).all(), (base, gap)
        coin = tie_hits("wavefront-warp", 0.0, optix_device, base=base, n_side=20)
        assert np.array_equal(tie_hits("optix", 0.0, optix_device, base=base, n_side=20), coin) and 0 < coin.mean() < 1


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["glass", "subsurface"])
def test_see_through_materials_agree_across_backends(optix_device, tmp_path, kind):
    # absorbing glass (volume stack, Beer-Lambert) and a random-walk cube: same means from the megakernel and both
    # wavefront tracers
    from alloy.camera.pbr import PbrRenderer, PbrScene
    from tests.camera.glass_packs import glass_material, sss_material, write_glass_pack
    mat = sss_material(1.0, 0.01)
    if kind == "glass":
        mat = glass_material(0.2, absorb_color=(0.4, 0.6, 0.9), density=8.0)
    half = (0.5, 0.5, 0.1) if kind == "glass" else (0.3, 0.3, 0.3)
    root = write_glass_pack(tmp_path / kind, mat, half=half, sky=0.5, eye_z=2.0)
    means = {}
    for backend in ("warp", "optix", "wavefront-warp"):
        ps = PbrScene.from_pack(root, device=optix_device, num_worlds=1)
        r = PbrRenderer(ps, 24, 24, _with_backend(ps.settings, backend))
        means[backend] = r.render(spp=256, seed=4)[0].numpy().reshape(-1, 3).astype(np.float64).mean(0)
    for b in ("optix", "wavefront-warp"):
        assert np.allclose(means[b], means["warp"], rtol=0.02)
