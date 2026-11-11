# SPDX-License-Identifier: Apache-2.0
"""UV-mapped image textures and corner colour attributes in the full-material tracer, and ``mesh_scene``: a checker
quad through a pack directory and through ``mesh_scene``, the four ways to ask for the UV, one module for every
mesh scene (CPU; the GPU test compares the Warp and OptiX backends)."""
from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from alloy.camera import Camera, CameraError, TexturedMesh, load_pack, mesh_scene
from alloy.camera.pbr.codegen import gen_material
from alloy.camera.pbr.dispatch import cache_dir, generate_source
from tests.camera.packs import (UV_SOURCES, attribute_colour_material, checker_image, principled_material,
                                textured_material, write_pack_v2, write_textured_pack)

log = logging.getLogger(__name__)
QUAD = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
TRIS = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)  # (0, 0) at the quad's bottom-left
TOP = (0.5, 0.5, 2.0), (0.5, 0.5, 0.0), (0.0, 1.0, 0.0)


def _quad_camera(size: int = 40, **kw) -> Camera:
    return Camera("c", look_at=TOP, width=size, height=size, spp=kw.pop("spp", 8), **kw)


def _quadrants(img: np.ndarray) -> dict:
    """Mean colour at the centre of each quadrant of the quad (it covers the middle ~70 % of the picture)."""
    n = img.shape[0]
    lo, hi = int(0.5 * n - 0.17 * n), int(0.5 * n + 0.17 * n)
    return {"tl": img[lo, lo], "tr": img[lo, hi], "bl": img[hi, lo], "br": img[hi, hi]}


def _assert_checker(img: np.ndarray) -> None:
    q = _quadrants(img)  # image row 0 = top: red top-left, green top-right, blue bottom-left, white bottom-right
    for key, ch in (("tl", 0), ("tr", 1), ("bl", 2)):
        c = np.asarray(q[key], np.float64)
        assert c.argmax() == ch and c[ch] > 2.0 * np.delete(c, ch).max(), (key, c)
    w = np.asarray(q["br"], np.float64)
    assert w.min() > 0.5 * w.max(), w


def _checker_scene(**kw):
    return mesh_scene([TexturedMesh("quad", QUAD, TRIS, uv=UV, texture=checker_image(), **kw)], device="cpu")


def test_mesh_scene_checker_quadrants():
    """UV (0, 0) is the bottom-left and image row 0 is the top: the stored picture appears upright."""
    img = _quad_camera().render(_checker_scene(), output="linear").numpy()[0]
    _assert_checker(img)


@pytest.mark.parametrize("source", UV_SOURCES)
def test_pack_checker_quadrants_for_every_uv_source(tmp_path, source):
    pytest.importorskip("PIL")
    scene = load_pack(write_textured_pack(tmp_path / source, source), device="cpu")
    cam = Camera("c", width=40, height=40, spp=8)
    _assert_checker(cam.render(scene, output="linear").numpy()[0])


def test_uv_sources_render_the_same_image(tmp_path):
    pytest.importorskip("PIL")
    imgs = []
    for source in UV_SOURCES:
        scene = load_pack(write_textured_pack(tmp_path / source, source), device="cpu")
        imgs.append(Camera("c", width=24, height=24, spp=4).render(scene, output="linear").numpy().copy())
    assert all(np.array_equal(imgs[0], i) for i in imgs[1:])


def test_corner_colour_attribute_is_interpolated(tmp_path):
    col = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * 2, np.float32)  # (2, 3, 3), every triangle alike
    root = write_pack_v2(tmp_path / "p", attribute_colour_material("Col"), sky=1.0, half=0.5, attrs={"Col": col})
    img = Camera("c", width=24, height=24, spp=4).render(load_pack(root, device="cpu"), output="linear").numpy()[0]
    assert img[..., 0].std() > 0.02 and img[..., 1].std() > 0.02  # varies across the quad
    assert np.isfinite(img).all()


def test_codegen_uv_nodes_and_errors():
    mat = textured_material("uvmap")
    tex = {"checker": {"file": "c.png", "colorspace": "sRGB"}}
    src, _ = gen_material("m", mat, tex)
    assert "v2 = uv\n" in src and "nd.image_cubic_repeat(img_0, v2[0], v2[1])" in src
    mat["tree"]["nodes"][-1]["props"]["uv_map"] = "Other"
    with pytest.raises(NotImplementedError, match="Other"):
        gen_material("m", mat, tex)
    ok, _ = gen_material("m", mat, tex, uv_names=("Other",))
    assert "= uv" in ok


def test_mesh_scene_validation_says_what_to_change():
    with pytest.raises(CameraError, match="no uv"):
        mesh_scene([TexturedMesh("q", QUAD, TRIS, texture=checker_image())], device="cpu")
    with pytest.raises(CameraError, match="uv"):
        mesh_scene([TexturedMesh("q", QUAD, TRIS, uv=np.zeros((3, 2)))], device="cpu")
    with pytest.raises(CameraError, match="no objects"):
        mesh_scene([], device="cpu")


def test_every_mesh_scene_shares_one_module(tmp_path, monkeypatch):
    """Other objects, colours and textures are data: the generated module (tag and file) does not change."""
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path))
    a = _checker_scene()
    files = sorted(p.name for p in cache_dir().glob("*.py")) if cache_dir().exists() else []
    tex = np.full((4, 4, 3), 0.25, np.float32)
    objs = [TexturedMesh("a", QUAD, TRIS, uv=UV, texture=tex, color=(0.9, 0.2, 0.1), roughness=0.2),
            TexturedMesh("b", QUAD + 2.0, TRIS, color=(0.1, 0.8, 0.3), metallic=1.0),
            TexturedMesh("c", QUAD - 2.0, TRIS, uv=UV, texture=checker_image(2))]
    b = mesh_scene(objs, device="cpu")
    assert a.disp.__name__ == b.disp.__name__
    assert files == (sorted(p.name for p in cache_dir().glob("*.py")) if cache_dir().exists() else [])
    ca, cb = _quad_camera(16, spp=1), Camera("d", look_at=((1, 1, 6), (1, 1, 0), (0, 1, 0)), width=16, height=16, spp=1)
    ca.render(a)
    t0 = time.time()
    cb.render(b)
    log.warning("first render of the second mesh scene: %.2f s", time.time() - t0)


def test_generic_source_does_not_depend_on_the_materials():
    def static(n):
        mats = {f"m{i}": {"generic": {"base_color": [i, 0, 0], "roughness": 0.5, "texture": None}} for i in range(n)}
        return {"materials": mats, "textures": {}}
    src1, info = generate_source(static(1), ["m0"])
    src5, _ = generate_source(static(5), [f"m{i}" for i in range(5)])
    assert src1 == src5 and info["generic"] == [True]
    mixed, minfo = generate_source({"materials": {"g": static(1)["materials"]["m0"], "n": principled_material()},
                                    "textures": {}}, ["g", "n"])
    assert minfo["generic"] == [True, False] and "mat_generic(mid, uv, im)" in mixed
    compile(mixed, "generated", "exec")


@pytest.mark.gpu
def test_checker_quad_matches_between_backends():
    pytest.importorskip("warp")
    import warp as wp
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    means = {}
    for backend in ("warp", "optix"):
        sc = mesh_scene([TexturedMesh("quad", QUAD, TRIS, uv=UV, texture=checker_image())], device="cuda:0")
        try:
            img = _quad_camera(48, spp=16, backend=backend).render(sc, output="linear").numpy()[0]
        except CameraError as exc:
            pytest.skip(f"{backend} unavailable: {exc}")
        _assert_checker(img)
        means[backend] = img.reshape(-1, 3).mean(0)
    assert np.allclose(means["warp"], means["optix"], rtol=0.03), means
