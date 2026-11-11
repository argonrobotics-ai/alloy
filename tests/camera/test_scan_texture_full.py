# SPDX-License-Identifier: Apache-2.0
"""Scan textures in the full-material tracer (SCAN-TEXTURES part C), on the CPU device.

- The bilinear image sampler (Image Texture "Linear", REPEAT) against numpy at random UVs, inside and outside 0..1.
- A palette quad through a pack with a "Linear" Image Texture: each block's radiance under a uniform sky of radiance
  1 is its sRGB colour decoded to linear (the albedo), Linear and Cubic agree inside the blocks, and a Mix
  (colour, MULTIPLY) by a factor scales it, as a glTF baseColorFactor times its baseColorTexture.
- Generic materials (how scan parts are drawn): ``texture_tint`` multiplies the texture by the base colour; without
  it the base colour is ignored next to a texture, as before.
- ``look_over_scan``: which look replaces a scan's own texture.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy.camera.pbr.nodes as nd
from alloy.camera import Camera, Lighting, TexturedMesh, load_pack, mesh_scene
from alloy.camera.card_looks import look_over_scan
from alloy.camera.looks import rgba
from alloy.camera.pbr.codegen import gen_material
from tests.camera.packs import _socket, textured_material, write_pack_v2

DEV = "cpu"
PALETTE = np.array([[[200, 60, 30], [40, 180, 90]], [[30, 70, 210], [128, 128, 128]]], np.uint8)  # sRGB, row 0 = top
QUAD = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
TRIS = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)  # Blender convention: (0, 0) at the bottom-left
TOP = (0.5, 0.5, 2.0), (0.5, 0.5, 0.0), (0.0, 1.0, 0.0)
FACTOR = (0.5, 0.25, 1.0)  # linear, as a glTF baseColorFactor


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64)
    return np.where(c < 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c < 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def palette_image(block: int = 8) -> np.ndarray:
    return np.kron(PALETTE, np.ones((block, block, 1), np.uint8))


def block_means(img: np.ndarray, half: int = 2) -> dict:
    """Mean colour of a small window at the centre of each block (the quad covers the picture's middle ~70 %)."""
    n = img.shape[0]
    lo, hi = int(0.5 * n - 0.17 * n), int(0.5 * n + 0.17 * n)
    win = lambda r, c: img[r - half:r + half + 1, c - half:c + half + 1].reshape(-1, 3).mean(0)  # noqa: E731
    return {(0, 0): win(lo, lo), (0, 1): win(lo, hi), (1, 0): win(hi, lo), (1, 1): win(hi, hi)}


def expected_albedo(factor=(1.0, 1.0, 1.0)) -> dict:
    return {(r, c): srgb_to_linear(PALETTE[r, c] / 255.0) * np.asarray(factor) for r in (0, 1) for c in (0, 1)}


# ---------------------------------------------------------------------------- the sampler
@wp.kernel
def _sample_linear(img: wp.array3d(dtype=float), uv: wp.array(dtype=wp.vec2), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = nd.image_linear_repeat(img, uv[i][0], uv[i][1])


def numpy_bilinear(img: np.ndarray, u: float, v: float) -> np.ndarray:
    """Bilinear, REPEAT; ``img`` row 0 = v 0; texel (i, j) centred at ((i + 0.5) / w, (j + 0.5) / h)."""
    h, w = img.shape[:2]
    x, y = u * w - 0.5, v * h - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    tx, ty = x - x0, y - y0
    px = lambda i, j: img[j % h, i % w, :3]  # noqa: E731
    lo = px(x0, y0) * (1 - tx) + px(x0 + 1, y0) * tx
    hi = px(x0, y0 + 1) * (1 - tx) + px(x0 + 1, y0 + 1) * tx
    return lo * (1 - ty) + hi * ty


def test_linear_sampler_matches_numpy_and_hits_texel_centres():
    rng = np.random.default_rng(3)
    img = rng.random((5, 7, 3)).astype(np.float32)
    uv = rng.uniform(-1.5, 2.5, (200, 2)).astype(np.float32)
    centres = np.array([[(i + 0.5) / 7, (j + 0.5) / 5] for j in range(5) for i in range(7)], np.float32)
    uv = np.concatenate([uv, centres])
    out = wp.zeros(len(uv), dtype=wp.vec3, device=DEV)
    wp.launch(_sample_linear, dim=len(uv), inputs=[wp.array(img, dtype=float, device=DEV),
                                                  wp.array(uv, dtype=wp.vec2, device=DEV), out], device=DEV)
    got = out.numpy()
    want = np.array([numpy_bilinear(img, u, v) for u, v in uv])
    assert np.allclose(got, want, atol=2e-5), np.abs(got - want).max()
    assert np.allclose(got[-35:], img.reshape(-1, 3), atol=2e-6)  # exactly the texel at its centre


# ---------------------------------------------------------------------------- node-tree materials (packs)
def palette_pack(root, interpolation: str = "Linear", factor=None):
    """A unit quad under a uniform sky of radiance 1 with a rough, non-specular Principled material whose Base Color
    is a ``PALETTE`` Image Texture (UVs unlinked, as a scan's material), optionally times ``factor`` (Mix MULTIPLY)."""
    from PIL import Image
    (root / "textures").mkdir(parents=True, exist_ok=True)
    Image.fromarray(palette_image()).save(root / "textures/palette.png")
    mat = textured_material("unlinked", image="palette")
    tex = next(n for n in mat["tree"]["nodes"] if n["type"] == "ShaderNodeTexImage")
    tex["props"]["interpolation"] = interpolation
    if factor is not None:
        mat["tree"]["nodes"].append({
            "name": "Factor", "type": "ShaderNodeMix",
            "props": {"data_type": "RGBA", "factor_mode": "UNIFORM", "blend_type": "MULTIPLY", "clamp_factor": True,
                      "clamp_result": False},
            "inputs": [_socket("Factor_Float", "Factor", 1.0),
                       _socket("A_Color", "A", [1, 1, 1, 1], True, "NodeSocketColor"),
                       _socket("B_Color", "B", [*factor, 1.0], False, "NodeSocketColor")]})
        mat["tree"]["links"] = [{"from": "Tex", "from_id": "Color", "to": "Factor", "to_id": "A_Color"},
                                {"from": "Factor", "from_id": "Result_Color", "to": "Principled BSDF",
                                 "to_id": "Base Color"}]
    corner = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    write_pack_v2(root, mat, sky=1.0, half=0.5, uv=corner[np.array([[0, 1, 2], [0, 2, 3]])], eye=(0.0, 0.0, 2.0),
                  textures={"palette": {"file": "textures/palette.png", "colorspace": "sRGB"}})
    return root


def render_pack(root, size: int = 48, spp: int = 16) -> np.ndarray:
    return Camera("c", width=size, height=size, spp=spp).render(load_pack(root, device=DEV), output="linear").numpy()[0]


def assert_blocks(img: np.ndarray, want: dict, rtol: float = 0.03) -> None:
    got = block_means(img)
    for key, w in want.items():
        assert np.allclose(got[key], w, rtol=rtol, atol=4e-3), (key, got[key], w)


def test_linear_image_texture_draws_each_block_at_its_albedo(tmp_path):
    pytest.importorskip("PIL")
    assert_blocks(render_pack(palette_pack(tmp_path / "lin")), expected_albedo())


def test_linear_and_cubic_agree_inside_the_blocks(tmp_path):
    pytest.importorskip("PIL")
    lin = block_means(render_pack(palette_pack(tmp_path / "lin")))
    cub = block_means(render_pack(palette_pack(tmp_path / "cub", "Cubic")))
    for key in lin:
        assert np.allclose(lin[key], cub[key], rtol=0.01, atol=2e-3), (key, lin[key], cub[key])


def test_mix_multiply_scales_the_texture_by_the_factor(tmp_path):
    pytest.importorskip("PIL")
    assert_blocks(render_pack(palette_pack(tmp_path / "fac", factor=FACTOR)), expected_albedo(FACTOR))


def test_codegen_emits_the_samplers_and_mix_and_names_what_it_lacks():
    tex = {"palette": {"file": "p.png", "colorspace": "sRGB"}}
    mat = textured_material("unlinked", image="palette")
    node = next(n for n in mat["tree"]["nodes"] if n["type"] == "ShaderNodeTexImage")
    cubic, _ = gen_material("m", mat, tex)
    assert "v1 = nd.image_cubic_repeat(img_0, v2[0], v2[1])" in cubic and "linear_repeat" not in cubic  # unchanged
    node["props"]["interpolation"] = "Linear"
    linear, meta = gen_material("m", mat, tex)
    assert "v1 = nd.image_linear_repeat(img_0, v2[0], v2[1])" in linear and "v1 = nd.srgb_to_linear3(v1)" in linear
    assert meta["images"] == [{"key": "palette", "file": "p.png", "srgb": True}]
    node["props"]["interpolation"] = "Closest"
    with pytest.raises(NotImplementedError, match="Closest"):
        gen_material("m", mat, tex)


def test_codegen_mix_colour_blends(tmp_path):
    pytest.importorskip("PIL")
    import json
    mat = json.loads((palette_pack(tmp_path / "m", factor=FACTOR) / "pack.json").read_text())["materials"]["mat0"]
    tex = {"palette": {"file": "p.png", "colorspace": "sRGB"}}
    src, _ = gen_material("m", mat, tex)
    assert "nd.mix_color(1.0, " in src and ", 1, 1, 0)" in src  # MULTIPLY, clamp factor, no clamp result
    mix = next(n for n in mat["tree"]["nodes"] if n["type"] == "ShaderNodeMix")
    mix["props"]["blend_type"] = "SCREEN"
    with pytest.raises(NotImplementedError, match="SCREEN"):
        gen_material("m", mat, tex)


# ---------------------------------------------------------------------------- generic materials (scan parts)
def generic_quad(texture, tint: bool = False, color=(0.8, 0.8, 0.8), device: str = DEV):
    flat = Lighting([], (1.0, 1.0, 1.0), 1.0)
    obj = TexturedMesh("scan", QUAD, TRIS, uv=UV, texture=texture, color=color, roughness=1.0, texture_tint=tint)
    img = Camera("c", look_at=TOP, width=48, height=48, spp=16).render(mesh_scene([obj], lighting=flat, device=device),
                                                                     output="linear")
    return img.numpy()[0]


def assert_tint_is_premultiplied(device: str = DEV) -> None:
    """Tinted by colour c == the untinted texture pre-multiplied by c (in linear), block by block. (The generic
    material reflects a little of the sky, colour-neutral, so the tint is checked against a render, not c x albedo.)"""
    tex = palette_image().astype(np.float32) / 255.0
    tint_srgb = linear_to_srgb(FACTOR)
    premult = linear_to_srgb(srgb_to_linear(tex) * np.asarray(FACTOR)).astype(np.float32)
    got = block_means(generic_quad(tex, tint=True, color=tuple(tint_srgb), device=device))
    want = block_means(generic_quad(premult, device=device))
    assert np.allclose(np.asarray(rgba(tint_srgb)[:3]), FACTOR, atol=1e-5)
    for key in got:
        assert np.allclose(got[key], want[key], rtol=0.01, atol=1e-3), (key, got[key], want[key])


def test_generic_texture_tint_multiplies_the_base_colour():
    assert_tint_is_premultiplied()


def test_generic_without_tint_ignores_the_base_colour_next_to_a_texture():
    tex = palette_image()
    a, b = generic_quad(tex, color=(0.9, 0.1, 0.1)), generic_quad(tex, color=(0.1, 0.1, 0.9))
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------- which look wins over a scan
def test_look_over_scan_order():
    labels = ["ball/visual", "ball"]
    look = {"materials": {"steel": {"preset": "steel_polished"}}, "assign": [{"match": "^ball", "material": "steel"}]}
    items = {"materials": {"ball:skin": {"preset": "rubber_black"}},
             "assign": [{"match": "^ball", "material": "ball:skin"}]}
    assert look_over_scan(labels, "rubber_ball_40mm", look, items) == ("steel", {"preset": "steel_polished"})
    assert look_over_scan(labels, "rubber_ball_40mm", None, items) == ("ball:skin", {"preset": "rubber_black"})
    name, spec = look_over_scan(labels, "rubber_ball_40mm")  # the card's own object look
    assert name.startswith("rubber_ball_40mm.") and "preset" in spec
    assert look_over_scan(labels, "block_alder_75x25x15") is None  # a material-default look: the texture wins
    assert look_over_scan(labels, None) is None
    other = {"materials": {}, "assign": [{"match": "^cup", "material": "glass_clear"}]}
    assert look_over_scan(labels, "block_alder_75x25x15", other) is None


@pytest.mark.gpu
def test_linear_texture_and_tint_on_cuda(tmp_path):
    """The same block colours on a GPU: the Linear pack and a tinted generic quad (both compile for CUDA)."""
    pytest.importorskip("PIL")
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    pack = Camera("c", width=48, height=48, spp=16).render(load_pack(palette_pack(tmp_path / "g"), device="cuda:0"),
                                                          output="linear").numpy()[0]
    assert_blocks(pack, expected_albedo())
    assert_tint_is_premultiplied("cuda:0")
    cpu, gpu = block_means(generic_quad(palette_image())), block_means(generic_quad(palette_image(), device="cuda:0"))
    for key in cpu:  # the two devices draw the same texture
        assert np.allclose(cpu[key], gpu[key], rtol=0.02, atol=2e-3), (key, cpu[key], gpu[key])
