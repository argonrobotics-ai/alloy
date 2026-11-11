# SPDX-License-Identifier: Apache-2.0
"""Image textures in the fast tracer (alloy.camera.tracer.textures): the albedo image of a textured quad against
its own texture, texel by texel and between texels; UV orientation; wrap; inst_color and inst_textured;
untextured instances."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.camera import Camera, CameraError, camera_scene
from alloy.camera.scene import IDENTITY_XF, MeshGeom, SceneData
from alloy.camera.tracer.textures import pack_textures, sample_host, srgb_to_linear
from tests.camera.helpers import cube, plane_scene

DEV = "cpu"
N = 8  # texels per side of the test texture; the quad is 1 m wide


def _texture(n: int = N, seed: int = 0) -> np.ndarray:
    """An sRGB8 image with no symmetry: random texels, a red top-left and a blue bottom-right corner."""
    img = np.random.default_rng(seed).integers(20, 236, (n, n, 3), dtype=np.uint8)
    img[0, 0], img[-1, -1] = (255, 0, 0), (0, 0, 255)
    return img


def _lin(tex: np.ndarray) -> np.ndarray:
    return srgb_to_linear(tex / 255.0)


def _quad_scene(tex: np.ndarray, uv_scale: float = 1.0, color=(1.0, 1.0, 1.0)) -> SceneData:
    """A 1 m square at z = 0 facing +z, image upright when seen from above with world +y up in the picture."""
    pts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], np.float32)
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32) * uv_scale  # v down: y = +0.5 is the top row
    quad = MeshGeom("quad", pts, np.array([[0, 1, 2], [0, 2, 3]], np.int32), uv=uv, texture=0)
    return SceneData.single_world([quad], [color], textures=[tex])


def _albedo(sd: SceneData, px: int, half: float = 0.5, backend: str = "warp") -> np.ndarray:
    """Albedo image of a top-down camera whose picture spans exactly ``2 * half`` metres at z = 0."""
    fov = float(np.degrees(2 * np.arctan(half)))
    cam = Camera("top", look_at=((0, 0, 1.0), (0, 0, 0), (0, 1, 0)), width=px, height=px, fov=fov, spp=1,
                 denoise="off", backend=backend)
    return cam.render(camera_scene(sd, device=DEV), output="all").albedo.numpy()[0]


@pytest.mark.law
def test_one_pixel_per_texel_reads_the_texture():
    """With one pixel centred on each texel the albedo image is the texture, made linear, within 0.002
    (the hit point is float32, so a pixel sits a hair off its texel centre)."""
    tex = _texture()
    alb = _albedo(_quad_scene(tex), N)
    want = _lin(tex)
    assert np.abs(alb - want).max() < 2e-3
    assert alb[0, 0, 0] > 0.99 and alb[-1, -1, 2] > 0.99  # red stays top-left, blue bottom-right: no flip


@pytest.mark.law
def test_between_texels_is_the_bilinear_blend():
    """Four pixels per texel: every pixel equals the host's bilinear lookup at its own UV, within 0.001."""
    tex, px = _texture(seed=1), 4 * N
    sd = _quad_scene(tex)
    alb = _albedo(sd, px)
    packed = pack_textures(sd, np.arange(1))
    want = np.array([[sample_host(packed, 0, (c + 0.5) / px, (r + 0.5) / px) for c in range(px)] for r in range(px)])
    assert np.abs(alb - want).max() < 1e-3
    assert np.abs(alb - _lin(np.repeat(np.repeat(tex, 4, 0), 4, 1))).max() > 0.05  # not nearest


@pytest.mark.law
def test_uvs_outside_zero_one_repeat():
    """UVs scaled by 2 tile the image 2 x 2 over the quad."""
    tex = _texture(seed=2)
    alb = _albedo(_quad_scene(tex, uv_scale=2.0), 2 * N)
    assert np.abs(alb - _lin(np.tile(tex, (2, 2, 1)))).max() < 2e-3


def test_inst_color_multiplies_and_inst_textured_false_draws_flat():
    """Albedo = inst_color * texture; an instance with inst_textured False reads inst_color alone; the lit image
    of the textured quad differs from the flat one."""
    tex, tint = _texture(seed=3), (0.5, 1.0, 0.25)
    sd = _quad_scene(tex, color=tint)
    assert np.abs(_albedo(sd, N) - _lin(tex) * tint).max() < 2e-3
    flat = _quad_scene(tex, color=tint)
    flat.inst_textured = np.array([False])
    assert np.allclose(_albedo(flat, N), tint, atol=1e-6)
    cam = Camera("top", look_at=((0, 0, 1.0), (0, 0, 0), (0, 1, 0)), width=N, height=N, fov=40, spp=2, denoise="off")
    a = cam.render(camera_scene(sd, device=DEV), output="linear", seed=1).numpy().copy()
    b = cam.render(camera_scene(flat, device=DEV), output="linear", seed=1).numpy()
    assert np.abs(a - b).max() > 0.05


def test_untextured_instances_keep_their_flat_colour_beside_a_textured_one():
    """A cube without a texture above the textured quad reads its own colour, and a scene without textures packs
    nothing."""
    sd = _quad_scene(_texture())
    both = SceneData(sd.meshes + [cube(0.1)], [0, 1], [0, 0], [-1, -1], [IDENTITY_XF, [0, 0, 0.3, 0, 0, 0, 1]],
                     [(1.0, 1.0, 1.0), (0.7, 0.1, 0.4)], 1, textures=sd.textures)
    alb = _albedo(both, 16)
    assert np.allclose(alb[8, 8], (0.7, 0.1, 0.4), atol=1e-6)
    corner = sample_host(pack_textures(both, np.arange(2)), 0, 0.5 / 16, 0.5 / 16)
    assert np.abs(alb[0, 0] - corner).max() < 1e-3
    assert not pack_textures(plane_scene(), np.arange(1)).any


def test_bad_texture_inputs_raise():
    sd = _quad_scene(_texture())
    sd.meshes[0].uv = np.zeros((3, 2), np.float32)
    with pytest.raises(CameraError, match="one UV per vertex"):
        _albedo(sd, 4)
    sd = _quad_scene(_texture())
    sd.meshes[0].texture = 3
    with pytest.raises(CameraError, match="index into textures"):
        _albedo(sd, 4)
    with pytest.raises(CameraError, match="uint8"):
        _albedo(_quad_scene(_texture().astype(np.float32)), 4)
