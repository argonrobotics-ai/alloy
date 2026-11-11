# SPDX-License-Identifier: Apache-2.0
"""The full-material tracer on CUDA with hand-made packs: furnace test, compiled node graphs, area lights."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.camera import Camera, load_pack
from tests.camera.packs import principled_material, write_pack_v2

pytestmark = pytest.mark.gpu


def _render(path, spp=64, output="linear"):
    scene = load_pack(path, device="cuda:0")
    return Camera("c", width=8, height=8, spp=spp).render(scene, output=output)


def test_white_furnace_plane(tmp_path):
    """A Lambert plane (specular 0) under a uniform sky of radiance L reflects albedo x L."""
    out = _render(write_pack_v2(tmp_path / "f", principled_material((0.5, 0.5, 0.5)), sky=0.5), output="all")
    lin = out.linear.numpy().reshape(-1, 3)
    assert np.allclose(lin.mean(0), 0.25, rtol=0.01), lin.mean(0)
    assert np.allclose(out.albedo.numpy().reshape(-1, 3), 0.5, atol=1e-4)
    assert np.allclose(out.normal.numpy().reshape(-1, 3), [0, 0, -1], atol=1e-4)


def test_noise_roughness_and_area_light(tmp_path):
    mat = principled_material((0.5, 0.5, 0.5), specular=0.5, noise_roughness=True)
    dark = _render(write_pack_v2(tmp_path / "a", mat, sky=0.2)).numpy().copy()
    lit = _render(write_pack_v2(tmp_path / "b", mat, sky=0.2, light_energy=50.0)).numpy()
    assert np.isfinite(dark).all() and dark.mean() > 0.05
    assert lit.mean() > dark.mean() * 1.2


def test_model_with_look_renders_full_materials():
    import warp as wp
    from alloy._vendor import newton
    from alloy.camera import camera_scene
    b = newton.ModelBuilder()
    b.add_ground_plane()
    box = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="box")
    b.add_shape_box(box, hx=0.08, hy=0.06, hz=0.04)
    s = newton.ModelBuilder()
    s.replicate(b, 2)
    model = s.finalize(device="cuda:0")
    look = {"assign": [{"match": "box", "material": "aluminum_bare"}]}
    cam = Camera("c", look_at=((0.5, -0.5, 0.35), (0.0, 0.0, 0.1)), width=16, height=16, spp=4)
    img = cam.render(camera_scene(model, look=look), model.state()).numpy()
    assert img.shape == (2, 16, 16, 3) and img.std() > 5
