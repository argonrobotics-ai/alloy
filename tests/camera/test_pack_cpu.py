# SPDX-License-Identifier: Apache-2.0
"""Scene packs on the host: readers, material code generation, camera and settings (no GPU kernels)."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.camera import CameraError, load_pack
from alloy.camera.pack_v1 import scene_from_pack_v1
from alloy.camera.pbr.dispatch import generate_source
from alloy.camera.pbr.pack_io import load_pack as read_pack
from alloy.camera.pbr.renderer import settings_from_pack
from alloy.camera.pbr.shade_data import camera_params
from tests.camera.packs import principled_material, write_pack_v1, write_pack_v2


def test_v2_pack_reads_back(tmp_path):
    pk = read_pack(write_pack_v2(tmp_path / "p", principled_material()))
    fr = pk.frame()
    assert pk.version == 2 and fr.frame == 0 and len(fr.meshes) == 1
    assert fr.meshes[0].tri.shape == (2, 3) and fr.meshes[0].n.shape == (2, 3, 3)
    st = settings_from_pack(pk.static, seed=5)
    assert st.max_bounces == 12 and st.clamp_indirect == 10.0 and st.seed == 5


def test_camera_params_follow_blender_sensor_fit(tmp_path):
    fr = read_pack(write_pack_v2(tmp_path / "p", principled_material())).frame()
    cp = camera_params(fr, 640, 480)
    assert cp["extent"][0] == pytest.approx(0.5 * 36.0 / 50.0)  # AUTO fit: sensor width spans the longer side
    assert cp["extent"][1] == pytest.approx(0.5 * 36.0 / 50.0 * 480 / 640)
    assert cp["aperture"] == 0.0 and np.allclose(cp["eye"], [0, 0, 1])


def test_material_codegen_for_linked_and_constant_inputs():
    static = {"materials": {"flat": principled_material(), "noisy": principled_material(noise_roughness=True)},
              "textures": {}}
    src, info = generate_source(static, ["flat", "noisy"])
    assert "def mat_noisy__Roughness(" in src and "nd.tex_noise_3d(" in src
    assert "q.roughness = mat_noisy__Roughness(p, n, uv, col, rnd)" in src
    assert "q.specular_ior_level = 0.0" in src and "MAT_PARAMS = [mat_params_0, mat_params_1, mat_params_2]" in src
    assert info["bump"] == [None, None] and info["images"] == []
    compile(src, "generated", "exec")


def test_unsupported_material_says_what_to_change():
    bad = principled_material()
    bad["principled"]["inputs"]["Emission Strength"]["value"] = 1.0
    with pytest.raises(CameraError, match="set Emission Strength to 0"):
        generate_source({"materials": {"lamp": bad}, "textures": {}}, ["lamp"])


def test_v1_pack_scene_and_camera(tmp_path):
    sd = scene_from_pack_v1(write_pack_v1(tmp_path / "v1"), num_worlds=2, static_names=["floor"])
    assert sd.num_instances == 3 and sorted(sd.inst_group.tolist()) == [-1, 0, 1]
    assert sd.camera["fov_deg"] == pytest.approx(np.degrees(0.9)) and len(sd.lighting.suns) == 1
    assert np.allclose(sd.inst_color[sd.inst_group == 0], [0.8, 0.1, 0.1])


def test_load_pack_rejects_other_folders(tmp_path):
    with pytest.raises(CameraError, match="pack.json"):
        load_pack(tmp_path, device="cpu")


def test_v1_pack_renders_on_cpu(tmp_path):
    from alloy.camera import Camera
    scene = load_pack(write_pack_v1(tmp_path / "v1"), device="cpu")
    img = Camera("v1", width=16, height=12, spp=2).render(scene).numpy()
    centre = img[0, 4:6, 6:10].reshape(-1, 3).astype(int)
    assert img.shape == (1, 12, 16, 3) and (centre[:, 0] > centre[:, 1] + 20).all()  # the red wall, in view
