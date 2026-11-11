# SPDX-License-Identifier: Apache-2.0
"""Glass, water and clear plastic (Principled transmission, Volume Absorption) and random-walk subsurface in the
full-material tracer, on the CPU: closed-form checks under a uniform sky, and the node-graph parsers."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.camera import Camera, CameraError, load_pack
from alloy.camera.pbr import nodes as nd
from alloy.camera.pbr.path import medium_empty, medium_enter_exit
from tests.camera.glass_packs import glass_material, slab_brightness, sss_material, write_glass_pack

SKY = 0.5


def _centre(root, material, half=(1.0, 1.0, 0.05), eye_z=0.6, spp=128, **pack) -> np.ndarray:
    scene = load_pack(write_glass_pack(root, material, half=half, sky=SKY, eye_z=eye_z, **pack), device="cpu")
    img = Camera("c", width=16, height=16, spp=spp).render(scene, output="linear").numpy()[0]
    return img[6:10, 6:10].mean(axis=(0, 1))


def test_clear_glass_is_invisible_in_a_uniform_sky(tmp_path):
    # every path through lossless smooth glass ends in the sky: the box vanishes (a white furnace test)
    assert np.allclose(_centre(tmp_path / "g", glass_material(0.0)), SKY, atol=2e-3)


def test_absorbing_slab_matches_beer_lambert(tmp_path):
    # sigma_a = (1 - 0.5) * 10 = 5 / m through 0.1 m of IOR 1.5 glass, every internal reflection included
    c = _centre(tmp_path / "a", glass_material(0.0, absorb_color=(0.5, 0.5, 0.5), density=10.0))
    assert np.allclose(c, SKY * slab_brightness(1.5, 5.0, 0.1), rtol=0.01)


def test_rough_glass_keeps_its_energy(tmp_path):
    # multiscatter GGX glass loses about 1 % (Cycles' energy tables are an approximation too)
    c = _centre(tmp_path / "r", glass_material(0.3))
    assert np.allclose(c, SKY, rtol=0.025)


def test_frosted_glass_over_white_matches_cycles_without_reflective_caustics(tmp_path):
    # 92 % rough glass over white diffuse, reflective caustics off. After a diffuse bounce inside the slab the glass
    # has no reflection, so at total internal reflection its sample weight is 0 while its rough transmission is not:
    # only light sampling reaches it (MIS weight 1). Reference: this pack in Blender 4.5.3 Cycles, 16x16 at 4096 spp.
    furnace = {k: 128 for k in ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces")}
    furnace.update(blur_glossy=0.0, caustics_reflective=False)
    c = _centre(tmp_path / "f", glass_material(0.8, transmission=0.92), spp=256, cycles=furnace)
    assert c.mean() / SKY == pytest.approx(1.0373, rel=0.01)


def test_a_world_that_is_not_a_light_is_counted_once(tmp_path):
    # sampling method NONE: the world is no light (no light sampling), escaped rays see it at full weight
    white = glass_material(1.0, transmission=0.0)
    auto = _centre(tmp_path / "a", white)
    assert np.allclose(_centre(tmp_path / "n", white, world_sampling="NONE"), auto, rtol=0.02)


def test_subsurface_is_close_to_diffuse_and_tiny_radii_fall_back_to_it(tmp_path):
    cube = {"half": (0.3, 0.3, 0.3), "eye_z": 2.0}
    diffuse = _centre(tmp_path / "d", sss_material(0.0, 0.01), **cube)
    walk = _centre(tmp_path / "w", sss_material(1.0, 0.01), **cube)
    tiny = _centre(tmp_path / "t", sss_material(1.0, 1e-12), **cube)
    assert np.allclose(walk, diffuse, rtol=0.06) and not np.allclose(walk, diffuse, rtol=1e-4)
    assert np.allclose(tiny, diffuse, atol=1e-5)  # radius < BSSRDF_MIN_RADIUS: the same Lambert lobe


def test_volume_absorption_parse_and_errors():
    from alloy.camera.pbr.volume import absorption
    mat = glass_material(0.0, absorb_color=(0.0, 0.5, 1.0), density=2.0)
    assert absorption(mat)["sigma_a"] == pytest.approx([2.0, 1.0, 0.0])
    assert absorption(glass_material(0.0)) is None
    mat["tree"]["nodes"][-1]["type"] = "ShaderNodeVolumeScatter"
    with pytest.raises(CameraError, match="Volume Absorption"):
        absorption(mat)


def _world_tree(rot_z: float = 0.0, camera_mix: bool = False) -> dict:
    nodes = [{"name": "Env", "type": "ShaderNodeTexEnvironment", "inputs": []},
             {"name": "Map", "type": "ShaderNodeMapping", "props": {"vector_type": "POINT"},
              "inputs": [{"name": "Location", "value": [0, 0, 0]}, {"name": "Rotation", "value": [0, 0, rot_z]},
                         {"name": "Scale", "value": [1, 1, 1]}]},
             {"name": "Bg", "type": "ShaderNodeBackground", "inputs": [{"name": "Strength", "value": 0.8}]},
             {"name": "Out", "type": "ShaderNodeOutputWorld", "inputs": []}]
    links = [{"from": "Map", "from_id": "Vector", "to": "Env", "to_id": "Vector"}]
    if camera_mix:
        nodes += [{"name": "Dark", "type": "ShaderNodeBackground",
                   "inputs": [{"name": "Color", "value": [0.1, 0.2, 0.3, 1]}, {"name": "Strength", "value": 2.0}]},
                  {"name": "LP", "type": "ShaderNodeLightPath", "inputs": []},
                  {"name": "Mix", "type": "ShaderNodeMixShader", "inputs": []}]
        links += [{"from": "LP", "from_id": "Is Camera Ray", "to": "Mix", "to_id": "Fac"},
                  {"from": "Bg", "from_id": "Background", "to": "Mix", "to_id": "Shader"},
                  {"from": "Dark", "from_id": "Background", "to": "Mix", "to_id": "Shader_001"},
                  {"from": "Mix", "from_id": "Shader", "to": "Out", "to_id": "Surface"}]
    return {"nodes": nodes, "links": links}


def test_world_mapping_rotation_and_camera_background():
    from alloy.camera.pbr.world_nodes import camera_background, lighting_background, mapping_rotation
    rot = mapping_rotation(_world_tree(np.pi / 2))
    assert np.allclose(rot @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)  # Blender: +x turns to +y about z
    tree = _world_tree(0.0, camera_mix=True)
    assert np.allclose(camera_background(tree), [0.2, 0.4, 0.6]) and lighting_background(tree)["name"] == "Bg"
    assert camera_background(_world_tree()) is None
    tree["links"][1]["from_id"] = "Is Shadow Ray"
    with pytest.raises(CameraError, match="Is Camera Ray"):
        camera_background(tree)


@wp.kernel
def _map_range_kernel(vals: wp.array(dtype=float), kinds: wp.array(dtype=wp.int32), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = nd.map_range(vals[i], 0.66, 0.72, 0.0, 1.0, 4.0, kinds[i], 1)


def test_map_range_node_matches_cycles():
    vals = np.array([0.6, 0.69, 0.8, 0.69, 0.69, 0.69], np.float32)
    kinds = np.array([0, 0, 0, 1, 2, 3], np.int32)
    out = wp.zeros(len(vals), dtype=float, device="cpu")
    wp.launch(_map_range_kernel, dim=len(vals), inputs=[wp.array(vals, device="cpu"), wp.array(kinds, device="cpu")],
              outputs=[out], device="cpu")
    f = (0.69 - 0.66) / 0.06
    expect = [0.0, f, 1.0, np.floor(f * 5.0) / 4.0, (3 - 2 * f) * f * f, f ** 3 * (f * (f * 6 - 15) + 10)]
    assert np.allclose(out.numpy(), expect, atol=1e-5)


@wp.kernel
def _medium_kernel(out: wp.array(dtype=wp.vec4i)):
    md = medium_empty()
    md = medium_enter_exit(md, 3, 1, False)   # enter glass (instance 3, material 1)
    md = medium_enter_exit(md, 7, 2, False)   # enter the water inside it
    md = medium_enter_exit(md, 7, 2, False)   # entering again changes nothing
    out[0] = md.inst
    md = medium_enter_exit(md, 3, 1, True)    # leave the glass: the water stays
    out[1] = md.inst
    md = medium_enter_exit(md, 9, 1, True)    # leaving an object the ray is not in changes nothing
    out[2] = md.inst


def test_volume_stack_enters_and_leaves_like_cycles():
    out = wp.zeros(3, dtype=wp.vec4i, device="cpu")
    wp.launch(_medium_kernel, dim=1, outputs=[out], device="cpu")
    assert out.numpy().tolist() == [[3, 7, -1, -1], [7, -1, -1, -1], [7, -1, -1, -1]]


def test_unsupported_subsurface_method_says_what_to_change(tmp_path):
    mat = sss_material(0.5, 0.01)
    mat["principled"]["subsurface_method"] = "BURLEY"
    with pytest.raises(CameraError, match="Random Walk"):
        Camera("c", width=8, height=8, spp=1).render(load_pack(write_glass_pack(tmp_path / "b", mat), device="cpu"))


@pytest.mark.parametrize("base", [0.0, 0.002])
def test_glass_touching_a_table_z_fights_like_cycles(base):
    # a glass standing exactly on a table: a quarter of the rays meet the glass first, the rest the table (Cycles'
    # z-fighting); a glass 0.2 um above it (a tilted sim pose) is seen floating, as in Cycles, and 0.2 um below it sunk
    from alloy.camera.tracer.geometry import TIE_P_TRANSMISSIVE
    from tests.camera.glass_packs import tie_hits
    for gap, want in ((1e-3, 1), (2e-7, 1), (-2e-7, 0)):
        assert (tie_hits("wavefront-warp", gap, base=base) == want).all(), gap
    assert tie_hits("wavefront-warp", 0.0, base=base, n_side=20).mean() == pytest.approx(TIE_P_TRANSMISSIVE, abs=0.07)


def test_random_walk_returns_what_an_analog_walk_does():
    # a 0.1 mm mean free path (C2's worktop, blue): starting the walk at the pushed first-cast origin (20 um) lost 4 %
    from tests.camera.sss_walks import analog_albedo, guided_albedo
    for base in (0.8, 0.5):
        assert guided_albedo(base, 0.0012) == pytest.approx(analog_albedo(base), rel=0.015)
