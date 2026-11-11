# SPDX-License-Identifier: Apache-2.0
"""Host-side camera maths, scene checks, errors, the view transform and PNG output (no kernels)."""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from alloy.camera import CameraError, image_extent, look_at
from alloy.camera.pose import as_poses, look_at_matrix
from alloy.camera.scene import MeshGeom, SceneData, Sun, quat_to_matrix
from alloy.camera.tracer.tables import filter_table, hash_uint
from alloy.camera.view import standard_u8, write_png
from tests.camera.helpers import plane_scene


def test_look_at_points_minus_z_at_target():
    eye, target = np.array([2.0, -2.0, 1.5]), np.array([0.0, 0.0, 0.3])
    xf = look_at(eye, target)
    r = quat_to_matrix(xf[3:])
    fwd = -r[:, 2]
    assert np.allclose(fwd, (target - eye) / np.linalg.norm(target - eye), atol=1e-6)
    assert r[2, 1] > 0  # camera up has a +z component
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-6)


def test_look_at_straight_down_is_finite():
    r = look_at_matrix((0, 0, 3), (0, 0, 0))
    assert np.isfinite(r).all() and np.allclose(-r[:, 2], [0, 0, -1])


def test_image_extent_fits():
    hx, hy = image_extent(60.0, 640, 480)
    assert hy == pytest.approx(np.tan(np.radians(30))) and hx == pytest.approx(hy * 640 / 480)
    hx, hy = image_extent(60.0, 640, 480, "AUTO")
    assert hx == pytest.approx(np.tan(np.radians(30)))


@pytest.mark.parametrize("bad", [0.0, 180.0, -5.0])
def test_bad_fov_says_what_to_change(bad):
    with pytest.raises(CameraError, match="between 0 and 180"):
        image_extent(bad, 64, 64)


def test_as_poses_broadcasts_and_normalises():
    p = as_poses([0, 0, 1, 0, 0, 0, 2.0], 3)
    assert p.shape == (3, 7) and np.allclose(p[:, 6], 1.0)
    with pytest.raises(CameraError, match="unit quaternion"):
        as_poses([0, 0, 1, 0, 0, 0, 0], 2)
    with pytest.raises(CameraError, match=r"\(2, 7\)"):
        as_poses(np.zeros((3, 7)), 2)


def test_scene_validation_messages():
    with pytest.raises(CameraError, match="triangle index"):
        MeshGeom("m", np.zeros((3, 3)), np.array([[0, 1, 3]]))
    sd = plane_scene()
    assert sd.num_instances == 1 and sd.inst_body[0] == -1
    with pytest.raises(CameraError, match="not a world"):
        SceneData(sd.meshes, [0], [2], [-1], sd.inst_local, sd.inst_color, num_worlds=1)
    err = CameraError("x", "do y")
    assert err.to_json() == {"error": "x", "fix": "do y"}


def test_sun_from_euler_points_up_for_identity():
    s = Sun.from_euler((0.0, 0.0, 0.0), 2.0)
    assert np.allclose(s.direction_to_sun, [0, 0, 1]) and s.strength == (2.0, 2.0, 2.0)


def test_filter_table_symmetric_and_bounded():
    t = filter_table("BLACKMAN_HARRIS", 1.5)
    assert t.shape == (1024,) and t.dtype == np.float32
    half = 511
    assert np.allclose(t[half - 50] + t[half + 50], 1.0, atol=1e-6)  # symmetric about 0.5
    assert np.all(np.diff(t[:-1]) >= 0) and t[0] == -1.0 and t[-2] == 2.0  # 3 px wide (BH width doubled)


def test_hash_uint_is_32_bit_and_mixes():
    vals = {hash_uint(k) for k in range(64)}
    assert len(vals) == 64 and all(0 <= v < 2 ** 32 for v in vals)


def test_standard_view_matches_srgb():
    img = standard_u8(np.array([[[0.0, 0.18, 1.0], [2.0, -1.0, 0.0031308]]], np.float32))
    assert img.tolist() == [[[0, 118, 255], [255, 0, 10]]]


def test_png_roundtrip(tmp_path):
    img = (np.arange(4 * 5 * 3) % 251).astype(np.uint8).reshape(4, 5, 3)
    p = write_png(tmp_path / "a.png", img)
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    idat = data.index(b"IDAT")
    n = int.from_bytes(data[idat - 4:idat], "big")
    raw = zlib.decompress(data[idat + 4:idat + 4 + n])
    rows = np.frombuffer(raw, np.uint8).reshape(4, 1 + 5 * 3)
    assert (rows[:, 0] == 0).all() and np.array_equal(rows[:, 1:].reshape(4, 5, 3), img)
