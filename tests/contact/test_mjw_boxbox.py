# SPDX-License-Identifier: Apache-2.0
"""BUL-0011: T12's block on the plywood sheet at the float32 frames of the tripping substep (WP7's repro,
[internal note]). MuJoCo Warp's own EPA flips the contacts on the default convex
path; the primitive collider that ``use_primitive_boxbox`` selects, and the vendored PR #1845 EPA, return MuJoCo C's."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
mjw = pytest.importorskip("alloy._vendor.mujoco_warp")
wp = pytest.importorskip("warp")

from alloy.contact.mjw_boxbox import box_box_pairs, set_primitive_boxbox, use_primitive_boxbox  # noqa: E402

SLAB_POS = np.array([0.3499999940395355, 0.0, -0.00634999992325902], np.float32)
BLOCK_POS = np.array([0.3924983739852905, -0.09146194159984589, 0.007498880848288536], np.float32)
BLOCK_MAT = np.array([-0.20900958776474, 0.9779136776924133, 0.00012808256724383682,
                      -0.9779136776924133, -0.20900952816009521, -0.00021551383542828262,
                      -0.00018398342945147306, -0.00017029812443070114, 1.0], np.float32)
XML = """<mujoco><worldbody>
  <geom name="slab" type="box" size="0.3 0.6 0.00635" pos="0.35 0 -0.00635"/>
  <body name="block" pos="0.4 0 0.0075"><freejoint/><geom type="box" size="0.0375 0.0041666668 0.0075"/></body>
  <body name="ball" pos="1 1 1"><freejoint/><geom type="sphere" size="0.01"/></body>
</worldbody></mujoco>"""


def _model() -> tuple:
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    d.geom_xpos[0], d.geom_xmat[0] = SLAB_POS, np.eye(3).ravel()
    d.geom_xpos[1], d.geom_xmat[1] = BLOCK_POS, BLOCK_MAT
    return m, d


def _contacts(primitive: bool) -> tuple[np.ndarray, np.ndarray]:
    """(dist, normal) of MuJoCo Warp's slab-block contacts at the trip frames, on CPU."""
    m, d = _model()
    with wp.ScopedDevice("cpu"):
        wm, wd = mjw.put_model(m), mjw.put_data(m, d, nworld=1, nconmax=8, njmax=32)
        if primitive:
            assert set_primitive_boxbox(wm)
        mjw.kinematics(wm, wd)
        gx, gm = wd.geom_xpos.numpy(), wd.geom_xmat.numpy()
        gx[0, 0], gm[0, 0], gx[0, 1], gm[0, 1] = SLAB_POS, np.eye(3, dtype=np.float32), BLOCK_POS, BLOCK_MAT.reshape(3, 3)
        wd.geom_xpos.assign(gx)
        wd.geom_xmat.assign(gm)
        mjw.collision(wm, wd)
        n = int(wd.nacon.numpy()[0])
        return wd.contact.dist.numpy()[:n], wd.contact.frame.numpy()[:n].reshape(n, 9)[:, :3]


def test_primitive_boxbox_matches_mujoco_c_on_the_bul0011_frames() -> None:
    m, d = _model()
    mujoco.mj_collision(m, d)
    ref = np.array([d.contact[i].dist for i in range(d.ncon)])
    dist, normal = _contacts(primitive=True)
    assert len(dist) == len(ref) == 2
    np.testing.assert_allclose(np.sort(dist), np.sort(ref), atol=2e-7)
    np.testing.assert_allclose(normal, [[0.0, 0.0, 1.0]] * 2, atol=1e-6)


def test_default_convex_path_on_the_bul0011_frames() -> None:
    """The defect the switch avoids is MuJoCo Warp's own EPA: at the pin (86e2fae0, ``ALLOY_MJW_EPA=stock``) the convex
    path flips the contacts (-27.7 mm, -z). If a new pin fixes it, the stock half says so. The vendored PR #1845 EPA
    (``alloy.contact.mjw_epa``, default) returns MuJoCo C's contacts there, as the primitive collider does."""
    m, d = _model()
    mujoco.mj_collision(m, d)
    ref = np.array([d.contact[i].dist for i in range(d.ncon)])
    dist, normal = _contacts(primitive=False)
    assert len(dist) == len(ref) == 2
    np.testing.assert_allclose(np.sort(dist), np.sort(ref), atol=2e-7)
    np.testing.assert_allclose(normal, [[0.0, 0.0, 1.0]] * 2, atol=1e-6)
    code = ("import json, sys; sys.path.insert(0, 'tests/contact'); import test_mjw_boxbox as T; "
            "d, n = T._contacts(primitive=False); print(json.dumps([d.tolist(), n[:, 2].tolist()]))")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, ALLOY_MJW_EPA="stock",
               PYTHONPATH=os.pathsep.join(p for p in (str(root), os.environ.get("PYTHONPATH")) if p))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=root, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    stock_dist, stock_nz = json.loads(r.stdout.strip().splitlines()[-1])
    assert all(z < -0.99 for z in stock_nz) and all(x < -0.027 for x in stock_dist), (stock_dist, stock_nz)


def test_switch_reports_why_nothing_changed() -> None:
    m, _ = _model()
    with wp.ScopedDevice("cpu"):
        wm = mjw.put_model(m)
    assert box_box_pairs(wm) == 1
    assert use_primitive_boxbox(SimpleNamespace(use_mujoco_cpu=True, mjw_model=wm)) is not None
    assert use_primitive_boxbox(SimpleNamespace(mjw_model=None)) == "no MuJoCo Warp model"
    flags = int(wm.opt.disableflags)
    assert use_primitive_boxbox(SimpleNamespace(mjw_model=wm)) is None
    assert int(wm.opt.disableflags) == flags | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    no_boxes = mujoco.MjModel.from_xml_string(XML.replace('type="box"', 'type="sphere"'))
    with wp.ScopedDevice("cpu"):
        wm2 = mjw.put_model(no_boxes)
    assert use_primitive_boxbox(SimpleNamespace(mjw_model=wm2)) == "no box-box pairs"
    assert int(wm2.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) == 0
