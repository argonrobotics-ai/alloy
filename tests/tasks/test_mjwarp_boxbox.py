# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-Warp's primitive ``box_box`` (used when NATIVECCD is disabled) on a box resting on its side on the floor box,
touching it by 0.005 mm. mujoco-warp 3.16.0 reported one extra contact 240.76 mm deep, and in the r5_bin50 Newton
replay with ``nativeccd=False`` (step 337, body o36) it threw the box out of the bin (6 m error at 2 s). The pinned
mujoco-warp 86e2fae0 (WP0, main c3df572) gives the true four contacts, as MuJoCo C (``mjc_BoxBox``) and MuJoCo-Warp's
GJK/EPA path do. This is now a regression test. See docs/notes/wp13-results.md (piles).

Data (``tests/tasks/data/mjwarp_boxbox_bin50.npz``): the floor and box geom world poses (``*_pos``, ``*_mat``) and
half-sizes at that state. The model is the static floor box plus one free box, no gravity.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

DATA = Path(__file__).parent / "data" / "mjwarp_boxbox_bin50.npz"
TRUE_DEPTH_MM = 0.005
BAD_DEPTH_MM = 240.76
NATIVECCD = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)


def _quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=np.float64))
    return q


def floor_and_box(disable: int) -> mujoco.MjModel:
    z = np.load(DATA)
    spec = mujoco.MjSpec()
    spec.option.gravity = np.zeros(3)
    spec.option.disableflags = disable
    floor = spec.worldbody.add_geom()
    floor.type, floor.size = mujoco.mjtGeom.mjGEOM_BOX, z["floor_size"]
    floor.pos, floor.quat = z["floor_pos"], _quat(z["floor_mat"])
    body = spec.worldbody.add_body()
    body.pos, body.quat = z["box_pos"], _quat(z["box_mat"])
    body.add_freejoint()
    box = body.add_geom()
    box.type, box.size = mujoco.mjtGeom.mjGEOM_BOX, z["box_size"]
    return spec.compile()


def cpu_depths_mm(m: mujoco.MjModel) -> list[float]:
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return [-1e3 * float(c.dist) for c in d.contact[: d.ncon]]


def mjwarp_depths_mm(m: mujoco.MjModel) -> list[float]:
    from alloy._vendor import mujoco_warp as mjw
    import warp as wp

    with wp.ScopedDevice("cpu"):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        mw, dw = mjw.put_model(m), mjw.put_data(m, d, nconmax=64, njmax=256)
        mjw.kinematics(mw, dw)
        mjw.collision(mw, dw)
        n = int(dw.nacon.numpy()[0])
        return [-1e3 * float(x) for x in dw.contact.dist.numpy()[:n]]


@pytest.mark.parametrize("disable", [0, NATIVECCD])
def test_cpu_mujoco_finds_the_resting_contact(disable: int) -> None:
    depths = cpu_depths_mm(floor_and_box(disable))
    assert len(depths) == 4 and max(depths) == pytest.approx(TRUE_DEPTH_MM, abs=2e-3)


def test_mjwarp_gjk_path_matches_cpu() -> None:
    m = floor_and_box(0)
    assert sorted(mjwarp_depths_mm(m)) == pytest.approx(sorted(cpu_depths_mm(m)), abs=2e-3)


def test_mjwarp_primitive_box_box_matches_cpu() -> None:
    m = floor_and_box(NATIVECCD)
    depths = mjwarp_depths_mm(m)  # mujoco-warp 3.16.0: [240.76, 0.005, 0.005, 0.004, 0.004]
    assert len(depths) == 4 and max(depths) == pytest.approx(TRUE_DEPTH_MM, abs=2e-3), (depths, BAD_DEPTH_MM)
    assert sorted(depths) == pytest.approx(sorted(cpu_depths_mm(m)), abs=2e-3)
