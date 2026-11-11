# SPDX-License-Identifier: Apache-2.0
"""BUL-0002 minimal repro: MuJoCo's native-CCD multi-contact (MULTICCD) reports a 41.6 mm deep contact between two
convex hulls that touch by 0.06 mm, when the hulls are cut to 64 vertices (Newton's default ``maxhullvert``). CPU
MuJoCo 3.16 and mujoco-warp 3.16 give the same wrong depth; multi-CCD off, libccd, or the full hulls the recording used
give the true one. MuJoCo 3.17 and mujoco-warp 3.17 (WP0's Newton bump, 2026-11-09) fixed it: each reports one contact
near the true depth. See docs/notes/wp13-bul0002.md.

Data (``tests/tasks/data/bul0002_hulls.npz``): the two source meshes as vendored (left D405 wrist-mount hull, right
finger-tip hull), their geom poses in the body frame and both bodies' world poses at the failing state (tick 61
substep 1 of the BUL-0002 replay). The model is two free bodies, one mesh geom each, no gravity.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from alloy.tasks.hull_split import outward_faces, split_hull

DATA = Path(__file__).parent / "data" / "bul0002_hulls.npz"
TRUE_DEPTH_M = 6.1e-5      # exact SAT penetration of the full hulls at this pose (notes/wp13-spikes/bul0002_geom.py)
BAD_DEPTH_M = 0.0416       # what the BUL-0002 replay saw from MuJoCo-Warp inside the scene
MULTICCD, NATIVECCD = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD), int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)


def _pieces(key: str, z: np.lib.npyio.NpzFile, split: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    if not split:
        return [(z[f"{key}_vert"].astype(np.float64), z[f"{key}_face"])]
    return [(p, outward_faces(p)) for p in split_hull(z[f"{key}_vert"].astype(np.float64)).pieces]


def hull_pair_model(maxhullvert: int, disable: int = 0, split: bool = False) -> mujoco.MjModel:
    """``split``: each hull replaced by its error-bounded pieces, as the converter writes T12 since the SB-6 split."""
    z = np.load(DATA)
    spec = mujoco.MjSpec()
    spec.option.gravity = np.zeros(3)
    spec.option.disableflags = disable
    for key in ("a", "b"):
        body = spec.worldbody.add_body()
        body.name, body.pos, body.quat = key, z[f"{key}_body_pos"], z[f"{key}_body_quat"]
        body.add_freejoint()
        for i, (vert, face) in enumerate(_pieces(key, z, split)):
            mesh = spec.add_mesh()
            mesh.name, mesh.maxhullvert = f"{key}{i}", maxhullvert
            mesh.uservert, mesh.userface = vert.ravel(), np.asarray(face).ravel()
            geom = body.add_geom()
            geom.type, geom.meshname = mujoco.mjtGeom.mjGEOM_MESH, f"{key}{i}"
            geom.pos, geom.quat = z[f"{key}_geom_pos"], z[f"{key}_geom_quat"]
    return spec.compile()


def cpu_depths(m: mujoco.MjModel) -> list[float]:
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return [-float(c.dist) for c in d.contact[: d.ncon]]


def mjwarp_depths(m: mujoco.MjModel) -> list[float]:
    from alloy._vendor import mujoco_warp as mjw
    import warp as wp

    with wp.ScopedDevice("cpu"):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        mw, dw = mjw.put_model(m), mjw.put_data(m, d, nconmax=64, njmax=256)
        mjw.kinematics(mw, dw)
        mjw.collision(mw, dw)
        n = int(dw.nacon.numpy()[0])
        return [-float(x) for x in dw.contact.dist.numpy()[:n]]


@pytest.mark.parametrize(("maxhullvert", "disable"), [(-1, 0), (64, MULTICCD), (64, NATIVECCD)])
def test_true_depth_without_the_failing_combination(maxhullvert: int, disable: int) -> None:
    depths = cpu_depths(hull_pair_model(maxhullvert, disable))
    assert depths and max(depths) == pytest.approx(TRUE_DEPTH_M, abs=2e-5)


def test_mjwarp_and_cpu_mujoco_report_the_true_depth_on_the_failing_hulls() -> None:
    """At 64 vertices with native CCD and multi-CCD on, each engine reports one contact near the true depth: CPU
    MuJoCo 3.17 0.049 mm, MuJoCo Warp 3.17 0.056 mm (true 0.061 mm; 3.16: 41.6 mm in both). MuJoCo Warp's depth is
    CPU MuJoCo's own with multi-CCD off (0.0565 mm)."""
    m = hull_pair_model(64)
    for depths in (cpu_depths(m), mjwarp_depths(m)):
        assert len(depths) == 1 and depths[0] == pytest.approx(TRUE_DEPTH_M, abs=2e-5), depths
    assert mjwarp_depths(m) == pytest.approx(cpu_depths(hull_pair_model(64, MULTICCD)), abs=1e-7)


def test_multiccd_true_depth_on_64_vertex_hulls() -> None:
    """BUL-0002 is fixed in MuJoCo 3.17: native-CCD multi-contact on these 64-vertex hulls reports the true depth
    (MuJoCo 3.16: 41.6 mm, ``BAD_DEPTH_M``; this was a strict xfail until WP0's bump)."""
    depths = cpu_depths(hull_pair_model(64))
    assert max(depths) == pytest.approx(TRUE_DEPTH_M, abs=2e-5), (depths, BAD_DEPTH_M)


def test_split_pieces_report_no_false_depth() -> None:
    """The SB-6 pieces (at most 64 vertices, within 0.2 mm, inside the recorded hulls) at the BUL-0002 pose, under the
    recorded flags (native CCD + multi-CCD): no contact deeper than the true 0.061 mm, in CPU MuJoCo and MuJoCo Warp."""
    m = hull_pair_model(-1, split=True)
    for depths in (cpu_depths(m), mjwarp_depths(m)):
        assert max(depths, default=0.0) <= TRUE_DEPTH_M + 1e-6, depths
