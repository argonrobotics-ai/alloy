# SPDX-License-Identifier: Apache-2.0
"""SCR-WP3-9 (BUL-0011): the MuJoCo factory puts box-box pairs on MuJoCo Warp's primitive collider right after
construction (both the implicitfast and the rk4 branch) and leaves models without box-box pairs alone."""

from __future__ import annotations

import pytest

newton = pytest.importorskip("alloy._vendor.newton")
wp = pytest.importorskip("warp")
pytest.importorskip("alloy._vendor.mujoco_warp")

from alloy._vendor.mujoco_warp import DisableBit  # noqa: E402

from alloy.scene.builtin_plugins import _mujoco  # noqa: E402

NATIVECCD = int(DisableBit.NATIVECCD)


def _model(box: bool) -> newton.Model:
    """A static slab and one free body resting on it: a box (one box-box pair) or a sphere (none)."""
    b = newton.ModelBuilder()
    b.add_shape_box(-1, hx=0.3, hy=0.3, hz=0.01, xform=wp.transform(wp.vec3(0.0, 0.0, -0.01), wp.quat_identity()))
    body = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0075), wp.quat_identity()), mass=0.1)
    if box:
        b.add_shape_box(body, hx=0.02, hy=0.01, hz=0.0075)
    else:
        b.add_shape_sphere(body, radius=0.0075)
    b.add_articulation([b.add_joint_free(body)])
    return b.finalize(device="cpu")


@pytest.mark.parametrize("integrator", ["implicitfast", "rk4"])
def test_factory_sets_primitive_boxbox_on_box_pairs(integrator: str) -> None:
    with wp.ScopedDevice("cpu"):
        solver = _mujoco(_model(box=True), {"integrator": integrator})
    assert int(solver.mjw_model.opt.disableflags) & NATIVECCD


def test_factory_leaves_models_without_box_pairs() -> None:
    with wp.ScopedDevice("cpu"):
        solver = _mujoco(_model(box=False), {})
    assert not int(solver.mjw_model.opt.disableflags) & NATIVECCD
