# SPDX-License-Identifier: Apache-2.0
"""DC4's findings on preloaded members (DIMM clips): the limit is active at q = 0 (joint margin), one step's snap
under the preload is bounded (``Member.snap_step``), a stiff stop holds the preload (``stop_stiffness``), and thin
touched boxes are flagged (``design.check_thin_boxes``)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy._vendor import newton

from alloy.latch import mjcf
from alloy.latch.builder import add_latch
from alloy.latch.design import check_spec, check_thin_boxes
from alloy.latch.mass import axis_inertia, member_props, snap_per_step
from alloy.latch.spec import Box, LatchSpec, Member, Spring

DT = 1.0 / 1200.0                                   # DC4's step
RATE, PRELOAD = 0.5, 0.2                           # N·m/rad, N·m: preload / rate = 0.4 rad past the stop


def _clip(snap_step: float | None = None, stop_stiffness: float | None = 100.0, ramp_t: float = 2e-3) -> LatchSpec:
    clip = Member("clip", "hinge", anchor=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0), travel=0.5,
                  spring=Spring(RATE, PRELOAD, 0.0),
                  shapes=(Box("clip_arm", (0.002, 0.003, 0.01), (0.0, 0.0, 0.01), role="tab"),
                          Box("clip_ramp", (0.5 * ramp_t, 0.003, 0.002), (0.003, 0.0, 0.018), role="ramp")),
                  mass=2.0e-3, snap_step=snap_step, stop_stiffness=stop_stiffness)
    return LatchSpec("dimm_clip_test", root_shapes=(Box("socket", (0.004, 0.004, 0.004)),), root_mass=0.01,
                     members=(clip,), static=True, design_dt=DT)


def _min_q(xml: str, spec: LatchSpec, seconds: float = 0.05) -> float:
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep, m.opt.gravity[:] = DT, 0.0
    d = mujoco.MjData(m)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, mjcf.joint_name(spec, "clip"))
    qs = []
    for _ in range(int(seconds / DT)):
        mujoco.mj_step(m, d)
        qs.append(d.qpos[m.jnt_qposadr[j]])
    return float(min(qs))


def test_limit_is_active_at_the_stop():
    """With the margin the preloaded clip stays within 2·preload/K of its stop from the first step; without it the
    first step is free and it overshoots."""
    spec = _clip()
    on = _min_q(mjcf.latch_mjcf(spec), spec)
    off = _min_q(mjcf.latch_mjcf(spec, margin=0.0), spec)
    assert 'margin="' not in mjcf.latch_mjcf(spec, margin=0.0)
    assert on >= -2.0 * PRELOAD / 100.0
    assert off < 2.0 * on


def test_builder_carries_the_margin():
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    h = add_latch(b, _clip())
    model = b.finalize(device="cpu")
    assert model.mujoco.limit_margin.numpy()[h.dofs["clip"]] == pytest.approx(mjcf.LIMIT_MARGIN)
    assert 'margin="1e-05"' in mjcf.latch_mjcf(_clip())


def test_snap_step_bounds_one_step_of_preload():
    m0, m1 = _clip().members[0], _clip(snap_step=0.005).members[0]
    s0 = snap_per_step(m0, DT, axis_inertia(m0, member_props(m0, DT)))
    s1 = snap_per_step(m1, DT, axis_inertia(m1, member_props(m1, DT)))
    assert s0 == pytest.approx(PRELOAD / RATE * 0.35 ** 2, rel=1e-6)      # omega·h floor only: 0.049 rad
    assert s1 == pytest.approx(0.005, rel=1e-6)
    assert any("snap_step" in i.message for i in check_spec(_clip()))
    assert not any("snap_step" in i.message for i in check_spec(_clip(snap_step=0.005)))


def test_thin_touched_boxes_are_flagged():
    thin = _clip(ramp_t=0.4e-3)
    assert [i.where for i in check_thin_boxes(thin, contact_depth=0.3e-3)] == ["clip_ramp"]
    assert not check_thin_boxes(_clip(), contact_depth=0.3e-3)
    assert any(i.where == "clip_ramp" for i in check_spec(thin, contact_depth=0.3e-3))
    np.testing.assert_equal(len(check_thin_boxes(thin, contact_depth=0.1e-3)), 0)
