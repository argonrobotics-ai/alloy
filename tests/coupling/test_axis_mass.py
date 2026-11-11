# SPDX-License-Identifier: Apache-2.0
"""Proxy sizing from the most mobile axis (known gap docs/known_gaps/coupling-proxy-isotropic-mass.md; WP3 2026-11-07).

A MuJoCo body's proxy keeps MuJoCo's isotropic mass 1 / body_invweight0; the exchange clock's interface bound, the
strong-pair mass and the exchange stiffness cap take its mass along its most mobile axis (``alloy.coupling.axis_mass``).
Rows: C7's gripper pads (0.56 kg on a jaw slider under a 1 kg carriage) have a 1.329 kg proxy and a 0.56 kg axis
mass; a 1.5 kg body on one slider has a 4.5 kg proxy and a hinged tip 3x its mass; a free body's agree."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.coupling.axis_mass import axis_mass_ratio

XML = """<mujoco><worldbody>
  <body name="free" pos="0 0 1"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1.5"/></body>
  <body name="slider" pos="1 0 1"><joint type="slide" axis="1 0 0"/>
    <geom type="box" size="0.05 0.05 0.05" mass="1.5"/></body>
  <body name="hinge" pos="2 0 1"><joint type="hinge" axis="0 1 0"/>
    <geom type="sphere" pos="0.05 0 0" size="0.002" mass="0.0012"/></body>
  <body name="carriage" pos="3 0 1"><joint type="slide" axis="0 0 1"/><geom type="box" size="0.05 0.05 0.02" mass="1"/>
    <body name="pad_a" pos="0 0.05 -0.05"><joint type="slide" axis="0 1 0"/>
      <geom type="box" size="0.01 0.01 0.02" mass="0.56"/></body>
    <body name="pad_b" pos="0 -0.05 -0.05"><joint type="slide" axis="0 -1 0"/>
      <geom type="box" size="0.01 0.01 0.02" mass="0.56"/></body>
  </body>
</worldbody></mujoco>"""
PAD_ISO = 3.0 / (1.0 / 0.56 + 1.0 / 2.12)          # trace fit of the pad's W: 1/0.56 along the jaw, 1/2.12 along z


def _solver() -> newton.solvers.SolverMuJoCo:
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(XML)
    return newton.solvers.SolverMuJoCo(b.finalize(device="cpu"), use_mujoco_cpu=True)


def test_axis_mass_ratio_per_body():
    s = _solver()
    ratio = axis_mass_ratio(s)
    labels = [str(lab).split("/")[-1] for lab in s.model.body_label]
    r = dict(zip(labels, ratio))
    assert r["free"] == pytest.approx(1.0, abs=1e-9)
    assert r["slider"] == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert r["hinge"] == pytest.approx(1.0 / 3.0, rel=1e-3)
    assert r["carriage"] == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert r["pad_a"] == pytest.approx(0.56 / PAD_ISO, rel=1e-6) == r["pad_b"]          # 0.421
    pad = s.mj_model.body("worldbody_carriage_pad_a").id
    assert 1.0 / s.mj_model.body_invweight0[pad, 0] == pytest.approx(PAD_ISO, rel=1e-6)   # 1.329 kg installed


def test_c7_gripper_sizes_its_clock_and_cap_from_the_pad_axis_mass():
    """C7's rig: the pair mass is the pads' 0.56 kg (not their 1.329 kg proxy), the clock bound is 0.5 sqrt(0.56 /
    2.5e5) s, the cap reads 0.56 kg for each pad, and the installed proxies keep 1.329 kg."""
    from alloy.coupling.pinch_rigs import SHAPE_KE, _cloth_scheduler, gripper_model
    from alloy.coupling.types import SchedulerPolicy

    model = gripper_model()
    sched = _cloth_scheduler(model, False, policy=SchedulerPolicy(exchange_stiffness_c=2.0))
    assert sched._pair_mass[("mjc", "vbd")] == pytest.approx(0.56, rel=1e-4)
    assert sched._interface_bound() == pytest.approx(0.5 * math.sqrt(0.56 / SHAPE_KE), rel=1e-4)
    es = next(e for e in sched._sets if e.kind == "body")
    labels = [str(lab) for lab in model.body_label]
    pads = [i for i, g in enumerate(es.src_global.numpy()) if "pad" in labels[g]]
    assert len(pads) == 2
    np.testing.assert_allclose(es.meff.numpy()[pads], PAD_ISO, rtol=1e-4)               # the proxy is unchanged
    np.testing.assert_allclose(es.m_axis[pads], 0.56, rtol=1e-4)
    cap = sched._caps[0]
    np.testing.assert_allclose(cap.cap_mass.numpy()[es.proxy_local.numpy()[pads]], 0.56, rtol=1e-4)
