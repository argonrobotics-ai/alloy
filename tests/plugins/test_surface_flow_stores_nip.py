# SPDX-License-Identifier: Apache-2.0
"""Surface flow, a roller's nip crossing (SCR-SOFT-3-4): a roller splits a cell when its contact line crosses it,
measured across its axis whatever the heading of each exchange's step; other shapes along the step (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store
from alloy.plugins.surface_flow.stores_meter import nip_heading
from alloy.plugins.surface_flow.tool_geometry import Pose, ToolSet
from tests.plugins.test_surface_flow_stores import HW, NX, R, VISCOUS, roller_scene


@wp.kernel
def _headings(ts: ToolSet, k: int, axis: wp.vec3, steps: wp.array[wp.vec2], out: wp.array[wp.vec3]):
    i = wp.tid()
    ps = Pose()
    ps.a = axis
    out[i] = nip_heading(ts, k, ps, steps[i])


def test_a_rollers_nip_crosses_across_its_axis_whatever_the_step():
    """A robot-held cover (SOFT-3's FR3) steps 0.39 mm with a heading wandering 0.1-0.6 deg: bands across each step
    tilted with it and skipped or re-split cells 114 mm out along the axis (full wall hide 0.975; 0.99995 with this
    rule). A roller with its axis along y crosses |step·x| along +-x for steps at +-30 deg; a disc keeps the step's
    own direction and length."""
    tools = (Applicator(body="cover", shape="roller", size=(R, HW), axis=(0.0, 1.0, 0.0), stiffness=1.0e3,
                        store=Store(capacity=1.0e-5)),
             Applicator(body="cover", shape="disc", size=(0.01, 0.01), center=(0.0, 0.0, -R), stiffness=1.0e3,
                        store=Store(capacity=1.0e-5)))
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=VISCOUS, inertia=False,
                       max_depth=1.0e-3, tools=tools)
    build_simulation(TestScene(build=roller_scene(R + 0.05, -0.1), ground=False, gravity=(0.0, 0.0, 0.0), worlds=1,
                               dt=2.0e-3), [flow], "cpu")
    a = np.radians(30.0)
    steps = np.asarray([[np.cos(a), np.sin(a)], [np.cos(a), -np.sin(a)], [-np.cos(a), np.sin(a)]]) * 3.9e-4
    st = wp.array(steps.astype(np.float32), dtype=wp.vec2, device="cpu")
    for k, want in ((0, [[1.0, 0.0, steps[0, 0]], [1.0, 0.0, steps[1, 0]], [-1.0, 0.0, -steps[2, 0]]]),
                    (1, [[s[0] / 3.9e-4, s[1] / 3.9e-4, 3.9e-4] for s in steps])):
        out = wp.zeros(3, dtype=wp.vec3, device="cpu")
        wp.launch(_headings, dim=3, inputs=[flow.tools.set, k, wp.vec3(0.0, 1.0, 0.0), st, out], device="cpu")
        assert np.allclose(out.numpy(), np.asarray(want, np.float32), rtol=0.0, atol=1.0e-7), (k, out.numpy())
