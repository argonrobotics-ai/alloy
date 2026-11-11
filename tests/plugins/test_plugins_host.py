# SPDX-License-Identifier: Apache-2.0
"""PluginHost: forces enter the step and leave State untouched; state arrays, inputs, time and errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins import ForcePlugin, PluginHost, SensorPlugin
from alloy.plugins.funcs import add_wrench


@wp.kernel
def _push(body: wp.array[wp.int32], fz: wp.array[wp.float32], body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    add_wrench(body_f, body[w], wp.spatial_vector(0.0, 0.0, fz[w], 0.0, 0.0, 0.0))


class Push(ForcePlugin):
    """Pushes the box up by input fz [N]; counts steps in state."""

    name = "push_test"
    summary = "test"

    @dataclass
    class Config:
        body: str = "box"

    def setup(self, ctx):
        self.body = ctx.bodies(self.config.body)
        self.fz = self.add_input("fz", ctx.zeros(ctx.world_count))
        self.count = self.add_state("count", ctx.zeros(ctx.world_count, wp.int32))

    def apply(self, state, forces, dt):
        wp.launch(_push, dim=self.ctx.world_count, inputs=[self.body, self.fz], outputs=[forces.body_f],
                  device=self.ctx.device)


class Bad(SensorPlugin):
    """Registers a non-world-major array."""

    name = "bad_test"

    def setup(self, ctx):
        self.add_state("x", ctx.zeros(ctx.world_count + 1))


class FakeStepper:
    def __init__(self):
        self.seen = []

    def step(self, state_in, state_out, control, contacts, dt):
        self.seen.append(state_in.body_f.numpy().copy())

    def state_arrays(self):
        return {"solver/x": wp.zeros(1)}


def _model(worlds=2):
    b = newton.ModelBuilder()
    box = b.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="box")
    b.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    root = newton.ModelBuilder()
    root.replicate(b, worlds)
    return root.finalize(device="cpu")


def test_forces_reach_the_step_and_state_is_restored():
    model = _model()
    host = PluginHost(model, [Push()], control_dt=0.01)
    stepper = host.install(FakeStepper())
    s0, s1 = model.state(), model.state()
    s0.body_f.assign(np.tile([1.0, 0, 0, 0, 0, 0], (2, 1)).astype(np.float32))   # another writer's force
    host["push_test"].set_input("fz", [5.0, 7.0])
    stepper.step(s0, s1, None, None, 0.01)
    np.testing.assert_allclose(stepper.seen[0][:, [0, 2]], [[1.0, 5.0], [1.0, 7.0]])
    np.testing.assert_allclose(s0.body_f.numpy()[:, 2], 0.0)                     # plug-in force removed
    np.testing.assert_allclose(s0.body_f.numpy()[:, 0], 1.0)                     # other writer kept
    arrays = stepper.state_arrays()
    assert {"solver/x", "plugin/time", "plugin/push_test/count"} <= set(arrays)
    np.testing.assert_allclose(arrays["plugin/time"].numpy(), [0.01, 0.01])


def test_host_errors_say_what_to_change():
    model = _model()
    with pytest.raises(ap.PluginError, match="first dimension must be the world count"):
        PluginHost(model, [Bad()], control_dt=0.01)
    with pytest.raises(ap.PluginError, match="label='...'"):
        PluginHost(model, [Push(), Push()], control_dt=0.01)
    with pytest.raises(ap.PluginError, match="only force and sensor"):
        PluginHost(model, [ap.create("hydraulic_cylinder")], control_dt=0.01)
    with pytest.raises(ap.PluginError, match="no body labelled 'pen'"):
        PluginHost(model, [Push(body="pen")], control_dt=0.01)
    host = PluginHost(model, [Push(label="a"), Push(label="b")], control_dt=0.01)
    stepper = host.install(FakeStepper())
    with pytest.raises(ap.PluginError, match="already runs a PluginHost"):
        PluginHost(model, [Push()], control_dt=0.01).install(stepper)
    with pytest.raises(ap.PluginError, match="inputs"):
        host["a"].set_input("nope", 1.0)
