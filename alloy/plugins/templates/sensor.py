# SPDX-License-Identifier: Apache-2.0
"""PluginName: an accelerometer (specific force in the body frame) on one body.

TEMPLATE: a complete, passing sensor plug-in. Replace the measurement and keep the structure: setup() allocates
readings with add_state(), measure() launches kernels that write them after every step, laws() state what must be
true. Check it with:  python -m alloy.plugins check <this file>
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins import Law, LawResult, SensorPlugin, TestScene, body_xform


@wp.kernel
def _accel_kernel(qd_in: wp.array[wp.spatial_vector], qd_out: wp.array[wp.spatial_vector],
                  q_out: wp.array[wp.transform], body: wp.array[wp.int32], gravity: wp.vec3, inv_dt: wp.float32,
                  reading: wp.array[wp.vec3]):
    w = wp.tid()
    b = body[w]
    a0, a1 = qd_in[b], qd_out[b]
    accel = (wp.vec3(a1[0], a1[1], a1[2]) - wp.vec3(a0[0], a0[1], a0[2])) * inv_dt   # COM acceleration, world
    rot = wp.transform_get_rotation(q_out[b])
    reading[w] = wp.quat_rotate_inv(rot, accel - gravity)                          # specific force, body frame


class PluginName(SensorPlugin):
    """Accelerometer: specific force (acceleration minus gravity) of a body's centre of mass, in the body frame
    [m/s²]. At rest on a table it reads +9.81 along the body's up axis; in free fall it reads 0.

    Example::

        imu = PluginName(body="box")
        sim = alloy.plugins.simulation(built, [imu]); sim.step_period(); imu.read()["acceleration"]   # [W, 3]
    """

    name = "plugin_name"
    summary = "accelerometer on one body (template)"

    @dataclass
    class Config:
        body: str = "box"              # label of the body carrying the sensor

    def setup(self, ctx) -> None:
        self.body = ctx.bodies(self.config.body)
        self.gravity = wp.vec3(*ctx.gravity)
        self.reading = self.add_state("acceleration", ctx.zeros(ctx.world_count, wp.vec3))

    def measure(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        wp.launch(_accel_kernel, dim=self.ctx.world_count,
                  inputs=[state_in.body_qd, state_out.body_qd, state_out.body_q, self.body, self.gravity,
                          1.0 / dt], outputs=[self.reading], device=self.ctx.device)

    def test_scene(self) -> TestScene:
        """A box dropped 2 cm onto the ground; it lands and rests."""
        return TestScene(build=_box(self.config.body, 0.07), periods=50)

    def laws(self) -> list[Law]:
        fall = TestScene(build=_box(self.config.body, 10.0), ground=False, periods=10)
        return [Law("reads_g_at_rest", _check_rest, None, "+9.81 m/s² up after landing"),
                Law("reads_zero_in_free_fall", _check_fall, fall, "0 in free fall")]


def _box(label: str, z: float):
    def build(b: newton.ModelBuilder) -> None:
        box = b.add_body(xform=body_xform(0.0, 0.0, z), label=label)
        b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0))
    return build


def _last_reading(run) -> np.ndarray:
    return run.trace.plugin(run.plugin.label, "acceleration")[-1, 0].astype(np.float64)   # world 0, last period


def _check_rest(run) -> LawResult:
    a = _last_reading(run)
    err = float(np.linalg.norm(a - np.array([0.0, 0.0, 9.81])))
    return LawResult.within(err / 9.81, 0.02, f"reading {np.round(a, 3).tolist()} m/s² at rest",
                            "specific force = (v_out - v_in)/dt - gravity, rotated into the body frame")


def _check_fall(run) -> LawResult:
    a = _last_reading(run)
    return LawResult.within(float(np.linalg.norm(a)), 0.01, f"reading {np.round(a, 4).tolist()} m/s² falling",
                            "subtract gravity: a free-falling accelerometer reads zero")
