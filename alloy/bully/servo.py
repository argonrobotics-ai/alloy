# SPDX-License-Identifier: Apache-2.0
"""A force- and speed-limited Cartesian servo for bully targets whose solver ignores effort limits.

``SolverXPBD`` does not support ``joint_effort_limit``, so a D6 drive with ``target_ke`` behaves like an unlimited
spring: jammed against an obstacle it stores ``ke e^2 / 2`` and releases it at once (BUL-0004, BUL-0005 in
``bench/bully/corpus``). A real arm is a servo with a speed and a force limit, which is what this module applies
instead. Per axis, before every substep:

    v_des = clamp(gain * (q_target - q), -v_lim, v_lim)
    f     = clamp(kd * (v_des - v) - m g, -f_max, f_max)        (gravity compensated, then saturated)

written to ``State.body_f`` of the driven body (the D6 axes carry no drive gains). ``q`` is the body's
displacement from its joint anchor along world x, y, z (a three-axis prismatic D6 with identity frames), and the
targets are read from ``Control.joint_target_q``, so the action space is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.core.errors import BullyError


@dataclass(frozen=True)
class ServoLimits:
    """Servo law parameters (SI)."""

    gain: float  # [1/s] position error -> desired speed
    kd: float  # [N s/m] speed error -> force
    v_lim: float  # [m/s] desired-speed limit
    f_max: float  # [N] force limit per axis


@wp.kernel
def _servo_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    target_q: wp.array[wp.float32],
    body: wp.array[wp.int32],
    q_start: wp.array[wp.int32],
    anchor: wp.array[wp.vec3],
    mass: wp.array[wp.float32],
    gravity: wp.array[wp.vec3],
    lim: wp.vec4,  # gain, kd, v_lim, f_max
    body_f: wp.array[wp.spatial_vector],
):
    w = wp.tid()
    b = body[w]
    s = q_start[w]
    q = wp.transform_get_translation(body_q[b]) - anchor[w]
    v = wp.spatial_top(body_qd[b])
    g = gravity[w]
    f = wp.vec3(0.0, 0.0, 0.0)
    for k in range(3):
        v_des = wp.clamp(lim[0] * (target_q[s + k] - q[k]), -lim[2], lim[2])
        f[k] = wp.clamp(lim[1] * (v_des - v[k]) - mass[b] * g[k], -lim[3], lim[3])
    body_f[b] = wp.spatial_vector(f, wp.vec3(0.0, 0.0, 0.0))


class CartesianServo:
    """Drives every D6 child body of ``model`` (one per world) towards its ``joint_target_q``; append ``apply`` to
    ``DirectSim.pre_substep``. Anchors are the bodies' positions at construction (joint coordinates zero)."""

    def __init__(self, model: newton.Model, state: newton.State, control: newton.Control, limits: ServoLimits) -> None:
        jt, child, q_start = (model.joint_type.numpy(), model.joint_child.numpy(), model.joint_q_start.numpy())
        d6 = np.nonzero(jt == int(newton.JointType.D6))[0]
        worlds = int(model.world_count)
        if d6.size != worlds:
            raise BullyError(f"CartesianServo needs one D6 joint per world, found {d6.size} for {worlds} worlds")
        bodies = child[d6].astype(np.int32)
        dev = model.device
        self._model, self._control = model, control
        self._body = wp.array(bodies, dtype=wp.int32, device=dev)
        self._q_start = wp.array(q_start[d6].astype(np.int32), dtype=wp.int32, device=dev)
        self._anchor = wp.array(state.body_q.numpy()[bodies, :3], dtype=wp.vec3, device=dev)
        self._lim = wp.vec4(limits.gain, limits.kd, limits.v_lim, limits.f_max)

    def apply(self, state: newton.State) -> None:
        m = self._model
        wp.launch(_servo_kernel, dim=self._body.shape[0], device=m.device,
                  inputs=[state.body_q, state.body_qd, self._control.joint_target_q, self._body, self._q_start,
                          self._anchor, m.body_mass, m.gravity, self._lim],
                  outputs=[state.body_f])
