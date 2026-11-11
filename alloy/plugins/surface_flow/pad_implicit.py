# SPDX-License-Identifier: Apache-2.0
"""The implicit pad (``Applicator(implicit=True)``): backward Euler on the tool's rigid motion against its pad.

The explicit pad (``p = k·δ`` plus its damper) is stable only below removal's ``pad_response`` bound, which a light
long tool on a stiff paste exceeds by orders of magnitude (WP6: a 0.12 kg jointer needs ~5e7 Pa/m at 1.25 ms, the
bound is ~5e5). Here the pad's stiffness and damping act implicitly on the tool body's twist ``ξ = (v, ω)``:

    K = Σ k·a·J_cᵀJ_c,   J_c = [nᵀ, (r_c × n)ᵀ]       (r_c from the tool's COM, n the pad's outward normal)
    (M + dt·c·K + dt²·K)·ξ' = M·ξ + dt·(G + m·g)      (G: the springs' wrench at the current penetration)
    W = G - (dt·K + c·K)·ξ'                            (the wrench added to the tool; its reaction to the chart's body)

with ``M`` the tool body's own mass and world inertia. It is stable for any stiffness. A tool held through joints is
heavier than its body, so the pad is a little softer than ``k`` (more give), and the resting pressure is still the
load over the area. ``ratio`` (|W·n| over the explicit normal force) carries the same pressure to the forming law and
to Coulomb friction. Limit: the chart's own motion is not in the solve (the paste's body is taken as immovable).
"""

from __future__ import annotations

import warp as wp


@wp.struct
class PadSums:
    """The pad's springs over its contact cells: stiffness blocks and the springs' wrench (world frame, at the COM)."""

    ktt: wp.mat33          # Σ k·a·n nᵀ
    ktr: wp.mat33          # Σ k·a·n (r × n)ᵀ
    krr: wp.mat33          # Σ k·a·(r × n)(r × n)ᵀ
    force: wp.vec3         # Σ k·a·δ·n (+ the damper's offset at the paste's sink speed)
    torque: wp.vec3


@wp.func
def pad_add(s: PadSums, ka: float, n: wp.vec3, r: wp.vec3, push: float) -> PadSums:
    """One contact cell: spring ``ka = k·a`` [N/m] at arm ``r``, pushing ``push`` [N] along ``n``."""
    rn = wp.cross(r, n)
    out = PadSums()
    out.ktt = s.ktt + ka * wp.outer(n, n)
    out.ktr = s.ktr + ka * wp.outer(n, rn)
    out.krr = s.krr + ka * wp.outer(rn, rn)
    out.force = s.force + push * n
    out.torque = s.torque + push * rn
    return out


@wp.func
def pad_zero() -> PadSums:
    s = PadSums()
    s.ktt = wp.mat33(0.0)
    s.ktr = wp.mat33(0.0)
    s.krr = wp.mat33(0.0)
    s.force = wp.vec3(0.0)
    s.torque = wp.vec3(0.0)
    return s


@wp.func
def pad_wrench(s: PadSums, mass: float, inertia: wp.mat33, v: wp.vec3, omega: wp.vec3, gravity: wp.vec3,
               damping: float, dt: float) -> wp.spatial_vector:
    """The implicit pad's wrench on the tool (force, torque at the COM): the 6x6 backward-Euler solve by its 3x3
    Schur complement. The tool's weight is in the predictor, so a tool resting on its pad under gravity keeps its
    static balance exactly; other loads (a robot's push) are not known here, and add ``dt·(dt + c)·a`` to the give."""
    g = dt + damping                                    # (dt·K + c·K) = g·K
    a11 = wp.diag(wp.vec3(mass)) + dt * g * s.ktt       # M + dt·(c + dt)·K, by blocks
    a12 = dt * g * s.ktr
    a22 = inertia + dt * g * s.krr
    b1 = mass * v + dt * (s.force + mass * gravity)
    b2 = inertia * omega + dt * s.torque
    a11_inv = wp.inverse(a11)
    schur = a22 - wp.transpose(a12) * a11_inv * a12
    x2 = wp.inverse(schur) * (b2 - wp.transpose(a12) * (a11_inv * b1))
    x1 = a11_inv * (b1 - a12 * x2)
    f = s.force - g * (s.ktt * x1 + s.ktr * x2)
    tau = s.torque - g * (wp.transpose(s.ktr) * x1 + s.krr * x2)
    return wp.spatial_vector(f[0], f[1], f[2], tau[0], tau[1], tau[2])


__all__ = ["PadSums", "pad_add", "pad_wrench", "pad_zero"]
