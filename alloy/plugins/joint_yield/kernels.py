# SPDX-License-Identifier: Apache-2.0
"""Device kernels of the joint-spring yield (SCR-WP7-15 P2): the 1-D return of ``alloy.solvers.plastic.j2`` per
yielding joint and world, and the write of the moved spring reference into MuJoCo Warp's per-world ``qpos_spring``.

Units: the "strain" is the joint coordinate past its build spring reference, q - s0 [rad or m]; E is the joint
stiffness k [N·m/rad or N/m]; σ_y0 the yield torque or force; H the hardening k_h (same units as k). The plastic
strain q_p is the set, so the spring reference is s0 + q_p and the spring torque k (q - s0 - q_p).

s0 here is the float32 build row of MuJoCo Warp's model. The C route springs about its float64 s0 (+ q_p), so there
the return's elastic part differs from the C spring by the float32 rounding of s0 (about 1e-8 rad at 0.3 rad).
"""

from __future__ import annotations

import warp as wp

from alloy.solvers.plastic.j2 import return_1d


@wp.kernel
def yield_step(joint_q: wp.array(dtype=float), q_index: wp.array2d(dtype=wp.int32), s0: wp.array2d(dtype=float),
               k: wp.array2d(dtype=float), limit: wp.array(dtype=float), hardening: wp.array(dtype=float),
               q_p: wp.array2d(dtype=float), ebar: wp.array2d(dtype=float)):
    """One return per (world, joint) on the step's converged coordinate; writes the set q_p and ε̄p."""
    w, i = wp.tid()
    eps = joint_q[q_index[w, i]] - s0[w, i]
    _sig, ep, eb = return_1d(eps, q_p[w, i], ebar[w, i], k[w, i], limit[i], hardening[i])
    q_p[w, i] = ep
    ebar[w, i] = eb


@wp.kernel
def write_springref(adr: wp.array(dtype=wp.int32), s0: wp.array2d(dtype=float), q_p: wp.array2d(dtype=float),
                    qpos_spring: wp.array2d(dtype=float)):
    """MuJoCo Warp's spring reference of each yielding joint: s0 + q_p (s0 exactly while nothing has yielded)."""
    w, i = wp.tid()
    qpos_spring[w, adr[i]] = s0[w, i] + q_p[w, i]


@wp.kernel
def book_set(joint_q: wp.array(dtype=float), q_index: wp.array2d(dtype=wp.int32), s0: wp.array2d(dtype=float),
             k: wp.array2d(dtype=float), q_p: wp.array2d(dtype=float), booked: wp.array2d(dtype=float),
             work: wp.array(dtype=wp.float64)):
    """Per world, the change since the last step of the spring energy that health's energy check over-counts once
    the reference has moved: it sums 0.5 k (q - s0)^2 about the build reference, the spring holds
    0.5 k (q - s0 - q_p)^2, and the difference is k q_p (q - s0 - q_p / 2). Joints in order (no float atomics)."""
    w = wp.tid()
    total = wp.float64(0.0)
    for i in range(q_p.shape[1]):
        set_ = q_p[w, i]
        over = k[w, i] * set_ * (joint_q[q_index[w, i]] - s0[w, i] - 0.5 * set_)
        total += wp.float64(over - booked[w, i])
        booked[w, i] = over
    work[w] = work[w] + total
