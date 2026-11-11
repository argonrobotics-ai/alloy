# SPDX-License-Identifier: Apache-2.0
"""Cloth bending hysteresis for the VBD path (gap #4, WP11): a fold holds a crease by fibre friction (KES 2HB).

Kawabata's bending law (KES-FB2): M = B K + HB sgn(dK/dt), the moment per unit width M of a fabric bent to
curvature K, B the bending rigidity (KES B) and 2HB the hysteresis width of the loop (KES 2HB). Released from a
fold, a fabric springs back until the elastic moment B K falls to HB, so it keeps the curvature HB / B; VBD's
elastic hinges spring all the way back (docs/known_gaps/cloth-sleeve-fold-springback.md).

The hinge (one VBD edge): the elastic hinge (stiffness k, the card's B through the plate map, authored rest angle
theta0) in parallel with a friction element of stiffness LOCK k whose rest angle theta_s slides so that its moment
LOCK k (theta - theta_s) stays within +-hb, hb = HINGE_C HB (the plate map's factor, so hb / k reads HB / B). Both
live in the one VBD hinge: stiffness (1 + LOCK) k, rest angle (theta0 + LOCK theta_s) / (1 + LOCK). Sliding, the
hinge's moment is k (theta - theta0) +- hb, Kawabata's law; reversed, the friction element holds until its moment
has crossed 2 hb, so a loop is 2 HB wide. theta_s and the rest angle are updated once per step on the converged
positions (as the J2 tets), so a step's increment is first taken elastically at (1 + LOCK) k.

Per edge: ``cloth:bend_hb`` (N, moment per unit length; 0 = elastic, the stock hinge) written by the emitter from the
card's ``elastic.bending_hysteresis`` (2HB); the emitter also scales the edge's stiffness by 1 + LOCK. State for
snapshots: ``_bend_theta_s`` and ``model.edge_rest_angle`` (``arrays()``). Without a marked edge nothing here runs
and the solver is the stock one bit for bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


@wp.func
def wrap_pi(a: float) -> float:
    return wp.atan2(wp.sin(a), wp.cos(a))


@wp.func
def hinge_angle(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> float:
    """VBD's hinge angle (alloy/solvers/film/hinge.py's, kept here: importing the film package loads this solver)."""
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    l1 = wp.length(n1)
    l2 = wp.length(n2)
    le = wp.length(e)
    if l1 < 1.0e-12 or l2 < 1.0e-12 or le < 1.0e-12:
        return 0.0
    return wp.atan2(wp.dot(wp.cross(n1 / l1, n2 / l2), e / le), wp.dot(n1 / l1, n2 / l2))


NAMESPACE = "cloth"
HB = "bend_hb"
LOCK = 10.0                    # friction element stiffness / the hinge's elastic stiffness (KES loop corner width)


def register_attributes(builder: Any) -> None:
    """Declare the per-edge friction moment on a ModelBuilder (idempotent)."""
    from alloy._vendor import newton  # noqa: PLC0415

    key = f"{NAMESPACE}:{HB}"
    if key not in builder.custom_attributes:
        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name=HB, namespace=NAMESPACE, frequency=newton.Model.AttributeFrequency.EDGE, dtype=wp.float32,
            default=0.0))


def mark_edges(builder: Any, start: int, end: int, hb: float) -> int:
    """Give the bending edges [start, end) of ``builder`` the friction moment ``hb`` (N) and stiffness (1 + LOCK) k;
    boundary edges (one wing) carry no bending and stay as they are. Returns the count marked."""
    register_attributes(builder)
    attr = builder.custom_attributes[f"{NAMESPACE}:{HB}"]
    if attr.values is None:
        attr.values = {}
    n = 0
    for e in range(start, end):
        o0, o1 = builder.edge_indices[e][0], builder.edge_indices[e][1]
        ke, kd = builder.edge_bending_properties[e]
        if o0 < 0 or o1 < 0 or ke <= 0.0:
            continue
        attr.values[e] = float(hb)
        builder.edge_bending_properties[e] = (float(ke) * (1.0 + LOCK), float(kd))
        n += 1
    return n


@wp.kernel(enable_backward=False)
def _slide(x: wp.array[wp.vec3], edge_indices: wp.array2d[wp.int32], bending: wp.array2d[float],
           hb: wp.array[float], rest0: wp.array[float], lock: float, theta_s: wp.array[float],
           rest_angle: wp.array[float]):
    """The friction element's rest slides to keep its moment within +-hb; the hinge's rest follows (module doc)."""
    e = wp.tid()
    h = hb[e]
    if h <= 0.0:
        return
    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    if i0 < 0 or i1 < 0:
        return
    th = hinge_angle(x[i0], x[i1], x[edge_indices[e, 2]], x[edge_indices[e, 3]])
    kf = bending[e, 0] * lock / (1.0 + lock)
    m = kf * wrap_pi(th - theta_s[e])
    if wp.abs(m) > h:
        theta_s[e] = wrap_pi(th - wp.sign(m) * h / kf)
    rest_angle[e] = wrap_pi(rest0[e] + lock / (1.0 + lock) * wrap_pi(theta_s[e] - rest0[e]))


@wp.kernel(enable_backward=False)
def _reset(edge_world: wp.array[wp.int32], world_mask: wp.array[wp.bool], hb: wp.array[float],
           rest0: wp.array[float], theta_s: wp.array[float], rest_angle: wp.array[float]):
    e = wp.tid()
    if hb[e] <= 0.0:
        return
    w = edge_world[e]
    if w < 0:
        w = world_mask.shape[0] - 1                  # Newton's mask: the last entry selects the global world (-1)
    if world_mask[w]:
        theta_s[e] = rest0[e]
        rest_angle[e] = rest0[e]


class BendHysteresis:
    """The hysteretic hinges of a VBD model (built only when some edge has ``cloth:bend_hb`` > 0)."""

    def __init__(self, model: Any, hb: wp.array) -> None:
        self.hb = hb
        self.rest0 = wp.clone(model.edge_rest_angle)
        self.theta_s = wp.clone(model.edge_rest_angle)
        ei = model.edge_indices.numpy()
        pw = model.particle_world.numpy() if getattr(model, "particle_world", None) is not None else None
        world = pw[ei[:, 2]] if pw is not None and len(ei) else np.zeros(len(ei), np.int32)
        self.edge_world = wp.array(np.asarray(world, np.int32), dtype=wp.int32, device=model.device)
        self.count = int((hb.numpy() > 0.0).sum())

    @classmethod
    def for_model(cls, model: Any) -> BendHysteresis | None:
        attrs = getattr(model, NAMESPACE, None)
        hb = getattr(attrs, HB, None) if attrs is not None else None
        if hb is None or not int(getattr(model, "edge_count", 0) or 0) or not float(hb.numpy().max(initial=0.0)) > 0:
            return None
        return cls(model, hb)

    def after_step(self, model: Any, particle_q: wp.array) -> None:
        """Explicit return map at the step's end: within one step the friction moment can pass hb by up to
        LOCK k |dtheta|, capped at the next update (dissipative); a flapping sheet sees a one-step stiffness
        transient (WP2 review 2026-11-08)."""
        wp.launch(_slide, dim=model.edge_count,
                  inputs=[particle_q, model.edge_indices, model.edge_bending_properties, self.hb, self.rest0,
                          float(LOCK), self.theta_s, model.edge_rest_angle], device=model.device)

    def reset(self, model: Any, world_mask: wp.array | None = None) -> None:
        if world_mask is None:
            wp.copy(self.theta_s, self.rest0)
            wp.copy(model.edge_rest_angle, self.rest0)
            return
        wp.launch(_reset, dim=model.edge_count,
                  inputs=[self.edge_world, world_mask, self.hb, self.rest0, self.theta_s, model.edge_rest_angle],
                  device=model.device)

    def arrays(self, model: Any) -> dict[str, wp.array]:
        """Per-edge state for snapshots: the friction rests and the hinges' rest angles."""
        return {"_bend_theta_s": self.theta_s, "_bend_rest_angle": model.edge_rest_angle}


def card_hb(material: Any) -> float:
    """hb (N) of a cloth material card: HINGE_C x HB, HB = elastic.bending_hysteresis (2HB) / 2; 0 without it."""
    from alloy.solvers.vbd.cloth_profile import HINGE_C  # noqa: PLC0415

    el = getattr(material, "elastic", None)
    q = getattr(el, "bending_hysteresis", None) if el is not None else None
    return float(HINGE_C) * 0.5 * float(q.value) if q is not None else 0.0


__all__ = ["HB", "LOCK", "NAMESPACE", "BendHysteresis", "card_hb", "mark_edges", "register_attributes"]
