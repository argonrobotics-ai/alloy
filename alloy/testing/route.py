# SPDX-License-Identifier: Apache-2.0
"""WP2 test helpers: build MaterialCard / ObjectCard / ResolvedObject in code (no WP1 library needed).

Values are tagged `source="model", note="test fixture"`; extents are the OBB of the shape source (l1 ≥ l2 ≥ l3).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from alloy.cards.types import (CurveSource, Elastic, Features, Fluid, LoadEnvelope, MaterialCard, ObjectCard,
                               ParticleSource, Plastic, Primitive, ResolvedObject, Surface)
from alloy.core.ids import digest_of
from alloy.core.quantity import Q
from alloy.route.types import SceneContext


def q(v: float) -> Q:
    return Q(float(v), "model", note="test fixture")


def material(name: str, density: float, *, E: float | None = None, nu: float = 0.3, mu: float = 0.5,
             e: float = 0.2, k: float | None = None, beta: float | None = None, fluid_mu: float | None = None,
             yield_stress: float | None = None, friction_angle: float | None = None) -> MaterialCard:
    surface = Surface(mu_static=q(mu * 1.1), mu_dynamic=q(mu), restitution=q(e),
                      contact_stiffness=q(k) if k is not None else None)
    elastic = None if E is None else Elastic(youngs=q(E), poisson=q(nu),
                                             rayleigh_beta=q(beta) if beta is not None else None)
    plastic = None
    if yield_stress is not None or friction_angle is not None:
        plastic = Plastic(yield_stress=q(yield_stress) if yield_stress is not None else None,
                          friction_angle=q(friction_angle) if friction_angle is not None else None)
    fluid = Fluid(viscosity=q(fluid_mu)) if fluid_mu is not None else None
    return MaterialCard(name=name, density=q(density), surface=surface, elastic=elastic, plastic=plastic,
                        fluid=fluid)


def extents_of(shape: Any) -> tuple[float, float, float]:
    """OBB extents l1 ≥ l2 ≥ l3 of a shape source [m]."""
    if isinstance(shape, Primitive):
        dims = _primitive_dims(shape)
    elif isinstance(shape, CurveSource):
        pts = np.asarray(shape.points)
        span = np.ptp(pts, axis=0)
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        dims = (max(length, *span), 2 * shape.radius, 2 * shape.radius)
    elif isinstance(shape, ParticleSource):
        dims = tuple(abs(h - lo) for lo, h in zip(*shape.region))
    else:
        raise TypeError(f"no extents for {type(shape).__name__}")
    return tuple(sorted((float(x) for x in dims), reverse=True))  # type: ignore[return-value]


def _primitive_dims(p: Primitive) -> tuple[float, float, float]:
    s = p.size
    if p.kind in ("box", "ellipsoid"):
        return (2 * s[0], 2 * s[1], 2 * s[2])
    if p.kind == "sphere":
        return (2 * s[0],) * 3
    if p.kind == "capsule":
        return (2 * (s[1] + s[0]), 2 * s[0], 2 * s[0])
    if p.kind in ("cylinder", "cone"):
        return (2 * s[1], 2 * s[0], 2 * s[0])
    return (10.0, 10.0, 0.0)


def obj(name: str, shape: Any, mat: MaterialCard, *, worst_case: Mapping[str, float] | None = None,
        **card_kw: Any) -> ResolvedObject:
    """ResolvedObject of a card built in code (material resolved, extents from the shape)."""
    card = ObjectCard(name=name, shape=shape, material=mat.name, **card_kw)
    return ResolvedObject(card=card, material=mat, part_materials={}, card_hash=digest_of((card, mat)),
                          extents=extents_of(shape), worst_case=dict(worst_case or {}), default_leaves=())


def loads(**kw: float) -> LoadEnvelope:
    return replace(LoadEnvelope(), **{k: q(v) for k, v in kw.items()})


def features(**kw: Any) -> Features:
    return replace(Features(), **{k: (q(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                                  for k, v in kw.items()})


def context(registered: frozenset[str], *, free_bodies: int = 1, touching: Mapping[str, tuple[str, ...]] = {},
            **extra: Any) -> SceneContext:
    return SceneContext(rigid_partners=dict(touching), free_bodies_per_world=free_bodies,
                        registered_routes=registered, **extra)


def rod(name: str, mat: MaterialCard, radius: float, length: float, **kw: Any) -> ResolvedObject:
    return obj(name, Primitive("cylinder", (radius, 0.5 * length)), mat, **kw)


def box(name: str, mat: MaterialCard, hx: float, hy: float, hz: float, **kw: Any) -> ResolvedObject:
    return obj(name, Primitive("box", (hx, hy, hz)), mat, **kw)


STEEL = dict(density=7850.0, E=200e9, nu=0.3, mu=0.4, e=0.6)
WOOD = dict(density=600.0, E=10e9, nu=0.35, mu=0.5, e=0.4)
ALUMINIUM = dict(density=2700.0, E=70e9, nu=0.33, mu=0.4, e=0.5)
RUBBER = dict(density=1100.0, E=2e6, nu=0.45, mu=0.9, e=0.6)
COTTON = dict(density=1500.0, E=1e6, nu=0.3, mu=0.6, e=0.1)
PU_FOAM = dict(density=30.0, E=1e5, nu=0.3, mu=0.8, e=0.1)
CABLE_PVC = dict(density=1300.0, E=5e7, nu=0.4, mu=0.6, e=0.2)


def circle_area(r: float) -> float:
    return math.pi * r * r
