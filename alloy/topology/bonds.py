# SPDX-License-Identifier: Apache-2.0
"""Breakable bonds as Newton custom attributes (SPEC §3.8, D19).

Each bond is one row of the custom frequency ``topology:bond``. It carries the native element that realises it: a
particle spring (CLOTH_EDGE, TET_CUT, SPRING) or a MuJoCo equality row (RIGID_WELD = weld, JOINT = connect;
SolverMuJoCo has no ``joint_enabled``, newton@ac7f7830:newton/_src/solvers/mujoco/solver_mujoco.py:562). Index
columns declare ``references=`` so ``replicate`` offsets them per world. Negative sentinels are kept
(sim/builder.py:5478-5489). The ``world`` column follows the builder's current world.

Builder helpers here are called by emitters (WP2) while the world template is built. ``bonds_from_model`` turns
the finalized rows into a :class:`~alloy.topology.types.BondSet`.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology.types import BondKind, BondSet

if TYPE_CHECKING:
    from alloy.cards.types import CardLibrary
    from alloy.scene.types import BuiltScene

log = logging.getLogger("alloy.topology")

NS = "topology"
BOND_FREQ = f"{NS}:bond"
INF = math.inf

# (name, dtype, default, references)
_BOND_COLUMNS: tuple[tuple[str, Any, Any, str | None], ...] = (
    ("bond_kind", wp.int32, 0, None),
    ("bond_world", wp.int32, -1, "world"),
    ("bond_particle_a", wp.int32, -1, "particle"),
    ("bond_particle_b", wp.int32, -1, "particle"),
    ("bond_body_a", wp.int32, -1, "body"),
    ("bond_body_b", wp.int32, -1, "body"),
    ("bond_spring", wp.int32, -1, "spring"),
    ("bond_eq", wp.int32, -1, "mujoco:equality_constraint"),
    ("bond_break_force", wp.float32, INF, None),
    ("bond_break_strain", wp.float32, INF, None),
    ("bond_cut_strain", wp.float32, INF, None),     # blade gate (TET_CUT), Failure.cut_min_strain
    ("bond_cut_force", wp.float32, INF, None),      # blade gate (TET_CUT), Failure.cut_min_force
    ("bond_rest", wp.float32, 0.0, None),           # reference length for strain [m] (springs, body bonds)
    ("bond_measure", wp.float32, 0.0, None),        # tributary seam length [m] or area [m^2]
    ("bond_normal", wp.vec3, wp.vec3(0.0, 0.0, 0.0), None),   # seam normal from a's region to b's
)


def register_custom_attributes(builder: newton.ModelBuilder) -> None:
    """Declare ``topology:*`` attributes (idempotent). Call before adding bonds or crumble sources."""
    if builder.has_custom_attribute(f"{NS}:bond_kind"):
        return
    ca = newton.ModelBuilder.CustomAttribute
    builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="bond", namespace=NS))
    for name, dtype, default, refs in _BOND_COLUMNS:
        builder.add_custom_attribute(ca(name=name, dtype=dtype, frequency=BOND_FREQ, namespace=NS, default=default,
                                        references=refs))
    edge = newton.Model.AttributeFrequency.EDGE
    for name in ("edge_bond0", "edge_bond1"):      # cross-seam bending element dropped when either bond breaks
        builder.add_custom_attribute(ca(name=name, dtype=wp.int32, frequency=edge, namespace=NS, default=-1,
                                        references=BOND_FREQ))
    particle, body = newton.Model.AttributeFrequency.PARTICLE, newton.Model.AttributeFrequency.BODY
    builder.add_custom_attribute(ca(name="crumble_body", dtype=wp.int32, frequency=particle, namespace=NS,
                                    default=-1, references="body"))
    builder.add_custom_attribute(ca(name="crumble_local", dtype=wp.vec3, frequency=particle, namespace=NS,
                                    default=wp.vec3(0.0, 0.0, 0.0)))
    builder.add_custom_attribute(ca(name="crumble_park_eq", dtype=wp.int32, frequency=body, namespace=NS,
                                    default=-1, references="mujoco:equality_constraint"))
    builder.add_custom_attribute(ca(name="crumble_trigger", dtype=wp.int32, frequency=body, namespace=NS,
                                    default=0))


def _finite_or_inf(x: float | None) -> float:
    return INF if x is None or not math.isfinite(float(x)) else float(x)


def _add_row(builder: newton.ModelBuilder, kind: BondKind, **cols: Any) -> int:
    register_custom_attributes(builder)
    vals = {f"{NS}:{name}": default for name, _, default, _ in _BOND_COLUMNS}   # every column, so no sparse rows
    vals.update({f"{NS}:bond_kind": int(kind), f"{NS}:bond_world": None})
    vals.update({f"{NS}:bond_{k}": v for k, v in cols.items()})
    return builder.add_custom_values(**vals)[f"{NS}:bond_kind"]


def add_spring_bond(builder: newton.ModelBuilder, i: int, j: int, ke: float, kd: float = 0.0, *,
                    kind: BondKind = BondKind.SPRING, break_force: float | None = None,
                    break_strain: float | None = None, cut_strain: float | None = None,
                    cut_force: float | None = None, normal: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    measure: float = 0.0) -> int:
    """Add a particle spring (rest length = current distance) and its bond row; returns the bond index."""
    if kind not in (BondKind.SPRING, BondKind.CLOTH_EDGE, BondKind.TET_CUT):
        raise AssemblyError(f"add_spring_bond: kind {kind!r} is not a particle bond")
    s = builder.spring_count
    builder.add_spring(int(i), int(j), float(ke), float(kd), 0)
    rest = float(builder.spring_rest_length[s])
    return _add_row(builder, kind, particle_a=int(i), particle_b=int(j), spring=s,
                    break_force=_finite_or_inf(break_force), break_strain=_finite_or_inf(break_strain),
                    cut_strain=_finite_or_inf(cut_strain), cut_force=_finite_or_inf(cut_force), rest=rest,
                    measure=float(measure), normal=wp.vec3(*normal))


def _body_rest(builder: newton.ModelBuilder, a: int, b: int) -> float:
    pa = np.array(builder.body_q[a][:3]) if a >= 0 else np.zeros(3)
    pb = np.array(builder.body_q[b][:3]) if b >= 0 else np.zeros(3)
    return float(np.linalg.norm(pb - pa))


def add_weld_bond(builder: newton.ModelBuilder, body_a: int, body_b: int = -1, *, break_force: float | None = None,
                  break_strain: float | None = None, solref: tuple[float, float] | None = None,
                  enabled: bool = True) -> int:
    """MuJoCo weld equality between two bodies (−1 = world) as a RIGID_WELD bond; relpose = current pose."""
    from alloy._vendor.newton.solvers import mujoco as nmj

    eq = nmj.add_equality_weld(builder, int(body_a), int(body_b), enabled=enabled, solref=solref)
    return _add_row(builder, BondKind.RIGID_WELD, body_a=int(body_a), body_b=int(body_b), eq=eq,
                    break_force=_finite_or_inf(break_force), break_strain=_finite_or_inf(break_strain),
                    rest=_body_rest(builder, body_a, body_b))


def add_connect_bond(builder: newton.ModelBuilder, body_a: int, body_b: int, anchor: tuple[float, float, float], *,
                     break_force: float | None = None, break_strain: float | None = None,
                     solref: tuple[float, float] | None = None) -> int:
    """MuJoCo connect (ball) equality, anchor in a's frame, as a JOINT bond."""
    from alloy._vendor.newton.solvers import mujoco as nmj

    eq = nmj.add_equality_connect(builder, int(body_a), int(body_b), anchor=wp.vec3(*anchor), solref=solref)
    return _add_row(builder, BondKind.JOINT, body_a=int(body_a), body_b=int(body_b), eq=eq,
                    break_force=_finite_or_inf(break_force), break_strain=_finite_or_inf(break_strain),
                    rest=_body_rest(builder, body_a, body_b))


def tag_cross_seam_edge(builder: newton.ModelBuilder, edge: int, bond0: int, bond1: int) -> None:
    """Mark a bending element that spans a seam: it is zeroed when either bond breaks."""
    register_custom_attributes(builder)
    for name, bond in (("edge_bond0", bond0), ("edge_bond1", bond1)):
        builder.custom_attributes[f"{NS}:{name}"].values = builder.custom_attributes[f"{NS}:{name}"].values or {}
        builder.custom_attributes[f"{NS}:{name}"].values[int(edge)] = int(bond)


def has_bonds(model: newton.Model) -> bool:
    ns = getattr(model, NS, None)
    return ns is not None and getattr(ns, "bond_kind", None) is not None and ns.bond_kind.shape[0] > 0


def bond_column(model: newton.Model, name: str) -> wp.array:
    return getattr(getattr(model, NS), f"bond_{name}")


def bonds_from_model(model: newton.Model) -> BondSet:
    """BondSet over every ``topology:bond`` row: a/b are particles or bodies by kind; all start intact."""
    dev = model.device
    if not has_bonds(model):
        z = lambda dt: wp.zeros(0, dtype=dt, device=dev)  # noqa: E731
        return BondSet(z(wp.int32), z(wp.int32), z(wp.int8), z(wp.int32), z(wp.float32), z(wp.float32),
                       z(wp.float32), z(wp.int8))
    kind = bond_column(model, "kind").numpy()
    is_body = np.isin(kind, (int(BondKind.RIGID_WELD), int(BondKind.JOINT)))
    a = np.where(is_body, bond_column(model, "body_a").numpy(), bond_column(model, "particle_a").numpy())
    b = np.where(is_body, bond_column(model, "body_b").numpy(), bond_column(model, "particle_b").numpy())
    n = len(kind)
    arr = lambda x, dt: wp.array(np.asarray(x), dtype=dt, device=dev)  # noqa: E731
    return BondSet(
        a=arr(a.astype(np.int32), wp.int32), b=arr(b.astype(np.int32), wp.int32),
        kind=arr(kind.astype(np.int8), wp.int8), world=arr(bond_column(model, "world").numpy(), wp.int32),
        break_force=arr(bond_column(model, "break_force").numpy(), wp.float32),
        break_strain=arr(bond_column(model, "break_strain").numpy(), wp.float32),
        damage=wp.zeros(n, dtype=wp.float32, device=dev), intact=arr(np.ones(n, np.int8), wp.int8),
        cut_strain=arr(bond_column(model, "cut_strain").numpy(), wp.float32),
        cut_force=arr(bond_column(model, "cut_force").numpy(), wp.float32))


def bonds_from_cards(built: BuiltScene, lib: CardLibrary) -> BondSet:
    """BondSet for an assembled scene. Thresholds were written from the cards at emit time (presplit, attachments);
    this reads them back and logs the per-kind counts."""
    bonds = bonds_from_model(built.model)
    if bonds.count:
        kinds = np.bincount(bonds.kind.numpy().astype(np.int64), minlength=len(BondKind))
        log.info("bonds_from_cards: %d bonds over %d worlds (%s)", bonds.count, built.world_count,
                 ", ".join(f"{k.name}={kinds[k]}" for k in BondKind if kinds[k]))
    return bonds
