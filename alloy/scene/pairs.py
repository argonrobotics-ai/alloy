# SPDX-License-Identifier: Apache-2.0
"""Pair table (D13, D22, SPEC §3.4): host build from cards + the Warp kernel `PairTable.apply` delegates to.

Per rigid contact the kernel writes into Newton's per-contact arrays (sim/contacts.py:331-345, which exist only with
`per_contact_shape_properties=True`):
- `rigid_contact_friction` = μ_pair / μ_mix(shape_a, shape_b): a **scale** on the solver's own mixed μ (D13);
  never 0 (0 = unset): a frictionless pair writes derive.TINY_SCALE.
- `rigid_contact_stiffness` = K (series of the two materials' K, D22; precision shape pairs override).
- `rigid_contact_damping` = 2ζ(e)·√(K·m_eff) (/√(1−imp) for SolverMuJoCo rows, derive.contact_damping), m_eff from
  the bodies' inverse mass and current-pose inverse inertia along the normal (translational only without a state).
Contacts whose shapes carry no material (id −1) are left untouched (0 = the solver's own values).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts, State

from alloy.cards.types import MaterialCard, PairCard
from alloy.scene import derive
from alloy.scene.types import PairTable

log = logging.getLogger("alloy.scene")
DEFAULT_K = 2.0e6           # N/m per material (D22)
MIX_MAX, MIX_MEAN, MIX_GEOMETRIC, MIX_HARMONIC, MIX_NONE = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class MaterialEntry:
    """One pair-table row: a material card, optionally with an object-specific contact stiffness (RIGID_COMPLIANT)."""

    key: str                  # table name: card name, or "<card>@<object>" for a compliant override
    card: MaterialCard
    k_override: float | None = None


@dataclass(frozen=True)
class PairValues:
    mu_s: float
    mu_d: float
    e: float
    lever: float
    k: float
    sourced: bool


def material_k(entry: MaterialEntry) -> float:
    if entry.k_override is not None:
        return entry.k_override
    k = entry.card.surface.contact_stiffness
    return float(k.value) if k is not None else DEFAULT_K


def _combine(rule: str, a: float, b: float) -> float:
    return max(a, b) if rule == "max" else derive.harmonic_mu(a, b)


def pair_values(a: MaterialEntry, b: MaterialEntry, card: PairCard | None, combine: str) -> PairValues:
    """Values for one material pair: the PairCard where sourced, else the scene's combine rule (harmonic, D13)."""
    sa, sb = a.card.surface, b.card.surface
    k = derive.pair_stiffness(material_k(a), material_k(b))
    e = _combine(combine, float(sa.restitution.value), float(sb.restitution.value))
    lever_a = float(sa.torsional_lever.value) if sa.torsional_lever is not None else 0.0
    lever_b = float(sb.torsional_lever.value) if sb.torsional_lever is not None else 0.0
    if card is None:
        return PairValues(_combine(combine, float(sa.mu_static.value), float(sb.mu_static.value)),
                          _combine(combine, float(sa.mu_dynamic.value), float(sb.mu_dynamic.value)),
                          e, max(lever_a, lever_b), k, False)
    mu_s, mu_d = float(card.mu_static.value), float(card.mu_dynamic.value)
    if card.single_mu == "static":
        mu_d = mu_s       # single-μ solvers get μ_static for this pair (object-model §4.5)
    if card.restitution is not None:
        e = float(card.restitution.value)
    lever = float(card.torsional_lever.value) if card.torsional_lever is not None else max(lever_a, lever_b)
    return PairValues(mu_s, mu_d, e, lever, k, True)


@dataclass
class TableBuild:
    """Host result of `build_tables`: [M, M] arrays plus the unsourced pairs for the RouteReport."""

    table: np.ndarray        # [M, M, 4] float64 (mu_s, mu_d, e, lever)
    stiffness: np.ndarray    # [M, M] float64
    unsourced: list[tuple[str, str]]


def pair_card(library, scene_pairs: Sequence[PairCard], a: str, b: str) -> PairCard | None:
    """The pair card of materials ``a`` and ``b``: the scene's own (``Scene.set_friction``), the library's, else the
    library's card of the materials they take their pairs from (``MaterialCard.pairs_from``, alloy.Scene's
    per-object materials), keyed onto ``a`` and ``b``. Example: ``pair_card(lib, (), "abs_for_b1", "abs_for_b2")``
    is the library's abs/abs card for two boxes given their own mass."""
    key = tuple(sorted((a, b)))
    card = next((p for p in scene_pairs if tuple(sorted((p.a, p.b))) == key), None) or library.pair(a, b)
    if card is not None:
        return card
    base = (_pairs_from(library, a), _pairs_from(library, b))
    card = library.pair(*base) if base != (a, b) else None
    return None if card is None else replace(card, a=a, b=b)


def _pairs_from(library, name: str) -> str:
    """The material whose pair cards ``name`` takes (itself unless its card says ``pairs_from``)."""
    has = getattr(library, "has_material", None)
    if has is not None and not has(name):       # a robot file's material: no card
        return name
    return getattr(library.material(name), "pairs_from", None) or name


def build_tables(entries: Sequence[MaterialEntry], pair_lookup, combine: str = "harmonic") -> TableBuild:
    """Fill the symmetric tables; `pair_lookup(a_name, b_name) -> PairCard | None` (scene pairs, then library)."""
    m = len(entries)
    table, stiff, unsourced = np.zeros((m, m, 4)), np.zeros((m, m)), []
    for i in range(m):
        for j in range(i, m):
            card = pair_lookup(entries[i].card.name, entries[j].card.name)
            v = pair_values(entries[i], entries[j], card, combine)
            table[i, j] = table[j, i] = (v.mu_s, v.mu_d, v.e, v.lever)
            stiff[i, j] = stiff[j, i] = v.k
            if not v.sourced:
                unsourced.append(tuple(sorted((entries[i].card.name, entries[j].card.name))))
    unsourced = sorted(set(unsourced))
    for a, b in unsourced:
        log.info("pair_unsourced %s/%s: %s rule", a, b, combine)
    return TableBuild(table, stiff, unsourced)


def make_pair_table(entries: Sequence[MaterialEntry], built: TableBuild, model: newton.Model, *,
                    mu_mix_rule: int, shape_imp: np.ndarray | None = None,
                    e_correction: Sequence[float] = (1.0,)) -> PairTable:
    """Device PairTable for `model` (alloy:shape_material / alloy:particle_material read from the Model)."""
    dev = model.device
    alloy = getattr(model, "alloy", None)
    shape_mat = _attr(alloy, "shape_material", model.shape_count, dev)
    part_mat = _attr(alloy, "particle_material", model.particle_count, dev)
    table = PairTable(materials=tuple(e.key for e in entries),
                      table=wp.array(built.table.astype(np.float32), dtype=wp.vec4, device=dev),
                      stiffness=wp.array(built.stiffness.astype(np.float32), dtype=wp.float32, device=dev),
                      shape_material=shape_mat, particle_material=part_mat,
                      shape_mu_mix=wp.clone(model.shape_material_mu),
                      mu_mix_rule=int(mu_mix_rule),
                      e_correction=wp.array(np.asarray(e_correction, np.float32), dtype=wp.float32, device=dev),
                      model=model,
                      shape_imp=None if shape_imp is None else wp.array(shape_imp.astype(np.float32),
                                                                        dtype=wp.float32, device=dev))
    return table


def _attr(ns, name: str, n: int, dev) -> wp.array:
    """`model.alloy.<name>` or an all −1 array (a frequency with no entities registers no array)."""
    arr = None
    if ns is not None:
        try:
            arr = getattr(ns, name)
        except AttributeError:
            arr = None
    return arr if arr is not None else wp.full(n, -1, dtype=wp.int32, device=dev)


def apply_pair_table(table: PairTable, contacts: Contacts, state: State | None = None, *,
                     precision_external: bool = True) -> None:
    """Write friction scale, stiffness and damping for every rigid contact (capture-safe, no host sync).

    Precision rows (both shapes in one precision pair) get friction only by default: the contact layer writes their
    K and damping in collide() (WP3: sdf persistent weights, hydroelastic area·p/|depth|). The fallback contact path
    (no layer) passes precision_external=False so they get the pair's override K."""
    if contacts.rigid_contact_friction is None:
        raise ValueError("PairTable.apply needs Contacts(per_contact_shape_properties=True) (D13)")
    from alloy.scene.pairs_kernel import launch_pair_kernel

    launch_pair_kernel(table, contacts, state, precision_external)


def pair_matrix(table: PairTable, world: int | None = None) -> np.ndarray:
    """Host copy of the [M, M, 4] table (a world's randomized table when present)."""
    wt = getattr(table, "world_table", None)
    if world is not None and wt is not None:
        return wt.numpy()[world]
    return table.table.numpy()


def unsourced_report(entries: Sequence[MaterialEntry], built: TableBuild) -> Mapping[str, tuple[str, str]]:
    return {f"{a}/{b}": (a, b) for a, b in built.unsourced}


def zeta_np(e: np.ndarray) -> np.ndarray:
    """Vectorised derive.zeta_from_restitution for host checks."""
    return np.vectorize(derive.zeta_from_restitution)(np.clip(e, 0.0, 1.0))


def mu_mix_np(rule: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    names = {MIX_MAX: "max", MIX_MEAN: "mean", MIX_GEOMETRIC: "geometric", MIX_HARMONIC: "harmonic", MIX_NONE: "none"}
    return np.vectorize(lambda x, y: derive.mu_mix(names[rule], float(x), float(y)))(a, b)


__all__ = ["MaterialEntry", "PairValues", "TableBuild", "apply_pair_table", "build_tables", "make_pair_table",
           "material_k", "pair_matrix", "pair_values"]
