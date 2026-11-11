# SPDX-License-Identifier: Apache-2.0
"""Card defaults: the values ``resolve`` fills in and the default leaves it reports.

A leaf is a *default leaf* when its ``Q.source`` is ``"default"`` after filling, i.e. nobody sourced it. Those that
change routing or contact are warned about at validation and counted against coverage (SPEC §3.2, SB-9).
"""

from __future__ import annotations

import dataclasses
from fnmatch import fnmatchcase

from alloy.cards import types as ct
from alloy.cards.paths import CardView, view_leaves
from alloy.core.quantity import Q

DEFAULT_CONTACT_STIFFNESS = 2.0e6     # N/m per material, series 1e6 for a pair (SPEC §3.2 Surface, §4.2)

ROUTING_LEAF_PATTERNS: tuple[str, ...] = (
    "loads.*", "features.*", "precision_features.*.clearance",
    "material.density", "material.elastic.*", "material.surface.*", "material.plastic.*", "material.fluid.viscosity",
    "part_materials.*.density", "part_materials.*.elastic.*", "part_materials.*.surface.*",
    "part_materials.*.plastic.*", "part_materials.*.fluid.viscosity",
)


def is_routing_leaf(path: str) -> bool:
    """True if a leaf at this object-relative path feeds the router or the contact model."""
    return any(fnmatchcase(path, pat) for pat in ROUTING_LEAF_PATTERNS)


def fill_material(m: ct.MaterialCard) -> ct.MaterialCard:
    """The material with numeric defaults filled: ``surface.contact_stiffness`` (None → 2e6 N/m, source default).

    Derived values stay None for their owners: torsional lever (Hertz, ``alloy.scene.derive``), damping ratio
    (from the pair's restitution, ``PairTable``)."""
    if m.surface.contact_stiffness is not None:
        return m
    surface = dataclasses.replace(m.surface, contact_stiffness=Q(DEFAULT_CONTACT_STIFFNESS, "default"))
    return dataclasses.replace(m, surface=surface)


def fill_view(view: CardView) -> CardView:
    parts = {k: fill_material(v) for k, v in view.parts.items()}
    return dataclasses.replace(view, material=fill_material(view.material), parts=parts)


def default_leaf_paths(view: CardView) -> tuple[str, ...]:
    """Object-relative paths of every leaf still at source ``default`` once defaults are filled."""
    return tuple(path for path, q in view_leaves(fill_view(view)) if q.source == "default")
