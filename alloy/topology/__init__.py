# SPDX-License-Identifier: Apache-2.0
"""Topology change on a fixed topology (WP10, SPEC §3.8, D19): bond breaking, tear, blade cut, crumble, ledger.

Types live in ``alloy.topology.types`` (WP0). Topology is fixed at assembly. Tear and cut lines are pre-split:
duplicated vertices are joined by breakable twin springs. A tear or a cut is a bond break, so VBD adjacency and
colouring never change.

Assembly (WP2 emitters call these while building the world template; call ``builder.color()`` afterwards):
- ``bonds.register_custom_attributes(builder)``: ``topology:*`` rows travel through ``replicate``.
- ``emit.add_presplit_cloth`` / ``emit.add_presplit_soft``: a pre-split mesh with CLOTH_EDGE / TET_CUT bonds.
- ``bonds.add_weld_bond`` / ``add_connect_bond`` / ``add_spring_bond``: breakable attachments.
- ``crumble.add_crumble_source``: spare MPM particles plus a parking weld for a rigid chunk.
- ``prefracture.prefracture``: offline Voronoi chunks.

Runtime:
- ``bonds.bonds_from_cards(built, lib)`` builds the BondSet.
- ``controller.TopologyController(model, bonds, budget, blades, entries)``: ``pre_step`` does blade gating and cut
  damage; ``apply`` does the break tests, the per-world capped commit, native toggles, crumble and the ledger.
- ``TopologyController.attach_self_contact(vbd)`` (``seam_contact``, SCR-SC5-2): a self-contact SolverVBD ignores a
  seam twin's partner faces only while the twins are joined, so cut faces collide; build it with
  ``particle_rest_shape_contact_exclusion_radius=0``.
- ``TopologyController.cut_front_blade`` / ``blade_half_thickness`` (``cut_front``, SCR-SC5-3): the health monitor's
  predicate for soft contacts between a Blade and a twin whose bond it is cutting (phase 1).
- ``soft_strain.SoftStrain(model).update(state)``: per-world max principal strain of live soft tets, for the health
  flag past ``SOFT_STRAIN_RANGE`` (60 %, ruling 2026-11-06T14:17Z; known gap: soft FEM squeezed by a flat tool).
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {"TopologyController": "controller", "bonds_from_cards": "bonds", "bonds_from_model": "bonds",
            "register_custom_attributes": "bonds", "prefracture": "prefracture"}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazy exports, so importing ``alloy.topology.types`` (e.g. via ``alloy.core.protocols``) stays light."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'alloy.topology' has no attribute {name!r}")
    return getattr(importlib.import_module(f"alloy.topology.{_EXPORTS[name]}"), name)
