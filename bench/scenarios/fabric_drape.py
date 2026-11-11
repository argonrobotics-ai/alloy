# SPDX-License-Identifier: Apache-2.0
"""D03M measured-fabric Cusick drape (SCR-WP2-14): published KES-F fabrics on Alloy's production route.

D03 (deform_cloth.py) drapes one synthetic cotton-like card on raw solvers. D03M drapes real fabrics: each pack entry
(``bench/scenarios/packs/fabric_drape_*.json``, schema ``alloy.bench.fabric_drape/1``) carries a paper's KES-F inputs
and the drape coefficient it measured on the same fabric. The card mapping, the rig and its scoring live in Alloy
(``alloy.scene.twins.fabric_rig``, ``fabric_score``, ``fabric_run``), shared with the R6 law's production-route twin.
This module adds the bench's mesh choice (``MESHES``: the R6 mesh or a square-grid disc) and the pack folder.
Runner: ``python -m bench.scenarios.fabric_drape_run`` (rows in bench/results/deform/D03M_*.jsonl).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alloy.scene.twins import fabric_rig as _FR
from alloy.scene.twins.fabric_rig import (ARMS, BAR_DC, FRAME_S, G_RAMP_S, H_MIN, HEADLINE, JITTER, NU, RIG_SPACING,
                                          SCORE_S, Rig, add_rig, bending, card_name, check_mesh, make_scene,
                                          material_card, membrane_Eh, min_edge, rigs_for, tensile, thickness,
                                          write_cards, youngs_L)

PACKS = Path(__file__).resolve().parent / "packs"
MESHES = ("r6", "grid")


def specimen_mesh(radius: float, edge: float, seed: int, kind: str = "r6") -> tuple[np.ndarray, np.ndarray]:
    """The specimen disc: vertices [V, 3] at z = 0, triangles [T, 3]. "r6": the R6 twin's jittered-hex mesh
    (``fabric_rig.specimen_mesh``); "grid": a square-grid disc (bench D03's ``deform_geom.disc_mesh``) with its outer
    nodes on the circle."""
    if kind == "grid":
        return _grid_disc(radius, edge)
    return _FR.specimen_mesh(radius, edge, seed)


def _grid_disc(radius: float, edge: float) -> tuple[np.ndarray, np.ndarray]:
    from bench.scenarios import deform_geom as GEO

    v, t = GEO.disc_mesh(radius, edge)
    v = np.column_stack([np.asarray(v, np.float64)[:, :2], np.zeros(len(v))])
    r = np.hypot(v[:, 0], v[:, 1])
    rim = r > radius - 0.75 * edge
    v[rim, :2] *= (radius / r[rim])[:, None]
    return v, np.asarray(t, np.int64)


__all__ = ["ARMS", "BAR_DC", "FRAME_S", "G_RAMP_S", "HEADLINE", "H_MIN", "JITTER", "MESHES", "NU", "PACKS",
           "RIG_SPACING", "Rig", "SCORE_S", "add_rig", "bending", "card_name", "check_mesh", "make_scene",
           "material_card", "membrane_Eh", "min_edge", "rigs_for", "specimen_mesh", "tensile", "thickness",
           "write_cards", "youngs_L"]
