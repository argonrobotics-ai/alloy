# SPDX-License-Identifier: Apache-2.0
"""Knife-work split rule for virtual-node copies (WP10 item 9 step 3; coordinator ruling 2, 2026-11-06).

The ``lead`` rule splits a tet as the edge comes near it, so the knife never presses and feels no cutting force.
The ``work`` rule cuts hub by hub, as the seam model cuts twin by twin (``cut.py``; RSS 2021 / IJRR 2023 Eq. 2):
- per hub (``vnode_cut.hub_work``, one thread per hub), the seam bond's phases: the hub starts welded
  (``PHASE_WELD``) and starts cutting once the edge reaches its contact ball (edge_radius + reach + particle radius)
  with the strain gate open: the largest tensile or compressive strain over its live tets (whole, or the fragments
  holding material) >= ``cut_strain`` (``Failure.cut_min_strain``; 0 = no gate). While cutting (and the edge leads)
  it gathers the knife work of ``cut.knife_work``: the edge's depth in the ball integrated over the edge's travel
  *relative to the hub* (a hub pushed along with the edge gathers little; with the edge's own travel the top layer
  was cut 0.9 mm early);
- the hub is cut (``PHASE_CUT``) once its work reaches grip^2 / softness, with grip = edge_radius + reach. That is
  the work that takes a twin spring from ``cut_spring_ke`` to 0 in ``cut.cut_damage_stiffness``;
- a tet near the edge splits once the knife has cut one of its corners near the plane (``vnode_split
  .near_corner_cut``); until then the hubs press the knife through their stock contacts with all their material
  (TOP-2's force). A hub being cut sits on the plane, on the side the edge's wedge pushed it
  (``vnode_split.cut_corner_side``);
- a split hub's far-side copy stays welded to it until the hub is cut (``vnode_split.weld_copies``) with the seam's
  cutting bond (``bond_stiffness``), and both press the knife, as a seam twin pair does.
Variants that failed (TOP-2 twin, scr/WP10.md 2026-11-06):
- per-tet work split tets spanning two layers early, so their lower hubs met the face and the force fell to ~0;
- splitting by the lead rule left uncut hubs stiffly welded to their far copies: VBD moves a stiffly welded pair
  only ~K_other / K_weld per sweep, so the knife met a rock (0.9-2.5 N at first touch);
- holding a tet's split until every reached corner was cut left cut hubs tied to the far side's material through
  whole tets: released hubs drifted across the knife and were kicked back at 2 m/s.
"""

from __future__ import annotations

import warp as wp

from alloy.topology.geometry import stretch_to_strain, tet_stretch_range

RULE_LEAD, RULE_WORK = 0, 1
PHASE_WELD, PHASE_CUTTING, PHASE_CUT = 0, 1, 2      # a hub's cut phase (``cut.py``'s bond phases, plus "cut")


@wp.struct
class WorkRule:
    rule: int                                  # RULE_LEAD or RULE_WORK
    reach: float                               # [m] contact reach beyond the edge radius (cut.default_reach)
    cut_strain: float                          # strain gate; 0 = always open
    radius: wp.array(dtype=wp.float32)         # particle radius
    poses: wp.array(dtype=wp.mat33)            # tet Dm^-1
    tet_indices: wp.array2d(dtype=wp.int32)
    work: wp.array(dtype=wp.float32)           # [H] accumulated knife work [m^2]
    depth: wp.array(dtype=wp.float32)          # [H] edge depth in the hub's contact ball, last substep [m]
    edge0: wp.array(dtype=wp.vec3)             # [H] edge end points relative to the hub, last substep
    edge1: wp.array(dtype=wp.vec3)
    tracked: wp.array(dtype=wp.int8)           # [H] 1 once edge0 / edge1 hold a substep
    cut: wp.array(dtype=wp.int32)              # [H] PHASE_*
    materials: wp.array2d(dtype=wp.float32)    # model tet_materials (the strain gate reads live tets only)
    split: wp.array(dtype=wp.int32)            # [C] the cutter's split marks
    bond: wp.array(dtype=wp.float32)           # [H] far-copy weld stiffness while uncut [N/m] (``bond_stiffness``)
    released: wp.array(dtype=wp.int8)          # [P] 1: a sided hub may meet the blade face (lead: all; work: cut)


@wp.func
def tet_strain(wr: WorkRule, t: int, x: wp.array(dtype=wp.vec3)) -> float:
    st = stretch_to_strain(tet_stretch_range(x[wr.tet_indices[t, 0]], x[wr.tet_indices[t, 1]],
                                             x[wr.tet_indices[t, 2]], x[wr.tet_indices[t, 3]], wr.poses[t]))
    return wp.max(st[0], st[1])


@wp.func
def work_threshold(wr: WorkRule, edge_radius: float, softness: float) -> float:
    """[m^2] grip^2 / softness: the work that takes a twin spring from cut_spring_ke to 0 (``cut.py``)."""
    grip = edge_radius + wr.reach
    return grip * grip / wp.max(softness, 1.0e-12)


@wp.func
def bond_stiffness(wr: WorkRule, r: int, cut_ke: float, threshold: float) -> float:
    """[N/m] Hub row r's far-copy weld under the ``work`` rule: ``cut.py``'s cutting twin spring, cut_ke (1 - work /
    threshold) (``cut.cut_damage_stiffness``), 0 once cut. Not the stiff phase-0 weld: a stiffly welded pair moves
    only ~K_other / K_weld per VBD sweep, so a pressed welded hub held the knife like a rock (1.7-3 N spikes as the
    next layer was reached; scr/WP10.md 2026-11-06T15:02Z)."""
    if wr.cut[r] == PHASE_CUT:
        return 0.0
    return wp.max(cut_ke * (1.0 - wr.work[r] / threshold), 0.0)
