# SPDX-License-Identifier: Apache-2.0
"""Online splitting of virtual-node copies by a blade (WP10 item 9 step 3, prototype; copies from ``tet_copies``).

``VNodeCutter.pre_step`` (before collision) runs four passes:
1. Dormant copies follow their hubs (position and velocity).
2. Per whole cuttable tet: split when its world's blade edge comes within ``lead`` of it, i.e. the edge segment crosses
   the tet or passes within ``lead`` of one of its edges (``segment_near_tet``) (rule ``lead``); or once the knife's
   work on it reaches the seam model's bond threshold (rule ``work``, ``vnode_work``); either while the edge leads: the blade's
   velocity along the face's edge-forward direction (-spine) is at least ``ADVANCE`` of its speed. A blade moving
   face-first or backing out cuts nothing; it pushes. (With the plane from the velocity alone, a sideways bully move
   split tets on near-horizontal planes and inverted them.)
   - The split plane holds the edge and is spanned by the edge direction and the edge's velocity (the swept
     triangle), or by the blade face normal when the edge slides along itself.
   - It is oriented so the + side is the blade face normal's side, the same in every tet.
   - Corner sides are signed distances to that plane. A corner within ``SNAP_FRACTION`` of its hub's mean edge length goes to
     the - side (the same threshold in every tet). A hub whose side an earlier split fixed keeps it: a corner
     geometry puts on the other side is placed on the plane, on the hub's side, so later splits agree with the welds
     even when the material has moved.
   - Each mesh edge is crossed once: the first split across it stores its crossing parameter (``edge_t``), and later
     tets on that edge reuse it, so neighbouring tets' cut faces meet on their shared faces. Per-tet planes mapped
     to rest left 2.2 mm kinks where a tet split after its neighbour had pushed a shared hub aside. The volume
     fraction comes from the edge parameters (``fraction_from_params``).
   - A tet left with every corner on one side stays whole.
   - A side under ``fold`` of the tet's volume (``FOLD``, 1 %; coordinator ruling 2026-11-06) folds into the other:
     that fragment takes all of T's material and the empty side's copies get no mass from T. Near-empty splits
     (111 of 316 under 1 % on bully traces) were the fragments that flipped.
3. Splits: T's material to 0, its fragments T+ and T- to f+ and f- of it, plus fragment damping
   ``FRAGMENT_DAMPING`` (k_mu + k_lambda) so a fragment's light virtual copies do not ring.
4. Per hub, in a fixed order over its tets split this substep:
   - mass handed to v+ and v- at the hub's velocity (momentum kept; deterministic order), in the ratio
     (f+ + ``MASS_FLOOR``) : (f- + ``MASS_FLOOR``). A copy shared only by sliver fragments then still has mass of
     order a tenth of a corner; with mass from f alone its nodes ran at 1-6 m/s in oblique strokes.
   - at the hub's first split, its side is fixed and the weld to that side's copy turns on. A later split that puts
     the hub on the other side is counted in ``conflicts`` and changes nothing (one cut per neighbourhood).

``filter_contacts`` (after collision) runs ``vnode_contacts``: copies muted, sided hubs onto their face.
"""


from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.topology.blade_contact import blade_face, blade_spine
from alloy.topology.cut import edge_depth, knife_work
from alloy.topology.tet_copies import TetCopies
from alloy.topology.types import Blade
from alloy.topology.vnode_contacts import CutterBlades, mute_copy_contacts, rewrite_hub_contacts
from alloy.topology.vnode_split import (FOLD, FRAGMENT_DAMPING, SNAP_FRACTION, HubTable, TetTable, apply_split,
                                        decide_split, edge_leads, follow_dormant, hand_over)
from alloy.topology.vnode_state import cutter_state_arrays, cutter_state_worlds
from alloy.topology.vnode_work import (PHASE_CUT, PHASE_CUTTING, PHASE_WELD, RULE_LEAD, RULE_WORK, WorkRule,
                                       bond_stiffness, tet_strain, work_threshold)


@wp.func
def strain_open(ht: HubTable, tt: TetTable, wr: WorkRule, r: int, x: wp.array(dtype=wp.vec3)) -> bool:
    """The work rule's strain gate on hub row r: the largest strain over its cuttable tets >= cut_strain."""
    if wr.cut_strain <= 0.0:
        return True
    strain = float(0.0)
    for e in range(ht.start[r], ht.start[r + 1]):
        row = ht.rows[e]
        if wr.split[row] == 0:
            strain = wp.max(strain, tet_strain(wr, tt.tet[row], x))
        else:
            for k in range(2):                          # a split tet: its fragments that hold material
                t = tt.copy_tet[row, k]
                if wr.materials[t, 0] != 0.0 or wr.materials[t, 1] != 0.0:
                    strain = wp.max(strain, tet_strain(wr, t, x))
    return strain >= wr.cut_strain


@wp.kernel
def hub_work(ht: HubTable, tt: TetTable, bl: CutterBlades, wr: WorkRule, body_q: wp.array(dtype=wp.transform),
             body_qd: wp.array(dtype=wp.spatial_vector), x: wp.array(dtype=wp.vec3)):
    """``vnode_work``: one substep of each hub's cut phase and knife work (one thread per hub, its own row only)."""
    r = wp.tid()
    if wr.cut[r] == PHASE_CUT or ht.start[r] == ht.start[r + 1]:
        return
    j = bl.of_world[tt.world[ht.rows[ht.start[r]]]]
    if j < 0:
        return
    h = ht.hub[r]
    xf = body_q[bl.body[j]]
    e0 = wp.transform_point(xf, bl.edge_a[j]) - x[h]
    e1 = wp.transform_point(xf, bl.edge_b[j]) - x[h]
    ball = bl.edge_radius[j] + wr.reach + wr.radius[h]
    depth = edge_depth(e0, e1, ball)
    if wr.cut[r] == PHASE_WELD and depth > 0.0 and strain_open(ht, tt, wr, r, x):
        wr.cut[r] = PHASE_CUTTING
    elif wr.cut[r] == PHASE_CUTTING and wr.tracked[r] != 0 and depth + wr.depth[r] > 0.0 and edge_leads(
            xf, wp.spatial_top(body_qd[bl.body[j]]), bl.spine[j]):
        wr.work[r] = wr.work[r] + knife_work(wr.edge0[r], wr.edge1[r], e0, e1, ball)
    wr.edge0[r] = e0
    wr.edge1[r] = e1
    wr.tracked[r] = wp.int8(1)
    wr.depth[r] = depth
    threshold = work_threshold(wr, bl.edge_radius[j], bl.softness[j])
    if wr.work[r] >= threshold:
        wr.cut[r] = PHASE_CUT
        wr.released[h] = wp.int8(1)
    wr.bond[r] = bond_stiffness(wr, r, bl.cut_ke[j], threshold)


def world_of(world: np.ndarray) -> np.ndarray:
    """Newton's world ids with global (-1, a model built without ``replicate``) as world 0."""
    return np.maximum(np.asarray(world), 0)


def mean_edge_length(rest: np.ndarray, copies: TetCopies) -> np.ndarray:
    """[H] each hub's mean rest length over its mesh edges in the cuttable region."""
    hub = copies.node_hub[copies.edge]                                  # [E, 2] particles
    length = np.linalg.norm(rest[hub[:, 0]] - rest[hub[:, 1]], axis=1)
    total = np.bincount(copies.edge.reshape(-1), weights=np.repeat(length, 2), minlength=len(copies.node_hub))
    count = np.bincount(copies.edge.reshape(-1), minlength=len(copies.node_hub))
    return total / np.maximum(count, 1)


def hub_csr(hub_row: np.ndarray, hubs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(start [H + 1], rows, corner): per hub row, its cuttable tet rows in ascending order."""
    flat = hub_row.reshape(-1)
    rows = np.repeat(np.arange(hub_row.shape[0]), 4)
    corner = np.tile(np.arange(4), hub_row.shape[0])
    order = np.lexsort((rows, flat))
    start = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=hubs))])
    return start, rows[order], corner[order]


class VNodeCutter:
    """Device tables, split state and the passes for one model's copies (``TetCopies`` with model indices)."""

    def __init__(self, model: newton.Model, copies: TetCopies, blades: Sequence[Blade], lead: float,
                 fragment_damping: float = FRAGMENT_DAMPING, fold: float = FOLD, rule: str = "lead",
                 reach: float | None = None, cut_strain: float = 0.0) -> None:
        """``fragment_damping`` [s]: extra fragment k_damp per unit (k_mu + k_lambda); ``fold``: volume-fraction
        floor of a split side; ``rule``: "lead" or "work" (``vnode_work``; ``reach`` defaults to
        ``cut.default_reach()``, ``cut_strain`` 0 = no gate)."""
        dev = model.device
        self.model, self.copies, self.lead = model, copies, float(lead)
        self.fragment_damping, self.fold = float(fragment_damping), float(fold)
        self.work_rule = self._work_rule(model, len(copies.node_hub), rule, reach, cut_strain)
        i32 = lambda a: wp.array(np.ascontiguousarray(a, np.int32), dtype=wp.int32, device=dev)  # noqa: E731
        f32 = lambda a: wp.array(np.ascontiguousarray(a, np.float32), dtype=wp.float32, device=dev)  # noqa: E731
        tt = TetTable()
        tt.world, tt.tet = i32(world_of(model.particle_world.numpy()[copies.hub[:, 0]])), i32(copies.tet)
        tt.hub, tt.hub_row, tt.copy_tet = i32(copies.hub), i32(copies.hub_row), i32(copies.copy_tet)
        tt.tet_edge, tt.tet_edge_flip = i32(copies.tet_edge), i32(copies.tet_edge_flip)
        tt.material = wp.array(copies.material.astype(np.float32), dtype=wp.vec3, device=dev)
        tt.mass = f32(copies.mass)
        ht = HubTable()
        start, rows, corner = hub_csr(copies.hub_row, len(copies.node_hub))
        ht.hub, ht.node, ht.weld, ht.weld_ke = i32(copies.node_hub), i32(copies.node), i32(copies.weld), \
            f32(copies.weld_ke)
        ht.start, ht.rows, ht.corner = i32(start), i32(rows), i32(corner)
        self.tets, self.hubs, self.blades = tt, ht, self._blade_table(model, blades)
        c, h = copies.count, len(copies.node_hub)
        self.split, self.split_now = wp.zeros(c, dtype=wp.int32, device=dev), wp.zeros(c, dtype=wp.int32, device=dev)
        self.side_mask, self.frac = wp.zeros(c, dtype=wp.int32, device=dev), wp.zeros(c, dtype=wp.float32, device=dev)
        self.front = wp.zeros(c, dtype=wp.int32, device=dev)          # 1: whole and across the cut (``weld_front``)
        self.edge_t = wp.full(len(copies.edge), -1.0, dtype=wp.float32, device=dev)   # crossing, -1 = none
        self.hub_eps = f32(SNAP_FRACTION * mean_edge_length(model.particle_q.numpy(), copies))
        self.hub_side = wp.full(h, -1, dtype=wp.int32, device=dev)
        self.conflicts = wp.zeros(h, dtype=wp.int32, device=dev)
        self.on_plane = wp.zeros(h, dtype=wp.int32, device=dev)      # 1: a split put the hub on the cut (snapped)
        self.particle_side = wp.full(model.particle_count, -1, dtype=wp.int8, device=dev)
        mask = np.zeros(model.particle_count, np.int8)
        mask[copies.node.reshape(-1)] = 1
        self.copy_mask = wp.array(mask, dtype=wp.int8, device=dev)  # 1 = a copy: leave out of particle-count and
        # throughput denominators (coordinator ruling 3, 2026-11-06); copies stay in the health checks
        self.added = wp.zeros(len(copies.node_hub), dtype=wp.float32, device=dev)
        self.work_rule.materials, self.work_rule.split = model.tet_materials, self.split
        self._world_maps: Mapping[str, wp.array] | None = None

    @staticmethod
    def _blade_table(model: newton.Model, blades: Sequence[Blade]) -> CutterBlades:
        dev = model.device
        of_world = np.full(max(int(model.world_count), 1), -1, np.int32)
        body_world = world_of(model.body_world.numpy())
        for j, b in enumerate(blades):
            if of_world[body_world[b.body]] >= 0:
                raise ValueError("cut_model: copies takes one blade per world, and blades "
                                 f"{of_world[body_world[b.body]]} and {j} are both in world {body_world[b.body]}. "
                                 "Keep one knife (an object with a "
                                 "`blade:` block) in each world, or give the objects a second knife must cut "
                                 "`cut_model: seams` (docs/guide/cutting.md)")
            of_world[body_world[b.body]] = j
        bt = CutterBlades()
        bt.of_world = wp.array(of_world, dtype=wp.int32, device=dev)
        bt.body = wp.array(np.array([b.body for b in blades], np.int32), dtype=wp.int32, device=dev)
        bt.edge_a = wp.array(np.array([b.edge_a for b in blades], np.float32), dtype=wp.vec3, device=dev)
        bt.edge_b = wp.array(np.array([b.edge_b for b in blades], np.float32), dtype=wp.vec3, device=dev)
        faces = [blade_face(model, b) for b in blades]
        bt.axis = wp.array(np.array([f.axis for f in faces], np.float32), dtype=wp.vec3, device=dev)
        bt.center = wp.array(np.array([f.center for f in faces], np.float32), dtype=wp.vec3, device=dev)
        bt.half = wp.array(np.array([f.half for f in faces], np.float32), dtype=wp.float32, device=dev)
        bt.edge_radius = wp.array(np.array([b.edge_radius for b in blades], np.float32), dtype=wp.float32, device=dev)
        spines = np.array([blade_spine(f, b) for f, b in zip(faces, blades, strict=True)], np.float32)
        bt.spine = wp.array(spines.reshape(-1, 3), dtype=wp.vec3, device=dev)
        bt.softness = wp.array(np.array([b.softness for b in blades], np.float32), dtype=wp.float32, device=dev)
        bt.cut_ke = wp.array(np.array([b.cut_spring_ke for b in blades], np.float32), dtype=wp.float32, device=dev)
        shape_blade = np.full(model.shape_count, -1, np.int32)
        shape_body = model.shape_body.numpy()
        for j, b in enumerate(blades):
            shape_blade[shape_body == b.body] = j
        bt.shape_blade = wp.array(shape_blade, dtype=wp.int32, device=dev)
        return bt

    @staticmethod
    def _work_rule(model: newton.Model, hubs: int, rule: str, reach: float | None, cut_strain: float) -> WorkRule:
        from alloy.topology.cut import resolve_reach

        if rule not in ("lead", "work"):
            raise ValueError(f"VNodeCutter rule must be 'lead' or 'work', got {rule!r}")
        wr = WorkRule()
        wr.rule = RULE_LEAD if rule == "lead" else RULE_WORK
        wr.reach, wr.cut_strain = resolve_reach(reach), float(cut_strain)
        wr.radius, wr.poses, wr.tet_indices = model.particle_radius, model.tet_poses, model.tet_indices
        wr.work = wp.zeros(hubs, dtype=wp.float32, device=model.device)
        wr.depth = wp.zeros(hubs, dtype=wp.float32, device=model.device)
        wr.edge0 = wp.zeros(hubs, dtype=wp.vec3, device=model.device)
        wr.edge1 = wp.zeros(hubs, dtype=wp.vec3, device=model.device)
        wr.tracked = wp.zeros(hubs, dtype=wp.int8, device=model.device)
        wr.cut = wp.zeros(hubs, dtype=wp.int32, device=model.device)
        wr.bond = wp.zeros(hubs, dtype=wp.float32, device=model.device)
        wr.released = wp.full(model.particle_count, 1 if rule == "lead" else 0, dtype=wp.int8, device=model.device)
        return wr

    def added_mass(self) -> np.ndarray:
        """[W] mass [kg] the copies of kinematic hubs took (not taken from any particle): the TOP-1 ledger row. Total
        particle mass = initial + this."""
        world = world_of(self.model.particle_world.numpy()[self.copies.node_hub])
        return np.bincount(world, weights=self.added.numpy().astype(np.float64),
                           minlength=max(int(self.model.world_count), 1))

    def state_arrays(self) -> Mapping[str, wp.array]:
        """Every array the next step reads that the cutter changes, model arrays included (``vnode_state``)."""
        return cutter_state_arrays(self)

    def state_worlds(self) -> Mapping[str, wp.array]:
        """Item -> world maps for ``state_arrays`` (WorldSnapshot ``extra_world``); static, built on the first call.
        The names match Simulation's topology slot (``state_arrays`` / ``state_worlds``)."""
        if self._world_maps is None:
            self._world_maps = cutter_state_worlds(self)
        return self._world_maps

    def reset(self) -> None:
        """Every tet whole and every copy dormant again (the model arrays are the caller's to restore)."""
        for a in (self.split, self.split_now, self.side_mask, self.frac, self.conflicts, self.added,
                  self.work_rule.work, self.work_rule.depth, self.work_rule.edge0, self.work_rule.edge1,
                  self.work_rule.tracked, self.work_rule.cut, self.work_rule.bond, self.front, self.on_plane):
            a.zero_()
        self.work_rule.released.fill_(1 if self.work_rule.rule == RULE_LEAD else 0)
        self.edge_t.fill_(-1.0)
        self.hub_side.fill_(-1)
        self.particle_side.fill_(-1)

    def pre_step(self, state: newton.State) -> None:
        """The passes (with the ``work`` rule, the hubs' knife work before the splits)."""
        m, dev, c = self.model, self.model.device, self.copies.count
        wp.launch(follow_dormant, dim=len(self.copies.node_hub), device=dev,
                  inputs=[self.hubs, self.hub_side, m.particle_mass, state.particle_q, state.particle_qd])
        if self.work_rule.rule == RULE_WORK:
            wp.launch(hub_work, dim=len(self.copies.node_hub), device=dev,
                      inputs=[self.hubs, self.tets, self.blades, self.work_rule, state.body_q, state.body_qd,
                              state.particle_q])
        wp.launch(decide_split, dim=c, device=dev,
                  inputs=[self.tets, self.blades, state.body_q, state.body_qd, state.particle_q, self.lead,
                          self.hub_side, self.hub_eps, self.split, self.split_now, self.side_mask, self.frac,
                          self.edge_t, self.fold, self.front, self.work_rule, self.on_plane])
        wp.launch(apply_split, dim=c, device=dev,
                  inputs=[self.tets, self.split_now, self.frac, self.fragment_damping, self.split, m.tet_materials])
        wp.launch(hand_over, dim=len(self.copies.node_hub), device=dev,
                  inputs=[self.hubs, self.tets, self.split_now, self.frac, self.side_mask, m.particle_mass,
                          m.particle_inv_mass, m.particle_flags, state.particle_qd, m.spring_stiffness, self.hub_side,
                          self.particle_side, self.conflicts, self.added, self.split, self.front, self.work_rule,
                          self.on_plane])

    def filter_contacts(self, state: newton.State, contacts: newton.Contacts | None, faces: bool = True) -> None:
        """Copies a hub stands for are muted; with ``faces``, sided hubs and free far copies meet the blade on their
        faces (``vnode_contacts``). ``faces=False`` keeps stock contacts, as the seam TOP-2 twin (no BladeContact)."""
        if contacts is None:
            return
        dev, n = self.model.device, contacts.soft_contact_max
        wp.launch(mute_copy_contacts, dim=n, device=dev,
                  inputs=[self.copy_mask, self.particle_side, state.body_q, self.model.shape_body, contacts.soft_contact_count, n,
                          contacts.soft_contact_indices, contacts.soft_contact_shape, contacts.soft_contact_body_pos,
                          contacts.soft_contact_normal])
        if not faces:
            return
        wp.launch(rewrite_hub_contacts, dim=n, device=dev,
                  inputs=[self.blades, self.particle_side, self.work_rule.released, state.body_q, state.particle_q, self.model.particle_radius,
                          contacts.soft_contact_count, n, contacts.soft_contact_indices, contacts.soft_contact_shape, contacts.soft_contact_body_pos,
                          contacts.soft_contact_body_vel, contacts.soft_contact_normal,
                          int(self.work_rule.rule == RULE_WORK)])
