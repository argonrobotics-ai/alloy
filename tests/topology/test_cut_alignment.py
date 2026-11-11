# SPDX-License-Identifier: Apache-2.0
"""The cut damage's seam-alignment gate (cut.py, ``TopologyController.Config.cut_alignment``; gap #9, 2026-11-09).

A blade parts material in its own plane. A seam its edge only crosses is not parted by it: before the gate, a blade
in the x seam of a block softened and broke the y seam along the line where its edge crossed it (SB-7's
``cut_block_stroke``), and a blade pressed face-on onto flat cut rounds split them into layers (JOBRK finding 1).

The rig is test_cut_damage's: a 20 x 15 x 15 mm pre-split block and one kinematic blade (face normal = body x, edge
along body y), no collision pipeline, so the blade acts through the cut damage model only.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.testing.topology import blade_pose, presplit_block
from alloy.topology import cut
from alloy.topology.blade_contact import BladeContact, BladeContactConfig, blade_axis
from alloy.topology.bonds import bond_column, bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import BondKind, TopologyBudget

ALIGN = cut.SEAM_ALIGNMENT                      # the gate's value: the blade contact's min_alignment
TOP_Z, EDGE_BELOW_BODY, SPEED, DT = 0.015, 0.02, 0.05, 1.0e-3
CENTRE = np.array([0.010, 0.0075])              # [m] the blade's x (the x seam) and the block's y centre
CROSSING = np.array([0.010, 0.010])             # [m] where the x and y seams meet


@dataclass
class Rig:
    scene: object
    ctl: TopologyController
    solver: newton.solvers.SolverVBD
    states: list
    cut_bonds: np.ndarray         # TET_CUT bond ids
    springs: np.ndarray           # their twin spring ids
    normal: np.ndarray            # [n, 3] their rest seam normals


@dataclass
class End:
    ke: np.ndarray                # [n] twin stiffness after the last pre_step
    ke0: np.ndarray               # [n] before the first
    phase: np.ndarray             # [n] the highest phase seen (0 weld, 1 cutting, 2 broken)
    damage: np.ndarray            # [n] the largest damage seen
    touched: np.ndarray           # [n] bool: in blade contact at some pre_step
    seam: np.ndarray              # [n, 3] current seam normals at the end (the blade contact's)


def _rig(device: str, alignment: float | None, plan: SplitPlan | None = None, bare: bool = False) -> Rig:
    """``bare``: the model's bonds carry no seam normal (a zero ``bond_normal``)."""
    sc = presplit_block(device, cells=(4, 3, 3), plan=plan, gravity=0.0, k_mu=5e4, k_lambda=5e4)
    m = sc.model
    if bare:
        bond_column(m, "normal").zero_()
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=64), sc.blades,
                             config=TopologyController.Config(cut_alignment=alignment))
    ids = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT))
    return Rig(sc, ctl, newton.solvers.SolverVBD(m, iterations=4), [m.state(), m.state()], ids,
               m.topology.bond_spring.numpy()[ids], bond_column(m, "normal").numpy()[ids])


def _descend(rig: Rig, yaw: float = 0.0, over: np.ndarray = CENTRE, travel: float = 0.02,
             saw: float = 2.0e-3) -> End:
    """The blade's edge from 2 mm above the block down ``travel`` at SPEED, sawing ``saw`` along itself; its face
    normal is (cos yaw, sin yaw, 0) and its edge passes over ``over``."""
    m, ctl = rig.scene.model, rig.ctl
    s0, s1 = rig.states
    c, n = m.control(), int(round(travel / (SPEED * DT)))
    edge = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    ke0 = m.spring_stiffness.numpy()[rig.springs]
    phase, damage, touched = np.zeros(len(ke0), np.int8), np.zeros(len(ke0)), np.zeros(len(ke0), bool)
    for k in range(n):
        along = saw * np.sin(2.0 * np.pi * DT * k / 0.1)
        p = np.array([*(over + along * edge[:2]), TOP_Z + 2.0e-3 - SPEED * DT * k + EDGE_BELOW_BODY])
        bq = s0.body_q.numpy()
        bq[rig.scene.blade_bodies] = blade_pose(p, edge, np.array([0.0, 0.0, -1.0]))
        s0.body_q.assign(bq)
        ctl.pre_step(s0, None, DT)
        phase = np.maximum(phase, ctl.phase.numpy()[rig.cut_bonds])
        damage = np.maximum(damage, ctl.bonds.damage.numpy()[rig.cut_bonds])
        touched |= ctl.contact.numpy()[rig.cut_bonds] >= 0
        ke = m.spring_stiffness.numpy()[rig.springs]
        rig.solver.step(s0, s1, c, None, DT)
        ctl.apply(s1)
        s0, s1 = s1, s0
    rig.states = [s0, s1]
    seam = BladeContact(m, ctl.bonds, rig.scene.blades).seam_normals(s0).numpy()[rig.cut_bonds]
    return End(ke, ke0, phase, damage, touched, seam)


def test_a_blade_in_the_x_seam_cuts_it_and_leaves_the_y_seam_welded(device):
    """SB-7's stroke: the blade lies in the x = 10 mm seam and its edge crosses the y = 10 mm seam. With the gate the
    y seam's bonds never see the blade; the x seam is cut exactly as without it. Without the gate (the rule before,
    this law's mutant) the edge softens and breaks y-seam bonds along the line where it crosses them."""
    gated, free = (_descend(_rig(device, a)) for a in (ALIGN, None))
    rig = _rig(device, ALIGN)
    x_seam = np.abs(rig.normal[:, 0]) > 0.5
    assert x_seam.sum() >= 10 and (~x_seam).sum() >= 10
    assert not gated.touched[~x_seam].any() and (gated.phase[~x_seam] == cut.PHASE_WELD).all()
    assert (gated.damage[~x_seam] == 0.0).all()
    np.testing.assert_array_equal(gated.ke[~x_seam], gated.ke0[~x_seam])
    cut_through = (gated.phase >= cut.PHASE_CUTTING) & (gated.ke == 0.0)
    assert cut_through[x_seam].sum() >= 10
    np.testing.assert_array_equal(gated.ke[x_seam], free.ke[x_seam])
    np.testing.assert_array_equal(gated.damage[x_seam], free.damage[x_seam])
    hurt = (free.damage > 0.0) & ~x_seam
    print(f"no gate: {int(hurt.sum())} y-seam bonds damaged, {int(((free.ke == 0.0) & ~x_seam).sum())} cut through")
    assert hurt.sum() >= 3


def test_every_bond_the_cutter_damages_is_one_the_blade_contact_opens(device):
    """The gate and ``BladeContact``'s active-twin rule read the same seam normal and the same threshold, so a
    damaged bond's twins are always put on the blade faces (never left to the stock contact in the kerf)."""
    rig = _rig(device, ALIGN)
    end = _descend(rig, yaw=np.deg2rad(20.0))
    face = np.array([np.cos(np.deg2rad(20.0)), np.sin(np.deg2rad(20.0)), 0.0])
    np.testing.assert_allclose(blade_axis(rig.scene.model, rig.scene.blades[0]), [1.0, 0.0, 0.0], atol=1e-12)
    aligned = np.abs(end.seam @ face) >= ALIGN
    assert (end.damage > 0.0).sum() >= 10 and aligned[end.damage > 0.0].all() and not end.touched[~aligned].any()


@pytest.mark.parametrize(("yaw_deg", "x_cut", "y_cut"), [(40.0, True, True), (70.0, False, True)])
def test_the_threshold_is_on_the_angle_between_the_seam_and_the_blade_face(device, yaw_deg, x_cut, y_cut):
    """The edge passes over the line where the seams meet. |n . face| is cos(yaw) for the x seam and sin(yaw) for
    the y seam: at 40 deg both pass 0.5 (0.77, 0.64); at 70 deg the x seam (0.34) does not."""
    rig = _rig(device, ALIGN)
    end = _descend(rig, yaw=np.deg2rad(yaw_deg), over=CROSSING)
    x_seam = np.abs(rig.normal[:, 0]) > 0.5
    assert bool((end.damage[x_seam] > 0.0).any()) is x_cut and bool((end.damage[~x_seam] > 0.0).any()) is y_cut


def test_a_blade_pressed_through_a_flat_seam_does_not_split_it(device):
    """JOBRK finding 1: a seam lying flat under a vertical blade (seam normal z, face normal x; rounds cut earlier,
    lying on the board). The edge passes through its plane and no bond is touched; without the gate it cuts them."""
    flat = SplitPlan(spacing=0.01, axes=(2,))
    gated, free = (_descend(_rig(device, a, flat)) for a in (ALIGN, None))
    assert len(gated.ke) >= 10 and not gated.touched.any() and (gated.damage == 0.0).all()
    np.testing.assert_array_equal(gated.ke, gated.ke0)
    assert ((free.ke == 0.0) & (free.phase >= cut.PHASE_CUTTING)).sum() >= 3


def test_alignment_follows_the_piece_not_its_rest_pose(device):
    """The block turned a quarter turn about z (through x = y = 10 mm): its rest x seam now lies across the blade and
    its rest y seam in the blade's plane. The gate reads the current normal F^-T n0, so the rest y seam is cut."""
    rig = _rig(device, ALIGN)
    for s in rig.states:
        q = s.particle_q.numpy()
        x, y = q[:, 0].copy(), q[:, 1].copy()
        q[:, 0], q[:, 1] = 0.010 - (y - 0.010), 0.010 + (x - 0.010)
        s.particle_q.assign(q)
    end = _descend(rig)
    x_seam = np.abs(rig.normal[:, 0]) > 0.5
    assert (end.damage[~x_seam] > 0.0).sum() >= 10 and not end.touched[x_seam].any()


@wp.kernel
def _pick(mid: wp.array[wp.vec3], n: wp.array[wp.vec3], blades: cut.CutBlades, reach: float, alignment: float,
          out: wp.array[wp.int32]):
    i = wp.tid()
    out[i] = cut.nearest_cutting_blade(mid[i], reach, 0, 2, blades, n[i], alignment)


def test_a_nearer_blade_that_only_crosses_the_seam_does_not_hide_an_aligned_one(device):
    """Two blades in reach of a bond at the origin with seam normal x: blade 0 passes 0.2 mm away with its face
    normal along y (it crosses the seam), blade 1 0.5 mm away with its face normal along x. The gate picks blade 1;
    with no gate, or for a bond without a normal, the nearest blade (0); with blade 1 out of reach, none."""
    v3 = lambda rows: wp.array(np.array(rows, np.float32), dtype=wp.vec3, device=device)  # noqa: E731
    f32 = lambda rows: wp.array(np.array(rows, np.float32), dtype=wp.float32, device=device)  # noqa: E731
    b = cut.CutBlades()
    b.world_start = wp.array(np.array([0, 2], np.int32), dtype=wp.int32, device=device)
    b.order = wp.array(np.array([0, 1], np.int32), dtype=wp.int32, device=device)
    b.p0, b.p1 = v3([[-0.05, 2e-4, 0.0], [5e-4, -0.05, 0.0]]), v3([[0.05, 2e-4, 0.0], [5e-4, 0.05, 0.0]])
    b.radius, b.softness, b.cut_ke = f32([1e-4, 1e-4]), f32([3.0, 3.0]), f32([500.0, 500.0])
    b.face = v3([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    cases = [((1.0, 0.0, 0.0), 1e-3, ALIGN, 1), ((1.0, 0.0, 0.0), 1e-3, 0.0, 0), ((0.0, 0.0, 0.0), 1e-3, ALIGN, 0),
             ((1.0, 0.0, 0.0), 3e-4, ALIGN, -1)]
    for n, reach, alignment, want in cases:
        out = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_pick, dim=1, device=device, inputs=[v3([[0.0, 0.0, 0.0]]), v3([n]), b, reach, alignment],
                  outputs=[out])
        assert int(out.numpy()[0]) == want, (n, reach, alignment)


def test_bonds_without_a_seam_normal_are_cut_ungated_and_counted(device, caplog):
    """A zero ``bond_normal`` gives the gate nothing to judge: those bonds follow the rule before, and the build
    says how many they are."""
    with caplog.at_level(logging.WARNING, logger="alloy.topology"):
        rig = _rig(device, ALIGN, bare=True)
    assert any(f"{len(rig.cut_bonds)} of {len(rig.cut_bonds)} TET_CUT bonds have no seam normal" in r.getMessage()
               for r in caplog.records)
    end, ref = _descend(rig), _descend(_rig(device, None))
    np.testing.assert_array_equal(end.ke, ref.ke)
    assert (end.damage > 0.0).sum() >= 10


def test_a_cutter_gated_under_the_contact_threshold_is_refused(device):
    """The blade contact opens seams from ``min_alignment`` up; a cutter gated lower would cut seams it leaves on the
    stock contact, so a ``BladeContact`` told of such a cutter raises."""
    sc = presplit_block(device, cells=(2, 2, 2))
    bonds = bonds_from_model(sc.model)
    for ok in (None, ALIGN, 0.8):
        BladeContact(sc.model, bonds, sc.blades, cut_alignment=ok)
    with pytest.raises(AssemblyError):
        BladeContact(sc.model, bonds, sc.blades, cut_alignment=0.3)
    BladeContact(sc.model, bonds, sc.blades, cut_alignment=0.3, config=BladeContactConfig(min_alignment=0.3))


def test_alignment_parameter_checks(device):
    assert cut.resolve_alignment(None) == 0.0 and cut.resolve_alignment(0.5) == 0.5
    assert BladeContactConfig().min_alignment == cut.SEAM_ALIGNMENT
    sc = presplit_block(device, cells=(2, 2, 2))
    for bad in (0.0, -0.5, 1.5, float("nan")):
        with pytest.raises(AssemblyError):
            TopologyController(sc.model, bonds_from_model(sc.model), TopologyBudget(), sc.blades,
                               config=TopologyController.Config(cut_alignment=bad))
