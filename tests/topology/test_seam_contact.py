# SPDX-License-Identifier: Apache-2.0
"""SCR-SC5-2: VBD self-contact ignores a seam twin's partner faces only while the twins are joined.

- ``attach_self_contact`` adds, to SolverVBD's own detector lists, each seam vertex -> the triangles around its
  twins and each edge -> the edges around its twins, gated on the device by the twin bonds. Newton reads the lists at
  every detection: cross-piece pairs appear exactly when the gate opens (tripwire for that contract).
- A broken bond keeps its twins excluded until they first move apart by the contact distance; then contact is
  restored. Restoring ``intact`` (a snapshot load) restores the exclusion.
- At seam junctions, copies stay excluded while joined through any intact path.
- Intact seams follow the same trajectory as with the 0.2 mm rest-shape radius.
- C9-like press: separated slices overlap their former partners by at most 0.1 mm (the radius: > 1 mm). The gate
  runs 2 worlds; the 16-world acceptance run and the radius control are ``slow``.
- The update is kernels only and replays from a CUDA graph.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.testing.topology import piece_rank, presplit_block, run_seam_press
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.presplit import SplitPlan
from alloy.topology.seam_contact import twin_groups
from alloy.topology.types import TopologyBudget

SPLIT_X = 0.0151                                  # one seam plane through the 30 mm block


def _block(device: str, plan: SplitPlan | None = None, gravity: float = 0.0, worlds: int = 1):
    plan = plan or SplitPlan(planes=(((1.0, 0.0, 0.0), SPLIT_X),))
    return presplit_block(device, cells=(6, 4, 4), plan=plan, worlds=worlds, gravity=gravity)


def _vbd(m: newton.Model, radius: float = 0.0, **kw) -> newton.solvers.SolverVBD:
    return newton.solvers.SolverVBD(m, iterations=5, particle_enable_self_contact=True,
                                    particle_self_contact_margin=5.0e-4, particle_self_contact_gap=2.5e-4,
                                    particle_rest_shape_contact_exclusion_radius=radius, **kw)


def _cross_pairs(m: newton.Model, det, q: np.ndarray, side: np.ndarray) -> tuple[int, int]:
    """Vertex-triangle and edge-edge pairs the detector reports between the two pieces at positions ``q``."""
    det.refit(wp.array(q, dtype=wp.vec3, device=m.device))
    det.vertex_triangle_collision_detection(1.0e-3)
    det.edge_edge_collision_detection(1.0e-3)
    tris, edges = m.tri_indices.numpy(), m.edge_indices.numpy()
    vt = det.vertex_colliding_triangles.numpy().reshape(-1, 2)
    vt = vt[vt[:, 0] >= 0]
    ee = det.edge_colliding_edges.numpy().reshape(-1, 2)
    ee = ee[ee[:, 0] >= 0]
    return (int((side[vt[:, 0]] != side[tris[vt[:, 1], 0]]).sum()),
            int((side[edges[ee[:, 0], 2]] != side[edges[ee[:, 1], 2]]).sum()))


def _rows_sorted(values: np.ndarray, offsets: np.ndarray) -> bool:
    return all(np.all(np.diff(values[offsets[r]:offsets[r + 1]]) >= 0) for r in range(offsets.size - 1))


def test_gate_opens_after_break_and_release_and_closes_on_restore(device):
    sc = _block(device)
    m = sc.model
    vbd = _vbd(m)
    det = vbd.trimesh_collision_detector
    newton_rows = det.vertex_triangle_filtering_list.numpy().size
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(max_events_per_step=1000), sc.blades)
    ctl.attach_self_contact(vbd)
    assert det.vertex_triangle_filtering_list.numpy().size > newton_rows      # twin entries added
    x0 = m.particle_q.numpy()
    side = piece_rank(m)
    state = m.state()
    assert _cross_pairs(m, det, x0, side) == (0, 0)                            # intact: twins excluded
    ctl.request_break(np.arange(ctl.bonds.count))
    ctl.apply(state)
    ctl.pre_step(state, None, 1e-3)
    assert ctl.bonds.intact.numpy().sum() == 0
    assert _cross_pairs(m, det, x0, side) == (0, 0)                            # broken, twins still together
    apart = x0.copy()
    apart[side == 1, 0] += 1.0e-3                                              # > contact distance (0.5 mm)
    state.particle_q.assign(apart)
    ctl.pre_step(state, None, 1e-3)
    released = ctl.state_arrays()["topology.contact_released"].numpy()
    np.testing.assert_array_equal(released[:ctl.bonds.count], 1)
    vt, ee = _cross_pairs(m, det, x0, side)                                    # pressed back: contact restored
    assert vt > 0 and ee > 0
    for vals, offs in ((det.vertex_triangle_filtering_list, det.vertex_triangle_filtering_list_offsets),
                       (det.edge_filtering_list, det.edge_filtering_list_offsets)):
        assert _rows_sorted(vals.numpy(), offs.numpy())                        # Newton's binary search holds
    ctl.bonds.intact.fill_(1)                                                  # snapshot load of an intact seam
    ctl.pre_step(state, None, 1e-3)
    assert ctl.state_arrays()["topology.contact_released"].numpy().sum() == 0
    assert _cross_pairs(m, det, x0, side) == (0, 0)


def test_junction_copies_stay_excluded_while_joined_through_any_path(device):
    sc = presplit_block(device, cells=(6, 4, 4), gravity=0.0)              # split along x and y
    m = sc.model
    bonds = bonds_from_model(m)
    a, b = bonds.a.numpy(), bonds.b.numpy()
    g = twin_groups(bonds.kind.numpy().astype(np.int64), a, b, m.particle_count)
    sizes = np.diff(g.start)
    assert set(sizes.tolist()) == {2, 4}                                       # seam faces and seam crossings
    ctl = TopologyController(m, bonds, TopologyBudget(), sc.blades)
    ctl.attach_self_contact(_vbd(m))
    k = int(np.flatnonzero(sizes == 4)[0])
    members = g.member[g.start[k]:g.start[k + 1]]
    ring = g.bond[g.bond_start[k]:g.bond_start[k + 1]]
    assert ring.size == 4                                                      # four copies joined in a cycle
    p = int(a[ring[0]])
    cut = [int(i) for i in ring if p in (a[i], b[i])]                          # p's two bonds
    q = m.particle_q.numpy()
    q[p] += 1.0e-3                                                             # p moves off its partners
    intact = bonds.intact.numpy()
    intact[cut[0]] = 0
    bonds.intact.assign(intact)
    ctl.pre_step(_with_q(m, q), None, 1e-3)
    comp = ctl.seam_contact.comp.numpy()
    assert len(set(comp[members].tolist())) == 1                               # still joined around the cycle
    intact[cut[1]] = 0
    bonds.intact.assign(intact)
    ctl.pre_step(_with_q(m, q), None, 1e-3)
    comp = ctl.seam_contact.comp.numpy()
    others = [int(x) for x in members if x != p]
    assert len(set(comp[others].tolist())) == 1 and comp[p] not in comp[others]   # p is cut loose


def _with_q(m: newton.Model, q: np.ndarray) -> newton.State:
    s = m.state()
    s.particle_q.assign(q)
    return s


def test_attach_checks_the_solver(device):
    sc = _block(device)
    m = sc.model
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(), sc.blades)
    with pytest.raises(AssemblyError, match="exclusion_radius"):
        ctl.attach_self_contact(_vbd(m, radius=2.0e-4))
    with pytest.raises(AssemblyError, match="self-contact is off"):
        ctl.attach_self_contact(newton.solvers.SolverVBD(m, iterations=1))
    with pytest.raises(AssemblyError, match="not SolverVBD"):
        ctl.attach_self_contact(newton.solvers.SolverXPBD(m))
    other = _block(device).model
    with pytest.raises(AssemblyError, match="different model"):
        ctl.attach_self_contact(_vbd(other))
    ctl.attach_self_contact(_vbd(m))
    with pytest.raises(AssemblyError, match="already attached"):
        ctl.attach_self_contact(_vbd(m))
    auto = TopologyController(m, bonds_from_model(m), TopologyBudget(), sc.blades, entries={"vbd": _vbd(m)})
    assert "topology.contact_released" in auto.state_arrays()                 # entries attach on their own
    plain = TopologyController(m, bonds_from_model(m), TopologyBudget(), sc.blades)
    assert "topology.contact_released" not in plain.state_arrays()


def test_intact_seams_match_the_rest_radius(device):
    """Gravity on a block split along two axes (seam faces and crossings): same trajectory as the 0.2 mm radius."""
    out = []
    for attach in (False, True):
        sc = presplit_block(device, cells=(6, 4, 4), gravity=-9.81)
        m = sc.model
        vbd = _vbd(m, radius=0.0 if attach else 2.0e-4)
        ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(), sc.blades)
        if attach:
            ctl.attach_self_contact(vbd)
        s0, s1, c = m.state(), m.state(), m.control()
        for _ in range(60):
            ctl.pre_step(s0, None, 1e-3)
            vbd.step(s0, s1, c, None, 1e-3)
            ctl.apply(s1)
            s0, s1 = s1, s0
        q = s0.particle_q.numpy()
        a, b = ctl.bonds.a.numpy(), ctl.bonds.b.numpy()
        assert np.linalg.norm(q[a] - q[b], axis=1).max() < 1.0e-4               # seams closed
        assert ctl.bonds.intact.numpy().all()
        out.append(q)
    np.testing.assert_array_equal(out[0], out[1])


def _check_press(worlds: int, device: str) -> None:
    r = run_seam_press(device, worlds=worlds, attach=True)
    print(f"seam press {worlds} worlds: overlap mm {np.round(r['overlap'] * 1e3, 3).tolist()} closest approach mm "
          f"{np.round(r['gap'] * 1e3, 3).tolist()} broken {r['broken']}/{r['bonds']} released {r['released']}")
    assert r["finite"] and r["broken"] == r["bonds"] and r["released"] == r["bonds"]
    assert r["overlap"].max() <= 1.0e-4, r["overlap"]
    assert 0.0 < r["gap"].min() and r["gap"].max() < 1.0e-3, r["gap"]                # pressed into contact


def test_cut_slices_pressed_back_do_not_overlap(device):
    """C9-like: both seams cut, slices pulled apart, pressed back with a per-world shift and tilt (gate size)."""
    _check_press(2, device)


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_cut_slices_pressed_back_do_not_overlap_16_worlds(device):
    """The SCR-SC5-2 acceptance size (about 3 min on CPU; run with ``-m slow``)."""
    _check_press(16, device)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_rest_radius_lets_pressed_slices_pass_through(device):
    """Control: with the permanent 0.2 mm rest-shape radius the same press overlaps by more than 1 mm."""
    r = run_seam_press(device, worlds=2, attach=False)
    assert r["finite"] and r["overlap"].min() > 1.0e-3, r["overlap"]


@pytest.mark.gpu
def test_seam_update_replays_from_a_graph(device):
    if not device.startswith("cuda"):
        pytest.skip("needs CUDA (ALLOY_DEVICE=cuda:0)")
    sc = _block(device)
    m = sc.model
    vbd = _vbd(m)
    det = vbd.trimesh_collision_detector
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(max_events_per_step=1000), sc.blades)
    ctl.attach_self_contact(vbd)
    state = m.state()
    closed = det.vertex_triangle_filtering_list.numpy().copy()
    with wp.ScopedCapture(device=device) as cap:
        ctl.pre_step(state, None, 1e-3)
    ctl.bonds.intact.zero_()
    x0 = m.particle_q.numpy()
    apart = x0.copy()
    apart[piece_rank(m) == 1, 0] += 1.0e-3
    state.particle_q.assign(apart)
    wp.capture_launch(cap.graph)
    opened = det.vertex_triangle_filtering_list.numpy()
    assert not np.array_equal(opened, closed)
    eager = _block(device).model
    vbd2 = _vbd(eager)
    ctl2 = TopologyController(eager, bonds_from_model(eager), TopologyBudget(), sc.blades)
    ctl2.attach_self_contact(vbd2)
    ctl2.bonds.intact.zero_()
    s2 = eager.state()
    s2.particle_q.assign(apart)
    ctl2.pre_step(s2, None, 1e-3)
    np.testing.assert_array_equal(vbd2.trimesh_collision_detector.vertex_triangle_filtering_list.numpy(), opened)
    ctl.bonds.intact.fill_(1)
    wp.capture_launch(cap.graph)
    np.testing.assert_array_equal(det.vertex_triangle_filtering_list.numpy(), closed)
