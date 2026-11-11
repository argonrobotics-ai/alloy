# SPDX-License-Identifier: Apache-2.0
"""Cut damage model (cut.py; clean-room, docs/notes/cleanroom-cut-log.md): behaviour required by
docs/notes/cleanroom-cut-spec.md §3 and §5.

The rig is a 20 x 15 x 15 mm pre-split tet block clamped at x = 0 with one kinematic blade per world. The blade
edge lies along y in the x = 10 mm seam plane and descends through it; there is no collision pipeline, so the
blade acts on the bonds through the cut damage model only. Stiffnesses are read right after ``pre_step``, before
``apply`` commits the breaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.core.snapshot import WorldSnapshot
from alloy.testing.topology import blade_pose, presplit_block, run_twin
from alloy.topology import cut
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.twin import KnifeForceTwin
from alloy.topology.types import Blade, BondKind, TopologyBudget

TOP_Z = 0.015                     # [m] block top
EDGE_BELOW_BODY = 0.02            # [m] presplit_block blade edge sits 20 mm below the body origin
SPEED = 0.05                      # [m/s] knife descent (the papers' velocity_y, RSS Table A2)


@dataclass
class Rig:
    scene: object
    ctl: TopologyController
    solver: newton.solvers.SolverVBD
    states: list
    cut_bonds: np.ndarray         # TET_CUT bond ids
    springs: np.ndarray           # their twin spring ids


@dataclass
class Trace:
    edge_z: np.ndarray            # [T] edge height at each pre_step [m]
    ke: np.ndarray                # [T, n] twin stiffness after pre_step
    phase: np.ndarray             # [T, n]
    damage: np.ndarray            # [T, n]
    contact: np.ndarray           # [T, n]


def _rig(device: str, blade_kw: list[dict], gravity: float = 0.0) -> Rig:
    sc = presplit_block(device, worlds=len(blade_kw), cells=(4, 3, 3), gravity=gravity, k_mu=5e4, k_lambda=5e4)
    m = sc.model
    blades = [Blade(b.body, b.shape, b.edge_a, b.edge_b, **kw) for b, kw in zip(sc.blades, blade_kw)]
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=64), blades)
    ids = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT))
    springs = m.topology.bond_spring.numpy()[ids]
    return Rig(sc, ctl, newton.solvers.SolverVBD(m, iterations=4), [m.state(), m.state()], ids, springs)


def _run(rig: Rig, edge_z: np.ndarray, dt: float, edge_y: np.ndarray | None = None) -> Trace:
    """Step the rig with the blade edge at height ``edge_z[k]`` and offset ``edge_y[k]`` along itself (x = 10 mm,
    centred in y by default) in every world."""
    m, ctl = rig.scene.model, rig.ctl
    s0, s1 = rig.states
    c = m.control()
    rows: dict[str, list] = {"ke": [], "phase": [], "damage": [], "contact": []}
    ys = np.zeros(len(edge_z)) if edge_y is None else edge_y
    for z, y in zip(edge_z, ys):
        bq = s0.body_q.numpy()
        pose = blade_pose(np.array([0.01, 0.0075 + y, z + EDGE_BELOW_BODY]), np.array([0.0, 1.0, 0.0]),
                          np.array([0.0, 0.0, -1.0]))
        bq[rig.scene.blade_bodies] = pose
        s0.body_q.assign(bq)
        ctl.pre_step(s0, None, dt)
        rows["ke"].append(m.spring_stiffness.numpy()[rig.springs])
        for key, arr in (("phase", ctl.phase), ("damage", ctl.bonds.damage), ("contact", ctl.contact)):
            rows[key].append(arr.numpy()[rig.cut_bonds])
        rig.solver.step(s0, s1, c, None, dt)
        ctl.apply(s1)
        s0, s1 = s1, s0
    rig.states = [s0, s1]
    return Trace(np.asarray(edge_z), *(np.stack(rows[k]) for k in ("ke", "phase", "damage", "contact")))


def _descent(travel: float, dt: float, start: float = TOP_Z + 2.0e-3) -> np.ndarray:
    return start - SPEED * dt * np.arange(int(round(travel / (SPEED * dt))))


def _saw(n: int, dt: float, amplitude: float = 2.0e-3, period: float = 0.1) -> np.ndarray:
    """[m] Sideways sawing offset along the edge (the slicing motion of IJRR §5.5-5.6)."""
    return amplitude * np.sin(2.0 * np.pi * dt * np.arange(n) / period)


def _cut_travel(tr: Trace, cols: np.ndarray) -> np.ndarray:
    """[m] Blade travel at which each listed bond's stiffness first reads exactly 0 (nan if never)."""
    out = np.full(len(cols), np.nan)
    for j, c in enumerate(cols):
        hit = np.flatnonzero((tr.phase[:, c] == cut.PHASE_CUTTING) & (tr.ke[:, c] == 0.0))
        if hit.size:
            out[j] = tr.edge_z[0] - tr.edge_z[hit[0]]
    return out


def test_cut_damage_is_monotone_and_reaches_exactly_zero(device):
    rig = _rig(device, [{}])                                   # default Blade cut parameters
    tr = _run(rig, _descent(0.02, 1.0e-3), 1.0e-3)
    entered = np.flatnonzero((tr.phase == cut.PHASE_CUTTING).any(axis=0))
    assert entered.size >= 10
    cutting = tr.phase == cut.PHASE_CUTTING
    for c in entered:
        ke = tr.ke[cutting[:, c], c]
        assert np.all(np.diff(ke) <= 0.0) and ke.min() >= 0.0
        np.testing.assert_allclose(tr.damage[cutting[:, c], c], 1.0 - ke / 500.0, atol=1e-6)
    zeroed = _cut_travel(tr, entered)
    print(f"cut damage: {entered.size} bonds cutting, {np.isfinite(zeroed).sum()} reached 0 exactly, "
          f"cut travel {1e3 * np.nanmin(zeroed):.2f}-{1e3 * np.nanmax(zeroed):.2f} mm")
    assert np.isfinite(zeroed).all()                           # every bond the blade passed is cut, in finite travel
    done = rig.ctl.cut_done.numpy()[rig.cut_bonds[entered]]
    assert done.all()                                          # ... and the commit broke them as blade cuts


def test_no_cut_damage_while_blade_withdrawn(device):
    rig = _rig(device, [{"softness": 1.0}])
    dt = 1.0e-3
    press = _descent(1.0e-3, dt)                               # edge stops 1 mm above the top row
    hold = np.full(50, 0.03)                                   # withdrawn: far out of reach
    tr1 = _run(rig, press, dt)
    tr2 = _run(rig, hold, dt)
    tr3 = _run(rig, _descent(0.02, dt, start=float(press[-1])), dt)
    part = np.flatnonzero((tr1.phase[-1] == cut.PHASE_CUTTING) & (tr1.damage[-1] > 0.0) & (tr1.damage[-1] < 1.0))
    print(f"withdrawn: {part.size} partly cut bonds, damage {tr1.damage[-1, part].round(3).tolist()}")
    assert part.size > 0
    assert np.all(tr2.contact == -1)
    np.testing.assert_array_equal(tr2.ke[:, part], np.broadcast_to(tr1.ke[-1, part], tr2.ke[:, part].shape))
    assert np.all(tr2.phase[:, part] == cut.PHASE_CUTTING)
    assert np.all(tr3.ke[-1, part] == 0.0)                     # the blade resumes and finishes the cut


def test_cut_progress_is_independent_of_dt(device):
    """The rate is per unit blade travel: a sawing descent cuts each bond at the same descent at dt and dt/2
    (gravity off, so the soft block does not swing; the material then only moves where the cut lets it)."""
    travel = {}
    for dt in (1.0e-3, 5.0e-4):
        rig = _rig(device, [{"softness": 0.5}])
        z = _descent(0.02, dt)
        tr = _run(rig, z, dt, _saw(len(z), dt))
        travel[dt] = _cut_travel(tr, np.flatnonzero((tr.phase == cut.PHASE_CUTTING).any(axis=0)))
    a, b = travel[1.0e-3], travel[5.0e-4]
    both = np.isfinite(a) & np.isfinite(b)
    rel = np.abs(a[both] - b[both]) / b[both]
    print(f"dt check: {both.sum()} bonds cut at both dt, cut travel max rel diff {100 * rel.max():.2f} %, "
          f"mean {100 * rel.mean():.2f} %")
    assert both.sum() >= 10
    np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
    assert rel.max() <= 0.05


def test_larger_softness_cuts_sooner(device):
    rig = _rig(device, [{"softness": 1.0}, {"softness": 4.0}])
    tr = _run(rig, _descent(0.02, 1.0e-3), 1.0e-3)
    world = rig.ctl.bonds.world.numpy()[rig.cut_bonds]
    touched = (tr.phase == cut.PHASE_CUTTING).any(axis=0)
    slow, fast = (_cut_travel(tr, np.flatnonzero((world == w) & touched)) for w in (0, 1))
    print(f"softness 1 vs 4: mean cut travel {1e3 * np.nanmean(slow):.3f} vs {1e3 * np.nanmean(fast):.3f} mm")
    assert np.isfinite(fast).all() and np.isfinite(slow).sum() >= 10
    ok = np.isfinite(slow)
    assert np.all(fast[ok] <= slow[ok]) and np.mean(fast[ok]) < np.mean(slow[ok])


def test_slicing_cuts_at_a_shallower_descent(device):
    """IJRR §5.5: travel along the edge does work too, so sawing while descending cuts sooner than pressing."""
    dt = 1.0e-3
    z = _descent(0.02, dt)
    press, saw = (_run(_rig(device, [{"softness": 0.5}]), z, dt, y) for y in (None, _saw(len(z), dt)))
    cols = np.flatnonzero((press.phase == cut.PHASE_CUTTING).any(axis=0))
    a, b = _cut_travel(press, cols), _cut_travel(saw, cols)
    print(f"press vs saw: mean descent to cut {1e3 * np.nanmean(a):.3f} vs {1e3 * np.nanmean(b):.3f} mm")
    assert np.isfinite(b).all() and np.all(b[np.isfinite(a)] < a[np.isfinite(a)])


def test_larger_softness_cuts_with_lower_knife_force(device):
    """Softness scales the damage per unit of knife work (Eq. 2), so the cutting force over the stroke is lower: the
    mean over the 1 / 2 / 3 mm samples. Pointwise it need not be: with the soft seam twins (SOFT_SEAM_KE_RATIO 1)
    softness 4 already presses the second layer at 2 mm (0.701 N) while softness 1 is in its first layer's dip
    (0.652 N); 1 / 3 mm read 0.573 vs 0.434 N and 0.907 vs 0.173 N."""
    twin = KnifeForceTwin()
    low, high = {"softness": 1.0, "cut_spring_ke": 1000.0}, {"softness": 4.0, "cut_spring_ke": 1000.0}
    sim = run_twin(twin, twin.conditions(quick=True)[0], [low, high], dt=5.0e-4, steps=450, device=device)
    print("knife force softness 1 vs 4:", {k: (round(float(v[0]), 4), round(float(v[1]), 4)) for k, v in sim.items()})
    mean = np.mean([np.asarray(v, np.float64) for v in sim.values()], axis=0)
    assert mean[1] < 0.75 * mean[0]


def test_cut_damage_track_is_in_the_snapshot(device):
    rig = _rig(device, [{"softness": 1.0}], gravity=-9.81)
    m, dt = rig.scene.model, 1.0e-3
    path = _descent(0.012, dt)
    _run(rig, path[:40], dt)
    arrays = rig.ctl.state_arrays()
    assert {"topology.cut_damage.blade", "topology.cut_damage.edge0", "topology.cut_damage.edge1"} <= set(arrays)
    snap = WorldSnapshot(m, rig.states[0], extra=arrays, extra_world=rig.ctl.state_worlds())
    snap.save(rig.states[0])
    first = _run(rig, path[40:], dt)
    snap.restore(rig.states[0])
    again = _run(rig, path[40:], dt)
    assert (first.phase == cut.PHASE_CUTTING).any()
    np.testing.assert_array_equal(first.ke, again.ke)
    np.testing.assert_array_equal(first.damage, again.damage)


def test_reach_default_and_parameter_checks(device):
    assert cut.default_reach() == pytest.approx(5.0e-4)
    assert cut.resolve_reach(None) == cut.default_reach() and cut.resolve_reach(0.0) == 0.0
    with pytest.raises(AssemblyError):
        cut.resolve_reach(-1.0e-4)
    sc = presplit_block(device, cells=(2, 2, 2))
    b = sc.blades[0]
    for bad in ({"softness": float("nan")}, {"softness": -1.0}, {"cut_spring_ke": 0.0}):
        with pytest.raises(AssemblyError):
            TopologyController(sc.model, bonds_from_model(sc.model), TopologyBudget(),
                               [Blade(b.body, b.shape, b.edge_a, b.edge_b, **bad)])
