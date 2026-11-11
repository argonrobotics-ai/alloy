# SPDX-License-Identifier: Apache-2.0
"""Virtual-node copies as a Simulation entry (alloy/topology/copies_model.py, vnode_topology.py; coordinator GO
2026-11-06T19:36Z), CPU.

A soft block's card opts in with ``topology: {cut_model: copies}``. A kinematic knife card with a ``blade:`` block
cuts it through ``alloy.build``. The checks:
- the copy rows survive ``replicate`` with per-world offsets;
- the copies are recoloured off their hubs;
- the cut goes through every world with mass kept and health OK;
- a cut, a rollback and the same cut again are equal bit for bit (cutter state, model arrays and SolverVBD's history
  are in the Simulation's snapshot);
- a card without the opt-in gets no copies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.snapshot import WorldSnapshot

KNIFE_Z, SPEED = 0.0309, 0.05         # [m] knife origin, edge 2 mm above the block; [m/s] descent
CARDS = {
    "materials/agar_cut.yaml": """schema: alloy.material/1
name: agar_cut
density: {value: 1010 kg/m^3, source: model, note: "agar gel"}
surface:
  mu_static: {value: 0.3, source: model, note: "wet gel"}
  mu_dynamic: {value: 0.25, source: model, note: "wet gel"}
  restitution: {value: 0.05, source: model, note: "gel"}
elastic:
  youngs: {value: 50 kPa, source: model, note: "soft agar"}
  poisson: {value: 0.45, source: model, note: "near incompressible"}
failure:
  fracture_strain: {value: 0.3, source: model, note: "gel"}
""",
    "objects/agar_block.yaml": """schema: alloy.object/1
name: agar_block
shape: {primitive: {kind: box, size: [0.01, 0.0075, 0.0075]}}
material: agar_cut
budget: {tets: 216}
topology: {cut_model: copies}
""",
    "objects/knife_kinematic.yaml": """schema: alloy.object/1
name: knife_kinematic
shape: {primitive: {kind: box, size: [0.06, 0.0125, 0.0006]}}
material: stainless_304
motion: kinematic
features:
  thickness: {value: 0.0012, source: model, note: "1.2 mm blade"}
  min_wall: {value: 0.0012, source: model, note: "1.2 mm blade"}
topology: {}
blade:
  edge_a: [-0.06, -0.0125, 0.0]
  edge_b: [0.06, -0.0125, 0.0]
""",
    "scenes/cut_block.yaml": f"""schema: alloy.scene/1
robots: []
objects:
  - {{object: agar_block, pose: [0, 0, 0.0089, 0, 0, 0, 1]}}
  - {{object: knife_kinematic, pose: [0, 0.0017, {KNIFE_Z}, 0.7071068, 0, 0, 0.7071068]}}
worlds: 2
control_dt: 0.01
""",
}


def _build(root: Path, opt_in: bool = True, failure: bool = True):
    import alloy

    for rel, text in CARDS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        text = text if opt_in else text.replace("topology: {cut_model: copies}", "topology: {}")
        if not failure:
            text = text.split("failure:")[0]
        (root / rel).write_text(text)
    scene = alloy.Scene.load(root / "scenes" / "cut_block.yaml")
    scene.add_ground()
    return alloy.build(scene, device="cpu")


@pytest.fixture(scope="module")
def cut_sim(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("copies_cards"))


def test_the_copy_rows_survive_replicate_per_world(cut_sim) -> None:
    from alloy.topology.copies_model import copies_from_model
    from alloy.topology.vnode_topology import VNodeTopology

    eng = cut_sim.engine
    assert isinstance(eng.topology, VNodeTopology)
    m, c = eng.model, copies_from_model(eng.model)
    pw = m.particle_world.numpy()
    tw = pw[m.tet_indices.numpy()[c.tet, 0]]
    assert c.count > 0 and np.bincount(tw).tolist() == [c.count // 2] * 2                 # every tet, both worlds
    assert np.bincount(pw[c.node_hub]).tolist() == [len(c.node_hub) // 2] * 2
    np.testing.assert_array_equal(pw[c.node], np.repeat(pw[c.node_hub][:, None], 2, axis=1))  # copies in their world
    springs = m.spring_indices.numpy().reshape(-1, 2)[c.weld]                              # weld = hub -> copy
    np.testing.assert_array_equal(np.sort(springs, axis=-1),
                                  np.sort(np.stack([np.repeat(c.node_hub[:, None], 2, 1), c.node], -1), axis=-1))
    tets = m.tet_indices.numpy()
    np.testing.assert_array_equal(pw[tets[c.copy_tet[:, 0], 0]], pw[tets[c.tet, 0]])
    color = np.full(m.particle_count, -1)
    for g, ids in enumerate(m.particle_color_groups):
        color[ids.numpy()] = g
    assert np.all(color[c.node] != color[c.node_hub][:, None])                            # recoloured off the hub
    # SolverVBD's spring kernel reads each end's colour from this array: left as finalize wrote it, a copy's weld
    # acted on its hub only (73 of 90 copies of this block) and cut pieces slid apart for good
    np.testing.assert_array_equal(m.particle_colors.numpy(), color)


def _knife(sim):
    """A test pose writer for the knife, by its own step count (``sim.drive`` keeps its own clock, outside the
    snapshot): the knife straight down at SPEED from KNIFE_Z, in every world."""
    eng = sim.engine
    m, bodies = eng.model, eng.topology.cutter.blades.body.numpy()
    joint = np.array([int(np.flatnonzero(m.joint_child.numpy() == b)[0]) for b in bodies])
    qs, ds = m.joint_q_start.numpy()[joint], m.joint_qd_start.numpy()[joint]
    state = {"k": 0}

    def write(s0, s1, ctrl, c, dt):
        state["k"] += 1
        z = KNIFE_Z - SPEED * dt * state["k"]
        for arr, idx in ((s1.joint_q, qs + 2), (s1.joint_qd, ds + 2)):
            a = arr.numpy()
            a[idx] = z if arr is s1.joint_q else -SPEED
            arr.assign(a)
        bq, bqd = s1.body_q.numpy(), s1.body_qd.numpy()
        bq[bodies, 2], bqd[bodies], bqd[bodies, 2] = z, 0.0, -SPEED
        s1.body_q.assign(bq)
        s1.body_qd.assign(bqd)

    eng.add_step_observer(write)
    return state


def _record(eng) -> dict[str, np.ndarray]:
    out = {k: v.numpy().copy() for k, v in eng.topology.state_arrays().items()}
    out["particle_q"], out["particle_qd"] = eng.state.particle_q.numpy().copy(), eng.state.particle_qd.numpy().copy()
    return out


def test_a_cut_a_rollback_and_the_same_cut_again_are_bitwise(tmp_path) -> None:
    sim = _build(tmp_path)
    eng = sim.engine
    knife = _knife(sim)
    assert int(eng.topology.cutter.split.numpy().sum()) == 0          # the edge 2 mm above the block: no split yet
    snap = WorldSnapshot(eng.model, eng.state, extra=eng.state_arrays(), extra_world=eng.state_world_maps())
    assert any(k.startswith("topology/") for k in eng.state_arrays())                    # the cutter's arrays
    assert any("particle_q_prev" in k for k in eng.state_arrays())                       # SolverVBD's history
    snap.save(eng.state)
    k0 = knife["k"]
    for _ in range(12):
        eng.step_period()
    first = _record(eng)
    assert eng.topology.ledger().events.min() > 0                                       # splits in every world
    split = eng.topology.cutter.split.numpy()
    assert np.all(np.bincount(eng.topology.cutter.tets.world.numpy()[split == 1], minlength=2) > 0)
    snap.restore(eng.state)
    assert int(eng.topology.cutter.split.numpy().sum()) == 0                             # the splits are undone
    knife["k"] = k0
    for _ in range(12):
        eng.step_period()
    again = _record(eng)
    for k, v in first.items():
        np.testing.assert_array_equal(again[k], v, err_msg=k)


def test_the_knife_cuts_through_every_world(cut_sim) -> None:
    eng = cut_sim.engine
    m0 = eng.model.particle_mass.numpy().astype(np.float64).sum()
    cut_sim.drive("knife_kinematic", velocity=(0.0, 0.0, -SPEED), duration=0.33)
    cut_sim.run(0.4, record=[])
    cut = eng.topology.cutter
    per_world = np.bincount(cut.tets.world.numpy()[cut.split.numpy() == 1], minlength=2)
    assert np.all(per_world == per_world[0]) and per_world[0] >= 30                     # through, alike
    assert eng.model.particle_mass.numpy().astype(np.float64).sum() == pytest.approx(m0, rel=1e-6)
    assert np.isfinite(eng.state.particle_q.numpy()).all()
    assert cut_sim.check()["ok"]


def test_cut_pieces_stand_and_stay_where_the_knife_left_them(tmp_path) -> None:
    """The example's cut (examples/topology/cut_anywhere.py): the knife crosses the block's long axis 1.7 mm off a
    mesh plane, so each piece keeps two layers of mesh nodes. After the cut both pieces stand (no node sinks or tips)
    and the larger one rests. With the copies' colour array left as built (``tet_copies.recolor_model``) its welds
    acted on one end only and it slid away at 6.5 mm/s for good. The smaller piece still creeps at SolverVBD's 10
    iterations (0.56 mm/s measured; 0.007 mm/s at 160: docs/known_gaps/cutting-copies-coarse-mesh.md)."""
    import alloy

    cards = dict(CARDS)
    cards["scenes/cut_block.yaml"] = cards["scenes/cut_block.yaml"].replace(
        f"pose: [0, 0.0017, {KNIFE_Z}, 0.7071068, 0, 0, 0.7071068]", f"pose: [0.0017, 0, {KNIFE_Z}, 0.5, 0.5, 0.5, 0.5]"
    ).replace("worlds: 2", "worlds: 1")
    assert "0.5, 0.5, 0.5, 0.5" in cards["scenes/cut_block.yaml"]
    for rel, text in cards.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    scene = alloy.Scene.load(tmp_path / "scenes" / "cut_block.yaml")
    scene.add_ground()
    sim = alloy.build(scene, device="cpu")
    eng = sim.engine
    hubs = np.unique(eng.topology.cutter.copies.node_hub)
    x0 = eng.state.particle_q.numpy()[hubs].copy()
    left, right = x0[:, 0] < 0.0017, x0[:, 0] > 0.0017
    sim.drive("knife_kinematic", velocity=(0.0, 0.0, -SPEED), duration=0.33)
    sim.step(50)
    x1 = eng.state.particle_q.numpy()[hubs].copy()
    sim.step(20)
    x2 = eng.state.particle_q.numpy()[hubs].copy()
    assert int((eng.topology.cutter.split.numpy() == 1).sum()) == 24                    # the slab the plane crosses
    assert np.abs((x2 - x0)[:, 2] + 0.42e-3).max() < 0.1e-3                             # all settled 0.42 mm, none more
    assert np.abs((x2 - x0)[:, 1]).max() < 0.05e-3
    speed = np.abs(x2 - x1)[:, 0] / 0.2                                                 # [m/s] over the last 0.2 s
    assert speed[left].max() < 0.1e-3                                                   # measured 0.005 mm/s
    assert speed[right].max() < 1.5e-3                                                  # measured 0.56 mm/s
    assert sim.check()["ok"]


def test_a_card_without_the_opt_in_gets_no_copies(tmp_path) -> None:
    from alloy.topology.copies_model import has_copies

    sim = _build(tmp_path, opt_in=False)
    assert not has_copies(sim.engine.model) and sim.engine.topology is None


def test_copies_need_no_failure_values_and_cloth_refuses_them(tmp_path) -> None:
    """A blade cuts copies, so the budget alone opts in (the strain gate is 0 without failure values; WP2 review: the
    material.failure gate dropped the copies silently). Cloth is pre-split only."""
    from types import SimpleNamespace

    from alloy.core.errors import AssemblyError
    from alloy.scene.emitters import topology as emit
    from alloy.topology.copies_model import has_copies
    from alloy.topology.types import TopologyBudget

    sim = _build(tmp_path, failure=False)
    assert has_copies(sim.engine.model) and sim.engine.topology.cutter.work_rule.cut_strain == 0.0
    cloth = SimpleNamespace(card=SimpleNamespace(name="sheet", topology=TopologyBudget(cut_model="copies")),
                            material=SimpleNamespace(failure=None))
    assert emit.wants_presplit(cloth)
    with pytest.raises(AssemblyError, match="cloth is pre-split"):
        emit.add_cloth(newton.ModelBuilder(), cloth, np.zeros((3, 3)), np.array([[0, 1, 2]]), {}, 1.0e-3)


def test_copies_beside_seam_bonds_are_refused(monkeypatch) -> None:
    """One cut model per scene (WP12 review): copies and pre-split seam bonds in one scene raise at wiring."""
    from types import SimpleNamespace

    from alloy.core.errors import AssemblyError
    from alloy.integration import wiring
    from alloy.topology import copies_model

    monkeypatch.setattr(copies_model, "has_copies", lambda model: True)
    with pytest.raises(AssemblyError, match="one model"):
        wiring._copies_topology(SimpleNamespace(model=None), SimpleNamespace(count=3), [])


def test_whole_tets_drops_only_fragments() -> None:
    from alloy.testing.topology import soft_kwargs
    from alloy.topology.copies_model import add_soft_copies, whole_tets

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    x = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]], float) * 1e-3
    add_soft_copies(b, x, np.array([[0, 1, 2, 3]]), soft_kwargs())
    m = b.finalize(device="cpu")
    assert m.tet_count == 3 and whole_tets(m, np.array([[0, 1, 2]])).tolist() == [[0]]
    assert isinstance(m.topology.copy_hub, wp.array)
