# SPDX-License-Identifier: Apache-2.0
"""Frame export for a live viewer (SCR-WP13-EXPORT): alloy.export.FrameExporter and alloy.api._frame_export.

Covers, on the CPU: the export equals ``sim.state`` bit for bit (bodies, cloth, liquid subset and live flags, world
ids, period, time, quaternion order); rod centre lines; stepping with export on equals export off; the deterministic
evenly strided particle downsample; the topology part sent once per world and only on change; ``topology_version``
counting broken bonds exactly at the cuts (and back to 0 on reset); the non-blocking capture/poll semantics (newest
frame wins, ``dropped`` counts); argument validation; and ``frame_export(sim, int)``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.api._frame_export import frame_export
from alloy.export import MAX_WORLDS, Frame, FrameExporter

DT = 0.01
SLOW = pytest.mark.timeout(300)       # cloth on the CPU steps ~10 s per period for 3 worlds (40 VBD iterations)


def _starts(worlds: int) -> np.ndarray:
    """One start per world, so the worlds differ."""
    x = np.linspace(-0.05, 0.05, worlds)
    return np.stack([x, np.zeros(worlds), np.full(worlds, 0.9)], axis=1)


def _mixed_scene(worlds: int = 3, cloth: bool = True) -> alloy.Scene:
    """Free box, cloth and a liquid in an open box, each world's box and cloth started elsewhere."""
    scene = alloy.Scene(worlds=worlds, dt=DT)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=_starts(worlds), mass=0.2)
    if cloth:
        scene.add_cloth("sheet", size=(0.2, 0.2), pos=_starts(worlds) + np.array([0.5, 0.0, -0.4]), resolution=4,
                        mass=0.05, self_contact=False)
    scene.add_open_box("cup", inner=(0.1, 0.1, 0.08), wall=0.01, pos=(-0.5, 0.0, 0.0))
    scene.add_liquid("water", size=(0.03, 0.03, 0.03), pos=(-0.5, 0.0, 0.03))
    return scene


def _liquid_scene(size: float, worlds: int = 1) -> alloy.Scene:
    scene = alloy.Scene(worlds=worlds, dt=DT)
    scene.add_ground()
    scene.add_open_box("cup", inner=(0.12, 0.12, 0.1), wall=0.01, pos=(0.0, 0.0, 0.0))
    scene.add_liquid("water", size=(size, size, 0.05), pos=(0.0, 0.0, 0.03))
    return scene


def _f32(a) -> np.ndarray:
    return np.asarray(a).astype(np.float32)


def _poll(exp: FrameExporter) -> list[Frame]:
    """Capture now and poll (the CPU path is synchronous, so the frame is ready)."""
    assert exp.capture()
    frames = exp.poll()
    assert frames is not None
    return frames


@pytest.fixture(scope="module")
def mixed_sim():
    sim = alloy.build(_mixed_scene(), device="cpu", seed=0)
    sim.step(2)
    return sim


# --- 1. export equals state --------------------------------------------------------------------------------------


@SLOW
def test_export_equals_state_bitwise(mixed_sim):
    sim = mixed_sim
    exp = FrameExporter(sim, [0, 2])
    frames = _poll(exp)
    assert [f.world for f in frames] == [0, 2]
    for f in frames:
        st = sim.state(world=f.world)
        b = f.topology.body_names.index("block")
        assert np.array_equal(f.body_pos[b], _f32(st["block.pos"]))
        assert np.array_equal(f.body_quat_xyzw[b], _f32(st["block.quat_xyzw"]))
        assert np.array_equal(f.meshes["sheet"], _f32(st["sheet.points"]))
        idx = f.topology.particles["water"]["indices"]
        assert np.array_equal(f.particles["water"]["pos"], _f32(st["water.points"])[idx])
        assert np.array_equal(f.particles["water"]["active"], np.asarray(st["water.active"])[idx].astype(bool))
        assert f.period == round(sim.t / sim.dt) and f.t == sim.t and f.quat_order == "xyzw"
    assert not np.array_equal(frames[0].body_pos, frames[1].body_pos)        # the worlds do differ


# --- 2. rod --------------------------------------------------------------------------------------------------------


@pytest.mark.timeout(300)
@pytest.mark.parametrize("clamp", [None, "start"])                   # the chain route, and rod_vbd (WP9)
def test_rod_centre_line_matches_state(clamp):
    scene = alloy.Scene(dt=DT)
    scene.add_ground()
    scene.add_rod("cord", length=0.3, radius=0.004, pos=(0.0, 0.0, 0.15), direction=(1.0, 0.0, 0.0), clamp=clamp)
    sim = alloy.build(scene, device="cpu")
    assert sim.routes()["cord"]["route"] == ("rod_vbd" if clamp else "chain")
    sim.step(2 if clamp else 5)
    frame = _poll(FrameExporter(sim, [0]))[0]
    ref = np.asarray(sim.state(world=0)["cord.points"])
    err = float(np.max(np.abs(frame.rods["cord"].astype(np.float64) - ref)))
    print(f"rod export vs state: max abs difference {err:.3e} m")
    assert frame.rods["cord"].shape == ref.shape
    assert err <= 1e-6
    info = frame.topology.rods["cord"]                               # JOBLIVE: a rod drawn by its own bodies
    assert info["route"] == sim.routes()["cord"]["route"] and info["segments"] == len(info["bodies"])
    assert all(frame.topology.body_names[b].startswith("cord") for b in info["bodies"])
    assert info["drawn_by_bodies"] == any(frame.topology.body_shapes[b] for b in info["bodies"])


# --- 3. export on == off -----------------------------------------------------------------------------------------


def test_stepping_with_export_equals_without():
    on = alloy.build(_mixed_scene(2, cloth=False), device="cpu", seed=3)
    off = alloy.build(_mixed_scene(2, cloth=False), device="cpu", seed=3)
    exp = FrameExporter(on, [0, 1])
    for _ in range(10):
        on.step(1)
        off.step(1)
        exp.capture()
        assert exp.poll() is not None
    a, b = on.state(), off.state()
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


# --- 4. deterministic downsample -----------------------------------------------------------------------------------


def test_downsample_is_deterministic_and_spread():
    sim = alloy.build(_liquid_scene(0.1), device="cpu")
    total = sim.state(world=0)["water.points"].shape[0]
    assert total > 1500
    e1, e2 = FrameExporter(sim, [0], max_points=500), FrameExporter(sim, [0], max_points=500)
    first = _poll(e1)[0]
    sim.step(2)
    second, other = _poll(e1)[0], _poll(e2)[0]
    i1 = first.topology.particles["water"]["indices"]
    assert 0 < len(i1) <= 500
    assert second.topology is None and second.particles["water"]["pos"].shape == (len(i1), 3)
    assert np.array_equal(i1, other.topology.particles["water"]["indices"])
    want = np.sort(np.random.default_rng(0).choice(total, len(i1), replace=False))   # WP12: no lattice stride
    assert np.array_equal(i1, want) and (np.diff(i1) > 0).all() and i1[-1] < total
    pts = np.asarray(sim.state(world=0)["water.points"])
    assert np.array_equal(second.particles["water"]["pos"], _f32(pts)[i1])
    assert np.array_equal(other.particles["water"]["pos"], second.particles["water"]["pos"])


# --- 5. topology part only on change --------------------------------------------------------------------------------


def test_topology_part_sent_once_with_statics_and_faces():
    scene = alloy.Scene(worlds=2, dt=DT)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.4, 0.2), pos=(0, 0, 0.1), static=True)
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.5), resolution=6, mass=0.05)
    sim = alloy.build(scene, device="cpu")
    exp = FrameExporter(sim, [0, 1])
    first = _poll(exp)
    assert all(f.topology is not None for f in first)
    sim.step(1)
    assert all(f.topology is None for f in _poll(exp))
    topo = first[0].topology
    kinds = {s["kind"] for s in topo.statics}
    assert {"plane", "box"} <= kinds
    box = next(s for s in topo.statics if s["kind"] == "box")
    assert np.allclose(np.asarray(box["size"]), [0.6, 0.4, 0.2])                   # full edges, as add_box
    assert np.array_equal(topo.mesh_faces["sheet"], sim.faces("sheet"))


# --- 6. topology_version at cuts -----------------------------------------------------------------------------------


def _welded_sim():
    """Two worlds, a block welded to a table by a bond (scene-card attachment) that never breaks on its own."""
    scene = alloy.Scene(worlds=2, dt=DT)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_box("blk", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.425), mass=0.2)
    scene.extra["attachments"] = [{"a": "table", "b": "blk", "kind": "weld", "break_force": 1.0e9}]
    return alloy.build(scene, device="cpu")


def test_topology_version_changes_exactly_at_cuts():
    sim = _welded_sim()
    ctl = sim.engine.topology
    assert ctl is not None and ctl.bonds.count == 2
    exp = FrameExporter(sim, [0, 1])
    f0, f1 = _poll(exp)
    assert (f0.topology_version, f1.topology_version) == (0, 0) and f0.topology and f1.topology
    sim.step(2)
    assert [f.topology is None for f in _poll(exp)] == [True, True]          # no change, nothing resent
    ctl.request_break([0], world_mask=[False, True])
    sim.step(2)
    f0, f1 = _poll(exp)
    assert (f0.topology_version, f1.topology_version) == (0, 1)
    assert f0.topology is None and f1.topology is not None                  # only the cut world resends
    sim.step(1)
    assert [f.topology is None for f in _poll(exp)] == [True, True]
    sim.reset()
    f0, f1 = _poll(exp)
    assert (f0.topology_version, f1.topology_version) == (0, 0)
    assert f0.topology is None and f1.topology is not None                  # back to 0 is a change in world 1


# --- 7. non-blocking semantics -------------------------------------------------------------------------------------


def test_poll_and_capture_semantics():
    sim = alloy.build(_mixed_scene(2, cloth=False), device="cpu")
    exp = FrameExporter(sim, [0])
    assert exp.poll() is None
    sim.step(1)
    assert exp.capture()
    first_period = round(sim.t / sim.dt)
    sim.step(1)
    assert exp.capture()
    frames = exp.poll()
    assert frames is not None and frames[0].period == first_period + 1
    assert exp.dropped == 1
    assert exp.poll() is None


def test_no_cuda_work_during_a_graph_capture(monkeypatch):
    """JOBLIVE's rule: capture() issues nothing while the device captures a graph (it refuses), and poll() makes no
    CUDA call at all (no launch, copy, event query or synchronize), so a publisher thread cannot break a capture."""
    import alloy.export.exporter as E

    sim = alloy.build(_mixed_scene(2, cloth=False), device="cpu")
    exp = FrameExporter(sim, [0, 1])
    sim.step(1)
    assert exp.capture()

    def forbidden(*a, **k):
        raise AssertionError("CUDA work issued")

    monkeypatch.setattr(E, "_capturing", lambda device: True)
    for name in ("launch", "copy", "synchronize_device", "synchronize", "record_event", "wait_event"):
        monkeypatch.setattr(E.wp, name, forbidden, raising=False)
    monkeypatch.setattr(E.K.wp, "launch", forbidden)
    assert exp.capture() is False and exp.refused == 1
    exp.flush()                                          # a no-op while capturing
    frames = exp.poll()                                  # the frame staged before the capture began
    assert frames is not None and [f.world for f in frames] == [0, 1]


def test_resend_topology_after_a_reconnect():
    sim = alloy.build(_mixed_scene(2, cloth=False), device="cpu")
    exp = FrameExporter(sim, [0, 1])
    first = _poll(exp)
    assert all(f.topology is not None for f in first)
    assert all(f.topology is None for f in _poll(exp))
    exp.resend_topology()
    assert all(f.topology is not None for f in _poll(exp))           # sent again, same version
    assert all(f.topology is None for f in _poll(exp))


# --- 8. validation ---------------------------------------------------------------------------------------------------


@SLOW
def test_invalid_world_picks_raise(mixed_sim):
    sim = mixed_sim
    with pytest.raises(ValueError):
        FrameExporter(sim, [])
    with pytest.raises(ValueError):
        FrameExporter(sim, list(range(MAX_WORLDS + 1)))
    with pytest.raises(ValueError):
        FrameExporter(sim, [0, 0])
    with pytest.raises(alloy.SceneError) as err:                     # OSS-3: an alloy error with a fix
        FrameExporter(sim, [sim.worlds])
    assert err.value.field == "worlds" and err.value.fix
    assert FrameExporter(sim, 1).plan.worlds == [1]                 # an int world, as frame_export takes
    with pytest.raises(alloy.FeatureUnavailableError):              # a processes= Sim has no engine
        FrameExporter(SimpleNamespace(worlds=4), [0])


# --- 9. frame_export with an int -----------------------------------------------------------------------------------


@SLOW
def test_frame_export_accepts_an_int(mixed_sim):
    frames = frame_export(mixed_sim, 1)
    assert len(frames) == 1 and frames[0].world == 1 and frames[0].topology is not None
    st = mixed_sim.state(world=1)
    assert np.array_equal(frames[0].meshes["sheet"], _f32(st["sheet.points"]))


# --- 10. soft-body tets and virtual-node cuts (slice 2) -------------------------------------------------------------


def _cube_mesh(edge: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """A closed cube, faces wound counter-clockwise seen from outside."""
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * edge
    f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3], [0, 4, 6],
                  [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    return v, f[:, ::-1]


@pytest.mark.timeout(300)
def test_meshed_soft_body_exports_its_tets_all_live():
    v, f = _cube_mesh()
    scene = alloy.Scene(dt=DT)
    scene.add_ground()
    scene.add_mesh("jelly", vertices=v, faces=f, pos=(0.0, 0.0, 0.05), material="silicone_40a")
    try:
        sim = alloy.build(scene, device="cpu")
    except alloy.FeatureUnavailableError:
        pytest.skip("the mesh extra is not installed")
    assert sim.routes()["jelly"]["route"] == "soft_vbd"
    frame = _poll(FrameExporter(sim, [0]))[0]
    topo, model = frame.topology, sim.engine.model
    start = sim.index.particles["jelly"].start
    tets = model.tet_indices.numpy().reshape(-1, 4)[sim.built.handles["jelly"].template.tet] - start
    assert topo.mesh_tets["jelly"].shape == (len(tets), 4) and len(tets) > 0
    assert np.array_equal(topo.mesh_tets["jelly"], tets)
    assert topo.mesh_tets["jelly"].min() >= 0 and topo.mesh_tets["jelly"].max() < len(frame.meshes["jelly"])
    assert topo.mesh_tet_active["jelly"].all() and topo.mesh_vertex_active["jelly"].all()
    assert frame.topology_version == 0                                    # no bond broken, no dead tet


# virtual-node copies: the card set of tests/topology/test_vnode_entry.py (CARDS: a 216-tet agar block + a blade)
_ENTRY = Path(__file__).resolve().parents[1] / "topology" / "test_vnode_entry.py"


def _vnode_sim(root: Path):
    spec = importlib.util.spec_from_file_location("_vnode_entry_cards", _ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod._build(root)


def _lower_knife_in_world(sim, mod, world: int) -> dict:
    """Step observer: the knife of ``world`` alone descends at SPEED from KNIFE_Z (the other world's blade stays)."""
    eng = sim.engine
    m, bodies = eng.model, eng.topology.cutter.blades.body.numpy()
    body = int(bodies[world])
    joint = int(np.flatnonzero(m.joint_child.numpy() == body)[0])
    q, qd = int(m.joint_q_start.numpy()[joint]) + 2, int(m.joint_qd_start.numpy()[joint]) + 2
    state = {"k": 0}

    def write(s0, s1, ctrl, c, dt):
        state["k"] += 1
        z = mod.KNIFE_Z - mod.SPEED * dt * state["k"]
        for arr, i, val in ((s1.joint_q, q, z), (s1.joint_qd, qd, -mod.SPEED)):
            a = arr.numpy()
            a[i] = val
            arr.assign(a)
        bq, bqd = s1.body_q.numpy(), s1.body_qd.numpy()
        bq[body, 2], bqd[body], bqd[body, 2] = z, 0.0, -mod.SPEED
        s1.body_q.assign(bq)
        s1.body_qd.assign(bqd)

    eng.add_step_observer(write)
    return state


def _masks(frame: Frame, name: str = "agar_block") -> tuple[np.ndarray, np.ndarray]:
    return frame.topology.mesh_tet_active[name].copy(), frame.topology.mesh_vertex_active[name].copy()


@pytest.mark.timeout(300)
def test_virtual_node_split_moves_the_version_and_masks_in_the_cut_world_only(tmp_path):
    mod, sim = _vnode_sim(tmp_path)
    eng, cutter = sim.engine, sim.engine.topology.cutter
    exp = FrameExporter(sim, [0, 1])
    f0, f1 = _poll(exp)
    assert f0.topology_version == f1.topology_version and f0.topology and f1.topology
    assert (~f1.topology.mesh_tet_active["agar_block"]).any()                           # dormant fragments
    assert not f1.topology.mesh_vertex_active["agar_block"].all()                       # dormant copy vertices
    tets0, verts0 = _masks(f1)
    v0, v1 = f0.topology_version, f1.topology_version
    knife = _lower_knife_in_world(sim, mod, 1)
    for _ in range(12):
        eng.step_period()
    split = cutter.split.numpy() == 1
    world = cutter.tets.world.numpy()
    assert split[world == 1].any() and not split[world == 0].any()                      # one world only
    g0, g1 = _poll(exp)
    assert g0.topology_version == v0 and g0.topology is None
    assert g1.topology_version != v1 and g1.topology is not None
    tets1, verts1 = _masks(g1)
    rows = exp.plan.tet_rows[1]
    cut = np.flatnonzero(np.isin(rows, cutter.copies.tet[split]))
    assert len(cut) > 0 and not tets1[cut].any() and tets0[cut].all()                  # the whole tets retired
    frags = np.flatnonzero(np.isin(rows, cutter.copies.copy_tet[split].ravel()))
    assert tets1[frags].all() and not tets0[frags].any()                                # fragments woke up
    assert (tets1 != tets0).sum() == len(cut) + len(frags)                              # and nothing else changed
    assert (verts1 & ~verts0).any() and not (verts0 & ~verts1).any()                    # copy vertices gained ACTIVE
    sim.reset()
    knife["k"] = 0
    h0, h1 = _poll(exp)
    assert (h0.topology_version, h1.topology_version) == (v0, v1)
    assert h0.topology is None and h1.topology is not None                              # back is a change too
    tets2, verts2 = _masks(h1)
    assert np.array_equal(tets2, tets0) and np.array_equal(verts2, verts0)


# --- 11. the version formula -------------------------------------------------------------------------------------


def test_version_formula_has_no_cancellation():
    from alloy.export.exporter import version

    def host(broken: int, retired: int, dead: int) -> dict:
        live = np.ones((1, 7), np.int8)
        live[0, :dead] = 0
        return {"counts": np.array([[broken, retired]], np.int32), "cut_live": live}

    assert version(host(3, 2, 4), 0) == 3 + 2 * 2**20 + 4 * 2**40
    assert version({"counts": np.zeros((1, 2), np.int32), "cut_live": np.zeros((1, 0), np.int8)}, 0) == 0
    base = version(host(3, 2, 4), 0)
    assert version(host(4, 2, 3), 0) != base                      # one more broken bond, one fewer dead tet
    assert version(host(4, 1, 4), 0) != base                      # one more bond, one fewer body
    assert version(host(4, 2, 3), 0) == base + 1 - 2**40


def test_a_folded_split_still_moves_the_version():
    """WP10: a split whose one side is under FOLD wakes one fragment only; T dies all the same. The version counts
    dead cuttable whole tets only (T), so it rises by 2**40 either way; the fragments stay in the draw mask."""
    from alloy.export import kernels as K
    from alloy.export.exporter import version

    mat = wp.array(np.array([[0.0, 0.0, 0.0], [1e4, 2e4, 0.1], [0.0, 0.0, 0.0]], np.float32), dtype=wp.float32,
                   device="cpu")                                    # T dead, T+ woke, T- folded (still dormant)
    cut = wp.array(np.array([[0, -1]], np.int32), dtype=wp.int32, device="cpu")   # T, then padding
    draw = wp.array(np.array([[0, 1, 2]], np.int32), dtype=wp.int32, device="cpu")
    cut_live = wp.zeros((1, 2), dtype=wp.int8, device="cpu")
    tet_live = wp.zeros((1, 3), dtype=wp.int8, device="cpu")
    wp.launch(K.tet_live, dim=(1, 2), inputs=[mat, cut], outputs=[cut_live], device="cpu")
    wp.launch(K.tet_live, dim=(1, 3), inputs=[mat, draw], outputs=[tet_live], device="cpu")
    assert cut_live.numpy().tolist() == [[0, 1]] and tet_live.numpy().tolist() == [[0, 1, 0]]
    before = {"counts": np.zeros((1, 2), np.int32), "cut_live": np.ones((1, 2), np.int8)}
    after = {"counts": np.zeros((1, 2), np.int32), "cut_live": cut_live.numpy()}
    assert version(after, 0) - version(before, 0) == 2**40
