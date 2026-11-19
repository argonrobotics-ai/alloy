# SPDX-License-Identifier: Apache-2.0
"""VBD self contact leaves other entries' particles out (alloy.coupling.liquid_cloth_selfcontact).

Root cause: the VBD entry's view keeps the liquid's particles, and Newton's
vertex-triangle self-contact detection paired them with the cloth's triangles. SPH rests water ~1.15 mm above the
cloth, inside VBD's 2 mm margin, so the penalty pushed the cloth off particles that VBD does not move (16,780 N on a
0.28 g vertex). The world went INVALID 0.02-0.05 s after the stream landed.

Laws (CPU, public API, one world; the rim scene: a 0.2 m cotton_woven sheet resting on the rim of a static
open box, a 20 ml pour at 0.5 m/s whose stream reaches the sheet at ~0.08 s):
- scope: the filter installs on the coupled VBD entry, with the liquid's particles as its foreign set, and not on a
  cloth-only scene; the masked positions keep the entry's own particles bit for bit;
- before / after: with the old detection (``ALLOY_VBD_FOREIGN_SELF_CONTACT=off``) the world is INVALID before
  t = 0.2 s (measured: 0.125 s, ENERGY_JUMP; the bound leaves 0.075 s). With the filter it stays OK to 0.3 s, and
  no particle goes through the sheet (the crossing meter, bound 0: a sheet is a closed surface);
- mechanism: across the pour, VBD's self-contact detection lists no pair between a liquid particle and a sheet
  triangle with the filter (bound 0: the liquid is no part of the cloth). With the old detection it lists them as
  the stream lands (50-60 pairs, from 0.08 s), which is what failed.
Measured, not a law: with the filter, self contact on and off give sheets within 0.64 mm over 0.3 s. With self contact
on, the sheet steps differently even with no self pair. Two candidate causes, neither verified: Newton's rigid-soft
DAT shares the self-contact motion reference at the rim, and the self-contact DAT truncation of each vertex's motion
between detections is on whenever self contact is.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy._vendor.newton._src.solvers.vbd.solver_vbd import SolverVBD
from alloy.api._liquid import all_solvers
from alloy.coupling import liquid_cloth_meter as meter
from alloy.coupling import liquid_cloth_selfcontact as lcs

DT = 0.005
SPACING = 4.0e-3


def _scene(self_contact: bool = True, liquid: bool = True, pour_x: float = 0.0) -> alloy.Scene:
    scene = alloy.Scene(dt=DT)
    scene.add_ground()
    scene.add_open_box("frame", inner=(0.16, 0.16, 0.10), wall=0.03, pos=(0, 0, 0))
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.1325), resolution=11, material="cotton_woven",
                    thickness=0.002, self_contact=self_contact)
    if liquid:
        scene.add_pour("water", 20e-6, pos=(pour_x, 0, 0.18), speed=0.5)
    return scene


def _vbd(sim: alloy.Simulation) -> SolverVBD:
    """The VBD entry's solver of a built simulation (one per scene here)."""
    found = [s for s in all_solvers(sim.engine) if isinstance(s, SolverVBD)]
    assert len(found) == 1, f"expected one VBD solver in the build, found {len(found)}"
    return found[0]


def _run(scene: alloy.Scene, seconds: float) -> dict:
    """Step ``seconds`` one control period at a time; stop 2 periods after the first INVALID."""
    sim = alloy.build(scene, device="cpu")
    pts, sheet, act, t_bad = [], [], [], None
    for _ in range(int(round(seconds / DT))):
        sim.step(1)
        s = sim.state(["sheet", "water"], world=0)
        pts.append(np.asarray(s["water.points"], np.float32))
        sheet.append(np.asarray(s["sheet.points"], np.float32))
        act.append(np.asarray(s["water.active"]).astype(bool))
        if t_bad is None and not sim.check()["ok"]:
            t_bad = sim.t
        if t_bad is not None and sim.t > t_bad + 2 * DT:
            break
    return {"sim": sim, "pts": np.stack(pts), "sheet": np.stack(sheet), "active": np.stack(act),
            "faces": np.asarray(sim.faces("sheet")), "t_invalid": t_bad, "ok": bool(sim.check()["ok"])}


def test_only_contact_free_entries_are_foreign():
    """The vbd entry is told the particle ids of the contact-free entries only (SPH, MPM), never those
    of another VBD-type entry (a film sheet is its own entry), as compact id ranges."""
    from types import SimpleNamespace as NS
    specs = [NS(capability="film", particles=(0, 1, 2), solver_kwargs={}),
             NS(capability="mpm", particles=(10, 11, 12, 20), solver_kwargs={}),
             NS(capability="sph", particles=(13, 14, 30), solver_kwargs={}),
             NS(capability="vbd", particles=(3, 4, 5), solver_kwargs={})]
    lcs.attach_contact_free(specs)
    assert specs[3].solver_kwargs[lcs.FREE_KEY] == ((10, 15), (20, 21), (30, 31))
    assert all(lcs.FREE_KEY not in sp.solver_kwargs for sp in specs[:3])
    only_film = [NS(capability="film", particles=(0, 1), solver_kwargs={}),
                 NS(capability="vbd", particles=(2,), solver_kwargs={})]
    lcs.attach_contact_free(only_film)
    assert lcs.FREE_KEY not in only_film[1].solver_kwargs


@pytest.mark.timeout(1800)
def test_filter_scope_and_masked_positions(monkeypatch):
    """The filter sits on the coupled VBD entry with the liquid as its foreign set and leaves own particles exact;
    a cloth-only scene, and the A/B switch, get none."""
    monkeypatch.delenv(lcs.ENV, raising=False)
    sim = alloy.build(_scene(), device="cpu")
    solver = _vbd(sim)
    filt = getattr(solver, "_alloy_foreign_filter", None)
    assert filt is not None, "no foreign-particle filter on the cloth + liquid VBD entry"
    m = solver.model
    q = m.particle_q
    own = (m.particle_flags.numpy() & 1) != 0           # ParticleFlags.ACTIVE
    on_mesh = np.zeros(m.particle_count, bool)
    on_mesh[m.tri_indices.numpy().ravel()] = True
    assert not (~own & on_mesh).any(), "a non-ACTIVE particle lies on the sheet's triangles"
    assert filt.foreign == int((~own).sum()) > 0
    masked = filt.masked(q).numpy()
    np.testing.assert_array_equal(masked[own], q.numpy()[own])
    assert np.all(masked[~own] == np.float32(lcs.FAR))
    assert solver.trimesh_collision_detector.refit == filt.refit

    dry = alloy.build(_scene(liquid=False), device="cpu")
    assert getattr(_vbd(dry), "_alloy_foreign_filter", None) is None, "a cloth-only scene must keep stock detection"
    monkeypatch.setenv(lcs.ENV, "off")
    off = alloy.build(_scene(), device="cpu")
    assert getattr(_vbd(off), "_alloy_foreign_filter", None) is None


@pytest.mark.law
@pytest.mark.timeout(3600)
def test_pour_on_cotton_before_and_after(monkeypatch):
    """Before (old detection): INVALID before 0.2 s. After: OK to 0.3 s and through == 0 (bound 0, module doc)."""
    monkeypatch.setenv(lcs.ENV, "off")
    before = _run(_scene(), 0.2)
    assert before["t_invalid"] is not None and before["t_invalid"] < 0.2, (
        f"the old detection no longer fails (INVALID at {before['t_invalid']}): re-check the root cause")
    monkeypatch.delenv(lcs.ENV)
    after = _run(_scene(), 0.3)
    assert after["ok"], f"INVALID at t = {after['t_invalid']} with the filter"
    assert after["active"][-1].sum() == after["pts"].shape[1], "the pour did not come out whole"
    c = meter.crossings(after["pts"], after["sheet"], after["faces"], after["active"], latch=6 * SPACING,
                        tol=0.25 * SPACING)
    assert c.tracked > 100, "the stream never reached the sheet: the rig is off"
    assert c.through == 0, f"{c.through} of {c.tracked} particles went through the sheet"


def _foreign_pairs_per_detection(monkeypatch, seconds: float) -> list[int]:
    """Run the pour; after every vertex-triangle detection of the VBD entry, the pairs whose vertex is a particle
    that is not ACTIVE in the entry's view (the liquid). Host reads after each detection: CPU only."""
    sim = alloy.build(_scene(), device="cpu")
    solver = _vbd(sim)
    det = solver.trimesh_collision_detector
    foreign = (solver.model.particle_flags.numpy() & 1) == 0           # ParticleFlags.ACTIVE
    counts: list[int] = []
    detect = det.vertex_triangle_collision_detection

    def watched(*args, **kwargs):
        detect(*args, **kwargs)
        counts.append(int(det.vertex_colliding_triangles_count.numpy()[foreign].sum()))

    monkeypatch.setattr(det, "vertex_triangle_collision_detection", watched)
    for _ in range(int(round(seconds / DT))):
        sim.step(1)
        if not sim.check()["ok"]:
            break
    return counts


@pytest.mark.law
@pytest.mark.timeout(3600)
def test_self_contact_lists_no_liquid_pair(monkeypatch):
    """Mechanism (module doc): 0 liquid-vertex / sheet-triangle pairs at every detection with the filter; more than
    0 with the old detection once the stream lands (before t = 0.15 s; measured from 0.08 s)."""
    monkeypatch.setenv(lcs.ENV, "off")
    before = _foreign_pairs_per_detection(monkeypatch, 0.15)
    assert max(before) > 0, "the old detection lists no liquid pair: the root cause is not reproduced"
    monkeypatch.delenv(lcs.ENV)
    after = _foreign_pairs_per_detection(monkeypatch, 0.15)
    assert len(after) > 50, "too few detections recorded: the rig is off"
    assert max(after) == 0, f"{max(after)} liquid pairs in one detection with the filter"


def _sheet_frames(scene: alloy.Scene, seconds: float) -> np.ndarray:
    sim = alloy.build(scene, device="cpu")
    out = []
    for _ in range(int(round(seconds / DT))):
        sim.step(1)
        out.append(np.asarray(sim.state(["sheet"], world=0)["sheet.points"]).copy())
    return np.stack(out)


@pytest.mark.law
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("case", ["cloth_only", "pour_beside"])
def test_filter_changes_nothing_without_liquid_contact(monkeypatch, case):
    """No side effect: where no liquid particle comes within the
    self-contact margin of the cloth, the sheet is the same, bit for bit, with the filter and with the old
    detection (ALLOY_VBD_FOREIGN_SELF_CONTACT=off) over 0.3 s. Cases: cloth only, where the filter does not
    install; and the coupled scene with the pour landing beside the box, where the filter installs and masks the
    liquid. The CPU repeats bit for bit, so the bound is exact equality."""
    scene = (lambda: _scene(liquid=False)) if case == "cloth_only" else (lambda: _scene(pour_x=0.25))
    monkeypatch.delenv(lcs.ENV, raising=False)
    new = _sheet_frames(scene(), 0.3)
    monkeypatch.setenv(lcs.ENV, "off")
    old = _sheet_frames(scene(), 0.3)
    np.testing.assert_array_equal(new, old)


def _stack_scene() -> alloy.Scene:
    """Two 0.2 m cotton sheets stacked 4 mm apart on a static table (one VBD entry: their contact is self contact),
    and a 20 ml pour that lands on the ground beside the table, so the filter installs but no liquid nears the cloth."""
    scene = alloy.Scene(dt=DT)
    scene.add_ground()
    scene.add_box("table", size=(0.3, 0.3, 0.1), pos=(0, 0, 0.05), static=True)
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.1025), resolution=11, material="cotton_woven",
                    thickness=0.002)
    scene.add_cloth("sheet2", size=(0.2, 0.2), pos=(0.01, 0.01, 0.1065), resolution=11, material="cotton_woven",
                    thickness=0.002)
    scene.add_pour("water", 20e-6, pos=(0.3, 0, 0.18), speed=0.5)
    return scene


def _stack_run(monkeypatch, seconds: float) -> tuple[np.ndarray, int, int, list]:
    """Frames of both sheets, the most own (cloth-cloth) pairs in one detection, the filter's foreign count, and per
    detection the own rows of the detector's results (pair counts, nearest distances: vertex-triangle and
    triangle-vertex; edge-edge)."""
    sim = alloy.build(_stack_scene(), device="cpu")
    solver = _vbd(sim)
    det = solver.trimesh_collision_detector
    own = (solver.model.particle_flags.numpy() & 1) != 0           # ParticleFlags.ACTIVE
    pairs: list[int] = []
    results: list = []
    detect, detect_ee = det.vertex_triangle_collision_detection, det.edge_edge_collision_detection

    def watched(*args, **kwargs):
        detect(*args, **kwargs)
        pairs.append(int(det.vertex_colliding_triangles_count.numpy()[own].sum()))
        results.append((det.vertex_colliding_triangles_count.numpy()[own].copy(),
                        det.vertex_colliding_triangles_min_dist.numpy()[own].copy(),
                        det.triangle_colliding_vertices_min_dist.numpy().copy()))

    def watched_ee(*args, **kwargs):
        detect_ee(*args, **kwargs)
        results.append((det.edge_colliding_edges_min_dist.numpy().copy(),))

    monkeypatch.setattr(det, "vertex_triangle_collision_detection", watched)
    monkeypatch.setattr(det, "edge_edge_collision_detection", watched_ee)
    frames = []
    for _ in range(int(round(seconds / DT))):
        sim.step(1)
        s = sim.state(["sheet", "sheet2"], world=0)
        frames.append(np.concatenate([np.asarray(s["sheet.points"]), np.asarray(s["sheet2.points"])]).copy())
    filt = getattr(solver, "_alloy_foreign_filter", None)
    return np.stack(frames), max(pairs), 0 if filt is None else filt.foreign, results


@pytest.mark.law
@pytest.mark.timeout(3600)
def test_filter_keeps_cloth_self_contact_bit_for_bit(monkeypatch):
    """No side effect on the cloth's own contact: two stacked sheets press on
    each other (cloth-cloth self-contact pairs, asserted > 0 so the case is not vacuous) while a pour lands beside the
    table (the filter installs, foreign count > 0). At every detection the cloth's own results (pair counts and
    nearest distances) and, over 0.3 s, the sheets' frames equal bit for bit those with the old detection
    (ALLOY_VBD_FOREIGN_SELF_CONTACT=off): the filter changes only what liquid particles see. Bound: exact equality
    (the CPU repeats bit for bit). The frame comparison alone has no power while the pair set is unchanged: detection
    positions feed only pair lists and distances, so a filter that moved the cloth's own detection positions (+1e-7 m,
    and +1 mm) left the frames identical; the per-detection comparison is what fails then."""
    monkeypatch.delenv(lcs.ENV, raising=False)
    new, pairs_new, foreign, det_new = _stack_run(monkeypatch, 0.3)
    assert foreign > 0, "the filter did not install: the case would not test it"
    assert pairs_new > 0, "the sheets never touched: no cloth self-contact pair to keep"
    monkeypatch.setenv(lcs.ENV, "off")
    old, pairs_old, _, det_old = _stack_run(monkeypatch, 0.3)
    assert pairs_new == pairs_old
    assert len(det_new) == len(det_old), "a different number of detections"
    for k, (a, b) in enumerate(zip(det_new, det_old)):      # what detection saw of the cloth, exactly
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y, err_msg=f"detection {k}: the cloth's own results differ")
    np.testing.assert_array_equal(new, old)
