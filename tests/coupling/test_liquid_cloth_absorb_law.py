# SPDX-License-Identifier: Apache-2.0
"""Cloth takes up liquid (opt-in): alloy/solvers/sph/cloth_absorb.py, alloy/coupling/liquid_cloth_absorb.py.

``add_cloth(absorb={"capacity": kg/m^2, "rate": kg/m^2/s})`` makes a sheet take up liquid resting on it, whole particles
into a per-vertex water mass up to capacity x area. The values here are test inputs, not fabric data.

Laws (CPU, public API, one world; bounds say whether they are exact, physical or measured-near):
- the request: films refuse it, bad values refuse with a fix, the card check runs on the library path, a scene's JSON
  keeps it, and without it nothing installs; a sheet too fine to take a particle warns at build, the laws' rig does
  not, and a dry object gets no water key;
- books: poured = liquid + absorbed in whole particles, water = absorbed mass exactly, no vertex past capacity, after
  every exchange while coupling passes re-run the uptake (A7);
- masses: every cloth vertex weighs dry + water and an absorbed particle 0 in the model health's energy book reads,
  so the water counts once; a planted double count fails that law;
- A5: absorb on a sheet the liquid never reaches changes nothing, bit for bit; the CPU repeats bit for bit, also
  after sim.reset();
- A1 capacity (the tank: standing water fills the sheet to capacity), A2 rate (uptake at q x A_wet within 20 %),
  A3 weight (a wet sheet sags like its dry twin within 5 %), A4 in a pour (the water reaches vertices the liquid never
  touched);
- on a GPU: identical worlds and repeat builds take up the same particles, and the CPU within 5 %.
The kernel laws (wicking at 4 D dt, A6 the exchange merge, the one-step gate) are in test_liquid_cloth_absorb_kernels.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import alloy
from alloy.api._liquid import liquid_solvers

ABSORB = {"capacity": 0.6, "rate": 2.0}       # test inputs [kg/m^2, kg/m^2/s]
DT = 0.005


def _rim(absorb: dict | None = ABSORB, pour_x: float | None = 0.0, material: str = "cotton_woven",
         resolution: int = 11) -> alloy.Scene:
    """The rim scene: a 0.2 m sheet resting on the rim of a static open box, a 20 ml pour at 0.5 m/s (none when
    ``pour_x`` is None)."""
    s = alloy.Scene(dt=DT)
    s.add_ground()
    s.add_open_box("frame", inner=(0.16, 0.16, 0.10), wall=0.03, pos=(0, 0, 0))
    s.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.1325), resolution=resolution, material=material,
                thickness=0.002, absorb=absorb)
    if pour_x is not None:
        s.add_pour("water", 20e-6, pos=(pour_x, 0, 0.18), speed=0.5)
    return s


@pytest.mark.parametrize("bad", [{"capacity": 0.0, "rate": 1.0}, {"capacity": 0.5}, {"capacity": 0.5, "rate": 1.0,
                                                                                         "speed": 1.0},
                                 {"capacity": 0.5, "rate": -1.0}, {"capacity": 0.5, "rate": 1.0, "wicking": -1e-6},
                                 0.5])
def test_bad_absorb_values_refuse_with_a_fix(bad):
    """A capacity or rate that is not > 0, a negative wicking, a missing or unknown key: SceneError with a fix."""
    with pytest.raises(alloy.SceneError) as e:
        _rim(absorb=bad)
    assert e.value.field == "absorb" and "capacity" in e.value.fix


def test_a_film_sheet_refuses_absorb():
    with pytest.raises(alloy.SceneError, match="film and paper sheets take up no liquid"):
        _rim(material="film_ldpe")


def test_scene_json_keeps_absorb():
    scene = _rim()
    again = alloy.Scene.from_dict(scene.to_dict())
    feats = again.items["sheet"]["card_extra"]["features"]
    assert {k: v["value"] for k, v in feats["absorb"].items()} == ABSORB
    assert "absorb" not in _rim(absorb=None).items["sheet"]["card_extra"]["features"]


def test_card_check_runs_on_the_library_path(tmp_path: Path):
    """A library object card with a bad absorb block is flagged by CardLibrary.object_issues (the path every build
    and importer takes): capacity and rate must be > 0, wicking >= 0."""
    from alloy.cards.library import CardLibrary, default_root

    card = yaml.safe_load((default_root() / "objects" / "cloth_cotton_sheet.yaml").read_text())
    card["name"] = "wet_sheet_bad"
    card["features"]["absorb"] = {"capacity": {"value": 0.0, "source": "model", "note": "test"},
                                  "rate": {"value": -1.0, "source": "model", "note": "test"},
                                  "wicking": {"value": -1e-6, "source": "model", "note": "test"}}
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / "wet_sheet_bad.yaml").write_text(yaml.safe_dump(card))
    lib = CardLibrary(roots=[tmp_path, default_root()])
    flagged = {i.field for i in lib.object_issues("wet_sheet_bad") if i.level == "error"}
    assert {"features.absorb.capacity", "features.absorb.rate", "features.absorb.wicking"} <= flagged


@pytest.mark.timeout(1800)
def test_nothing_installs_without_absorb():
    """absorb left out: no uptake on the liquid's solver and no water / absorbed state keys."""
    sim = alloy.build(_rim(absorb=None), device="cpu")
    assert all(getattr(s, "cloth_absorb", None) is None for s in liquid_solvers(sim.engine))
    sim.step(1)
    st = sim.state(["sheet", "water"], world=0)
    assert "sheet.water" not in st and "water.absorbed" not in st


TOO_FINE = "hold under a third of a liquid particle"


@pytest.mark.timeout(1800)
def test_a_sheet_too_fine_to_take_a_particle_warns_at_build(caplog):
    """Resolution 40 (5 mm cells): capacity x area is 5-20 mg per vertex and a particle 64 mg, so no vertex has room for
    the largest share of a particle (at least a third) and the sheet would take up nothing. The build warns once,
    naming the sheet and the fix, and the install line counts no vertex that can take a particle. Building with the
    capacity the warning names silences it. Build only."""
    import re

    with caplog.at_level("INFO", logger="alloy.coupling"):
        alloy.build(_rim(resolution=40), device="cpu")
    msgs = [r.getMessage() for r in caplog.records]
    warn = [m for m in msgs if TOO_FINE in m]
    assert len(warn) == 1 and "1681 of 1681 vertices of 'sheet'" in warn[0] and "liquid_accuracy" in warn[0], msgs
    assert any("0 of 1681 vertices of absorbing sheets can take up a particle" in m for m in msgs), msgs
    named = float(re.search(r"capacity of at least ([0-9.e+-]+) kg/m\^2", warn[0]).group(1))
    caplog.clear()
    with caplog.at_level("INFO", logger="alloy.coupling"):
        alloy.build(_rim(absorb={**ABSORB, "capacity": named}, resolution=40), device="cpu")
    assert not any(TOO_FINE in r.getMessage() for r in caplog.records), (named, caplog.records)


@pytest.mark.timeout(1800)
def test_the_laws_sheet_does_not_warn_and_a_dry_sheet_gets_no_water_key(caplog):
    """The laws' rig (resolution 11, capacity 0.6) has room for a particle's share at every vertex: no warning. A dry
    sheet beside it neither counts among the absorbing vertices nor gets a water key. Build only."""
    s = _rim()
    s.add_cloth("dry", size=(0.1, 0.1), pos=(0.3, 0, 0.005), resolution=11, thickness=0.002)
    with caplog.at_level("INFO", logger="alloy.coupling"):
        sim = alloy.build(s, device="cpu")
    msgs = [r.getMessage() for r in caplog.records]
    assert not any(TOO_FINE in m for m in msgs), msgs
    assert any("144 of 144 vertices of absorbing sheets can take up a particle" in m for m in msgs), msgs
    st = sim.state(["sheet", "dry"], world=0)
    assert "sheet.water" in st and "dry.water" not in st


# ------------------------------------------------------------------------------------------------ the sim laws
def _parts(sim: alloy.Simulation):
    """(SPH solver, its ClothAbsorb, the coupled solver's ClothAbsorbCoupling) of a built simulation."""
    from alloy.coupling.liquid_cloth_absorb import ClothAbsorbCoupling

    sph = liquid_solvers(sim.engine)[0]
    seen, todo, cpl = set(), [sim.engine], None
    while todo and cpl is None:
        o = todo.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        for v in list(getattr(o, "__dict__", {}).values()):
            if isinstance(v, ClothAbsorbCoupling):
                cpl = v
                break
            if hasattr(v, "__dict__") and type(v).__module__.startswith("alloy"):
                todo.append(v)
    return sph, sph.cloth_absorb, cpl


def _books(sim: alloy.Simulation) -> dict:
    """The uptake books of world 0 from the engine's arrays (exact integers where the code keeps them so)."""
    from alloy.solvers.sph.cloth_absorb import ABSORBED, W_SCALE_HOST

    sph, ca, cpl = _parts(sim)
    phase = sph.layout.phase.numpy()
    liquid = phase != -1
    m = sph.model.particle_mass.numpy()
    st = sim.state(["sheet", "water"], world=0)
    return {"absorbed": int((phase == ABSORBED).sum()), "absorbed_key": int(st["water.absorbed"]),
            "count": int(ca.soaked_count.numpy()[0]), "soaked_fx": int(ca.soaked_mass.numpy()[0]),
            "water_fx": int(ca.water.numpy().sum()),
            "expect_fx": int(np.round(m[phase == ABSORBED].astype(np.float64) * W_SCALE_HOST).sum()),
            "over_cap": int((ca.water.numpy() - ca.cap.numpy()).max()),
            "slots": int(liquid.sum()), "by_phase": {int(k): int(v) for k, v in zip(*np.unique(phase[liquid],
                                                                                                return_counts=True))},
            "water_key_kg": float(np.sum(st["sheet.water"]))}


def _rest_areas(sim: alloy.Simulation) -> tuple[np.ndarray, np.ndarray]:
    """(global ids, rest area per sheet vertex [m^2]) computed here from the sheet's start mesh: a third of each
    incident triangle's area (independent of the engine's own areas)."""
    x = np.asarray(sim.state(["sheet"], world=0)["sheet.points"], np.float64)
    f = np.asarray(sim.faces("sheet"))
    a = 0.5 * np.linalg.norm(np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]]), axis=1)
    area = np.zeros(len(x))
    for c in range(3):
        np.add.at(area, f[:, c], a / 3.0)
    r = sim.index.particles["sheet"]
    return np.arange(r.start, r.start + len(r)), area


def _per_sheet_vertex(sim: alloy.Simulation, values: np.ndarray) -> np.ndarray:
    """An engine array per compact soft vertex, in sheet vertex order (0 for a sheet vertex not among them)."""
    from alloy.coupling.liquid_cloth_absorb import solver_l2g

    sph, ca, _ = _parts(sim)
    vid = solver_l2g(sph, ca.soft_walls.vid.numpy())
    ids, _ = _rest_areas(sim)
    pos = {int(g): k for k, g in enumerate(vid)}
    return np.array([values[pos[int(g)]] if int(g) in pos else 0 for g in ids])


@pytest.mark.timeout(1800)
def test_capacity_is_capacity_times_rest_area():
    """Each sheet vertex holds at most capacity x its share of the sheet's rest area: checked against areas computed
    here from the start mesh, per vertex (within the fixed-point and float32 rounding) and in total (0.6 kg/m^2 x
    0.04 m^2 = 0.024 kg)."""
    from alloy.solvers.sph.cloth_absorb import W_SCALE_HOST

    sim = alloy.build(_rim(), device="cpu")
    _, area = _rest_areas(sim)
    cap = _per_sheet_vertex(sim, _parts(sim)[1].cap.numpy()) / W_SCALE_HOST
    np.testing.assert_allclose(cap, ABSORB["capacity"] * area, rtol=1e-5, atol=2e-12)
    assert abs(cap.sum() - 0.024) < 1e-7


@pytest.fixture(scope="module")
def wet_run():
    """The rim pour with absorb on, 0.5 s, books read after every control step."""
    sim = alloy.build(_rim(), device="cpu")
    rows = []
    for _ in range(int(round(0.5 / DT))):
        sim.step(1)
        rows.append(_books(sim))
    return sim, rows


@pytest.mark.law
@pytest.mark.timeout(3600)
def test_books_close_exactly_in_particles(wet_run):
    """poured = liquid + absorbed in whole particles; the absorbed count, the per-world counter and the state key
    agree; the water held equals the absorbed particles' mass exactly (int64 fixed point); no vertex holds more than
    its capacity. The rig must take up particles (more than 50). A7: coupling passes re-run uptake from one start
    (more acceptances than absorbed particles across all passes), and the books still close exactly after every
    exchange, so only the pass that sticks counts."""
    sim, rows = wet_run
    for r in rows:
        assert r["absorbed"] == r["count"] == r["absorbed_key"], r
        assert r["water_fx"] == r["soaked_fx"] == r["expect_fx"], r
        assert r["over_cap"] <= 0, r
        assert sum(r["by_phase"].values()) == r["slots"], r
        assert abs(r["water_key_kg"] - r["water_fx"] / 2.0 ** 40) <= 1e-9 + 1e-6 * r["water_key_kg"], r
    assert rows[-1]["absorbed"] > 50, f"the sheet took up only {rows[-1]['absorbed']} particles: the rig is off"
    diag = _parts(sim)[1].diagnostics()
    assert diag["accepted"] > rows[-1]["absorbed"], "no re-run pass took up a particle: A7 is not exercised"
    assert sim.check()["ok"], sim.check()["problems"]


def _mass_error(sim: alloy.Simulation) -> float:
    """Largest |parent mass - its expected value| over the cloth vertices and the liquid [kg]: a cloth vertex's dry
    mass plus its water, an absorbed particle 0, any other liquid particle its build mass. The cloth view too."""
    from alloy.solvers.sph.cloth_absorb import ABSORBED

    sph, ca, cpl = _parts(sim)
    assert cpl is not None, "no exchange-end coupling on the coupled solver"
    par = cpl.model.particle_mass.numpy().astype(np.float64)
    err = 0.0
    for c in cpl.maps:
        vtx, g = c.vtx.numpy(), c.es.src_global.numpy()
        want = c.dry.m.numpy().astype(np.float64) + np.where(vtx >= 0, ca.water_kg()[np.maximum(vtx, 0)], 0.0)
        view = c.src_view.particle_mass.numpy()[c.es.src_local.numpy()].astype(np.float64)
        err = max(err, float(np.abs(par[g] - want).max()), float(np.abs(view - want).max()))
    phase = sph.layout.phase.numpy()
    liquid = (phase != -1) & (cpl.l2g.numpy()[: len(phase)] >= 0)
    g = cpl.l2g.numpy()[: len(phase)][liquid]
    want = np.where(phase[liquid] == ABSORBED, 0.0, cpl.m0.numpy()[liquid].astype(np.float64))
    return max(err, float(np.abs(par[g] - want).max()))


@pytest.mark.law
@pytest.mark.timeout(3600)
def test_masses_follow_the_water(wet_run):
    """After the exchanges, every cloth vertex weighs its dry mass plus its water (parent model and the cloth's view)
    and every absorbed particle's parent mass is 0, so health's energy book (it reads the parent's masses) counts the
    water once, on the cloth. Bound: float32 rounding of m_dry + w (1e-6 relative of a 0.3 g vertex: 3e-10 kg)."""
    sim, rows = wet_run
    assert rows[-1]["absorbed"] > 50
    assert _mass_error(sim) < 3e-10


@pytest.mark.law
@pytest.mark.timeout(3600)
def test_planted_double_count_is_caught():
    """Mutant: the exchange end leaves the absorbed particles' parent mass in place (the double count the design
    guards against). The mass law must catch it: the error is then a whole particle's mass (64 mg at 4 mm)."""
    sim = alloy.build(_rim(), device="cpu")
    sim.step(1)                                     # the coupling may be built at the first exchange
    _, _, cpl = _parts(sim)
    assert cpl is not None
    cpl.n = 0                                       # planted: skip the liquid mass writes
    for _ in range(int(round(0.3 / DT)) - 1):
        sim.step(1)
    assert _books(sim)["absorbed"] > 0
    assert _mass_error(sim) > 1e-5, "the planted double count went unseen by the mass law"


@pytest.mark.law
@pytest.mark.timeout(3600)
def test_absorb_on_a_dry_sheet_changes_nothing():
    """absorb on, but the pour lands beside the box: no particle reaches the sheet, so every frame of the sheet and the
    water equals the run without absorb, bit for bit (the dry masses are written back exactly)."""
    def frames(absorb):
        sim = alloy.build(_rim(absorb=absorb, pour_x=0.25), device="cpu")
        out = []
        for _ in range(int(round(0.3 / DT))):
            sim.step(1)
            st = sim.state(["sheet", "water"], world=0)
            out.append(np.concatenate([np.asarray(st["sheet.points"]).ravel(), np.asarray(st["water.points"]).ravel()]))
        return np.stack(out)

    np.testing.assert_array_equal(frames(ABSORB), frames(None))


@pytest.mark.law
@pytest.mark.timeout(3600)
def test_repeat_runs_and_reset_give_the_same_uptake(wet_run):
    """The CPU repeats bit for bit: a second build, and the first sim after sim.reset(), take up the same particles
    into the same water arrays as the first run (the lowest-index rule and integer books)."""
    sim, rows = wet_run
    _, ca, _ = _parts(sim)
    first = ca.water.numpy().copy()
    other = alloy.build(_rim(), device="cpu")
    for _ in range(len(rows)):
        other.step(1)
    np.testing.assert_array_equal(_parts(other)[1].water.numpy(), first)
    sim.reset()
    assert _books(sim)["absorbed"] == 0 and int(ca.water.numpy().sum()) == 0
    assert _mass_error(sim) < 3e-10, "sim.reset() left the cloth or the liquid at its wet masses"
    for _ in range(len(rows)):
        sim.step(1)
    np.testing.assert_array_equal(ca.water.numpy(), first)


# ------------------------------------------------------------------------------------- capacity, rate, weight, wicking
def _tank(absorb: dict) -> alloy.Scene:
    """The cloth-floor tank of the leak laws: a 0.2 m sheet pinched between a box rim (0.10 m opening) and a static
    frame 1.5 mm above it, so 60 ml poured inside the frame stands on the sheet."""
    wall, depth, gap, fh, fw, out = 0.04, 0.08, 1.5e-3, 0.06, 0.06, 0.22
    rim = depth + wall
    zc, o = rim + 0.002 + gap + fh / 2, out / 2 - fw / 2
    s = alloy.Scene(dt=DT)
    s.add_ground()
    s.add_open_box("box", inner=(0.10, 0.10, depth), wall=wall, pos=(0, 0, 0))
    s.add_cloth("sheet", size=(0.20, 0.20), pos=(0, 0, rim + 0.0025), resolution=14, material="cotton_woven",
                thickness=2e-3, absorb=absorb)
    for name, size, pos in (("bar_n", (out, fw, fh), (0, o, zc)), ("bar_s", (out, fw, fh), (0, -o, zc)),
                            ("bar_e", (fw, 0.10, fh), (o, 0, zc)), ("bar_w", (fw, 0.10, fh), (-o, 0, zc))):
        s.add_box(name, size=size, pos=pos, static=True)
    s.add_pour("water", 60e-6, pos=(0, 0, rim + 0.07), speed=0.5)
    return s


def _run_for(sim: alloy.Simulation, seconds: float, every: int = 10) -> list[tuple[float, float]]:
    """Step ``seconds``; every ``every`` periods (t, absorbed mass [kg] of world 0)."""
    _, ca, _ = _parts(sim)
    out = []
    for k in range(int(round(seconds / DT))):
        sim.step(1)
        if (k + 1) % every == 0:
            out.append((sim.t, float(ca.soaked()[0][0])))
    return out


@pytest.mark.law
@pytest.mark.timeout(7200)
def test_a1_standing_water_fills_the_sheet_to_capacity():
    """A1: where water stands on the sheet (the tank's opening), every vertex ends within one particle's mass of its
    capacity, and no vertex anywhere holds more (exact). Measured: 49 of 49 opening vertices, mean fill 0.95."""
    from alloy.solvers.sph.cloth_absorb import W_SCALE_HOST

    sim = alloy.build(_tank({"capacity": 0.6, "rate": 20.0}), device="cpu")
    _run_for(sim, 1.3)
    sph, ca, _ = _parts(sim)
    w, cap = ca.water.numpy(), ca.cap.numpy()
    x = sph.model.particle_q.numpy()[sph.soft.vid.numpy()]
    inner = (np.abs(x[:, 0]) < 0.045) & (np.abs(x[:, 1]) < 0.045) & (cap > 0)
    assert inner.sum() >= 25, "the opening holds too few sheet vertices: the rig is off"
    assert (w <= cap).all()
    short = (cap[inner] - w[inner]) / W_SCALE_HOST
    assert (short <= ca.burst / W_SCALE_HOST).all(), f"opening vertices short of capacity by up to {short.max():.3g} kg"
    assert sim.check()["ok"], sim.check()["problems"]


@pytest.mark.law
@pytest.mark.timeout(7200)
def test_a2_uptake_runs_at_the_rate_limit():
    """A2: with room to spare (capacity 5 kg/m^2) and water standing on the sheet:
    - no vertex has taken more than q x its rest area x the time since the build (exact: the allowance starts at 0 and
      fills at that rate), with areas computed here from the start mesh;
    - the absorbed mass grows at 70-100 % of q x A_wet over the hold after the pour lands (0.5-1.3 s), A_wet the rest
      area of the vertices that took water. Measured 0.86: vertices at the water's edge are wet only part of the
      time, so the whole wetted area takes less than its full rate."""
    from alloy.solvers.sph.cloth_absorb import W_SCALE_HOST

    q = 1.0
    sim = alloy.build(_tank({"capacity": 5.0, "rate": q}), device="cpu")
    _, area = _rest_areas(sim)
    rows = _run_for(sim, 1.3)
    sph, ca, _ = _parts(sim)
    w = _per_sheet_vertex(sim, ca.water.numpy()) / W_SCALE_HOST
    assert (w <= q * area * sim.t + 1e-9).all(), float((w - q * area * sim.t).max())
    a_wet = float(area[w > 0].sum())
    t = np.array([r[0] for r in rows])
    m = np.array([r[1] for r in rows])
    sel = t >= 0.5
    slope = float(np.polyfit(t[sel], m[sel], 1)[0])
    ratio = slope / (q * a_wet)
    print(f"A2: slope {slope * 1e3:.2f} g/s, q x A_wet {q * a_wet * 1e3:.2f} g/s, ratio {ratio:.3f}")
    assert a_wet > 0.005, f"water reached only {a_wet:.4f} m^2 of the sheet: the rig is off"
    assert 0.7 <= ratio <= 1.0, (slope, q * a_wet)
    assert sim.check()["ok"], sim.check()["problems"]


def _sheet_mass_twin(water: np.ndarray, seconds: float) -> np.ndarray:
    """The dry twin: the rim sheet without liquid or absorb, its vertex masses set to m_dry + water, settled."""
    from alloy._vendor.newton._src.solvers.vbd.solver_vbd import SolverVBD
    from alloy.api._liquid import all_solvers

    sim = alloy.build(_rim(absorb=None, pour_x=None), device="cpu")
    vbd = [s for s in all_solvers(sim.engine) if isinstance(s, SolverVBD)][0]
    m = vbd.model.particle_mass.numpy()
    assert len(m) == len(water)
    new = (m + water).astype(np.float32)
    vbd.model.particle_mass.assign(new)
    vbd.model.particle_inv_mass.assign(np.where(new > 0, 1.0 / np.where(new > 0, new, 1.0), 0.0).astype(np.float32))
    for _ in range(int(round(seconds / DT))):
        sim.step(1)
    assert sim.check()["ok"]
    return np.asarray(sim.state(["sheet"], world=0)["sheet.points"])


@pytest.mark.law
@pytest.mark.timeout(7200)
def test_a3_a_wet_sheet_sags_like_its_dry_twin():
    """A3: after the pour has been taken up or run off, the wet sheet sags like a dry twin whose vertex masses are its
    dry masses plus the water it holds: the deepest sag within 5 % of the twin's. Measured: 11.76 mm and 11.44 mm
    (2.8 %; 24 particles still on the wet sheet add a little load), largest height difference 0.35 mm."""
    sim = alloy.build(_rim({"capacity": 0.6, "rate": 20.0, "wicking": 2e-4}), device="cpu")
    _run_for(sim, 1.5)
    st = sim.state(["sheet"], world=0)
    water, wet = np.asarray(st["sheet.water"]), np.asarray(st["sheet.points"])
    assert water.sum() > 0.01, "the sheet took up too little water for the comparison"
    twin = _sheet_mass_twin(water, 1.5)
    top = 0.1325
    sag_wet, sag_twin = top - wet[:, 2].min(), top - twin[:, 2].min()
    assert abs(sag_wet / sag_twin - 1.0) <= 0.05, (sag_wet, sag_twin)
    assert sim.check()["ok"]


@pytest.mark.law
@pytest.mark.timeout(7200)
def test_a4_wicking_spreads_the_water_in_a_pour():
    """A4 in a run: with wicking the water reaches vertices the liquid never touched (measured 144 against 111 of
    144), the books still close, and no vertex goes past its capacity."""
    def run(D):
        sim = alloy.build(_rim({"capacity": 0.6, "rate": 20.0, "wicking": D}), device="cpu")
        _run_for(sim, 1.0)
        b = _books(sim)
        assert b["water_fx"] == b["soaked_fx"] and b["over_cap"] <= 0, b
        return int((_parts(sim)[1].water.numpy() > 0).sum())

    dry_spread, wick_spread = run(0.0), run(2e-4)
    assert wick_spread > dry_spread, (wick_spread, dry_spread)


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.timeout(3600)
def test_gpu_uptake_repeats_and_matches_the_cpu_within_5_percent():
    """On a GPU: 4 identical worlds take up the same number of particles, and a second build repeats the counts and the
    water bit for bit (the lowest-index rule and integer books; measured on one L4: 220 in each of 16 worlds, two runs
    with the same water hash). Against the CPU the count differs because the liquid's CPU and GPU steps are the same
    model, not the same bits: within 5 %, a measured-near bound from one rig on one L4 (220 against 224 at 0.5 s)."""
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device")

    def run(device: str, worlds: int) -> tuple[np.ndarray, np.ndarray]:
        sim = alloy.build(_rim(), device=device, worlds=worlds)
        for _ in range(int(round(0.5 / DT))):
            sim.step(1)
        _, ca, _ = _parts(sim)
        return ca.soaked()[1][:worlds].copy(), ca.water.numpy().copy()

    a_count, a_water = run("cuda:0", 4)
    b_count, b_water = run("cuda:0", 4)
    assert (a_count == a_count[0]).all(), a_count
    np.testing.assert_array_equal(a_count, b_count)
    np.testing.assert_array_equal(a_water, b_water)
    cpu_count, _ = run("cpu", 1)
    assert abs(int(a_count[0]) / int(cpu_count[0]) - 1.0) <= 0.05, (a_count[0], cpu_count[0])
