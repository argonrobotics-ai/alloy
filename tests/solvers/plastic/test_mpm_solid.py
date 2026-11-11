# SPDX-License-Identifier: Apache-2.0
"""A j2 solid on implicit MPM, opt-in by the card's route override (GAP #6 slice 1; alloy/solvers/plastic/mpm_solid.py).

(a) the router: the override gives a voxel number, a reason and the tight-solve effect; without it the object routes
as before (soft_vbd); a j2 sheet with the override raises. (b) the per-particle numbers and the hardening refusal.
(c) law U1 through the real route on the CPU: a 12 mm block between a frictionless floor and a kinematic plate has its
plateau stress at sigma_y and its elastic slope at E, with no attribute set after the build.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import textwrap

import numpy as np
import pytest

import alloy
from alloy.cards.types import Plastic, RouteOverride
from alloy.core.errors import AssemblyError, RouteError
from alloy.core.registry import isolated_registry, registered_routes
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.scene.emitters.deformable import mpm_values
from alloy.solvers.plastic import mpm_solid
from alloy.testing import route as T

E, NU, SY = 3e6, 0.3, 1e4
OVERRIDE = RouteOverride("paste_mpm", "GAP #6 test: a j2 solid on implicit MPM")


@pytest.fixture(scope="module")
def routes():
    with isolated_registry(empty=True):
        register_builtin_solvers()
        yield registered_routes()


def _j2(hardening: float | None = None):
    mat = T.material("j2_soft", 1076.0, E=E, nu=NU, yield_stress=SY)
    h = T.q(hardening) if hardening is not None else None
    return dataclasses.replace(mat, plastic=Plastic(yield_stress=T.q(SY), model="j2", hardening_modulus=h))


def test_override_routes_a_j2_volume_to_mpm_with_a_voxel(routes):
    d = Router().route(T.box("blk", _j2(), 0.006, 0.006, 0.006, route=OVERRIDE), T.context(routes))
    assert d.route == Route.PASTE_MPM and d.overridden
    assert d.numbers["voxel"] == pytest.approx(0.003)           # min(d_feature / 4, 5 mm), d_feature 12 mm
    assert any("von Mises solid" in r and "sigma_y / sqrt(3)" in r for r in d.reasons), d.reasons
    assert mpm_solid.EFFECT in d.scene_effects


def test_without_the_override_a_j2_volume_is_unchanged(routes):
    d = Router().route(T.box("blk", _j2(), 0.006, 0.006, 0.006), T.context(routes))
    assert d.route == Route.SOFT_VBD and not d.overridden
    assert "voxel" not in d.numbers and mpm_solid.EFFECT not in d.scene_effects


def test_a_j2_sheet_with_the_override_raises(routes):
    with pytest.raises(RouteError, match="solid volumes only"):
        Router().route(T.box("sheet", _j2(), 0.1, 0.1, 0.001, route=OVERRIDE), T.context(routes))


def test_j2_values_are_von_mises_on_mpm():
    v = mpm_values(_j2())
    assert v == mpm_solid.j2_mpm_values(_j2())
    assert v["yield_stress"] == pytest.approx(SY / np.sqrt(3.0)) and mpm_solid.SQRT3 == pytest.approx(3 ** 0.5)
    assert (v["young_modulus"], v["poisson_ratio"], v["friction"], v["tensile_yield_ratio"]) == (E, NU, 0.0, 1.0)
    assert v["yield_pressure"] == 1e15 and v["hardening"] == v["viscosity"] == v["dilatancy"] == v["damping"] == 0.0


def test_a_hardening_modulus_is_refused_and_zero_is_allowed():
    with pytest.raises(AssemblyError, match="hardening_modulus"):
        mpm_solid.j2_mpm_values(_j2(hardening=1e6))
    assert mpm_solid.j2_mpm_values(_j2(hardening=0.0))["hardening"] == 0.0


def test_other_materials_keep_their_mpm_values():
    paste = T.material("paste", 1100.0, E=1e4, yield_stress=200.0)
    assert mpm_values(paste)["yield_stress"] == 200.0 and "tensile_yield_ratio" not in mpm_values(paste)


def test_the_tolerance_is_inside_the_ruled_range():
    from alloy.integration import mpm_solve

    lo, hi = mpm_solve.TOLERANCE_RANGE
    assert lo <= mpm_solid.TOLERANCE <= hi


def test_sim_routes_names_the_tolerance_the_entry_runs(tmp_path):
    """Sand-like neighbour + a j2 solid in one MPM entry: both rows say what the entry really runs (WP12's M1)."""
    s = alloy.Scene(dt=DT, roots=[_card_root(tmp_path)])
    s.add_ground(friction=1e-4)
    s.add_box("blk", size=(H, H, H), pos=(0, 0, H / 2), material="j2_soft_solid",
              card_extra={"route": {"route": "paste_mpm", "reason": "M1 test"}})
    s.add_box("dough", size=(H, H, H), pos=(0.05, 0, H / 2), material="dough_wheat")
    sim = alloy.build(s, device="cpu")
    rows = sim.routes()
    assert rows["blk"]["route"] == rows["dough"]["route"] == "paste_mpm"
    for name in ("blk", "dough"):
        note = rows[name]["mpm_j2_solid"]
        assert "tolerance 1e-05" in note and "at most 1000 iterations" in note and "dense grid" in note, note
        assert "every MPM material in the entry pays for it" in note
        assert "Floors thinner than a voxel leak on the dense grid" in note, note
        assert "drifts (0.5 mm" not in note


def test_the_sparse_grid_note_names_its_drift_and_early_contact():
    class FakeSolver:
        tolerance, max_iterations = 1e-4, 250

    class FakeView:
        class mpm:
            class tensile_yield_ratio:
                @staticmethod
                def numpy():
                    return np.array([1.0])

    solver = FakeSolver()
    mpm_solid.install_notes(solver, FakeView, "sparse")
    note = solver.alloy_route_notes()["mpm_j2_solid"]
    assert "sparse grid" in note and "a block resting on the ground drifts" in note, note
    assert "from 2 mm above its surface" in note and "Floors thinner" not in note


def test_a_j2_solid_on_the_cpu_asks_for_the_dense_grid_unless_the_user_chose(tmp_path, monkeypatch):
    def grid(env: str | None) -> str:
        if env is None:
            monkeypatch.delenv("ALLOY_MPM_GRID", raising=False)
        else:
            monkeypatch.setenv("ALLOY_MPM_GRID", env)
        s = alloy.Scene(dt=DT, roots=[_card_root(tmp_path / (env or "unset"))])
        s.add_box("blk", size=(H, H, H), pos=(0, 0, 0.1), material="j2_soft_solid",
                  card_extra={"route": {"route": "paste_mpm", "reason": "grid test"}})
        return alloy.build(s, device="cpu").engine.solvers[0].grid_type

    assert grid(None) == "dense" and grid("dense") == "dense" and grid("sparse") == "sparse"


# --- law U1 through the route --------------------------------------------------------------------------------------

CARD = textwrap.dedent("""\
    schema: alloy.material/1
    name: j2_soft_solid
    density: {value: 1076.0, source: model, note: "law U1 fixture"}
    surface:
      mu_static: {value: 1.0e-4, source: model, note: "frictionless fixture"}
      mu_dynamic: {value: 1.0e-4, source: model, note: "frictionless fixture"}
      restitution: {value: 0.0, source: model, note: "fixture"}
    elastic:
      youngs: {value: 3.0e6, source: model, note: "law U1 fixture"}
      poisson: {value: 0.3, source: model, note: "law U1 fixture"}
    plastic:
      yield_stress: {value: 1.0e4, source: model, note: "law U1 fixture"}
      model: j2
    """)
H, DT, TRAVEL = 0.012, 0.005, 1.4e-3


def _card_root(root) -> str:
    (root / "materials").mkdir(parents=True)
    (root / "materials" / "j2_soft_solid.yaml").write_text(CARD)
    return str(root)


def _press(tmp_path, speed: float, caplog) -> dict[str, np.ndarray]:
    """Press the block at ``speed`` m/s; per period: mean particle sigma_zz and elastic strain. Also
    ``iterations``: the most iterations any rheology solve used (Newton keeps no count; with ``solver.verbose`` it logs
    "iteration #N" per batch of 25, which is read from the log here)."""
    s = alloy.Scene(dt=DT, roots=[_card_root(tmp_path)])
    s.add_ground(friction=1e-4)
    s.add_box("blk", size=(H, H, H), pos=(0, 0, H / 2), material="j2_soft_solid",
              card_extra={"route": {"route": "paste_mpm", "reason": "law U1"}})
    s.add_box("plate", size=(2 * H, 2 * H, 0.01), pos=(0, 0, H + 0.005), kinematic=True, friction=1e-4)
    sim = alloy.build(s, device="cpu")
    assert sim.routes()["blk"]["route"] == "paste_mpm"
    eng, solver = sim.engine, sim.engine.solvers[0]
    assert solver.tolerance == pytest.approx(mpm_solid.TOLERANCE)
    sim.step(20)
    solver.verbose = True
    caplog.set_level(logging.INFO)
    caplog.clear()
    sim.drive("plate", velocity=(0, 0, -speed), duration=TRAVEL / speed + 1)
    rows = []
    for _ in range(int(round(TRAVEL / speed / DT))):
        sim.step(1)
        sig = eng.state.mpm.particle_stress.numpy().reshape(-1, 3, 3).mean(axis=0)
        fe = eng.state.mpm.particle_elastic_strain.numpy().reshape(-1, 3, 3).mean(axis=0)
        rows.append((sig[2, 2], fe[2, 2] - 1.0))
    assert sim.check()["ok"]
    health = sim.health()
    out = np.array(rows)
    counts = [int(m.group(1)) for r in caplog.records if (m := re.search(r"iteration #(\d+)", r.getMessage()))]
    return {"stress": out[:, 0], "strain": -out[:, 1], "iterations": np.array(counts, dtype=float),
            "health_ok": bool(health["ok"]) and not health["invalid_worlds"] and not health["degraded_worlds"]}


def _plateau_and_slope(r: dict[str, np.ndarray]) -> tuple[float, float]:
    stress, strain = r["stress"], r["strain"]
    plateau = float(stress.max())                           # the block barrels and softens after ~4 % (U1.md)
    before = np.arange(len(stress)) < int(np.argmax(stress))     # the loading rows (not the barrelled block after)
    low = before & (strain > 1e-4) & (stress < 0.9 * plateau)    # the elastic rows (few: one per 5 ms period)
    assert low.any(), "no elastic row before the plateau"
    slope = float(np.median(stress[low] / strain[low]))
    return plateau, slope


def _check_press(r: dict[str, np.ndarray]) -> float:
    """Plateau within 5 % of sigma_y, slope within 10 % of E. The iteration log is read (``iterations``) but not
    capped: at tolerance 1e-4 about 60 % of the solves end on the 250 cap; the plateau holds (note)."""
    plateau, slope = _plateau_and_slope(r)
    assert plateau == pytest.approx(SY, rel=0.05), (plateau, r["stress"])
    assert slope == pytest.approx(E, rel=0.10), slope
    assert r["iterations"].size > 0, "no solver iteration log was read"
    assert r["health_ok"], "a world is not OK (UNCONVERGED would show here)"
    return plateau


@pytest.mark.law
@pytest.mark.timeout(600)
def test_law_u1_uniaxial_compression_through_the_route(tmp_path, monkeypatch, caplog):
    # no ALLOY_MPM_GRID: the j2 entry asks for the dense grid itself on a CPU (the sparse one drifts; U1.md)
    _check_press(_press(tmp_path, 1.4e-3, caplog))


@pytest.mark.law
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_law_u1_half_speed_plateau_is_the_same(tmp_path, monkeypatch, caplog):
    """Quasi-static at the route's tolerance: half the press speed moves the plateau by less than 5 %."""
    fast = _check_press(_press(tmp_path / "a", 1.4e-3, caplog))
    half = _check_press(_press(tmp_path / "b", 0.7e-3, caplog))
    assert half == pytest.approx(fast, rel=0.05)


def test_the_entry_solves_to_the_punch_tolerance_and_cap():
    """Slice 2: a punch read 0.84 of its converged force at 1e-4 / 250 and 0.99 at 1e-5 / 1,000 (4,000 particles)."""
    from alloy.integration import mpm_solve

    assert (mpm_solid.TOLERANCE, mpm_solid.MAX_ITERATIONS) == (1e-5, 1000)
    lo, hi = mpm_solve.TOLERANCE_RANGE
    assert lo <= mpm_solid.TOLERANCE <= hi


def test_sweeps_needed_follows_the_measured_rows():
    """WP7 slice 2: E 3 MPa converged at 1,000 sweeps, 30 MPa at 4,000; brass needs about 1.6 million."""
    assert mpm_solid.sweeps_needed(3e6, 1076.0, 0.003) == pytest.approx(387, rel=0.01)
    assert mpm_solid.sweeps_needed(3e7, 1076.0, 0.003) == pytest.approx(3872, rel=0.01)
    assert mpm_solid.sweeps_needed(9.7e10, 8500.0, 0.003) == pytest.approx(1.585e6, rel=0.01)
    assert mpm_solid.sweeps_needed(3e6, 1076.0, 0.0015) == pytest.approx(4 * 387, rel=0.01)   # (dt / h)^2


def test_a_metal_card_on_the_route_warns_that_it_is_too_stiff(routes, monkeypatch):
    said = []
    monkeypatch.setattr("alloy.api._say.warn", lambda fmt, *a: said.append(fmt % a))
    monkeypatch.setattr("alloy.route.router._STIFF_WARNED", set())
    metal = dataclasses.replace(T.material("j2_metal", 8500.0, E=9.7e10, nu=0.34, yield_stress=3.1e8),
                                plastic=Plastic(yield_stress=T.q(3.1e8), model="j2"))
    for _ in range(2):
        d = Router().route(T.box("blk", metal, 0.006, 0.006, 0.006, route=OVERRIDE), T.context(routes))
    assert d.route == Route.PASTE_MPM
    assert any("too stiff for implicit MPM" in r and "about 1,600,000 by a rule" in r for r in d.reasons), d.reasons
    assert len(said) == 1 and "metals are not supported on this route yet" in said[0], said
    soft = Router().route(T.box("soft", _j2(), 0.006, 0.006, 0.006, route=OVERRIDE), T.context(routes))
    assert not any("too stiff" in r for r in soft.reasons)


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_on_a_gpu_a_j2_scene_runs_without_graphs_on_the_dense_grid(tmp_path):
    """Slice 2, one L4: the graph grid (sparse) met a punch 2 mm early and read 34 % low; dense matched the CPU."""
    s = alloy.Scene(dt=DT, roots=[_card_root(tmp_path)])
    s.add_ground(friction=1e-4)
    s.add_box("blk", size=(H, H, H), pos=(0, 0, H / 2), material="j2_soft_solid",
              card_extra={"route": {"route": "paste_mpm", "reason": "GPU grid test"}})
    sim = alloy.build(s, device="cuda:0")
    solver = sim.engine.solvers[0]
    assert sim.engine.use_graph is False
    assert getattr(solver, "alloy_grid", {}).get("grid_type") == "dense", getattr(solver, "alloy_grid", None)
    sim.step(2)
    assert sim.check()["ok"]


PUNCH_CARD = CARD.replace("j2_soft_solid", "j2_punch_solid").replace("yield_stress: {value: 1.0e4",
                                                                   "yield_stress: {value: 5.0e4")
PUNCH_CONVERGED_N = 0.2251   # the twin below at 1e-5 / 4,000 sweeps (WP7 slice 2, notes/wp7-dent/slice2/twin)


@pytest.mark.law
@pytest.mark.timeout(600)
def test_law_a_punch_reads_its_converged_force_at_the_route_settings(tmp_path, monkeypatch):
    """Fast twin of slice 2's punch rows (AUDIT-BENCH's ask): a 3 mm kinematic ball pressed at 2 mm/s into a
    15 x 15 x 9 mm block (E 3 MPa, yield 50 kPa, voxel pinned at 3 mm as the rig does, 600 particles, CPU dense)
    pushes back with its converged force at 0.5 mm: 0.2250 N at the route's 1e-5 / 1,000 against 0.2251 N at 4,000
    sweeps. At slice 1's 1e-4 / 250 it read 0.0909 N (0.40)."""
    import alloy.route.checks as ck

    monkeypatch.setattr(ck, "mpm_voxel", lambda d, vmax: 0.003)    # the rig's voxel (the router picks 2.25 mm)
    (tmp_path / "materials").mkdir()
    (tmp_path / "materials" / "j2_punch_solid.yaml").write_text(PUNCH_CARD)
    lx, lz, r = 0.015, 0.009, 0.003
    s = alloy.Scene(dt=DT, roots=[str(tmp_path)])
    s.add_ground(friction=1.0)
    s.add_box("blk", size=(lx, lx, lz), pos=(0, 0, lz / 2), material="j2_punch_solid",
              card_extra={"route": {"route": "paste_mpm", "reason": "punch twin"}})
    s.add_sphere("tool", r, pos=(0, 0, lz + r + 0.001), kinematic=True, friction=1e-4)
    sim = alloy.build(s, device="cpu")
    eng = sim.engine
    solver = eng.solvers[0]
    assert (solver.tolerance, solver.max_iterations) == (mpm_solid.TOLERANCE, mpm_solid.MAX_ITERATIONS)
    body = int(np.where(np.asarray(eng.model.body_label).astype(str) == "tool")[0][0])
    sim.step(1)
    step_dt = DT / sim.physics_steps

    def depth() -> float:
        return lz - (float(sim.state(["tool"], world=0)["tool.pos"][2]) - r)

    sim.drive("tool", velocity=(0, 0, -0.01), duration=10)
    while depth() < -0.5e-3:
        sim.step(1)
    sim.stop_drives()
    sim.drive("tool", velocity=(0, 0, -0.002), duration=100)
    while depth() < 0.5e-3 - 1e-9:
        sim.step(1)
    imp, _, cid = solver.collect_collider_impulses(eng.state)
    imp, cid, cbi = imp.numpy(), cid.numpy(), solver.collider_body_index.numpy()
    force = float(imp[np.where(cid >= 0, cbi[np.maximum(cid, 0)], -2) == body][:, 2].sum()) / step_dt
    assert sim.check()["ok"]
    assert force == pytest.approx(PUNCH_CONVERGED_N, rel=0.02), force
