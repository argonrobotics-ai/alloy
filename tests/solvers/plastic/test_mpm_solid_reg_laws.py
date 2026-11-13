# SPDX-License-Identifier: Apache-2.0
"""GAP #6 slice 2b laws for vendored patch 0015 (the j2 route's Delassus regularization 1e-12), on slice 2's small
punch rig: a 5 mm kinematic ball pressed at 2 mm/s into a 30 x 30 x 15 mm block on a rough floor, 3 mm voxels (4,000
particles, CPU dense grid), the force read at 1.0 mm ([internal note] notes/wp7-dent/slice2, rig3.py; b8 README).

Brass (E 97 GPa, Poisson 0.34, yield 310 MPa) with quasi-static mass scaling x16,000 (density x s, gravity / s, so
its weight stays true) is the same problem, in E / yield and Poisson, as the toy below (E 3 MPa, yield 9.6 kPa,
Poisson 0.34), so its force is the toy's times the yield ratio (32,292). The scaling is done here by hand after the
build, as the rig does; slice 2b's build option comes later. Slow (minutes each on a CPU): the gates skip them, so
each is run by hand and its reading kept in the patch's VENDORED.md row."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

import alloy
from alloy.solvers.plastic import mpm_solid

R, LX, LZ, GAP, DT = 0.005, 0.03, 0.015, 0.002, 0.005
TOY = {"youngs": 3.0e6, "poisson": 0.34, "yield": 9.6e3, "density": 1076.0}
BRASS = {"youngs": 9.7e10, "poisson": 0.34, "yield": 3.1e8, "density": 8500.0}
TOY_N = 0.84633              # the toy at the route's settings, F at 1.0 mm (this helper, CPU; 0.13 % under its
#                              value at tolerance 1e-7 and 4,000 sweeps on rig3, notes/wp7-dent/slice2/b7)
YIELD_RATIO = BRASS["yield"] / TOY["yield"]


def _card(m: dict) -> str:
    return textwrap.dedent(f"""\
        schema: alloy.material/1
        name: reg_law_j2
        density: {{value: {m['density']}, source: model, note: "slice 2b law"}}
        surface:
          mu_static: {{value: 0.5, source: model, note: "rig3"}}
          mu_dynamic: {{value: 0.5, source: model, note: "rig3"}}
          restitution: {{value: 0.0, source: model, note: "rig3"}}
        elastic:
          youngs: {{value: {m['youngs']}, source: model, note: "slice 2b law"}}
          poisson: {{value: {m['poisson']}, source: model, note: "slice 2b law"}}
        plastic:
          yield_stress: {{value: {m['yield']}, source: model, note: "slice 2b law"}}
          model: j2
        """)


def punch(tmp_path, monkeypatch, material: dict, mass_scale: float = 1.0) -> dict:
    """The small rig; returns the force at 1.0 mm (N), the mean rheology sweeps per solve and the check."""
    import alloy.route.checks as ck
    from alloy._vendor.newton._src.solvers.implicit_mpm import solve_rheology as SR

    monkeypatch.setattr(ck, "mpm_voxel", lambda d, vmax: 0.003)    # the rig's voxel
    (tmp_path / "materials").mkdir(parents=True, exist_ok=True)
    (tmp_path / "materials" / "reg_law_j2.yaml").write_text(_card(material))
    s = alloy.Scene(dt=DT, roots=[str(tmp_path)])
    s.add_ground(friction=1.0)
    s.add_box("blk", size=(LX, LX, LZ), pos=(0, 0, LZ / 2), material="reg_law_j2",
              card_extra={"route": {"route": "paste_mpm", "reason": "slice 2b law"}})
    s.add_sphere("tool", R, pos=(0, 0, LZ + R + GAP), kinematic=True, friction=1e-4)
    sim = alloy.build(s, device="cpu")
    eng, solver = sim.engine, sim.engine.solvers[0]
    if mass_scale != 1.0:                       # quasi-static mass scaling with the weight kept true (rig3)
        eng.model.particle_mass.assign(eng.model.particle_mass.numpy() * mass_scale)
        eng.model.set_gravity((0.0, 0.0, -9.81 / mass_scale))
        solver._mpm_model.notify_particle_material_changed()
    sweeps: list[int] = []
    loop = SR._run_solver_loop

    def counted(rs, *a, **k):
        n, solve = [0], rs.solve

        def one(*x, **y):
            n[0] += 1
            return solve(*x, **y)
        rs.solve = one
        try:
            return loop(rs, *a, **k)
        finally:
            del rs.solve
            sweeps.append(n[0])

    with pytest.MonkeyPatch.context() as mp:            # undone after each run: two runs in one test do not nest
        mp.setattr(SR, "_run_solver_loop", counted)
        body = int(np.where(np.asarray(eng.model.body_label).astype(str) == "tool")[0][0])
        sim.step(10)                                # rest 0.05 s, as the rig
        step_dt = DT / (sim.physics_steps // 10)

        def depth() -> float:
            return LZ - (float(sim.state(["tool"], world=0)["tool.pos"][2]) - R)

        sim.drive("tool", velocity=(0, 0, -0.01), duration=10)
        while depth() < -0.8e-3:
            sim.step(1)
        sim.stop_drives()
        sim.drive("tool", velocity=(0, 0, -0.002), duration=100)
        while depth() < 1.0e-3 - 1e-9:
            sim.step(1)
        imp, _, cid = solver.collect_collider_impulses(eng.state)
        imp, cid, cbi = imp.numpy(), cid.numpy(), solver.collider_body_index.numpy()
        force = float(imp[np.where(cid >= 0, cbi[np.maximum(cid, 0)], -2) == body][:, 2].sum()) / step_dt
    return {"force": force, "sweeps": float(np.mean(sweeps)), "ok": bool(sim.check()["ok"]),
            "regularization": solver.delassus_regularization}


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_law_a_soft_card_reads_the_same_at_newtons_regularization(tmp_path, monkeypatch):
    """Patch 0015's ruling: the route's 1e-12 leaves a soft card's punch force where Newton's 1e-6 had it.
    Measured: 0.846343 N (1e-12) and 0.846284 N (1e-6), 6.9e-5 apart; 186 s each on a CPU core."""
    after = punch(tmp_path / "a", monkeypatch, TOY)
    monkeypatch.setattr(mpm_solid, "DELASSUS_REGULARIZATION", 1.0e-6)
    before = punch(tmp_path / "b", monkeypatch, TOY)
    print("LAW soft", after, before)                           # the readings the VENDORED.md row quotes (pytest -s)
    assert (after["regularization"], before["regularization"]) == (1.0e-12, 1.0e-6)
    assert after["ok"] and before["ok"]
    assert after["force"] == pytest.approx(TOY_N, rel=2e-3), after
    assert after["force"] == pytest.approx(before["force"], rel=1e-4), (after, before)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_law_mass_scaled_brass_reads_its_force_at_the_routes_regularization(tmp_path, monkeypatch):
    """Brass x16,000 reads the toy's force times the yield ratio at the route's 1e-12, within its sweeps, and fails
    at Newton's 1e-6 (the code before patch 0015). Measured: 27,420 N = 1.003 of 27,330 in 106 sweeps a solve
    (152 s); at 1e-6 11,658 N = 0.427 in 814 (298 s)."""
    ref = TOY_N * YIELD_RATIO
    after = punch(tmp_path / "a", monkeypatch, BRASS, mass_scale=16000.0)
    assert after["ok"] and after["regularization"] == 1.0e-12
    assert 0.99 <= after["force"] / ref <= 1.02, (after, ref)
    assert after["sweeps"] < 300, after                       # the solve converges well inside the cap of 1,000
    monkeypatch.setattr(mpm_solid, "DELASSUS_REGULARIZATION", 1.0e-6)
    before = punch(tmp_path / "b", monkeypatch, BRASS, mass_scale=16000.0)
    print("LAW brass", after["force"] / ref, after, before["force"] / ref, before)
    assert before["force"] / ref < 0.6, (before, ref)
