# SPDX-License-Identifier: Apache-2.0
"""Cloth and liquid with kinematic objects and no free rigid body (gap wave 2, the follow-up of gaps-now #1), CPU.

Cloth + liquid is a coupled scene with two entries (VBD, SPH) and no rigid one. Newton's coupler shows an entry only
the bodies it owns or receives as proxies, and nothing owned the kinematic bodies: the water fell out of a kinematic
jug (median z 0.091 m under a floor at 0.104 m) and the cloth fell through a kinematic tray (PEN_FAIL at 0.07 s, the
world frozen). The build warned about the liquid only. Now the kinematic bodies have their own entry
(``alloy.integration.kinematic_carrier``), proxied one-way into both solvers.

Laws: in the cloth + liquid scene each material meets its kinematic object as it does in its own single-solver scene
(one SPH solver, one VBD solver; no coupler there).
- At rest and through a 30 deg tilt of the jug about its lip, every water particle stays in the jug, on its floor or
  above, and the water's height in the jug's frame is the liquid-only scene's within 1 mm.
- The cloth lands on the tray and rides up with it (0.1 m/s for 0.3 s): its height over the tray is the cloth-only
  scene's within 1 mm, and no vertex is under the tray's top.
- The structure: the carrier owns the kinematic bodies, is the source of one one-way pair per solver, and leaves the
  scene's own pairs alone; a scene with a rigid entry, one entry, or no kinematic body gets no carrier.
- Liquid and sand (SPH + MPM, the other pair of solvers that builds with no rigid entry): sand rides a driven
  kinematic bin exactly as in the sand-only scene (0 um), with a jug of water beside it.
- A clamped rod (an entry that owns bodies) beside the carrier moves as beside static walls (0 um).
- No carrier under ADMM or beside a one-way pair; sim.routes() names no rigid solver.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy._vendor import newton
from alloy.api import _liquid
from alloy.coupling.types import PairPolicy
from alloy.integration import kinematic_carrier as kc
from alloy.integration import stepper as stepper_mod
from alloy.integration.stepper import coupler_pairs, dynamic_entries
from alloy.scene.types import EntrySpec

IN, WALL, Z = 0.05, 0.004, 0.10
LIP = (IN / 2 + WALL, 0.0, Z + WALL + IN)              # the jug's +x rim, outer top edge
BIN_WALL = 0.006                 # the sand bin's floor: thicker than MPM's 5 mm voxel (mpm-sparse-floor-offset.md)
TRAY_X, TRAY_TOP, LIFT, LIFT_S = 0.4, 0.205, 0.1, 0.3    # tray centre x, its top face z [m]; lift speed [m/s], time [s]


def _local(p: np.ndarray, pos: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Points in the frame of a body at ``pos`` with xyzw quaternion ``q``."""
    x, y, z, w = (float(c) for c in q)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return (p - pos) @ r


def _scene(water: bool, cloth: bool, worlds: int = 1) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    if water:
        sc.add_open_box("jug", inner=(IN, IN, IN), wall=WALL, pos=(0, 0, Z), kinematic=True)
        sc.add_liquid("water", size=(0.044, 0.044, 0.016), pos=(0, 0, Z + WALL + 0.009))
    if cloth:
        sc.add_box("tray", size=(0.2, 0.2, 0.01), pos=(TRAY_X, 0, TRAY_TOP - 0.005), kinematic=True)
        sc.add_cloth("sheet", size=(0.1, 0.1), pos=(TRAY_X, 0, TRAY_TOP + 0.01), resolution=6, mass=0.01)
    return sc


def _run(water: bool, cloth: bool, device: str = "cpu", worlds: int = 1, world: int = 0) -> tuple[dict, alloy.Sim]:
    """Rest 0.1 s, then tilt the jug 30 deg about its lip and lift the tray, both over 0.3 s; 0.05 s more.
    Returns the water in the jug floor's frame (rest, end) and the cloth's heights over the tray's top (rest, end),
    read in ``world``."""
    sim = alloy.build(_scene(water, cloth, worlds), device=device)
    out: dict = {}

    def read(tag: str) -> None:
        if water:
            s = sim.state(["water", "jug"], world=world)
            out[f"water_{tag}"] = _local(s["water.points"], s["jug.pos"], s["jug.quat_xyzw"])
        if cloth:
            s = sim.state(["sheet", "tray"], world=world)
            out[f"sheet_{tag}"] = s["sheet.points"][:, 2] - (float(s["tray.pos"][2]) + 0.005)

    sim.step(10)
    read("rest")
    if water:
        sim.drive("jug", angular_velocity=(0, math.radians(100), 0), pivot=LIP, duration=0.3)
    if cloth:
        sim.drive("tray", velocity=(0, 0, LIFT), duration=LIFT_S)
    sim.step(35)
    read("end")
    return out, sim


@pytest.fixture(scope="module")
def both():
    out, sim = _run(water=True, cloth=True)
    yield out, sim
    sim.close()


@pytest.fixture(scope="module")
def built_only():
    sim = alloy.build(_scene(True, True), device="cpu")
    yield sim
    sim.close()


def test_the_carrier_owns_the_kinematic_bodies_and_feeds_both_solvers(built_only):
    sim = built_only
    built, stepper = sim.engine.built, sim.engine.stepper
    flags = built.model.body_flags.numpy()
    kinematic = {b for b in range(built.model.body_count) if int(flags[b]) & int(newton.BodyFlags.KINEMATIC)}
    assert len(kinematic) == 2                                           # the jug (one body) and the tray
    dyn = dynamic_entries(built)
    assert sorted(e.capability for e in dyn) == ["sph", "vbd"]          # the assembler's entries are unchanged
    spec = kc.carrier_spec(dyn, built.model)
    assert set(spec.bodies) == kinematic and len(spec.joints) == 2 and not spec.particles
    pairs = {(p.source, p.destination): p for p in coupler_pairs(built, [*dyn, spec])}
    assert set(pairs) == {("vbd", "sph"), (kc.CARRIER, "vbd"), (kc.CARRIER, "sph")}
    assert pairs["vbd", "sph"] == built.coupler.pairs[0]                 # the scene's own pair, as assembled
    assert all(p.one_way and p.mode == "lagged" for (src, _), p in pairs.items() if src == kc.CARRIER)
    assert type(stepper).__name__ == "Scheduler" and kc.CARRIER in stepper._entries
    assert isinstance(stepper._entries[kc.CARRIER].solver, kc.SolverKinematicCarrier)
    assert _liquid.unseen_kinematic(sim.engine) == 0


def test_no_carrier_without_unowned_kinematic_bodies_or_beside_a_rigid_entry(built_only):
    sim = built_only
    model = sim.engine.built.model
    dyn = dynamic_entries(sim.engine.built)
    assert kc.carrier_spec(dyn[:1], model) is None                       # one entry steps the full model
    hub = EntrySpec(name="mujoco", capability="mujoco", bodies=(), particles=(), joints=(), shapes=(), substeps=1,
                    target_dt=0.001)
    assert kc.carrier_spec([*dyn, hub], model) is None                   # adopt_kinematic gives them to the hub
    spec = kc.carrier_spec(dyn, model)
    owner = dataclasses.replace(dyn[0], bodies=spec.bodies)
    assert kc.carrier_spec([owner, dyn[1]], model) is None               # every kinematic body is owned
    assert kc.carrier_pairs(dyn) == () and kc.without_carrier([*dyn, spec]) == list(dyn)


def test_no_carrier_under_admm_or_beside_a_one_way_pair(built_only, caplog):
    """WP4's review: the carrier's pairs would make a follower leaf's destination a chain, which the one-way stage
    refuses, and ADMM has no one-way stage. There the bodies stay unowned, with the warning, as before."""
    built = built_only.engine.built
    dyn = dynamic_entries(built)
    assert [e.name for e in stepper_mod._with_carrier(dyn, built.model, built.coupler)][-1] == kc.CARRIER
    follower = SimpleNamespace(kind="proxy", pairs=(PairPolicy("vbd", "sph", mode="lagged", one_way=True),))
    for coupler in (follower, SimpleNamespace(kind="admm", pairs=built.coupler.pairs)):
        with caplog.at_level("WARNING", logger="alloy.integration"):
            caplog.clear()
            assert stepper_mod._with_carrier(dyn, built.model, coupler) == list(dyn)
        assert sum("no rigid entry to own them" in r.getMessage() for r in caplog.records) == 1


def test_the_kinematic_objects_report_no_rigid_solver(built_only):
    """OSS-3's review: sim.routes() must not name a rigid solver the scene lacks."""
    routes = built_only.routes()
    assert {routes[n]["route"] for n in ("jug_floor", "tray")} == {"static"}
    assert "mujoco" not in str({n: routes[n] for n in ("jug_floor", "tray")}).lower()


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_water_in_a_kinematic_jug_beside_cloth_matches_the_liquid_only_scene(both):
    out, sim = both
    ref, ref_sim = _run(water=True, cloth=False)
    assert type(ref_sim.engine.stepper).__name__ != "Scheduler"         # the reference: one solver, no coupler
    tilt = 2 * math.asin(abs(float(sim.state(["jug"], world=0)["jug.quat_xyzw"][1])))
    assert math.degrees(tilt) == pytest.approx(30.0, abs=0.5)
    for tag in ("rest", "end"):
        loc = out[f"water_{tag}"]
        inside = (np.abs(loc[:, 0]) < IN / 2 + 1e-3) & (np.abs(loc[:, 1]) < IN / 2 + 1e-3)
        assert inside.all(), (tag, int((~inside).sum()))
        assert float(loc[:, 2].min()) > WALL / 2 - 1.5e-3, (tag, float(loc[:, 2].min()))    # on the floor
        level, ref_level = float(np.median(loc[:, 2])), float(np.median(ref[f"water_{tag}"][:, 2]))
        print(f"water {tag}: {int((~inside).sum())} outside, min {loc[:, 2].min() * 1e3:.2f} mm, level "
              f"{level * 1e3:.2f} mm (liquid only {ref_level * 1e3:.2f})")
        assert level == pytest.approx(ref_level, abs=1.0e-3, rel=0), tag
    assert sim.check()["ok"], sim.check()["problems"]
    ref_sim.close()


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_cloth_on_a_kinematic_tray_beside_liquid_matches_the_cloth_only_scene(both):
    out, sim = both
    ref, ref_sim = _run(water=False, cloth=True)
    assert type(ref_sim.engine.stepper).__name__ != "Scheduler"
    lifted = float(sim.state(["tray"], world=0)["tray.pos"][2]) - (TRAY_TOP - 0.005)
    assert lifted == pytest.approx(LIFT * LIFT_S, abs=1e-6, rel=0)       # the tray went up 30 mm
    for tag in ("rest", "end"):
        h = out[f"sheet_{tag}"]
        assert float(h.min()) > -0.5e-3, (tag, float(h.min()))           # no vertex under the tray's top
        print(f"sheet {tag}: over the tray min {h.min() * 1e3:.2f} mean {h.mean() * 1e3:.2f} max {h.max() * 1e3:.2f} "
              f"mm (cloth only mean {ref[f'sheet_{tag}'].mean() * 1e3:.2f})")
        assert float(h.mean()) == pytest.approx(float(ref[f"sheet_{tag}"].mean()), abs=1.0e-3, rel=0), tag
    assert float(out["sheet_end"].max()) < 5e-3                          # lying on it, not left behind or thrown
    ref_sim.close()


def _sand_in_a_driven_bin(water: bool) -> tuple[list[np.ndarray], alloy.Sim]:
    """Sand in a kinematic bin: rest 0.05 s, lift at 0.1 m/s for 0.15 s, 0.05 s more; with ``water`` a jug of water
    stands beside it (SPH + MPM, no rigid entry), else the sand is the scene's one solver. Returns the sand's heights
    over the bin's frame at the three reads."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("bin", inner=(0.06, 0.06, 0.05), wall=BIN_WALL, pos=(0.3, 0, Z), kinematic=True)
    sc.add_box("sand", size=(0.04, 0.04, 0.02), pos=(0.3, 0, Z + BIN_WALL + 0.011), material="sand_silica_dry")
    sc.set_friction("bin_floor", "sand", 0.7)
    if water:
        sc.add_open_box("jug", inner=(IN, IN, IN), wall=WALL, pos=(0, 0, Z), kinematic=True)
        sc.add_liquid("water", size=(0.044, 0.044, 0.016), pos=(0, 0, Z + WALL + 0.009))
    sim = alloy.build(sc, device="cpu")
    out = []

    def read() -> None:
        s = sim.state(["sand", "bin"], world=0)
        out.append(s["sand.points"][:, 2] - float(s["bin.pos"][2]))

    sim.step(5)
    read()
    sim.drive("bin", velocity=(0, 0, 0.1), duration=0.15)
    sim.step(15)
    read()
    sim.step(5)
    read()
    return out, sim


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_sand_in_a_driven_kinematic_bin_beside_liquid_matches_the_sand_only_scene():
    """WP12's review: a kinematic body that moves against sand through the carrier. The sand rides the bin up 15 mm
    exactly as in the sand-only scene, where the one MPM solver sees the bin directly (measured: the same to the
    bit over 0.3 s), and the jug beside it holds its water. The law covers one MPM substep per exchange, the rung
    this scene runs at; for more, the MPM entry carries its proxy bodies through its substeps
    (``carried_into_mpm``), which the law pins as installed."""
    got, sim = _sand_in_a_driven_bin(water=True)
    stepper = sim.engine.stepper
    assert {r["route"] for r in sim.routes().values()} >= {"liquid_sph", "grain_mpm"}
    assert sorted(stepper._entries) == [kc.CARRIER, "mpm", "sph"]
    mpm = stepper._entries["mpm"].solver
    assert getattr(mpm, "_alloy_body_carry", False)                       # proxies carried through MPM's substeps
    rungs = dict(stepper._clock.rungs)
    print(f"substeps per exchange: {rungs}")
    assert rungs["mpm"] == 1 and rungs[kc.CARRIER] == 1, rungs            # the case this law covers
    friction = mpm._mpm_model.collider.material_friction.numpy()      # set_friction("bin_floor", "sand", 0.7):
    assert friction.size > 5 and np.isclose(friction, 0.7, rtol=1e-6, atol=0).any()     # the bin is MPM's collider
    lifted = float(sim.state(["bin"], world=0)["bin.pos"][2]) - (Z + BIN_WALL / 2)
    assert lifted == pytest.approx(0.015, abs=1e-6, rel=0)
    water = sim.state(["water"], world=0)["water.points"][:, 2]
    assert float(water.min()) > Z + WALL - 1.5e-3, float(water.min())    # the jug's water is on its floor
    assert sim.check()["ok"], sim.check()["problems"]
    sim.close()
    ref, ref_sim = _sand_in_a_driven_bin(water=False)
    assert type(ref_sim.engine.stepper).__name__ != "Scheduler"         # the reference: one solver, no coupler
    ref_sim.close()
    for k, (h, r) in enumerate(zip(got, ref, strict=True)):
        gap = float(np.abs(h - r).max())
        print(f"read {k}: sand over the bin's frame min {h.min() * 1e3:.3f} median {np.median(h) * 1e3:.3f} mm; "
              f"against sand only {gap * 1e6:.3f} um")
        assert gap < 1.0e-6, (k, gap)
        assert float(h.min()) > BIN_WALL / 2 - 1.0e-3, (k, float(h.min()))   # on the bin's floor, not through


def _rod_beside_a_jug(**jug) -> tuple[list[np.ndarray], alloy.Sim]:
    """A clamped rod (the elastic-rod entry owns its bodies) beside water in a jug; the rod's centre line at the
    start and after 10 and 20 periods."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("jug", inner=(IN, IN, IN), wall=WALL, pos=(0, 0, Z), **jug)
    sc.add_liquid("water", size=(0.044, 0.044, 0.016), pos=(0, 0, Z + WALL + 0.009))
    sc.add_rod("held", length=0.3, radius=0.003, pos=(0, 0.3, 0.3), direction=(1, 0, 0), clamp="start")
    sim = alloy.build(sc, device="cpu")
    out = [sim.state(["held"], world=0)["held.points"].copy()]
    for _ in range(2):
        sim.step(10)
        out.append(sim.state(["held"], world=0)["held.points"].copy())
    return out, sim


@pytest.mark.law
@pytest.mark.timeout(1200)
def test_an_entry_that_owns_bodies_keeps_them_beside_the_carrier():
    """WP4's review: the carrier's step copies every body row of its own states, and Newton writes back an entry's
    owned rows only (solver_coupled.py, the owned_rows copy), so the rod's bodies are its solver's. The clamped rod
    swings down beside a kinematic jug exactly as beside a static one, and the jug holds its water."""
    carried, sim = _rod_beside_a_jug(kinematic=True)
    assert sorted(sim.engine.stepper._entries) == [kc.CARRIER, "sph", "vbd"]
    owned = set(next(e for e in dynamic_entries(sim.engine.built) if e.name == "vbd").bodies)
    assert owned and not owned & set(kc.carrier_spec(dynamic_entries(sim.engine.built), sim.engine.built.model).bodies)
    water = sim.state(["water"], world=0)["water.points"][:, 2]
    assert float(water.min()) > Z + WALL - 1.5e-3 and sim.check()["ok"]
    sim.close()
    plain, ref = _rod_beside_a_jug(static=True)
    assert sorted(ref.engine.stepper._entries) == ["sph", "vbd"]
    ref.close()
    gap = max(float(np.abs(a - b).max()) for a, b in zip(carried[1:], plain[1:], strict=True))
    print(f"rod beside the carrier vs beside static walls: {gap * 1e6:.3f} um; tip fell "
          f"{(carried[0][-1, 2] - carried[-1][-1, 2]) * 1e3:.1f} mm")
    assert gap < 1.0e-6, gap
    assert carried[0][-1, 2] - carried[-1][-1, 2] > 0.1                   # the rod moved: its free end swung down
    assert float(np.abs(carried[-1][0] - carried[0][0]).max()) < 1e-6     # and its clamp held
