# SPDX-License-Identifier: Apache-2.0
"""Start overlaps and fast spins in health (SCR-WP5-14 and its addendum, 2026-11-08; CPU).

Before: WP13's YCB cans placed 16-22 mm inside a bin wall read OK. Health judged depth only on each check step's end
state, after the solver had pushed the overlap out, and MuJoCo-route depths raise no PEN flag (SCR-SC0-1). A body spun
to 0.4-0.7 rad per step by rim contacts (WP3's markers) had no flag before it was flung about 15 ms later.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy

WALL, INNER = 0.01, 0.30


def _bin(over: float, shape: str = "box", **build):
    """A 6 cm block (or a 33 mm-radius can) whose outer side is ``over`` past a bin wall's inner face (< 0: clear)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("bin", inner=(INNER, INNER, 0.15), wall=WALL, pos=(0, 0, 0))
    if shape == "box":
        scene.add_box("block", size=(0.06, 0.06, 0.06), pos=(INNER / 2 + over - 0.03, 0, WALL + 0.031), mass=0.4)
    else:
        scene.add_cylinder("can", 0.033, 0.10, pos=(INNER / 2 + over - 0.033, 0, WALL + 0.051), mass=0.4)
    return alloy.build(scene, device="cpu", **build)


@pytest.mark.law
@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_a_body_placed_18_mm_inside_a_wall_is_start_pen_by_name_and_depth(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _bin(0.018)
    sim.step(1)
    s = sim.health()["worlds"][0]["start_overlap"]
    assert {s["a"], s["b"]} == {"block", "bin_wall0"} and s["level"] == "fail"
    assert s["depth_mm"] == pytest.approx(18.0, abs=1.0)     # MuJoCo's box-box depth reads 17.25 for 18 geometric
    sim.run(0.5)                                             # the solver has pushed it out (or the world latched)
    w = sim.health()["worlds"][0]
    assert "START_PEN" in w["flags"] and w["start_overlap"] == s and w["severity"] in ("DEGRADED", "INVALID")
    c = sim.check()
    assert not c["ok"] and any("block" in p["what"] and "bin_wall0" in p["what"] for p in c["problems"])
    assert sum("at the start" in str(p.get("what")) for p in c["problems"]) == 1   # one line per overlap (OSS-3)


@pytest.mark.law
@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_a_shallow_start_overlap_is_degraded_and_a_clear_start_is_ok(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _bin(0.004)                                        # 4 mm in: past pen_warn, under pen_fail
    sim.run(0.3)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "DEGRADED" and "START_PEN" in w["flags"] and w["start_overlap"]["level"] == "warn"
    assert w["start_overlap"]["depth_mm"] == pytest.approx(4.0, abs=1.0) and sim.check()["ok"]
    clear = _bin(-0.002)                                     # 2 mm clear of the wall, 1 mm above the floor
    clear.run(0.3)
    w = clear.health()["worlds"][0]
    assert "START_PEN" not in w["flags"] and "start_overlap" not in w and clear.check()["ok"]


@pytest.mark.law
def test_a_can_18_mm_into_the_wall_names_the_can() -> None:
    sim = _bin(0.018, "can")
    sim.step(1)
    s = sim.health()["worlds"][0]["start_overlap"]
    assert {s["a"], s["b"]} == {"can", "bin_wall0"} and s["level"] == "fail" and s["depth_mm"] > 15.0


def test_under_the_precision_preset_a_deep_start_overlap_is_invalid() -> None:
    from alloy.core.errors import PrecisionError

    sim = _bin(0.018, profile="precision")
    with pytest.raises((PrecisionError, alloy.SimulationError)):
        sim.step(1)
        sim.engine.health.poll()
    w = sim.health()["worlds"][0]
    assert w["severity"] == "INVALID" and "PEN_FAIL" in w["flags"] and "START_PEN" in w["flags"]


def test_a_cutter_marked_by_relax_thin_part_rule_is_not_judged_at_the_start() -> None:
    sim = _bin(0.004)
    shapes = list(sim.engine.model.shape_label)
    sim.engine.health.relax_thin_part_rule([shapes.index("bin_wall0")])
    sim.run(0.2)
    assert "START_PEN" not in sim.health()["worlds"][0]["flags"]


def _spinner(spins: dict[str, float], sphere: bool = False):
    """Free boxes (or spheres) high in the air (no contacts), each spinning about z [rad/s] from the start: set before
    the first step, with the energy reference taken again (a mid-run kick is an unexplained gain: ENERGY_JUMP)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for k, name in enumerate(spins):
        if sphere:
            scene.add_sphere(name, 0.005, pos=(0.3 * k, 0, 2.0), mass=0.004)
        else:
            scene.add_box(name, size=(0.08, 0.01, 0.01), pos=(0.3 * k, 0, 2.0), mass=0.05)
    sim = alloy.build(scene, device="cpu")
    eng = sim.engine
    qd = eng.state.joint_qd.numpy()
    starts = eng.model.joint_qd_start.numpy()
    for k, w in enumerate(spins.values()):
        qd[starts[k] + 5] = w                                # a free joint's qd: linear (3), then angular (3)
    eng.state.joint_qd.assign(qd)
    from alloy._vendor import newton

    newton.eval_fk(eng.model, eng.state.joint_q, eng.state.joint_qd, eng.state)     # body_qd: the energy's source
    eng.health.reset_reference(eng.state)
    return sim


@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_a_body_turning_more_than_half_a_radian_per_step_is_spin_by_name(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _spinner({"slow": 200.0, "fast": 900.0})           # rad/s; the physics step is 1.25 ms: 0.25 and 1.1 rad
    sim.step(1)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "DEGRADED" and "SPIN" in w["flags"]
    assert w["problem"].startswith("fast turned") and "in one physics step (limit 0.5 rad)" in w["problem"]
    rad = float(sim.engine.health.poll().spin_rad[0])
    assert 0.5 < rad < 2.0, rad


def test_slow_spins_and_relaxed_spinners_are_not_spin() -> None:
    sim = _spinner({"slow": 200.0})
    sim.step(1)
    assert "SPIN" not in sim.health()["worlds"][0]["flags"]
    wheel = _spinner({"wheel": 900.0})
    wheel.engine.health.relax_spin([list(wheel.engine.model.body_label).index("wheel")])
    wheel.step(1)
    assert "SPIN" not in wheel.health()["worlds"][0]["flags"]
    assert np.all(wheel.engine.health.poll().spin_body == -1)


@pytest.mark.law
def test_every_pair_past_a_centimetre_is_counted_and_the_penetration_advice_points_at_the_start() -> None:
    """WP13 (11-08): only the deepest pair was named, and check()'s penetration line advised stiffer contacts."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("bin", inner=(INNER, INNER, 0.15), wall=WALL, pos=(0, 0, 0))
    for k, (over, y) in enumerate([(0.030, -0.09), (0.022, 0.0), (0.016, 0.09)]):
        scene.add_box(f"b{k}", size=(0.06, 0.06, 0.06), pos=(INNER / 2 + over - 0.03, y, WALL + 0.031), mass=0.4)
    sim = alloy.build(scene, device="cpu")
    sim.step(1)
    s = sim.health()["worlds"][0]["start_overlap"]
    assert (s["a"], s["b"]) in (("b0", "bin_wall0"), ("bin_wall0", "b0")) and s["more_count"] == 2
    assert [{m["a"], m["b"]} for m in s["more"]] == [{"b1", "bin_wall0"}, {"b2", "bin_wall0"}]
    c = sim.check()
    assert c["problems"][0]["check"] == "start_overlap" and "also" in c["problems"][0]["what"]
    pen = [p for p in c["problems"] if p.get("check") == "penetration"]
    assert all(p["fix"].startswith("caused by the start overlap") for p in pen if "b0" in p["what"])


def test_a_sphere_rolling_fast_is_not_spin() -> None:
    """WP3: turning a sphere leaves its contacts as they were (a 5 mm marble at 2.5 m/s turns 0.625 rad per step)."""
    sim = _spinner({"marble": 900.0}, sphere=True)
    sim.step(1)
    assert "SPIN" not in sim.health()["worlds"][0]["flags"]


def test_spin_is_judged_per_rigid_step_when_an_exchange_runs_several(monkeypatch) -> None:
    """WP3: a coupled MuJoCo entry can take k steps per exchange of dt_x; |w| dt_x overstates its turn k times."""
    sim = _spinner({"fast": 900.0})                          # 1.1 rad per 1.25 ms exchange
    eng = sim.engine
    clock = eng.stepper.choose_clock
    monkeypatch.setattr(eng.stepper, "choose_clock", lambda c, last: (clock(c, last)[0], {
        n: 4 for n in clock(c, last)[1]}))                   # as if the rigid entry ran 4 steps per exchange
    sim.step(1)
    assert eng.health.start.rigid_substeps() == 4 and "SPIN" not in sim.health()["worlds"][0]["flags"]
    assert 0.2 < float(eng.health.poll().spin_rad[0]) < 0.35          # 1.1 rad / 4


@pytest.mark.law
def test_in_a_coupled_scene_spin_reads_the_rigid_entrys_substeps() -> None:
    """WP3's 2-entry pin: cloth (VBD) beside a box (MuJoCo); health's rigid steps per exchange follow sim.substeps."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.03), mass=0.2)
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.1), resolution=6, mass=0.02, self_contact=False)
    sim = alloy.build(scene, device="cpu")
    sim.step(1)
    eng = sim.engine
    rigid = [e.name for e in eng.built.entries if len(e.bodies)]
    assert len(eng.built.entries) >= 2 and rigid
    assert eng.health.start.rigid_substeps() == max(eng.substeps[n] for n in rigid)


@pytest.mark.law
def test_a_first_period_inside_a_graph_capture_skips_the_start_judge_without_host_reads(monkeypatch, caplog) -> None:
    """WP10's red JOB1.first_press_w0 (CUDA error 906 on rc-1109; coordinator 2026-11-09): SliceSim recorded its first
    period into a CUDA graph with health inside, and START_PEN's judge read the host there. Under capture the judge
    is skipped once, with a warning and no host read; a later eager update does not judge a state past the start."""
    from alloy.precision import health_start as HS

    def host_read(*_a):
        raise AssertionError("START_PEN read the host inside a capture")

    sim = _bin(0.018)                                         # 18 mm into the wall: START_PEN when judged
    monkeypatch.setattr(HS, "capturing", lambda device: True)
    for name in ("_newton", "_mujoco"):
        monkeypatch.setattr(HS.StartChecks, name, host_read)
    with caplog.at_level("WARNING", logger="alloy.precision"):
        sim.step(1)
    monkeypatch.undo()
    sim.step(5)
    w = sim.health()["worlds"][0]
    assert "START_PEN" not in w["flags"] and "start_overlap" not in w
    assert sum("not judged" in r.getMessage() for r in caplog.records) == 1
    assert sim.engine.health.start.skipped and not sim.engine.health.start.pending


def _prism_in_ground(depth: float, device: str = "cpu"):
    """OSS-18's agent smoke (2026-11-09): a convex 6 cm hexagonal prism (the agent's OBJ) with its bottom ``depth``
    below the ground (< 0: above it)."""
    a = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    ring = np.stack([0.03 * np.cos(a), 0.03 * np.sin(a)], axis=1)
    v = np.concatenate([np.c_[ring, np.full(6, -0.03)], np.c_[ring, np.full(6, 0.03)]])
    f = [[i, (i + 1) % 6, 6 + (i + 1) % 6] for i in range(6)] + [[i, 6 + (i + 1) % 6, 6 + i] for i in range(6)]
    f += [[0, k + 1, k] for k in range(1, 5)] + [[6, 6 + k, 7 + k] for k in range(1, 5)]
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_mesh("prism", vertices=v, faces=np.array(f), collision="convex", pos=(0, 0, 0.03 - depth), mass=0.2)
    return alloy.build(scene, device=device)


def _ground_start_pen(sim) -> None:
    sim.step(1)
    w = sim.health()["worlds"][0]
    s = w["start_overlap"]
    assert {s["a"], s["b"]} == {"prism", "ground"} and s["level"] == "fail" and "START_PEN" in w["flags"]
    assert s["depth_mm"] == pytest.approx(28.0, abs=2.5)     # MuJoCo's contacts read ~4 % under (26.2 mm)
    assert not sim.check()["ok"]


@pytest.mark.law
@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_a_body_started_28_mm_into_the_ground_is_start_pen(route, monkeypatch) -> None:
    """Before: the ground plane's own thickness (inf) read as a cutter's (``relax_thin_part_rule``), so the judge left
    every ground contact out: ENERGY_JUMP and LATCHED at 1.25 ms (INVALID), no START_PEN, on every route."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    _ground_start_pen(_prism_in_ground(0.028))


@pytest.mark.law
@pytest.mark.gpu
def test_a_body_started_28_mm_into_the_ground_is_start_pen_on_the_gpu() -> None:
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    _ground_start_pen(_prism_in_ground(0.028, "cuda:0"))


@pytest.mark.law
@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_a_body_standing_on_the_ground_is_not_start_pen(route, monkeypatch) -> None:
    """The ground is judged now: a body standing on it, or 0.5 mm above, starts clean (no START_PEN, ok)."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    for depth in (0.0, -0.0005):
        sim = _prism_in_ground(depth)
        sim.run(0.2)
        w = sim.health()["worlds"][0]
        assert "START_PEN" not in w["flags"] and "start_overlap" not in w and sim.check()["ok"], (depth, w)
