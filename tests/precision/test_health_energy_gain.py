# SPDX-License-Identifier: Apache-2.0
"""ENERGY_JUMP's gain tier (PHYSA-1, 2026-11-09; CPU): a passive world whose energy rises with nothing adding it.

Before: a free mesh cube with ``collision="sdf"`` pumped itself from rest to 127 mm and 1.0 m/s in 3 s (44 % of
E_scale) with health OK and ``sim.check()["ok"]`` True: under 1 % per step and under the 50 % cumulative bound. And
after a fall the old bounds compare with the start energy, so a pump was unseen until the body was above its start.

Laws (``alloy.precision.health_energy_gain``): a gain of more than 10 % of E_scale over the lowest earlier 0.25 s
block maximum is ENERGY_JUMP (DEGRADED, sticky, ``check()`` not ok), from rest and after a fall; a bounce, a slide
to rest, a pendulum, a teleport, a booked push and what a start releases in its first block are not; a reset clears
it. The pump here is a force written into
``body_f`` that nothing books (the engine's contact bug is WP3's to fix, so no law may need it).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.precision.health_energy_gain import BLOCK_S, GAIN_REL, EnergyGain, energy_ref

M, R, G = 0.5, 0.05, 9.81
SCALE = M * G * 2 * R                           # E_scale of the lone ball: m g L_char, L_char its diameter
DT = 0.004                                      # 62.5 checks per block: no check lands on a block edge
BLOCK = int(BLOCK_S / DT)                       # 62 checks in the first block (the 63rd starts the second)


def _series(net: np.ndarray, *, dt: float | None = DT, particles: bool = False) -> tuple[int | None, float]:
    """(first step the gain trips or None, peak gain) for one world's net energy per check; E_scale = 1 J."""
    model = SimpleNamespace(device="cpu", particle_count=int(particles),
                            particle_world=wp.zeros(1, dtype=wp.int32, device="cpu"))
    gain, ref = EnergyGain(model, 1), energy_ref(1, "cpu")
    ref.e0.assign([net[0]])
    ref.e_scale.assign([1.0])
    gain.rebase(wp.ones(1, dtype=wp.int32, device="cpu"), ref)
    energy, first = wp.zeros(1, dtype=wp.float64, device="cpu"), None
    for k, e in enumerate(net):
        energy.assign([e])
        gain.launch(energy, ref, dt)
        if first is None and gain.state.tripped.numpy()[0]:
            first = k
    return first, float(gain.state.peak.numpy()[0])


def test_a_steady_pump_under_the_per_step_bound_trips_at_the_gain_bound() -> None:
    net = 0.003 * np.arange(200)                                  # +0.3 % of E_scale per check: no per-step jump
    first, peak = _series(net)
    floor = net[BLOCK - 1]                                        # the first block only sets the floor
    assert first == BLOCK + 33 and net[first - 1] - floor < GAIN_REL < net[first] - floor
    assert peak == pytest.approx(net[-1] - floor)


def test_a_pump_after_a_fall_is_measured_from_the_floor_not_the_start() -> None:
    fall, rest = np.linspace(0.0, -3.0, 100), np.full(int(1.0 / DT), -3.0)
    pump = -3.0 + 0.001 * np.arange(1, 201) + 0.0005              # +20 % of E_scale, still 2.8 J under the start
    first, peak = _series(np.concatenate([fall, rest, pump]))
    assert first == 100 + rest.size + 99 and peak == pytest.approx(0.2005)   # the 100th pump step: 10.05 %
    first, peak = _series(np.concatenate([fall, rest, pump]), particles=True)
    assert first is None and peak == 0.0                          # strain energy is not counted: the floor stays


def test_a_dip_shorter_than_a_block_is_a_bounce_not_a_gain() -> None:
    dip = int(0.6 * BLOCK_S / DT)                                 # pressed into the floor: energy health cannot see
    one = np.concatenate([np.zeros(60), np.full(dip, -0.9), np.full(60, -0.05)])
    net = np.concatenate([one, one - 0.05, one - 0.1])            # three bounces, each ending lower
    first, peak = _series(net)
    assert first is None and peak == 0.0


def test_the_first_block_and_a_run_without_step_lengths_are_not_judged() -> None:
    burst = np.concatenate([np.linspace(0.0, 0.4, 20), np.full(180, 0.4)])   # a start overlap pushed apart at once
    assert _series(burst) == (None, 0.0)                          # START_PEN's to report; the floor is 0.4
    assert _series(0.003 * np.arange(200), dt=None) == (None, 0.0)           # no block ever ends: the tier is off


def test_rebase_restarts_the_floor_and_clear_forgets_the_masked_worlds() -> None:
    model = SimpleNamespace(device="cpu", particle_count=0, particle_world=None)
    gain, ref = EnergyGain(model, 2), energy_ref(2, "cpu")
    ref.e_scale.assign([1.0, 1.0])
    every, one = (wp.array(np.array(m, np.int32), dtype=wp.int32, device="cpu") for m in ([1, 1], [0, 1]))
    energy = lambda a, b: wp.array(np.array([a, b]), dtype=wp.float64, device="cpu")  # noqa: E731
    gain.rebase(every, ref)
    for _ in range(BLOCK + 1):
        gain.launch(energy(0.0, 0.0), ref, DT)                    # one block at rest: the floor is 0
    gain.launch(energy(0.2, 0.3), ref, DT)
    assert gain.state.tripped.numpy().tolist() == [1, 1]
    np.testing.assert_allclose(gain.state.peak.numpy(), [0.2, 0.3])
    ref.e0.assign([0.2, 0.3])                                     # a teleport: the reference moves to the state now
    gain.rebase(one, ref)
    gain.clear(one)
    assert gain.state.tripped.numpy().tolist() == [1, 0] and gain.state.peak.numpy()[1] == 0.0
    gain.launch(energy(0.2, 0.9), ref, DT)                        # world 1 is in a first block again: not judged
    assert gain.state.tripped.numpy().tolist() == [1, 0] and gain.state.ready.numpy().tolist() == [1, 0]
    gain.clear()
    assert gain.state.tripped.numpy().tolist() == [0, 0]


def _ball(worlds: int = 2, z: float = R, **kw):
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", R, pos=(0, 0, z), mass=M, **kw)
    return alloy.build(scene, device="cpu")


def _pump(sim, world: int, force: float) -> None:
    """A world-frame x force on world ``world``'s ball that nothing books (``Sim.apply_wrench``'s work is booked)."""
    eng = sim.engine
    for st in (eng.state, eng._next):
        f = st.body_f.numpy()
        f[world, 0] = force
        st.body_f.assign(f)


def _gain(sim) -> np.ndarray:
    return sim.engine.health.gain.state.peak.numpy().copy()


@pytest.mark.parametrize("route", ["mjc", "warp"])      # the users' CPU default (MuJoCo C) and the suite's
def test_a_ball_that_gains_energy_from_nowhere_is_flagged_and_the_check_is_not_ok(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _ball()
    sim.run(0.3)                                    # at rest for a block: the floor is the resting energy
    _pump(sim, 1, 0.5)                              # 0.5 N: 0.71 m/s² rolling, < 0.3 mJ a step (1 % of E_scale: 4.9 mJ)
    sim.run(0.7)                                    # 10 % of E_scale (49 mJ) at ~0.5 s; 50 % (INVALID) only at ~1.2 s
    h, check = sim.health(), sim.check()
    assert h["worlds"][0]["severity"] == "OK" and "energy_gain" not in h["worlds"][0]
    w1 = h["worlds"][1]
    assert w1["severity"] == "DEGRADED" and w1["flags"] == ["ENERGY_JUMP"] and h["degraded_worlds"] == [1]
    ke = 0.5 * 1.4 * M * float((sim.state(["ball"])["ball.vel"][1] ** 2).sum())        # rolling: 7/10 m v²
    assert w1["energy_gain"]["percent"] == pytest.approx(100 * ke / SCALE, rel=0.05) and ke / SCALE < 0.5
    assert w1["energy_gain"]["limit_percent"] == 100 * GAIN_REL and "nothing adding it" in w1["problem"]
    assert not check["ok"]
    [p] = [p for p in check["problems"] if p.get("check") == "energy_gain"]
    assert p["world"] == 1 and p["level"] == "fail" and "gained energy" in p["what"]
    ref = sim.engine.health._ref                    # the older bounds alone saw a gain too small for either
    assert float((ref.e_max.numpy()[1] - ref.e0.numpy()[1]) / ref.e_scale.numpy()[1]) < 0.5
    import builtins                                 # the health-only path (no alloy.check): the same gain row
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name))
                        if name == "alloy.check" else real(name, *a, **k))
    plain = sim.check()
    assert plain["source"] == "health" and not plain["ok"]
    assert [q for q in plain["problems"] if q.get("world") == 1] == [p]


@pytest.fixture
def mjc(monkeypatch) -> None:
    """The users' CPU route (MuJoCo C): the suite's Warp route runs these scenes about 20 s per simulated second."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")


def test_a_pump_after_a_drop_is_flagged_while_the_ball_is_still_below_its_start(mjc) -> None:
    sim = _ball(z=0.3 + R)                          # falls 0.3 m: 3 E_scale lost before anything pumps
    sim.run(1.5)
    assert _gain(sim).max() < 0.2 * GAIN_REL and sim.health()["degraded_worlds"] == []
    _pump(sim, 0, 0.5)
    sim.run(0.7)
    h = sim.health()
    assert h["degraded_worlds"] == [0] and h["worlds"][0]["flags"] == ["ENERGY_JUMP"] and not sim.check()["ok"]
    ref = sim.engine.health._ref                    # never above the start: the older bounds are blind here
    assert float((ref.e_max.numpy()[0] - ref.e0.numpy()[0]) / ref.e_scale.numpy()[0]) < 1e-4
    assert h["worlds"][1]["severity"] == "OK"


def _pendulum():
    arm = """<mujoco><compiler angle="radian"/><worldbody><body name="link" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" ref="0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody></mujoco>"""
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("pend", mjcf=arm, pos=(0, 0, 0.5))
    return alloy.build(scene, device="cpu")


def _drop_bounce():
    return _ball(worlds=1, z=0.5, restitution=0.8)


def _slide_to_rest():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("ramp", size=(0.6, 0.3, 0.02), pos=(0, 0, 0.2), rpy=(0, 0.5, 0), static=True)
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(-0.15, 0, 0.33), rpy=(0, 0.5, 0), mass=0.2, friction=0.3)
    return alloy.build(scene, device="cpu")


def _stack():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for k in range(4):
        scene.add_box(f"b{k}", size=(0.06, 0.06, 0.06), pos=(0.004 * k, 0, 0.04 + 0.07 * k), mass=0.3)
    return alloy.build(scene, device="cpu")


@pytest.mark.parametrize("build", [_drop_bounce, _slide_to_rest, _stack, _pendulum])
def test_healthy_passive_scenes_gain_nothing(build, mjc) -> None:
    sim = build()
    assert sim.engine.health._energy_on             # passive: the energy check runs
    sim.run(2.0)
    h = sim.health()
    assert h["degraded_worlds"] == [] and h["invalid_worlds"] == [] and sim.check()["ok"]
    assert _gain(sim).max() < 0.2 * GAIN_REL        # five times under the bound


def test_a_teleport_a_booked_push_and_a_reset_do_not_leave_a_gain(mjc) -> None:
    sim = _ball()
    sim.run(0.3)
    sim.set_pose("ball", pos=(0, 0, 0.4))           # +0.35 m: 3.5 E_scale, on purpose (the reference moves with it)
    sim.run(1.0)
    sim.apply_wrench("ball", force=(0.5, 0.0, 0.0))  # its work is booked (health_work.WrenchWork)
    sim.run(0.6)
    assert sim.health()["degraded_worlds"] == [] and _gain(sim).max() < 0.2 * GAIN_REL and sim.check()["ok"]
    sim.reset()                                     # the wrench and its booked work are gone: passive again
    sim.run(0.3)
    _pump(sim, 0, 1.0)
    sim.run(0.5)
    assert sim.health()["degraded_worlds"] == [0]
    _pump(sim, 0, 0.0)                              # a raw body_f is not sim.reset()'s to clear
    sim.reset()
    sim.run(0.5)
    assert sim.health()["degraded_worlds"] == [] and _gain(sim).max() < 0.2 * GAIN_REL and sim.check()["ok"]


def test_a_world_with_booked_outside_work_is_judged_by_the_older_bounds_only(mjc) -> None:
    """Main red, 2026-11-10: the SFP insertion rig books its crosshead's work as f . v dt, off by 18 % of E_scale
    at the latch's snap, and the gain tier flagged it. Books are only as exact as the booker, so the tier judges a
    world only while no outside work is booked in it; the 50 % bound still catches books that are far off."""
    sim = _ball()
    sim.run(0.3)
    sim.apply_wrench("ball", force=np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))   # world 0: a booked 0.5 N push
    _pump(sim, 0, 1.0)                              # ... while 1.0 N acts: its books miss half of the work
    _pump(sim, 1, 0.5)                              # world 1: the missing half alone, nothing booked
    sim.run(0.7)
    h, ref = sim.health(), sim.engine.health._ref
    unexplained = (ref.e_max.numpy() - ref.e0.numpy()) / ref.e_scale.numpy()
    assert GAIN_REL < unexplained[0] < 0.5 and GAIN_REL < unexplained[1] < 0.5      # both between the two bounds
    assert h["worlds"][0]["severity"] == "OK" and "energy_gain" not in h["worlds"][0] and _gain(sim)[0] == 0.0
    assert h["worlds"][1]["flags"] == ["ENERGY_JUMP"] and h["degraded_worlds"] == [1]
    sim.run(0.8)                                    # world 0's books are now more than 50 % of E_scale off
    w0 = sim.health()["worlds"][0]
    assert w0["severity"] == "INVALID" and "ENERGY_JUMP" in w0["flags"]


def test_a_rollback_takes_the_flag_back_for_the_restored_worlds_only(mjc) -> None:
    """WP5's review: a pass that spiked and was discarded must not leave a sticky flag on the kept trajectory. The
    engine restores a re-run world's health from ``state_arrays`` (``IntegratedSimulation._health_snapshot``)."""
    from alloy._vendor import newton
    from alloy.core.snapshot import WorldSnapshot
    from alloy.integration.simulation import _health_arrays

    sim = _ball()
    sim.run(0.3)
    health = sim.engine.health
    arrays = {f"health/{k}": v for k, v in _health_arrays(health).items()}
    assert {"health/health.energy_gain_tripped", "health/health.energy_gain_peak"} <= set(arrays)
    snap, state = WorldSnapshot(sim.engine.model, newton.State(), arrays), newton.State()
    snap.save(state)                                # the period start, both worlds at rest
    _pump(sim, 0, 0.5)
    _pump(sim, 1, 0.5)
    sim.run(0.7)                                    # the pass that spikes
    assert health.gain.state.tripped.numpy().tolist() == [1, 1]
    snap.restore(state, wp.array(np.array([0, 1], np.int32), dtype=wp.int32, device="cpu"))   # world 1 is re-run
    assert health.gain.state.tripped.numpy().tolist() == [1, 0] and health.gain.state.peak.numpy()[1] == 0.0
    h = sim.health()
    assert h["degraded_worlds"] == [0] and "energy_gain" not in h["worlds"][1]


BALL_SPRING = """<mujoco><compiler angle="radian"/><worldbody><body name="link" pos="0 0 0.6">
  <joint name="ball" type="ball" stiffness="1.5" damping="0.002"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody></mujoco>"""


def _spring_pendulum():
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", mjcf=BALL_SPRING, pos=(0, 0, 0.4))
    return alloy.build(scene, device="cpu")


def test_a_slow_spring_the_energy_sum_leaves_out_is_not_a_gain(mjc) -> None:
    """WP5's review: a ball-joint spring is not in E. The arm falls from level, the spring catches it and throws it
    back at about 1 Hz, so E dips for longer than a block and comes back: a floor that followed the dip read the
    return as a gain. Such a world keeps its first floor."""
    sim = _spring_pendulum()
    gain = sim.engine.health.gain
    assert sim.engine.health._energy_on and gain.state.lowers.numpy().tolist() == [0]
    sim.run(5.0)
    assert gain.state.tripped.numpy().tolist() == [0] and sim.health()["degraded_worlds"] == [] and sim.check()["ok"]
    assert float(gain.state.peak.numpy()[0]) < 0.2 * GAIN_REL
    lowering = _spring_pendulum()                   # the same world with a floor that follows the block maxima
    lowering.engine.health.gain.state.lowers.fill_(1)
    lowering.run(5.0)
    assert lowering.engine.health.gain.state.tripped.numpy().tolist() == [1]   # the false gain this rule prevents
