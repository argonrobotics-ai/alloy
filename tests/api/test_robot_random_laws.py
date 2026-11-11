# SPDX-License-Identifier: Apache-2.0
"""Laws for ``Scene.randomize`` on robots and gravity: each world's motion shows the value ``sim.randomized()`` says
it drew, at build and after a reset, on both CPU routes (MuJoCo C and MuJoCo Warp).

No law reads the model: each recovers the value from a rest state or a motion, per world, and compares it with
``sim.randomized()`` at a stated tolerance; each also asserts that the worlds' draws spread far wider than that
tolerance. Every row is ``when="reset"``, and every law is measured twice: after the build, and after a reset that
must change the draws of the reset worlds only (``sim.engine.reset(mask)`` on worlds 0 and 2; ``sim.reset()`` for
friction) with the motion following the new values. ``sim.check()["ok"]`` and an empty flag list are part of each law.

==================  ==========================================================================  ===========
field               measurement (4 worlds, dt = 10 ms)                                          tolerance
==================  ==========================================================================  ===========
gravity ``z``       rest sag of a servoed vertical slide: |g| = sag kp / m                      2e-4
gravity ``x``       rest offset of a servoed slide along x: g_x = q kp / m                      1e-3 m/s^2
``mass_scale``      sag against the plain rig's sag in the same world (same gravity)            2e-4
``mass_add``        the same ratio: m0 (ratio - 1)                                              2e-4 kg
both on one link    m = m0 x scale + add from the same ratio                                    2e-4
``kp_scale``        plain sag / sag                                                             2e-4
``com_x/_y/_z``     droop of a servoed hinge holding a link level: lever = kp theta / (m g cos)  2e-5 m
``armature_scale``  period of a mass on a spring slide: m + armature = k T^2 / (4 pi^2)          1e-3
``armature``        the same, absolute                                                          1e-3
``kd_scale``        position servo: slow decay rate L of an overdamped slide, kv = m L + kp / L  1e-2
``kd_scale``        velocity servo: fall speed under gravity, kv = m g / v                       1e-3
``friction``        a pushed box on the robot file's own floor: mu = (F / m - a) / g            2e-2
==================  ==========================================================================  ===========

The kd decay law carries the integrator's first-order error (0.3 % measured at the 1.25 ms step), the friction law
the box's small rocking while it slides (0.2-0.6 % measured). The last test is about the draws alone: build rows
never change, a reset changes exactly the reset worlds, and a world's draw depends on the seed and its id only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest
import warp as wp

import alloy.api as alloy

log = logging.getLogger(__name__)

pytestmark = [pytest.mark.compile_heavy, pytest.mark.timeout(900)]     # a cold MuJoCo Warp compile passes 180 s

W, G, MASKED = 4, 9.81, (0, 2)
M, KP = 2.0, 400.0                       # the servoed slide rig
LEVER, ARM_KP, ARM_M = 0.2, 200.0, 1.0   # the hinge rig
K, SPRING_M, REF = 100.0, 1.0, 0.1       # the spring rig
DECAY_M, DECAY_KP, DECAY_KV = 0.5, 100.0, 40.0
FALL_KV = 200.0
PUSH, PUCK_M = 12.0, 1.0
SOLVERS = {"mjc": "MjcCoreSolver", "warp": "SolverMuJoCo"}
BOX = '<geom type="box" size="0.05 0.05 0.05" mass="{m}" contype="0" conaffinity="0"/>'


def _slide(axis: str) -> str:
    """A 2 kg mass on a slide along ``axis`` held at 0 by a position servo: it rests at m g_axis / kp."""
    return f"""<mujoco><compiler angle="radian"/><worldbody><body name="mass" pos="0 0 0.5">
 <joint name="j" type="slide" axis="{axis}" range="-0.45 0.45" damping="5" armature="0.1"/>{BOX.format(m=M)}
 </body></worldbody><actuator><position joint="j" kp="{KP}" kv="40"/></actuator></mujoco>"""


def _arm(axis: str, at: str, euler: str = "0 0 0") -> str:
    """A 1 kg mass ``LEVER`` out along a level direction from a servoed hinge: it droops by m g lever cos / kp."""
    return f"""<mujoco><compiler angle="radian"/><worldbody><body name="link" pos="0 0 0.5" euler="{euler}">
 <joint name="hinge" type="hinge" axis="{axis}" range="-1 1" damping="2"/>
 <geom type="box" pos="{at}" size="0.05 0.05 0.05" mass="{ARM_M}" contype="0" conaffinity="0"/>
 </body></worldbody><actuator><position joint="hinge" kp="{ARM_KP}" kv="10"/></actuator></mujoco>"""


ARMS = {"cx": ("com_x", _arm("0 1 0", "0.2 0 0")), "cy": ("com_y", _arm("1 0 0", "0 0.2 0")),
        "cz": ("com_z", _arm("0 1 0", "0 0 0.2", euler="0 1.5707963 0"))}      # link z along world x
SPRING = f"""<mujoco><worldbody><body name="mass" pos="0 0 0.5">
 <joint name="slide" type="slide" axis="1 0 0" stiffness="{K}" springref="{REF}" armature="1"/>{BOX.format(m=SPRING_M)}
 </body></worldbody></mujoco>"""
DECAY = f"""<mujoco><worldbody><body name="mass" pos="0 0 0.5">
 <joint name="slide" type="slide" axis="0 1 0"/>{BOX.format(m=DECAY_M)}</body></worldbody>
 <actuator><position joint="slide" kp="{DECAY_KP}" kv="{DECAY_KV}"/></actuator></mujoco>"""
FALL = f"""<mujoco><worldbody><body name="mass" pos="0 0 2"><joint name="drop" type="slide" axis="0 0 1"/>
 {BOX.format(m=M)}</body></worldbody><actuator><velocity joint="drop" kv="{FALL_KV}"/></actuator></mujoco>"""
PUCK = f"""<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1"/>
 <body name="puck" pos="0 0 0.0205"><freejoint/><geom type="box" size="0.1 0.1 0.02" mass="{PUCK_M}"/></body>
 </worldbody></mujoco>"""


@dataclass
class Stage:
    """One measurement: what ``sim.randomized()`` held, what the motion showed, and the health of the run."""

    drawn: dict[str, np.ndarray]
    measured: dict[str, np.ndarray]
    ok: bool
    flags: list[list[str]]


@dataclass
class Law:
    """A scene measured after its build and again after a reset of ``reset_worlds``."""

    stages: list[Stage]
    reset_worlds: tuple[int, ...]


@pytest.fixture(scope="module", params=[pytest.param("mjc", marks=pytest.mark.cpu_route), "warp"])
def route(request):
    """Each CPU route in turn (the suite's conftest pins ``ALLOY_CPU_ROUTE=warp``; "mjc" is the users' default)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALLOY_CPU_ROUTE", request.param)
        yield request.param


def _build(route: str, robots: dict[str, str], rows: list[tuple], worlds: int = W, seed: int = 1) -> Any:
    """``robots`` side by side with the randomize ``rows``, built on ``route`` (asserted from the solver)."""
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    for i, (name, mjcf) in enumerate(robots.items()):
        scene.add_robot(name, mjcf=mjcf, pos=(0.4 * (i % 4) - 0.6, 0.4 * (i // 4), 0.0))   # all within 4 m
    for target, field, lo, hi, *when in rows:
        scene.randomize(target, field, lo, hi, when=when[0] if when else "reset")
    sim = alloy.build(scene, device="cpu", seed=seed)
    assert [type(s).__name__ for s in sim.engine.stepper.solvers()] == [SOLVERS[route]]
    assert {r["solver"] for r in sim.routes().values()} == {"mujoco"}
    return sim


def _masked_reset(sim: Any) -> None:
    mask = np.zeros(sim.worlds, np.int32)
    mask[list(MASKED)] = 1
    sim.engine.reset(wp.array(mask, dtype=wp.int32, device=sim.engine.model.device))


def _measure_twice(sim: Any, measure: Callable[[Any], dict], reset: Callable[[Any], Any] = _masked_reset,
                   reset_worlds: tuple[int, ...] = MASKED) -> Law:
    stages = []
    for again in (False, True):
        if again:
            reset(sim)
        drawn, measured = sim.randomized(), measure(sim)
        flags = [list(w["flags"]) for w in sim.health()["worlds"]]
        stages.append(Stage(drawn, measured, bool(sim.check()["ok"]), flags))
    sim.close()
    return Law(stages, reset_worlds)


def _assert_law(law: Law, key: str, tol: float, *, relative: bool = True, measured: str | None = None,
                expected: Callable[[dict], np.ndarray] | None = None) -> None:
    """Both stages: the worlds differ by far more than ``tol``, each world's measured value is its draw within
    ``tol``, the run is healthy; and the reset changed the draw of exactly ``law.reset_worlds``."""
    for stage in law.stages:
        want = stage.drawn[key] if expected is None else expected(stage.drawn)
        got = stage.measured[measured or key]
        scale = np.abs(want).max() if relative else 1.0
        log.info("%s: max error %.1e (%s, tolerance %.0e)", key, np.abs(got / want - 1.0).max() if relative
                 else np.abs(got - want).max(), "relative" if relative else "absolute", tol)
        assert np.ptp(want) > 20.0 * tol * scale, f"{key}: the draws do not spread: {want}"
        np.testing.assert_allclose(got, want, rtol=tol if relative else 0.0, atol=0.0 if relative else tol,
                                   err_msg=key)
        assert stage.ok and stage.flags == [[]] * W, f"{key}: health {stage.flags}"
    if expected is None:
        changed = law.stages[0].drawn[key] != law.stages[1].drawn[key]
        assert np.flatnonzero(changed).tolist() == list(law.reset_worlds), f"{key}: reset changed worlds {changed}"


# --- loads and servos: gravity z, mass, kp, kd, centre of mass (no armature row in this scene) ---------------------

def _decay_kv(sim: Any) -> np.ndarray:
    """kv of the overdamped servo slide from its slow decay rate after the target steps back to 0."""
    sim.set_targets({"dec/slide": 0.0})
    sim.step(40)                                                  # the fast mode (>= 34 /s) is gone
    traj = sim.run(0.5, record=["dec/slide"])
    q, t = traj["dec/slide.q"], traj["t"]
    rate = np.log(q[0] / q[-1]) / (t[-1] - t[0])
    return DECAY_M * rate + DECAY_KP / rate


def _measure_loads(sim: Any) -> dict[str, np.ndarray]:
    sim.set_targets({"dec/slide": 0.2})
    traj = sim.run(3.0, record=["fall/drop"])
    q, t, drop = sim.state(), traj["t"], traj["fall/drop.q"]
    plain = -q["ref/j.q"]                                     # the unrandomized rig's sag: M |g| / KP
    g = plain * KP / M
    speed = -(drop[-1] - drop[-51]) / (t[-1] - t[-51])
    out = {"gravity.z": -g, "ms.mass_scale": -q["ms/j.q"] / plain, "ma/mass.mass_add": M * (-q["ma/j.q"] / plain - 1),
           "kp.kp_scale": plain / -q["kp/j.q"], "mix.mass": M * -q["mix/j.q"] / plain,
           "fall.kd_scale": M * g / speed / FALL_KV}
    for name, (field, _) in ARMS.items():
        theta = np.abs(q[f"{name}/hinge.q"])
        out[f"{name}/link.{field}"] = ARM_KP * theta / (ARM_M * g * np.cos(theta)) - LEVER
    out["dec.kd_scale"] = _decay_kv(sim) / DECAY_KV
    return out


@pytest.fixture(scope="module")
def loads(route) -> Law:
    robots = {n: _slide("0 0 1") for n in ("ref", "ms", "ma", "kp", "mix")}
    robots.update({n: mjcf for n, (_, mjcf) in ARMS.items()}, dec=DECAY, fall=FALL)
    rows = [("gravity", "z", -12.0, -6.0), ("ms", "mass_scale", 0.5, 2.0), ("ma/mass", "mass_add", -1.0, 2.0),
            ("kp", "kp_scale", 0.5, 2.0), ("mix", "mass_scale", 0.5, 2.0), ("mix/mass", "mass_add", 0.5, 2.0),
            ("dec", "kd_scale", 0.5, 2.0), ("fall", "kd_scale", 0.5, 2.0)]
    rows += [(f"{n}/link", field, -0.1, 0.1) for n, (field, _) in ARMS.items()]
    return _measure_twice(_build(route, robots, rows), _measure_loads)


def test_gravity_z_is_each_worlds_rest_sag(loads):
    _assert_law(loads, "gravity.z", 2e-4)


def test_mass_scale_scales_each_worlds_weight(loads):
    _assert_law(loads, "ms.mass_scale", 2e-4)


def test_mass_add_adds_to_each_worlds_weight(loads):
    _assert_law(loads, "ma/mass.mass_add", 2e-4, relative=False)


def test_mass_scale_and_mass_add_compose(loads):
    _assert_law(loads, "mix", 2e-4, measured="mix.mass",
                expected=lambda d: M * d["mix.mass_scale"] + d["mix/mass.mass_add"])


def test_kp_scale_stiffens_each_worlds_servo(loads):
    _assert_law(loads, "kp.kp_scale", 2e-4)


def test_kd_scale_sets_each_worlds_position_servo_decay(loads):
    _assert_law(loads, "dec.kd_scale", 1e-2)


def test_kd_scale_sets_each_worlds_velocity_servo_fall_speed(loads):
    _assert_law(loads, "fall.kd_scale", 1e-3)


@pytest.mark.parametrize("name", sorted(ARMS))
def test_com_offset_moves_each_worlds_lever(loads, name):
    _assert_law(loads, f"{name}/link.{ARMS[name][0]}", 2e-5, relative=False)


# --- springs: armature, and gravity x ------------------------------------------------------------------------------

def _period(t: np.ndarray, q: np.ndarray) -> float:
    """Mean time between upward crossings of the oscillation's centre (linear interpolation between frames)."""
    s = q - 0.5 * (q.max() + q.min())
    i = np.flatnonzero((s[:-1] < 0.0) & (s[1:] >= 0.0))
    assert len(i) >= 2, "the record holds less than one full period"
    cross = t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i])
    return float((cross[-1] - cross[0]) / (len(cross) - 1))


def _inertia(traj: Any, key: str) -> np.ndarray:
    """m + armature of a spring slide from its period, per world."""
    periods = np.array([_period(traj["t"], traj[key][:, w]) for w in range(W)])
    return K * periods**2 / (4.0 * np.pi**2)


def _measure_springs(sim: Any) -> dict[str, np.ndarray]:
    traj = sim.run(2.4, record=["as/slide", "aa/slide"])          # the longest period is 1.09 s
    return {"as.armature_scale": _inertia(traj, "as/slide.q") - SPRING_M,          # nominal armature 1
            "aa/slide.armature": _inertia(traj, "aa/slide.q") - SPRING_M,
            "gravity.x": sim.state()["gx/j.q"] * KP / M}


@pytest.fixture(scope="module")
def springs(route) -> Law:
    robots = {"as": SPRING, "aa": SPRING, "gx": _slide("1 0 0")}
    rows = [("as", "armature_scale", 0.5, 2.0), ("aa/slide", "armature", 0.2, 2.0), ("gravity", "x", -3.0, 3.0)]
    return _measure_twice(_build(route, robots, rows), _measure_springs)


def test_armature_scale_sets_each_worlds_spring_period(springs):
    _assert_law(springs, "as.armature_scale", 1e-3)


def test_armature_sets_each_worlds_spring_period(springs):
    _assert_law(springs, "aa/slide.armature", 1e-3)


def test_gravity_x_is_each_worlds_sideways_offset(springs):
    _assert_law(springs, "gravity.x", 1e-3, relative=False)


# --- friction ------------------------------------------------------------------------------------------------------

def _measure_friction(sim: Any) -> dict[str, np.ndarray]:
    sim.step(20)
    sim.apply_wrench("rig/puck", force=(PUSH, 0.0, 0.0))
    traj = sim.run(0.4, record=["rig/puck"])
    t, x = traj["t"] - traj["t"][0], traj["rig/puck.pos"][:, :, 0]
    accel = np.array([2.0 * np.polyfit(t[5:], x[5:, w], 2)[0] for w in range(W)])
    return {"rig.friction": (PUSH / PUCK_M - accel) / G}


def test_friction_sets_each_worlds_sliding_drag(route):
    """The robot file's floor and box both take the draw (``"rig"``: every geom of the file), so the pair's friction
    is the draw; ``sim.reset()`` redraws every world and removes the push."""
    sim = _build(route, {"rig": PUCK}, [("rig", "friction", 0.2, 0.8)])
    law = _measure_twice(sim, _measure_friction, reset=lambda s: s.reset(), reset_worlds=tuple(range(W)))
    _assert_law(law, "rig.friction", 2e-2)


# --- the draws: when, masks, seeds ---------------------------------------------------------------------------------

DRAW_ROWS = [("rig", "mass_scale", 0.5, 2.0), ("rig", "kp_scale", 0.5, 2.0, "build"), ("gravity", "z", -12.0, -6.0),
             ("gravity", "x", -1.0, 1.0, "build")]
RESET_KEYS, BUILD_KEYS = ("rig.mass_scale", "gravity.z"), ("rig.kp_scale", "gravity.x")


def _draw_history(route: str, worlds: int = W, seed: int = 1) -> list[dict[str, np.ndarray]]:
    """``sim.randomized()`` after the build, after ``sim.reset()`` and after a masked reset."""
    sim = _build(route, {"rig": _slide("0 0 1")}, DRAW_ROWS, worlds=worlds, seed=seed)
    out = [sim.randomized()]
    sim.reset()
    out.append(sim.randomized())
    _masked_reset(sim)
    out.append(sim.randomized())
    sim.close()
    return out


def test_draws_follow_when_the_mask_the_seed_and_the_world_id(route):
    built, reset, masked = history = _draw_history(route)
    others = [w for w in range(W) if w not in MASKED]
    for key in BUILD_KEYS:
        assert np.array_equal(built[key], reset[key]) and np.array_equal(built[key], masked[key])
    for key in RESET_KEYS:
        assert np.all(built[key] != reset[key])
        assert np.all(reset[key][list(MASKED)] != masked[key][list(MASKED)])
        assert np.array_equal(reset[key][others], masked[key][others])              # bitwise
    eight, reseeded = _draw_history(route, worlds=2 * W), _draw_history(route, seed=2)
    for mine, again, wide, other in zip(history, _draw_history(route), eight, reseeded):
        for key in RESET_KEYS + BUILD_KEYS:
            assert np.array_equal(mine[key], again[key])                            # same seed: same draws
            assert np.all(mine[key] != other[key])
        for key in BUILD_KEYS:
            assert np.array_equal(mine[key], wide[key][:W])                         # world w of 8 is world w of 4
    for key in RESET_KEYS:
        assert np.array_equal(built[key], eight[0][key][:W]) and np.array_equal(reset[key], eight[1][key][:W])
