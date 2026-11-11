# SPDX-License-Identifier: Apache-2.0
"""JOINT_FAIL law for a torn MuJoCo weld / connect, read the way users read it (audit ERR-11, GAP-E3; CPU).

Rig (the audit's i03): a 5 kg body ``a`` on an x slide (position servo) and a free 20 kg body ``b``, joined by a
default-solref equality at b's centre; a held ``sim.apply_wrench`` pulls b sideways (y) in each world.

The bound is derived, not fitted. A MuJoCo equality is a soft constraint: a row carrying force F rests at
    r = F · W · (1 - d) / (d · k),   k = d / (dmax² τ² ζ²),   d = d(r) in [d0, dmax]
(MuJoCo "Solver parameters": reference acceleration -b v - k r, regulariser R = (1 - d)/d · diagApprox;
engine_core_constraint.c, mj_makeImpedance / getimpedance). W is the row's diagApprox, the bodies' translational
``body_invweight0`` summed: 1/m_b for the free body plus 1/(3 m_a) for the slide body (its one axis averaged over
three). SPEC §4.2 / the ladder: JOINT_FAIL when the violation is over max(1e-3 m, 3 x that steady state at the design
load). The design load here is b's weight (no design force is declared), so whatever d the rule is read at, the limit
lies between the 1 mm floor and ``LIMIT_MAX`` = 3 x the d0 (softest) sag under the weight, 3.06 mm.

Laws:
- the simulated violation follows the formula (so the bound stands on the rig's own physics);
- a world whose violation is under the floor is OK with no flag; a world over ``LIMIT_MAX`` lists JOINT_FAIL in
  ``sim.health()``, DEGRADED (the ladder's severity for JOINT_FAIL), each world on its own load;
- DEGRADED passes ``sim.check()["ok"]`` (AGENTS.md §8: ok is False for INVALID worlds only);
- latch semantics: only INVALID is sticky in the record (health_kernels.classify); a DEGRADED flag is the latest
  check's, so once the load is released and the weld is back under the floor the world reads OK again.

Before the fix the "standard" cases fail: the weld check runs on exchanges 0, 10, 20, ... (full_every=10), the next
cheap check overwrites its flag, and a control period (8 exchanges at dt=0.01) always ends on an odd exchange, so
``sim.health()`` never saw it (7 / 23 mm gaps read OK). The "precision" case (full check every exchange) passed.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import alloy
from alloy.core.defaults import JOINT_LIMIT_FLOOR_M
from alloy.precision.ladder import LADDER
from alloy.precision.types import Severity

M_A, M_B, G = 5.0, 20.0, 9.81
TAU, ZETA = 0.02, 1.0                                   # MuJoCo's default equality solref
D0, DMAX, WIDTH, MID, POWER = 0.9, 0.95, 0.001, 0.5, 2.0  # and solimp
K0 = 1.0 / (DMAX**2 * TAU**2 * ZETA**2)                 # 2770 s^-2
W_ROW = 1.0 / M_B + 1.0 / (3.0 * M_A)                   # diagApprox of a translation row [1/kg]
OFFSET = np.array([0.1, 0.0, 0.0])                      # b - a as built: the equality's target
PULL_N = (100.0, 3000.0, 10000.0)                       # world 0 quiet; worlds 1, 2 torn
EQUALITIES = {"weld": '<weld body1="a" body2="b"/>',
              "connect": '<connect body1="a" body2="b" anchor="0.1 0 0"/>'}   # at b's centre: no swing
CASES = [("mjc", "standard", "weld"), ("warp", "standard", "weld"), ("mjc", "precision", "weld"),
         ("mjc", "standard", "connect"),
         ("mjc", "fast", "weld"), ("warp", "fast", "weld"),       # audit ERR-13: fast never measured joints
         # the hold flag is fixed when a period's CUDA graph is captured: the graph key must split the patterns
         pytest.param(("cuda", "standard", "weld"), marks=pytest.mark.gpu)]


def _mjcf(equality: str) -> str:
    return f"""<mujoco><compiler angle="radian"/><worldbody>
<body name="a" pos="0 0 0.6"><joint name="ja" type="slide" axis="1 0 0" range="-1 1"/>
  <geom type="box" size="0.03 0.03 0.03" mass="{M_A}"/></body>
<body name="b" pos="0.1 0 0.6"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="{M_B}"/></body>
</worldbody><equality>{equality}</equality>
<actuator><position joint="ja" kp="5000" kv="200"/></actuator></mujoco>"""


def _impedance(r: float) -> float:
    """MuJoCo's d(r) for the default solimp (getimpedance): d0 at r = 0 rising to dmax at r = width."""
    x = min(abs(r) / WIDTH, 1.0)
    y = x**POWER / MID ** (POWER - 1) if x <= MID else 1.0 - (1.0 - x) ** POWER / (1.0 - MID) ** (POWER - 1)
    return D0 + y * (DMAX - D0)


def _sag(force: float, d: float) -> float:
    return force * W_ROW * (1.0 - d) / (d * d * K0)


def steady_violation(force: float) -> float:
    """Self-consistent r = sag(F, d(r)) [m] by bisection (sag falls as r grows)."""
    lo, hi = 0.0, _sag(force, D0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _sag(force, _impedance(mid)) > mid else (lo, mid)
    return 0.5 * (lo + hi)


WEIGHT = M_B * G
LIMIT_MAX = max(JOINT_LIMIT_FLOOR_M, 3.0 * _sag(WEIGHT, D0))               # 3.06 mm: no reading of §4.2 is looser
EXPECTED = np.array([steady_violation(math.hypot(f, WEIGHT)) for f in PULL_N])   # 0.66, 7.0, 23.3 mm
# Flags that leave a world OK (the ladder's OK rows): a GPU world always carries DET_DOWNGRADED, so "no flag" means none
# of the others.
NOTICES = frozenset(r.key for r in LADDER if r.severity == Severity.OK)


def _problems(row: dict[str, Any]) -> list[str]:
    """A health row's flags without the notices."""
    return [f for f in row["flags"] if f not in NOTICES]


@dataclass(frozen=True)
class Reading:
    violation_m: np.ndarray       # per world, |(b - a) - OFFSET|
    health: dict[str, Any]
    check: dict[str, Any]


@dataclass(frozen=True)
class RigRun:
    loaded: Reading               # after 1 s of the held pull
    released: Reading             # 1 s after the pull is taken off


def _read(sim) -> Reading:
    st = sim.state(["rig/a", "rig/b"])
    gap = np.linalg.norm(st["rig/b.pos"] - st["rig/a.pos"] - OFFSET, axis=1)
    return Reading(gap, sim.health(), sim.check())


@pytest.fixture(scope="module", params=CASES, ids=["-".join(getattr(c, "values", (c,))[0]) for c in CASES])
def rig(request) -> RigRun:
    route, profile, kind = request.param
    with pytest.MonkeyPatch.context() as mp:
        if route != "cuda":                        # ALLOY_CPU_ROUTE takes only "mjc" / "warp" (a GPU build reads it too)
            mp.setenv("ALLOY_CPU_ROUTE", route)    # users' CPU default (MuJoCo C) and the suite's (MuJoCo Warp)
        scene = alloy.Scene(worlds=len(PULL_N), dt=0.01)
        scene.add_ground()
        scene.add_robot("rig", mjcf=_mjcf(EQUALITIES[kind]))
        sim = alloy.build(scene, device="cuda:0" if route == "cuda" else "cpu", profile=profile,
                          welds="mujoco")               # MuJoCo's soft row: the law tears it on purpose
        sim.step(5)
        sim.apply_wrench("rig/b", force=np.array([[0.0, f, 0.0] for f in PULL_N]))
        sim.run(1.0)                               # 50 solref time constants: the equality rests at its sag
        loaded = _read(sim)
        sim.apply_wrench("rig/b", force=np.zeros((len(PULL_N), 3)))
        sim.run(1.0)
        return RigRun(loaded, _read(sim))


def test_the_bound_is_derived_and_the_loads_straddle_it() -> None:
    assert K0 == pytest.approx(2770.1, rel=1e-4) and W_ROW == pytest.approx(0.11667, rel=1e-4)
    assert LIMIT_MAX == pytest.approx(3.06e-3, rel=5e-3)
    assert EXPECTED[0] < JOINT_LIMIT_FLOOR_M                 # under the floor: no reading of the rule flags it
    assert (EXPECTED[1:] > 2.0 * LIMIT_MAX).all()            # over the loosest limit, with a factor 2 to spare
    row = next(r for r in LADDER if r.key == "JOINT_FAIL")
    assert row.severity == Severity.DEGRADED and row.upper is None


@pytest.mark.law
def test_the_violation_follows_mujocos_soft_constraint(rig: RigRun) -> None:
    np.testing.assert_allclose(rig.loaded.violation_m, EXPECTED, rtol=0.03)
    assert (rig.released.violation_m < JOINT_LIMIT_FLOOR_M).all()          # back to the sag under b's weight
    np.testing.assert_allclose(rig.released.violation_m, steady_violation(WEIGHT), rtol=0.05)


@pytest.mark.law
def test_a_torn_equality_is_joint_fail_in_its_own_world(rig: RigRun) -> None:
    worlds = rig.loaded.health["worlds"]
    assert worlds[0]["severity"] == "OK" and _problems(worlds[0]) == []    # 100 N: 0.66 mm, under the floor
    for w in (1, 2):                                                       # 3 kN: 7 mm; 10 kN: 23 mm
        assert "JOINT_FAIL" in worlds[w]["flags"], (w, worlds[w], rig.loaded.violation_m[w])
        assert worlds[w]["severity"] == "DEGRADED"                         # the ladder's severity for JOINT_FAIL
        assert worlds[w]["fixes"] and "joint" in worlds[w]["problem"]
    assert rig.loaded.health["degraded_worlds"] == [1, 2] and rig.loaded.health["invalid_worlds"] == []


@pytest.mark.law
def test_a_degraded_joint_fail_passes_check(rig: RigRun) -> None:
    assert "JOINT_FAIL" in rig.loaded.health["worlds"][2]["flags"]         # the case under test is the flagged one
    assert rig.loaded.check["ok"] is True                                  # ok is False for INVALID worlds only
    assert rig.loaded.check.get("invalid_worlds", []) == []


@pytest.mark.law
def test_joint_fail_clears_when_the_load_is_released(rig: RigRun) -> None:
    assert "JOINT_FAIL" in rig.loaded.health["worlds"][1]["flags"]         # it was up while the weld was torn
    for w, row in enumerate(rig.released.health["worlds"]):                # DEGRADED flags are not sticky
        assert row["severity"] == "OK" and _problems(row) == [], (w, row)
    assert rig.released.check["ok"] is True


FLIP_N, FLIP_PERIODS = 30000.0, 5                # the audit's ERR-13 probe: the pull reverses every 5 periods


def _flipping_pull_seen(full_every: int | None) -> tuple[float, list[str]]:
    """Stiff welds at a pinned 20 ms rigid step (one physics step per control period): a 30 kN pull that reverses
    every 5 periods opens the weld 2.7-3.2 mm at each half-cycle's end and swings it through 0 after each reversal.
    Returns (largest opening seen at period ends, m; the world's seen_flags)."""
    scene = alloy.Scene(dt=0.02)
    scene.rigid_step = 0.02
    scene.add_ground()
    scene.add_robot("rig", mjcf=_mjcf(EQUALITIES["weld"]))
    sim = alloy.build(scene, device="cpu", welds="stiff")
    if full_every is not None:
        monitor = sim.engine.health
        monitor.cfg = dataclasses.replace(monitor.cfg, full_every=full_every)
    sim.step(5)
    peak = 0.0
    for k in range(60):
        pull = FLIP_N if (k // FLIP_PERIODS) % 2 == 0 else -FLIP_N
        sim.apply_wrench("rig/b", force=np.array([0.0, pull, 0.0]))
        sim.step(1)
        st = sim.state(["rig/a", "rig/b"])
        peak = max(peak, float(np.linalg.norm(st["rig/b.pos"] - st["rig/a.pos"] - OFFSET)))
    return peak, sim.health()["worlds"][0]["seen_flags"]


@pytest.mark.law
def test_a_load_in_step_with_the_measurement_is_seen_when_joints_are_measured_every_step() -> None:
    """Measured every 10th physics step (standard), a load whose cycle is 10 steps is always measured at the same
    phase, right after it reverses, where the weld passes through 0: a 2.7-3.2 mm opening read 0.3-0.6 mm and no flag
    (audit ERR-13's second case). Measured every step, the opening is seen."""
    peak, seen = _flipping_pull_seen(full_every=1)
    assert peak > 2.0 * JOINT_LIMIT_FLOOR_M and "JOINT_FAIL" in seen, (peak, seen)


@pytest.mark.law
def test_measuring_every_tenth_step_can_miss_a_load_in_step_with_it() -> None:
    """The limit as it stands for profile standard (docs/guide/health.md, JOINT_FAIL row): kept as a law so that
    measuring more often, or less, is a visible change."""
    peak, seen = _flipping_pull_seen(full_every=None)
    assert peak > 2.0 * JOINT_LIMIT_FLOOR_M and "JOINT_FAIL" not in seen, (peak, seen)
