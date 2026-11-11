# SPDX-License-Identifier: Apache-2.0
"""CARD-7 / ROD-1 / ROD-2: the CHAIN cantilever self-test on SolverMuJoCo (CPU) at the grid dt 1.25 ms.

Gate (CPU suite): 30 links, both integrators, γ = 0.5: |sag| ≤ 5 %, release drop ≤ 5 %, period ≤ 2 %.
Sweep (slow): 20/30/40/80 links × {discrete, implicitfast + armature} × γ ∈ {0.5, 5}; the stock-chain regression rows
(≤ 8 % at 20, ≤ 2 % at 80) are checked as reported, not as the gate (SCR-WP2-5)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from alloy.route import rodcheck
from alloy.route.types import RodCard
from alloy.testing import scene as S

L, R, RHO, NU = 0.5, 0.003, 1200.0, 0.3


def cable(links: int, integrator: str, gamma: float) -> RodCard:
    area = math.pi * R * R
    EI = RHO * area * 9.81 * L**3 / gamma
    return RodCard(length=L, radius=R, density=RHO, EI=EI, GJ=EI / (1 + NU), EA=1e9 * area, links=links,
                   gamma=gamma, integrator=integrator)


def run(links: int, integrator: str, gamma: float) -> dict[str, float]:
    out = rodcheck.self_test(cable(links, integrator, gamma), f"test-cable-g{gamma}", use_cache=False)
    for metric, key, bound in (("CARD-7.sag", "sag_err", rodcheck.SAG_BOUND),
                               ("ROD-2.release_drop", "drop_err", rodcheck.DROP_BOUND),
                               ("ROD-2.period", "period_err", rodcheck.PERIOD_BOUND)):
        S.record(metric, out[key], "pass" if abs(out[key]) <= bound else "fail", links=links,
                 integrator=integrator, gamma=gamma, dt=1.25e-3, **{k: v for k, v in out.items() if k != key})
    return out


@pytest.mark.compile_heavy
@pytest.mark.timeout(180)
@pytest.mark.parametrize("integrator", ["discrete", "implicitfast"])
def test_card7_rod2_30_links(integrator):
    out = run(30, integrator, 0.5)
    assert out["finite"] == 1.0
    assert abs(out["sag_err"]) <= 0.03                    # 3 % stretch target (5 % gate)
    assert abs(out["drop_err"]) <= rodcheck.DROP_BOUND
    assert abs(out["period_err"]) <= rodcheck.PERIOD_BOUND
    assert rodcheck.passes(out)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("gamma", [0.5, 5.0])
@pytest.mark.parametrize("integrator", ["discrete", "implicitfast"])
@pytest.mark.parametrize("links", [20, 30, 40, 80])
def test_card7_sweep(links, integrator, gamma):
    out = run(links, integrator, gamma)
    if links == 30:
        assert abs(out["sag_err"]) <= rodcheck.SAG_BOUND
    if links == 20:
        assert abs(out["sag_err"]) <= 0.08
    if links == 80:
        assert abs(out["sag_err"]) <= 0.02


def test_cache_keys_on_rod_physics_not_the_card(tmp_path, monkeypatch):
    """SCR-OSS22-3: cards with the same rod physics (other names, other curves) share one self-test, in-process and
    on disk; another physics field, link count, integrator or dt runs it again; an entry under the old card-hash key
    is found and copied to the new key."""
    monkeypatch.setenv("ALLOY_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(rodcheck, "_MEMO", {})
    calls = []
    monkeypatch.setattr(rodcheck, "_static_test", lambda rod, dt: calls.append((rod, dt)) or {"sag_err": 0.01})
    monkeypatch.setattr(rodcheck, "_release_test", lambda rod, dt: {"drop_err": 0.0, "period_err": 0.0})
    rod = cable(30, "discrete", 0.5)
    assert rodcheck.self_test(rod, "card-a") == rodcheck.self_test(rod, "card-b")
    rodcheck._MEMO.clear()                                              # a new process reads the file
    rodcheck.self_test(rod, "card-c")
    assert len(calls) == 1
    for other in (replace(rod, EI=rod.EI * 1.001), replace(rod, links=31), replace(rod, integrator="implicitfast")):
        rodcheck.self_test(other, "card-a")
    rodcheck.self_test(rod, "card-a", dt=1.0e-3)
    assert len(calls) == 5
    old = replace(rod, radius=R * 1.1)
    path = tmp_path / "rodcheck.json"
    data = json.loads(path.read_text())
    data[rodcheck._legacy_key("old-card", old, 1.25e-3)] = {"sag_err": 0.02, "drop_err": 0.0, "period_err": 0.0,
                                                             "passed": 1.0}
    path.write_text(json.dumps(data))
    rodcheck._MEMO.clear()
    assert rodcheck.self_test(old, "old-card")["sag_err"] == 0.02 and len(calls) == 5
    assert rodcheck._cache_key(old, 1.25e-3) in json.loads(path.read_text())


def test_warm_fills_the_cache_once(tmp_path, monkeypatch):
    """SCR-OSS1-9: warm() routes every library cable (and the test helpers' cable) in both scene contexts and runs
    each self-test once; a second warm, or a fresh process, runs none."""
    monkeypatch.setenv("ALLOY_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(rodcheck, "_MEMO", {})
    calls = []
    monkeypatch.setattr(rodcheck, "_static_test", lambda rod, dt: calls.append(rod) or {"sag_err": 0.0})
    monkeypatch.setattr(rodcheck, "_release_test", lambda rod, dt: {"drop_err": 0.0, "period_err": 0.0})
    first = rodcheck.warm()
    assert first["cables"] == 4 and first["failed"] == 0                               # 3 library cables + helper
    assert first["self_tests_run"] == len(calls) >= 4
    assert {round(r.length, 3) for r in calls} >= {1.0}                                # usb_cable_1m among them
    rodcheck._MEMO.clear()
    assert rodcheck.warm()["self_tests_run"] == 0 and len(calls) == first["self_tests_run"]
