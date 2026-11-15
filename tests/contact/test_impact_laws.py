# SPDX-License-Identifier: Apache-2.0
"""Laws for impact-matched native gains (tests/contact/test_impact_gains.py has the unit laws; physics rulings
2026-11-11 (i)-(iii); 2026-11-13 option A, 2026-11-14 option (1): they go on balls, one-point contacts with no resting
floor, in rk4 entries and in free-body clutter on implicitfast with the friction cap: test_impact_scope.py).
Every law here drops a ball. CPU, MuJoCo C, and the Warp path where marked.

(i) Stability: a ball dropped on a static slab, 10 g to 100 kg, 0.5-10 m/s, at the fixed step and (MuJoCo C) 0.42 ms,
    restitution 0 and 0.5: no NONFINITE, ENERGY_JUMP or TUNNEL; a world that comes to rest rests within 0.3 mm; and
    no world fails PEN_FAIL that today's gains (``native_gains.planning_rule``) keep valid.
(C4) Impact stiffness: a ball of at least 10 kg at the fixed step (1 kg at 0.42 ms) goes no deeper than the card's K
    spring, v sqrt(m / K), where today's gains went 2-8x deeper (14.3-14.9 mm for SAMPLE-CLASSIC's 6.8 kg ball).
(e) Restitution (leaving speed / impact speed), a 4.1 kg steel ball on a static slab: e 0 rebounds under 0.03 at
    0.3125 ms and 0.07 at 1.25 ms (the static rule: 0.13-0.15); e 0.5 and 0.8 within 0.05 of the card at 1-2 m/s on
    rk4 (the ball alone; clutter on MuJoCo C runs rk4 too). In clutter on Warp (8 more resting balls: implicitfast +
    friction cap) within 0.08 (physics ruling 2026-11-13): there the contact runs at omega h about 0.5 (about 6 steps
    per contact) and one contact solve per step resolves the release coarsely, so e scatters with impact speed
    (0.57 / 0.46 at 1 / 2 m/s for e 0.5; the static rule reads a steady 0.56 / 0.54); rk4's stages hold it. Both
    routes, the entry's integrator and cap asserted. Measured by leaving speed, not apex height: the ball leaves at
    zero push while still about 1 mm into the slab, so at e 0 it never clears the surface and the apex read 0 under
    either rule.
Rows (CPU, 2026-11-11): [internal note] scr/WP3.md, 11:xxZ lines.
"""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.contact import mjw_friction_cap as FC
from alloy.scene import integrator_route as IR
from alloy.scene import native_gains as NG

_spec = importlib.util.spec_from_file_location("impact_scope", Path(__file__).with_name("test_impact_scope.py"))
SCOPE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SCOPE)

pytestmark = [pytest.mark.law, pytest.mark.timeout(3000)]
K, G, TOP = 1.0e6, 9.81, 0.2
BAD = {"NONFINITE", "ENERGY_JUMP", "TUNNEL"}


def _drop(m: float, e: float, step, speeds, radius: float, material: str | None = None, today: bool = False,
          dt: float = 0.01, clutter: bool = False):
    """Peak overlap (health, m), end overlap from the position, end speed and flags per world (one per speed), the
    trajectory and the entry (``test_impact_scope.entry``). ``clutter``: 8 more steel balls rest on the slab."""
    scene = alloy.Scene(worlds=len(speeds), dt=dt)
    scene.rigid_step = step
    scene.add_box("slab", size=(1.0, 1.0, TOP), pos=(0, 0, TOP / 2), static=True, restitution=e)
    h = np.array([v * v / (2 * G) for v in speeds])
    pos = np.column_stack([np.zeros(len(speeds)), np.zeros(len(speeds)), TOP + radius + h])
    kw = {"material": material} if material else {"mass": m}
    scene.add_sphere("ball", radius, pos=pos, restitution=e, **kw)
    for k in range(IR.R1_MIN_FREE if clutter else 0):
        scene.add_sphere(f"rest{k}", 0.03, pos=(0.4, -0.42 + 0.12 * k, TOP + 0.03 + 1e-4), material="steel_carbon")
    with NG.planning_rule() if today else contextlib.nullcontext():      # today's gains are written at build
        sim = alloy.build(scene, device="cpu")
    ent = SCOPE.entry(sim)
    tr = sim.run(float(np.sqrt(2 * h.max() / G)) + 1.0, record=["ball"])
    health = sim.health()["worlds"]
    sim.close()
    z, vel = tr["ball.pos"][:, :, 2], np.linalg.norm(tr["ball.vel"], axis=-1)
    return [{"peak": w.get("peak_penetration_m", 0.0), "end": TOP + radius - z[-1, i], "speed": vel[-1, i],
             "flags": set(w["flags"]) | set(w.get("seen_flags", ())), "sev": w["severity"]}
            for i, w in enumerate(health)], tr, ent


# MuJoCo C at both steps, Warp at the fixed step: Warp at 0.42 ms took 3-10 min a case on a loaded CPU host (WP13)
SWEEP = [(route, m, e, step) for route in ("mujoco_c", "warp") for m in (0.01, 1.0, 100.0) for e in (0.0, 0.5)
         for step in ("fixed", 0.00041667) if route == "mujoco_c" or step == "fixed"]


@pytest.mark.compile_heavy
@pytest.mark.parametrize("route,m,e,step", SWEEP, ids=[f"{r}-{m:g}kg-e{e:g}-{s}" for r, m, e, s in SWEEP])
def test_drops_stay_stable_and_fail_no_more_than_today(monkeypatch, route, m, e, step):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp" if route == "warp" else "mjc")      # the suite's default is warp
    radius, speeds = (0.05 if m <= 10.0 else 0.12), (0.5, 2.0, 10.0)
    rows, _, _ = _drop(m, e, step, speeds, radius)
    base, _, _ = _drop(m, e, step, speeds, radius, today=True)
    for v, r, b in zip(speeds, rows, base, strict=True):
        assert not (r["flags"] & BAD), (v, r["flags"])
        assert "PEN_FAIL" not in r["flags"] or "PEN_FAIL" in b["flags"], (v, r, b)
        if r["sev"] != "INVALID" and r["speed"] < 0.01:
            assert r["end"] <= 3.0e-4, (v, r)


C4 = [(10.0, "fixed"), (100.0, "fixed"), (1.0, 0.00041667), (10.0, 0.00041667)]


@pytest.mark.compile_heavy
@pytest.mark.parametrize("m,step", C4, ids=[f"{m:g}kg-{s}" for m, s in C4])
def test_a_heavy_impact_goes_no_deeper_than_the_card_spring(m, step, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")                                       # the rows' route (docstring)
    speeds = (0.5, 2.0, 5.0)
    rows, _, _ = _drop(m, 0.5, step, speeds, 0.05 if m <= 10.0 else 0.12)
    for v, r in zip(speeds, rows, strict=True):
        assert r["peak"] <= v * np.sqrt(m / K), (m, step, v, r["peak"])


def _rebound(e: float, step, clutter: bool = False) -> tuple[list[float], tuple]:
    """Restitution of a 4.1 kg steel ball (r 5 cm) from 1 and 2 m/s: its fastest upward speed after the impact (it
    leaves at zero push) over the impact speed, on 1.25 ms frames; and the entry. Not the apex: the ball leaves while
    still about 1 mm into the slab, so at e 0 it never clears the surface and the apex reads 0 under either rule."""
    speeds = (1.0, 2.0)
    _, tr, ent = _drop(0.0, e, step, speeds, 0.05, material="steel_carbon", dt=0.00125, clutter=clutter)
    vz = tr["ball.vel"][:, :, 2]
    out = []
    for w, v in enumerate(speeds):
        up = int(np.argmax(vz[:, w] > 0.0))                      # first frame moving up: after the impact
        out.append(float(vz[up:, w].max()) / v)
    return out, ent


REBOUND = [(False, 0.0003125, 0.03), (False, 0.00125, 0.07), (True, 0.00125, 0.07)]
E_TOL = {True: 0.08, False: 0.05}                   # capped implicitfast, else (docstring (e))


@pytest.mark.compile_heavy
@pytest.mark.parametrize("route", ["mjc", "warp"])
@pytest.mark.parametrize("clutter,step,e0_max", REBOUND, ids=["rk4-0.3125ms", "rk4-1.25ms", "clutter-1.25ms"])
def test_a_heavy_ball_rebounds_at_its_card(clutter, step, e0_max, route, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    capped = clutter and route == "warp"            # MuJoCo C keeps rk4 for clutter (integrator_route.r1_off)
    want = ("mjINT_IMPLICITFAST", FC.CAP_DEFAULT) if capped else ("mjINT_RK4", None)
    e0, ent = _rebound(0.0, step, clutter)
    assert ent == (*want, route == "mjc"), ent
    assert max(e0) <= e0_max, e0
    for e in (0.5, 0.8):
        got, _ = _rebound(e, step, clutter)
        assert got == pytest.approx([e, e], abs=E_TOL[capped]), (e, step, clutter, route)
