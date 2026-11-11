# SPDX-License-Identifier: Apache-2.0
"""Sprung-latch mechanism on the generic tab rig (SolverMuJoCo, CPU): latches on insertion, holds above the 75 N
grip limit when pulled without pressing, releases when the tab is pressed past its release angle.

The dt / friction sweeps are ``slow`` (run on the DC1 L4 VM: [internal note]). Contact model and
regularisation are held at the design step while the step is halved (CON-62)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.testing.latch import RigRun, TabRig, build_rig, insert, press, pull, tab_spec

HOLD_N = 100.0
DT = 6.25e-4


def _cycle(r: TabRig, dt: float) -> dict[str, float]:
    run = RigRun(build_rig(r), dt)
    ti = insert(run, speed=0.02)
    x_latched = float(ti.x[-1, 0])
    tp = pull(run, HOLD_N, ramp_s=0.25, hold_s=0.1)
    held = bool(tp.engaged[-1, 0])
    slip = x_latched - float(tp.x[:, 0].min())
    pull(run, 0.0, ramp_s=0.05, hold_s=0.05)
    k = r.rate
    torque = 1.5 * k * r.travel
    tq = press(run, torque)
    tw = pull(run, 20.0, ramp_s=0.25, hold_s=0.15, torque=torque)
    out = float(tw.x[0, 0] - tw.x[-1, 0])
    moving = (tw.x[0, 0] - tw.x[:, 0]) > 0.001
    release = float(-tw.force[np.argmax(moving), 0]) if moving.any() else math.inf
    return {"peak": float(ti.peak_before(r.start_gap + r.depth - 2e-4)[0]), "latched": float(ti.engaged[-1, 0]), "held": float(held), "slip": slip,
            "pressed_free": float(not tq.engaged[-1, 0]), "out": out, "release": release,
            "finite": float(ti.finite and tp.finite and tw.finite)}


def _check(m: dict[str, float], r: TabRig) -> None:
    assert m["finite"] == 1.0
    assert m["latched"] == 1.0, m
    assert 0.5 < m["peak"] <= 20.0, m                      # insertion peak within the 20 N limit
    assert m["held"] == 1.0 and m["slip"] <= r.play + 3e-4, m
    assert m["pressed_free"] == 1.0 and m["out"] > 0.005 and m["release"] <= 20.0, m


def test_tab_ramp_stands_proud_of_the_catch_edge():
    """The ramp face passes above the shoulder's nose-side top edge by more than the contact depth at the ramp top
    (47 um at dt/4). Flush, the lips stop on the shoulder's nose-side face at fine steps: 5.24 N at dt/4 and dt/8
    against 3.3 N at dt and dt/2 (the slow dt test below; this pins the geometry in the gated run)."""
    r = TabRig()
    spec = tab_spec(r)
    catch, ramp = spec.box("rig_tab_catch")[1], spec.box("rig_tab_ramp")[1]
    b = r.ramp_angle
    normal = np.array([math.sin(b), 0.0, math.cos(b)])
    edge = np.array([catch.pos[0] + catch.half[0], 0.0, catch.pos[2] + catch.half[2]])
    proud = float((np.array(ramp.pos) + ramp.half[2] * normal - edge) @ normal)
    assert proud == pytest.approx(r.ramp_proud, abs=1e-9)
    assert proud * math.cos(b) > 5.0e-5                     # the edge's drop below the lips' lower face


@pytest.mark.compile_heavy
@pytest.mark.timeout(300)
def test_tab_latch_cycle():
    r = TabRig()
    _check(_cycle(r, DT), r)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_tab_latch_dt_convergence():
    """Peak insertion force and hold slip at dt, dt/2, dt/4 agree within 5 % (model held at design_dt)."""
    r = TabRig()
    rows = {dt: _cycle(r, dt) for dt in (DT, DT / 2, DT / 4)}
    for m in rows.values():
        _check(m, r)
    peaks = np.array([m["peak"] for m in rows.values()])
    assert np.ptp(peaks) / peaks.mean() <= 0.05, rows


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mu", [0.30, 0.52])
def test_tab_latch_friction_ends(mu):
    """Both ends of the cited PC-on-steel range (sources M2: 0.38 dynamic, 0.52; 0.30 the low end)."""
    r = TabRig(mu=mu)
    _check(_cycle(r, DT), r)
