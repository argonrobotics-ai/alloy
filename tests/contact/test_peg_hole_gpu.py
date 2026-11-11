# SPDX-License-Identifier: Apache-2.0
"""CT-1 (precision-pair penetration p99/max, SB-3) and CT-12 (no dropped contacts) on the meshed peg-in-hole.

Scenario per (mode, native): 16 worlds, peg 0.2 mm above the bore bottom with random tilt ≤ 0.2° and lateral offset
≤ 30 µm; settle 0.1 s; 50 N downward ramp over 0.5 s and hold 0.1 s (the A8 ramp); 5 N lateral push at the peg top
in +x then -y for 0.2 s each. dt is the §3.5 L2 step for a slow-approach card (contact_speed 0.05 m/s). Penetration
is GapMonitor's (exact winding-number geometry). Targets (D2): p99 ≤ max_pen = 50 µm, max ≤ 100 µm, on every
(mode, native) a router may pick; hydroelastic on native='newton' is refused (``precision.UNSUPPORTED_NATIVE``).
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.precision import supported_natives
from alloy.core.errors import PrecisionSpecError
from alloy.testing.contact import peg_hole_scene

from peg_harness import advance, l2_dt, make_run, overflow_total, perturb, ramp, record, set_wrench

# 317-612 s per case on one L4 (2026-11-10), over the gpu marker's 600 s default once: the wall time, not the physics
pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1800)]
WORLDS = 16
GUARD = 1.1            # hydroelastic-mujoco's max: hard regression guard on 2 x max_pen, outside the xfail
HYDRO_MAX_GAP = ("GPU spread at the max bound (docs/known_gaps/precision-hydroelastic-side-lean.md, Number): on one "
                 "L4 the deepest single sample read 100.5 um against 2 x max_pen = 100 um in 1 of 3 runs at main "
                 "f207ccaff (p99 19 um against 50 um); its parent 0f647f2a2 passed 3 of 3, which at one failing run "
                 "in six does not establish a difference between them. MuJoCo Warp's GPU contact solve does not "
                 "repeat run to run. Not strict: the other runs pass. p99, tunnels, dropped contacts and the 110 um "
                 "guard fail hard")
DOWN = np.array([0.0, 0.0, -1.0])


def _scenario(run) -> None:
    advance(run, int(0.1 / run.dt))
    ramp(run, DOWN, 0.0, 50.0, 0.5)
    advance(run, int(0.1 / run.dt))
    for d in (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])):
        set_wrench(run, np.tile(d * 5.0, (WORLDS, 1)), at_top=True)
        advance(run, int(0.2 / run.dt))


def _ct1(mode: str, native: str):
    scene = peg_hole_scene(WORLDS, device="cuda:0", mode=mode)
    run = make_run(scene, native, l2_dt(scene))
    perturb(run, np.random.default_rng(1))
    _scenario(run)
    rep = run.monitor.report()
    max_pen = scene.spec.resolved_max_penetration()
    row = dict(metric="CT-1", mode=mode, native=native, worlds=WORLDS, dt=run.dt, steps=run.steps,
               p99_m=float(rep.pair_p99_m[0]), max_m=float(rep.pair_max_m[0]), samples=int(rep.pair_samples[0]),
               tunnels=int(rep.tunnel_events.sum()), rungs=list(run.rungs), reference=rep.reference[0],
               max_pen_m=max_pen, wall_s=run.wall_s, overflow=overflow_total(run),
               supported=native in supported_natives(scene.spec))
    record("ct1", row)
    return row, max_pen


@pytest.mark.smoke
@pytest.mark.xfail(strict=False, raises=AssertionError, reason=HYDRO_MAX_GAP)
def test_ct1_hydro_mujoco():
    row, max_pen = _ct1("hydroelastic", "mujoco")
    if not (row["samples"] > 0 and row["tunnels"] == 0 and row["p99_m"] <= max_pen
            and row["max_m"] <= GUARD * 2 * max_pen and sum(row["overflow"].values()) == 0):
        pytest.fail(f"CT-1 hydroelastic-mujoco outside its hard bounds: {row}")     # not an AssertionError: no xfail
    assert row["max_m"] <= 2 * max_pen, row


def test_ct1_hydro_newton_unsupported():
    """SB-3: no router may pick hydroelastic precision pairs on native='newton' (``UNSUPPORTED_NATIVE``): the layer
    refuses it. The row measured with ``measure_unsupported`` is the evidence (max over 2·max_pen when a lateral push
    wedges the peg at the bore edge; native='mujoco' stays within), not a gate."""
    scene = peg_hole_scene(WORLDS, device="cuda:0", mode="hydroelastic")
    with pytest.raises(PrecisionSpecError, match="native='mujoco'"):
        AlloyContactLayer(scene.model, [scene.spec], native="newton")
    row, _ = _ct1("hydroelastic", "newton")
    assert not row["supported"] and row["samples"] > 0 and row["tunnels"] == 0


@pytest.mark.parametrize(("mode", "native"), [("sdf", "mujoco"), ("sdf", "newton")])
def test_ct1(mode, native):
    row, max_pen = _ct1(mode, native)
    assert row["samples"] > 0 and row["tunnels"] == 0
    assert row["p99_m"] <= max_pen and row["max_m"] <= 2 * max_pen, row
    assert sum(row["overflow"].values()) == 0, row["overflow"]
