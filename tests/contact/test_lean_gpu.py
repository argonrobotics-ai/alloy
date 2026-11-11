# SPDX-License-Identifier: Apache-2.0
"""CT-6: a seated peg pushed 5 N sideways at its top: lean / allowed ≤ 1.05 and wall penetration ≤ 0.05 mm.

Allowed lean: ``d·cosθ + l·sinθ = D`` with l = bore depth (CON-46 baseline: 4° against 1.27°, 0.35 mm). Both
precision modes and both native paths; 16 worlds with small random tilt/offset; L2 dt for a 0.05 m/s card.

The law's value is 1.05. On Newton 1.8 the sdf pair on the MuJoCo Warp path reads 1.0475-1.0554 for the +x push, so
that case is an xfail (docs/known_gaps/precision-sdf-side-lean-direction.md), not strict because the reading
straddles the bound. The cause: the lean past the limit is the penetration of the two wall patches over the 19.9 mm
between them. With one loaded row in each patch (K 2e6 N/m) the rows sink 3.8 and 1.3 µm: 1.026. D22 shares K
equally among the loaded rows of a patch, so each further row that Newton's contact reduction keeps on the rim
(18-22° from the deepest one, 0.4-3 µm deep) takes its share of K and the deepest sinks to 6.2-8.6 µm: 1.035-1.055.
Which rows are kept follows the push direction against the reduction's normal bins. On the old Newton pin +x fell
inside a bin (1.026) and 45° near an edge (1.045); Newton 1.8 turned the bins: +x reads up to 1.055 and 22.5° reads
1.023. ``test_ct6_lean_by_direction`` holds 45° and 135° to 1.05; 7.5° straddles it too (1.046-1.052) and is the
same kind of xfail.

Hard guards on every sdf case, which no xfail covers (``pytest.fail`` is not an AssertionError): lean / allowed ≤
1.08 (16 µm of summed wall penetration: the worst reading, 11 µm, plus the 0.01 a world can swing for a few steps
when its bottom row is re-picked, and 0.015 to spare) and wall penetration ≤ 0.05 mm.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.testing.contact import peg_hole_scene

from peg_harness import advance, l2_dt, make_run, perturb, record, set_wrench, tilt_deg
from test_clearance_gpu import closed_form_angle

pytestmark = pytest.mark.gpu
WORLDS = 16


LEAN_GAP = ("known gap docs/known_gaps/precision-hydroelastic-side-lean.md: a hydroelastic fit leans 1.24-1.34x the "
            "limit with 100-130 um wall penetration (L4, 2026-11-01 to 2026-11-10); sdf 1.02-1.06")
_HYDRO = pytest.mark.xfail(strict=True, raises=AssertionError, reason=LEAN_GAP)
# Not strict: the reading straddles 1.05 (15 runs on one L4, 2026-11-09/10: 1.0475-1.0554, 7 of them under 1.05), so
# a strict xfail would fail every run that reads under the bound. The guards in _hard_guards still fail hard.
# The mark goes, and 1.05 is a hard assert again, when the depth-shared patch stiffness lands (the page's planned fix).
SDF_GAP = ("known gap docs/known_gaps/precision-sdf-side-lean-direction.md: on Newton 1.8 the +x push reads "
           "1.0475-1.0554 on the MuJoCo Warp path (15 runs, 7 under 1.05: not strict); the 1.08 guard and the 50 um "
           "wall penetration bound fail hard; a hard assert again once the patch stiffness is shared by depth")
_SDF_MUJOCO = pytest.mark.xfail(strict=False, raises=AssertionError, reason=SDF_GAP)
# The 7.5 deg push straddles 1.05 the same way (3 runs on one L4, 2026-11-10: 1.0475, 1.0463, 1.0520). Same rules.
SDF_GAP_7P5 = ("known gap docs/known_gaps/precision-sdf-side-lean-direction.md: on Newton 1.8 the push at 7.5 deg "
               "reads 1.0463-1.0520 on the MuJoCo Warp path (3 runs, 2 under 1.05: not strict); the 1.08 guard and "
               "the 50 um wall penetration bound fail hard; a hard assert again once the patch stiffness is shared "
               "by depth")
_SDF_7P5 = pytest.mark.xfail(strict=False, raises=AssertionError, reason=SDF_GAP_7P5)
LAW = 1.05              # lean / allowed
GUARD = 1.08            # sdf regression guard on lean / allowed
WALL_PEN = 5e-5         # [m]


def _lean_row(mode: str, native: str, push_deg: float = 0.0) -> dict:
    """Seat the peg, push 5 N at its top along ``push_deg`` (from +x, in the x-y plane) for 0.5 s; the CT-6 row."""
    scene = peg_hole_scene(WORLDS, device="cuda:0", mode=mode, start_gap=5e-5)
    g = scene.geometry
    run = make_run(scene, native, l2_dt(scene))
    perturb(run, np.random.default_rng(6), lateral=1e-5, tilt_deg=0.05)
    advance(run, int(0.1 / run.dt))
    push = np.array([5.0, 0.0, 0.0])
    if push_deg != 0.0:
        push = 5.0 * np.array([math.cos(math.radians(push_deg)), math.sin(math.radians(push_deg)), 0.0])
    set_wrench(run, np.tile(push, (WORLDS, 1)), at_top=True)
    advance(run, int(0.5 / run.dt))
    allowed = math.degrees(closed_form_angle(g.peg_d, g.bore_d, g.depth))
    lean = tilt_deg(run)
    rep = run.monitor.report()
    row = dict(metric="CT-6", mode=mode, native=native, push_deg=push_deg, dt=run.dt, allowed_deg=allowed,
               lean_max_deg=float(lean.max()), lean_mean_deg=float(lean.mean()), ratio=float(lean.max() / allowed),
               ratio_mean=float(lean.mean() / allowed),
               wall_pen_max_m=float(rep.pair_max_m[0]), p99_m=float(rep.pair_p99_m[0]), rungs=list(run.rungs))
    record("ct6", row)
    return row


def _hard_guards(row: dict) -> None:
    """The sdf bounds an xfail must not swallow: ``pytest.fail`` raises ``Failed``, not ``AssertionError``."""
    if row["ratio"] > GUARD or row["wall_pen_max_m"] > WALL_PEN:
        pytest.fail(f"CT-6 guard: lean / allowed > {GUARD} or wall penetration > {WALL_PEN} m: {row}")


@pytest.mark.parametrize(("mode", "native"), [pytest.param("hydroelastic", "mujoco", marks=_HYDRO),
                                              pytest.param("hydroelastic", "newton", marks=_HYDRO),
                                              pytest.param("sdf", "mujoco", marks=_SDF_MUJOCO), ("sdf", "newton")])
def test_ct6_lean(mode, native):
    row = _lean_row(mode, native)
    if mode == "sdf":
        _hard_guards(row)
    assert row["ratio"] <= LAW, row
    assert row["wall_pen_max_m"] <= WALL_PEN, row


@pytest.mark.parametrize("push_deg", [pytest.param(7.5, marks=_SDF_7P5, id="7.5"), pytest.param(45.0, id="45"),
                                      pytest.param(135.0, id="135")])
def test_ct6_lean_by_direction(push_deg):
    """The sdf pair on the MuJoCo Warp path pushed off the world axes. Newton 1.8, 3 runs each: 7.5° 1.046-1.052
    (xfail, not strict), 45° 1.040, 135° 1.039-1.040."""
    row = _lean_row("sdf", "mujoco", push_deg)
    _hard_guards(row)
    assert row["ratio"] <= LAW, row
