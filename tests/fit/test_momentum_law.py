# SPDX-License-Identifier: Apache-2.0
"""A17.vbd_momentum (coordinator ruling SC8 ~23:30Z, SCR-WP11-2): VBD cloth momentum under a non-uniform load.

The free-particle control must pass (it is the harness check: masses, force applied once per step, no damping). The
law is red on Newton's stock SolverVBD (newton@ac7f7830; the strict xfail turns into a failure the day the stock
solver passes) and green with Alloy's coarse translation step (`alloy.solvers.vbd`, `coarse_step=True`, the
`vbd` capability's default).
docs/notes/wp11-vbd-momentum.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.bench.law import registered_laws
from alloy.bench.runner import run_law
from alloy.fit import momentum_law as ML


def _run(quick: bool, coarse: bool | None):
    return run_law(ML.LAW, ML.VbdMomentumTwin(coarse), None, device="cpu", solver="vbd", quick=quick)


@pytest.fixture(scope="module")
def quick_run():
    return _run(True, coarse=False)


@pytest.fixture(scope="module")
def quick_coarse():
    return _run(True, coarse=True)


def test_law_is_registered_with_sourced_tolerances():
    ids = {spec.id for spec, _ in registered_laws("analytic")}
    assert ML.LAW.id in ids
    assert all(t.abs == 0.01 and t.source for t in ML.LAW.metrics.values())


def test_meshes_and_loads():
    for cond in ML.GROUPS:
        v, t = ML.mesh_for(cond)
        sel = ML.loaded(cond, v)
        assert 0 < sel.sum() < len(v) and t.max() < len(v)
    v, _ = ML.cap()
    assert np.allclose(np.linalg.norm(v, axis=1), 0.04)          # every cap vertex on the sphere


def test_free_particle_control_passes(quick_run, record_row):
    """Harness check: with no elastic energy the momentum gain equals the impulse to float32 precision."""
    free = next(v for v in quick_run.verdicts if v.name == "free_err")
    record_row("A17.free_err (quick, CPU)", free.error, free.verdict)
    assert free.verdict == "pass" and free.error < 1e-3


def test_a17_is_green_on_alloys_default_vbd_path(record_row):
    """SCR-WP11-2 default switch: the `vbd` capability without `coarse_step` (what SB-12 runs) is the coarse step."""
    res = _run(True, coarse=None)
    assert res.monitors["solver_class"] == "SolverVBDMomentum"
    for v in res.verdicts:
        record_row(f"A17.{v.name} (quick, CPU, default vbd path)", v.error, v.verdict)
    assert res.verdict == "pass"


def test_stock_vbd_measures_the_leak(quick_run, record_row):
    """The twin resolves the error SC8 reported: stock VBD's curved-shell momentum is far from F·t (cap ~2.5x on
    Newton ac7f7830, ~0.6x with the membrane Hessian fix of the SCR-WP11-3 pin; either way a leak)."""
    mon = quick_run.monitors
    for k in ("flat_half", "flat_row", "cylinder_bottom", "cap_bottom"):
        record_row(f"A17.{k}:P_over_FT (quick, CPU, stock VBD)", mon[f"{k}:P_over_FT"], "measured")
    assert abs(mon["cap_bottom:P_over_FT"] - 1.0) > 0.3


@pytest.mark.xfail(strict=True, reason="Newton SolverVBD leaks linear momentum (unsolved net residual of the "
                                        "Gauss-Seidel sweep); docs/notes/wp11-vbd-momentum.md")
def test_a17_passes_on_stock_vbd(quick_run):
    assert quick_run.verdict == "pass"


def test_a17_passes_with_the_coarse_step(quick_coarse, record_row):
    """SCR-WP11-2 acceptance item 1: every group within 1 % on Alloy's VBD path with the coarse step."""
    assert quick_coarse.monitors["solver_class"] == "SolverVBDMomentum"
    for v in quick_coarse.verdicts:
        record_row(f"A17.{v.name} (quick, CPU, coarse step)", v.error, v.verdict)
    assert quick_coarse.verdict == "pass"
    assert all(v.error < 1e-3 for v in quick_coarse.verdicts)


@pytest.mark.law
@pytest.mark.parametrize("coarse", [False, True], ids=["stock", "coarse"])
def test_a17_full_run(coarse, record_row):
    """Full duration (0.3 s) on CPU: the rows for docs/notes/wp11-vbd-momentum.md."""
    res = _run(False, coarse)
    label = "coarse step" if coarse else "stock VBD"
    for v in res.verdicts:
        record_row(f"A17.{v.name} (full, CPU, {label})", v.error, v.verdict)
    assert next(v for v in res.verdicts if v.name == "free_err").verdict == "pass"
    assert res.verdict == ("pass" if coarse else "fail")
