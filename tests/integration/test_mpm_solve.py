# SPDX-License-Identifier: Apache-2.0
"""The implicit-MPM rheology tolerance (``alloy.integration.mpm_solve``, SCR-WP12-3 (3), coordinator ruling 2026-11-08
10:05Z): set within the ruled range, refused outside it, applied by the MPM factory. CPU."""

from __future__ import annotations

import dataclasses

import pytest

from alloy.core.errors import SolverOptionError
from alloy.core.registry import isolated_registry
from alloy.integration import mpm_solve as M
from alloy.testing import integration as I
from alloy.testing import scene as S


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(M.TOLERANCE_ENV, raising=False)


def test_unset_is_the_ruled_default() -> None:
    """3e-4 since the 2026-11-08 law re-pass (a revert of that commit restores Newton's 1e-4)."""
    assert M.DEFAULT_TOLERANCE == 3e-4 and M.solve_kwargs() == {"tolerance": 3e-4}


@pytest.mark.parametrize("raw, tol", [("3e-4", 3e-4), (" 1e-4 ", 1e-4), ("1e-5", 1e-5)])
def test_a_tolerance_in_the_ruled_range_is_passed(monkeypatch, raw: str, tol: float) -> None:
    monkeypatch.setenv(M.TOLERANCE_ENV, raw)
    assert M.solve_kwargs() == {"tolerance": tol}


@pytest.mark.parametrize("raw", ["1e-3", "5e-4", "0", "-1e-4", "nan", "inf", "tight"])
def test_a_tolerance_outside_the_range_is_refused(monkeypatch, raw: str) -> None:
    """The ruling rejected 1e-3 (every solve stops unconverged): nothing looser than 3e-4."""
    monkeypatch.setenv(M.TOLERANCE_ENV, raw)
    with pytest.raises(SolverOptionError, match=M.TOLERANCE_ENV):
        M.solve_kwargs()


def _solver(kwargs: dict | None = None):
    """sand_heap's MPM solver built on the CPU, with ``kwargs`` added to its entry's solver kwargs."""
    from alloy._vendor.newton.solvers import SolverImplicitMPM
    from alloy.integration import build_simulation

    _, card, lib = next(x for x in S.mixed_scenes() if x[0] == "sand_heap")
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 1)
        if kwargs:
            built = dataclasses.replace(built, entries=[
                dataclasses.replace(e, solver_kwargs={**dict(e.solver_kwargs), **kwargs}) if e.capability == "mpm"
                else e for e in built.entries])
        sim = build_simulation(built, "standard", graph="off")
    return next(s for s in sim.solvers if isinstance(s, SolverImplicitMPM))


def test_the_mpm_factory_applies_the_tolerance_and_an_entrys_own_kwarg_wins(monkeypatch) -> None:
    assert _solver().tolerance == pytest.approx(M.DEFAULT_TOLERANCE or 1e-4)
    monkeypatch.setenv(M.TOLERANCE_ENV, "2e-5")
    assert _solver().tolerance == pytest.approx(2e-5)
    assert _solver({"tolerance": 2e-4}).tolerance == pytest.approx(2e-4)        # the entry's own kwarg wins


def test_an_entrys_own_tolerance_is_held_to_the_range_too() -> None:
    """WP4's review: a card's or entry's solver kwargs cannot pass a looser tolerance than the ruled 3e-4."""
    with pytest.raises(SolverOptionError, match="entry"):
        _solver({"tolerance": 1e-3})


def test_the_route_names_the_tolerance_it_ran_with(monkeypatch) -> None:
    """WP2's review: the environment is not in Scene JSON, so ``sim.routes()`` reasons carry the effective value."""
    assert M.describe() == f"rheology solve tolerance {M.DEFAULT_TOLERANCE or M.NEWTON_TOLERANCE:g} (default)"
    monkeypatch.setenv(M.TOLERANCE_ENV, "1e-4")
    assert M.describe() == f"rheology solve tolerance 0.0001 ({M.TOLERANCE_ENV})"
    _, card, lib = next(x for x in S.mixed_scenes() if x[0] == "sand_heap")
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 1)
    reasons = [r for d in built.report.decisions.values() for r in d.reasons if "rheology solve tolerance" in r]
    assert reasons and all(r.endswith(f"rheology solve tolerance 0.0001 ({M.TOLERANCE_ENV})") for r in reasons)
