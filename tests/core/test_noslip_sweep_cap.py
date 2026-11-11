# SPDX-License-Identifier: Apache-2.0
"""ALLOY_NOSLIP_SWEEPS (opt-in, TRAIN-FIX item 2): caps the scoped noslip's sweeps at install; unset changes nothing.

On a GPU the scoped noslip's launch lasts as long as its slowest world, so a few worlds that stop improving above the
tolerance hold every world for all the sweeps (PickPlace-v0, 2048 worlds, random actions: 15.5k env steps/s at 100
sweeps, 23.7k at 20; zero actions 48k; one L4). The law here is the mechanism on the CPU: at a tolerance no world
reaches, each world runs exactly the capped number of sweeps, which solver_niter counts after the main solve's.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.contact.noslip import ALL, SWEEPS_ENV, NoslipSpec, capped_sweeps, install_noslip
from alloy.core.errors import SolverOptionError
from alloy.testing.scenes import incline


def _iterations_after_one_step(monkeypatch: pytest.MonkeyPatch, cap: str | None, sweeps: int = 50) -> np.ndarray:
    """solver_niter per world after one step of two boxes on an incline with noslip over every row at tolerance 1e-30
    (no world converges, so each runs all its sweeps)."""
    if cap is None:
        monkeypatch.delenv(SWEEPS_ENV, raising=False)
    else:
        monkeypatch.setenv(SWEEPS_ENV, cap)
    r = incline(2, 0.8, 0.4, device="cpu").runner("mujoco", cone="elliptic")
    r.step(150)                                                      # the boxes rest: contacts on every world
    install_noslip(r.solver, NoslipSpec(sweeps, 1e-30, scope=ALL))
    assert int(r.solver.mjw_model.opt.noslip_iterations) == capped_sweeps(sweeps)
    r.step(1)
    return r.solver.mjw_data.solver_niter.numpy().astype(int)


@pytest.mark.compile_heavy
def test_the_cap_cuts_every_world_to_its_sweeps_and_unset_keeps_them(monkeypatch: pytest.MonkeyPatch) -> None:
    full = _iterations_after_one_step(monkeypatch, None)
    capped = _iterations_after_one_step(monkeypatch, "7")
    above = _iterations_after_one_step(monkeypatch, "500")           # the cap never raises the scene's own value
    assert full.size == 2 and np.array_equal(full - capped, [43, 43]) and np.array_equal(above, full)


@pytest.mark.parametrize("raw", ["0", "-3", "twenty", "2.5"])
def test_a_cap_that_is_not_a_whole_number_of_sweeps_is_refused(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(SWEEPS_ENV, raw)
    with pytest.raises(SolverOptionError, match=SWEEPS_ENV):
        capped_sweeps(100)


def test_unset_or_blank_leaves_the_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SWEEPS_ENV, " ")
    assert capped_sweeps(100) == 100
    monkeypatch.delenv(SWEEPS_ENV)
    assert capped_sweeps(37) == 37
