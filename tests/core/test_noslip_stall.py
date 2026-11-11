# SPDX-License-Identifier: Apache-2.0
"""ALLOY_NOSLIP_STALL (opt-in, TRAIN-FIX item 2 B): the scoped noslip stops a world whose sweep lowers the cost by no
more than this fraction of all its sweeps so far, the float32 floor, instead of running every sweep.

On a GPU the launch lasts as long as its slowest world, so worlds stuck at the floor hold the batch (see
test_noslip_sweep_cap.py). Unset, the stop is off and the arithmetic is unchanged: 300 steps of the scene below hash the
same on main 6df958068 and with this change (CPU). Here: on two boxes resting on an incline, noslip over every row at a
tolerance no world reaches, the stop cuts the sweeps to a few and ends within 1e-6 of the full-sweep state.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.contact.noslip import ALL, STALL_ENV, NoslipSpec, install_noslip, stall_fraction
from alloy.core.errors import SolverOptionError
from alloy.testing.scenes import incline


def _run(monkeypatch: pytest.MonkeyPatch, stall: str | None, force_space: bool,
         steps: int = 300) -> tuple[np.ndarray, float]:
    """(body_q after ``steps``, mean solver_niter per world-step) with noslip at 50 sweeps and tolerance 1e-30;
    ``force_space`` picks the Delassus-space sweep loop (noslip_dl) or the qacc-space one (noslip_block)."""
    if stall is None:
        monkeypatch.delenv(STALL_ENV, raising=False)
    else:
        monkeypatch.setenv(STALL_ENV, stall)
    r = incline(2, 0.8, 0.4, device="cpu").runner("mujoco", cone="elliptic")
    r.step(150)                                                      # the boxes rest: contacts in every world
    install_noslip(r.solver, NoslipSpec(50, 1e-30, scope=ALL, force_space=force_space))
    iters = []
    for _ in range(steps):
        r.step(1)
        iters.append(r.solver.mjw_data.solver_niter.numpy().copy())
    return r.state.body_q.numpy().copy(), float(np.mean(iters))


@pytest.mark.compile_heavy
@pytest.mark.parametrize("force_space", [True, False], ids=["force_space", "qacc_space"])
def test_the_stall_stop_ends_floor_sweeps_and_keeps_the_state(monkeypatch: pytest.MonkeyPatch,
                                                              force_space: bool) -> None:
    full_q, full_it = _run(monkeypatch, None, force_space)
    q, it = _run(monkeypatch, "1e-6", force_space)
    print(f"force_space={force_space}: niter {it:.1f} vs {full_it:.1f}, max |dq| {np.abs(q - full_q).max():.2e}")
    assert it < 0.5 * full_it, (it, full_it)                         # 4.0 against 26.2-26.3 (both loops)
    assert np.abs(q - full_q).max() < 1e-6                           # 1.6-1.7e-7


@pytest.mark.parametrize("raw", ["0", "1", "-1e-5", "small"])
def test_a_stall_fraction_outside_zero_to_one_is_refused(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(STALL_ENV, raw)
    with pytest.raises(SolverOptionError, match=STALL_ENV):
        stall_fraction()


def test_unset_or_blank_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STALL_ENV, "  ")
    assert stall_fraction() == 0.0
    monkeypatch.delenv(STALL_ENV)
    assert stall_fraction() == 0.0
