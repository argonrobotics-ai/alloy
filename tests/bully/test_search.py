# SPDX-License-Identifier: Apache-2.0
"""B1 acceptance (SPEC §7 WP7): on CPU with 64 worlds, B1 finds FakeSolver's seeded instability within 5
generations, and the finding replays 10/10."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.bully.search import search, search_detailed
from alloy.bully.types import BullyLimits, BullySeverity, SearchConfig
from alloy.core.newton_compat import NEWTON_PIN
from alloy.testing.bully import seeded_fake_target, stable_fake_target

CFG = SearchConfig(worlds=64, generations=5, knots=8, horizon_s=2.0, sigma0=0.3, budget_gpu_h=0.05, seed=0)


def test_b1_finds_seeded_instability_without_b0_seed():
    # the search alone (mean at zero force) must climb to the blow-up: no B0 trace to start from
    out = search_detailed(seeded_fake_target(), CFG, BullyLimits(), "cpu", seed_from_b0=False)
    assert out.generations_run == 5
    assert 1 in out.first_trip_generation, out.best_badness
    assert out.first_trip_generation[1] <= 4  # generations 0..4
    s1 = [f for f in out.findings if f.severity == BullySeverity.S1]
    assert s1, out.replays
    assert out.replays[s1[0].monitor] == (10, 10)  # replay 10/10 on CPU


def test_b1_seeded_from_b0_and_spec_signature():
    findings = search(seeded_fake_target(), CFG, BullyLimits(), "cpu")
    assert findings and findings[0].severity == BullySeverity.S1
    f = findings[0]
    assert f.value > f.threshold
    assert len(f.scene_hash) == 64 and f.newton_sha == NEWTON_PIN


def test_stable_twin_gives_no_finding():
    out = search_detailed(stable_fake_target(), SearchConfig(worlds=16, generations=2, seed=1), BullyLimits(), "cpu")
    assert out.findings == []
    assert np.all(np.isfinite(out.best_badness))
    assert max(out.best_badness) < 1.0


def test_budget_stops_the_search():
    out = search_detailed(seeded_fake_target(), SearchConfig(worlds=8, generations=50, budget_gpu_h=1e-9),
                          BullyLimits(), "cpu", seed_from_b0=False, replay_runs=1)
    assert out.generations_run == 0


@pytest.mark.parametrize("seed", [0, 1])
def test_search_is_repeatable(seed):
    cfg = SearchConfig(worlds=16, generations=2, seed=seed)
    a = search_detailed(seeded_fake_target(), cfg, BullyLimits(), "cpu", seed_from_b0=False, replay_runs=1)
    b = search_detailed(seeded_fake_target(), cfg, BullyLimits(), "cpu", seed_from_b0=False, replay_runs=1)
    assert a.best_badness == b.best_badness


def test_ipop_restarts_after_25_generations_under_1_percent_and_doubles_the_population():
    """SCR-WP7-6: the best b improving by under 1 % over 25 generations restarts CMA from the best candidate with
    twice the population."""
    from alloy.bully.cma import BatchedCMAES
    from alloy.bully.search import RESTART_STALL_GENS, _Ipop

    cma = _Ipop(BatchedCMAES(np.full(4, 0.5), 0.3, lam=8, seed=0), 0.3, seed=0)
    for g in range(RESTART_STALL_GENS):
        cma.note(1.0 + 0.0003 * g)  # +0.75 % over 25 generations
    assert not cma.stalled()  # needs 25 generations after the reference one
    cma.note(1.0 + 0.0003 * RESTART_STALL_GENS)
    assert cma.stalled()
    cma.es.ask()
    cma.es.tell(np.arange(8.0))
    best = cma.es.state.best_x.copy()
    cma.restart(worlds=8)
    assert cma.batches == 2 and cma.es.params.lam == 16 and cma.best == []
    np.testing.assert_allclose(cma.es.state.mean, best)
    for g in range(RESTART_STALL_GENS + 1):
        cma.note(1.0 + 0.01 * g)  # 1 % per generation: no restart
    assert not cma.stalled()


def test_b1_runs_a_doubled_population_as_two_batches(monkeypatch: pytest.MonkeyPatch):
    from alloy.bully import search as S

    # make every generation count as stalled: generation 1 restarts and generation 2 runs 2 x 8 candidates
    monkeypatch.setattr(S, "RESTART_STALL_GENS", 1)
    monkeypatch.setattr(S, "RESTART_STALL_REL", 10.0)
    out = search_detailed(stable_fake_target(), SearchConfig(worlds=8, generations=3, seed=2), BullyLimits(), "cpu",
                          seed_from_b0=False)
    assert out.restarts == [1] and out.generations_run == 3 and len(out.best_badness) == 3
