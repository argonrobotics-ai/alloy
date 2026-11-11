# SPDX-License-Identifier: Apache-2.0
"""GPU bully runs on the XPBD box-stack target (SPEC §7 WP7 GPU acceptance, SB-7).

``smoke``: a 128-world, 1-generation, 0.5 s B1 (merge-gate GPU subset, kept short for the shared gate). ``nightly``: the acceptance run (4,096 worlds, 20
generations, 1 L4-hour budget) and SB-7 (300k random-legal episodes). Each writes its JSON record to
``bench/results/bully/`` (``gpu_run.sh`` copies that directory back).
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy.bully.corpus import Corpus
from alloy.bully.rate import blowup_rate_from_severity
from alloy.bully.replay import EpisodeRunner
from alloy.bully.search import alloy_sha, search_detailed
from alloy.bully.stress import random_legal_traces
from alloy.bully.targets import get_target
from alloy.bully.types import BullyLimits, SearchConfig

pytestmark = pytest.mark.gpu
RESULTS = Path("bench/results/bully")


def _cuda() -> str:
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return "cuda:0"


def _write(name: str, rec: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "alloy_sha": alloy_sha(), "device_name": wp.get_device("cuda:0").name,
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    return path


def _b1(worlds: int, generations: int, horizon: float, budget_h: float, corpus: Path) -> dict:
    device = _cuda()
    target = get_target("xpbd_box_stack")
    cfg = SearchConfig(worlds=worlds, generations=generations, horizon_s=horizon, budget_gpu_h=budget_h)
    out = search_detailed(target, cfg, BullyLimits(), device)
    filed = [Corpus(corpus).add(f, out.traces[f.id], 1.0 / target.actions.rate_hz, cfg.seed, target=target.name).name
             for f in out.findings]
    return {"target": target.name, "config": dataclasses.asdict(cfg), "wall_s": out.wall_s,
            "generations_run": out.generations_run, "best_badness": out.best_badness,
            "first_trip_generation": out.first_trip_generation, "replays": out.replays, "filed": filed,
            "findings": [dataclasses.asdict(f) for f in out.findings]}


@pytest.mark.smoke
@pytest.mark.timeout(900)
def test_b1_xpbd_box_stack_smoke(tmp_path):
    rec = _b1(128, 1, 0.5, 0.1, tmp_path / "corpus")
    _write("b1_xpbd_box_stack_smoke", rec)
    assert rec["generations_run"] == 1 and np.all(np.isfinite(rec["best_badness"]))


@pytest.mark.nightly
@pytest.mark.slow
@pytest.mark.timeout(5400)
def test_b1_xpbd_box_stack_4096_within_budget(tmp_path):
    rec = _b1(4096, 20, 2.0, 1.0, RESULTS / "corpus")
    _write("b1_xpbd_box_stack_4096", rec)
    assert rec["wall_s"] <= 3600.0 + 600.0  # budget 1 L4-hour for the search; replays may add minutes
    assert rec["generations_run"] >= 1


@pytest.mark.nightly
@pytest.mark.slow
@pytest.mark.timeout(5400)
def test_sb7_xpbd_box_stack():
    device = _cuda()
    target = get_target("xpbd_box_stack")
    runner = EpisodeRunner(target, 4096, device, BullyLimits())
    sev, t0, episodes = [], time.perf_counter(), 300_032  # 74 batches of 4,096 >= 300k
    for batch in range(episodes // 4096 + (episodes % 4096 > 0)):
        sev.append(runner.run(random_legal_traces(target, 4096, 2.0, seed=batch)).severity)
    s = np.concatenate(sev)
    rate = blowup_rate_from_severity(s)
    _write("sb7_xpbd_box_stack", {"metric": "SB-7", "target": target.name, "episodes": int(s.size),
                                  "rate": dataclasses.asdict(rate), "passes": rate.passes,
                                  "s2_episodes": int(np.sum(s == 2)), "wall_s": time.perf_counter() - t0})
    assert rate.episodes >= 300_000
