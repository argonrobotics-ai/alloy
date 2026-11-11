# SPDX-License-Identifier: Apache-2.0
"""R1 twin end to end on the synthetic dataset, quick mode, through ``alloy.bench.runner.run_law`` (GPU).

Quick set: tilt {1, 2}° × offset 0.05 mm, chamfered; 4 worlds per condition (nominal + 3 jittered), stage 5 mm/s,
12,800 steps at the L2 dt. The synthetic data come from Whitney's quasi-static model (``alloy.contact.rig_synth``),
so the verdicts measure the twin against that model, not against a rig: the test asserts that the pipeline ran
cleanly (no error verdicts, finite state, no dropped contacts) and records every verdict with its numbers, plus the
CT-3 curve error per condition, to bench/results/wp3/r1.jsonl. Physics agreement is reported, not asserted.
Diagnostics: the nominal world of each condition against the model at the same pose, as median sim/model ratios over
the two-point range of the insertion force, the holder's lateral force and moment (friction-independent in two-point
contact: geometry and compliance only) and the tilt; per-world series go to r1_quick_profiles.npz.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from alloy.bench.runner import run_law
from alloy.contact import rig_synth as RS
from alloy.contact.rig_r1 import curve_error, force_profile, mean_profile
from alloy.contact.rig_synth import r1_dataset
from alloy.contact.rig_twins import R1_LAW, R1Twin

from peg_harness import RESULTS, record

pytestmark = pytest.mark.gpu
DEVICE = "cuda:0"


def _num(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def _verdict_rows(res) -> list[dict]:
    return [dict(name=v.name, verdict=v.verdict, real=_num(v.real.mean) if v.real else None,
                 real_std=_num(v.real.std) if v.real else None, real_n=v.real.n if v.real else 0,
                 sim=_num(v.sim.mean), sim_std=_num(v.sim.std), sim_n=v.sim.n, error=_num(v.error),
                 bound=_num(v.bound), tur=_num(v.tur)) for v in res.verdicts]


def _ct3(twin: R1Twin, data) -> tuple[dict[str, float | None], dict[str, np.ndarray]]:
    """CT-3 curve error per condition: twin mean profile vs the mean of the valid trials' profiles."""
    errs, arrays = {}, {}
    for cond in data.conditions:
        trials = [t for t in data.split("valid") if t.condition_id == cond.id]
        z, real = mean_profile([(t.series["depth"], t.series["force"]) for t in trials], twin.proto)
        sim = twin.profiles[cond.id][1]
        errs[cond.id] = _num(curve_error(sim, real))
        arrays.update({f"{cond.id}__z": z, f"{cond.id}__real": real, f"{cond.id}__sim": sim})
    return errs, arrays


def _onset(z: np.ndarray, f: np.ndarray, level: float = 0.3) -> float | None:
    hit = np.flatnonzero(np.nan_to_num(f) > level)
    return float(z[hit[0]]) if hit.size else None


def _diagnose(twin: R1Twin, data) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Nominal world per condition vs the Whitney model at the same pose (median sim/model over two-point depths)."""
    ids, out = [c.id for c in twin.worlds], {}
    for cond in data.conditions:
        w = ids.index(cond.id)                                    # expand: sample 0 is nominal
        pair = RS.WhitneyPair.build(twin.geometry, twin.holder, cond.params["tilt"], cond.params["offset"],
                                    cond.params["chamfer"], twin.mu)
        z = force_profile(np.zeros(1), np.zeros(1), twin.proto)[0]
        qs = RS.quasi_static(pair, z)
        sim = {k: force_profile(twin.series["depth"][w], twin.series[k][w], twin.proto)[1]
               for k in ("force", "fx", "my", "tilt")}
        two = np.flatnonzero(qs.phase == RS.TWO)
        row = {"onset_model_m": _onset(z, qs.P), "onset_sim_m": _onset(z, sim["force"])}
        if two.size:
            sel = (z >= z[two[0]] + 1e-3) & (z <= twin.proto.window[1]) & (qs.phase == RS.TWO)
            for k, model in (("force", qs.P), ("fx", qs.fx), ("my", qs.m), ("tilt", qs.th)):
                row[f"{k}_ratio_median"] = _num(float(np.nanmedian(sim[k][sel] / model[sel])))
        out[cond.id] = row
    arrays = {f"world__{k}": v for k, v in twin.series.items()}
    return out, arrays


def test_r1_quick_synthetic() -> None:
    data = r1_dataset(seed=0, quick=True)
    twin = R1Twin(device=DEVICE, seed=0)
    t0 = time.time()
    res = run_law(R1_LAW, twin, data, device=DEVICE, quick=True)
    wall = time.time() - t0
    mon = res.monitors
    ct3, arrays = _ct3(twin, data) if twin.profiles else ({}, {})
    diag, world_arrays = _diagnose(twin, data) if twin.series else ({}, {})
    arrays.update(world_arrays)
    if arrays:
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez(RESULTS / "r1_quick_profiles.npz", **arrays)
    row = dict(metric="R1", law=res.law, verdict=res.verdict, wall_s=wall, dt=R1_LAW.dt, steps=mon.get("steps"),
               worlds=mon.get("worlds"), stage_speed=mon.get("stage_speed"), verdicts=_verdict_rows(res),
               per_condition=mon.get("per_condition"), world_values={k: [_num(x) for x in v]
                                                                     for k, v in twin.world_values.items()},
               calib_error=mon.get("calib_error"), ct3_curve_error=ct3, overflow=mon.get("overflow"),
               finite=mon.get("finite"), stage_stopped=mon.get("stage_stopped"),
               max_tilt_deg=mon.get("max_tilt_deg"), model_vs_sim=diag, error=mon.get("error"),
               dataset_sha=data.sha256)
    record("r1", row)
    assert "error" not in mon, mon.get("error")
    assert mon["finite"], "non-finite state"
    assert sum(mon["overflow"].values()) == 0, mon["overflow"]
    assert not [v.name for v in res.verdicts if v.verdict == "error"], _verdict_rows(res)
    judged = [v for v in res.verdicts if v.verdict in ("pass", "fail", "inconclusive")]
    assert len(judged) == 6, _verdict_rows(res)       # t1: sliding + jam; t2: all four (t1 has no two-point peak)
