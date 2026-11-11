# SPDX-License-Identifier: Apache-2.0
"""FIT-2: wall time to fit one new object's laws (R3 friction + R4 restitution) on one L4, <= 1 h (SPEC §5.2).

Both fits run at GPU scale (cfg.worlds = 1024 candidate-conditions per batch, 30 generations, the laws' full
durations and world counts) on synthetic twin data at known values, one after the other in one process, and the
summed wall time (data generation excluded, cold kernel compiles included) is FIT-2. Recovery (FIT-3) and FIT-1
rows are recorded from the same runs. R4 runs the §3.4 damping path at dt/4 (see test_fit_synthetic.py).
"""

from __future__ import annotations

import math
import time

import pytest
import warp as wp

from alloy.bench.twins.bounce import LAW as R4
from alloy.bench.twins.incline import LAW as R3
from alloy.bench.types import Condition
from alloy.fit import fit
from alloy.fit.problem import run_spec
from alloy.fit.twins import FitBounceTwin, FitInclineTwin, SqueezeTwin
from alloy.fit.twins.squeeze import LAW as SQUEEZE
from alloy.fit.types import FitConfig, FitParam, FitProblem
from alloy.testing.fit import SyntheticSpec, twin_dataset

BUDGET_S = 3600.0
DEVICE = "cuda:0"
R3_TRUTH = {"mu_static": 0.42, "mu_dynamic": 0.36}
R4_TRUTH = {"restitution": 0.62}
DROPS = tuple(Condition(f"drop{int(h * 100)}cm", {"drop_height": h}) for h in (0.05, 0.10, 0.15))


def _require_cuda() -> None:
    if not wp.is_cuda_available():
        pytest.skip("FIT-2 needs a CUDA device")


def _r3(cfg: FitConfig) -> FitProblem:
    data = twin_dataset(run_spec(R3, cfg), FitInclineTwin(), R3_TRUTH, (Condition("pp_on_steel", {}),),
                        SyntheticSpec({"slip_angle": math.radians(0.05), "mu_kinetic": 0.003, "creep": 1e-7}),
                        solver="mujoco", device=DEVICE)
    params = (FitParam("pair:pp_homopolymer__steel_carbon.mu_static", 0.05, 1.2),
              FitParam("pair:pp_homopolymer__steel_carbon.mu_dynamic", 0.05, 1.2))
    return FitProblem(R3, FitInclineTwin(), data, params, solver="mujoco")


def _r4(cfg: FitConfig) -> FitProblem:
    data = twin_dataset(run_spec(R4, cfg), FitBounceTwin(), R4_TRUTH, DROPS, SyntheticSpec({"restitution": 0.005}),
                        solver="mujoco", device=DEVICE)
    params = (FitParam("material:pp_homopolymer.surface.restitution", 0.05, 0.99),)
    return FitProblem(R4, FitBounceTwin(), data, params, solver="mujoco")


def _rel(res, truth) -> float:
    return max(abs(q.value - truth[k.rsplit(".", 1)[-1]]) / truth[k.rsplit(".", 1)[-1]] for k, q in res.params.items())


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.timeout(int(BUDGET_S) + 600)   # the conftest's 600 s GPU default would cut a 1 h budget
def test_fit2_r3_r4_one_object_under_one_hour(record_row):
    _require_cuda()
    pytest.importorskip("alloy.scene.pairs", reason="R4 damping path needs WP2's PairTable")
    cfg3 = FitConfig(worlds=1024, generations=30, wall_budget_s=BUDGET_S / 2)
    cfg4 = FitConfig(worlds=1024, generations=30, wall_budget_s=BUDGET_S / 2, per_condition=1, substeps=4)
    p3, p4 = _r3(cfg3), _r4(cfg4)
    t0 = time.perf_counter()
    r3 = fit(p3, cfg3, DEVICE)
    r4 = fit(p4, cfg4, DEVICE)
    wall = time.perf_counter() - t0
    gpu = wp.get_device(DEVICE).name
    record_row("FIT-2", wall, "pass" if wall <= BUDGET_S else "fail", unit="s", device=gpu,
               r3_wall_s=r3.wall_s, r4_wall_s=r4.wall_s, r3_evals=r3.evaluations, r4_evals=r4.evaluations,
               r3_worlds=r3.report["worlds"], r4_worlds=r4.report["worlds"])
    for res, truth, law in ((r3, R3_TRUTH, "R3"), (r4, R4_TRUTH, "R4")):
        rel = _rel(res, truth)
        record_row("FIT-3", rel, "pass" if rel <= 0.02 else "fail", law=law, device=gpu,
                   fitted={k: q.value for k, q in res.params.items()}, truth=truth)
        ok = all(v.verdict == "pass" for v in res.verdicts)
        record_row("FIT-1", max(res.valid_error.values()), "pass" if ok else "fail", law=law, device=gpu,
                   valid={v.name: v.verdict for v in res.verdicts})
    assert wall <= BUDGET_S
    assert _rel(r3, R3_TRUTH) <= 0.02 and _rel(r4, R4_TRUTH) <= 0.02


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.timeout(3600)
def test_fit3_k_youngs_squeeze_gpu(record_row):
    """FIT-3 k on the L4: E of the 4-cell VBD cube (100 iterations), hybrid (CMA-ES + Gauss-Newton), log leaf."""
    _require_cuda()
    truth = {"youngs": 6.0e4}
    cfg = FitConfig(worlds=512, generations=30, method="hybrid", grad_steps=5)
    conds = (Condition("F1N", {"force": 1.0}), Condition("F2N", {"force": 2.0}))
    data = twin_dataset(run_spec(SQUEEZE, cfg), SqueezeTwin(), truth, conds, SyntheticSpec({"compression": 2e-5}),
                        solver="vbd", device=DEVICE)
    p = FitProblem(SQUEEZE, SqueezeTwin(), data,
                   (FitParam("material:pu_foam_sponge.elastic.youngs", 1e4, 1e6, scale="log"),), solver="vbd")
    res = fit(p, cfg, DEVICE)
    q = res.params["material:pu_foam_sponge.elastic.youngs"]
    rel = abs(q.value - truth["youngs"]) / truth["youngs"]
    record_row("FIT-3", rel, "pass" if rel <= 0.10 else "fail", law="squeeze", solver="vbd", leaf="youngs",
               device=wp.get_device(DEVICE).name, fitted=q.value, truth=truth, wall_s=res.wall_s,
               gn_unc95=res.report.get("gn_unc95"), evaluations=res.evaluations)
    assert rel <= 0.10 and all(v.verdict == "pass" for v in res.verdicts)


@pytest.mark.gpu
@pytest.mark.smoke
def test_fit_gpu_smoke():
    """A tiny R4 native fit on CUDA: the batched path runs on device (merge smoke, <= 1 min warm)."""
    _require_cuda()
    cfg = FitConfig(worlds=64, generations=2, quick=True, per_condition=1)
    data = twin_dataset(run_spec(R4, cfg), FitBounceTwin(path="native"), R4_TRUTH, DROPS[:1],
                        SyntheticSpec({"restitution": 0.005}, trials_per_condition=6, calib_per_condition=3),
                        solver="xpbd", device=DEVICE)
    p = FitProblem(R4, FitBounceTwin(path="native"), data,
                   (FitParam("material:pp_homopolymer.surface.restitution", 0.05, 0.99),), solver="xpbd")
    res = fit(p, cfg, DEVICE)
    assert abs(res.params["material:pp_homopolymer.surface.restitution"].value - 0.62) < 0.03
