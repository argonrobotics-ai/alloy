# SPDX-License-Identifier: Apache-2.0
"""FIT-3 (recovery of known values from synthetic twin data) and FIT-1 (held-out error) on CPU, SPEC §5.2.

Data come from the fit twin itself at known values plus noise (alloy.testing.fit.twin_dataset), so recovery measures
the fitter, not the twin's model error. Targets: mu +/- 2 % (FIT-3); e +/- 2 % (FIT-3 names no e bound; held to the
mu figure); k +/- 10 % (Young's modulus on the static squeeze twin); FIT-1: every valid verdict passes. Rows go to bench/out/fit/results.jsonl (docs/notes/wp11-results.md).
"""

from __future__ import annotations

import math

import pytest

from alloy.bench.twins.bounce import LAW as R4
from alloy.bench.twins.incline import LAW as R3
from alloy.bench.types import Condition
from alloy.fit import fit
from alloy.fit.problem import run_spec
from alloy.fit.twins import FitBounceTwin, FitInclineTwin, SqueezeTwin
from alloy.fit.twins.squeeze import LAW as SQUEEZE
from alloy.fit.types import FitConfig, FitParam, FitProblem
from alloy.testing.fit import SyntheticSpec, twin_dataset

REL = 0.02
R3_TRUTH = {"mu_static": 0.42, "mu_dynamic": 0.36}
R3_NOISE = SyntheticSpec({"slip_angle": math.radians(0.05), "mu_kinetic": 0.003, "creep": 1e-7})
R4_NOISE = SyntheticSpec({"restitution": 0.005})
DROPS = (Condition("drop5cm", {"drop_height": 0.05}), Condition("drop15cm", {"drop_height": 0.15}))


class _NoStart(FitInclineTwin):
    """R3 twin without the closed-form start: the search alone has to find the values."""

    initial_values = None  # type: ignore[assignment]


def _check(res, truth: dict[str, float], record_row, metric: str, **extra) -> None:
    worst = 0.0
    for leaf, q in res.params.items():
        rel = abs(q.value - truth[leaf.rsplit(".", 1)[-1]]) / truth[leaf.rsplit(".", 1)[-1]]
        worst = max(worst, rel)
        assert q.source == "fit" and q.note.startswith(res.fit_id) and q.unc[0] <= q.value <= q.unc[1]
    verdicts = {v.name: v.verdict for v in res.verdicts}
    ok = worst <= REL
    record_row(metric, worst, "pass" if ok else "fail", fitted={k: q.value for k, q in res.params.items()},
               truth=truth, wall_s=res.wall_s, evaluations=res.evaluations, valid=verdicts, **extra)
    fit1 = all(v == "pass" for v in verdicts.values())
    record_row("FIT-1", max(res.valid_error.values()), "pass" if fit1 else "fail", law=res.report["law"],
               valid_error=res.valid_error, gap=res.heldout_gap)
    assert ok, f"{metric}: worst relative error {worst:.4%} > {REL:.0%}: {res.params}"
    assert fit1, verdicts


def _r3_problem(cfg: FitConfig, twin=None) -> FitProblem:
    spec = run_spec(R3, cfg)
    data = twin_dataset(spec, FitInclineTwin(), R3_TRUTH, (Condition("pp_on_steel", {}),), R3_NOISE,
                        solver="mujoco", device="cpu")
    params = (FitParam("pair:pp_homopolymer__steel_carbon.mu_static", 0.05, 1.2),
              FitParam("pair:pp_homopolymer__steel_carbon.mu_dynamic", 0.05, 1.2))
    return FitProblem(R3, twin or FitInclineTwin(), data, params, solver="mujoco")


def _r4_problem(cfg: FitConfig, path: str, solver: str) -> FitProblem:
    spec = run_spec(R4, cfg)
    data = twin_dataset(spec, FitBounceTwin(path=path), {"restitution": 0.62}, DROPS, R4_NOISE,
                        solver=solver, device="cpu")
    params = (FitParam("material:steel_carbon.surface.restitution", 0.05, 0.99),)
    return FitProblem(R4, FitBounceTwin(path=path), data, params, solver=solver)


@pytest.mark.law
def test_fit3_r3_mu_mujoco(record_row):
    cfg = FitConfig(worlds=16, generations=3, quick=True, per_condition=4)
    res = fit(_r3_problem(cfg), cfg, "cpu")
    _check(res, R3_TRUTH, record_row, "FIT-3", law="R3", solver="mujoco", leaf="mu")


@pytest.mark.law
def test_fit3_r4_e_xpbd_native(record_row):
    cfg = FitConfig(worlds=64, generations=6, quick=True, per_condition=1)
    res = fit(_r4_problem(cfg, "native", "xpbd"), cfg, "cpu")
    _check(res, {"restitution": 0.62}, record_row, "FIT-3", law="R4", solver="xpbd", path="native")


@pytest.mark.law
@pytest.mark.slow
def test_fit3_r4_e_mujoco_damping_path(record_row):
    """R4 through the §3.4 damping path (WP2 PairTable -> per-contact kd -> MuJoCo solref) at dt/4.

    At the law's dt (1.25 ms) a K = 1e6 N/m contact on the 33 g ball lasts 0.58 ms, under one step, and the
    realised e is erratic per drop height (docs/notes/wp11-results.md); dt/4 resolves it (fit_dt_only)."""
    pytest.importorskip("alloy.scene.pairs", reason="R4 damping path needs WP2's PairTable")
    cfg = FitConfig(worlds=64, generations=6, quick=True, per_condition=1, substeps=4)
    res = fit(_r4_problem(cfg, "damping", "mujoco"), cfg, "cpu")
    assert res.report["fit_dt_only"]
    _check(res, {"restitution": 0.62}, record_row, "FIT-3", law="R4", solver="mujoco", path="damping",
           start=res.report["start"])


@pytest.mark.law
@pytest.mark.slow
def test_fit3_r3_mu_search_without_start(record_row):
    cfg = FitConfig(worlds=32, generations=14, quick=True, per_condition=4)
    res = fit(_r3_problem(cfg, _NoStart()), cfg, "cpu")
    assert res.report["start"] == {}
    _check(res, R3_TRUTH, record_row, "FIT-3", law="R3", solver="mujoco", leaf="mu", start="none")


@pytest.mark.law
@pytest.mark.slow
def test_fit3_k_youngs_squeeze_vbd(record_row):
    """Stiffness: E of a VBD soft cube from its settled compression under two loads (log-scale leaf, no start)."""
    truth = {"youngs": 6.0e4}
    cfg = FitConfig(worlds=16, generations=12, quick=True, method="hybrid", grad_steps=3)
    conds = (Condition("F1N", {"force": 1.0}), Condition("F2N", {"force": 2.0}))
    data = twin_dataset(run_spec(SQUEEZE, cfg), SqueezeTwin(cells=3, iterations=60), truth, conds,
                        SyntheticSpec({"compression": 2e-5}), solver="vbd", device="cpu")
    p = FitProblem(SQUEEZE, SqueezeTwin(cells=3, iterations=60), data,
                   (FitParam("material:pu_foam_sponge.elastic.youngs", 1e4, 1e6, scale="log"),), solver="vbd")
    res = fit(p, cfg, "cpu")
    rel = abs(res.params["material:pu_foam_sponge.elastic.youngs"].value - truth["youngs"]) / truth["youngs"]
    record_row("FIT-3", rel, "pass" if rel <= 0.10 else "fail", law="squeeze", solver="vbd", leaf="youngs",
               fitted=res.params["material:pu_foam_sponge.elastic.youngs"].value, truth=truth, wall_s=res.wall_s,
               gn_unc95=res.report.get("gn_unc95"), method=res.report["method"])
    assert rel <= 0.10, res.params
    assert all(v.verdict == "pass" for v in res.verdicts)
