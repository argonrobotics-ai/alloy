# SPDX-License-Identifier: Apache-2.0
"""alloy.fit.problem: calib-only targets, setup checks, run settings, start values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench.twins.bounce import LAW as R4
from alloy.bench.twins.incline import LAW as R3
from alloy.bench.types import Condition, LawSpec, RigCard, RigDataset, RigTrial, Tolerance
from alloy.fit.errors import FitError, SplitError
from alloy.fit.problem import calib_means, calib_targets, choose_solver, prepare, run_spec, start_values
from alloy.fit.twins import FitBounceTwin, FitInclineTwin
from alloy.fit.types import FitConfig, FitParam, FitProblem
from alloy.testing.fit import guarded


def _r4_data(values=((0.7, 0.71), (0.69, 0.72)), conds=("h5", "h10")) -> RigDataset:
    trials = []
    for c, (a, b) in zip(conds, values):
        trials += [RigTrial(c, 0, "calib", {"t": np.zeros(2)}, {"restitution": a}),
                   RigTrial(c, 1, "calib", {"t": np.zeros(2)}, {"restitution": b}),
                   RigTrial(c, 2, "valid", {"t": np.zeros(2)}, {"restitution": 0.0})]
    card = RigCard(R4.id, "t", "test/1", {}, {"restitution": {"u95": 0.004}}, {}, {})
    return RigDataset(card, tuple(Condition(c, {"drop_height": 0.05}) for c in conds), tuple(trials), "sha")


def test_targets_read_calib_only():
    data, guard = guarded(_r4_data())
    tg = calib_targets(R4, data)                       # the guard raises if any valid trial field is read
    assert guard.touched == []
    assert [t.key for t in tg.targets] == ["restitution@h5", "restitution@h10"]
    assert tg.targets[0].mean == pytest.approx(0.705) and tg.targets[0].u_instr == 0.004
    assert tg.targets[0].bound == pytest.approx(0.03) and tg.n_calib == 4


def test_condition_filter_and_unknown_condition():
    tg = calib_targets(R4, _r4_data(), only=("h10",))
    assert [c.id for c in tg.conditions] == ["h10"]
    with pytest.raises(FitError):
        calib_targets(R4, _r4_data(), only=("nope",))


def test_no_calib_trials_raises():
    data = _r4_data()
    only_valid = RigDataset(data.card, data.conditions, tuple(t for t in data.trials if t.split == "valid"), "s")
    with pytest.raises(FitError):
        calib_targets(R4, only_valid)


def test_prepare_checks_the_twin_and_leaves():
    p = (FitParam("material:steel.surface.restitution", 0.05, 0.99),)
    setup = prepare(FitProblem(R4, FitBounceTwin(path="native"), _r4_data(), p, solver="xpbd"), FitConfig())
    assert setup.fields == ("restitution",) and setup.solver == "xpbd"
    assert setup.start == {"material:steel.surface.restitution": pytest.approx(0.705)}   # mean calib e
    with pytest.raises(FitError, match="cannot realise"):        # the native path has no stiffness
        prepare(FitProblem(R4, FitBounceTwin(path="native"), _r4_data(),
                           (FitParam("material:steel.surface.contact_stiffness", 1e4, 1e8, "log"),)), FitConfig())
    with pytest.raises(FitError, match="not a fit twin"):
        prepare(FitProblem(R4, object(), _r4_data(), p), FitConfig())
    used = FitBounceTwin(path="native")
    used.worlds.append(Condition("x", {}))
    with pytest.raises(FitError, match="fresh twin"):
        prepare(FitProblem(R4, used, _r4_data(), p), FitConfig())
    two = (FitParam("material:a.surface.restitution", 0.1, 0.9), FitParam("material:b.surface.restitution", 0.1, 0.9))
    with pytest.raises(FitError, match="same twin field"):
        prepare(FitProblem(R4, FitBounceTwin(), _r4_data(), two), FitConfig())


def test_prepare_rejects_non_rig_law_bad_solver_and_objective():
    p = (FitParam("material:steel.surface.restitution", 0.05, 0.99),)
    analytic = LawSpec("A.x", "analytic", {"restitution": Tolerance(abs=0.1, source="t")}, {}, 1e-3, 1, 0.1)
    with pytest.raises(FitError, match="rig law"):
        prepare(FitProblem(analytic, FitBounceTwin(), _r4_data(), p), FitConfig())
    with pytest.raises(FitError, match="does not run on solver"):
        prepare(FitProblem(R4, FitBounceTwin(), _r4_data(), p, solver="vbd"), FitConfig())
    with pytest.raises(FitError, match="objective 'nrmse'"):
        prepare(FitProblem(R4, FitBounceTwin(), _r4_data(), p, objective="nrmse"), FitConfig())


def test_zero_bound_raises():
    p = (FitParam("material:steel.surface.restitution", 0.05, 0.99),)
    law = LawSpec("R.zero", "rig", {"restitution": Tolerance(source="t")}, {"restitution": "mean"}, 1e-3, 1, 0.1)
    with pytest.raises(FitError, match="bound is 0"):
        prepare(FitProblem(law, FitBounceTwin(), _r4_data(), p), FitConfig())


def test_run_spec_overrides():
    q = run_spec(R3, FitConfig(quick=True, per_condition=4))
    assert q.duration_s == R3.quick["duration_s"] and q.worlds_per_condition == 4 and q.production_dt
    s = run_spec(R4, FitConfig(substeps=4))
    assert s.substeps == 4 and not s.production_dt
    assert run_spec(R4, FitConfig()) is R4
    with pytest.raises(FitError):
        run_spec(R4, FitConfig(substeps=0))
    assert choose_solver(FitProblem(R4, FitBounceTwin(), _r4_data(), ())) == R4.solvers[0]


def test_incline_start_values_from_calib_means():
    trials = [RigTrial("c", k, "calib", {"t": np.zeros(2)},
                       {"slip_angle": math.atan(0.4), "mu_kinetic": 0.35, "creep": 0.0}) for k in range(2)]
    card = RigCard(R3.id, "t", "test/1", {}, {}, {}, {})
    data = RigDataset(card, (Condition("c", {}),), tuple(trials), "s")
    tg = calib_targets(R3, data)
    assert calib_means(tg) == pytest.approx({"slip_angle": math.atan(0.4), "mu_kinetic": 0.35, "creep": 0.0})
    from alloy.fit.params import ParamSpace
    space = ParamSpace((FitParam("pair:a__b.mu_static", 0.05, 1.5), FitParam("pair:a__b.mu_dynamic", 0.05, 1.5)))
    start = start_values(FitInclineTwin(), space, ("mu_static", "mu_dynamic"), tg)
    assert start == pytest.approx({"pair:a__b.mu_static": 0.4, "pair:a__b.mu_dynamic": 0.35})


def test_guarded_trial_raises_when_armed():
    data, guard = guarded(_r4_data())
    valid = [t for t in data.trials if t.split == "valid"]
    with pytest.raises(SplitError):
        _ = valid[0].scalars
    guard.armed = False
    assert valid[0].scalars == {"restitution": 0.0}
    assert guard.touched == ["h5#2.scalars"]
