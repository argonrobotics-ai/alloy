# SPDX-License-Identifier: Apache-2.0
"""Fit twins (R3, R4) and the batched evaluator on CPU: hooks, per-world values, virtual conditions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench.twins.bounce import LAW as R4
from alloy.bench.twins.incline import LAW as R3, ROLE_KINETIC
from alloy.bench.types import Condition, RigCard
from alloy.fit.batch import TwinBatch, bind_candidates, split_virtual, virtual_id
from alloy.fit.errors import FitError
from alloy.fit.problem import prepare, run_spec
from alloy.fit.twins import FitBounceTwin, FitInclineTwin, FitValues
from alloy.fit.types import FitConfig, FitParam, FitProblem
from alloy.testing.fit import SyntheticSpec, twin_dataset, twin_truth

DROPS = (Condition("drop5cm", {"drop_height": 0.05}), Condition("drop15cm", {"drop_height": 0.15}))


def test_fit_values_precedence():
    v = FitValues({"mu_static": 0.3})
    assert v.get("c", "mu_static") == 0.3
    v.fix({"mu_static": 0.4})
    v.bind("c~1", {"mu_static": 0.5})
    assert v.get("c", "mu_static") == 0.4 and v.get("c~1", "mu_static") == 0.5
    assert v.get("c", "mu_dynamic", "mu_static") == 0.4            # fallback field
    with pytest.raises(FitError):
        FitValues().get("c", "restitution")


def test_virtual_ids_round_trip_and_binding():
    assert split_virtual(virtual_id("a~b", 7)) == ("a~b", 7)
    tw = FitBounceTwin(path="native")
    groups = bind_candidates(tw, DROPS, ("restitution",), np.array([[0.3], [0.6]]), first=4)
    assert [[c.id for c in g] for g in groups] == [["drop5cm~4", "drop15cm~4"], ["drop5cm~5", "drop15cm~5"]]
    assert tw.values.get("drop15cm~5", "restitution") == 0.6 and groups[1][1].params == {"drop_height": 0.15}


def test_incline_twin_roles_and_kinetic_tilt():
    tw = FitInclineTwin()
    tw.bind_values("c~0", {"mu_static": 0.5, "mu_dynamic": 0.7})        # unphysical mu_d > mu_s
    assert tw.mu_for("c~0", kinetic=False) == 0.5 and tw.mu_for("c~0", kinetic=True) == 0.7
    assert tw.mu_fit["c~0"] == 0.5
    cond = Condition("c~0", {})
    for s in tw.expand(cond, False, 4):
        tw.worlds.append(cond)
        tw.samples.append(s)
    th = tw._theta()
    kin = [i for i, s in enumerate(tw.samples) if s["role"] == ROLE_KINETIC][0]
    assert th[kin] == pytest.approx(math.atan(0.7) + math.radians(5.0))   # slides for any candidate
    tw.bind_values("d~0", {"mu_dynamic": 0.4})                              # one leaf: the single mu
    assert tw.mu_for("d~0", kinetic=False) == 0.4


def test_bind_calib_only_without_values():
    card = RigCard(R4.id, "t", "t/1", {}, {}, {}, {})
    from alloy.bench.types import RigTrial
    calib = [RigTrial("drop5cm", 0, "calib", {}, {"restitution": 0.5})]
    tw = FitBounceTwin(path="native")
    tw.bind_calib(card, calib)                     # no values: WP6's closed form becomes the default
    assert tw.values.get("drop5cm", "restitution") == 0.5
    fixed = FitBounceTwin(path="native")
    fixed.fix_values({"restitution": 0.8})
    fixed.bind_calib(card, calib)
    assert fixed.values.get("drop5cm", "restitution") == 0.8


def test_native_bounce_realises_e_per_world():
    spec = run_spec(R4, FitConfig(quick=True, per_condition=1))
    out = twin_truth(spec, FitBounceTwin(path="native"), {"restitution": 0.6}, DROPS, solver="xpbd", device="cpu")
    for c in DROPS:   # XPBD restitution: e_sim within 0.01 of the card value
        assert out[c.id]["restitution"] == pytest.approx(0.6, abs=0.01)


def test_batch_scores_candidates_independently():
    cfg = FitConfig(worlds=8, quick=True, per_condition=1)
    spec = run_spec(R4, cfg)
    data = twin_dataset(spec, FitBounceTwin(path="native"), {"restitution": 0.6}, DROPS,
                        SyntheticSpec({"restitution": 0.0}, trials_per_condition=4, calib_per_condition=2),
                        solver="xpbd", device="cpu")
    p = (FitParam("material:steel_carbon.surface.restitution", 0.05, 0.99),)
    setup = prepare(FitProblem(R4, FitBounceTwin(path="native"), data, p, solver="xpbd"), cfg)
    batch = TwinBatch(setup, "cpu", worlds=2)      # chunk = 1 candidate: three models, same answers
    res = batch.evaluate(np.array([[0.6], [0.4], [0.8]]))
    assert res.objective[0] == pytest.approx(0.0, abs=1e-12)          # data came from this twin at 0.6
    assert res.objective[1] > 10 and res.objective[2] > 10
    assert res.worlds == 6 and batch.evaluations == 3
    one = TwinBatch(setup, "cpu", worlds=64).evaluate(np.array([[0.6], [0.4], [0.8]]))
    np.testing.assert_allclose(one.objective, res.objective, atol=1e-12)   # chunking does not change scores


def test_damping_twin_pair_table_per_world():
    pytest.importorskip("alloy.scene.pairs", reason="R4 damping path needs WP2's PairTable")
    spec = run_spec(R4, FitConfig(quick=True, per_condition=1, substeps=4))
    lo = twin_truth(spec, FitBounceTwin(), {"restitution": 0.4}, DROPS[:1], solver="mujoco", device="cpu")
    hi = twin_truth(spec, FitBounceTwin(), {"restitution": 0.8}, DROPS[:1], solver="mujoco", device="cpu")
    e_lo, e_hi = lo["drop5cm"]["restitution"], hi["drop5cm"]["restitution"]
    assert 0.3 < e_lo < 0.55 and 0.7 < e_hi < 0.9            # monotonic and near the card value at dt/4


def test_squeeze_twin_compression_falls_with_youngs():
    from alloy.fit.twins import SqueezeTwin
    from alloy.fit.twins.squeeze import LAW as SQ, lame
    assert lame(5e4, 0.3) == pytest.approx((5e4 / 2.6, 5e4 * 0.3 / (1.3 * 0.4)))
    conds = (Condition("F1N", {"force": 1.0}),)
    spec = run_spec(SQ, FitConfig(quick=True))
    soft = twin_truth(spec, SqueezeTwin(cells=2, iterations=30), {"youngs": 5e4}, conds, solver="vbd", device="cpu")
    hard = twin_truth(spec, SqueezeTwin(cells=2, iterations=30), {"youngs": 2e5}, conds, solver="vbd", device="cpu")
    c_soft, c_hard = soft["F1N"]["compression"], hard["F1N"]["compression"]
    assert 0.0 < c_hard < c_soft < 0.01 and math.isfinite(c_soft)
