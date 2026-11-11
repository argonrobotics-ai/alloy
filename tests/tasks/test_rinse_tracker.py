# SPDX-License-Identifier: Apache-2.0
"""Pan sanitizer clock and progress-v1 (alloy/tasks/rinse_tracker.py), host side only."""

import math
from dataclasses import replace

from alloy.tasks.rinse_tracker import (PanSample, PanTracker, carried_detergent_kg, clean_term, pan_progress,
                                         pan_score, rinsed_term, soap_limit_kg)
from alloy.tasks.rinse_score import Verdict

LIMIT = soap_limit_kg(0.14)


def free(t: float, **kw) -> PanSample:
    return PanSample(t=t, gauge=0.0, carried=0.0, z=0.9, inside_wet=0.0, **kw)


def dunk(t: float, **kw) -> PanSample:
    base = dict(pool="sanitize", pool_frac=1.0, touch_sanitizer=True, in_bowl=True, inside_wet=0.95)
    return replace(free(t), **{**base, **kw})


def run_dunk(tr: PanTracker, t0: float, n: int, **kw) -> float:
    for i in range(n):
        tr.sample(dunk(t0 + 0.01 * i, **kw))
    return t0 + 0.01 * n


def test_qualifying_dunk_accrues_30s():
    tr = PanTracker(LIMIT)
    tr.sample(free(0.0))
    run_dunk(tr, 0.01, 3000)
    assert tr.session_ok and tr.live
    assert math.isclose(tr.sanitized_s, 30.0, abs_tol=1e-6)
    assert tr.rinse_owed is None and tr.entries[0].qualifies


def test_mouth_down_reads_zero():
    tr = PanTracker(LIMIT)
    tr.sample(free(0.0))
    run_dunk(tr, 0.01, 500, inside_wet=0.84)
    assert tr.session_ok and not tr.live and tr.sanitized_s == 0.0


def test_wash_water_resets():
    tr = PanTracker(LIMIT)
    tr.sample(free(0.0))
    t = run_dunk(tr, 0.01, 200)
    assert tr.sanitized_s > 1.9
    tr.sample(dunk(t, other_wet="wash"))
    assert tr.sanitized_s == 0.0 and not tr.session_ok
    assert "wash" in tr.resets[-1][1]


def test_dirty_entry_never_starts_and_owes_rinse():
    tr = PanTracker(LIMIT)
    tr.sample(replace(free(0.0), gauge=2e-5))
    run_dunk(tr, 0.01, 100, gauge=2e-5)
    assert tr.sanitized_s == 0.0 and not tr.session_ok
    assert tr.rinse_owed is not None and not tr.entries[0].clean


def test_entry_uses_max_of_prior_sample():
    tr = PanTracker(LIMIT)
    tr.sample(replace(free(0.0), carried=2 * LIMIT))
    tr.sample(dunk(0.01))                  # film given to the bowl at once: this sample carries 0
    assert not tr.entries[0].within_soap_limit and tr.rinse_owed is not None


def full_rinse(t: float) -> PanSample:
    return replace(free(t), pool="rinse", pool_frac=1.0, other_wet="rinse", in_bowl=True, inside_wet=0.95)


def test_full_rinse_immersion_clears_owed_and_enables_session():
    tr = PanTracker(LIMIT)
    tr.sample(replace(free(0.0), gauge=2e-5))
    tr.sample(dunk(0.01, gauge=2e-5))
    tr.sample(free(0.02))
    assert tr.rinse_owed is not None
    tr.sample(replace(full_rinse(0.03), inside_wet=0.5))        # underside touch is not a rinse
    assert tr.rinse_owed is not None
    tr.sample(full_rinse(0.04))
    assert tr.rinse_owed is None and tr.rinses_after_owed
    tr.sample(free(0.05))
    tr.sample(dunk(0.06))
    assert tr.entries[-1].qualifies and tr.session_ok


def test_floor_touch_resets_and_owes():
    tr = PanTracker(LIMIT)
    tr.sample(free(0.0))
    t = run_dunk(tr, 0.01, 100)
    tr.sample(replace(free(t), z=0.2))
    assert tr.sanitized_s == 0.0 and not tr.session_ok
    assert tr.rinse_owed is not None and tr.rinse_owed.why == "touched the floor"


def stream(tr: PanTracker, t: float, ml_each: float, n: int, base: float = 0.0, inside: bool = False) -> None:
    for i in range(n):
        w = base + ml_each * (i + 1)
        tr.sample(free(t + 0.01 * i, outlet_running=True, water_ml=w, inside_water_ml=w if inside else 0.0))


def test_stream_rinse_over_5ml_resets():
    tr = PanTracker(LIMIT)
    tr.sample(free(0.0))
    t = run_dunk(tr, 0.01, 100)
    tr.sample(free(t, water_ml=0.0))
    stream(tr, t + 0.01, 1.0, 5)                  # 4 gains x 1 mL (first sample sets the base): below 5
    assert tr.session_ok or tr.sanitized_s > 0
    tr.sample(free(t + 0.1, outlet_running=True, water_ml=10.0))
    assert tr.sanitized_s == 0.0 and "running outlet" in tr.resets[-1][1]


def test_stream_on_inside_clears_owed_rinse():
    tr = PanTracker(LIMIT)
    tr.sample(replace(free(0.0), gauge=2e-5))
    tr.sample(dunk(0.01, gauge=2e-5))
    tr.sample(free(0.02))
    assert tr.rinse_owed is not None
    stream(tr, 0.03, 2.0, 5, inside=True)         # 4 gains x 2 mL = 8 mL > 5
    assert tr.rinse_owed is None


def test_carried_detergent_cavity_rule():
    c_rinse = LIMIT / 5e-6
    assert carried_detergent_kg(1e-9, 1e-9, 1e-5, c_rinse) == 1e-9          # weak standing water ignored
    assert carried_detergent_kg(1e-9, 1e-3 * 1e-5 * 10, 1e-5, c_rinse) > 1e-9


def test_progress_stage_products():
    assert pan_progress(1.0, 0.0, 30.0, True)[0] == 0.25
    assert pan_progress(1.0, 1.0, 15.0, False)[0] == round((1 + 1 + 0.5) / 4, 4)
    assert pan_progress(1.0, 1.0, 99.0, True)[0] == 1.0
    assert pan_progress(0.0, 1.0, 30.0, True)[0] == 0.0
    assert pan_progress(0.5, 1.0, 30.0, True)[0] == round((0.5 + 0.5 + 0.5 + 0.5) / 4, 4)


def test_rinsed_term_at_ref_and_limit():
    ref, lim = 13.5e-6, 16.5e-9
    assert rinsed_term(ref, ref, lim) == 0.0
    assert math.isclose(rinsed_term(lim, ref, lim), 1.0)
    assert rinsed_term(1e-3, ref, lim) == 0.0 and rinsed_term(0.0, ref, lim) == 1.0
    assert math.isclose(rinsed_term(math.sqrt(ref * lim), ref, lim), 0.5)


def test_clean_term_log_share():
    g0 = 1e-3
    assert clean_term([g0], [1e-5], [1.0]) == 1.0
    assert math.isclose(clean_term([g0, 5e-6], [1e-4, 1e-3], [1.0, 1.0]), 0.5)   # unsoiled gauge is ignored: 0.5
    assert clean_term([5e-6], [1e-3], [1.0]) == 1.0


def test_pan_score_rules():
    ok = {k: (True, None) for k in ("placed", "drained", "soap_free")}
    assert pan_score(Verdict(criteria=dict(ok)), 1, 1, 30, True).score == 1.0
    bad = Verdict(criteria={**ok, "sanitized": (False, 0)})
    pan_score(bad, 1.0, 0.0, 0.0, True)
    assert bad.progress == 0.25 and bad.score == round(0.9 * 0.25, 4)
    none = pan_score(Verdict(criteria=dict(ok)), 1, 1, 30, False)
    assert none.score is None and none.progress is None
