# SPDX-License-Identifier: Apache-2.0
"""CPL-4 under the amended protocol (coordinator 2026-11-04 20:46Z): rate over the whole run, every 1 s window, worst
height, and no verdict for a run shorter than 60 s."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.bench import cpl4 as C


def test_series_stats_reads_the_whole_run_every_window_and_the_largest_step():
    t = np.arange(1, 6001) * 0.01                       # 60 s at 10 ms
    v = np.where(t >= 0.5, 2e-3, 0.0) + 1e-5 * t        # one 2e-3 onset jump plus a slow drift, E_scale 1
    s = C.series_stats(t, v, 1.0)
    assert s.per_min == pytest.approx(v[-1] * 60.0 / 60.0)          # no extrapolation: 60 s is one minute
    assert s.max_1s == pytest.approx(2e-3 + 1e-5 * 1.0, rel=1e-3)   # the window holding the jump
    assert s.max_step == pytest.approx(2e-3, rel=1e-3)
    assert s.passes()
    assert not C.SeriesStats(per_min=1.1e-2, max_1s=0.0, max_step=0.0).passes()
    assert not C.SeriesStats(per_min=0.0, max_1s=5.1e-2, max_step=0.0).passes()


def _result(rates: list[float], t_end: float = 60.0, partial: bool = False) -> C.SweepResult:
    rows = [C.HeightRow(dz_mm=0.5 * k, e_scale=9.3, stats={"interface_work": C.SeriesStats(r, r / 10, r / 20)})
            for k, r in enumerate(rates)]
    return C.SweepResult(spec=C.SweepSpec("dissipative"), t_end=t_end, rows=rows, exchanges=1, rollbacks=0,
                         wall_s=1.0, partial=partial, versions={})


def test_the_row_is_the_worst_height_and_no_height_may_fail():
    res = _result([1e-3, 4e-3, 2e-3, 1e-4, 3e-3, 1e-3, 2e-3, 1e-3])
    assert res.worst("interface_work", "per_min") == (4e-3, 0.5)
    assert res.verdict() == "pass"
    assert _result([1e-3] * 7 + [1.2e-2]).verdict() == "fail"


def test_a_partial_or_short_run_has_no_verdict():
    assert _result([1e-3] * 8, partial=True).verdict() == "incomplete"
    assert _result([1e-3] * 8, t_end=2.0).verdict() == "incomplete"


def test_result_json_names_the_worst_height_per_bound(tmp_path):
    out = tmp_path / "cpl4.json"
    C.write_result(_result([1e-3, 4e-3, 2e-3, 1e-4, 3e-3, 1e-3, 2e-3, 1e-3]), out)
    d = json.loads(out.read_text())
    assert d["schema"] == "alloy.bench.cpl4/1" and d["verdict"] == "pass"
    assert d["worst"]["interface_work"]["per_min"] == {"value": 4e-3, "dz_mm": 0.5}
    assert len(d["rows"]) == 8
    assert "worst 0.4 %/sim-min (dz 0.5 mm)" in C.summary(_result([1e-3, 4e-3, 2e-3, 1e-4, 3e-3, 1e-3, 2e-3, 1e-3]))


def test_a_short_real_sweep_keeps_one_ledger_sample_per_accepted_exchange(tmp_path):
    spec = C.SweepSpec("conservative", seconds=0.04, heights_m=(0.0, 0.0035), checkpoint_s=0.02)
    res = C.run_sweep(spec, tmp_path / "c.json")
    assert res.t_end == pytest.approx(0.04) and res.verdict() == "incomplete"
    assert [r.dz_mm for r in res.rows] == [0.0, 3.5]
    assert all(np.isfinite(s.per_min) for r in res.rows for s in r.stats.values())
    assert set(res.rows[0].stats) == {"interface_work", "gain"}
    assert json.loads((tmp_path / "c.json").read_text())["partial"] is False


def test_an_onset_re_run_starts_a_new_attempt_on_every_tree():
    """Main 61d3b39a counts the onset refiner's re-runs apart from rollbacks; both start a new attempt, whose
    exchanges replace the rolled-back one's."""
    assert C.attempts(SimpleNamespace(rollbacks=1)) == 1                    # e52980ae: rollbacks counts both
    assert C.attempts(SimpleNamespace(rollbacks=1, onset_reruns=3)) == 4    # 61d3b39a on
