# SPDX-License-Identifier: Apache-2.0
"""Headline rows SB-3..SB-11 fed from producer rows (coordinator task 2026-11-01): value, unit, verdict, source."""

from __future__ import annotations

from pathlib import Path

import pytest

from alloy.bench import report as R
from alloy.bench import sb9 as SB9
from alloy.bench import scoreboard as SB
from alloy.bench.headline import HEADLINES, measured_verdict
from alloy.bench.types import LawResult, MetricVerdict, Quantity, ScoreRow
from alloy.integration.results import ResultRow, collect, to_law_results

NOTES = Path(__file__).resolve().parents[2] / "docs" / "notes"


def _row(pkg: str, mid: str, label: str, value: str, verdict: str, line: int) -> ResultRow:
    return ResultRow(pkg, mid, label, value, verdict, "cmd", "abc1234", f"docs/notes/{pkg.lower()}-results.md:{line}")


def _board(tmp_path: Path, rows: list[ResultRow]) -> dict[str, ScoreRow]:
    for r in to_law_results(rows, "2026-11-01"):
        R.write_report(r, tmp_path)
    return {x.metric_id: x for x in SB.scoreboard([tmp_path])}


def test_sb5_verdict_on_scoped_noslip_with_mujoco_warp_beside(tmp_path) -> None:
    rows = [
        _row("WP8", "PAR-3", "PAR-3 / SB-5 (T12 aggregate, one L4 >= 30x RT)", "177.0x RT at 4,096 worlds", "pass", 35),
        _row("WP0", "PAR-3", "PAR-3 / SB-5 T12 noslip 50 (one L4)", "18.7x RT (noslip 10: 27.0x)", "fail", 40),
        _row("WP0", "PAR-3", "PAR-3 / SB-5 T12 noslip off (one L4)", "185.5x RT", "pass", 41),
        _row("WP3", "SB-5", "SB-5 T12 (L4, recorded noslip 50 scoped)", "162.3x RT at 4,096 worlds", "pass", 74),
        _row("WP8", "PAR-4", "PAR-4 (T09 aggregate >= 5x RT)", "6.1x RT at 1,024 worlds", "pass", 39),
    ]
    by = _board(tmp_path / "a", rows)
    sb5, every = by["SB-5"], by["SB-5:all configurations"]
    assert (sb5.value, sb5.unit, sb5.verdict) == (162.3, "x RT", "pass")  # default route: scoped noslip 50 + T09
    assert "docs/notes/wp3-results.md:74" in sb5.source_rows and "docs/notes/wp0-results.md:40" not in sb5.source_rows
    assert (every.value, every.verdict) == (18.7, "fail")  # MuJoCo Warp's noslip 50 beside it
    assert "all configurations (MuJoCo Warp noslip 50 included): 18.7 x RT fail" in sb5.reason
    assert by["SB-5:T12 noslip off (WP8)"].value == 177.0 and by["SB-5:T12 noslip off (WP0)"].value == 185.5
    assert by["SB-5:T12 MuJoCo Warp noslip 50 (WP0)"].verdict == "fail" and by["SB-5:T09 (WP8)"].value == 6.1
    t09_fail = _row("WP8", "PAR-4", "PAR-4 (T09 aggregate >= 5x RT)", "no valid number: NaN at 1.1 s", "fail", 39)
    assert _board(tmp_path / "b", rows[:4] + [t09_fail])["SB-5"].verdict == "fail"  # T09 still decides
    assert _board(tmp_path / "c", rows[:4])["SB-5"].verdict == "inconclusive"  # no T09 row: partial


def test_fast_caveat_fallback_keeps_wp2s_cable_creep_numbers() -> None:
    import re

    fast_profile = pytest.importorskip("alloy.scene.fast_profile")
    from alloy.bench.headline import _FAST_FALLBACK

    creep = r"cable creep [\d.]+-[\d.]+ mm per 10 s"
    assert re.search(creep, _FAST_FALLBACK).group() == re.search(creep, fast_profile.FAST_CAVEAT).group()


def test_sb5_t09_fast_profile_never_decides_and_carries_its_caveat(tmp_path) -> None:
    from alloy.bench.headline import _FAST_STATUS, _fast_caveat

    base = [_row("WP3", "SB-5", "SB-5 T12 scoped noslip 50", "162.3x RT", "pass", 74),
            _row("WP8", "PAR-4", "PAR-4 (T09 aggregate >= 5x RT)", "1.21x RT at 2,048 worlds", "fail", 40)]
    fast_label = _row("WP8", "PAR-4", "PAR-4 T09, fast route profile", "6.86x RT at 2,048 worlds", "pass", 41)
    fast_text = _row("WP12", "PAR-4", "PAR-4 (T09)", "7.02x RT at 1,024 worlds (mujoco:profile=fast)", "pass", 42)
    by = _board(tmp_path, base + [fast_label, fast_text])
    sb5 = by["SB-5"]
    assert (sb5.value, sb5.verdict) == (162.3, "fail")  # value is T12; the default route T09 (1.21x) fails it, not 6.86x
    fast = by["SB-5:T09 fast route profile (opt-in) (WP8)"]
    assert (fast.value, fast.verdict) == (6.86, "pass") and fast.reason.startswith(f"caveat: {_fast_caveat()}")
    # coordinator 2026-11-03: the fast profile's numbers are marked while WP12 bisects its INVALID worlds
    assert fast.reason.startswith("caveat: invalid on main from 8d513582 to aa2a613a") == bool(_FAST_STATUS)
    assert "informational, not part of the verdict" in fast.reason
    assert by["SB-5:T09 fast route profile (opt-in) (WP12)"].value == 7.02  # marked in the value cell only
    assert f"(info) 6.86 x RT pass (caveat: {_fast_caveat()})" in sb5.reason  # beside the default verdict
    assert "cable creep" in _fast_caveat()


_WP12_NOTE = """| metric id | value | verdict | command | commit |
|---|---|---|---|---|
| WP12-T09-RT default route (the SB-5 verdict) | 1.64x RT at 1,024 worlds with WP8 caps | fail (needs 5x) | `--worlds 1024` | 8a9250e |

| setting | RT from the card start | RT seated | plug creep | cable creep (far end) | source |
|---|---|---|---|---|---|
| default route (the verdict, above) | 1.64x at 1,024 | about 0.77x at 1,024 | 0.044-0.112, fail | 0.13-0.22, fail | WP12 8a9250e |
| opt-in fast profile (WP2 e089c64) | **7.94x at 1,024, 7.77x at 2,048** | **5.78x at 1,024**; **5.69x at 2,048** | 0.0000, pass | 0.73-0.94, fail | WP12 1bf764d |
"""


def test_sb5_reads_wp12_t09_rows_from_its_note(tmp_path, monkeypatch) -> None:
    from alloy.bench import note_rows
    from alloy.bench.headline import _fast_caveat

    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "wp12-results.md").write_text(_WP12_NOTE)
    monkeypatch.setattr(note_rows, "NOTES", tmp_path / "notes")
    by = _board(tmp_path / "b", [_row("WP3", "SB-5", "SB-5 T12 scoped noslip 50", "162.3x RT", "pass", 74)])
    t09 = by["SB-5:T09 default route (WP12)"]
    assert (t09.value, t09.verdict, t09.source_rows) == (1.64, "fail", ("docs/notes/wp12-results.md:3",))
    assert by["SB-5"].verdict == "fail" and "docs/notes/wp12-results.md:3" in by["SB-5"].source_rows
    start, seated = by["SB-5:T09 fast route profile, card start (opt-in) (WP12)"], \
        by["SB-5:T09 fast route profile, seated (opt-in) (WP12)"]
    assert (start.value, start.verdict, seated.value, seated.verdict) == (7.94, "pass", 5.78, "pass")
    assert start.reason.startswith(f"caveat: {_fast_caveat()}") and "7.77x at 2,048" in start.reason
    assert "T09 fast route profile, seated (opt-in) [WP12] (info) 5.78 x RT pass (caveat:" in by["SB-5"].reason


def test_sb4_superseded_reruns_are_shown_not_decided(tmp_path) -> None:
    by = _board(tmp_path, [
        _row("WP3", "CT-2", "CT-2 routed / mujoco", "0 tunnels, p99 11 µm", "pass", 35),
        _row("WP9", "LIQ-8", "LIQ-8", "0 leaked particles and 0 escaped in 10,496 bully episodes", "pass", 45),
        _row("WP9", "LIQ-8", "LIQ-8 rerun after the blow-up fix", "two particles left the tray", "fail", 47),
        _row("WP9", "LIQ-8", "LIQ-8 fixes: regressions", "bitwise equal", "pass", 49)])
    assert (by["SB-4"].value, by["SB-4"].verdict) == (0.0, "pass")
    old = by["SB-4:SPH leaks, superseded rerun (WP9)"]
    assert old.verdict == "fail" and old.source_rows == ("docs/notes/wp9-results.md:47",)
    assert "SPH leaks, superseded rerun [WP9]" in by["SB-4"].reason


_WP5_NOTE = """| Metric | Mode | Value | Verdict | Commit |
|---|---|---|---|---|
| P3-1 R1 | NONE | 0.0 % of worlds equal after the last call, first differing call 0 | fail | 703432e |
| P3-1 R4 | ALL with sorts (7) | world 0 alone vs in 1,024: equal on 100 % of calls | pass | 242f6d0 |

| Metric | Mode | Value | Verdict | Commit |
|---|---|---|---|---|
| P3-1 R1 | route, graph replays | 1,024 worlds, 120 calls: 100.0 % of worlds equal after every call | pass | 49f03b4 |
| P3-1 R1 / R2 / R4 | T12 route, 1,024 worlds, 120 graph calls | 100.0 % / 100.0 % / 100.0 % | pass | cf93378 |
| P3-1 cost | T12 route, 1,024 worlds | graph step 13.41 vs NONE 10.59 ms (1.27x); 1.411 vs 1.239 GiB (1.14x) | pass | cf93378 |
| P3-1 R1 / R2 | T11 / T14 route, 64 worlds, router capacities | 40 graph calls: 100.0 % / 100.0 % on both | pass | 7855d30 |
| P3-1 cost | T11 / T14 route, 64 worlds | 3.75 vs 0.95 ms (3.93x) / 1.94 vs 1.02 ms (1.90x); memory 1.20x | fail | 7855d30 |
| P3-1 R1 / R2 / R4 | T12 route, 1,024 worlds, graph and eager | 100.0 % / 100.0 % / 100.0 % | pass | 4525d4e |
| P3-1 cost | T12 route, 1,024 worlds | graph step 12.98 vs NONE 10.31 ms (1.26x); 1.441 vs 1.266 GiB (1.14x) | pass | 4525d4e |
| P3-1 N=1 (RUN_TO_RUN, beside SB-6) | T12 route, one-block rank sorts | 1.426 vs 1.197 ms per call (1.19x) | measured | dff9282 |
| P3-1 N=1 (RUN_TO_RUN, beside SB-6) | T12 route, + tree sums | 1.307 vs 1.197 ms (1.09x) | measured | 4525d4e |
"""


def test_sb11_counts_solvermujoco_on_the_ordered_route_only(tmp_path, monkeypatch) -> None:
    from alloy.bench import note_rows

    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "wp5-results.md").write_text(_WP5_NOTE)
    monkeypatch.setattr(note_rows, "NOTES", tmp_path / "notes")
    by = _board(tmp_path / "b", [_row("WP5", "PRC-4", "PRC-4 (GPU, D9 population)", "FakeSolver: R1, R2 equal",
                                      "pass", 29)])
    assert (by["SB-11"].value, by["SB-11"].verdict) == (1.0, "pass")
    t12 = by["SB-11:P3-1 R1 / R2 / R4, T12 route, 1,024 worlds, graph and eager (WP5)"]  # newest T12 row decides
    assert (t12.value, t12.verdict, t12.source_rows) == (1.0, "pass", ("docs/notes/wp5-results.md:13",))
    assert "SB-11:P3-1 R1 / R2 / R4, T12 route, 1,024 worlds, 120 graph calls (superseded) (WP5)" in by
    assert by["SB-11:P3-1 R1 / R2, T11 / T14 route, 64 worlds, router capacities (WP5)"].value == 1.0
    assert not any(k.startswith(("SB-11:P3-1 R1, NONE", "SB-11:P3-1 R1, route")) for k in by)  # off-route, part 1
    cost = by["SB-11:P3-1 cost, T12 route, 1,024 worlds (WP5)"]  # info beside: newest only, the 1.27x row dropped
    assert (cost.value, cost.verdict) == (1.26, "pass") and "SB-11:P3-1 cost, T12 route, 1,024 worlds (WP5) #2" not in by
    few = by["SB-11:P3-1 cost, T11 / T14 route, 64 worlds (WP5)"]
    assert (few.value, few.verdict) == (3.93, "fail")  # the worst scene of the row, not the first number
    assert "repeat under RUN_TO_RUN, an opt-in preset" in by["SB-11"].reason
    (tmp_path / "notes" / "wp5-results.md").write_text(_WP5_NOTE.replace(
        "| 100.0 % / 100.0 % / 100.0 % | pass | 4525d4e", "| 100.0 % / 99.2 % / 100.0 % | fail | 4525d4e"))
    by = _board(tmp_path / "c", [_row("WP5", "PRC-4", "PRC-4 (GPU)", "FakeSolver: R1, R2 equal", "pass", 29)])
    assert (by["SB-11"].value, by["SB-11"].verdict) == (0.992, "fail")


def test_sb6_newest_wp3_row_decides_and_run_to_run_is_beside(tmp_path, monkeypatch) -> None:
    from alloy.bench import note_rows

    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "wp3-results.md").write_text(
        "| Metric | Value | Verdict |\n|---|---|---|\n"
        "| SB-6 variants on the new noslip (ladder, same VM) | WP5's determinism sorts (bits 3): 0.877x (0.950 ms) "
        "| info (sorts: fail) |\n"
        "| SB-6 RUN_TO_RUN N=1 (T12, default route + WP5's route, older) | 0.95x | info |\n"
        "| SB-6 RUN_TO_RUN N=1 (T12, default route + WP5's route) | 1.035x (0.805 ms per 0.833 ms step) | info |\n")
    (tmp_path / "notes" / "wp5-results.md").write_text(_WP5_NOTE)
    monkeypatch.setattr(note_rows, "NOTES", tmp_path / "notes")
    by = _board(tmp_path / "b", [
        _row("WP3", "SB-6", "SB-6 T12 (N = 1, default route: scoped noslip 50)", "0.64x (1.30 ms)", "fail", 83),
        _row("WP3", "SB-6", "SB-6 T12 re-measure on split hulls", "N = 1: 0.88x", "fail", 134),
        _row("WP3", "SB-6", "SB-6 T12 (N = 1, default route)", "**1.0996x / 1.0988x** (0.758 ms)", "pass", 199),
        _row("WP8", "PAR-8", "PAR-8 / SB-6 (T12 single-world RTF >= 1x)", "0.877x on main d86d75f", "fail", 37)])
    assert (by["SB-6"].value, by["SB-6"].verdict) == (1.0996, "pass")  # WP3 lists rounds oldest first
    assert by["SB-6:T12 (N = 1, default route: scoped noslip 50) (superseded) (WP3)"].value == 0.64
    r2r = by["SB-6:T12 N=1 RUN_TO_RUN (opt-in) (WP3)"]  # WP3's newest RUN_TO_RUN row only, never decides
    assert (r2r.value, r2r.verdict, r2r.source_rows) == (1.035, "pass", ("docs/notes/wp3-results.md:5",))
    assert "SB-6:T12 N=1 RUN_TO_RUN (opt-in) (WP3) #2" not in by  # the 0.877x variants row is no longer read
    assert "SB-6 counts the determinism route off" in by["SB-6"].reason
    assert "T12 N=1 RUN_TO_RUN (opt-in) [WP3] (info) 1.035 x RT pass" in by["SB-6"].reason
    cost = by["SB-6:T12 N=1 RUN_TO_RUN step cost (opt-in) (WP5)"]  # WP5's newest N=1 row only
    assert (cost.value, cost.verdict, cost.source_rows) == (1.09, "pass", ("docs/notes/wp5-results.md:16",))


def test_sb10_scoped_decides_mujoco_warp_beside(tmp_path) -> None:
    by = _board(tmp_path, [
        _row("WP0", "PAR-15", "PAR-15 / SB-10 T12 noslip 50", "$8.56 per M sim-s at 18.7x", "fail", 40),
        _row("WP3", "SB-10", "SB-10 T12 (scoped noslip 50)", "$0.985 per M sim-s at 4,096 worlds", "pass", 75),
    ])
    assert (by["SB-10"].value, by["SB-10"].verdict) == (0.985, "pass")
    assert (by["SB-10:all configurations"].value, by["SB-10:all configurations"].verdict) == (8.56, "fail")


def test_sb10_wp3s_newest_t12_row_decides(tmp_path) -> None:
    """WP3 lists T12 cost rows oldest first (the light-body row at :305 supersedes :83), as for SB-5."""
    by = _board(tmp_path, [
        _row("WP3", "SB-10", "SB-10 T12 (scoped noslip 50)", "$0.944 per M sim-s at 4,096 worlds", "pass", 83),
        _row("WP3", "SB-10", "SB-10 T12 (scoped noslip 50, light bodies aeac0ab1)", "$0.928 per M sim-s", "pass", 305),
    ])
    assert (by["SB-10"].value, by["SB-10"].verdict) == (0.928, "pass")
    assert by["SB-10:T12 scoped noslip 50 (superseded) (WP3)"].value == 0.944


def test_value_unit_from_producer_text(tmp_path) -> None:
    by = _board(tmp_path, rows := [
        _row("WP3", "CT-1", "CT-1 sdf / mujoco", "16 worlds, dt 0.3125 ms: p99 26 µm, max 27.1 µm", "pass", 21),
        _row("WP3", "CT-1", "CT-1 hydroelastic / newton", "p99 17 µm, max 102.7 µm", "fail", 24),
        _row("WP6", "CPL-4", "CPL-4 (dissipative, 8-height sweep, 60 s)",
             "**0.40 %/sim-min** at dz 3.0 mm, worst 1 s window 0.011 %", "pass", 77),
        _row("WP4", "CPL-4", "CPL-4 (dissipative)", "box on cloth (μ 0.5): 0.04 %/sim-min, max 1 s 0.011 %", "pass",
             31),  # the 2 s single-height smoke form: feeds neither SB-8 nor CPL-4
        _row("WP7", "SB-7", "SB-7 (task class)", "121 S1 in 20,480 episodes: upper bound 6.87e-3 > 1e-5", "fail", 23),
        _row("WP3", "SB-10", "SB-10 T12 (scoped noslip 50)", "$8.56 per M sim-s at 18.7x", "fail", 37),
        _row("WP5", "PRC-4", "PRC-4 (GPU)", "R1, R2 and R3 equal at 1 and 1024 worlds", "pass", 29),
    ])
    assert (by["SB-3"].value, by["SB-3"].unit, by["SB-3"].verdict) == (26.0, "µm", "pass")  # default route
    every = by["SB-3:all configurations"]
    assert (every.value, every.verdict) == (26.0, "fail") and "hydroelastic / newton [WP3] fail" in every.reason
    assert (by["SB-8"].value, by["SB-8"].unit, by["SB-8"].verdict) == (0.40, "%/sim-min", "pass")
    assert by["SB-8"].source_rows == ("docs/notes/wp6-results.md:77",)
    assert (by["CPL-4"].value, by["CPL-4"].source_rows) == (0.40, ("docs/notes/wp6-results.md:77",))
    rerun = _row("WP6", "CPL-4", "CPL-4 (dissipative, 8-height sweep, 60 s, Newton 8f3466f9)",
                 "**0.16 %/sim-min** at dz 3.0 mm", "pass", 90)       # a re-pin's rerun, listed after the old sweep
    sb8 = _board(tmp_path / "rerun", [r for r in rows if r.metric_id == "CPL-4"] + [rerun])
    assert (sb8["SB-8"].value, sb8["SB-8:dissipative, 8-height sweep, 60 s (superseded) (WP6)"].value) == (0.16, 0.40)
    assert by["SB-7"].value == pytest.approx(6.87e-3) and by["SB-7"].verdict == "fail"
    assert (by["SB-10"].value, by["SB-10"].verdict) == (8.56, "fail")
    assert (by["SB-11"].value, by["SB-11"].verdict) == (1.0, "pass")


def test_partial_evidence_never_passes(tmp_path) -> None:
    routed = _row("WP3", "CT-2", "CT-2 routed / mujoco", "0 tunnels, p99 19 µm", "pass", 35)
    liq = _row("WP9", "LIQ-8", "LIQ-8", "B0 target registered", "not_run", 43)
    assert _board(tmp_path / "a", [routed])["SB-4"].verdict == "inconclusive"  # LIQ-8 has no row
    sb4 = _board(tmp_path / "b", [routed, liq])["SB-4"]
    assert sb4.verdict == "inconclusive" and sb4.value == 0.0 and "SPH leaks [WP9] not_run" in sb4.reason
    bad = _row("WP3", "CT-2", "CT-2 routed / newton", "2 tunnel sample-events in 1 of 256 worlds", "fail", 36)
    by = _board(tmp_path / "c", [routed, liq, bad])
    assert by["SB-4"].verdict == "inconclusive"  # default route: mujoco 0 events, LIQ-8 not measured
    assert by["SB-4:all configurations"].verdict == "fail"  # a measured fail beats not_run
    assert measured_verdict(["pass", "not_run"]) == "inconclusive" and measured_verdict(["not_run"]) == "not_run"


def test_not_run_only_when_nothing_measured(tmp_path) -> None:
    by = _board(tmp_path, [_row("WP2", "CARD-1", "CARD-1, CARD-11 coverage", "coverage", "not_run", 60),
                           _row("WP8", "PAR-8", "PAR-8 / SB-6 (T12 single-world)", "1.32x (0.629 ms)", "pass", 36),
                           _row("WP3", "CT-2", "CT-2 detector", "counts pass-throughs in all 256 worlds", "pass", 34)])
    assert by["SB-9"].verdict == "not_run" and "CARD-1" in by["SB-9"].reason
    assert by["SB-6"].verdict == "inconclusive" and "noslip off" in by["SB-6"].reason  # no scoped row yet
    assert by["SB-4"].verdict == "inconclusive" and "update alloy.bench.headline" in by["SB-4"].reason
    assert by["SB-3"].verdict == "not_run" and by["SB-3"].reason.startswith("no producer row for CT-1")


def test_default_route_verdict_with_all_configurations_beside(tmp_path) -> None:
    by = _board(tmp_path, [
        _row("WP3", "CT-1", "CT-1 sdf / mujoco", "p99 26 µm, max 27.1 µm", "pass", 21),
        _row("WP3", "CT-1", "CT-1 sdf / newton", "p99 26 µm, max 26.6 µm", "pass", 22),
        _row("WP3", "CT-1", "CT-1 hydroelastic / mujoco", "p99 19 µm, max 86.1 µm", "pass", 23),
        _row("WP3", "CT-1", "CT-1 hydroelastic / newton", "p99 17 µm, max 102.7 µm", "fail", 24),
    ])
    sb3 = by["SB-3"]
    assert sb3.verdict == "pass" and sb3.source_rows == ("docs/notes/wp3-results.md:21",)
    assert sb3.reason.startswith("verdict on the Alloy default route (sdf precision pairs, native_contacts mujoco")
    assert "all configurations: 26 µm fail" in sb3.reason and by["SB-3:all configurations"].verdict == "fail"
    order = [r.metric_id for r in SB.scoreboard([tmp_path]) if r.metric_id.startswith("SB-3")]
    assert order[:2] == ["SB-3", "SB-3:all configurations"]  # shown next to each other


def _producer_ids(sb: str) -> set[str]:
    return {f.metric for f in HEADLINES[sb].feeds}


def test_every_sb_with_a_measured_producer_is_fed(tmp_path) -> None:
    # the live results notes: any measured producer row must reach its SB row, with a pointer to the row
    collect(NOTES, [], tmp_path, "2026-11-01")
    reports = SB.load_reports([tmp_path])
    rows = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    measured = {v.name.split(":")[0] for _, r in reports for v in r.verdicts if v.verdict != "not_run"}
    for sb in HEADLINES:
        row = rows[sb]
        if _producer_ids(sb) & measured:
            assert row.verdict != "not_run", f"{sb} not fed although {_producer_ids(sb) & measured} measured"
            assert any(".md:" in s for s in row.source_rows), f"{sb} has no source pointer"
        assert row.reason or row.verdict == "pass", f"{sb} {row.verdict} without a reason"


def _law(law: str, solver: str, verdict: str, date: str = "2026-11-01") -> LawResult:
    mv = MetricVerdict("m", "1", None, Quantity("m", "1", 0.1, 0.0, 1, 0.0), 0.1, 1.0, None, "mean", verdict)  # type: ignore[arg-type]
    prov = {"suite": "analytic", "quick": "True", "alloy": "abc", "date": date}
    return LawResult(law, solver, "cpu", (mv,), {}, prov, 0.1, True)


def test_sb12_counts_laws_that_ran_once(tmp_path) -> None:
    for i, r in enumerate([_law("A1.freefall", "mujoco", "fail", "2026-10-31"), _law("A1.freefall", "mujoco", "pass"),
                           _law("A2.pendulum", "mujoco", "fail"), _law("A9", "any", "not_run"),
                           _law("A9.thread", "mujoco", "pass"), _law("A2.pendulum", "xpbd", "fail"),
                           _law("A15.neohookean", "vbd", "pass"), _law("A15.neohookean_explicit", "semi_implicit", "fail"),
                           _law("A7.hydroelastic", "mujoco", "pass")]):
        R.write_report(r, tmp_path / str(i))
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    sb12, every = by["SB-12"], by["SB-12:all configurations"]
    # default route: A1, A2 on mujoco, A9, A15 on vbd (newest A1 run counts once); xpbd, semi_implicit and A7 are not
    assert sb12.value == pytest.approx(3 / 4) and sb12.verdict == "fail"
    assert "A2.pendulum/mujoco/cpu fail" in sb12.reason and "not counted (not_run): A9" in sb12.reason
    assert "all configurations: 0.571 fail" in sb12.reason and every.value == pytest.approx(4 / 7)
    assert "A2.pendulum/xpbd/cpu fail" in every.reason and "xpbd" not in sb12.reason.split("; ", 2)[2]


def test_sb12_newer_not_run_replaces_an_older_error(tmp_path) -> None:
    # a law that now declines the device (WP3's A8 on CPU) must leave the population, not keep its old error
    R.write_report(_law("A8.push", "mujoco", "error", "2026-11-01T01:00"), tmp_path / "old")
    R.write_report(_law("A8.push", "mujoco", "not_run", "2026-11-01T02:00"), tmp_path / "new")
    R.write_report(_law("A1.freefall", "mujoco", "pass"), tmp_path / "a1")
    sb12 = next(x for x in SB.scoreboard([tmp_path]) if x.metric_id == "SB-12")
    assert sb12.value == 1.0 and sb12.verdict == "inconclusive" and "not counted (not_run): A8.push" in sb12.reason


def test_sb6_scoped_decides_noslip_off_beside(tmp_path) -> None:
    # coordinator round 2: SB-6 on the default route (scoped noslip), WP8's noslip-off ladder row beside
    by = _board(tmp_path, [
        _row("WP8", "PAR-8", "PAR-8 / SB-6 (T12 single-world RTF >= 1x)", "1.32x (0.629 ms per 0.833 ms)", "pass", 36),
        _row("WP0", "PAR-8", "PAR-8 / SB-6 T12 noslip off (N = 1, <gpu-vm>)", "0.79x RT (median 1.05 ms)", "fail", 39),
        _row("WP0", "PAR-8", "PAR-8 / SB-6 T12 noslip 50 (N = 1)", "0.03x RT (26.2 ms)", "fail", 38),
        _row("WP3", "SB-6", "SB-6 T12 (N = 1, scoped noslip 50)", "0.41x (2.01 ms per period)", "fail", 76),
    ])
    sb6, off = by["SB-6"], by["SB-6:noslip off (WP8 ladder)"]
    assert (sb6.value, sb6.verdict) == (0.41, "fail") and sb6.source_rows == ("docs/notes/wp3-results.md:76",)
    assert (off.value, off.verdict) == (1.32, "pass") and off.source_rows == ("docs/notes/wp8-results.md:36",)
    assert "noslip off (WP8 ladder): 1.32 x RT pass" in sb6.reason
    assert by["SB-6:T12 N=1 noslip off, other VM (WP0)"].value == 0.79


def test_sb4_out_of_envelope_line_is_informational(tmp_path) -> None:
    newton = ('same run on native="newton": 0 tunnels, p99 10 µm. The earlier 2 events (0e801fc) and 3 events on '
              'native="mujoco" (seed 4) were all in the press phase: coins driven onto the plate edge at 4.2 m/s')
    by = _board(tmp_path, [_row("WP3", "CT-2", "CT-2 routed / mujoco", "0 tunnels, p99 11 µm", "pass", 35),
                           _row("WP3", "CT-2", "CT-2 routed / newton", newton, "pass", 36),
                           _row("WP9", "LIQ-8", "LIQ-8", "0 escaped", "pass", 43)])
    assert (by["SB-4"].value, by["SB-4"].verdict) == (0.0, "pass")  # the in-envelope sweep decides
    assert by["SB-4:all configurations"].verdict == "pass"
    out_n = by["SB-4:outside the CT-2 envelope, newton (old press) (WP3)"]
    out_m = by["SB-4:outside the CT-2 envelope, mujoco (old press, seed 4) (WP3)"]
    assert (out_n.value, out_n.verdict, out_m.value, out_m.verdict) == (2.0, "fail", 3.0, "fail")
    assert out_n.reason.startswith("informational, not part of the verdict") and out_n.source_rows == (
        "docs/notes/wp3-results.md:36",)
    assert "outside the CT-2 envelope, mujoco (old press, seed 4) [WP3] (info) 3 events fail" in by["SB-4"].reason


def test_sb12_known_gap_is_shown_not_counted(tmp_path) -> None:
    R.write_report(_law("A5.slide", "mujoco", "pass"), tmp_path / "a")
    R.write_report(_law("A5.slide_recipe", "mujoco", "fail"), tmp_path / "b")
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert (by["SB-12"].value, by["SB-12"].verdict) == (1.0, "pass")
    assert "known gaps, not counted (SPEC §5.3): A5.slide_recipe/mujoco/cpu fail" in by["SB-12"].reason
    assert by["law:A5.slide_recipe:mujoco:cpu"].reason.startswith("known gap, not gated: SPEC §5.3 A5 (SCR-WP3-7)")


def test_sb12_informational_law_is_shown_not_counted(tmp_path) -> None:
    R.write_report(_law("A4.momentum", "mujoco", "pass"), tmp_path / "a")
    R.write_report(_law("A4.collide_small_lever", "mujoco", "fail"), tmp_path / "b")
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert (by["SB-12"].value, by["SB-12"].verdict) == (1.0, "pass")
    assert "informational, not counted: A4.collide_small_lever/mujoco/cpu fail" in by["SB-12"].reason
    assert by["law:A4.collide_small_lever:mujoco:cpu"].reason.startswith("informational, not gated: the A4 collide")


def test_sb12_counts_a10_as_fail_until_the_ruling_is_met(tmp_path, monkeypatch) -> None:
    import dataclasses

    from alloy.bench import law as L

    def ruling(met: bool) -> None:
        table = {**L.CONDITIONAL, "A10.capstan": dataclasses.replace(L.CONDITIONAL["A10.capstan"], met=met)}
        monkeypatch.setattr(L, "CONDITIONAL", table)
        monkeypatch.setattr(SB, "CONDITIONAL", table)

    R.write_report(_law("A4.momentum", "mujoco", "pass"), tmp_path / "a")
    R.write_report(_law("A10.capstan", "mujoco", "pass"), tmp_path / "b")
    ruling(False)
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert (by["SB-12"].value, by["SB-12"].verdict) == (0.5, "fail")  # measured pass, counted fail (ruling unmet)
    assert "A10.capstan/mujoco/cpu measured pass, counted fail: caveat: the route's CHAIN emitter floors" \
        in by["SB-12"].reason and "coordinator 2026-11-02" in by["SB-12"].reason
    law_row = by["law:A10.capstan:mujoco:cpu"]
    assert law_row.verdict == "fail" and law_row.reason.startswith("measured pass, counted fail; caveat:")
    assert "## Rulings" in SB.render_markdown(list(by.values()))
    ruling(True)
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert (by["SB-12"].value, by["SB-12"].verdict) == (1.0, "pass")  # once WP2 meets it, the pass counts
    assert by["law:A10.capstan:mujoco:cpu"].verdict == "pass" and "counted" not in by["law:A10.capstan:mujoco:cpu"].reason


def test_sb12_a_cuda_only_law_that_ran_on_cuda_is_not_missing(tmp_path) -> None:
    import dataclasses

    R.write_report(_law("A4.momentum", "mujoco", "pass"), tmp_path / "a")
    R.write_report(_law("A8.push", "mujoco", "not_run"), tmp_path / "b")  # CPU: texture SDFs bake on CUDA only
    assert {x.metric_id: x for x in SB.scoreboard([tmp_path])}["SB-12"].verdict == "inconclusive"
    R.write_report(dataclasses.replace(_law("A8.push", "mujoco", "pass"), device="cuda:0"), tmp_path / "c")
    sb12 = {x.metric_id: x for x in SB.scoreboard([tmp_path])}["SB-12"]
    assert (sb12.value, sb12.verdict) == (1.0, "pass") and "not counted (not_run)" not in sb12.reason


def test_board_keeps_the_newest_report_per_key(tmp_path) -> None:
    R.write_report(_law("A5.slide", "mujoco", "fail", "2026-11-01T01:00"), tmp_path / "old")
    R.write_report(_law("A5.slide", "mujoco", "pass", "2026-11-01T02:00"), tmp_path / "new")
    rows = [x for x in SB.scoreboard([tmp_path]) if x.metric_id == "law:A5.slide:mujoco:cpu"]
    assert len(rows) == 1 and rows[0].verdict == "pass"


def test_sb9_decides_on_the_coverage_row_on_the_pinned_newton(tmp_path) -> None:
    """WP11's re-measure on the pin decides; WP2's v1 row (earlier Newton) is beside, as are the findings."""
    pin = SB9.PIN
    v1 = _row("WP2", "CARD-1", "CARD-1 coverage (SB-9 v1)", "62/76 = 81.6 % (target >= 80 %)", "pass", 61)
    rows = [v1, _row("WP2", "CARD-1", "CARD-1 rest-law jitter vs damping", "travel 1.12 mm", "inconclusive", 62),
            _row("WP11", "CARD-1", f"CARD-1 coverage (SB-9) re-measured after the re-pin: Newton {pin}, L4",
                 "152 / 181 covered (84.0 %); earlier tree: stock 151 / 180", "pass", 116)]
    by = _board(tmp_path / "a", rows)
    assert (by["SB-9"].value, by["SB-9"].verdict) == (84.0, "pass")
    assert by["SB-9"].source_rows == ("docs/notes/wp11-results.md:116", "docs/notes/wp2-results.md:61")
    assert f"SB-9:coverage on Newton {pin} (WP11)" in by
    assert by["SB-9:coverage on an earlier Newton (superseded) (WP2)"].value == 81.6
    off_pin = _board(tmp_path / "b", [v1])["SB-9"]              # no row on the pin yet: never a pass
    assert (off_pin.value, off_pin.verdict) == (None, "inconclusive")


def test_sb4_value_counts_sph_leaks_too(tmp_path) -> None:
    by = _board(tmp_path, [_row("WP3", "CT-2", "CT-2 routed / mujoco", "0 tunnels, p99 11 µm", "pass", 35),
                           _row("WP9", "LIQ-8", "LIQ-8", "8 leaked particles in 1 of 12,288 episodes", "fail", 43)])
    assert (by["SB-4"].value, by["SB-4"].verdict) == (8.0, "fail")



def test_newest_class_is_per_results_note() -> None:
    """A class's newest row is chosen within one producer's note: WP5's later-file audit row (no number) must not
    supersede WP3's newest measured T12 row (main 2026-11-03, `SB-6 T12 N=1 overflow`)."""
    from alloy.bench import headline as H
    from alloy.bench.headline_types import Feed, ProducerRow

    feed = Feed("SB-6", r"([\d.]+)x", "x RT", newest=r"\bT12\b", newest_last=True)

    def row(source: str, label: str, value: float | None) -> ProducerRow:
        return ProducerRow(feed, source.split("-")[0], label, "", value, "pass", source, "", "", "", "")

    rows = [row("docs/notes/wp3-results.md:10", "T12 (N = 1) old", 0.88),
            row("docs/notes/wp3-results.md:20", "T12 (N = 1) new", 1.0996),
            row("docs/notes/wp5-results.md:5", "T12 N=1 overflow", None)]
    out = H._supersede(feed, rows)
    assert [p.label for p in out if p.feed.decides] == ["T12 (N = 1) new", "T12 N=1 overflow"]
    assert [p.feed.tag for p in out if not p.feed.decides] == ["T12 (N = 1) old (superseded)"]


def test_sb12_traded_known_gap_is_shown_beside_with_both_numbers(tmp_path) -> None:
    """Coordinator ruling 2026-11-03 (SCR-WP3-10): the slide spread's growth is a stated trade, visible beside SB-12."""
    R.write_report(_law("A5.slide", "mujoco", "pass"), tmp_path / "a")
    q = Quantity("accel_spread", "1", 0.1028, 0.0, 1, 0.0)
    mv = MetricVerdict("accel_spread", "1", None, q, 0.1028, 0.01, None, "mean", "fail")  # type: ignore[arg-type]
    prov = {"suite": "analytic", "quick": "True", "alloy": "abc", "date": "2026-11-03"}
    R.write_report(LawResult("A5.slide_spread", "mujoco", "cpu", (mv,), {}, prov, 0.1, True), tmp_path / "b")
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert (by["SB-12"].value, by["SB-12"].verdict) == (1.0, "pass")  # not counted
    gap = by["SB-12:known gap A5.slide_spread/mujoco/cpu"]
    assert (gap.value, gap.verdict, gap.target) == (0.1028, "fail", "<= 0.01 (not gated)")
    assert "A5.speed fixed, slide spread 0.034 -> 0.103" in gap.reason and "accel_spread 0.103" in gap.reason
    assert "traded known gap (beside): SCR-WP3-10 300 fill-in" in by["SB-12"].reason


def test_sb12_route_known_gap_is_shown_beside_and_not_counted(tmp_path) -> None:
    """Coordinator ruling 2026-11-04 01:36Z (c): the static-hold creep stays a default-route gap, shown beside SB-12."""
    R.write_report(_law("A5.slide", "mujoco", "pass"), tmp_path / "a")
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert (by["SB-12"].value, by["SB-12"].verdict) == (1.0, "pass")  # not counted
    gap = by["SB-12:known gap static hold (soft friction creep)"]
    assert (gap.value, gap.unit, gap.verdict, gap.status) == (0.109, "mm/s", "fail", "active")
    assert "creeps below the cone on the default route (0.11 mm/s at 0.25 mu*N push, real: 0)" in gap.reason
    assert "alloy:noslip_all, T12 SB-6 cost 1.12x -> 0.21x" in gap.reason and "rk4 route entries keep the creep" in gap.reason
    assert "route known gap (beside): soft friction creeps" in by["SB-12"].reason

