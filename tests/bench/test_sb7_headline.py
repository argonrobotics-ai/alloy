# SPDX-License-Identifier: Apache-2.0
"""SB-7 (bully S1 rate per scene class): per class WP7's newest campaign row decides; triage rows are beside."""

from __future__ import annotations

from alloy.bench import sb7 as SB7
from tests.bench.test_headline import _board, _row


def test_sb7_newest_row_per_scene_class_decides(tmp_path) -> None:
    by = _board(tmp_path, [
        _row("WP7", "SB-7", "SB-7 (xpbd_box_stack class)", "0 S1 in 303,104; one-sided 95 % bound 3/N = 9.90e-6",
             "pass", 24),
        _row("WP7", "SB-7", "SB-7 (task class: task_t12 on the route, after WP3's cone fix b85c590)",
             "2 S1 in 303,104: rate 6.60e-6, Clopper-Pearson one-sided 95 % upper bound 2.08e-5", "fail", 25),
        _row("WP7", "SB-7", "SB-7 (task class: task_t12 on the route, after WP3's BUL-0010 fix)",
             "10 S1 in 49,152: Clopper-Pearson 95 % upper bound 3.6e-4", "fail", 27),
        _row("WP7", "SB-7", "SB-7 (task_t12, before the hull fix)", "Clopper-Pearson 95 % upper bound 6.87e-3 > 1e-5",
             "fail", 43),
        _row("WP7", "SB-7", "SB-7 (other scene classes)", "cloth, liquid, cutting", "not_run", 44)])
    assert (by["SB-7"].value, by["SB-7"].verdict) == (2.08e-5, "fail")  # task_t12's newest campaign, not the worst
    old = by["SB-7:task_t12, before the hull fix (superseded) (WP7)"]
    assert (old.value, old.verdict, old.source_rows) == (6.87e-3, "fail", ("docs/notes/wp7-results.md:43",))
    assert "task class: task_t12 on the route, after WP3's BUL-0010 fix (superseded) [WP7]" in by["SB-7"].reason
    assert by["SB-7:xpbd_box_stack class (WP7)"].verdict == "pass"


_PASS = "0 S1 in 303,104 episodes: rate 0, one-sided 95 % bound 3/N = 9.90e-6 <= 1e-5"


def _emb(cls: str, value: str, verdict: str, line: int):
    return _row("WP7", "SB-7", f"SB-7 ({cls} class, main ab00a559)", value, verdict, line)


def _routes(first_line: int) -> list:
    """A passing route-variant row per embodiment (the classes SB-7 counts since 2026-11-05 21:04Z)."""
    return [_emb(f"emb_{e}_x_route", _PASS, "pass", first_line + k) for k, e in enumerate(SB7.ROUTE_EMBODIMENTS)]


def test_sb7_triage_rows_are_beside_and_the_newest_campaign_decides(tmp_path) -> None:
    """A triage row replays a campaign's S1 episodes; the campaign row carries them, so a triage fail of a superseded
    campaign never keeps SB-7 red (WP7's light-body campaign on aeac0ab1, 2026-11-05)."""
    rows = [
        _row("WP7", "SB-7", "SB-7 (xpbd_box_stack class)", "0 S1 in 303,104; one-sided 95 % bound 3/N = 9.90e-6",
             "pass", 24),
        _row("WP7", "SB-7", "SB-7 (task class: task_t12 on WP3's light-body rule, main aeac0ab1)",
             "0 S1 in 303,104 random-legal OU episodes: rate 0, one-sided 95 % bound 3/N = 9.90e-6", "pass", 25),
        _row("WP7", "SB-7", "SB-7 (task class: task_t12 on the ordered route, box-box on the primitive collider)",
             "1 S1 in 303,104: Clopper-Pearson one-sided 95 % upper bound 1.57e-5 > 1e-5", "fail", 27),
        _row("WP7", "SB-7", "SB-7 triage (SCR-WP3-9 campaign) -> BUL-0014", "100100:316 replays alone", "fail", 28),
        _row("WP7", "SB-7", "SB-7 triage (ordered route) -> BUL-0011", "Both S1 replay alone", "fail", 33),
        *_routes(60)]
    by = _board(tmp_path / "a", rows)
    assert (by["SB-7"].value, by["SB-7"].verdict) == (9.9e-6, "pass")
    triage = by["SB-7:triage (SCR-WP3-9 campaign) -> BUL-0014 (WP7)"]
    assert (triage.verdict, triage.source_rows) == ("fail", ("docs/notes/wp7-results.md:28",))
    assert "triage (SCR-WP3-9 campaign) -> BUL-0014 [WP7] fail" in by["SB-7"].reason   # shown beside
    other = _row("WP7", "SB-7", "SB-7 (other scene classes)", "cloth, liquid, cutting", "not_run", 53)
    assert _board(tmp_path / "b", [*rows, other])["SB-7"].verdict == "inconclusive"   # classes not run: never pass


def test_task_t12_s2_count_is_labelled_arrival_penetration_with_wp3s_depths(tmp_path) -> None:
    """Coordinator 2026-11-05 14:58Z: the S2 count reads as transient arrival depth, not resting sink-in; only the
    class's newest campaign carries the label (WP3's percentiles are aeac0ab1's), and it adds no board row."""
    new = ("0 S1 in 303,104 episodes: rate 0, one-sided 95 % bound 3/N = 9.90e-6 <= 1e-5. S2 in 141,062 episodes "
           "(46.5 %), every one past the 1 mm max_pen line")
    old = "1 S1 in 303,104: upper bound 1.57e-5 > 1e-5 (0 S1 would have given 9.88e-6). S2 in 157,557 episodes (52 %)"
    by = _board(tmp_path, [
        _row("WP7", "SB-7", "SB-7 (task class: task_t12 with WP3's light-body contact rule, main aeac0ab1)", new,
             "pass", 25),
        _row("WP7", "SB-7", "SB-7 (task class: task_t12 on the ordered route, main 9ac45a5)", old, "fail", 28)])
    label = ("S2 141,062 episodes (46.5 %) = arrival penetration (transient), cause under re-derivation: p50 / p95 / "
             "p99 0.97 / 2.02 / 2.76 mm")
    newest = by["SB-7:task class: task_t12 with WP3's light-body contact rule, main aeac0ab1 (WP7)"]
    assert (newest.verdict, newest.value) == ("pass", 9.9e-6) and label in newest.reason
    assert label in by["SB-7"].reason and "not resting sink-in" in by["SB-7"].reason
    assert "arrival" not in by["SB-7:task class: task_t12 on the ordered route, main 9ac45a5 (superseded) (WP7)"].reason
    sb7 = sorted(k for k in by if k.startswith("SB-7"))
    assert sb7 == ["SB-7", "SB-7:task class: task_t12 on the ordered route, main 9ac45a5 (superseded) (WP7)",
                   "SB-7:task class: task_t12 with WP3's light-body contact rule, main aeac0ab1 (WP7)"]


def test_s2_aside_reads_the_count_or_nothing() -> None:
    got = SB7.s2_aside("", "S2 in 4,481 episodes (1.48 %, penetration > 1 mm)")
    assert got.startswith("S2 4,481 episodes (1.48 %) = ")
    assert SB7.s2_aside("", "0 S1 in 3,072 episodes") == ""


def test_embodiment_classes_count_on_their_route_variants_and_source_scenes_are_reported(tmp_path) -> None:
    """Coordinator 2026-11-05 21:04Z: the source scenes' soft contact measures the scene, so their rows are reported,
    not counted; each embodiment counts on its _route variant, and SB-7 reads fail until every counted class passes."""
    source = [_emb("emb_e1_fr3", "213 S1 in 2,048: one-sided 95 % upper bound 0.116 > 1e-5", "fail", 25),
              _emb("emb_e3_ur10e_2f85_stiff", "1 S1 in 2,048: one-sided 95 % upper bound 0.00231", "fail", 26)]
    base = [_row("WP7", "SB-7", "SB-7 (xpbd_box_stack class, main ab00a559)", _PASS, "pass", 24), *source]
    by = _board(tmp_path / "a", base)
    assert (by["SB-7"].value, by["SB-7"].verdict) == (9.9e-6, "fail")       # no route variant measured yet
    assert "emb_e8 route variant (emb_e8_*_route): not run yet, a counted class" in by["SB-7"].reason
    e1 = by["SB-7:emb_e1_fr3 class, main ab00a559 (WP7)"]
    assert e1.verdict == "fail" and "fail: source contact parameters (τ = 20 ms); reported, not counted" in e1.reason
    assert "(τ = 5 ms)" in by["SB-7:emb_e3_ur10e_2f85_stiff class, main ab00a559 (WP7)"].reason
    routes = _routes(30)
    assert _board(tmp_path / "b", base + routes)["SB-7"].verdict == "pass"   # every counted class passes
    seven = _board(tmp_path / "c", base + routes[:-1])["SB-7"]
    assert seven.verdict == "fail" and "emb_e8 route variant" in seven.reason


def test_a_class_mujoco_c_reproduces_is_reported_not_counted(tmp_path) -> None:
    """Coordinator 2026-11-05 22:33Z: E7 H1-2's route variant fails on the source hand, reproduced in MuJoCo C
    float64, so it is reported, not counted; SB-7 counts e1-e6 and e8 and can pass without it."""
    assert "e7" not in SB7.ROUTE_EMBODIMENTS and len(SB7.ROUTE_EMBODIMENTS) == 7
    e7 = [_emb("emb_e7_h1_2_route", "153 S1 in 2,048: upper bound 0.085; reproduced in MuJoCo C: 153 S1 (rate "
               "7.47e-2)", "fail", 50),
          _emb("emb_e7_h1_2", "2,044 S1 in 2,048: one-sided 95 % upper bound 0.999", "fail", 51)]
    by = _board(tmp_path, [_emb("xpbd_box_stack", _PASS, "pass", 24), *_routes(30), *e7])
    assert by["SB-7"].verdict == "pass"
    route = by["SB-7:emb_e7_h1_2_route class, main ab00a559 (WP7)"]
    assert route.verdict == "fail" and "not counted" in route.reason
    assert "reproduced in MuJoCo C: 153 S1 (rate 7.47e-2); reported, not counted" in route.reason
    assert "its route variant is reported too" in by["SB-7:emb_e7_h1_2 class, main ab00a559 (WP7)"].reason


def test_each_counted_class_row_shows_its_reproduced_s1_count_and_rate(tmp_path) -> None:
    """Coordinator 2026-11-05 23:27Z: SB-7 measures what Alloy adds over MuJoCo C float64; a reproduced S1 is listed,
    not counted, and every counted class row shows the reproduced count and rate (or says the row states none)."""
    e5 = _emb("emb_e5_fr3_duo_route", "0 counted S1 in 303,104 clean episodes, one-sided 95 % bound 3/N = 9.90e-6; "
              "reproduced in MuJoCo C: 1 S1 (rate 3.3e-6), 100015:183", "pass", 40)
    cut = _row("WP7", "SB-7", "SB-7 (cut_block_stroke class: cutting)", "1,372 S1 in 2,048: upper bound 0.687",
               "fail", 41)
    by = _board(tmp_path, [_emb("xpbd_box_stack", _PASS, "pass", 24), e5, cut])
    assert "reproduced in MuJoCo C float64: 1 S1 (rate 3.3e-6), listed, not counted" in (
        by["SB-7:emb_e5_fr3_duo_route class, main ab00a559 (WP7)"].reason)
    assert "reproduced in MuJoCo C float64: 0 S1" in by["SB-7:xpbd_box_stack class, main ab00a559 (WP7)"].reason
    assert "not stated in the row" in by["SB-7:cut_block_stroke class: cutting (WP7)"].reason
    assert by["SB-7"].reason.startswith("SB-7 measures the failures Alloy adds over MuJoCo C float64")


_CUT = "SB-7 (cut_block_stroke class: cutting, main 42283f1d), calibration"
_CUT_VALUE = "1,372 S1 in 2,048 random-legal OU episodes: rate 0.67, Clopper-Pearson one-sided 95 % upper bound 0.687"


def _cut_board(tmp_path, monkeypatch, verdict_cell: str, label: str = _CUT):
    """A passing route row per embodiment plus the cutting class's only row, whose note line 83 reads `verdict_cell`."""
    from alloy.bench import note_rows as NR

    notes = tmp_path / "notes"
    notes.mkdir(parents=True)
    (notes / "wp7-results.md").write_text("\n" * 80 + "| metric id | value | verdict | command | commit |\n"
                                          f"|---|---|---|---|---|\n| {label} | {_CUT_VALUE} | {verdict_cell} | x | y |\n")
    monkeypatch.setattr(NR, "NOTES", notes)
    assert NR.verdict_cell("docs/notes/wp7-results.md:83") == verdict_cell and NR.verdict_cell("no/such.md:1") == ""
    return _board(tmp_path, _routes(30) + [_row("WP7", "SB-7", label, _CUT_VALUE, "fail", 83)])


def test_a_row_its_note_marks_superseded_does_not_decide(tmp_path, monkeypatch) -> None:
    """Audit AB-inv-reports-12: SB-7's 0.687 was the bound of a row whose note says "superseded by <commit>, re-run
    pending". Such a row is shown beside; its class has no current row, so the headline shows no number and cannot
    read pass. The same row without the mark decides as before."""
    by = _cut_board(tmp_path / "a", monkeypatch, "fail")
    assert (by["SB-7"].value, by["SB-7"].verdict) == (0.687, "fail")
    by = _cut_board(tmp_path / "b", monkeypatch, "fail; superseded by eb75be132 (cut faces collide). Re-run pending")
    head, tag = by["SB-7"], "calibration (superseded in its note, no current row)"
    assert (head.value, head.verdict) == (None, "inconclusive"), head
    assert f"{tag}: not in the value or the verdict, re-run pending" in head.reason
    beside = by[f"SB-7:(cut_block_stroke class: cutting, main 42283f1d), {tag} (WP7)"]
    assert (beside.value, beside.verdict) == (0.687, "fail")
    by = _cut_board(tmp_path / "c", monkeypatch, "fail", label=_CUT.replace("calibration", "superseded"))
    assert (by["SB-7"].value, by["SB-7"].verdict) == (None, "inconclusive")      # marked in the label instead


def test_every_published_rate_carries_the_population_changes() -> None:
    """Audit AB-laws-7: the S1 thresholds are fixed but the counted population was narrowed by rulings. Each change
    has a date, a commit and a reason; the headline text carries the count and the board the table."""
    assert len(SB7.POPULATION) >= 10 and [c.date for c in SB7.POPULATION] == sorted(c.date for c in SB7.POPULATION)
    assert all(c.date and c.commit and c.what and c.removed and c.ruling for c in SB7.POPULATION)
    assert {c.kind for c in SB7.POPULATION} == {"counted", "rule", "board", "scorer"}
    assert sum(c.widens for c in SB7.POPULATION) == 3 and "3 widened it" in SB7.population_line()
    for commit in ("a173d4220", "20a7132b3", "db0e58259", "6f4929b7e", "d0a527682", "86addd7f9", "ef7339732"):
        assert any(c.commit == commit for c in SB7.POPULATION), commit
    line = SB7.population_line()
    assert f"counts changed {len(SB7.POPULATION)} times" in line and line in SB7.note()
    table = SB7.population_md()
    assert "## SB-7 counted population" in table and len([r for r in table if r.startswith("| 2026-")]) == len(SB7.POPULATION)
