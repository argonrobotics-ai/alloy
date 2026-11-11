# SPDX-License-Identifier: Apache-2.0
"""Scoreboard (SPEC §5, §3.10 `scoreboard`): one `ScoreRow` per §5 metric, per law result and per baseline scenario.

Inputs are report/1 files (`alloy.bench.report/1`, one LawResult each) and run/1 JSON lines (`alloy.bench.run/1`,
the repo-root `bench/` harness). Rules: a metric with no producing result is `not_run`; `not_run` never counts as a
pass; deferred metrics carry `status="deferred"` and `release_rows` drops them.
A verdict feeds a §5 id when its name is that id (or starts with `<id>:`), or through `LAW_METRIC_IDS`. The headline
rows SB-3..SB-11 take value, unit and verdict from their producer rows (`alloy.bench.headline`); a row is `not_run`
only when nothing feeding it was measured. `source_rows` point at the producer rows (`<notes>.md:<line>`) when the
report came from the results collector, else at the report file.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alloy.bench import default_route as DR
from alloy.bench import full_mode as FM
from alloy.bench import gpu_repeat as GR
from alloy.bench import sb7 as SB7
from alloy.bench.baseline import RANK, key_of, law_verdict, mode_of
from alloy.bench.headline import HEADLINES, headline_rows, measured_verdict
from alloy.bench.law import (CONDITIONAL, INFORMATIONAL, KNOWN_GAPS, ROUTE_GAPS, SB12_DETAIL, TRADES, caveat,
                              counted_verdict, law_prefix, not_gated)
from alloy.bench.metric_catalogue import BY_ID, DEFERRED, METRICS, MetricDef
from alloy.bench.provenance import commit_short
from alloy.bench.report import SCHEMA as REPORT_SCHEMA
from alloy.bench.report import from_dict
from alloy.bench.review import mark_under_review, review_json, review_md
from alloy.bench.types import RUN_SCHEMA, LawResult, ScoreRow

log = logging.getLogger("alloy.bench")

SB12_LAWS = frozenset(f"A{i}" for i in range(1, 18)) | {"D1"}  # A17: SCR-WP11-2
# (law prefix, metric name) -> §5 id, for law metrics whose names are physical rather than §5 ids
LAW_METRIC_IDS: dict[tuple[str, str], str] = {("A5", "creep"): "BEN-3", ("R3", "creep"): "BEN-3",
                                              ("A16", "selftest_pass_rate"): "BEN-1"}
# §5 ids whose value is a rate derived from the fed error (A16 reports 1 - pass rate as its error)
VALUE_FROM_ERROR = {"BEN-1": lambda e: 1.0 - e}
Loaded = tuple[str, LawResult]


def _iter_files(roots: Sequence[Path], suffix: str) -> Iterable[Path]:
    for root in roots:
        root = Path(root)
        if root.is_file() and root.suffix == suffix:
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob(f"*{suffix}"))


def load_reports(roots: Sequence[Path]) -> list[Loaded]:
    out = []
    for path in _iter_files(roots, ".json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("schema") == REPORT_SCHEMA:
            out.append((str(path), from_dict(data)))
    return out


def load_run_rows(roots: Sequence[Path]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for path in _iter_files(roots, ".jsonl"):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("schema") == RUN_SCHEMA:
                out.append((str(path), row))
    return out


def verdict_reason(r: LawResult, v: Any) -> str:
    """Why a verdict did not pass: `monitors["reasons"][name]` (multi-verdict reports, e.g. WP12's collector),
    else `monitors["not_run_reason"]` (`not_run_result`), else `monitors["error"]` (`run_law` failures)."""
    if v.verdict == "pass":
        return ""
    per = r.monitors.get("reasons") or {}
    text = per.get(v.name) if isinstance(per, dict) else None
    return str(text or r.monitors.get("not_run_reason") or r.monitors.get("error") or "")


def _join(reasons: Iterable[str]) -> str:
    return "; ".join(dict.fromkeys(x for x in reasons if x))


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def run_scenario_rows(rows: Sequence[tuple[str, dict[str, Any]]]) -> list[ScoreRow]:
    """One row per run/1 scenario: worst `accuracy_error` over ok rows, verdict from `metrics.law_pass`."""
    by_scn: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path, row in rows:
        by_scn[row["scenario"]].append((path, row))
    return [_scenario_row(scn, items) for scn, items in sorted(by_scn.items())]


def _scenario_row(scn: str, items: list[tuple[str, dict[str, Any]]]) -> ScoreRow:
    ok = [r for _, r in items if r.get("status") == "ok" and r.get("accuracy_error") is not None]
    passes = [r.get("metrics", {}).get("law_pass") for r in ok]
    if not ok:
        verdict = "error" if any(r.get("status") == "error" for _, r in items) else "not_run"
    elif any(p is False for p in passes) or not all(r.get("stable", True) for r in ok):
        verdict = "fail"
    elif all(p is True for p in passes):
        verdict = "pass"
    else:
        verdict = "inconclusive"  # rows without a law verdict cannot pass
    value = max((float(r["accuracy_error"]) for r in ok), default=None)
    bounds = [r.get("metrics", {}).get("law_bound_mm") for r in ok]
    unit = ok[0].get("accuracy_unit", "") if ok else ""
    target = f"<= {_fmt(min(b for b in bounds if b is not None))} {unit}" if any(b is not None for b in bounds) else ""
    if ok and ok[0].get("accuracy_ref"):
        target = f"{target} ({ok[0]['accuracy_ref']})".strip()
    last = max(items, key=lambda it: it[1].get("timestamp", ""))[1]
    reason = "" if ok else (_join(str(r.get("error") or "") for _, r in items) or "no run/1 row with status ok")
    return ScoreRow(scn, value, unit, target, verdict, tuple(sorted({p for p, _ in items})), last.get("command", ""),
                    str(last.get("versions", {}).get("alloy", "")), last.get("timestamp", ""), reason=reason)


def _tag(r: LawResult, sep: str = "/") -> str:
    """`law/solver/device`, with `/full` for an explicit full-mode run (quick is the board's default mode)."""
    full = str(r.provenance.get("quick", "")) == "False" and mode_of(r) == "full"
    parts = [r.law, r.solver, r.device] + (["full"] if full else [])
    return sep.join(parts)


def law_rows(reports: Sequence[Loaded]) -> list[ScoreRow]:
    """One row per (law, solver, device): the worst metric error ratio is not summed; verdict = worst metric."""
    out = []
    for path, r in reports:
        worst = max(r.verdicts, key=lambda v: RANK[v.verdict], default=None)
        value = worst.error if worst is not None and math.isfinite(worst.error) else None
        target = f"<= {_fmt(worst.bound)} {worst.unit} ({worst.name})" if worst and math.isfinite(worst.bound) else ""
        measured = law_verdict(r)
        verdict = counted_verdict(r.law, measured)
        reason = _join(verdict_reason(r, v) for v in r.verdicts if v.verdict == measured)
        counted = f"measured {measured}, counted {verdict}" if verdict != measured else ""
        reason = _join([not_gated(r.law), counted, caveat(r.law), GR.sample_label(r.device), reason])
        out.append(ScoreRow(f"law:{_tag(r, ':')}", value, worst.unit if worst else "", target,
                            verdict, (path,), r.command, r.provenance.get("alloy", ""),  # type: ignore[arg-type]
                            r.provenance.get("date", ""), reason=reason))
    return out


def _feeds(r: LawResult) -> Iterable[tuple[str, Any]]:
    for v in r.verdicts:
        mid = v.name.split(":", 1)[0]
        if mid in BY_ID:
            yield mid, v
        elif (law_prefix(r.law), v.name.split("@", 1)[0]) in LAW_METRIC_IDS:  # rig metrics are "<name>@<cond>"
            yield LAW_METRIC_IDS[(law_prefix(r.law), v.name.split("@", 1)[0])], v


def latest_per_key(reports: Sequence[Loaded]) -> list[Loaded]:
    """The newest report per (law, solver, device, mode), in input order of first appearance; older reruns drop out."""
    newest: dict[str, Loaded] = {}
    for path, r in sorted(reports, key=lambda pr: (pr[1].provenance.get("date", ""), pr[0])):
        newest[key_of(r)] = (path, r)
    keep = {id(r) for _, r in newest.values()}
    return [(p, r) for p, r in reports if id(r) in keep]


def sb12_latest(reports: Sequence[Loaded]) -> list[Loaded]:
    """The newest A1-A17 / D1 result per (law, solver, device, mode), whatever its verdict."""
    return [(p, r) for p, r in latest_per_key(reports)
            if law_prefix(r.law) in SB12_LAWS and not not_gated(r.law)]


def known_gap_results(reports: Sequence[Loaded], table: Mapping[str, str] = KNOWN_GAPS) -> list[str]:
    """`law/solver/device verdict` of the newest result per key of every law in `table` (shown, never counted)."""
    newest: dict[str, LawResult] = {}
    for _, r in sorted(reports, key=lambda pr: (pr[1].provenance.get("date", ""), pr[0])):
        if r.law in table:
            newest[key_of(r)] = r
    return sorted(f"{_tag(r)} {law_verdict(r)}" for r in newest.values())


def sb12_results(reports: Sequence[Loaded]) -> list[Loaded]:
    """Newest results that actually ran: a newer not_run (e.g. a law now declining CPU) replaces an older error."""
    return [(p, r) for p, r in sb12_latest(reports) if law_verdict(r) != "not_run"]


def _counted(r: LawResult) -> str:
    """The verdict SB-12 counts: the full-mode row's for a law whose quick mode is outside its range (FM), else the
    measured verdict under the law's ruling, if any."""
    full = FM.counted(r.law, law_verdict(r))
    return full.verdict if full is not None else counted_verdict(r.law, law_verdict(r))


def _counted_tag(r: LawResult) -> str:
    """`law/solver/device`, or `law/solver/<device>/full` where the full-mode row decides (the board names the mode)."""
    fm = FM.FULL_MODE.get(r.law)
    return f"{r.law}/{r.solver}/{fm.device}/full" if fm is not None else _tag(r)


def _full_mode(rel: Sequence[Loaded]) -> tuple[str, list[str]]:
    """("; <which mode fed each law>" notes, full-mode row sources) for the results whose full-mode row decides."""
    got = [c for c in (FM.counted(r.law, law_verdict(r)) for _, r in rel) if c is not None]
    notes = sorted({c.note for c in got})
    return "".join(f"; {n}" for n in notes), sorted({c.source for c in got if c.source})


def _measured_counted(r: LawResult) -> str:
    measured, counted = law_verdict(r), _counted(r)
    return measured if measured == counted else f"measured {measured}, counted {counted}"


def _caveat_note(rel: Sequence[Loaded]) -> str:
    """"; <law/solver/device> measured <v>, counted <v>: caveat: ..." for every result whose law carries a ruling."""
    got = sorted({f"{_tag(r)} {_measured_counted(r)}: {caveat(r.law)}" for _, r in rel if r.law in CONDITIONAL})
    return "".join(f"; {x}" for x in got)


def _not_run_anywhere(tried: Sequence[LawResult], default_only: bool = False) -> list[str]:
    """Laws whose newest results are all not_run (a law that runs on CUDA only and ran there is not missing)."""
    pick = [r for r in tried if not default_only or DR.on_default_route(r.law, r.solver)]
    ran = {r.law for r in pick if law_verdict(r) != "not_run"}
    return sorted({r.law for r in pick if law_verdict(r) == "not_run"} - ran)


def _sb12_population(rel: Sequence[Loaded], skipped: Sequence[str]) -> tuple[float, str, str]:
    """(pass share, verdict, reason) over results that ran; laws that did not run keep it from passing."""
    failing = sorted(f"{_counted_tag(r)} {_counted(r)}" for _, r in rel if _counted(r) != "pass")
    n_pass = len(rel) - len(failing)
    reason = f"{n_pass}/{len(rel)} (law, solver, device) results that ran pass"
    reason += f"; not counted (not_run): {', '.join(skipped)}" if skipped else ""
    reason += f"; not passing: {', '.join(failing)}" if failing else ""
    verdict = "fail" if failing else ("inconclusive" if skipped else "pass")
    return n_pass / len(rel), verdict, reason


def _sb12(reports: Sequence[Loaded], m: MetricDef) -> list[ScoreRow]:
    """SB-12 on Alloy's default route (verdict), then over all configurations (`SB-12:all configurations`)."""
    every = sb12_results(reports)
    if not every:
        return [_empty(m)]
    tried = [r for _, r in sb12_latest(reports)]
    skipped = _not_run_anywhere(tried)
    default = [(p, r) for p, r in every if DR.on_default_route(r.law, r.solver)]
    skipped_default = _not_run_anywhere(tried, default_only=True)
    v_all, verdict_all, why_all = _sb12_population(every, skipped)
    gaps, info = known_gap_results(reports), known_gap_results(reports, INFORMATIONAL)
    gap_note = f"; known gaps, not counted (SPEC §5.3): {', '.join(gaps)}{_trade_note()}" if gaps else ""
    gap_note += f"; informational, not counted: {', '.join(info)}" if info else ""
    gap_note += "".join(f"; route known gap (beside): {g.text}" for g in ROUTE_GAPS)
    gap_note += "".join(f"; {SB12_DETAIL[k]}" for k in sorted({law_prefix(r.law) for _, r in every} & set(SB12_DETAIL)))
    full_note, full_src = _full_mode(every)
    gap_note = full_note + gap_note
    why_all += _caveat_note(every) + gap_note
    rows = []
    if default:
        v_def, verdict_def, why_def = _sb12_population(default, skipped_default)
        why = (f"verdict on the Alloy default route ({DR.describe()}); all configurations: {v_all:.3f} "
               f"{verdict_all}; {why_def}{_caveat_note(default)}{gap_note}")
        rows.append(_row(m, v_def, verdict_def, [p for p, _ in default] + full_src, [r for _, r in default],
                         reason=why))
    else:
        rows.append(_empty(m, why="no result on the Alloy default route"))
    sub = _row(m, v_all, verdict_all, [p for p, _ in every] + full_src, [r for _, r in every], reason=why_all)
    return (rows + [dataclasses.replace(sub, metric_id=f"{m.id}:all configurations")] + _trade_rows(reports, m)
            + _route_gap_rows(m))


def _trade_rows(reports: Sequence[Loaded], m: MetricDef) -> list[ScoreRow]:
    """`SB-12:known gap <law/solver/device>` for each law a ruling traded (`law.TRADES`): its newest value, the
    numbers before and after the change, and the ruling. Not counted in SB-12 (a known gap)."""
    newest: dict[str, Loaded] = {}
    for p, r in sorted(reports, key=lambda pr: (pr[1].provenance.get("date", ""), pr[0])):
        if r.law in TRADES and law_verdict(r) != "not_run":
            newest[key_of(r)] = (p, r)
    rows = []
    for p, r in newest.values():
        t = TRADES[r.law]
        v = next((x for x in r.verdicts if x.name == t.metric), None)
        if v is None:
            continue
        why = f"known gap, not counted in SB-12; trade: {t.why}; this result {t.metric} {v.error:.3g}"
        row = _row(m, v.error, v.verdict, [p], [r], reason=why)
        rows.append(dataclasses.replace(row, metric_id=f"{m.id}:known gap {_tag(r)}", unit=v.unit,
                                        target=f"<= {v.bound:g} (not gated)"))
    return sorted(rows, key=lambda x: x.metric_id)


def _route_gap_rows(m: MetricDef) -> list[ScoreRow]:
    """`SB-12:known gap <name>` for each default-route gap a ruling measured (`law.ROUTE_GAPS`); the verdict is the
    gap's own against its target, never counted in SB-12."""
    return [ScoreRow(f"{m.id}:known gap {g.name}", g.value, g.unit, g.target,
                     "pass" if g.value <= g.bound else "fail", g.sources, "", "", "",
                     "active", f"known gap, not counted in SB-12: {g.text}; {g.detail}") for g in ROUTE_GAPS]


def _trade_note() -> str:
    return "".join(f"; traded known gap (beside): {t.why}" for t in TRADES.values())


def held_out_gaps(r: LawResult) -> list[float]:
    """BEN-2 terms of one rig result: (RE(valid) - RE(calib)) / bound per judged metric@condition."""
    calib = r.monitors.get("calib_error", {}) or {}
    out = []
    for v in r.verdicts:
        c = calib.get(v.name)
        if c is not None and math.isfinite(v.error) and v.bound > 0:
            out.append((v.error - float(c)) / v.bound)
    return out


def _ben2(reports: Sequence[Loaded], m: MetricDef) -> ScoreRow:
    rel = [(p, r) for p, r in reports if r.provenance.get("suite") == "rig" and held_out_gaps(r)]
    if not rel:
        return _empty(m)
    worst = max(g for _, r in rel for g in held_out_gaps(r))
    return _row(m, worst, "pass" if worst <= 0.5 else "fail", [p for p, _ in rel], [r for _, r in rel])


def _empty(m: MetricDef, status: str | None = None, why: str = "") -> ScoreRow:
    st = status or ("deferred" if m.id in DEFERRED else "active")
    reason = why or f"no result fed {m.id} (producer: {m.producer})"
    if st == "deferred":
        reason = f"deferred (SPEC §5): {reason}"
    return ScoreRow(m.id, None, m.unit, m.target, "not_run", (), "", "", "", st, reason)  # type: ignore[arg-type]


def _row(m: MetricDef, value: float | None, verdict: str, paths: Sequence[str], results: Sequence[LawResult],
         status: str | None = None, reason: str = "") -> ScoreRow:
    st = status or ("deferred" if m.id in DEFERRED else "active")
    last = max(results, key=lambda r: r.provenance.get("date", "")) if results else None
    prov = last.provenance if last else {}
    if st == "deferred":
        reason = f"deferred (SPEC §5){': ' + reason if reason else ''}"
    return ScoreRow(m.id, value, m.unit, m.target, verdict, tuple(sorted(set(paths))),  # type: ignore[arg-type]
                    last.command if last else "", prov.get("alloy", ""), prov.get("date", ""),
                    st, reason)  # type: ignore[arg-type]


def _real_rig(r: LawResult) -> bool:
    """A rig result on recorded data (synthetic datasets carry protocol_version "synthetic/...")."""
    return r.provenance.get("suite") == "rig" and not str(r.data.get("protocol", "synthetic")).startswith("synthetic")


def catalogue_rows(reports: Sequence[Loaded]) -> list[ScoreRow]:
    """One row per §5 metric id, from the verdicts that feed it (worst value, worst verdict)."""
    fed: dict[str, list[tuple[str, LawResult, Any]]] = defaultdict(list)
    for path, r in reports:
        for mid, v in _feeds(r):
            fed[mid].append((path, r, v))
    rows = []
    for m in METRICS:
        if m.id in HEADLINES:
            rows += headline_rows(HEADLINES[m.id], m, reports)
            continue
        if m.id in ("SB-12", "BEN-2"):
            rows += _sb12(reports, m) if m.id == "SB-12" else [_ben2(reports, m)]
            continue
        status = "deferred" if m.id == "SB-2" and not any(_real_rig(r) for _, r in reports) else None
        items = fed.get(m.id, [])
        if not items:
            rows.append(_empty(m, status, "no recorded rig data (synthetic datasets only)" if status else ""))
            continue
        rows.append(_fed_row(m, items, status))
        solvers = sorted({r.solver for _, r, _ in items})
        if len(solvers) > 1:
            for sv in solvers:
                sub = [it for it in items if it[1].solver == sv]
                rows.append(dataclasses.replace(_fed_row(m, sub, status), metric_id=f"{m.id}:{sv}"))
    return rows


def _fed_row(m: MetricDef, items: list[tuple[str, LawResult, Any]], status: str | None) -> ScoreRow:
    """Worst error and worst verdict over the verdicts feeding one §5 id (value transformed where defined)."""
    errors = [v.error for _, _, v in items if math.isfinite(v.error)]
    value = max(errors) if errors else None
    if value is not None and m.id in VALUE_FROM_ERROR:
        value = VALUE_FROM_ERROR[m.id](value)
    worst = measured_verdict([v.verdict for _, _, v in items])
    shown = "not_run" if worst == "inconclusive" and any(v.verdict == "not_run" for _, _, v in items) else worst
    reason = _join(verdict_reason(r, v) for _, r, v in items if v.verdict == shown)
    if worst == "inconclusive" and shown == "not_run":
        reason = f"partial: measured parts pass, others not_run: {reason}"
    if worst != "pass" and not reason:
        laws = ", ".join(sorted({r.law for _, r, _ in items}))
        reason = f"{worst} verdicts from {laws} carry no reason"
    paths = [_source(p, r, v) for p, r, v in items]
    return _row(m, value, worst, paths, [r for _, r, _ in items], status, reason)


def _source(path: str, r: LawResult, v: Any) -> str:
    """The producer row behind a verdict (`<notes>.md:<line>` from the results collector), else the report path."""
    rows = r.monitors.get("rows") or {}
    meta = rows.get(v.name) if isinstance(rows, dict) else None
    return str(meta.get("source") or path) if isinstance(meta, dict) else path


def scoreboard(roots: Sequence[Path]) -> list[ScoreRow]:
    """All score rows from report/1 and run/1 files under `roots`; `not_run` never counts as a pass."""
    reports = latest_per_key(load_reports(roots))  # a rerun replaces the older report of the same key
    runs = load_run_rows(roots)
    log.info("scoreboard: %d reports, %d run/1 rows", len(reports), len(runs))
    return mark_under_review(catalogue_rows(reports) + law_rows(reports) + run_scenario_rows(runs))


def release_rows(rows: Sequence[ScoreRow]) -> list[ScoreRow]:
    """Rows that count for a release: deferred rows are excluded."""
    return [r for r in rows if r.status == "active"]


def pass_rate(rows: Sequence[ScoreRow]) -> float:
    """Share of active rows that pass; not_run, inconclusive, fail and error all count against it."""
    active = release_rows(rows)
    return sum(r.verdict == "pass" for r in active) / len(active) if active else 0.0


def write_scoreboard(rows: Sequence[ScoreRow], path: Path = Path("bench/SCOREBOARD.json")) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "alloy.bench.scoreboard/1", "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "tree": commit_short(), "pass_rate_active": pass_rate(rows), "rows": [asdict(r) for r in rows],
               "rulings": [{"law": k, "met": c.met, "caveat": c.caveat, "ruling": c.ruling}
                           for k, c in CONDITIONAL.items()],
               "under_review": review_json(), "gpu_determinism": {"claim": GR.CLAIM, "ruling": GR.RULING},
               "sb7_population": [dataclasses.asdict(c) for c in SB7.POPULATION]}
    path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    return path


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


BUILT = ("Built {date} from tree {tree}: the board was assembled then, from the results notes and report files of "
         "that tree. It is not a re-run: each row's `run` column gives the date and commit of the run behind its "
         "number (the commit alone for a row read from a results note, whose own text dates it; empty where the "
         "producer states none), and a row older than a change to what it measures is stale until re-run.")


def render_markdown(rows: Sequence[ScoreRow]) -> str:
    """A header saying when and from which tree the board was built, then one table row per score row with the date
    and commit of its own run (audit AB-inv-reports-12); `source` lists producer-row pointers (`<notes>.md:<line>`),
    not report paths."""
    lines = [BUILT.format(date=time.strftime("%Y-%m-%d %H:%M %z"), tree=commit_short()), "",
             "| id | value | unit | target | verdict | status | run | reason | source |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        val = "" if r.value is None else _fmt(r.value)
        src = ", ".join(s for s in r.source_rows if ".md:" in s)
        noted = any(".md:" in s for s in r.source_rows)     # a note row: its date here is the collection's, not the run's
        run = " @ ".join(x for x in ("" if noted else r.date[:10], r.commit) if x and x != "—")
        lines.append(f"| {_cell(r.metric_id)} | {val} | {r.unit} | {_cell(r.target)} | {r.verdict} | {r.status} | "
                     f"{_cell(run)} | {_cell(r.reason)} | {src} |")
    return "\n".join(lines + _rulings_md() + SB7.population_md() + GR.board_md() + review_md())


def _rulings_md() -> list[str]:
    """Board notes: rulings on laws whose pass holds under a condition (alloy.bench.law.CONDITIONAL)."""
    if not CONDITIONAL:
        return []
    out = ["", "## Rulings", ""]
    for law_id, c in CONDITIONAL.items():
        state = "met: the pass counts" if c.met else "not met: a pass counts as fail"
        out.append(f"- **{law_id}** ({state}). Caveat: {c.caveat}. Ruling: {c.ruling}.")
    return out


__all__ = ["scoreboard", "release_rows", "pass_rate", "write_scoreboard", "render_markdown", "load_reports",
           "load_run_rows", "run_scenario_rows", "law_rows", "catalogue_rows", "verdict_reason", "sb12_results",
           "LAW_METRIC_IDS"]
