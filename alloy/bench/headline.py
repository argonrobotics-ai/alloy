# SPDX-License-Identifier: Apache-2.0
"""Headline rows SB-3..SB-11 (SPEC §5.1) fed from the producer rows that measured them.

A producer row is one verdict of a collected results report (`alloy.integration.results`: name `<id>:<package>:<i>`,
row text in `monitors["rows"]`) or of any report/1 file whose verdict is named after a §5 id. Each headline lists its
feeds: which rows count (producer id, label pattern), how to read the number from the value cell (regex, unit) and
whether the row decides the verdict (`decides=False` rows are reported beside it, e.g. SB-5 at noslip off).

Rules:
- verdict = worst verdict over the measured deciding rows; a deciding feed with no measured row turns an
  all-pass into `inconclusive` (partial evidence never passes);
- `not_run` only when no producer row was measured; measured rows that match no deciding feed give `inconclusive`;
- value and unit come from the first deciding feed with a number (worst over its rows); rows reported beside
  the verdict only appear in sub-rows and the reason;
- `source_rows` point at the producer rows (`docs/notes/<file>.md:<line>`, or the report path);
- one sub-row `SB-n:<variant> (<package>)` per matched row keeps every number visible;
- headlines with a default route (SB-3, SB-4; `alloy.bench.default_route`) take value and verdict on the rows Alloy
  uses by default and add a sub-row `SB-n:all configurations` with the value and verdict over every row;
- opt-in route profiles never decide: a T09 row from WP2's fast route profile (SCR-WP3-8) is shown beside SB-5 with
  the profile's caveat (`alloy.scene.fast_profile.FAST_CAVEAT`, cable creep) printed next to its speed;
- rows the results collector does not take (no §5 id, no verdict column) are read from their note by a feed's
  `note` (`alloy.bench.note_rows`), e.g. WP12's T09 rows;
- a superseded rerun (the producer's verdict cell names the commit that fixed it) is shown beside, not decided on.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from alloy.bench import default_route as DR
from alloy.bench import note_rows as NR
from alloy.bench import sb7 as SB7
from alloy.bench import sb9 as SB9
from alloy.bench.baseline import RANK
from alloy.bench.headline_types import NO_CURRENT, NOT_MEASURED, Feed, Headline, ProducerRow, Summary, demote_superseded
from alloy.bench.metric_catalogue import DEFERRED, MetricDef
from alloy.bench.types import LawResult, ScoreRow

log = logging.getLogger("alloy.bench")
Loaded = tuple[str, LawResult]
_NUM = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_RT = _NUM + r"x RT"
_NOSLIP_ON = r"noslip [1-9]"
_SCOPED = r"scoped"
_USD = r"\$" + _NUM + r" per (?:M|million) sim-s"
_NOSLIP_NOTE = ("noslip 50 is the task setting (docs/notes/wp0-noslip.md); on the Alloy default route a scene that asks "
                "for noslip gets Alloy's scoped noslip (alloy.contact.noslip; SCR-WP3-4); MuJoCo Warp's noslip 50 "
                "is shown beside; rows that do not name noslip are noslip off")
_OUT_OF_ENVELOPE = ("informational, not part of the verdict: the earlier press protocol drove coins onto the plate edge "
                    "beyond the CT-2 envelope (impacts <= 2 m/s; WP3 measured 4.2 m/s), where both natives tunnelled")
_MJW_BESIDE = "all configurations (MuJoCo Warp noslip 50 included)"
_FAST = r"fast (?:route )?profile|profile=fast"  # WP2's opt-in T09 route profile (SCR-WP3-8), label or value cell
_FAST_FALLBACK = ("fast route profile (SCR-WP3-8), chain armature 1e-5: cable creep 0.99-1.06 mm per 10 s against "
                  "0.13-0.22 on the default route (CHAIN known gap, CREEP_GENERAL 0.1 mm)")  # WP2 c9d3caa


_FAST_INFO = "informational, not part of the verdict: WP2's opt-in fast route profile (SCR-WP3-8)"
# Board status of the fast-profile numbers (coordinator 2026-11-03); clear it once a rerun on fixed main replaces them.
_FAST_STATUS = ("invalid on main from 8d513582 to aa2a613a, not rerun since: the fast profile read 319-326 of 1,024 "
                "worlds INVALID seated (WP2's home11 rows, under review); WP12's bisect: 8d513582 turned WP5's "
                "ordered MuJoCo route on for T09's fast Newton path, whose contact sort moved T09's appended "
                "precision-pair contacts (WP5's fix on main aa2a613a; route off: 0 of 64 bad); the speeds beside ran "
                "with the route off (uncovered then) and stand for that code")
_WP12_FAST = r"^opt-in fast profile"
_SUPERSEDED = r"rerun|regression"  # WP9's earlier LIQ-8 reruns ("fail (fixed in <commit>)") and its regression check
# Coordinator 2026-11-04 14:33Z put SB-4's liquid half under review for the box-tank wall-seam leak; WP9's census cleared it.
_SEAM_CENSUS = ("SPH leaks rechecked by WP9's wall-seam census (wp9-results 'Triple-corner wall leak' row, fix e9eb9035): "
                "LIQ-8 replay batches 0-3, 1,024 episodes on sph_pour's box tanks, old vs fixed projection: 0 in solid, "
                "0 leaks, 0 escaped; the fix never acted there, so the LIQ-8 rows stand; box-box line contacts "
                "(BUL-0014, main d4864988, ruling 20:08Z re-feed) touch no SB-4 input: CT-2 is a cylinder coin on a "
                "box plate and the SPH half has no box-box pair (WP3's library census: contact counts identical line "
                "on vs off in all 33 scenes with box-box pairs); SCR-JOB2-5 check (ruling 2026-11-05 00:31Z): no SB-4 "
                "liquid row ran with static walls masked: every one is sph_pour, whose receiver and tray are "
                "world-attached and whose SPH entry declares no shapes, so its view keeps all 10 static walls")
# WP5's P3-1 rows on Alloy's ordered SolverMuJoCo route, per scene and world count (mode "T12 route, 1,024 worlds",
# round 5 part 2 on); earlier rounds' NONE / ATOMICS / ALL / sorts rows measured MuJoCo Warp off the route, outside
# the D9 population, and round 5 part 1's T12 rows (mode "route, ...") are superseded by part 3's
_ROUTE_CLASS = r"T\d+(?: / T\d+)? route, \d[\d,]* worlds"
_MJW_ROUTE = NR.NoteSource("wp5", r"^P3-1 R\d(?: / R\d)*, " + _ROUTE_CLASS, label_column="mode")
_MJW_ROUTE_COST = NR.NoteSource("wp5", r"^P3-1 cost, " + _ROUTE_CLASS, label_column="mode")
_R2R_COST = ("informational: the step cost of RUN_TO_RUN on the ordered route vs NONE, worst scene of the row "
             "(WP5 P3-1 round 5; M7 allows 1.3x)")
_WP5_N1 = NR.NoteSource("wp5", r"^P3-1 N=1 \(RUN_TO_RUN, beside SB-6\)", label_column="mode")
_R2R_N1_COST = ("informational, not part of the verdict: RUN_TO_RUN's step cost at N=1 (T12 graph step, 1 world, L4, "
                "vs NONE; WP5 P3-1, newest row; M7 allows 1.3x)")
_WP3_R2R_RT = NR.NoteSource("wp3", r"^SB-6 RUN_TO_RUN N=1", verdict_column="")
_R2R_INFO = ("informational, not part of the verdict: T12 N=1 real time with RUN_TO_RUN on (WP5's ordered route, an "
             "opt-in preset; coordinator 2026-11-02), WP3's newest latency-pack row (its sorts-on 0.877x at dee7f38 "
             "predates WP5's N=1 path)")


def _fast_caveat() -> str:
    """WP2's caveat for any fast-profile speed (alloy.scene.fast_profile.FAST_CAVEAT), after the profile's board
    status while one is set (`_FAST_STATUS`)."""
    try:
        from alloy.scene.fast_profile import FAST_CAVEAT
    except ImportError as exc:  # a broken producer must not take the scoreboard down
        log.warning("headline: fast-profile caveat unavailable, using WP6's copy (%s)", exc)
        FAST_CAVEAT = _FAST_FALLBACK
    return f"{_FAST_STATUS}; {FAST_CAVEAT}" if _FAST_STATUS else FAST_CAVEAT


# §5.2 CPL-4 as amended (coordinator 2026-11-04 20:46Z, docs/notes/wp0-scr-log.md "SB-8 / CPL-4 protocol"): only the
# 8-height phase-sweep rows (>= 60 s of sim, every 1 s window, worst height) feed SB-8 and CPL-4; per scene WP6's
# newest row decides (its note lists sweeps oldest first, e.g. a Newton re-pin's rerun after the previous pin's).
_CPL4 = Feed("CPL-4", _NUM + r" ?%/(?:sim-)?min", "%/sim-min", label=r"8-height",
             newest=r"\b(?:dissipative|conservative)\b", newest_last=True)
_CPL4_NOTE = ("worst CPL-4 scene at its worst height: 8-height phase sweep on box on cloth, >= 60 s of sim, every 1 s "
              "window (§5.2 CPL-4 as amended 2026-11-04 20:46Z, on main with D20's welded-body E_scale); per scene "
              "the newest sweep decides, earlier ones are beside as superseded; WP4's 2 s single-height rows are the "
              "smoke form and feed neither")


HEADLINES: dict[str, Headline] = {h.id: h for h in (
    Headline("SB-3", (Feed("CT-1", r"p99 " + _NUM + r" ?[µμu]m", "µm", default=DR.precision_label()),), "lower",
             "worst p99 over the CT-1 rows; each row's verdict also holds max <= 2 max_penetration",
             default_route=True),
    Headline("SB-4", (Feed("CT-2", r"(\d+) tunnel", "events", label="routed", default=DR.precision_label("routed"),
                           quantity="pass-through"),
                      Feed("LIQ-8", r"(\d+) (?:escaped|leak)", "events", tag="SPH leaks", quantity="pass-through",
                           exclude=_SUPERSEDED),
                      Feed("LIQ-8", r"(\d+) (?:escaped|leak)", "events", tag="SPH leaks, superseded rerun",
                           label=r"rerun", decides=False, quantity="pass-through"),
                      Feed("CT-2", r"earlier (\d+) events", "events", tag="outside the CT-2 envelope, newton (old press)",
                           label="routed / newton", info=_OUT_OF_ENVELOPE, fail_above=0.0),
                      Feed("CT-2", r"(\d+) events on native=\W?mujoco", "events",
                           tag="outside the CT-2 envelope, mujoco (old press, seed 4)", label="routed / newton",
                           info=_OUT_OF_ENVELOPE, fail_above=0.0)), "lower",
             "tunnel events in the routed thin-wall sweep (impacts <= 2 m/s, seated press) plus SPH leaks; " + _SEAM_CENSUS,
             default_route=True),
    Headline("SB-5", (Feed("SB-5", _NUM + "x", "x RT", tag="T12 scoped noslip 50", label=_SCOPED, quantity="T12",
                           newest=r"\bT12\b", newest_last=True),   # WP3 lists T12 rounds oldest first (BUL-0014 rerun)
                      Feed("PAR-3", _RT, "x RT", tag="T12 MuJoCo Warp noslip 50", label="noslip 50", exclude=_SCOPED,
                           alt=True, quantity="T12"),
                      Feed("PAR-4", _RT, "x RT", tag="T09", quantity="T09", not_marker=_FAST),
                      Feed("PAR-4", _RT, "x RT", tag="T09 default route", quantity="T09",
                           note=NR.NoteSource("wp12", r"^WP12-T09-RT default route")),
                      Feed("PAR-4", _RT, "x RT", tag="T09 fast route profile (opt-in)", quantity="T09", marker=_FAST,
                           info=_FAST_INFO, fail_below=5.0, caveat=_fast_caveat()),
                      Feed("PAR-4", _NUM + r"x at", "x RT", tag="T09 fast route profile, card start (opt-in)",
                           quantity="T09", info=_FAST_INFO, fail_below=5.0, caveat=_fast_caveat(),
                           note=NR.NoteSource("wp12", _WP12_FAST, column="rt from the card start", verdict_column="")),
                      Feed("PAR-4", _NUM + r"x at", "x RT", tag="T09 fast route profile, seated (opt-in)",
                           quantity="T09", info=_FAST_INFO, fail_below=5.0, caveat=_fast_caveat(),
                           note=NR.NoteSource("wp12", _WP12_FAST, column="rt seated", verdict_column="")),
                      Feed("PAR-3", _RT, "x RT", tag="T12 noslip off", exclude=_NOSLIP_ON, decides=False,
                           quantity="T12")),
             "higher", _NOSLIP_NOTE, default_route=True, beside=_MJW_BESIDE),
    Headline("SB-6", (Feed("SB-6", _NUM + "x", "x RT", label=r"\bT12\b", quantity="T12", newest=r"\bT12\b",
                           newest_last=True),
                      Feed("SB-6", r"^\**" + _NUM + "x", "x RT", tag="T12 N=1 RUN_TO_RUN (opt-in)",
                           quantity="T12", info=_R2R_INFO, fail_below=1.0, note=_WP3_R2R_RT, newest=r"^SB-6\b",
                           newest_last=True),
                      Feed("SB-6", r"\(" + _NUM + "x", "x step time", tag="T12 N=1 RUN_TO_RUN step cost (opt-in)",
                           quantity="T12 cost", info=_R2R_N1_COST, fail_above=1.3, note=_WP5_N1, newest=r"^P3-1 N=1\b",
                           newest_last=True),
                      Feed("PAR-8", _NUM + "x", "x RT", tag="T12 N=1 noslip off", exclude=r"noslip", alt=True,
                           quantity="T12"),
                      Feed("PAR-8", _NUM + "x", "x RT", tag="T12 N=1 noslip off, other VM", label="noslip off",
                           decides=False, alt=True, quantity="T12"),
                      Feed("PAR-8", _NUM + "x", "x RT", tag="T12 N=1 MuJoCo Warp noslip 50", label="noslip 50",
                           exclude=_SCOPED, decides=False, alt=True, quantity="T12")),
             "higher", "verdict at the task's noslip 50 on the Alloy default route, scoped noslip (coordinator round 2, "
             "2026-11-01; SCR-WP3-4 item 4 had noslip off), WP3's newest T12 row deciding (its note lists rounds "
             "oldest first); ruling (coordinator 2026-11-02): SB-6 counts the determinism route off, because "
             "repeatability (RUN_TO_RUN) is an opt-in preset; RUN_TO_RUN at N=1 is shown beside as information, real "
             "time from WP3's latency pack and step cost vs NONE from WP5; WP8's noslip-off ladder row beside",
             default_route=True, beside="noslip off (WP8 ladder)", beside_feed="T12 N=1 noslip off"),
    Headline("SB-7", SB7.FEEDS, "lower", SB7.note()),
    Headline("SB-8", (_CPL4,), "lower", "coupling injection per sim-minute, " + _CPL4_NOTE),
    Headline("CPL-4", (_CPL4,), "lower", "coupling injection per sim-minute (the conservative scene's passive-gain "
             "bound is in its row's verdict), " + _CPL4_NOTE),
    Headline("SB-9", SB9.FEEDS, "higher", SB9.note()),
    Headline("SB-10", (Feed("SB-10", _USD, "USD per M sim-s", tag="T12 scoped noslip 50", label=_SCOPED,
                            quantity="T12", newest=r"\bT12\b", newest_last=True),   # as SB-5: WP3 lists oldest first
                       Feed("PAR-15", _USD, "USD per M sim-s", tag="T12 MuJoCo Warp noslip 50", label="noslip 50",
                            exclude=_SCOPED, alt=True, quantity="T12"),
                       Feed("PAR-15", _USD, "USD per M sim-s", tag="T12 noslip off", exclude=_NOSLIP_ON,
                            decides=False, quantity="T12")),
             "lower", _NOSLIP_NOTE, default_route=True, beside=_MJW_BESIDE),
    Headline("SB-11", (Feed("PRC-4", r"share of digests equal " + _NUM, "share", pass_value=1.0),
                       Feed("PRC-4", _NUM + r" ?%", "share", scale=0.01, every="min", note=_MJW_ROUTE,
                            newest=r"R\d(?: / R\d)*, " + _ROUTE_CLASS, newest_last=True),
                       Feed("PRC-4", r"ms \(" + _NUM + r"x", "x step time", every="max", info=_R2R_COST,
                            fail_above=1.3, note=_MJW_ROUTE_COST, newest=_ROUTE_CLASS, newest_last=True)), "higher",
             "ordered route only: on the default route a GPU MuJoCo scene does not repeat run to run (share 0 on "
             "T12, bench/xbench/OUR_TESTS.md; section 'GPU determinism' below), so this 1.0 is not the default's; "
             "a pass row states every compared digest equal (share 1); D9 GPU population: FakeSolver (PRC-4) and "
             "SolverMuJoCo on Alloy's ordered route (WP5 P3-1 round 5, native_share counts it since 60f64eb: T12, T11 "
             "and T14; per scene and world count WP5's newest row decides, its note lists rounds oldest first). The "
             "MuJoCo-routed scenes repeat under RUN_TO_RUN, an opt-in preset (the ordered route turns on where a "
             "HealthMonitor is built with a RUN_TO_RUN preset), at the step cost shown beside, per scene and world count (M7 "
             "allows 1.3x). Not in it (native_share 0): dense-Jacobian scenes "
             "(T02, H1, peg, nut-and-bolt) and T09 (CG)"),
)}


def _matches(feed: Feed, label: str, text: str = "") -> bool:
    if feed.label and not re.search(feed.label, label, re.IGNORECASE):
        return False
    if feed.exclude and re.search(feed.exclude, label, re.IGNORECASE):
        return False
    both = f"{label} {text}"
    if feed.marker and not re.search(feed.marker, both, re.IGNORECASE):
        return False
    return not (feed.not_marker and re.search(feed.not_marker, both, re.IGNORECASE))


def _number(feed: Feed, text: str, verdict: str) -> float | None:
    if verdict in NOT_MEASURED:
        return None
    found = [float(m.group(1)) for m in re.finditer(feed.value, text.replace(",", ""))]
    if found:
        x = {"": found[0], "max": max(found), "min": min(found)}[feed.every]
        return x if feed.scale == 1.0 else float(f"{x * feed.scale:.12g}")
    return feed.pass_value if verdict == "pass" else None


def _row_meta(path: str, r: LawResult, v: Any) -> dict[str, str]:
    """Label, value text, source, commit and command of one verdict (collected row, else the report itself)."""
    rows = r.monitors.get("rows") or {}
    meta = rows.get(v.name) if isinstance(rows, dict) else None
    if isinstance(meta, dict):
        return {k: str(meta.get(k, "")) for k in ("label", "value", "source", "commit", "command")}
    value = f"{v.error:g} {v.unit}" if math.isfinite(v.error) else ""
    return {"label": v.name, "value": value, "source": path, "commit": str(r.provenance.get("alloy", "")),
            "command": r.command}


def _reason_of(r: LawResult, v: Any) -> str:
    per = r.monitors.get("reasons") or {}
    text = per.get(v.name) if isinstance(per, dict) else None
    return str(text or r.monitors.get("not_run_reason") or r.monitors.get("error") or "")


def producer_rows(reports: Sequence[Loaded], metric: str) -> list[tuple[str, LawResult, Any, dict[str, str]]]:
    """Every verdict of `reports` that reports §5 id `metric` (name `<metric>` or `<metric>:...`)."""
    out = []
    for path, r in reports:
        for v in r.verdicts:
            if v.name == metric or v.name.startswith(metric + ":"):
                out.append((path, r, v, _row_meta(path, r, v)))
    return out


def matched_rows(h: Headline, reports: Sequence[Loaded]) -> list[ProducerRow]:
    """Producer rows matched by each feed of `h`, in feed order (a row counts once, for its first feed; info feeds may
    re-read a row and keep it only when their number is in its text)."""
    seen: set[tuple[str, str]] = set()
    out: list[ProducerRow] = []
    for feed in h.feeds:
        if feed.note is not None:
            out += _supersede(feed, note_producer_rows(feed))
            continue
        got = []
        for path, r, v, meta in producer_rows(reports, feed.metric):
            key = (path, v.name)
            if (key in seen and not feed.info) or not _matches(feed, meta["label"], meta["value"]):
                continue
            row = _producer_row(feed, path, r, v, meta)
            if feed.info and row.value is None:
                continue
            seen.add(key)
            got.append(row)
        out += _supersede(feed, got)
    return demote_superseded(out)


def _note_line(p: ProducerRow) -> tuple[str, int]:
    path, _, line = p.source.rpartition(":")
    return (path, int(line)) if line.isdigit() else (p.source, 0)


def _supersede(feed: Feed, rows: list[ProducerRow]) -> list[ProducerRow]:
    """The newest row of each class `feed.newest` names decides (the first in note order, the last with
    `feed.newest_last`); the class's other rows are shown beside, tagged superseded (an info feed drops them). Rows
    naming no class decide. A class is per results note: note order says nothing across producers."""
    if not feed.newest:
        return rows
    first: set[tuple[str, str]] = set()
    out = []
    for p in sorted(rows, key=_note_line, reverse=feed.newest_last):
        m = re.search(feed.newest, p.label, re.IGNORECASE)
        cls = (_note_line(p)[0], m.group(0).lower()) if m else None
        if cls is None or cls not in first:
            first.update([cls] if cls else [])
            out.append(p)
            continue
        if feed.info:
            continue
        out.append(dataclasses.replace(p, feed=dataclasses.replace(feed, decides=False, tag=f"{p.tag} (superseded)",
                                                                   aside=None)))
    return sorted(out, key=_note_line)


def note_producer_rows(feed: Feed) -> list[ProducerRow]:
    """Rows of `feed.note` read from the results note (info feeds keep a row only when it has a number)."""
    out = []
    for n in NR.read(feed.note) if feed.note is not None else ():
        verdict = n.verdict or "inconclusive"
        value = _number(feed, n.text, "pass" if feed.info else verdict)
        if feed.info and value is None:
            continue
        reason = f"{feed.info}; {n.text[:200]}" if feed.info else ""
        out.append(ProducerRow(feed, n.package, n.label, n.text, value, _info_verdict(feed, value) if feed.info
                               else verdict, n.source, n.commit, n.command, "", reason))
    return out


def _info_verdict(feed: Feed, value: float | None) -> str:
    over = value is not None and feed.fail_above is not None and value > feed.fail_above
    under = value is not None and feed.fail_below is not None and value < feed.fail_below
    return "fail" if over or under else "pass"


def _producer_row(feed: Feed, path: str, r: LawResult, v: Any, meta: dict[str, str]) -> ProducerRow:
    package = v.name.split(":")[1] if v.name.count(":") >= 2 else r.law
    value = _number(feed, meta["value"], "pass" if feed.info else v.verdict)
    verdict = _info_verdict(feed, value) if feed.info else v.verdict
    return ProducerRow(feed, package, meta["label"], meta["value"], value, verdict, meta["source"], meta["commit"],
                       meta["command"], str(r.provenance.get("date", "")), feed.info or _reason_of(r, v))


def worst(verdicts: Iterable[str]) -> str:
    return max(verdicts, key=lambda x: RANK[x], default="not_run")


def measured_verdict(verdicts: Sequence[str], missing: bool = False) -> str:
    """Worst measured verdict; `not_run` only when nothing was measured; an all-pass with gaps is inconclusive."""
    done = [v for v in verdicts if v not in NOT_MEASURED]
    if not done:
        return "not_run"
    got = worst(done)
    return "inconclusive" if got == "pass" and (missing or len(done) < len(verdicts)) else got


def _verdict(h: Headline, rows: Sequence[ProducerRow], stray: Sequence[str],
             feeds: Sequence[Feed]) -> tuple[str, list[str]]:
    """(verdict, notes on gaps) over the deciding feeds of this population."""
    deciding = [p for p in rows if p.feed.decides and not p.feed.info]
    measured = {(p.feed.metric, _quantity(p.feed)) for p in deciding}  # one producer id from two sources counts once
    gaps = [f.tag or f.metric for f in feeds
            if f.decides and not f.info and (f.metric, _quantity(f)) not in measured]
    verdict = measured_verdict([p.verdict for p in deciding], missing=bool(gaps))
    notes = [f"{g}: no measured producer row" for g in gaps]
    unmet = [f.required for f in feeds if f.required and not any(p.feed == f and p.measured for p in deciding)]
    if unmet and verdict in ("pass", "inconclusive"):
        verdict = "fail"
    notes += [f"{u}: not run yet, a counted class (the headline reads fail until it passes)" for u in unmet]
    stale = [p.tag for p in rows if p.feed.tag.endswith(f"({NO_CURRENT})")]
    verdict = "inconclusive" if stale and verdict == "pass" else verdict
    notes += [f"{t}: not in the value or the verdict, re-run pending" for t in stale]
    if verdict == "not_run" and stray:
        verdict = "inconclusive"
        notes.append("measured producer rows match no feed (update alloy.bench.headline): " + ", ".join(stray))
    elif verdict == "not_run" and any(p.measured for p in rows):
        verdict = "inconclusive"
        notes.append("measured only beside the verdict: " + ", ".join(p.tag for p in rows if p.measured))
    return verdict, notes


def _quantity(f: Feed) -> str:
    return f.quantity or f.metric


def _value(h: Headline, rows: Sequence[ProducerRow]) -> tuple[float | None, str]:
    """Worst number over the deciding rows that measure the headline's primary quantity (its first deciding feed's);
    rows reported beside the verdict never set it."""
    pick = max if h.better == "lower" else min
    primary = next((f for f in h.feeds if f.decides), h.feeds[0])
    if any(p.feed.tag.endswith(f"({NO_CURRENT})") for p in rows):       # no number over a population with a hole
        return None, primary.unit
    vals = [p.value for p in rows if p.feed.decides and not p.feed.info and _quantity(p.feed) == _quantity(primary)
            and p.value is not None]
    return (pick(vals) if vals else None), primary.unit


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4g}"


def _part(p: ProducerRow) -> str:
    """One clause of the headline reason."""
    who = f"{p.tag} [{p.package}]"
    if p.feed.caveat and p.measured:
        info = " (info)" if p.feed.info else ""
        return f"{who}{info} {_fmt(p.value)} {p.feed.unit} {p.verdict} (caveat: {p.feed.caveat})"
    if p.feed.info:
        return f"{who} (info) {_fmt(p.value)} {p.feed.unit} {p.verdict}"
    if not p.measured:
        return f"{who} not_run: {p.reason or p.text}"
    if (p.feed.decides and p.verdict != "pass") or p.value is None:
        return f"{who} {p.verdict}: {p.text[:160]}"
    aside = _aside(p)
    return f"{who} {_fmt(p.value)} {p.feed.unit} {p.verdict}" + (f" ({aside})" if aside else "")


def _aside(p: ProducerRow) -> str:
    return p.feed.aside(p.label, p.text) if p.feed.aside is not None and p.measured else ""


def _stray(h: Headline, reports: Sequence[Loaded], rows: Sequence[ProducerRow]) -> list[str]:
    """Measured producer rows of the headline's ids that no feed matched (the feed table is out of date)."""
    used = {(p.source, p.label) for p in rows}
    out = []
    for metric in dict.fromkeys(f.metric for f in h.feeds):
        for _, _, v, meta in producer_rows(reports, metric):
            if v.verdict not in NOT_MEASURED and (meta["source"], meta["label"]) not in used:
                out.append(meta["label"])
    return out


def _status(m: MetricDef) -> Literal["active", "deferred"]:
    return "deferred" if m.id in DEFERRED else "active"


def summarize(h: Headline, rows: Sequence[ProducerRow], stray: Sequence[str] = (),
              feeds: Sequence[Feed] | None = None) -> Summary:
    verdict, gaps = _verdict(h, rows, stray, h.feeds if feeds is None else feeds)
    value, unit = _value(h, rows)
    key = [p for p in rows if p.feed.decides and not p.feed.info and p.verdict == verdict] or list(rows)
    parts = [_part(p) for p in rows if p.verdict != "pass" or not p.feed.decides or p.feed.info or _aside(p)]
    return Summary(value, unit, verdict, "; ".join(x for x in [*gaps, *parts] if x), key[0] if key else None)


def on_default(p: ProducerRow) -> bool:
    if p.feed.alt:
        return False
    return p.feed.default is None or re.search(p.feed.default, p.label, re.IGNORECASE) is not None


def headline_rows(h: Headline, m: MetricDef, reports: Sequence[Loaded]) -> list[ScoreRow]:
    """The headline row, the "all configurations" row when the headline has a default route, then one sub-row per
    matched producer row."""
    rows = matched_rows(h, reports)
    stray = _stray(h, reports, rows) if not any(p.measured and p.feed.decides for p in rows) else []
    if not rows and not stray:
        reason = f"no producer row for {', '.join(dict.fromkeys(f.metric for f in h.feeds))} ({m.producer})"
        unit = next((f.unit for f in h.feeds if f.decides), h.feeds[0].unit)
        return [ScoreRow(m.id, None, unit, m.target, "not_run", (), "", "", "", _status(m), reason)]
    every = summarize(h, rows, stray)
    date = max((p.date for p in rows), default="")
    sources = tuple(dict.fromkeys(p.source for p in rows))
    if h.default_route:
        out = _default_route_rows(h, m, rows, stray, every, sources, date)
    else:
        out = [_score(m, every, "; ".join(x for x in [h.note, every.reason] if x), sources, date)]
    return out + _sub_rows(m, rows)


def _default_route_rows(h: Headline, m: MetricDef, rows: Sequence[ProducerRow], stray: Sequence[str],
                        every: Summary, sources: tuple[str, ...], date: str) -> list[ScoreRow]:
    """Headline on the default-route rows, then the row shown beside it ("all configurations", or one alt feed)."""
    default = [p for p in rows if on_default(p)]
    head = summarize(h, default, stray, [f for f in h.feeds if not f.alt])
    if head.verdict == "not_run" and any(p.measured for p in rows):
        off = ", ".join(dict.fromkeys(p.tag for p in rows if p.measured))
        head = dataclasses.replace(head, verdict="inconclusive",
                                   reason="; ".join(x for x in [f"measured only off the default route: {off}",
                                                                head.reason] if x))
    if h.beside_feed:
        alt = [f for f in h.feeds if f.tag == h.beside_feed]
        every = summarize(dataclasses.replace(h, feeds=tuple(alt)), [p for p in rows if p.feed in alt], (), alt)
        sources = tuple(dict.fromkeys(p.source for p in rows if p.feed in alt)) or sources
    beside = f"{h.beside}: {_fmt(every.value)} {every.unit} {every.verdict}"
    reason = "; ".join(x for x in [f"verdict on the Alloy default route ({DR.describe()})", h.note, beside,
                                   head.reason] if x)
    head_sources = tuple(dict.fromkeys(p.source for p in default)) or sources
    sub_id = h.beside if h.beside_feed else "all configurations"
    return [_score(m, head, reason, head_sources, date),
            _score(dataclasses.replace(m, id=f"{m.id}:{sub_id}"), every,
                   "; ".join(x for x in [h.note, every.reason] if x), sources, date)]


def _score(m: MetricDef, s: Summary, reason: str, sources: tuple[str, ...], date: str) -> ScoreRow:
    return ScoreRow(m.id, s.value, s.unit, m.target, s.verdict, sources,  # type: ignore[arg-type]
                    s.key.command if s.key else "", s.key.commit if s.key else "", date, _status(m), reason)


def _sub_rows(m: MetricDef, rows: Sequence[ProducerRow]) -> list[ScoreRow]:
    out: list[ScoreRow] = []
    seen: dict[str, int] = {}
    for p in rows:
        rid = f"{m.id}:{p.tag} ({p.package})"
        seen[rid] = seen.get(rid, 0) + 1
        rid = rid if seen[rid] == 1 else f"{rid} #{seen[rid]}"
        why = p.reason if p.feed.info else ("" if p.verdict == "pass" else (p.reason if not p.measured else p.text[:200]))
        why = "; ".join(x for x in [f"caveat: {p.feed.caveat}" if p.feed.caveat else "", _aside(p), why] if x)
        out.append(ScoreRow(rid, p.value, p.feed.unit, m.target, p.verdict, (p.source,),  # type: ignore[arg-type]
                            p.command, p.commit, p.date, _status(m), why))
    return out


__all__ = ["Feed", "Headline", "HEADLINES", "ProducerRow", "Summary", "headline_rows", "matched_rows",
           "measured_verdict", "on_default", "producer_rows", "summarize", "worst"]
