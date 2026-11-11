# SPDX-License-Identifier: Apache-2.0
"""Scorecard charts: pure functions from `Row` records to matplotlib figures, and `write_charts` to SVG files.

matplotlib is optional: it is imported inside the functions, and a missing install raises
`alloy.FeatureUnavailableError` naming the command. Rules every chart follows: log axes for ratios and error
sizes (no truncated linear bars), the device / mode caveat found in the rows is in the title, a loss is drawn the
same way as a win, state is never carried by colour alone (every mark has a text label or a distinct symbol), and
rows marked `internal=True` are not drawn unless the caller asks (the title then starts with "RESTRICTED").

    python -m alloy.bench.scorecard charts --root scorecard --out scorecard/charts
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from alloy.bench.scorecard.rows_io import PUBLIC_NAME, read_rows
from alloy.bench.scorecard.types import Row

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SELFTEST = "A16.selftest"
INK, INK2, GRID = "#111111", "#4d4d4d", "#dddddd"
# Okabe-Ito colour-blind-safe palette.
BLUE, VERMILLION, ORANGE, SKY = "#0072B2", "#D55E00", "#E69F00", "#56B4E9"
GREEN, YELLOW, PINK, GREY = "#009E73", "#F0E442", "#CC79A7", "#999999"
VERDICT_COLOR = {"pass": BLUE, "fail": VERMILLION, "inconclusive": ORANGE, "error": PINK}
VERDICT_RANK = {"pass": 0, "inconclusive": 1, "fail": 2, "error": 3}
ENGINE_COLORS = (BLUE, VERMILLION, GREEN, ORANGE, PINK, SKY, GREY)
FLOOR_DECADES = 3.0                   # a zero error is drawn this many decades below the smallest non-zero one


def _figure(width: float, height: float) -> Figure:
    try:
        from matplotlib.figure import Figure  # noqa: PLC0415 - optional dependency
    except ImportError as e:
        from alloy.api.errors import FeatureUnavailableError  # noqa: PLC0415

        raise FeatureUnavailableError("scorecard charts need matplotlib, which is not installed",
                                      "pip install matplotlib") from e
    return Figure(figsize=(width, height), dpi=150, facecolor="white")


def public(rows: Sequence[Row], internal: bool = False) -> list[Row]:
    """The rows a chart may draw: internal ones only when asked."""
    return [r for r in rows if internal or not r.internal]


def caveat_line(rows: Sequence[Row], extra: str = "") -> str:
    """The device and mode caveat found in the rows, for the chart's subtitle."""
    parts = []
    devices = sorted({r.device for r in rows if r.device})
    if devices:
        parts.append("device: " + ", ".join(devices))
    if any("quick mode" in r.caveat for r in rows):
        parts.append("quick mode: a draft, not the launch number")
    if extra:
        parts.append(extra)
    return "; ".join(parts)


def _heading(fig: Figure, title: str, subtitle: str, top: float) -> None:
    """Title and subtitle at the left, above the axes (`top` is the axes' top edge, figure fraction)."""
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.1 / h, title, fontsize=11, weight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.4 / h, subtitle, fontsize=8, color=INK2, va="top")
    fig.subplots_adjust(top=top)


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# --- row 3: error against tolerance per law -----------------------------------------------------------------------

@dataclass(frozen=True)
class LawBar:
    label: str                         # law id (plus engine when the law ran on several)
    metric: str                        # the worst metric
    ratio: float | None                # |error| / tolerance of that metric; None: no number
    verdict: str

    @property
    def text(self) -> str:
        if self.verdict == "not_run":
            return "not run"
        if self.ratio is None:
            return f"{self.verdict}, no ratio"
        return f"{self.verdict} {_times(self.ratio)}"


def _times(ratio: float) -> str:
    return "0x (exact)" if ratio == 0 else f"{ratio:.2g}x" if ratio < 100 else f"{ratio:.0f}x"


def _ratio(r: Row) -> float | None:
    if r.value is None or r.hi is None or r.hi <= 0 or not math.isfinite(r.value):
        return None
    ratio = abs(r.value) / r.hi
    return None if r.verdict == "pass" and ratio > 1 else ratio     # a signed metric the verdict reads differently


def _worst(group: Sequence[Row]) -> tuple[str, float | None, str]:
    """(metric, ratio, verdict) of the group's worst metric; a metric with a number outranks one without."""
    verdicts = {r.verdict for r in group}
    if verdicts == {"not_run"}:
        return group[0].metric, None, "not_run"
    ran = [r for r in group if r.verdict != "not_run"]
    verdict = max((r.verdict for r in ran), key=lambda v: VERDICT_RANK.get(v, 0))
    numbered = [(r, _ratio(r)) for r in ran if _ratio(r) is not None]
    if not numbered:
        return ran[0].metric, None, verdict
    r, ratio = max(numbered, key=lambda p: p[1])
    return r.metric, ratio, verdict


def law_bars(rows: Sequence[Row]) -> list[LawBar]:
    """One bar per law on an Alloy engine (`alloy/*`), in file order; a law on several engines gets one each."""
    kept = [r for r in rows if r.engine.startswith("alloy/") and r.task != SELFTEST]
    groups: dict[tuple[str, str], list[Row]] = {}
    for r in kept:
        groups.setdefault((r.task, r.engine), []).append(r)
    engines_per_law = Counter(task for task, _ in groups)
    bars = []
    for (task, engine), group in groups.items():
        metric, ratio, verdict = _worst(group)
        label = task if engines_per_law[task] == 1 else f"{task} [{engine.split('/', 1)[1]}]"
        bars.append(LawBar(label, metric, ratio, verdict))
    return bars


def _x_range(bars: Sequence[LawBar]) -> tuple[float, float, float]:
    """(left edge, right edge, floor) on the log axis: tolerance 1.0 and every drawn value stay inside."""
    positive = [b.ratio for b in bars if b.ratio]
    lowest = min(positive + [1.0])
    floor = lowest / 10 ** FLOOR_DECADES if any(b.ratio == 0 for b in bars) else lowest
    highest = max(positive + [1.0])
    return floor / 3, highest * 40, floor


def law_error_chart(rows: Sequence[Row], internal: bool = False) -> Figure:
    """Row 3: each law's worst metric, error / tolerance on a log axis; 1.0 is the tolerance, right of it fails."""
    rows = public(rows, internal)
    bars = law_bars(rows)
    if not bars:
        raise ValueError("row 3 chart: no rows on an alloy/* engine")
    fig = _figure(8.5, 1.8 + 0.24 * len(bars))
    ax = fig.add_subplot(111)
    left, right, floor = _x_range(bars)
    for i, b in enumerate(bars):
        _draw_law_bar(ax, i, b, left, floor)
    ax.axvline(1.0, color=INK, linewidth=1.2)
    ax.text(1.0, -0.9, "tolerance = 1", fontsize=7.5, color=INK, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_ylim(len(bars) - 0.4, -1.4)
    ax.set_yticks(range(len(bars)), [f"{b.label}  ({b.metric})" for b in bars], fontsize=7)
    ax.set_xlabel("worst metric: |error| / tolerance (log scale; bars start at the axis edge, read where they end)",
                  fontsize=8, color=INK2)
    _style_axes(ax)
    fails = sum(b.verdict in ("fail", "error") for b in bars)
    title = f"Physics against exact answers: {fails} of {len(bars)} law runs outside tolerance"
    _heading(fig, title, caveat_line(rows, "laws that did not run have no bar"), top=1 - 1.1 / fig.get_figheight())
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.55 / fig.get_figheight() + 0.02)
    return fig


def _draw_law_bar(ax, i: int, b: LawBar, left: float, floor: float) -> None:
    if b.ratio is None:
        ax.text(left * 1.15, i, b.text, fontsize=7, color=INK2, va="center", ha="left", style="italic")
        return
    shown = max(b.ratio, floor)
    color = VERDICT_COLOR.get(b.verdict, GREY)
    hatch = "////" if b.verdict in ("fail", "error") else None
    ax.barh(i, shown - left, left=left, height=0.62, color=color, hatch=hatch, edgecolor="white", linewidth=0.0)
    ax.text(shown * 1.25, i, b.text, fontsize=7, color=INK, va="center", ha="left",
            weight="bold" if b.verdict != "pass" else "normal")


# --- row 6: capability grid and Gym-Anything tiers ------------------------------------------------------------------

STATUS_MARK = {"yes": "✔ yes", "partial": "◐ part", "no": "✖ no", "unknown": "? unk"}
STATUS_FILL = {"yes": "#cfe6f4", "partial": "#fbe7b8", "no": "#f4cdb8", "unknown": "#e8e8e8"}
CAPABILITY_PREFIX = "capability/"
GA_PREFIX = "gym-anything/"
TIER_NAMES = {"A": "port, own checks judge all gates", "B": "own checks judge some gates", "C": "own-rule analogue",
              "D": "proof run", "E": "class judgment only (no port)"}
TIER_COLORS = {"A": BLUE, "B": SKY, "C": YELLOW, "D": PINK, "E": GREY}
TIER_TEXT = {"A": "white", "B": INK, "C": INK, "D": INK, "E": INK}


def capability_grid(rows: Sequence[Row]) -> tuple[list[str], list[str], dict[tuple[str, str], str]]:
    """(capabilities, engines with Alloy first, status per (capability, engine)) from the status records."""
    cells = [r for r in rows if r.task.startswith(CAPABILITY_PREFIX) and r.metric == "status"]
    caps = list(dict.fromkeys(r.task[len(CAPABILITY_PREFIX):] for r in cells))
    engines = sorted(dict.fromkeys(r.engine for r in cells), key=lambda e: e != "alloy")
    status = {(r.task[len(CAPABILITY_PREFIX):], r.engine): (r.verdict or "unknown") for r in cells}
    return caps, engines, status


def capability_chart(rows: Sequence[Row], internal: bool = False) -> Figure:
    """Row 6: capabilities down, engines across; every cell has a symbol and a word, not only a colour."""
    rows = public(rows, internal)
    caps, engines, status = capability_grid(rows)
    if not caps:
        raise ValueError("row 6 capability chart: no capability status rows")
    fig = _figure(1.5 + 0.95 * len(engines), 1.9 + 0.3 * len(caps))
    ax = fig.add_subplot(111)
    for j, eng in enumerate(engines):
        for i, cap in enumerate(caps):
            s = status.get((cap, eng), "unknown")
            ax.add_patch(_cell_patch(j, i, STATUS_FILL.get(s, STATUS_FILL["unknown"])))
            ax.text(j, i, STATUS_MARK.get(s, STATUS_MARK["unknown"]), ha="center", va="center", fontsize=7, color=INK)
    ax.set_xlim(-0.5, len(engines) - 0.5)
    ax.set_ylim(len(caps) - 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(engines)), engines, fontsize=7.5, rotation=30, ha="left")
    ax.set_yticks(range(len(caps)), caps, fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    note = "Alloy's yes needs a passing test; another engine's yes means the feature is in its public docs or code"
    _heading(fig, "Capability coverage: Alloy first, losses shown as plainly as wins",
             caveat_line(rows, note), top=1 - 1.45 / fig.get_figheight())
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.2 / fig.get_figheight())
    return fig


def _cell_patch(x: int, y: int, color: str):
    from matplotlib.patches import Rectangle  # noqa: PLC0415

    return Rectangle((x - 0.48, y - 0.46), 0.96, 0.92, facecolor=color, edgecolor="white", linewidth=1.0)


def tier_counts(rows: Sequence[Row]) -> dict[str, int]:
    """Gym-Anything tasks per evidence-tier letter (A to E), zero-filled."""
    counts = Counter(str(r.extra.get("evidence_tier") or "?")[:1]
                     for r in rows if r.task.startswith(GA_PREFIX) and r.metric == "verdict_class")
    return {t: counts.get(t, 0) for t in TIER_NAMES} | ({"?": counts["?"]} if counts.get("?") else {})


def ga_tier_chart(rows: Sequence[Row], internal: bool = False) -> Figure:
    """Row 6: one stacked bar of the Gym-Anything tasks by how much of GA's own checking the Alloy port does."""
    rows = public(rows, internal)
    counts = tier_counts(rows)
    total = sum(counts.values())
    if not total:
        raise ValueError("row 6 Gym-Anything chart: no gym-anything task rows")
    fig = _figure(8.5, 3.4)
    ax = fig.add_subplot(111)
    left = 0
    for i, (tier, n) in enumerate(counts.items()):
        ax.barh(0, n, left=left, height=0.5, color=TIER_COLORS.get(tier, GREY), edgecolor="white", linewidth=1.5,
                label=f"{tier}: {TIER_NAMES.get(tier, 'no tier')} ({n})")
        _segment_label(ax, left, n, total, tier, i)
        left += n
    ax.set_xlim(0, total)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xlabel(f"tasks (of {total}); linear axis from zero", fontsize=8, color=INK2)
    _style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=7.5, frameon=False)
    sub = "Alloy's re-implementation of GA's criteria on Alloy's port (scripted actor); not run through GA's verifier"
    _heading(fig, "Gym-Anything tasks by evidence tier: most are a class judgment, not a port", sub, top=0.78)
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.36)
    return fig


def _segment_label(ax, left: float, n: int, total: int, tier: str, index: int) -> None:
    if not n:
        return
    middle = left + n / 2
    if n / total >= 0.06:
        ax.text(middle, 0, str(n), ha="center", va="center", fontsize=10, weight="bold",
                color=TIER_TEXT.get(tier, INK))
        return
    height = 0.42 + 0.2 * (index % 2)
    ax.annotate(f"{tier}: {n}", (middle, 0.25), (middle + 0.03 * total * (tier == "A"), height), fontsize=8, ha="center", color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.7})


# --- speed rows: ratio to a baseline engine -----------------------------------------------------------------------

def _by_task(rows: Sequence[Row], metric: str) -> dict[str, dict[str, Row]]:
    out: dict[str, dict[str, Row]] = {}
    for r in rows:
        if r.metric == metric and r.value is not None and r.value > 0:
            out.setdefault(r.task, {})[r.engine] = r
    return out


def ratio_points(rows: Sequence[Row], metric: str, baseline_engine: str) -> dict[str, dict[str, tuple[float, float | None, float | None]]]:
    """Per task and engine: (value / baseline, lo / baseline, hi / baseline). Tasks without a baseline are left out."""
    out = {}
    for task, by_engine in _by_task(rows, metric).items():
        base = by_engine.get(baseline_engine)
        if base is None:
            continue
        out[task] = {e: (r.value / base.value, _rel(r.lo, base.value), _rel(r.hi, base.value))
                     for e, r in by_engine.items() if e != baseline_engine}
    return out


def _rel(bound: float | None, base: float) -> float | None:
    return bound / base if bound is not None and bound > 0 else None


def ratio_chart(rows: Sequence[Row], metric: str, baseline_engine: str, internal: bool = False,
                higher_is_better: bool = True) -> Figure:
    """Speed rows: each engine's `metric` / the baseline engine's, per task, log axis, the bound as an error bar.

    Internal rows are not drawn unless `internal=True`; the title then starts with "RESTRICTED"."""
    rows = public(rows, internal)
    points = ratio_points(rows, metric, baseline_engine)
    engines = list(dict.fromkeys(e for p in points.values() for e in p))
    if not engines:
        raise ValueError(f"ratio chart: no {metric!r} rows with a {baseline_engine!r} baseline to compare with")
    fig = _figure(8.5, 1.9 + 0.32 * len(points) * max(1, len(engines) // 2 + 1))
    ax = fig.add_subplot(111)
    for k, engine in enumerate(engines):
        _draw_engine_points(ax, points, engine, k, len(engines))
    ax.axvline(1.0, color=INK, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_ylim(len(points) - 0.4, -0.6)
    ax.set_yticks(range(len(points)), list(points), fontsize=7.5)
    direction = "higher is better" if higher_is_better else "lower is better"
    ax.set_xlabel(f"{metric} relative to {baseline_engine} = 1 (log scale; {direction}; bars are the row's bound)",
                  fontsize=8, color=INK2)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _style_axes(ax)
    prefix = "RESTRICTED: " if internal else ""
    _heading(fig, f"{prefix}{metric}: ratio to {baseline_engine}", caveat_line(rows), top=1 - 1.1 / fig.get_figheight())
    fig.subplots_adjust(left=0.24, right=0.78, bottom=0.6 / fig.get_figheight() + 0.02)
    return fig


def _draw_engine_points(ax, points, engine: str, k: int, n: int) -> None:
    xs, ys, lo, hi = [], [], [], []
    for i, by_engine in enumerate(points.values()):
        if engine not in by_engine:
            continue
        ratio, rlo, rhi = by_engine[engine]
        xs.append(ratio)
        ys.append(i + (k - (n - 1) / 2) * 0.22)
        lo.append(ratio - (rlo if rlo is not None else ratio))
        hi.append((rhi if rhi is not None else ratio) - ratio)
    color = ENGINE_COLORS[k % len(ENGINE_COLORS)]
    marker = "osD^v<>"[k % 7]
    ax.errorbar(xs, ys, xerr=[lo, hi], fmt=marker, color=color, markersize=5, linewidth=1.1, capsize=2, label=engine)
    for x, y in zip(xs, ys, strict=True):
        ax.text(x * 1.08, y, f"{x:.2g}x", fontsize=6.5, color=INK, va="center")


# --- files ----------------------------------------------------------------------------------------------------------

def write_charts(root: Path, out: Path) -> list[Path]:
    """Read `<root>/row<id>/rows.jsonl` for rows 3 and 6 where the file exists and write their SVG charts to `out`."""
    out = Path(out)
    jobs = (("3", (("row3_laws.svg", law_error_chart),)),
            ("6", (("row6_capabilities.svg", capability_chart), ("row6_gym_anything.svg", ga_tier_chart))))
    written = []
    for row, charts in jobs:
        path = Path(root) / f"row{row}" / PUBLIC_NAME
        if not path.is_file():
            continue
        rows = read_rows(path)
        out.mkdir(parents=True, exist_ok=True)
        for name, build in charts:
            try:
                fig = build(rows)
            except ValueError:
                continue                  # the file has no rows for this chart: write nothing for it
            fig.savefig(out / name, format="svg", bbox_inches="tight")
            written.append(out / name)
    return written


__all__ = ["LawBar", "capability_chart", "capability_grid", "caveat_line", "ga_tier_chart", "law_bars",
           "law_error_chart", "public", "ratio_chart", "ratio_points", "tier_counts", "write_charts"]
