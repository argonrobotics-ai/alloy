# SPDX-License-Identifier: Apache-2.0
"""Row 4 report: the scored cells as a Markdown table (public, or with the cells whose runs set internal_only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.scorecard.train.score import Cell, load_thresholds, score_files

TASKS = ("go2_flat", "franka_lift", "cube_reorient")


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "not reached"
    if seconds == float("inf"):
        return "not reached"
    return f"{seconds / 60:.1f} min" if seconds >= 90 else f"{seconds:.0f} s"


def _fmt_steps(steps: float | None) -> str:
    if steps is None or steps == float("inf"):
        return "-"
    return f"{steps / 1e6:.1f} M"


def cell_text(cell: Cell | None) -> str:
    """One table cell: median wall time (min-max), env steps, seeds reached."""
    if cell is None:
        return "not run"
    spread = ""
    if cell.wall_s_min is not None and cell.wall_s_max is not None:
        spread = f" ({_fmt_time(cell.wall_s_min)}-{_fmt_time(cell.wall_s_max)})"
    rate = f", {cell.steps_per_s_median / 1e3:.0f}k steps/s" if cell.steps_per_s_median else ""
    return (
        f"{_fmt_time(cell.wall_s_median)}{spread}, {_fmt_steps(cell.env_steps_median)} steps, "
        f"{cell.seeds_reached}/{cell.seeds} seeds{rate}"
    )


def table(cells: list[Cell], variant: str) -> str:
    by_key = {(c.framework, c.task): c for c in cells if c.variant == variant}
    frameworks = sorted({f for f, _ in by_key})
    lines = ["| framework | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for fw in frameworks:
        row = " | ".join(cell_text(by_key.get((fw, task))) for task in TASKS)
        lines.append(f"| {fw} | {row} |")
    return "\n".join(lines)


def render(cells: list[Cell], public: bool, header: str = "") -> str:
    """``public`` drops the cells whose runs set internal_only; ``header`` (internal reports: the licence notice) goes first."""
    shown = [c for c in cells if not (public and c.internal_only)]
    parts = [header, ""] if header else []
    parts.append("# Row 4: time to train (median of seeds; wall-clock from the first training iteration)")
    for variant in sorted({c.variant for c in shown}):
        parts += ["", f"## {variant}", "", table(shown, variant)]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Markdown table of the time-to-train cells.")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--thresholds", type=Path, default=Path(__file__).with_name("thresholds.json"))
    ap.add_argument("--public", action="store_true", help="leave out cells whose runs set internal_only")
    ap.add_argument("--header-file", type=Path, help="text to put first (the licence notice of an internal report)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    paths = [p for r in args.runs for p in (sorted(r.glob("**/*.jsonl")) if r.is_dir() else [r])]
    header = args.header_file.read_text().strip() if args.header_file else ""
    text = render(score_files(paths, load_thresholds(args.thresholds)), args.public, header)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
