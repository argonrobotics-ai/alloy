# SPDX-License-Identifier: Apache-2.0
"""`python -m alloy.bench.scorecard <row>`: run one scorecard row, or `page` to build the one-page scorecard.

    python -m alloy.bench.scorecard list
    python -m alloy.bench.scorecard 3 --device cpu --out scorecard
    python -m alloy.bench.scorecard page --root scorecard --out scorecard/SCORECARD.md
    python -m alloy.bench.scorecard charts --root scorecard --out scorecard/charts    # SVGs; needs matplotlib

A row writes `rows.jsonl` and `headline.json` under `<out>/row<id>/`. Exit codes: 0 done, 2 the row is not wired
yet (its keeper is named), 1 an error.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from alloy.bench.scorecard import page, rows_io
from alloy.bench.scorecard.registry import ROWS
from alloy.bench.scorecard.types import RunOptions

log = logging.getLogger("alloy.bench.scorecard")
EXIT_NOT_WIRED = 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.bench.scorecard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("row", choices=[*ROWS, "list", "page", "charts"],
                   help="a row id, `list`, `page` or `charts`")
    p.add_argument("--out", type=Path, default=Path("scorecard"), help="output (row: <out>/row<id>/; page: file; charts: folder)")
    p.add_argument("--root", type=Path, default=Path("scorecard"), help="page, charts: folder holding the rows' outputs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true", help="a short draft run; never the launch number")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, default=None, help="result files a row reads instead of measuring")
    p.add_argument("--jobs", type=int, default=1, help="row 3: laws run in this many processes (resumes a stopped run)")
    p.add_argument("--internal", action="store_true", help="page: include rows that may not be published")
    return p


def cmd_list() -> int:
    for spec in ROWS.values():
        state = "ready" if spec.runner else "not wired"
        print(f"{spec.id:3s} {spec.title:36s} {state:10s} {spec.lead:20s} {spec.machine}")  # CLI output
    return 0


def cmd_page(args: argparse.Namespace) -> int:
    out = args.out if args.out.suffix == ".md" else args.out / "SCORECARD.md"
    print(page.write_page(args.root, out, internal=args.internal))
    return 0


def cmd_charts(args: argparse.Namespace) -> int:
    from alloy.bench.scorecard import charts  # noqa: PLC0415 - matplotlib is optional

    paths = charts.write_charts(args.root, args.out)
    for path in paths:
        print(path)
    if not paths:
        print(f"no row3 / row6 rows.jsonl under {args.root}: run `python -m alloy.bench.scorecard 3|6` first",
              file=sys.stderr)
    return 0 if paths else 1


def cmd_row(args: argparse.Namespace, command: str) -> int:
    spec = ROWS[args.row]
    if spec.runner is None:
        print(f"row {spec.id} ({spec.title}) is not wired into this command yet; keeper: {spec.lead}. {spec.note}",
              file=sys.stderr)
        return EXIT_NOT_WIRED
    out = page.row_dir(args.out, spec.id)
    rows, head = spec.runner(RunOptions(out=out, device=args.device, quick=args.quick, seed=args.seed,
                                        command=command, data=args.data, jobs=args.jobs))
    for path in rows_io.write_rows(rows, out):
        log.info("wrote %s", path)
    rows_io.write_headline(head, out)
    print(f"row {head.row}: {head.number}. {head.bound}. {head.caveat}".rstrip(". ") + ".")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stderr)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    if args.row == "list":
        return cmd_list()
    if args.row == "page":
        return cmd_page(args)
    if args.row == "charts":
        return cmd_charts(args)
    return cmd_row(args, "python -m alloy.bench.scorecard " + " ".join(shlex.quote(a) for a in argv))


__all__ = ["EXIT_NOT_WIRED", "main"]
