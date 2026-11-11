# SPDX-License-Identifier: Apache-2.0
"""Run one task-regression check in this process and write its metrics as JSON (the suite runs each in a subprocess).

    python -m alloy.bench.task_regression.one --task ID --device cpu --json OUT
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

from alloy.bench.task_regression.registry import get

log = logging.getLogger("alloy.bench.task_regression")


def resolve(runner: str) -> Callable[[Sequence[int], str], dict[str, float]]:
    """`module:function` -> the callable."""
    module, _, name = runner.partition(":")
    return getattr(importlib.import_module(module), name)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.bench.task_regression.one")
    ap.add_argument("--task", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    task = get(args.task)
    try:
        metrics = resolve(task.runner)(task.seeds, args.device)
        payload = {"task": task.id, "metrics": {k: float(v) for k, v in metrics.items()}}
        code = 0
    except Exception as exc:                                           # reported, never a pass
        log.error("task %s failed to run: %s", task.id, exc)
        payload = {"task": task.id, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-4000:]}
        code = 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
