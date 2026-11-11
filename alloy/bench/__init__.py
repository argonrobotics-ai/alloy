# SPDX-License-Identifier: Apache-2.0
"""Validation bench (WP6, SPEC §3.10, D1): laws, twins, comparators, reports, baseline and scoreboard.

`python -m alloy.bench run --suite {analytic,determinism,rig} [--law ID] [--solver NAME] [--device cpu|cuda:0]
[--quick] [--baseline PATH]` runs registered laws (A16 self-test first; producers that have not merged show
`not_run`) and writes report/1 files; `python -m alloy.bench scoreboard` builds bench/SCOREBOARD.json.
Producers register laws with `alloy.bench.law.register_law` from their own `laws.py`; twins follow
`alloy.bench.types.SimTwin` plus the optional hooks listed in `alloy.bench.runner`. Types live in
`alloy.bench.types` (WP0); `alloy` never imports the repo-root `bench/` harness.
"""

from alloy.bench.baseline import compare_to_baseline
from alloy.bench.compare import compare_curve, compare_distribution, compare_rate, compare_scalar
from alloy.bench.law import register_law, registered_laws
from alloy.bench.report import write_report
from alloy.bench.rigs import load_rig
from alloy.bench.runner import run_law

# `scoreboard(roots)` stays in its module (alloy.bench.scoreboard.scoreboard): exporting it here would shadow the
# submodule of the same name.
__all__ = ["compare_scalar", "compare_curve", "compare_distribution", "compare_rate", "run_law", "write_report",
           "load_rig", "register_law", "registered_laws", "compare_to_baseline"]
