# SPDX-License-Identifier: Apache-2.0
"""Task-regression suite (coordinator ruling 2026-11-06 05:55Z, scr/XB4-rulings.md "engine focus").

Finished tasks (T12, DC2/DC3/..., JOB1-3, the SC scenes that pass) are kept as checks of the engine: each has its
last recorded result, a stated tolerance and a small fixed seed set (`registry.TASKS`, `last_results.json`). The CPU
part runs in the post-merge full run and gates last-green; the GPU part runs nightly. CLI: `python -m
alloy.bench.task_regression` (list / run / rebaseline).
"""
