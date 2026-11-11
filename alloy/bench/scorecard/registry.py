# SPDX-License-Identifier: Apache-2.0
"""The scorecard's rows: id, title, keeper, machine and runner. A row without a runner is not wired yet."""

from __future__ import annotations

from alloy.bench.scorecard import row3, row6
from alloy.bench.scorecard.types import RowSpec

GPU = "g4-standard-48 (one RTX PRO 6000 Blackwell), spot"
CPU = "n2-standard-16, spot"

ROWS: dict[str, RowSpec] = {s.id: s for s in (
    RowSpec("1", "Speed in Skild's protocol", "scorecard maintainers", GPU,
            note="the protocol's 20 tasks, physics only, 5 ms env step, each engine at its best batch; ratio to MuJoCo Warp"),
    RowSpec("1b", "Robot speed fix", "scorecard maintainers", GPU, note="reported inside row 1 once the fix is supported"),
    RowSpec("2", "Quality beside every speed number", "scorecard maintainers", GPU,
            note="glitch rate with its 95 % bound, contact held, penetration, on row 1's seeds and worlds"),
    RowSpec("3", row3.TITLE, "scorecard maintainers", CPU, runner=row3.run,
            note="the analytic laws against their closed-form answers"),
    RowSpec("4", "Time to train", "scorecard maintainers", GPU, note="PPO to a fixed reward, 5 seeds, three tasks"),
    RowSpec("5", "Camera throughput", "scorecard maintainers", GPU, note="images/s and GPU memory with physics on"),
    RowSpec("6", row6.TITLE, "scorecard maintainers", CPU, runner=row6.run, note="capability matrix and the 200 Gym-Anything tasks"),
    RowSpec("7", "New-environment test", "scorecard maintainers", CPU,
            note="cold agents build 20 or more environments from one sentence each"),
    RowSpec("8", "Real robot", "not started", note="not measured"),
)}


def row_ids() -> list[str]:
    return list(ROWS)


__all__ = ["CPU", "GPU", "ROWS", "row_ids"]
