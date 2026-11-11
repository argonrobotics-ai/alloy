# SPDX-License-Identifier: Apache-2.0
"""GPU determinism, as the board and the task-regression suite state it (coordinator ruling 2026-11-06 06:23Z,
scr/WP4.md "MuJoCo Warp GPU determinism: the claim is accepted as written").

The claim, verbatim: GPU SolverMuJoCo repeats bit for bit only on Alloy's ordered route, when asked for, on its covered
path. Every other GPU MuJoCo run is NONE: it can differ from the first step. CPU repeats. Wherever a GPU result is
quoted from one run, it is labelled a sample.

So every GPU result the board quotes from a single run (a law report on CUDA, a full-mode GPU row, a CARD-1 row from an
L4) carries `SAMPLE`; CPU results carry nothing. The task-regression suite marks each row the same way (`repeats`), and
its GPU tasks are samples until they request the route (WP5's fix, GO in the same ruling). The D1-D3 laws measuring
determinism are alloy.bench.determinism.
"""

from __future__ import annotations

RULING = "coordinator 2026-11-06 06:23Z"
CLAIM = ("GPU SolverMuJoCo repeats bit for bit only on Alloy's ordered route, when asked for, on its covered path. "
         "Every other GPU MuJoCo run is NONE: it can differ from the first step. CPU repeats. Wherever a GPU result "
         "is quoted from one run, it is labelled a sample.")
SAMPLE = f"GPU, one run: a sample ({RULING})"


def is_gpu(device: str) -> bool:
    return str(device).startswith("cuda")


def sample_label(device: str) -> str:
    """`SAMPLE` for a result measured on a GPU, "" for one measured on the CPU (which repeats)."""
    return SAMPLE if is_gpu(device) else ""


def repeats(device: str, route_requested: bool = False) -> str:
    """How a result from `device` repeats: CPU bit for bit; GPU only on the ordered route when it was requested."""
    if not is_gpu(device):
        return "bitwise (CPU)"
    return "bitwise (GPU, ordered route requested)" if route_requested else "sample (GPU run is NONE)"


def board_md() -> list[str]:
    """The board's GPU-determinism section (SCOREBOARD.md)."""
    return ["", "## GPU determinism", "", f"- {CLAIM} ({RULING}, scr/WP4.md; WP6 board text, WP0 docs)"]


__all__ = ["CLAIM", "RULING", "SAMPLE", "board_md", "is_gpu", "repeats", "sample_label"]
