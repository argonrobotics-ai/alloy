# SPDX-License-Identifier: Apache-2.0
"""GPU determinism text (coordinator ruling 2026-11-06 06:23Z): the claim verbatim on the board, every one-run GPU
result labelled a sample, CPU results unlabelled, and the task-regression rows saying how they repeat."""

from __future__ import annotations

from alloy.bench import gpu_repeat as GR
from alloy.bench import scoreboard as SB
from alloy.bench.types import LawResult, MetricVerdict, Quantity


def _law(device: str) -> LawResult:
    """A one-metric law result, built as tests/bench/test_scoreboard.py builds its own."""
    q = Quantity("peak_pen_m", "m", 0.002, 0.0, 1, 0.0)
    mv = MetricVerdict("peak_pen_m", "m", None, q, 0.002, 0.001, None, "mean", "fail")  # type: ignore[arg-type]
    prov = {"suite": "analytic", "quick": "True", "alloy": "abc", "date": "2026-11-06"}
    return LawResult("A8.push", "mujoco", device, (mv,), {}, prov, 0.1, False, command="cmd")


def test_the_claim_is_verbatim_and_on_the_board():
    assert GR.CLAIM == ("GPU SolverMuJoCo repeats bit for bit only on Alloy's ordered route, when asked for, on its "
                        "covered path. Every other GPU MuJoCo run is NONE: it can differ from the first step. CPU "
                        "repeats. Wherever a GPU result is quoted from one run, it is labelled a sample.")
    md = SB.render_markdown([])
    assert "## GPU determinism" in md and GR.CLAIM in md


def test_a_gpu_law_result_is_a_sample_and_a_cpu_one_is_not():
    gpu, cpu = SB.law_rows([("a.json", _law("cuda:0")), ("b.json", _law("cpu"))])
    assert GR.SAMPLE in gpu.reason and GR.SAMPLE not in cpu.reason
    assert (GR.repeats("cpu"), GR.repeats("cuda:0")) == ("bitwise (CPU)", "sample (GPU run is NONE)")
    assert GR.repeats("cuda:0", route_requested=True) == "bitwise (GPU, ordered route requested)"
