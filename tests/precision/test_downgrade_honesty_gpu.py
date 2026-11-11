# SPDX-License-Identifier: Apache-2.0
"""PRC-13 (SPEC §5.2): effective_determinism agrees with measured R1 (same process) and R2 (fresh process) for every
solver x preset; silent downgrades (RUN_TO_RUN claimed, digests differ) must be 0. Solver kwargs per preset follow
apply_precision: fast -> no deterministic mode; standard -> XPBD/Steel deterministic + sorted contacts, SolverMuJoCo
built NOT_GUARANTEED with sorted contacts and Alloy's ordered route enabled through the preset (P3-1 round 5;
``enable_solver_determinism``); 'requested' runs Warp's lowering on SolverMuJoCo at 16 worlds."""

from __future__ import annotations

import functools

import pytest

from alloy.precision.config import preset
from alloy.precision.determinism import effective_determinism
from alloy.precision.repeat import repeat_test
from alloy.precision.types import Determinism, GpuTier
from alloy.testing.precision import det_sim, write_row

pytestmark = [pytest.mark.gpu, pytest.mark.det, pytest.mark.slow]
CMD = "scripts/gpu_run.sh WP5 'tests/precision/test_downgrade_honesty_gpu.py' --vm <gpu-vm>"

# (solver, preset, det switch used by that preset, worlds, boxes)
CASES = [
    ("xpbd", "fast", False, 1024, 27), ("xpbd", "standard", True, 1024, 27),
    ("steel", "fast", False, 1024, 27), ("steel", "standard", True, 1024, 27),
    ("featherstone", "fast", False, 1024, 27), ("featherstone", "standard", True, 1024, 27),
    ("mujoco", "fast", False, 1024, 8), ("mujoco", "standard", False, 1024, 8),
    ("mujoco", "requested", True, 16, 8),
]
_SUMMARY: list[dict] = []


@pytest.mark.parametrize("kind, preset_name, det, worlds, boxes", CASES)
def test_prc13_agreement(kind: str, preset_name: str, det: bool, worlds: int, boxes: int) -> None:
    cfg = preset("fast" if preset_name == "fast" else "standard", GpuTier.B)
    sorted_contacts = det or preset_name == "standard"  # the standard preset sorts the bulk pipeline
    make = functools.partial(det_sim, kind, det=det, boxes=boxes, sorted_contacts=sorted_contacts, precision=cfg)
    sim = make(worlds)
    claim, reasons = effective_determinism(cfg, sim.solvers, worlds)
    periods = 4 if kind == "mujoco" and det else 16
    r1 = repeat_test(make, "R1", steps=periods, world_counts=(worlds,))
    r2 = repeat_test(make, "R2", steps=periods, world_counts=(worlds,))
    measured = r1.equal[worlds] and r2.equal[worlds]
    silent = claim >= Determinism.RUN_TO_RUN and not measured
    agree = (claim >= Determinism.RUN_TO_RUN) == measured
    row = {"solver": kind, "preset": preset_name, "worlds": worlds, "claim": claim.name, "r1": r1.equal[worlds],
           "r2": r2.equal[worlds], "agree": agree, "silent_downgrade": silent, "reasons": reasons}
    _SUMMARY.append(row)
    write_row("PRC-13 case", f"{kind}/{preset_name}@{worlds}: claim {claim.name}, R1 {r1.equal[worlds]}, "
              f"R2 {r2.equal[worlds]}", "fail" if silent else ("pass" if agree else "conservative"), CMD, **row)
    assert not silent, row


def test_prc13_summary() -> None:
    if len(_SUMMARY) < len(CASES):
        pytest.skip("run with the case tests in the same session")
    agree = sum(r["agree"] for r in _SUMMARY)
    silent = sum(r["silent_downgrade"] for r in _SUMMARY)
    write_row("PRC-13", f"agreement {agree}/{len(_SUMMARY)}; silent downgrades {silent}",
              "pass" if silent == 0 and agree == len(_SUMMARY) else ("fail" if silent else "partial"), CMD,
              cases=_SUMMARY)
    assert silent == 0
