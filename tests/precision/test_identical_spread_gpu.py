# SPDX-License-Identifier: Apache-2.0
"""PRC-14 (SPEC §5.2): for downgraded (NOT_GUARANTEED) paths, the outcome spread across 1,024 identical worlds
(final pose of every free body) against the spread a seeded 1-ulp perturbation produces in deterministic mode (the
reference the SPEC takes from a CPU run; here the same deterministic solver on the GPU, so only the nudges differ).
Reported: median/p90 spread, KS distance and p-value. Target: spread <= the nudge spread (or indistinguishable)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from alloy.testing.precision import det_sim, write_row

pytestmark = [pytest.mark.gpu, pytest.mark.det, pytest.mark.slow]
CMD = "scripts/gpu_run.sh WP5 'tests/precision/test_identical_spread_gpu.py' --vm <gpu-vm>"
PERIODS = 30


def _final_positions(sim, periods: int) -> np.ndarray:
    for _ in range(periods):
        sim.step_period()
    q = sim.state.body_q.numpy()[:, :3].astype(np.float64)
    world = sim.model.body_world.numpy()
    W = int(sim.model.world_count)
    return np.stack([q[world == w] for w in range(W)])  # [W, B, 3]


def _spread(pos: np.ndarray) -> np.ndarray:
    """Per world: max over bodies of the distance to the cross-world median pose [m]."""
    med = np.median(pos, axis=0)
    return np.linalg.norm(pos - med, axis=2).max(axis=1)


@pytest.mark.parametrize("kind, worlds, ref_worlds, boxes", [
    ("xpbd", 1024, 1024, 27), ("steel", 1024, 1024, 27), ("mujoco", 1024, 64, 8),
])
def test_prc14_spread(kind: str, worlds: int, ref_worlds: int, boxes: int) -> None:
    ng = _spread(_final_positions(det_sim(kind, worlds, det=False, boxes=boxes), PERIODS))
    ref = _spread(_final_positions(det_sim(kind, ref_worlds, det=True, boxes=boxes, nudge_worlds=True), PERIODS))
    ks = stats.ks_2samp(ng, ref)
    med_ng, med_ref = float(np.median(ng)), float(np.median(ref))
    ok = med_ng <= med_ref or ks.pvalue >= 0.05
    value = (f"{kind}: NONE spread median {med_ng * 1e3:.4g} mm / p90 {np.quantile(ng, 0.9) * 1e3:.4g} mm "
             f"({worlds} worlds) vs 1-ulp nudge median {med_ref * 1e3:.4g} mm / p90 "
             f"{np.quantile(ref, 0.9) * 1e3:.4g} mm ({ref_worlds} worlds); KS D {ks.statistic:.3f} p {ks.pvalue:.3g}")
    write_row("PRC-14", value, "pass" if ok else "fail", CMD, solver=kind, periods=PERIODS, boxes=boxes,
              ratio=med_ng / med_ref if med_ref > 0 else None)
