# SPDX-License-Identifier: Apache-2.0
"""PRC-7 (SPEC §5.2): determinism overhead vs NONE, 1,024 and 4,096 worlds, graph mode where the mode allows it
(Warp's deterministic lowering does not fit a captured period for XPBD, research §3.3, so XPBD det is eager), time
per control period (median of 5 x 3 periods after warm-up) and mempool high-water memory, each config in a fresh
process. Target <= 1.3x time, <= 1.5x memory; timing counts only from an exclusive run (own VM, PRF-08)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from alloy.testing.precision import write_row

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
CMD = "scripts/gpu_run.sh WP5 'tests/precision/test_det_overhead_gpu.py' --vm <gpu-vm> --exclusive"

CHILD = r"""
import json, sys, time
import numpy as np, warp as wp
from alloy.testing.precision import det_sim
kind, n, det, graph = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1"
sim = det_sim(kind, n, det=det, use_graph=graph)
for _ in range(2):
    sim.step_period()
wp.synchronize()
times = []
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(3):
        sim.step_period()
    wp.synchronize()
    times.append((time.perf_counter() - t0) / 3)
mem = wp.get_mempool_used_mem_high("cuda:0")
print(json.dumps({"median_s": float(np.median(times)), "min_s": float(min(times)), "max_s": float(max(times)),
                  "mem_high": int(mem)}))
"""


def _measure(kind: str, n: int, det: bool, graph: bool) -> dict:
    out = subprocess.run([sys.executable, "-c", CHILD, kind, str(n), str(int(det)), str(int(graph))],
                         capture_output=True, text=True, timeout=3000)
    if out.returncode != 0:
        return {"error": out.stderr[-400:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("kind, graph_det", [("steel", True), ("xpbd", False), ("fake", True)])
@pytest.mark.parametrize("n", [1024, 4096])
def test_prc7_overhead(kind: str, graph_det: bool, n: int) -> None:
    ng = _measure(kind, n, det=False, graph=True)
    det = _measure(kind, n, det=True, graph=graph_det)
    if "error" in ng or "error" in det:
        write_row("PRC-7", f"{kind}@{n}: error {ng.get('error') or det.get('error')}", "fail", CMD)
        pytest.fail(str(ng.get("error") or det.get("error")))
    t_ratio = det["median_s"] / ng["median_s"]
    m_ratio = det["mem_high"] / max(ng["mem_high"], 1)
    ok = t_ratio <= 1.3 and m_ratio <= 1.5
    write_row("PRC-7", f"{kind}@{n}: time x{t_ratio:.2f} ({det['median_s'] * 1e3:.2f} vs {ng['median_s'] * 1e3:.2f} "
              f"ms/period; det {'graph' if graph_det else 'eager'}), memory x{m_ratio:.2f} "
              f"({det['mem_high'] / 2**20:.0f} vs {ng['mem_high'] / 2**20:.0f} MiB)", "pass" if ok else "fail", CMD,
              solver=kind, worlds=n, det=det, ng=ng)
