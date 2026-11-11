# SPDX-License-Identifier: Apache-2.0
"""P3-1 (BOARD, SPEC §8 open question 13): does MuJoCo Warp's own DeterminismType make SolverMuJoCo repeatable?

T12 (bimanual block stacking, WP8's TaskRung with WP8-sized capacities) on one L4, modes NONE, ATOMICS, sorts
(CONTACTS|CONSTRAINT) and ALL. At mujoco_warp@86e2fae0 the CONTACTS and CONSTRAINT bits call stubs; route (c) fills
them with Alloy's sorts (``alloy.precision.mjw_order``), which ``set_mjw_determinism`` installs. Checks:
- R1: two runs in one process.
- R2: a second process.
- R4: world 0 run alone vs world 0 of the 1,024-world batch.
- Cost against NONE: M7 allows <= 1.3x step time and <= 1.5x peak memory.
Every run is a fresh ``alloy.tools.mjw_det_probe`` process, so each mode gets its own memory peak. Acceptance
needs R1 and R2 bitwise at >= 1,024 worlds plus R4. The test records the verdicts as rows; it fails only when a
measurement could not be taken.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alloy.testing.precision import compare_digests, probe_run, repeat_verdict, write_row

pytestmark = [pytest.mark.gpu, pytest.mark.nightly, pytest.mark.slow, pytest.mark.timeout(1200)]
ROOT = Path(__file__).resolve().parents[2]
CMD = "scripts/gpu_run.sh WP5 'tests/precision/test_mjw_determinism_gpu.py -m nightly -k {k}' --vm <gpu-vm>"
WORLDS = 1024
CALLS = int(os.environ.get("ALLOY_P31_CALLS", "60"))  # one call = 2 steps of 0.833 ms on T12
MODES = {"none": 0, "atomics": 4, "sorts": 3, "all": 7}  # sorts / all: CONTACTS and CONSTRAINT filled by Alloy
M7_TIME, M7_MEM = 1.3, 1.5


_probe, _compare, _verdict = probe_run, compare_digests, repeat_verdict


@pytest.mark.parametrize("mode", list(MODES))
def test_p31_repeat(mode: str, tmp_path: Path) -> None:
    bits = str(MODES[mode])
    a, meta, _ = _probe(tmp_path, "a", "--bits", bits, "--worlds", str(WORLDS), "--calls", str(CALLS), "--repeat", "2")
    b, _, _ = _probe(tmp_path, "b", "--bits", bits, "--worlds", str(WORLDS), "--calls", str(CALLS))
    one, _, _ = _probe(tmp_path, "one", "--bits", bits, "--worlds", "1", "--calls", str(CALLS))
    cmd = CMD.format(k=f"repeat and {mode}")
    assert a.shape[:2] == (2, CALLS), meta.get("error")
    tag = f"T12 {WORLDS} worlds, {CALLS} calls, mujoco_warp bits {bits} ({mode}), njmax {meta['njmax']}"
    for case, (x, y) in {"R1": (a[0], a[1]), "R2": (a[0], b[0])}.items():
        share, first, overall = _compare(x, y)
        write_row(f"P3-1 {case}", f"{tag}: {share:.1%} of worlds equal after the last call, first differing call "
                  f"{first}, {overall:.1%} of world-calls equal", _verdict(share, first), cmd, mode=mode)
    share, first, overall = _compare(one[0][:, :1], a[0][:, :1])
    write_row("P3-1 R4", f"{tag}: world 0 alone vs world 0 of {WORLDS}: equal on {overall:.1%} of calls, first "
              f"differing call {first}", _verdict(share, first), cmd, mode=mode)
    distinct = len({tuple(x) for x in a[0][-1]})
    write_row("P3-1 worlds", f"{tag}: {distinct} distinct world digests after the last call (identical replicas)",
              "pass" if distinct == 1 else "fail", cmd, mode=mode)


def test_p31_cost(tmp_path: Path) -> None:
    """Eager step time, graph step time and peak memory per mode at 1,024 worlds, against NONE (M7)."""
    runs = {m: _probe(tmp_path, m, "--bits", str(b), "--worlds", str(WORLDS), "--calls", "12", "--graph")[1:]
            for m, b in MODES.items()}
    base, base_t = runs["none"]
    assert base["error"] is None and base["graph_ms"] is not None, base
    base_eager = float(np.median(base_t[0][2:]))
    for mode, (meta, t) in runs.items():
        if mode == "none" or meta["error"] is not None:
            continue
        eager = float(np.median(t[0][2:]))
        graph = meta["graph_ms"]
        t_ratio = (graph / base["graph_ms"]) if graph is not None else eager * 1e3 / base["graph_ms"]
        m_ratio = meta["mem_high_bytes"] / base["mem_high_bytes"]
        graph_txt = f"{graph:.2f} ms" if graph is not None else f"not capturable ({str(meta['graph_error'])[:90]})"
        value = (f"T12 {WORLDS} worlds ({mode}): step {eager * 1e3:.1f} ms eager vs NONE {base_eager * 1e3:.1f} ms "
                 f"eager ({eager / base_eager:.1f}x); graph {graph_txt} vs NONE graph {base['graph_ms']:.2f} ms, so "
                 f"{t_ratio:.1f}x the production step; peak memory {meta['mem_high_bytes'] / 2**30:.2f} GiB vs "
                 f"{base['mem_high_bytes'] / 2**30:.2f} GiB ({m_ratio:.1f}x)")
        ok = t_ratio <= M7_TIME and m_ratio <= M7_MEM
        write_row("P3-1 cost", value, "pass" if ok else "fail", CMD.format(k="cost"), mode=mode)


def test_p31_capacity(tmp_path: Path) -> None:
    """ATOMICS at the router's capacities (njmax 14,160/world): 1,024 worlds and the 16-world R1."""
    _, big, _ = _probe(tmp_path, "router1024", "--bits", "4", "--worlds", str(WORLDS), "--calls", "2",
                       "--caps", "router")
    write_row("P3-1 capacity", f"T12 {WORLDS} worlds, ATOMICS, router capacities (njmax {big['njmax'] or 'n/a'}): "
              f"{big['error'] or 'ran'}", "fail" if big["error"] else "pass", CMD.format(k="capacity"), mode="atomics")
    small, meta, _ = _probe(tmp_path, "router16", "--bits", "4", "--worlds", "16", "--calls", "30", "--caps", "router",
                            "--repeat", "2")
    assert small.shape[0] == 2, meta.get("error")
    share, first, overall = _compare(small[0], small[1])
    write_row("P3-1 R1", f"T12 16 worlds, 30 calls, ATOMICS, router capacities: {share:.1%} of worlds equal after the "
              f"last call, first differing call {first}", _verdict(share, first), CMD.format(k="capacity"),
              mode="atomics")
