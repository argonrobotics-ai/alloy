# SPDX-License-Identifier: Apache-2.0
"""P3-1 round 5 acceptance on the L4: Alloy's ordered SolverMuJoCo route on T12 (WP8 capacities).

- R1 (two runs in one process), R2 (a second process) and R4 (world 0 alone vs in the batch) at 1,024 worlds, with
  digests taken after every graph replay (``--graph-series``) and, separately, after eager calls.
- M7: captured step time and peak memory against NONE at 1,024 worlds (<= 1.3x, <= 1.5x).
- N = 1 step time with and without the route (SB-6 must not get worse).
- The same R1 / R2 / R4 with box-box pairs on MuJoCo Warp's primitive collider (``--boxbox``, SCR-WP3-9 / BUL-0011).
- T09 on the CG path (SCR-DC5-4 (ii)): R1 / R2 / R4, the route-off control and cost at 1,024 worlds.

Every run is a fresh ``alloy.tools.mjw_det_probe`` process. Rows go to bench/results/wp5_results.jsonl.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from alloy.testing.precision import compare_digests as _compare
from alloy.testing.precision import probe_run as _probe
from alloy.testing.precision import repeat_verdict as _verdict
from alloy.testing.precision import write_row

pytestmark = [pytest.mark.gpu, pytest.mark.nightly, pytest.mark.slow, pytest.mark.timeout(3000)]
CMD = "scripts/gpu_run.sh WP5 'tests/precision/test_mjw_route_gpu.py -m nightly -k {k}' --vm <gpu-vm>"
WORLDS = int(os.environ.get("ALLOY_P31_WORLDS", "1024"))
CALLS = int(os.environ.get("ALLOY_P31_CALLS", "120"))  # one call = 2 steps of 0.833 ms on T12
TASK = os.environ.get("ALLOY_P31_TASK", "t12")
M7_TIME, M7_MEM = 1.3, 1.5


def _route(*flags: str) -> tuple[str, ...]:
    return ("--route", "--bits", "0", "--task", TASK, *flags)


@pytest.mark.parametrize("series", ["graph", "eager"])
def test_route_repeat(series: str, tmp_path: Path) -> None:
    extra = ("--graph-series",) if series == "graph" else ()
    common = ("--calls", str(CALLS), *extra)
    a, meta, _ = _probe(tmp_path, "a", *_route("--worlds", str(WORLDS), "--repeat", "2", *common))
    assert meta["route_reason"] is None and meta["error"] is None, meta
    b, _, _ = _probe(tmp_path, "b", *_route("--worlds", str(WORLDS), *common))
    one, _, _ = _probe(tmp_path, "one", *_route("--worlds", "1", *common))
    tag = (f"{TASK} {WORLDS} worlds, {CALLS} calls ({series}), Alloy ordered route, njmax {meta['njmax']}, "
           f"replaced launches {sum(meta['substituted'].values())}, primitive box-box {meta['primitive_boxbox']} "
           f"({meta['boxbox_pairs']} pairs)")
    cmd = CMD.format(k=f"repeat and {series}")
    rows = {"R1": _compare(a[0], a[1]), "R2": _compare(a[0], b[0])}
    for case, (share, first, _) in rows.items():
        write_row(f"P3-1 {case}", f"{tag}: {share:.1%} of worlds equal after every call, first differing call {first}",
                  _verdict(share, first), cmd, mode="route", series=series, task=TASK)
    r4 = (a[0][:, :1] == one[0]).all(axis=-1)
    write_row("P3-1 R4", f"{tag}: world 0 alone vs in {WORLDS}: equal on {r4.mean():.1%} of calls",
              "pass" if r4.all() else "fail", cmd, mode="route", series=series, task=TASK)
    distinct = len({tuple(x) for x in a[0][-1]})
    write_row("P3-1 worlds", f"{tag}: {distinct} distinct world digests after the last call (identical replicas)",
              "measured", cmd, mode="route", series=series, task=TASK)
    assert all(_verdict(s, f) == "pass" for s, f, _ in rows.values()) and r4.all()


def _cost(tmp: Path, worlds: int) -> tuple[dict, dict]:
    base = _probe(tmp, f"none{worlds}", "--bits", "0", "--task", TASK, "--worlds", str(worlds), "--calls", "12",
                  "--graph")[1]
    routed = _probe(tmp, f"route{worlds}", *_route("--worlds", str(worlds), "--calls", "12", "--graph"))[1]
    assert base["graph_ms"] is not None and routed["graph_ms"] is not None, (base, routed)
    return base, routed


def test_route_cost(tmp_path: Path) -> None:
    base, routed = _cost(tmp_path, WORLDS)
    t_ratio = routed["graph_ms"] / base["graph_ms"]
    m_ratio = routed["mem_high_bytes"] / base["mem_high_bytes"]
    ok = t_ratio <= M7_TIME and m_ratio <= M7_MEM
    write_row("P3-1 cost", f"{TASK} {WORLDS} worlds, Alloy ordered route: graph step {routed['graph_ms']:.2f} vs NONE "
              f"{base['graph_ms']:.2f} ms ({t_ratio:.2f}x); peak memory {routed['mem_high_bytes'] / 2**30:.3f} vs "
              f"{base['mem_high_bytes'] / 2**30:.3f} GiB ({m_ratio:.2f}x); M7 allows {M7_TIME}x / {M7_MEM}x",
              "pass" if ok else "fail", CMD.format(k="cost"), mode="route", task=TASK)
    assert ok


def test_route_n1(tmp_path: Path) -> None:
    """N = 1 graph step with and without the route: 300 replays each, NONE and route alternated 3 times, medians
    (the beside-SB-6 number; GPU clocks move between processes)."""
    flags = ("--task", TASK, "--worlds", "1", "--calls", "12", "--graph", "--graph-launches", "300")
    none, routed = [], []
    for i in range(3):
        none.append(_probe(tmp_path, f"none{i}", "--bits", "0", *flags)[1]["graph_ms"])
        routed.append(_probe(tmp_path, f"route{i}", "--route", "--bits", "0", *flags)[1]["graph_ms"])
    base, mid = float(np.median(none)), float(np.median(routed))
    write_row("P3-1 N=1", f"{TASK} 1 world: graph step {mid:.3f} ms with the route vs {base:.3f} ms without "
              f"({mid / base:.2f}x; medians of 3 alternated runs x 300 replays; route {sorted(routed)}, NONE "
              f"{sorted(none)})", "measured", CMD.format(k="n1"), mode="route", task=TASK)
    assert np.isfinite(mid / base)


SCENE_WORLDS = int(os.environ.get("ALLOY_P31_SCENE_WORLDS", "64"))  # router capacities: T11 needs 3.5 GB at 64
SCENES = ("task_t11", "task_t14", "task_t02", "task_h1", "task_t09")


SCENE_CALLS = {"task_t09": 8}  # T09's 115-dof cable: 40 calls at router capacities pass the 20 min GPU cap
# Per-world capacities near WP8's rule: about 2.5x the peaks seen over 60 held-control calls at 64 worlds
# (T11: 66 rows, 7 contacts; T14: 268 rows, 43 contacts; router capacities are 39,264 / 16,224 rows)
SCENE_CAPS = {"task_t11": (256, 32), "task_t14": (512, 96)}


@pytest.mark.parametrize("scene", SCENES)
def test_route_scene(scene: str, tmp_path: Path) -> None:
    """R1 / R2 and cost on the other MuJoCo-routed task scenes; an uncovered path records its downgrade reason."""
    task = f"scene:{scene}"
    calls = SCENE_CALLS.get(scene, 40)
    common = ("--task", task, "--worlds", str(SCENE_WORLDS), "--calls", str(calls), "--graph-series")
    a, meta, _ = _probe(tmp_path, "a", "--route", "--bits", "0", "--repeat", "2", *common)
    cmd = CMD.format(k=f"scene and {scene}")
    tag = f"{scene} {SCENE_WORLDS} worlds, {calls} calls (graph), router capacities (njmax {meta['njmax']})"
    if meta["route_reason"] is not None:
        share, first, _ = _compare(a[0], a[1])
        write_row("P3-1 scene", f"{tag}: route refused ({meta['route_reason']}); SolverMuJoCo stays NONE; measured R1 "
                  f"{share:.1%} of worlds equal, first differing call {first}", "downgraded", cmd, scene=scene)
        return
    b, _, _ = _probe(tmp_path, "b", "--route", "--bits", "0", *common)
    rows = {"R1": _compare(a[0], a[1]), "R2": _compare(a[0], b[0])}
    for case, (share, first, _) in rows.items():
        write_row(f"P3-1 {case}", f"{tag}, route: {share:.1%} of worlds equal after every call, first differing call "
                  f"{first}", _verdict(share, first), cmd, mode="route", scene=scene)
    base = _probe(tmp_path, "none", "--bits", "0", "--task", task, "--worlds", str(SCENE_WORLDS), "--calls", "8",
                  "--graph")[1]
    routed = _probe(tmp_path, "cost", "--route", "--bits", "0", "--task", task, "--worlds", str(SCENE_WORLDS),
                    "--calls", "8", "--graph")[1]
    t_ratio = routed["graph_ms"] / base["graph_ms"]
    m_ratio = routed["mem_high_bytes"] / base["mem_high_bytes"]
    write_row("P3-1 cost", f"{tag}, route: graph step {routed['graph_ms']:.2f} vs NONE {base['graph_ms']:.2f} ms "
              f"({t_ratio:.2f}x); peak memory {m_ratio:.2f}x", "pass" if t_ratio <= M7_TIME and m_ratio <= M7_MEM
              else "fail", cmd, mode="route", scene=scene)
    assert all(_verdict(s, f) == "pass" for s, f, _ in rows.values())


@pytest.mark.parametrize("scene", sorted(SCENE_CAPS))
def test_route_scene_caps(scene: str, tmp_path: Path) -> None:
    """R1 / R2 and M7 cost at 1,024 worlds with capacities sized to the scene (``SCENE_CAPS``)."""
    njmax, nconmax = SCENE_CAPS[scene]
    base = ("--task", f"scene:{scene}", "--worlds", str(WORLDS), "--njmax", str(njmax), "--nconmax", str(nconmax))
    a, meta, _ = _probe(tmp_path, "a", "--route", "--bits", "0", "--repeat", "2", "--calls", "60", "--graph-series",
                        *base)
    assert meta["route_reason"] is None and meta["error"] is None, meta
    b, _, _ = _probe(tmp_path, "b", "--route", "--bits", "0", "--calls", "60", "--graph-series", *base)
    cmd = CMD.format(k=f"scene_caps and {scene}")
    tag = (f"{scene} {WORLDS} worlds, 60 calls (graph), njmax {njmax} / nconmax {nconmax} per world (peaks "
           f"{meta['nefc_peak']} rows / {meta['ncon_peak']} contacts), primitive box-box {meta['primitive_boxbox']}")
    rows = {"R1": _compare(a[0], a[1]), "R2": _compare(a[0], b[0])}
    for case, (share, first, _) in rows.items():
        write_row(f"P3-1 {case}", f"{tag}, route: {share:.1%} of worlds equal after every call, first differing call "
                  f"{first}", _verdict(share, first), cmd, mode="route", scene=scene)
    none = _probe(tmp_path, "none", "--bits", "0", "--calls", "8", "--graph", *base)[1]
    routed = _probe(tmp_path, "cost", "--route", "--bits", "0", "--calls", "8", "--graph", *base)[1]
    t_ratio = routed["graph_ms"] / none["graph_ms"]
    m_ratio = routed["mem_high_bytes"] / none["mem_high_bytes"]
    ok = t_ratio <= M7_TIME and m_ratio <= M7_MEM
    write_row("P3-1 cost", f"{tag}, route: graph step {routed['graph_ms']:.2f} vs NONE {none['graph_ms']:.2f} ms "
              f"({t_ratio:.2f}x); peak memory {routed['mem_high_bytes'] / 2**30:.3f} vs "
              f"{none['mem_high_bytes'] / 2**30:.3f} GiB ({m_ratio:.2f}x)", "pass" if ok else "fail", cmd, mode="route",
              scene=scene)
    assert all(_verdict(s_, f) == "pass" for s_, f, _ in rows.values())


@pytest.mark.parametrize("task", ["t12", "scene:task_t11", "scene:task_t14"])
def test_route_boxbox(task: str, tmp_path: Path) -> None:
    """R1 / R2 / R4 with box-box pairs on the primitive collider (SCR-WP3-9), graph replays; T11 / T14 at the
    sized capacities of ``SCENE_CAPS``."""
    caps = SCENE_CAPS.get(task.removeprefix("scene:"))
    sized = ("--njmax", str(caps[0]), "--nconmax", str(caps[1])) if caps else ()
    calls = CALLS if task == "t12" else 60
    flags = ("--route", "--bits", "0", "--boxbox", "--task", task, "--calls", str(calls), "--graph-series", *sized)
    a, meta, _ = _probe(tmp_path, "a", *flags, "--worlds", str(WORLDS), "--repeat", "2")
    assert meta["error"] is None, meta
    cmd = CMD.format(k=f"boxbox and {task.removeprefix('scene:')}")
    tag = f"{task} {WORLDS} worlds, {calls} calls (graph), primitive box-box ({meta['boxbox']}), njmax {meta['njmax']}"
    if meta["route_reason"] is not None:
        share, first, _ = _compare(a[0], a[1])
        write_row("P3-1 boxbox", f"{tag}: route refused ({meta['route_reason']}); R1 {share:.1%}", "downgraded", cmd,
                  mode="route+boxbox", task=task)
        pytest.fail(f"route refused with the primitive box-box collider: {meta['route_reason']}")
    b, _, _ = _probe(tmp_path, "b", *flags, "--worlds", str(WORLDS))
    one, _, _ = _probe(tmp_path, "one", *flags, "--worlds", "1")
    rows = {"R1": _compare(a[0], a[1]), "R2": _compare(a[0], b[0])}
    for case, (share, first, _) in rows.items():
        write_row(f"P3-1 {case}", f"{tag}, route: {share:.1%} of worlds equal after every call, first differing call "
                  f"{first}", _verdict(share, first), cmd, mode="route+boxbox", task=task)
    r4 = (a[0][:, :1] == one[0]).all(axis=-1)
    write_row("P3-1 R4", f"{tag}, route: world 0 alone vs in {WORLDS}: equal on {r4.mean():.1%} of calls",
              "pass" if r4.all() else "fail", cmd, mode="route+boxbox", task=task)
    assert all(_verdict(s_, f) == "pass" for s_, f, _ in rows.values()) and r4.all()


T09_CAPS = (384, 96)  # 1.5x the peaks at 16 worlds, router capacities (245 rows / 64 contacts per world)


@pytest.mark.timeout(7200)
def test_route_cg_t09(tmp_path: Path) -> None:
    """SCR-DC5-4 (ii): T09 (CHAIN, CG solver; the cable tree takes MuJoCo Warp's sparse L'DL path) on the route at
    1,024 worlds: R1 / R2 / R4 from graph replays, the route-off R1 as the control, and M7 cost."""
    njmax, nconmax = T09_CAPS
    common = ("--task", "scene:task_t09", "--njmax", str(njmax), "--nconmax", str(nconmax), "--calls", "60",
              "--graph-series")
    a, meta, _ = _probe(tmp_path, "a", "--route", "--route-cg", "--bits", "0", "--repeat", "2", "--worlds", str(WORLDS), *common)
    assert meta["route_reason"] is None and meta["error"] is None, meta
    b, _, _ = _probe(tmp_path, "b", "--route", "--route-cg", "--bits", "0", "--worlds", str(WORLDS), *common)
    one, _, _ = _probe(tmp_path, "one", "--route", "--route-cg", "--bits", "0", "--worlds", "1", *common)
    off, off_meta, _ = _probe(tmp_path, "off", "--bits", "0", "--repeat", "2", "--worlds", str(WORLDS), *common)
    cmd = CMD.format(k="cg_t09")
    tag = (f"task_t09 (CG) {WORLDS} worlds, 60 calls (graph), njmax {njmax} / nconmax {nconmax} per world (peaks "
           f"{meta['nefc_peak']} rows / {meta['ncon_peak']} contacts); substituted {sorted(meta['substituted'] or {})}")
    rows = {"R1": _compare(a[0], a[1]), "R2": _compare(a[0], b[0])}
    for case, (share, first, _) in rows.items():
        write_row(f"P3-1 {case}", f"{tag}, route: {share:.1%} of worlds equal after every call, first differing call "
                  f"{first}", _verdict(share, first), cmd, mode="route", scene="task_t09")
    r4 = (a[0][:, :1] == one[0]).all(axis=-1)
    write_row("P3-1 R4", f"{tag}, route: world 0 alone vs in {WORLDS}: equal on {r4.mean():.1%} of calls",
              "pass" if r4.all() else "fail", cmd, mode="route", scene="task_t09")
    share, first, _ = _compare(off[0], off[1])
    write_row("P3-1 R1 control", f"task_t09 (CG) {WORLDS} worlds, route off (NONE): {share:.1%} of worlds equal "
              f"after every call, first differing call {first}", "measured", cmd, mode="none", scene="task_t09")
    none = _probe(tmp_path, "none", "--bits", "0", "--calls", "8", "--graph", "--worlds", str(WORLDS), *common[:6])[1]
    routed = _probe(tmp_path, "cost", "--route", "--route-cg", "--bits", "0", "--calls", "8", "--graph", "--worlds",
                    str(WORLDS), *common[:6])[1]
    t_ratio = routed["graph_ms"] / none["graph_ms"]
    m_ratio = routed["mem_high_bytes"] / none["mem_high_bytes"]
    write_row("P3-1 cost", f"task_t09 (CG) {WORLDS} worlds, route: graph step {routed['graph_ms']:.2f} vs NONE "
              f"{none['graph_ms']:.2f} ms ({t_ratio:.2f}x); peak memory {m_ratio:.2f}x",
              "pass" if t_ratio <= M7_TIME and m_ratio <= M7_MEM else "fail", cmd, mode="route", scene="task_t09")
    assert all(_verdict(s_, f) == "pass" for s_, f, _ in rows.values()) and r4.all()
