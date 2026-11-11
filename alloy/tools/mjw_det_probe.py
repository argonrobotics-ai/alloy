# SPDX-License-Identifier: Apache-2.0
"""Measure MuJoCo Warp ``DeterminismType`` modes on a WP13 task scene (P3-1): digests, step time, memory.

    python -m alloy.tools.mjw_det_probe --bits 4 --worlds 1024 --calls 60 --out run.npz [--caps wp8|router]
        [--repeat 2] [--graph] [--task t12] [--route] [--graph-series] [--boxbox]

``--task scene:<card scene>`` runs a card-library scene through ``alloy.sim.Simulation`` (router capacities, held
controls) instead of a WP8 task rung. ``--route`` enables Alloy's ordered route (``alloy.precision.mjw_route.enable_route``, P3-1 round 5) instead of
setting ``--bits``. ``--graph-series`` runs the first call eagerly, captures one ``step()`` and replays it for the
other calls, so the digests (R1/R2/R4) come from graph replays.

Builds the task's ``TaskRung`` (WP8's ladder runner: assembled scene, Alloy entry stepper, trace replay) once per
repeat, sets ``mjw_model.opt.deterministic`` and runs ``calls`` eager ``step()`` calls. After each call it records
the per-world digest of the Newton state plus MuJoCo Warp's ``qpos``, ``qvel`` and ``qacc_warmstart``. The npz holds
``digests`` [repeat, calls, worlds, 2] uint64, ``eager_s`` [repeat, calls], and a JSON ``meta`` string with the
memory peak (mempool high-water above the pre-build level, first repeat), graph timing, errors and sizes. Repeats
in one process give R1; separate invocations give R2. ``--caps wp8`` sizes njmax/nconmax as WP8 does (1.5x the
p99 rows/contacts measured on T12: 402 / 96); ``router`` keeps the router's defaults. Exit 0 = ran, 3 = the mode
failed (the error is in ``meta``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

log = logging.getLogger("alloy.tools")

T12_ROWS_P99, T12_CONTACTS_P99 = 402, 96  # WP8 status 2026-11-01 (T12 p99 rows / contacts per world)


@dataclass(frozen=True)
class ProbeArgs:
    bits: int
    worlds: int
    calls: int
    out: str
    caps: str = "wp8"
    repeat: int = 1
    graph: bool = False
    task: str = "t12"
    device: str = "cuda:0"
    route: bool = False
    graph_series: bool = False
    njmax: int = 0  # scene runs: per-world capacities (0 = the router's)
    nconmax: int = 0
    graph_launches: int = 20  # replays timed by --graph
    boxbox: bool = False  # box-box pairs on MuJoCo Warp's primitive collider (alloy.contact.mjw_boxbox, SCR-WP3-9)
    route_cg: bool = False  # --route also takes the CG solver (opt-in, PrecisionConfig.mjw_route_cg; SCR-DC5-4)


@dataclass
class ProbeMeta:
    bits: int
    worlds: int
    calls: int
    caps: str
    njmax: int = 0
    naconmax: int = 0  # contact capacity of the whole pack
    nv: int = 0
    mem_high_bytes: int = 0
    mem_live_bytes: int = 0
    build_s: float = 0.0
    graph_ms: float | None = None
    graph_error: str | None = None
    error: str | None = None
    calls_done: int = 0
    filled_bits: int = 0  # DeterminismType bits Alloy's sorts implement in this run (route c)
    route: bool = False
    route_reason: str | None = None  # why enable_route refused (None: enabled, or not asked)
    substituted: dict | None = None  # mjw_launch.substituted_calls() after the first repeat
    nefc_peak: int = 0  # most rows in one world after any call (first repeat)
    ncon_peak: int = 0  # most contacts in one world after any call (first repeat)
    boxbox: str | None = None  # --boxbox: "on", or why use_primitive_boxbox changed nothing
    primitive_boxbox: bool = False  # NATIVECCD disabled on the Warp model as built (the router sets it, SCR-WP3-9)
    boxbox_pairs: int = 0  # box-box geom pairs the Warp model collides


class SceneRun:
    """A card scene through ``alloy.sim.Simulation`` (router capacities, held controls), shaped like a TaskRung
    for this probe: ``solver``, ``model``, ``s0`` and ``step()`` (one control period)."""

    def __init__(self, scene: str, worlds: int, device: str, njmax: int = 0, nconmax: int = 0) -> None:
        from alloy.cards.library import CardLibrary
        from alloy.parallel.types import PackSpec
        from alloy.scene.assembler import SceneAssembler
        from alloy.sim import Simulation

        caps = None
        if njmax and nconmax:  # per-world capacities, as WP7's routed task build passes them
            caps = PackSpec(device=device, world_ids=tuple(range(worlds)), n=worlds, stream_priority=0,
                            kind="throughput", rigid_contact_max=nconmax * worlds, njmax=njmax, nconmax=nconmax,
                            shape_pairs_max=0, predicted_bytes=0)
        lib = CardLibrary()
        built = SceneAssembler(lib).assemble(lib.scene(scene), device, world_count=worlds, capacities=caps)
        with wp.ScopedDevice(device):
            self.sim = Simulation(built)
        mjc = [s for s in self.sim.stepper.solvers() if type(s).__name__ == "SolverMuJoCo"]
        if len(mjc) != 1:
            raise ValueError(f"scene {scene!r} has {len(mjc)} SolverMuJoCo entries (expected 1)")
        self.solver, self.model = mjc[0], built.model

    @property
    def s0(self):
        return self.sim.state

    def step(self) -> None:
        self.sim.step_period()


def _rung(args: ProbeArgs):
    if args.task.startswith("scene:"):
        return SceneRun(args.task[len("scene:"):], args.worlds, args.device, args.njmax, args.nconmax)
    from alloy.parallel.costmodel import capacities_from_counts
    from alloy.parallel.rung_tasks import TaskRung, pack_spec
    from alloy.parallel.types import PackPolicy

    spec = None
    if args.caps == "wp8":
        caps = capacities_from_counts(T12_CONTACTS_P99, T12_ROWS_P99, T12_CONTACTS_P99, args.worlds, PackPolicy())
        spec = pack_spec(args.worlds, args.device, caps)
    return TaskRung(args.task, args.worlds, args.device, capacities=spec)


def _peaks(rung, meta: ProbeMeta) -> None:
    d = rung.solver.mjw_data
    meta.nefc_peak = max(meta.nefc_peak, int(d.nefc.numpy().max()))
    n = int(d.nacon.numpy()[0])
    if n:
        per = np.bincount(d.contact.worldid.numpy()[:min(n, int(d.naconmax))], minlength=1)
        meta.ncon_peak = max(meta.ncon_peak, int(per.max()))


def _digests(rung) -> np.ndarray:
    from alloy.precision.digest import world_digests

    d = rung.solver.mjw_data
    extras = {"mjw_qpos": d.qpos, "mjw_qvel": d.qvel, "mjw_qacc_warmstart": d.qacc_warmstart}
    return world_digests(rung.model, rung.s0, extras).copy()


def _graph_series(rung, calls: int, device: str, meta: ProbeMeta) -> tuple[list[np.ndarray], list[float]]:
    """Call 0 eager (loads modules, allocates), then one captured ``step()`` replayed for the other calls."""
    digests, times = _series(rung, 1, device, meta)
    with wp.ScopedCapture(device) as cap:
        rung.step()
    for _ in range(calls - 1):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)
        times.append(time.perf_counter() - t0)
        digests.append(_digests(rung))
        _peaks(rung, meta)
        meta.calls_done += 1
    return digests, times


def _series(rung, calls: int, device: str, meta: ProbeMeta) -> tuple[list[np.ndarray], list[float]]:
    digests, times = [], []
    for _ in range(calls):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        rung.step()
        wp.synchronize_device(device)
        times.append(time.perf_counter() - t0)
        digests.append(_digests(rung))
        _peaks(rung, meta)
        meta.calls_done += 1
    return digests, times


def _graph_ms(rung, device: str, launches: int = 20) -> tuple[float | None, str | None]:
    """Capture one ``step()`` call and time ``launches`` replays (after the digest series; state moves on)."""
    try:
        with wp.ScopedCapture(device) as cap:
            rung.step()
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(launches):
            wp.capture_launch(cap.graph)
        wp.synchronize_device(device)
    except (RuntimeError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"[:300]
    return (time.perf_counter() - t0) / launches * 1e3, None


def _collider(m, meta: ProbeMeta) -> None:
    from alloy._vendor.mujoco_warp import DisableBit

    from alloy.contact.mjw_boxbox import box_box_pairs

    meta.primitive_boxbox = bool(int(m.opt.disableflags) & int(DisableBit.NATIVECCD))
    meta.boxbox_pairs = box_box_pairs(m)


def _one(args: ProbeArgs, meta: ProbeMeta, first: bool) -> tuple[list[np.ndarray], list[float]]:
    from alloy.precision.mjw_det import filled_bits, set_mjw_determinism

    mem0 = wp.get_mempool_used_mem_current(args.device)
    t0 = time.perf_counter()
    rung = _rung(args)
    if args.boxbox:  # before enable_route, so its coverage probe steps the primitive box-box path
        from alloy.contact.mjw_boxbox import use_primitive_boxbox

        meta.boxbox = use_primitive_boxbox(rung.solver) or "on"
    if first:
        _collider(rung.solver.mjw_model, meta)
    if args.route:
        from alloy.precision.mjw_route import enable_route

        meta.route, meta.route_reason = True, enable_route(rung.solver, allow_cg=args.route_cg)
        meta.bits = int(rung.solver.mjw_model.opt.deterministic)
    else:
        set_mjw_determinism(rung.solver, args.bits)
    meta.filled_bits = filled_bits() & meta.bits
    if first:
        d, m = rung.solver.mjw_data, rung.solver.mjw_model
        meta.build_s, meta.nv = time.perf_counter() - t0, int(m.nv)
        meta.njmax, meta.naconmax = int(d.njmax), int(d.naconmax)
        meta.mem_live_bytes = wp.get_mempool_used_mem_current(args.device) - mem0
    run_calls = _graph_series if args.graph_series else _series
    digests, times = run_calls(rung, args.calls, args.device, meta)
    if first:
        from alloy.precision.mjw_launch import substituted_calls

        meta.substituted = substituted_calls()
        meta.mem_high_bytes = wp.get_mempool_used_mem_high(args.device) - mem0
        if args.graph:
            meta.graph_ms, meta.graph_error = _graph_ms(rung, args.device, args.graph_launches)
    return digests, times


def run(args: ProbeArgs) -> ProbeMeta:
    wp.init()
    meta = ProbeMeta(args.bits, args.worlds, args.calls, args.caps)
    series, times = [], []
    try:
        for r in range(args.repeat):
            d, t = _one(args, meta, first=r == 0)
            series.append(np.stack(d))
            times.append(t)
    except (RuntimeError, ValueError) as e:
        meta.error = f"{type(e).__name__}: {e}"[:400]
        log.warning("mjw_det_probe bits=%d worlds=%d: %s", args.bits, args.worlds, meta.error)
    digests = np.stack(series) if series else np.zeros((0, 0, args.worlds, 2), np.uint64)
    np.savez(args.out, digests=digests, eager_s=np.array(times, dtype=np.float64), meta=json.dumps(asdict(meta)))
    return meta


def parse(argv: list[str] | None = None) -> ProbeArgs:
    p = argparse.ArgumentParser(prog="python -m alloy.tools.mjw_det_probe")
    p.add_argument("--bits", type=int, required=True)
    p.add_argument("--worlds", type=int, required=True)
    p.add_argument("--calls", type=int, default=60)
    p.add_argument("--out", required=True)
    p.add_argument("--caps", choices=("wp8", "router"), default="wp8")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--task", default="t12")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--route", action="store_true", help="enable Alloy's ordered route instead of --bits")
    p.add_argument("--graph-series", action="store_true", help="digests from graph replays after call 0")
    p.add_argument("--njmax", type=int, default=0, help="scene runs: rows per world (0 = router capacity)")
    p.add_argument("--nconmax", type=int, default=0, help="scene runs: contacts per world (0 = router capacity)")
    p.add_argument("--graph-launches", type=int, default=20, help="replays timed by --graph")
    p.add_argument("--boxbox", action="store_true", help="box-box pairs on MuJoCo Warp's primitive collider")
    p.add_argument("--route-cg", action="store_true", help="--route also takes the CG solver (opt-in)")
    a = p.parse_args(argv)
    return ProbeArgs(a.bits, a.worlds, a.calls, a.out, a.caps, a.repeat, a.graph, a.task, a.device, a.route,
                     a.graph_series, a.njmax, a.nconmax, a.graph_launches, a.boxbox, a.route_cg)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    meta = run(parse(argv))
    sys.stdout.write(json.dumps(asdict(meta)) + "\n")
    return 3 if meta.error else 0


if __name__ == "__main__":
    sys.exit(main())
