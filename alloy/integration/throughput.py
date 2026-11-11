# SPDX-License-Identifier: Apache-2.0
"""Per-world cost of a card scene through ``IntegratedSimulation`` on one device (SB-5 / PAR-4 diagnosis).

    python -m alloy.integration.throughput --scene task_t09 --worlds 64,256,1024 [--route-profile fast]
        [--set KEY=VALUE ...] [--profile KEY=VALUE ...] [--periods 10] [--kernels N] [--label NAME]

For each world count: assemble the card scene (WP2 ``SceneAssembler``; ``--route-profile fast`` opts in to the fast
route profile, and its rows and log lines carry the profile's cable creep caveat, SCR-WP3-8), apply ``--set`` to the MuJoCo entry's
solver kwargs (``none`` drops a key, e.g. ``alloy:noslip=none``) and ``--armature`` to CHAIN joint dofs,
``build_simulation`` with the profile and
``--profile`` overrides (every part wired), run 2 warm-up periods (eager warm-up, capture), then ``--periods`` timed
graph periods. Each point reports ms per period, RT (sim-s per GPU-s = N·control_dt·periods / wall), host syncs and
the MJWarp counters of the last step (solver iterations p50/p99/max, rows, contacts, overflow / iteration-cap
worlds). The world counts are fitted to T(N) = t_floor + N·c_w (WP8 ``fit_cost_model``); the SB-5 ceiling is
control_dt / c_w. ``--kernels K`` adds one eager period (at ``--kernel-worlds``) under Warp's kernel timer: GPU time per
kernel (top K), which is where c_w goes.

Rows go to ``bench/results/wp12/throughput.jsonl`` (one per point, one per fit, one per kernel table).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from alloy.integration.built_copy import replace_built

log = logging.getLogger("alloy.integration")

__all__ = ["Point", "kernel_table", "measure", "parse_sets", "route_caveat", "run"]

OUT = Path("bench/results/wp12/throughput.jsonl")
WARMUP_PERIODS = 2          # eager warm-up period + the capture period


@dataclasses.dataclass
class Point:
    """One timed world count."""

    label: str
    scene: str
    worlds: int
    periods: int
    ms_per_period: float
    rt: float                       # sim-s per GPU-s
    host_syncs: int
    steps_per_period: int
    counters: dict[str, Any]
    finite: bool
    invalid_worlds: int
    world0_pos: list[list[float]] = dataclasses.field(default_factory=list)   # body positions of world 0 [m]
    min_z: float = 0.0                  # lowest body origin, all worlds [m] (WP8's T09 floor gate: > -0.01)


def parse_value(text: str) -> Any:
    """``none`` -> None, ``true``/``false`` -> bool, int, float, ``(a,b)`` tuples, else the string."""
    low = text.strip().lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    if text.startswith("(") and text.endswith(")"):
        return tuple(parse_value(t) for t in text[1:-1].split(",") if t.strip())
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_sets(items: Sequence[str]) -> dict[str, Any]:
    """``KEY=VALUE`` strings -> dict (``parse_value``)."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        out[key.strip()] = parse_value(value)
    return out


def with_entry_kwargs(built: Any, sets: dict[str, Any], capability: str = "mujoco") -> Any:
    """``built`` with ``sets`` merged into the solver kwargs of its ``capability`` entries (None drops a key). The copy
    keeps the assembler's extras, so its draws read back (``built_copy.replace_built``)."""
    if not sets:
        return built
    entries = []
    for e in built.entries:
        if e.capability == capability:
            kw = {**dict(e.solver_kwargs), **sets}
            kw = {k: v for k, v in kw.items() if v is not None}
            e = dataclasses.replace(e, solver_kwargs=kw)
        entries.append(e)
    return replace_built(built, entries=entries)     # keeps _plan / randomized / randomized_seed (SCR-DC3-8)


def chain_armature(built: Any, value: float, twist_only: bool = False) -> int:
    """Armature ``value`` [kg·m²] (max with the current one) on the non-free joint dofs of CHAIN objects in every
    world; ``twist_only``: only the dof whose axis is the child link's least-inertia (long) axis. WP2's analytic rods
    use 1e-6 for the same float32 factorisation NaN under MJW's Newton solver (alloy/scene/analytic_rods.py), and
    D7 plans implicitfast + armature for chains in L1 scenes. Template joint ids are per world; returns dofs set."""
    from alloy.route.types import Route  # noqa: PLC0415

    m = built.model
    jstart, qd, jtype, child = (a.numpy() for a in (m.joint_world_start, m.joint_qd_start, m.joint_type,
                                                     m.joint_child))
    axes, inertia, arm = m.joint_axis.numpy(), m.body_inertia.numpy(), m.joint_armature.numpy()
    dofs: list[int] = []
    for h in built.handles.values():
        if h.decision.route != Route.CHAIN:
            continue
        for w in range(built.world_count):
            for g in (int(jstart[w]) + j for j in h.template.joint):
                if jtype[g] != _FREE:
                    dofs += _joint_dofs(range(qd[g], qd[g + 1]), axes, inertia[child[g]], twist_only)
    arm[dofs] = np.maximum(arm[dofs], value)
    m.joint_armature.assign(arm)
    log.info("chain armature %g kg·m² on %d dofs (%d per world, %s)", value, len(dofs), len(dofs) // built.world_count,
             "twist only" if twist_only else "all joint dofs")
    return len(dofs)


_FREE = 4   # newton.JointType.FREE


def _joint_dofs(dofs: range, axes: np.ndarray, inertia: np.ndarray, twist_only: bool) -> list[int]:
    """All of ``dofs``, or the one most aligned with the least-inertia principal axis of the child body."""
    if not twist_only:
        return list(dofs)
    w, v = np.linalg.eigh(np.asarray(inertia, np.float64))
    long_axis = v[:, int(np.argmin(w))]
    return [max(dofs, key=lambda d: abs(float(np.dot(axes[d], long_axis))))]


def assemble(scene: str, device: str, worlds: int, route_profile: str = "default") -> Any:
    """The card scene at ``worlds`` worlds (WP2 assembler with its own router, as WP8's ladder). ``route_profile``
    "fast" opts in to WP2's fast route profile (SCR-WP3-8); "default" is the route."""
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    lib = CardLibrary()
    return SceneAssembler(lib, route_profile=route_profile).assemble(lib.scene(scene), device, world_count=worlds)


def route_caveat(sim: Any) -> str | None:
    """The fast route profile's caveat (its cable creep, SCR-WP3-8) when the build carries it. The ruling: print it
    beside every speed from that profile."""
    from alloy.scene.fast_profile import FAST_CAVEAT  # noqa: PLC0415

    built = getattr(getattr(sim, "parts", None), "built", None)
    warnings = getattr(getattr(built, "report", None), "warnings", ()) or ()
    return FAST_CAVEAT if FAST_CAVEAT in warnings else None


def mjw_counters(sim: Any) -> dict[str, Any]:
    """Iterations / rows / contacts of the last step of every MuJoCo entry (WP8 ``mujoco_counters``)."""
    from alloy.parallel.counters import mujoco_counters  # noqa: PLC0415

    out: dict[str, Any] = {}
    for name, solver in sim.parts.solvers.items():
        if getattr(solver, "mjw_data", None) is None:
            continue
        s = mujoco_counters(solver, sim.model)()
        out[name] = {"iters": _pcts(s.iters), "rows": _pcts(s.rows), "contacts": _pcts(s.contacts),
                     "overflow_worlds": s.overflow, "iter_cap_worlds": s.iter_limit,
                     "opt": _mjw_opt(solver)}
    return out


def _pcts(a: np.ndarray | None) -> dict[str, float] | None:
    if a is None or a.size == 0:
        return None
    return {"p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def _mjw_opt(solver: Any) -> dict[str, Any]:
    m, d = solver.mjw_model, solver.mjw_data
    o = m.opt
    keys = ("solver", "iterations", "ls_iterations", "cone", "integrator", "noslip_iterations")
    out = {k: _plain(getattr(o, k, None)) for k in keys}
    for k in ("tolerance", "impratio", "ls_tolerance", "noslip_tolerance"):
        v = getattr(o, k, None)
        out[k] = float(v.numpy().ravel()[0]) if hasattr(v, "numpy") else _plain(v)
    inv = getattr(o, "impratio_invsqrt", None)
    if hasattr(inv, "numpy"):
        out["impratio"] = float(inv.numpy().ravel()[0]) ** -2
    out.update(nv=int(m.nv), njmax=int(d.njmax), naconmax=int(getattr(d, "naconmax", 0)), nworld=int(d.nworld),
               is_sparse=bool(getattr(m, "is_sparse", False)))
    return out


def _plain(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def build(scene: str, device: str, worlds: int, sets: dict[str, Any], prof: dict[str, Any],
          armature: float = 0.0, twist_only: bool = False, *, route_profile: str = "default") -> Any:
    """Assembled scene (``route_profile``) -> ``IntegratedSimulation`` with ``sets`` on the MuJoCo entry, ``prof``
    overrides and ``armature`` on CHAIN dofs (0: none)."""
    from alloy.integration.wiring import build_simulation  # noqa: PLC0415

    dense_max = int(prof.pop("mjw_dense_max", 0))
    if dense_max:
        _mjw_dense_max(dense_max)
    scope = prof.pop("noslip_scope", None)
    built = with_entry_kwargs(assemble(scene, device, worlds, route_profile), sets)
    if armature > 0.0:
        chain_armature(built, armature, twist_only)
    with wp.ScopedDevice(device), _noslip_scope(scope):
        return build_simulation(built, prof.pop("profile", "standard"), **prof)


@contextlib.contextmanager
def _noslip_scope(scope: str | None):
    """Experiment: the scene's scoped noslip (WP3, installed by WP2's MuJoCo factory with scope "robot") with
    ``scope`` instead ("all": every frictional row, so the §3.5 creep rule's noslip branch covers precision pairs)."""
    from alloy.contact import noslip as ns  # noqa: PLC0415

    if scope is None:
        yield
        return
    orig = ns.install_noslip
    ns.install_noslip = lambda solver, spec=ns.NoslipSpec(): orig(solver, dataclasses.replace(spec, scope=scope))
    try:
        yield
    finally:
        ns.install_noslip = orig


def _mjw_dense_max(size: int) -> None:
    """Experiment: MJWarp factors M's diagonal blocks (kinematic trees) up to ``M_BLOCK_DENSE_MAX`` dofs with tile
    Cholesky and larger ones with level-by-level sparse LDL (mujoco_warp/_src/io.py m_block_layout); T09's cable tree
    has 93 dofs. Read at ``put_model`` time, so it must be set before SolverMuJoCo is built."""
    from alloy._vendor.mujoco_warp._src import types as mjw_types  # noqa: PLC0415

    log.info("MJWarp M_BLOCK_DENSE_MAX %d -> %d", mjw_types.M_BLOCK_DENSE_MAX, size)
    mjw_types.M_BLOCK_DENSE_MAX = size


def measure(sim: Any, label: str, scene: str, periods: int) -> Point:
    """Warm-up, then ``periods`` timed periods (synchronised wall clock)."""
    for _ in range(WARMUP_PERIODS):
        sim.step_period()
    wp.synchronize_device(sim.model.device)
    syncs0 = sim.host_syncs
    t0 = time.perf_counter()
    for _ in range(periods):
        sim.step_period()
    wp.synchronize_device(sim.model.device)
    wall = time.perf_counter() - t0
    n = int(sim.model.world_count)
    rep = sim.report()
    health = rep.get("health")
    invalid = len(getattr(health, "invalid_worlds", ()) or ())
    q = sim.state.body_q.numpy()
    nb = int(sim.model.body_count) // n
    return Point(label, scene, n, periods, 1e3 * wall / periods, n * sim.clock.control_dt * periods / wall,
                 sim.host_syncs - syncs0, sim.clock.steps_per_period, mjw_counters(sim), bool(np.isfinite(q).all()),
                 invalid, np.round(q[:nb, :3], 7).tolist(), float(np.nanmin(q[:, 2])))


_HASH = re.compile(r"_[0-9a-f]{6,}$")


def kernel_table(sim: Any, top: int) -> list[dict[str, Any]]:
    """One eager period under Warp's kernel timer: [{kernel, ms, launches, share}] sorted by GPU time."""
    graphs, sim._graphs = sim._graphs, None
    try:
        wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
        sim.step_period()
        results = wp.timing_end()
    finally:
        sim._graphs = graphs
    total: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for r in results:
        name = _HASH.sub("", r.name.replace("forward kernel ", ""))
        total[name] += r.elapsed
        count[name] += 1
    grand = sum(total.values()) or 1.0
    rows = sorted(total, key=total.get, reverse=True)
    return [{"kernel": k, "ms": round(total[k], 4), "launches": count[k], "share": round(total[k] / grand, 4)}
            for k in rows[:top]] + [{"kernel": "TOTAL", "ms": round(grand, 3), "launches": sum(count.values()),
                                     "share": 1.0}]


def fit(points: Sequence[Point], control_dt: float) -> dict[str, Any] | None:
    """(t_floor, c_w) over the points' world counts; ceiling RT = control_dt / c_w."""
    from alloy.parallel.costmodel import fit_cost_model  # noqa: PLC0415

    if len({p.worlds for p in points}) < 2:
        return None
    t_floor, c_w = fit_cost_model([p.worlds for p in points], [p.ms_per_period * 1e-3 for p in points])
    return {"t_floor_ms": 1e3 * t_floor, "c_w_ms": 1e3 * c_w,
            "ceiling_rt": control_dt / c_w if c_w > 0 else float("inf")}


def _write(row: dict[str, Any], out: Path) -> None:
    from alloy.testing.integration import git_commit  # noqa: PLC0415

    out.parent.mkdir(parents=True, exist_ok=True)
    row = {"commit": git_commit(), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
    with out.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run(args: argparse.Namespace) -> list[Point]:
    """Every world count in ``args.worlds``; kernel table at the largest when asked; fit; rows to ``args.out``."""
    sets, prof = parse_sets(args.set), parse_sets(args.profile)
    meta = {"label": args.label, "scene": args.scene, "route_profile": args.route_profile, "sets": sets,
            "profile": prof, "device": args.device, "armature": args.armature, "twist_only": args.twist_only}
    points, sim = [], None
    for n in sorted(args.worlds):
        sim = build(args.scene, args.device, n, dict(sets), dict(prof), args.armature, args.twist_only,
                    route_profile=args.route_profile)
        p = measure(sim, args.label, args.scene, args.periods)
        caveat = route_caveat(sim)
        log.info("%s N=%d: %.1f ms/period, RT %.3f, syncs %d, counters %s", args.label, n, p.ms_per_period, p.rt,
                 p.host_syncs, json.dumps(p.counters))
        if caveat:
            log.warning("%s N=%d RT %.3f: %s", args.label, n, p.rt, caveat)
        _write({**meta, "kind": "point", "caveat": caveat, **dataclasses.asdict(p)}, args.out)
        points.append(p)
        if args.kernels and n == (args.kernel_worlds or max(args.worlds)):
            table = kernel_table(sim, args.kernels)
            _write({**meta, "kind": "kernels", "worlds": n, "table": table}, args.out)
            for row in table:
                log.info("  %-60s %9.3f ms %6d launches %5.1f%%", row["kernel"][:60], row["ms"], row["launches"],
                         100 * row["share"])
        del sim
    f = fit(points, args.control_dt) if points else None
    if f:
        _write({**meta, "kind": "fit", "worlds": [p.worlds for p in points], **f}, args.out)
        log.info("%s fit: t_floor %.1f ms, c_w %.3f ms/world/period, ceiling %.2fx RT", args.label, f["t_floor_ms"],
                 f["c_w_ms"], f["ceiling_rt"])
    return points


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.integration.throughput", description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", default="task_t09")
    ap.add_argument("--route-profile", default="default", help="SceneAssembler route profile: default | fast "
                    "(opt-in, SCR-WP3-8; its speed rows carry the cable creep caveat)")
    ap.add_argument("--worlds", type=lambda s: [int(x) for x in s.split(",")], default=[64, 256, 1024])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--periods", type=int, default=10)
    ap.add_argument("--set", action="append", default=[], help="MuJoCo entry solver kwarg KEY=VALUE (none drops)")
    ap.add_argument("--profile", action="append", default=[], help="profile field KEY=VALUE (profile=NAME too)")
    ap.add_argument("--kernels", type=int, default=0, help="per-kernel GPU time table (top K), one eager period")
    ap.add_argument("--kernel-worlds", type=int, default=0, help="world count of the kernel table (default: largest;"
                    " eager MJWarp allocates its narrow-phase scratch per call, 0.85 GB at 256 T09 worlds)")
    ap.add_argument("--armature", type=float, default=0.0, help="armature [kg·m²] on CHAIN joint dofs (0: none)")
    ap.add_argument("--twist-only", action="store_true", help="--armature on the twist (long-axis) dof only")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--control-dt", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    wp.config.quiet = True
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
