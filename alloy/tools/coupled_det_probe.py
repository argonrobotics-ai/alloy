# SPDX-License-Identifier: Apache-2.0
"""Repeat probe for VBD, MuJoCo + VBD and MuJoCo + SPH scenes (SCR-DC5-4 (i): their PRC-13 classification).

    python -m alloy.tools.coupled_det_probe --case mjc_sph_route --worlds 1024 --periods 25 --out run.npz
        [--repeat 2] [--world-ids 0] [--device cuda:0]

Each case builds its scene the way its users run it:
- Card scenes go through ``alloy.integration.build_simulation(built, "standard")``: the router, the coupler WP4's
  ``make_stepper`` picks (the Scheduler for two entries), and the HealthMonitor, which turns on Alloy's ordered
  SolverMuJoCo route where the path is covered.
- WP9's ``sph_pour`` goes through its own ``make_pour_sim`` (``SolverCoupledProxy``: MuJoCo source, SolverDFSPH
  destination). Then the standard preset's ``enable_solver_determinism`` runs, as the HealthMonitor would.
- A ``*_route`` variant builds the MuJoCo entry on the route's proven configuration (``ROUTE_KWARGS``), so the route
  is on inside the coupled step. Everything else is as its scene builds it.

One process builds the scene ``--repeat`` times. After every control period it records the world digests (state
plus the solvers' registered internals):
- ``digests``: extras without a world axis are hashed into every world (R1 compares builds, R2 compares processes);
- ``digests_world``: such extras are left out (R4: a ``--worlds 1 --world-ids k`` run against world k of a batch).
The npz also holds ``period_s`` [repeat, periods] and a JSON ``meta`` (claim, reasons, route state per SolverMuJoCo,
solver classes, graph). Exit 0 = ran, 3 = the build or a step failed (the error is in ``meta``).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import json
import logging
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.tools")

# The ordered route's proven SolverMuJoCo path (alloy.precision.mjw_route.static_reason)
ROUTE_KWARGS: Mapping[str, str] = {"solver": "newton", "cone": "elliptic", "jacobian": "sparse",
                                   "integrator": "implicitfast"}


@dataclass(frozen=True)
class Built:
    sim: Any                      # SimulationLike: model, state, step_period(), state_arrays()
    solvers: tuple[Any, ...]      # what effective_determinism reads (coupled solvers are unwrapped there)
    config: Any                   # the PrecisionConfig that ran
    drive: Callable[[Any, int], None] | None = None  # sets the controls before period k (None: held)


# --- scenes ----------------------------------------------------------------------------------------------------


def _keep(*names: str) -> Callable[[Any], Any]:
    """Scene card edit: keep only the named objects; drop attachments and emitters."""
    return lambda card: dataclasses.replace(card, objects=tuple(p for p in card.objects if p.object in names),
                                            attachments=(), emitters=())


def _foam_on_table(card: Any) -> Any:
    """Scene card edit: the table and the 10 cm PU foam block (SOFT_VBD, as DC5's pad) 1 mm above it plus its particle
    contact radius: a SOFT_VBD solid rests one radius above its support (SCR-XB4-5-1). The block centre was at 26 mm,
    which put its bottom 24 mm inside the table (top at z = 0; the block is 10 cm, not 5 cm)."""
    from alloy.cards.library import CardLibrary
    from alloy.cards.types import Placement
    from alloy.scene.emitters.deformable import box_mesh, soft_particle_radius

    lib = CardLibrary()
    table = next(p for p in card.objects if p.object == "table_top_plywood")
    block = lib.object("foam_block")
    top = float(table.pose[2]) + float(lib.object("table_top_plywood").shape.size[2])
    z = top + float(block.shape.size[2]) + 1.0e-3 + soft_particle_radius(*box_mesh(block.shape, block.budget.tets))
    foam = Placement("foam_block", (0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0))
    return dataclasses.replace(card, objects=(table, foam), attachments=(), emitters=())


def _with_kwargs(entry: Any, kwargs: Mapping[str, Any]) -> Any:
    return dataclasses.replace(entry, solver_kwargs={**dict(entry.solver_kwargs), **kwargs})


def card_case(scene: str, worlds: int, device: str, world_ids: Sequence[int] | None, *,
              edit: Callable[[Any], Any] | None = None, mjc_kwargs: Mapping[str, Any] | None = None,
              graph: str = "auto", route: bool = False) -> Built:
    """A card scene through ``build_simulation(built, "standard")`` (``world_ids``: global ids, SCR-WP8-2;
    ``graph``: the profile's graph mode, "off" for eager periods; ``route``: the profile's ``mjw_route``, which asks for
    Alloy's ordered MuJoCo route on a sparse Jacobian, coordinator ruling 2026-11-06 06:23Z)."""
    from alloy.cards.library import CardLibrary
    from alloy.integration.wiring import build_simulation
    from alloy.scene.assembler import SceneAssembler

    lib = CardLibrary()
    card = lib.scene(scene)
    card = edit(card) if edit is not None else card
    built = SceneAssembler(lib).assemble(card, device, world_count=worlds, world_ids=world_ids)
    if mjc_kwargs:
        built = dataclasses.replace(built, entries=[_with_kwargs(e, mjc_kwargs) if e.capability == "mujoco" else e
                                                    for e in built.entries])
    with wp.ScopedDevice(device):
        sim = build_simulation(built, "standard", graph=graph, **({"mjw_route": True} if route else {}))
    cfg = sim.precision
    if mjc_kwargs is ROUTE_KWARGS:   # *_route cases ask for Alloy's ordered route explicitly (ruling 2026-11-03 23:15Z)
        from alloy.precision.determinism import enable_solver_determinism, refresh_coupled_constants

        cfg = dataclasses.replace(cfg, mjw_route=True)
        enable_solver_determinism(cfg, list(sim.parts.solvers.values()))   # before the first step / capture
        refresh_coupled_constants(sim.parts.stepper)                        # the coupler's proxy masses follow
    return Built(sim, tuple(sim.parts.solvers.values()), cfg)


def mixed_case(name: str, worlds: int, device: str, world_ids: Sequence[int] | None, *, graph: str = "auto") -> Built:
    """One of ``alloy.testing.scene.mixed_scenes`` (sand_heap: MPM; cable_on_table: rod) through
    ``build_simulation(built, "standard")``, assembled as the integration tests do (rod self-test stubbed)."""
    from alloy.integration.wiring import build_simulation
    from alloy.testing import integration as I
    from alloy.testing import scene as S

    card, lib = {n: (c, lb) for n, c, lb in S.mixed_scenes()}[name]
    built = I.assemble(card, lib, device, worlds)
    with wp.ScopedDevice(device):
        sim = build_simulation(built, "standard", graph=graph)
    return Built(sim, tuple(sim.parts.solvers.values()), sim.precision)


@contextlib.contextmanager
def mujoco_defaults(kwargs: Mapping[str, Any]) -> Iterator[None]:
    """Within the block, SolverMuJoCo options its builder leaves at None take ``kwargs`` (a probe-only scope for
    scenes that build their SolverMuJoCo inline, as ``make_pour_sim`` does)."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    with solver_defaults(SolverMuJoCo, kwargs):
        yield


@contextlib.contextmanager
def solver_defaults(cls: type, kwargs: Mapping[str, Any]) -> Iterator[None]:
    """Within the block, options of ``cls`` its builder leaves at None take ``kwargs`` (probe-only)."""
    init = cls.__init__

    @functools.wraps(init)
    def patched(self: Any, *args: Any, **kw: Any) -> None:
        init(self, *args, **{**kw, **{k: v for k, v in kwargs.items() if kw.get(k) is None}})

    cls.__init__ = patched
    try:
        yield
    finally:
        cls.__init__ = init


def vbd_defaults(on: bool) -> contextlib.AbstractContextManager:
    """``--vbd-det``: every SolverVBD built in the block runs Newton's ordered mode, ``deterministic=RUN_TO_RUN``
    (Warp's deterministic lowering on its atomic modules, per-particle contact gather; ordered-VBD spike)."""
    if not on:
        return contextlib.nullcontext()
    from alloy._vendor.newton.solvers import SolverVBD

    return solver_defaults(SolverVBD, {"deterministic": wp.DeterministicMode.RUN_TO_RUN})


def pour_drive(sim: Any, period: int) -> None:
    """``sph_pour`` pour: the carriage ramps to +19 mm at 0.1 m/s (the cup's wall meets the receiver's, a MuJoCo
    contact) while the cup tilts to 2 rad at 4 rad/s, the same targets in every world."""
    t = (period + 1) * sim.control_dt
    q = sim.control.joint_target_q.numpy().reshape(sim.model.world_count, -1)
    q[:, 0], q[:, 1] = min(0.019, 0.1 * t), min(2.0, 4.0 * t)
    sim.control.joint_target_q.assign(q.reshape(-1))


def pour_case(worlds: int, device: str, world_ids: Sequence[int] | None, *,
              mjc_kwargs: Mapping[str, Any] | None = None, pour: bool = False) -> Built:
    """WP9's ``sph_pour`` (identical worlds: ``world_ids`` is not used) plus the standard preset's
    ``enable_solver_determinism``; ``pour``: driven by ``pour_drive`` instead of held."""
    from alloy.precision.config import gpu_tier, preset
    from alloy.precision.determinism import enable_solver_determinism
    from alloy.solvers.sph.pour_target import PourTargetSpec, make_pour_sim

    with mujoco_defaults(mjc_kwargs or {}), wp.ScopedDevice(device):
        sim = make_pour_sim(PourTargetSpec(), worlds, device)
    cfg = preset("standard", gpu_tier(device))
    if mjc_kwargs is ROUTE_KWARGS:   # *_route cases ask for the ordered route explicitly (ruling 2026-11-03 23:15Z)
        cfg = dataclasses.replace(cfg, mjw_route=True)
    enable_solver_determinism(cfg, [sim.solver])
    return Built(sim, (sim.solver,), cfg, pour_drive if pour else None)


CASES: dict[str, Callable[..., Built]] = {
    "vbd_cloth": functools.partial(card_case, "demo_cloth_drop", edit=_keep("table_top_plywood", "towel_cotton")),
    "vbd_soft": functools.partial(card_case, "demo_cloth_drop", edit=_foam_on_table),
    "mjc_vbd": functools.partial(card_case, "demo_cloth_drop"),
    "mjc_vbd_route": functools.partial(card_case, "demo_cloth_drop", mjc_kwargs=ROUTE_KWARGS),
    "mjc_vbd_eager": functools.partial(card_case, "demo_cloth_drop", graph="off"),
    "mjc_vbd_route_eager": functools.partial(card_case, "demo_cloth_drop", mjc_kwargs=ROUTE_KWARGS, graph="off"),
    "mjc_sph": pour_case,
    "mjc_sph_route": functools.partial(pour_case, mjc_kwargs=ROUTE_KWARGS),
    "mjc_sph_pour": functools.partial(pour_case, pour=True),
    "mjc_sph_pour_route": functools.partial(pour_case, mjc_kwargs=ROUTE_KWARGS, pour=True),
    # task-regression GPU checks (coordinator ruling 2026-11-06 06:23Z): as the standard profile builds them, and with
    # the profile asking for the ordered route (sparse Jacobian)
    **{f"{name}{'_route' if r else ''}": functools.partial(card_case, scene, route=r)
       for name, scene in (("t12", "task_t12"), ("dc_sled", "task_dc_rack_sled"), ("job_bottle", "task_job_bottle"))
       for r in (False, True)},
    "demo_pour": functools.partial(card_case, "demo_pour"),                  # SPH water into a tumbler (card)
    # GPU smoke families without a library card (alloy.tools.gpu_smoke)
    **{f"mixed_{name}": functools.partial(mixed_case, name) for name in ("sand_heap", "cable_on_table")},
}


# --- probe -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeArgs:
    case: str
    worlds: int
    periods: int
    out: str
    repeat: int = 1
    device: str = "cuda:0"
    world_ids: tuple[int, ...] | None = None
    max_graph_steps: int = 0  # > 0: IntegratedSimulation's steps per captured period graph (WP4's crash check)
    vbd_det: bool = False     # SolverVBD with deterministic=RUN_TO_RUN (ordered-VBD spike)


@dataclass
class ProbeMeta:
    case: str
    worlds: int
    periods: int
    world_ids: list[int] | None = None
    claim: str = ""
    reasons: list[str] = field(default_factory=list)
    solvers: list[str] = field(default_factory=list)       # unwrapped solver classes
    routes: list[dict] = field(default_factory=list)       # per SolverMuJoCo: enabled, reason, integrator, cone
    native_share: float = 0.0
    use_graph: bool = False
    stepper: str = ""
    build_s: float = 0.0
    mem_high_bytes: int = 0
    periods_done: int = 0
    invalid_worlds: list[int] = field(default_factory=list)  # per repeat: worlds latched INVALID at the end; a
    error: str | None = None                                  # latched world freezes, so its repeat proves nothing


def _route_state(solver: Any) -> dict:
    from alloy._vendor.mujoco_warp._src import types

    from alloy.precision.mjw_route import route_enabled, route_reason

    opt = solver.mjw_model.opt
    return {"enabled": route_enabled(solver), "reason": route_reason(solver),
            "integrator": types.IntegratorType(opt.integrator).name, "cone": types.ConeType(opt.cone).name,
            "solver": types.SolverType(opt.solver).name, "sparse": bool(solver.mjw_model.is_sparse)}


def _describe(b: Built, worlds: int, meta: ProbeMeta) -> None:
    from alloy.precision.determinism import unwrap_solvers, effective_determinism, native_share

    mode, reasons = effective_determinism(b.config, b.solvers, worlds)
    flat = list(unwrap_solvers(b.solvers))
    meta.claim, meta.reasons = mode.name, list(reasons)
    meta.solvers = [type(s).__name__ for s in flat]
    meta.routes = [_route_state(s) for s in flat if getattr(s, "mjw_model", None) is not None]
    meta.native_share = native_share(b.solvers)
    meta.use_graph = bool(getattr(b.sim, "use_graph", False))
    meta.stepper = type(getattr(b.sim, "stepper", None) or getattr(b.sim, "solver", None)).__name__


def _mem(device: str, high: bool = False) -> int:
    """Mempool bytes in use (or the high-water mark) on a CUDA device; 0 on CPU (no mempool)."""
    if not wp.get_device(device).is_cuda:
        return 0
    return int(wp.get_mempool_used_mem_high(device) if high else wp.get_mempool_used_mem_current(device))


def _one(args: ProbeArgs, meta: ProbeMeta, first: bool) -> tuple[np.ndarray, np.ndarray, list[float]]:
    from alloy.precision.repeat import sim_parts

    mem0 = _mem(args.device)
    t0 = time.perf_counter()
    with vbd_defaults(args.vbd_det):
        b = CASES[args.case](args.worlds, args.device, args.world_ids)
    if args.max_graph_steps and hasattr(b.sim, "max_graph_steps"):
        b.sim.max_graph_steps = args.max_graph_steps
    if first:
        meta.build_s = time.perf_counter() - t0
        _describe(b, args.worlds, meta)
    parts = sim_parts(b.sim)
    full, per, times = [parts.digests(b.sim.state)], [parts.digests(b.sim.state, False)], []
    for k in range(args.periods):
        if b.drive is not None:
            b.drive(b.sim, k)
        wp.synchronize_device(args.device)
        t = time.perf_counter()
        b.sim.step_period()
        wp.synchronize_device(args.device)
        times.append(time.perf_counter() - t)
        full.append(parts.digests(b.sim.state))
        per.append(parts.digests(b.sim.state, False))
        meta.periods_done += 1 if first else 0
    if first:
        meta.mem_high_bytes = _mem(args.device, high=True) - mem0
    health = getattr(b.sim, "health", None)
    if health is not None:
        meta.invalid_worlds.append(len(health.poll().invalid_worlds))
    return np.stack(full), np.stack(per), times


def run(args: ProbeArgs) -> ProbeMeta:
    wp.init()
    meta = ProbeMeta(args.case, args.worlds, args.periods, list(args.world_ids) if args.world_ids else None)
    full, per, times = [], [], []
    try:
        for r in range(args.repeat):
            f, p, t = _one(args, meta, first=r == 0)
            full.append(f)
            per.append(p)
            times.append(t)
    except (RuntimeError, ValueError, TypeError) as e:
        meta.error = f"{type(e).__name__}: {e}"[:400]
        log.warning("coupled_det_probe %s worlds=%d: %s", args.case, args.worlds, meta.error)
    empty = np.zeros((0, 0, args.worlds, 2), np.uint64)
    np.savez(args.out, digests=np.stack(full) if full else empty, digests_world=np.stack(per) if per else empty,
             period_s=np.array(times, dtype=np.float64), meta=json.dumps(asdict(meta)))
    return meta


def parse(argv: list[str] | None = None) -> ProbeArgs:
    p = argparse.ArgumentParser(prog="python -m alloy.tools.coupled_det_probe")
    p.add_argument("--case", choices=sorted(CASES), required=True)
    p.add_argument("--worlds", type=int, required=True)
    p.add_argument("--periods", type=int, default=25)
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--world-ids", type=int, nargs="*", default=None, help="global world ids (card scenes)")
    p.add_argument("--max-graph-steps", type=int, default=0, help="steps per captured period graph (0: default)")
    p.add_argument("--vbd-det", action="store_true", help="SolverVBD with deterministic=RUN_TO_RUN (Newton's ordered mode)")
    a = p.parse_args(argv)
    return ProbeArgs(a.case, a.worlds, a.periods, a.out, a.repeat, a.device,
                     tuple(a.world_ids) if a.world_ids else None, a.max_graph_steps, a.vbd_det)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    meta = run(parse(argv))
    sys.stdout.write(json.dumps(asdict(meta)) + "\n")
    return 3 if meta.error else 0


if __name__ == "__main__":
    sys.exit(main())
