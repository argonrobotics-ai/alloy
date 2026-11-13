# SPDX-License-Identifier: Apache-2.0
"""``alloy.estimate(scene, worlds, seconds)``: memory, step time, wall time and compile time before a run.

It assembles one world of the scene (no stepping; the explicit MuJoCo pair rows are counted, not built:
``mujoco_pairs.count_only``) to count bodies, joints and particles and to see which solvers the scene routes to, then
applies a per-step cost model calibrated on an NVIDIA L4 and on CPU (``COST`` in this module): warm wall time within
1.3x on its calibration rows (``tests/api/test_estimate_calibration.py``), further off for scenes unlike them, and
``caveats`` names what it knows it leaves out for the scene (mesh-on-mesh contact at the start); ``measure=True``
times a short real run instead (costs a compile).

Example::

    est = alloy.estimate(scene, worlds=1024, seconds=4.0)
    print(est["wall_time_s"], est["memory_bytes"] / 1e9, "GB", est["compile_s_first_run"])
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from alloy.api import _liquid_accuracy as LA
from alloy.api._estimate_caveats import caveats, touching_mesh_pairs
from alloy.scene import cloth_accuracy as CA

#: per physics step (s): ``floor_s`` once; then per pass (``_exchange_clock``: a coupled scene steps its entries once
#: per pass of each force exchange) each solver entry's fixed cost (``fixed_s``: its launches and syncs, the part that
#: does not grow with worlds; "coupled" is the exchange's own, beside an SPH liquid), and per world: the rigid terms per
#: body / dof / shape (MuJoCo C on the CPU route; MuJoCo Warp on a CPU that the route refused (``mjw_scale`` x), as on
#: a GPU), ``sph_particle_s`` per SPH particle in flow (a pour's join as they leave the nozzle, ``_active_share``; x
#: ``coupled_particle`` beside a robot, whose contacts it also feeds) and ``particle_s`` x SOLVER_FACTOR for others
#: (MPM). Fit 2026-11-09 (FIX-LIST 7; least squares in log space with a weak pull to the earlier values) on the 55 rows
#: in ``bench/results/estimate_calib/`` (robot + water, bins, arm, cloth, pour at 1 / 4 / 64 worlds, also 256 / 512 on
#: the L4; one process per row on spot n2d CPUs, one L4; warm graphs): every row within 1.25x (the law holds 1.3x).
#: ``pair_s`` (``wp7_s1_rows.json``): per explicit MuJoCo pair row (``mujoco_pairs``: up to one per
#: shape pair) and world, x rk4_factor. MuJoCo C tests each every step: 4-32 decomposed bowls (4,482-362,448 rows) on
#: a spot n2d stepped 1.06-3.41x the estimate without it, 0.93-1.07x with; on an L4 0.5-1.0x without it: 0 there.
COST: dict[str, dict[str, Any]] = {
    "cuda": {"floor_s": 7.2e-5, "body_s": 4.0e-8, "dof_s": 5.0e-9, "particle_s": 2.0e-8, "contact_s": 4.5e-8,
             "fixed_s": {"mujoco": 4.9e-4, "sph": 2.0e-3, "coupled": 1.9e-3}, "sph_particle_s": 6.7e-8,
             "coupled_particle": 1.27, "mjw_scale": 1.0, "rk4_factor": 2.85, "pair_s": 0.0},
    "cpu": {"floor_s": 6.6e-4, "body_s": 9.8e-6, "dof_s": 3.9e-6, "particle_s": 2.0e-6, "contact_s": 1.1e-6,
            "fixed_s": {"mujoco": 5.1e-5, "mujoco_warp": 3.6e-3, "sph": 6.5e-3, "coupled": 9.0e-3},
            "sph_particle_s": 2.3e-5, "coupled_particle": 1.28, "mjw_scale": 1.03, "rk4_factor": 2.14,
            "pair_s": 1.47e-8},
}
SOLVER_FACTOR = {"mujoco": 1.0, "mpm": 8.0}     # particle_s multiples (MPM); SPH: sph_particle_s; VBD: VBD_COST
#: one VBD solve over all worlds: step_call_s + iterations x iter_call_s (launches) + particles x (step_s + iterations x
#: iter_s), particles over every world, x self_contact when the entry collides its cloth with itself; "coupled" when a
#: force exchange with a rigid entry runs (the solve then also does body contact every iteration). CPU fit 2026-11-08,
#: host-1, one core, 1 world, 40 iterations with self-contact: a sheet over a static ball, 121 / 441 / 961 particles at
#: 56 / 115 / 214 ms per solve; the same sheet coupled to a robot link, 81 / 256 / 676 at 94 / 150 / 281 ms; stored
#: divided by the 1.7 of self-contact (a 20 x 20 sheet over a ball: 57.2 s with, 33.7 s without); held on the FIX-LIST 7
#: rows (a 0.6 m sheet on a table at 10 / 40 / 60 iterations, 1 / 4 worlds: 0.98-1.19x). L4 (FIX-LIST 7): the same
#: sheet, 1 / 4 / 64 worlds: 0.14 ms per iteration whatever the worlds (iter_call_s x 1.5), per world as the 2026-11-08
#: drape fit (64 worlds of 1,354 particles, 13.4 / 35.5 / 51.3 ms per step at 10 / 40 / 60 iterations, divided by the
#: guide's 1.5x for self-contact). No coupled L4 fit, so coupled GPU solves use the free one. A VBD rod's links are
#: bodies of the entry: link_iter_s per link and iteration (CPU, a 0.4 m USB cable on rod_vbd at 10 iterations: 10 / 20 /
#: 40 links at 4.2 / 4.7 / 5.4 ms per solve on the ground, 13.6 / 13.9 / 14.7 ms beside a wrist's pads; the fixed
#: per-iteration part is the cloth fit's within 30 %). No GPU rod fit (links stay in COST's body term there).
VBD_COST: dict[str, dict[str, Any]] = {
    "cuda": {"free": {"step_call_s": 2.4e-4, "iter_call_s": 9.3e-5, "step_s": 4.6e-8, "iter_s": 5.7e-9,
                      "link_iter_s": 0.0}, "self_contact": 1.5},
    "cpu": {"free": {"step_call_s": 0.0, "iter_call_s": 4.9e-4, "step_s": 0.0, "iter_s": 2.8e-6, "link_iter_s": 3.8e-6},
            "coupled": {"step_call_s": 0.0, "iter_call_s": 1.0e-3, "step_s": 0.0, "iter_s": 4.6e-6,
                        "link_iter_s": 3.8e-6},
            "self_contact": 1.7},
}
COMPILE_S = {"cuda": 60.0, "cpu": 240.0}     # first run of a new scene kind; later runs read the kernel cache
BYTES_PER_PARTICLE = 400                      # state, neighbour lists and solver scratch (SPH / VBD); MPM: mpm_memory
#: GPU device memory [B], fitted to one L4 (bench/results/estimate_calib/memory_rows.json: what nvidia-smi books to the
#: process after 20 control periods, or the CUDA context + Warp's pool peak where the CUDA graph was dropped)
#: a cloth or cable (VBD) coupled to rigid bodies (wp7_s1_rows.json): its own contact pipeline and the
#: coupling's kernels: B, and B per world x colliding shapes ** COUPLED_POWER (it grew faster than its contact slots:
#: 20 / 43 / 93 kB per shape and world at 19 / 111 / 445 shapes)
COUPLED_BASE, COUPLED_SHAPE, COUPLED_POWER = 1.2e9, 4.36e3, 1.5
GPU_MEMORY = {"base": 3.8e8,                  # the CUDA context (2.3e8), the kernels and the first CUDA graph
              "pools": 5.5e8,                 # more with self-colliding cloth / soft bodies (VBD) or a robot in liquid
              "pools_plain": 2.7e8,           # more with cloth that does not collide with itself
              "row": 125.0, "row_dof": 5.5,   # per MuJoCo constraint row (njmax), and per row and dof
              "contact": 160.0,               # in-step scratch per contact slot (nconmax), held by the CUDA graph
              "contact_ccd": 5.4e3,           # the same on MuJoCo's convex path (meshes, cylinders: multiccd)
              "rk4": 1.5,                     # rk4 holds its stages' scratch
              "vbd_particle": 3.5e3,          # per VBD particle with self-contact (its pair buffers)
              "vbd_particle_plain": 1.5e3, "sph_particle": 1.25e3,
              "coupled": COUPLED_BASE, "coupled_shape": COUPLED_SHAPE}


def estimate(scene: Any, worlds: int | None = None, seconds: float = 1.0, device: str | None = None, *,
             measure: bool = False) -> dict[str, Any]:
    """Estimate a run of ``seconds`` simulated time in ``worlds`` worlds (JSON-ready dict).

    Keys: physics_steps_per_world, step_time_s, wall_time_s, world_steps_per_s, real_time_factor, memory_bytes,
    compile_s_first_run, counts (per world), solvers, rigid_step (its "fast" row: the rigid fast mode's step, time and
    cost, for a scene of free bodies), routes (per object: route, solver, reasons, as ``Sim.routes()``), vbd (the
    cloth / soft solver's particles per world, iterations and self-contact), caveats (sentences: what the model
    leaves out for this scene, e.g. meshes resting on meshes, which step slower than estimated), basis.
    Wall time is of warm stepping, fitted to measured runs (``bench/results/estimate_calib/``: robot + water, bins,
    an arm, cloth, a pour; 1-64 worlds, CPU and L4): within 1.3x on those 55 rows. Right after a build or reset a GPU
    run captures CUDA graphs first (up to 2.65x on a 1 s run); a scene far from those rows can be further off. Memory
    on a GPU is what the run needs to capture its CUDA graphs, fitted to 38 L4 rows (``memory_rows.json``: within
    1.2x); many worlds of scanned meshes (YCB, Google Scanned Objects) are not among them. ``measure=True`` times a
    short real run instead (wall time; no memory). Example: ``alloy.estimate(scene, 64, 2.0)``."""
    from alloy.api._build import _as_scene, pick_device  # noqa: PLC0415

    scene = _as_scene(scene)
    n = int(worlds or scene.worlds)
    dev = pick_device(device)
    if measure:
        return _measured(scene, n, float(seconds), dev)
    out = _modelled(scene, n, float(seconds), dev)
    robots = any(it.get("kind") == "robot" for it in scene.items.values())      # no fast mode for robots yet
    if out["rigid_step"]["eligible"] and not robots:    # free bodies only: the rigid fast mode beside (SPEED-NOW S1)
        out["rigid_step"]["fast"] = _fast_row(scene, n, float(seconds), dev, out)
    if LA.has_liquid(scene.items):                  # both liquid settings, cost vs accuracy (SCR-WP9-LIQRES)
        out["liquid_accuracy"] = scene.liquid_accuracy
        out["liquid"] = {acc: _liquid_row(scene, acc, n, float(seconds), dev, out) for acc in LA.SPACING}
    if any(it.get("kind") == "cloth" for it in scene.items.values()):     # both cloth settings (SCR-WP2-13)
        _cloth_rows(scene, out)
    return out


def _cloth_rows(scene: Any, out: dict[str, Any]) -> None:
    """``out["cloth"]``: each cloth setting's step and wall time for this scene (only the VBD term changes: the
    setting's iteration count, ``cloth_accuracy.KWARGS``; ``out`` already charges the scene's own setting at its
    built count), its cost against the default and its measured accuracy."""
    kind = "cuda" if out["device"].startswith("cuda") else "cpu"
    own, vbd = scene.cloth_accuracy, out["vbd"]
    passes = int(vbd.get("exchange_clock", {}).get("passes", 1))
    base = _vbd_call_s(vbd, kind, out["worlds"])
    step = {acc: out["step_time_s"] + (0.0 if acc == own else passes * (_vbd_call_s(
        {**vbd, "iterations": CA.KWARGS[acc]["iterations"]}, kind, out["worlds"]) - base)) for acc in CA.CLOTH_ACCURACY}
    out["cloth_accuracy"] = own
    out["cloth"] = {acc: {"step_time_s": step[acc], "wall_time_s": step[acc] * out["physics_steps_per_world"],
                          "cost_vs_default": step[acc] / step["default"], "accuracy": CA.ACCURACY[acc]}
                    for acc in CA.CLOTH_ACCURACY}


def _fast_row(scene: Any, n: int, seconds: float, dev: str, current: dict[str, Any]) -> dict[str, Any]:
    """The rigid fast mode's step and time for this scene (``profile="fast"``, or ``scene.rigid_step = "fast"``) and
    what it gives up (``alloy.scene.rigid_step.FAST_COST``); the scene's own row when it already steps in it, else
    one more template assembly (an estimate of a free-body scene takes about twice as long)."""
    from alloy.scene.rigid_step import FAST_COST  # noqa: PLC0415

    row = current if current["rigid_step"]["mode"] == "fast" else _modelled(
        type(scene).from_dict({**scene.to_dict(), "rigid_step": "fast"}), n, seconds, dev)
    return {"dt_s": row["rigid_step"]["dt_s"], "reason": row["rigid_step"]["reason"],
            "physics_steps_per_world": row["physics_steps_per_world"], "wall_time_s": row["wall_time_s"],
            "real_time_factor": row["real_time_factor"], "gives_up": FAST_COST}


def _run_time(template: tuple, kind: str, worlds: int, seconds: float, dt: float) -> tuple[float, int]:
    """(step time, physics steps per world) of ``seconds`` in ``worlds`` worlds of a ``_template_counts`` result on
    ``kind``'s table, with the SPH particles in flow over that time (``_active_share``)."""
    counts, solvers, steps_per_period, _, _, vbd, load = template
    sph = load["sph_particles"] - sum(k * (1.0 - _active_share(t, seconds)) for k, t in load["pours"])
    step_s = _step_time(counts, solvers, worlds, kind, vbd, {**load, "sph_active": sph})
    return step_s, max(1, int(round(seconds / dt))) * steps_per_period


def _modelled(scene: Any, n: int, seconds: float, dev: str) -> dict[str, Any]:
    template = _template_counts(scene, dev)
    counts, solvers, _, rigid, routes, vbd, _ = template
    kind = "cuda" if dev.startswith("cuda") else "cpu"
    step_s, steps = _run_time(template, kind, n, seconds, scene.dt)
    wall = steps * step_s
    return {"worlds": n, "seconds": seconds, "device": dev, "physics_steps_per_world": steps,
            "step_time_s": step_s, "wall_time_s": wall, "world_steps_per_s": n * steps / wall if wall else math.inf,
            "real_time_factor": n * seconds / wall if wall else math.inf,
            "memory_bytes": _memory(counts, n, template if kind == "cuda" else None),
            "compile_s_first_run": COMPILE_S[kind], "counts": counts, "solvers": sorted(solvers), "rigid_step": rigid,
            "routes": routes, "vbd": vbd, "caveats": caveats(template[6]),
            "basis": f"cost model ({kind} table, alloy.api._estimate.COST and VBD_COST; a coupled scene at its force "
                     "exchange clock, exchange_clock in vbd); warm wall time within 1.3x on its calibration rows"}


def _liquid_row(scene: Any, acc: str, n: int, seconds: float, dev: str, current: dict[str, Any]) -> dict[str, Any]:
    """One liquid setting's particles (every particle in the scene at that setting, cloth included), time and memory
    (the scene's own setting reuses ``current``), and its measured accuracy (``_liquid_accuracy.ACCURACY``)."""
    if acc == scene.liquid_accuracy:
        row = current
    else:
        other = type(scene).from_dict({**scene.to_dict(), "liquid_accuracy": acc})
        row = _modelled(other, n, seconds, dev)
    return {"spacing_mm": LA.SPACING[acc] * 1e3, "particles": row["counts"]["particles"],
            "step_time_s": row["step_time_s"], "wall_time_s": row["wall_time_s"], "memory_bytes": row["memory_bytes"],
            "accuracy": LA.ACCURACY[acc]}


def _template_counts(scene: Any, device: str) -> tuple[dict[str, int], set[str], int, dict[str, Any],
                                                          dict[str, dict], dict[str, Any], dict[str, Any]]:
    """Per-world counts, solvers, physics steps per control period, the rigid step (``alloy.scene.rigid_step``:
    {"dt_s", "fixed_s", "reason"}), the route rows (``Sim.routes()``), the VBD load (``_vbd_load``) and the rest of the
    load ``_step_time`` charges (``_load``) of ``scene``, assembled with one world."""
    from alloy.api._materialize import materialize  # noqa: PLC0415
    from alloy.api._build import _with_worlds  # noqa: PLC0415
    from alloy.api.errors import wrapped  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="alloy-estimate-") as tmp, wrapped("estimate"):
        from alloy.cards.library import CardLibrary  # noqa: PLC0415
        from alloy.cards.roots import assets_root  # noqa: PLC0415
        from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415
        from alloy.scene.mujoco_pairs import count_only  # noqa: PLC0415
        from alloy.scene.rigid_step_warn import PREFIX as GAP_PREFIX  # noqa: PLC0415

        from alloy.api import _pour, _rigid_step  # noqa: PLC0415

        mat = materialize(_pour.at_spacing(_with_worlds(scene, 1, trim=True), LA.spacing(scene.liquid_accuracy)),
                          Path(tmp))
        from alloy.api import catalog  # noqa: PLC0415
        from alloy.api._build import fetch_imported  # noqa: PLC0415

        lib = CardLibrary(roots=catalog.card_roots([tmp, *scene.roots]), assets=assets_root())
        fetch_imported(lib, mat.cards)
        step = _rigid_step.request(scene, "standard", (), mat.per_world_pos, announce=False)       # as the build asks
        with count_only() as pair_rows:              # the explicit MuJoCo pairs counted, not built
            built = SceneAssembler(lib, LA.router_for(scene), rigid_step=step).assemble(
                lib.scene(mat.scene_file.stem), device, world_count=1)
        m = built.model
        from alloy.integration.mpm_memory import mpm_particles  # noqa: PLC0415

        counts = {"bodies": int(m.body_count), "dofs": int(m.joint_dof_count), "particles": int(m.particle_count),
                  "shapes": int(m.shape_count), "mpm_particles": int(mpm_particles(built)[0])}
        solvers = {e.capability for e in built.entries if e.name}
        target = min((e.target_dt for e in built.entries if e.name), default=built.control_dt)
        steps_per_period = max(1, 2 ** math.ceil(math.log2(max(1.0, built.control_dt / target - 1e-9))))
        steps_per_period *= max(1, round(scene.dt / built.control_dt))     # precise fixtures: finer card period
        rs = built.report.rigid_step
        rigid = {"dt_s": built.rigid_dt, "fixed_s": rs.today if rs else built.rigid_dt,
                 "reason": rs.reason if rs else "", "mode": "fast" if rs and rs.fast else step.mode,
                 "eligible": bool(rs and rs.eligible),       # "auto" / "fast" could step it (rigid_step.why_not)
                 "warnings": [w for w in built.report.warnings if w.startswith(GAP_PREFIX)]}   # the build's, as given
        from alloy.api.sim import route_rows  # noqa: PLC0415

        routes = route_rows(built.report)                  # the pick and why, as Sim.routes()
        clock = _exchange_clock(built)
        vbd = {**_vbd_load(built.entries), "coupled": clock["coupled"], "exchange_clock": clock}
        if clock["coupled"]:            # a physics step per force exchange, or per the finest entry's own step
            steps_per_period = max(steps_per_period, clock["exchanges"] * max(1, round(scene.dt / built.control_dt)))
        load = {**_load(scene, built, device), "pair_rows": int(sum(pair_rows)),
                "touching_mesh_pairs": touching_mesh_pairs(m)}
    return counts, solvers, steps_per_period, rigid, routes, vbd, load


def _load(scene: Any, built: Any, device: str) -> dict[str, Any]:
    """What ``_step_time`` charges beyond the counts: SPH particles per world, each pour's (particles, seconds it
    pours), whether an SPH entry exchanges forces, and which MuJoCo steps the rigid entry ("c": the CPU route on
    MuJoCo C; "warp": MuJoCo Warp, on a GPU or where the CPU route refuses the entry; None: no MuJoCo entry)."""
    sph = [e for e in built.entries if e.name and "sph" in e.capability]
    names = {e.name for e in sph}
    pairs = tuple(built.coupler.pairs) if built.coupler.kind != "none" else ()
    pours = []
    for name, it in scene.items.items():
        handle = built.handles.get(name) if it.get("pour") else None
        if handle is not None:
            p = it["pour"]
            pours.append((len(handle.template.particle), float(p["volume"]) / float(p["rate"])))
    from alloy._vendor import newton  # noqa: PLC0415

    mj = [e for e in built.entries if e.name and e.capability == "mujoco"]
    flags = built.model.shape_flags.numpy() if built.model.shape_count else np.zeros(0, np.int32)
    return {"sph_particles": sum(len(e.particles) for e in sph), "pours": pours, "mujoco": _mujoco_route(built, device),
            "sph_coupled": any(p.source in names or p.destination in names for p in pairs),
            "rk4": any(e.solver_kwargs.get("integrator") == "rk4" for e in mj),
            **_capacities(mj, int(np.count_nonzero(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)))),
            "ccd": bool(mj) and (any(e.solver_kwargs.get("enable_multiccd") for e in mj) or _has_mesh(built.model)),
            "collide": int(np.count_nonzero(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)))}


def _capacities(mujoco_entries: list, collide_shapes: int) -> dict[str, int]:
    """MuJoCo's contact slots and constraint rows per world over the entries, as the build sizes them: the entry's
    own ``nconmax`` / ``njmax``, else the default rule (8 per colliding shape, at least 64; 6 rows per contact)."""
    from alloy.scene.entries import default_capacities  # noqa: PLC0415

    ncon = nrow = 0
    for e in mujoco_entries:
        c, r = default_capacities(collide_shapes)
        ncon += int(e.solver_kwargs.get("nconmax") or c)
        nrow += int(e.solver_kwargs.get("njmax") or r)
    return {"nconmax": ncon, "njmax": nrow}


def _has_mesh(model: Any) -> bool:
    """A colliding mesh shape in the model: MuJoCo collides it on its convex path (GPU memory: ``contact_ccd``)."""
    from alloy._vendor import newton  # noqa: PLC0415

    if not model.shape_count:
        return False
    kinds, flags = model.shape_type.numpy(), model.shape_flags.numpy()
    mesh = np.isin(kinds, [int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)])
    return bool(np.any(mesh & ((flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0)))


def _mujoco_route(built: Any, device: str) -> str | None:
    """"c", "warp" or None (``_load``), by the CPU route's own rule (``mjc_core_config.requested_route``,
    ``mjc_core_factory.refusal``: kinematic bodies, couplers, ...)."""
    mj = [e for e in built.entries if e.name and e.capability == "mujoco"]
    if not mj:
        return None
    if not device.startswith("cpu"):
        return "warp"
    from alloy.route.mjc_core_config import MJC, requested_route  # noqa: PLC0415
    from alloy.route.mjc_core_factory import refusal  # noqa: PLC0415

    kw = dict(mj[0].solver_kwargs)
    return "c" if requested_route(kw) == MJC and refusal(built.model, kw) is None else "warp"


def _active_share(pour_s: float, seconds: float) -> float:
    """The mean share of a pour's particles in flow over a run of ``seconds`` that pours for ``pour_s`` (they join
    as they leave the nozzle: SPH's cost follows them, 8x over a CPU pour from 12 % to all)."""
    if pour_s <= 0.0 or seconds <= 0.0:
        return 1.0
    return seconds / (2.0 * pour_s) if seconds <= pour_s else 1.0 - pour_s / (2.0 * seconds)


def _exchange_clock(built: Any) -> dict[str, Any]:
    """A coupled scene's force-exchange clock as the coupling scheduler sets it with its pairs in contact
    (``alloy.coupling.clocks``): dt_x = the control grid value under min over pairs of safety x omega_dt_max x
    sqrt(m / k_c), with k_c the pair's interface stiffness, else max(soft_contact_ke, shape ke), and m the lightest
    body of the pair's source entry (the scheduler's axis mass is that or heavier); passes per exchange: 1 one-way,
    ``lagged_iterations`` lagged, the staggered minimum. Entry substep rungs are not modelled. Uncoupled:
    {"coupled": False, "passes": 1}."""
    from alloy.core.timebase import grid_dt  # noqa: PLC0415
    from alloy.coupling.clocks import grid_level, interface_bound  # noqa: PLC0415
    from alloy.coupling.types import SchedulerPolicy  # noqa: PLC0415

    pairs = tuple(built.coupler.pairs) if built.coupler.kind != "none" else ()
    if not pairs:
        return {"coupled": False, "passes": 1}
    pol, m = SchedulerPolicy(), built.model
    ke = max(float(getattr(m, "soft_contact_ke", 0.0) or 0.0),
             float(m.shape_material_ke.numpy().max()) if m.shape_count else 0.0)
    entries = {e.name: e for e in built.entries if e.name}
    bound, m_eff = math.inf, math.inf
    for p in pairs:
        if "sph" in getattr(entries.get(p.destination), "capability", ""):
            continue                                      # contact-free destination: no bound (scheduler._pair_k_c)
        mass = _lightest(m, entries.get(p.source))
        bound = min(bound, interface_bound(float(p.interface_stiffness or ke), mass, pol.clock.safety,
                                           pol.clock.omega_dt_max))
        m_eff = min(m_eff, mass)
    j, _ = grid_level(min(bound, pol.clock.dt_x_max), built.control_dt, pol.clock.j_max)
    dt_x = grid_dt(built.control_dt, j)
    staggered = any(p.mode == "staggered" for p in pairs)
    passes = (1 if all(p.one_way for p in pairs) else
              max(pol.tol.k_min, pol.lagged_iterations) if staggered else pol.lagged_iterations)
    return {"coupled": True, "dt_x_s": dt_x, "exchanges": max(1, round(built.control_dt / dt_x)), "passes": passes,
            "k_c": float(ke), "m_eff_kg": m_eff}


def _lightest(model: Any, entry: Any) -> float:
    """The lightest positive body mass of ``entry`` (its particles' when it has no bodies) [kg]; inf if none."""
    if entry is None:
        return math.inf
    if entry.bodies:
        mass = model.body_mass.numpy()[list(entry.bodies)]
    else:
        mass = model.particle_mass.numpy()[list(entry.particles)] if entry.particles else []
    mass = [float(x) for x in mass if x > 0]
    return min(mass) if mass else math.inf


def _vbd_load(entries: Any) -> dict[str, Any]:
    """The VBD entries' particles (per world), iterations and self-contact, from the kwargs the build gives them
    (``alloy.scene.entries.solver_kwargs``: capability default, cloth_accuracy, the self-contact rule)."""
    vbd = [e for e in entries if e.name and "vbd" in e.capability]
    if not vbd:
        return {"particles": 0, "links": 0, "iterations": 0, "self_contact": False}
    return {"particles": sum(len(e.particles) for e in vbd), "links": sum(len(e.bodies) for e in vbd),
            "iterations": max(int(e.solver_kwargs.get("iterations", 10)) for e in vbd),
            "self_contact": any(bool(e.solver_kwargs.get("particle_enable_self_contact")) for e in vbd)}


def _factor(solvers: set[str]) -> float:
    return max([f for key, f in SOLVER_FACTOR.items() if any(key in s for s in solvers)] or [1.0])


def _step_time(counts: dict[str, int], solvers: set[str], worlds: int, kind: str,
               vbd: dict[str, Any] | None = None, load: dict[str, Any] | None = None) -> float:
    """One physics step (one force exchange in a coupled scene) of ``worlds`` worlds (``COST``): the floor, then per
    pass (``_exchange_clock``) the entries' fixed costs, the per-world rigid, SPH and other particle terms, and the
    VBD solve (``_vbd_call_s``). ``load``: ``_load`` plus "sph_active" (SPH particles in flow per world)."""
    c, fixed = COST[kind], COST[kind]["fixed_s"]
    vbd, load = vbd or {"particles": 0}, load or {}
    mj, sph, coupled = load.get("mujoco"), float(load.get("sph_active", 0.0)), bool(load.get("sph_coupled"))
    shapes = load.get("collide", counts["shapes"])          # colliding shapes (a robot's visual meshes are free)
    rigid = (counts["bodies"] * c["body_s"] + counts["dofs"] * c["dof_s"] + shapes * c["contact_s"]
             + int(load.get("pair_rows", 0)) * c["pair_s"])
    once = 0.0
    if mj is not None:
        once += fixed["mujoco_warp"] if mj == "warp" and "mujoco_warp" in fixed else fixed["mujoco"]
        rigid *= c["mjw_scale"] if mj == "warp" else 1.0
        if load.get("rk4"):                                  # four evaluations per step
            once, rigid = once * c["rk4_factor"], rigid * c["rk4_factor"]
    if load.get("sph_particles"):
        once += fixed["sph"] + (fixed["coupled"] if coupled else 0.0)
    others = max(0, counts["particles"] - int(vbd["particles"]) - int(load.get("sph_particles", 0)))
    per_world = (rigid + sph * c["sph_particle_s"] * (c["coupled_particle"] if coupled else 1.0)
                 + others * c["particle_s"] * _factor(solvers))
    passes = int(vbd.get("exchange_clock", {}).get("passes", 1))
    return c["floor_s"] + passes * (once + worlds * per_world + _vbd_call_s(vbd, kind, worlds))


def _vbd_call_s(vbd: dict[str, Any], kind: str, worlds: int) -> float:
    """One VBD solve over ``worlds`` worlds (``VBD_COST``): its launches, particles and rod links; 0 without either."""
    if not (vbd.get("particles") or vbd.get("links")):
        return 0.0
    table = VBD_COST[kind]
    c = table["coupled"] if vbd.get("coupled") and "coupled" in table else table["free"]
    it, n, links = int(vbd["iterations"]), int(vbd["particles"]) * worlds, int(vbd.get("links", 0)) * worlds
    t = c["step_call_s"] + it * c["iter_call_s"] + n * (c["step_s"] + it * c["iter_s"])
    return t * (table["self_contact"] if vbd["self_contact"] else 1.0) + it * links * c["link_iter_s"]


def _memory(counts: dict[str, int], worlds: int, gpu_template: tuple | None = None) -> int:
    """Device bytes for ``worlds`` worlds of the template ``counts``. With ``gpu_template`` (a ``_template_counts``
    result, for a CUDA device) it is `_gpu_memory`, the measured model. Without, the CPU's: the rigid terms, 400 B
    per SPH / VBD particle and WP12's MPM model for MPM particles (``alloy.integration.mpm_memory.predict``: 6 kB
    each, 32 kB beside a body; the same numbers the build refuses on). Example: ``_memory({"bodies": 1, "dofs": 6,
    "particles": 0, "shapes": 2}, 64)``."""
    from alloy.integration.mpm_memory import predict  # noqa: PLC0415
    from alloy.parallel.costmodel import MemoryTerms, estimate_bytes_per_world  # noqa: PLC0415

    if gpu_template is not None:
        return _gpu_memory(gpu_template, worlds)
    contacts = 8 * counts["shapes"]
    terms = MemoryTerms(nv=counts["dofs"], rows_max=4 * contacts, contacts_max=contacts,
                        pairs_max=counts["shapes"] ** 2, n_bodies=counts["bodies"], n_particles=0)
    mpm = int(counts.get("mpm_particles", 0))
    per_world = estimate_bytes_per_world(terms) + (counts["particles"] - mpm) * BYTES_PER_PARTICLE
    total = per_world * worlds + 64e6
    if mpm:
        total += predict(mpm * worlds, moving=counts["bodies"] > 0)
    return int(total)


def _gpu_memory(template: tuple, worlds: int) -> int:
    """GPU bytes the run needs (``GPU_MEMORY``; audit WORLDS-7: the old rigid terms read 4-5x low for many arm
    worlds and left the CUDA context out). Per world: MuJoCo's constraint rows as the build sizes them (``njmax``,
    at least 384) times a row's bytes, its in-step scratch per contact slot (``nconmax``, at least 64; 34x more on the
    convex path, 1.5x under rk4), and the particles (cloth that collides with itself holds its pair buffers); plus
    the base. Within 1.2x on 38 L4 rows (arms, a ball, a bin, the Panda scene, cloth, liquid: 1 to 16,384 worlds)."""
    from alloy.integration.mpm_memory import predict  # noqa: PLC0415

    counts, solvers, _, _, _, vbd, load = template
    g = GPU_MEMORY
    touching = bool(vbd and vbd.get("self_contact"))
    mpm, sph = int(counts.get("mpm_particles", 0)), int(load["sph_particles"])
    scratch = load["nconmax"] * (g["contact_ccd"] if load["ccd"] else g["contact"]) * (g["rk4"] if load["rk4"] else 1.0)
    per_world = (load["njmax"] * (g["row"] + g["row_dof"] * counts["dofs"]) + scratch + sph * g["sph_particle"]
                 + max(counts["particles"] - sph - mpm, 0) * g["vbd_particle" if touching else "vbd_particle_plain"])
    pools = g["pools"] if touching or load["sph_coupled"] else g["pools_plain"] if "vbd" in solvers else 0.0
    if _vbd_coupled(vbd, load):                       # its own contact pipeline beside MuJoCo's
        pools += g["coupled"]
        per_world += g["coupled_shape"] * int(load["collide"]) ** COUPLED_POWER
    total = g["base"] + pools + per_world * worlds
    if mpm:
        total += predict(mpm * worlds, moving=counts["bodies"] > 0)
    return int(total)


def _vbd_coupled(vbd: dict[str, Any] | None, load: dict[str, Any]) -> bool:
    """A VBD entry exchanging forces with a MuJoCo entry: its own proxy contact pipeline (coupling.proxy_contacts)."""
    return bool(vbd and vbd.get("coupled") and (vbd.get("particles") or vbd.get("links")) and load.get("mujoco"))


def _measured(scene: Any, worlds: int, seconds: float, device: str) -> dict[str, Any]:
    """Build the scene, run one warm-up period, time a few periods and scale (includes the compile)."""
    from alloy.api._build import build  # noqa: PLC0415

    t0 = time.perf_counter()
    sim = build(scene, worlds=worlds, device=device)
    sim.step(1)
    compile_s = time.perf_counter() - t0
    k = max(2, min(20, int(round(0.25 / sim.dt))))
    steps0, t1 = sim.physics_steps, time.perf_counter()
    sim.step(k)
    _sync(device)
    dt_wall = (time.perf_counter() - t1) / k
    steps_per_period = max(1, (sim.physics_steps - steps0) // k)
    periods = max(1, int(round(seconds / sim.dt)))
    wall = dt_wall * periods
    routes = sim.routes()
    sim.close()
    return {"worlds": worlds, "seconds": seconds, "device": device, "physics_steps_per_world": periods *
            steps_per_period, "step_time_s": dt_wall / steps_per_period, "wall_time_s": wall,
            "world_steps_per_s": worlds * periods * steps_per_period / wall, "real_time_factor": worlds * seconds / wall,
            "memory_bytes": None, "compile_s_first_run": compile_s, "routes": routes,
            "basis": f"measured: {k} periods after warm-up"}


def _sync(device: str) -> None:
    import warp as wp  # noqa: PLC0415

    wp.synchronize_device(device)
