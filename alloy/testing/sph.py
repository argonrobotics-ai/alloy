# SPDX-License-Identifier: Apache-2.0
"""SPH reference scenes for tests, laws and bench runs (WP9-owned test helpers, SPEC §6.1).

Scenes are built directly with ``newton.ModelBuilder`` (no cards), so SolverDFSPH can be tested before the card
pipeline lands. Each returns an ``SPHScene`` with the finalized model, the solver config and the geometry needed to
evaluate laws (tank inner box, base area, per-world fill).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.solvers.sph.build import add_box_tank, add_floating_box, add_fluid_block, add_fluid_points, add_spares
from alloy.solvers.sph.build import wall_box
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes


@dataclass
class SPHScene:
    model: newton.Model
    config: DFSPHConfig
    inner_lo: tuple[float, float, float]
    inner_hi: tuple[float, float, float]
    fill: list[tuple[int, int, int]]  # lattice counts per world
    tank: list[dict[str, int]] = field(default_factory=list)  # shape ids per world
    bodies: dict[str, float] = field(default_factory=dict)

    @property
    def base_area(self) -> float:
        return (self.inner_hi[0] - self.inner_lo[0]) * (self.inner_hi[1] - self.inner_lo[1])

    def make_solver(self, **kwargs):
        from alloy.solvers.sph.solver import SolverDFSPH

        return SolverDFSPH(self.model, self.config, **kwargs)


def default_config(r: float = 1.0e-3, **overrides) -> DFSPHConfig:
    cfg = DFSPHConfig(particle_radius=r, **overrides)
    cfg.validate()
    return cfg


def water_column(nx: int = 6, ny: int = 6, nz: int = 20, *, worlds: int = 1, fills: list[int] | None = None,
                 cfg: DFSPHConfig | None = None, headroom: int = 8, spares: int = 0,
                 device: str | None = None) -> SPHScene:
    """A resting column nx x ny x nz on the 2r lattice in a box tank (walls h thick) per world.

    ``fills`` overrides the column height (nz) per world, for per-world independence tests.
    """
    cfg = cfg or default_config()
    fills = fills or [nz] * worlds
    s = cfg.spacing
    hmax = max(fills) + headroom
    inner_lo = (0.0, 0.0, 0.0)
    inner_hi = (nx * s, ny * s, hmax * s)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    tanks = []
    for w in range(len(fills)):
        b.begin_world()
        tanks.append(add_box_tank(b, inner_lo, inner_hi, wall=cfg.h))
        add_fluid_block(b, cfg, inner_lo, (nx, ny, fills[w]))
        if spares:
            add_spares(b, cfg, spares)
        b.end_world()
    model = b.finalize(device=device)
    return SPHScene(model, cfg, inner_lo, inner_hi, [(nx, ny, f) for f in fills], tanks)


def column_height(solver, state: newton.State, base_area: float) -> np.ndarray:
    """LIQ-2 SPH-volume height per world [m]: sum_i m_i / (max(rho_i, rho0_i) A_base) over active liquid
    (deficient surface particles count at rho0, so only compaction lowers it; SPEC §5.2)."""
    ratio, active = solver.density_ratio(state)
    m = solver.model
    mass = m.particle_mass.numpy().astype(np.float64)
    world = m.particle_world.numpy()
    out = np.zeros(solver.layout.world_count)
    rho0 = solver.layout.rest_density.numpy().astype(np.float64)
    vol = mass / (rho0 * np.maximum(ratio, 1.0))
    np.add.at(out, world[active], vol[active])
    return out / base_area


def run(solver, s0: newton.State, s1: newton.State, steps: int, dt: float, record=None):
    """Step ``steps`` times (ping-pong states); ``record(k, state, report)`` after each step. Returns the last state
    and the spare one."""
    for k in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        if record is not None:
            record(k, s0, solver.report())
    return s0, s1


def dam_break(width_cells: int = 20, height_cells: int = 40, depth_cells: int = 4, tank_cells: int = 80, *,
              cfg: DFSPHConfig | None = None, device: str | None = None) -> SPHScene:
    """Martin & Moyce style dam break: a block a x 2a (n^2 = 2 when height = 2 width) against the x_lo wall."""
    cfg = cfg or default_config()
    s = cfg.spacing
    inner_lo = (0.0, 0.0, 0.0)
    inner_hi = (tank_cells * s, depth_cells * s, (height_cells + 10) * s)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    tank = add_box_tank(b, inner_lo, inner_hi, wall=cfg.h)
    add_fluid_block(b, cfg, inner_lo, (width_cells, depth_cells, height_cells))
    b.end_world()
    model = b.finalize(device=device)
    return SPHScene(model, cfg, inner_lo, inner_hi, [(width_cells, depth_cells, height_cells)], [tank])


def floating_box(nx: int = 12, ny: int = 12, nz: int = 12, box_cells: int = 6, box_density: float = 500.0, *,
                 box_height_cells: float | None = None, cfg: DFSPHConfig | None = None,
                 device: str | None = None) -> SPHScene:
    """A free box (box_cells x box_cells x box_height_cells lattice cells, density ``box_density``) floating at its
    expected draft in a resting pool; SolverDFSPH owns and integrates the body (``integrate_bodies="dynamic"``, D5
    monolithic). Liquid lattice cells inside the submerged part of the box are left out."""
    cfg = cfg or default_config(integrate_bodies="dynamic")
    bz = float(box_cells if box_height_cells is None else box_height_cells)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    fb = add_floating_box(b, cfg, (nx, ny, nz), (box_cells, box_cells, bz), box_density)
    b.end_world()
    model = b.finalize(device=device)
    s = cfg.spacing
    return SPHScene(model, cfg, (0.0, 0.0, 0.0), (nx * s, ny * s, (nz + 2 * bz) * s), [(nx, ny, nz)], [fb["tank"]],
                    {"box": fb["body"], "draft": fb["draft"]})


def few_particles(walls: list[tuple[tuple[float, float, float], tuple[float, float, float]]], points,
                  velocity: tuple[float, float, float], *, r: float = 2.0e-3, gravity: float = 0.0,
                  device: str | None = None) -> SPHScene:
    """Liquid particles at ``points`` (one ``velocity``) among static box walls (center, half extents): the LIQ-8
    corner cases (slit, fast shot at a wall, a collapsed cluster)."""
    cfg = default_config(r)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    register_custom_attributes(b)
    b.begin_world()
    for center, half in walls:
        wall_box(b, center, half)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    add_fluid_points(b, cfg, pts, 1000.0, 1.0e-3, velocity)
    b.end_world()
    return SPHScene(b.finalize(device=device), cfg, tuple(pts.min(0)), tuple(pts.max(0)), [(len(pts), 1, 1)])


def one_particle(walls: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
                 point: tuple[float, float, float], velocity: tuple[float, float, float], *, r: float = 2.0e-3,
                 device: str | None = None) -> SPHScene:
    """One liquid particle at ``point`` with ``velocity`` among static box walls, no gravity (``few_particles``)."""
    return few_particles(walls, [point], velocity, r=r, device=device)


def collapsed_cluster(n: int = 4, spread: float = 1.0e-6, height: float = 4.0e-3, r: float = 2.0e-3,
                      device: str | None = None) -> SPHScene:
    """``n`` particles at rest within ``spread`` (Gaussian, seed 0) of one point ``height`` above an h-thick floor,
    with gravity: four of them make rho > rho0 with a vanishing density gradient (pressure.py, degenerate gradients)."""
    h = 4.0 * r
    pts = np.array([0.0, 0.0, height]) + spread * np.random.default_rng(0).standard_normal((n, 3))
    return few_particles([((0.0, 0.0, -0.5 * h), (0.05, 0.05, 0.5 * h))], pts, (0.0, 0.0, 0.0), r=r, gravity=9.81,
                         device=device)


def slit(gap: float, offset: float, speed: float, r: float = 2.0e-3, device: str | None = None) -> SPHScene:
    """A particle ``offset`` off the centre of a slit ``gap`` wide between two h-thick walls (x), moving at
    ``speed`` along +x (the pour-scene blow-up: cup and receiver walls 2.4-3.0 mm apart, r = 2 mm)."""
    t, side = 2.0 * r, 0.02
    walls = [((-t, 0.0, 0.0), (t, side, side)), ((gap + t, 0.0, 0.0), (t, side, side))]
    return one_particle(walls, (0.5 * gap + offset, 0.0, 0.0), (speed, 0.0, 0.0), r=r, device=device)


def closing_wedge_max_speed(theta_deg: float = 5.0, speed: float = 0.5, offset: tuple[float, float] = (-1.0e-3, 1.0e-3),
                            steps: int = 400, device: str | None = None) -> float:
    """The LIQ-8 pour escape on its own: a kinematic box (an 8 mm cup wall, bottom corner first, tilted ``theta_deg``)
    closes at ``speed`` on a static wall until its corner overlaps it by 1 mm, with one lone particle in the wedge
    (``offset`` (x, z) from the corner), gravity on, pour substep. Returns the particle's max speed [m/s]."""
    import math

    from alloy.solvers.sph.solver import SolverDFSPH

    dt, hx, hz, zc = 0.02 / 64, 4.0e-3, 10.0e-3, 0.02
    cfg = default_config(2.0e-3, integrate_bodies="none")
    th = math.radians(-theta_deg)
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), th)
    corner = (hx * math.cos(th) - hz * math.sin(th), -hx * math.sin(th) - hz * math.cos(th))  # (x, z) in the world
    x, x_stop = -3.0e-3 - corner[0], 1.0e-3 - corner[0]
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    wall_box(b, (4.0e-3, 0.0, 0.02), (4.0e-3, 0.02, 0.02))  # static wall, face at x = 0
    cup = b.add_body(xform=wp.transform((x, 0.0, zc), q), label="cup")
    wall_box(b, (0.0, 0.0, 0.0), (hx, 0.02, hz), body=cup)
    p0 = np.array([[corner[0] + x + offset[0], 0.0, zc + corner[1] + offset[1]]])
    add_fluid_points(b, cfg, p0, 1000.0, 1.0e-3, (0.0, 0.0, 0.0))
    b.end_world()
    m = b.finalize(device=device)
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()
    vmax = 0.0
    for _ in range(steps):
        v = speed if x < x_stop else 0.0
        a.body_q.assign([wp.transform((x, 0.0, zc), q)])
        a.body_qd.assign(np.array([[v, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        x = min(x + v * dt, x_stop)
        s.step(a, c, None, None, dt)
        a, c = c, a
        vmax = max(vmax, float(np.linalg.norm(a.particle_qd.numpy()[0])))
    return vmax


def lifting_floor_energy(speed: float = 0.05, settle: int = 200, steps: int = 400, n: int = 4, layers: int = 6,
                         device: str | None = None) -> dict:
    """A kinematic floor lifts a resting n x n x ``layers`` column (r = 2 mm) at ``speed`` inside static side walls,
    pour substep. Returns the liquid's energy change over the lift (KE + m g z) and the walls' work on it
    (``alloy_shape_work``) split into the moving floor and the static walls [J]."""
    from alloy.solvers.sph.solver import SolverDFSPH

    dt, r = 0.02 / 64, 2.0e-3
    hw, top = n * r, 4.0e-3  # inner half width; floor box top at z = 4 mm
    cfg = default_config(r, integrate_bodies="none")
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # static side walls, 4 mm thick, 60 mm high
        wall_box(b, (sx * (hw + r), sy * (hw + r), 0.03), (r if sx else hw + 2 * r, r if sy else hw + 2 * r, 0.03))
    floor = b.add_body(xform=wp.transform_identity(), label="floor")
    wall_box(b, (0.0, 0.0, 0.0), (hw, hw, top), body=floor)
    g = (np.arange(n) + 0.5) * 2.0 * r - hw
    pts = np.array([(x, y, top + r + 2.0 * r * k) for k in range(layers) for y in g for x in g])
    add_fluid_points(b, cfg, pts, 1000.0, 1.0e-3, (0.0, 0.0, 0.0))
    b.end_world()
    m = b.finalize(device=device)
    s = SolverDFSPH(m, cfg)
    moving = m.shape_body.numpy() >= 0
    mass = m.particle_mass.numpy()
    work = wp.zeros(m.shape_count, dtype=wp.float64, device=m.device)
    a, c = m.state(), m.state()

    def energy(st):
        q, v = st.particle_q.numpy(), st.particle_qd.numpy()
        return float(np.sum(mass * (0.5 * np.sum(v * v, axis=1) + 9.81 * q[:, 2])))

    z, e0, w_floor, w_static = 0.0, 0.0, 0.0, 0.0
    for k in range(settle + steps):
        v = speed if k >= settle else 0.0
        a.body_q.assign([wp.transform((0.0, 0.0, z), wp.quat_identity())])
        a.body_qd.assign(np.array([[0.0, 0.0, v, 0.0, 0.0, 0.0]], dtype=np.float32))
        z += v * dt
        if k == settle:
            e0 = energy(a)
        s.step(a, c, None, None, dt)
        a, c = c, a
        if k >= settle:
            s.alloy_shape_work(work)
            w = work.numpy()
            w_floor, w_static = w_floor + float(w[moving].sum()), w_static + float(w[~moving].sum())
    return {"de": energy(a) - e0, "w_floor": w_floor, "w_static": w_static, "lift": z}


def synthetic_pour_rig(geometry=None, rates: tuple[float, ...] = (10.0, 20.0, 40.0), theta_max_deg: float = 120.0,
                       duration_s: float = 14.0, noise_kg: float = 5.0e-4, noise_deg: float = 0.3,
                       trials_per_condition: int = 5, calib_per_condition: int = 2, seed: int = 0):
    """An R7 ``RigDataset`` from the quasi-static pour model (``pour.quasi_static_pour``) plus Gaussian noise:
    10 Hz scale series ``m_poured`` [kg], scalars onset angle [deg], poured mass [kg], spill [kg] (0 + noise)."""
    from alloy.bench.types import Condition, RigCard, RigDataset, RigTrial
    from alloy.core.ids import digest_of
    from alloy.solvers.sph.pour import PourGeometry, onset_angle_static, quasi_static_pour, retained_volume, tilt

    g = geometry or PourGeometry()
    rng = np.random.default_rng(seed)
    conds = tuple(Condition(f"tilt{r:g}", {"tilt_rate_deg_s": float(r), "theta_max_deg": theta_max_deg})
                  for r in rates)
    t = np.arange(0.0, duration_s + 1e-9, 0.1)
    trials = []
    for cond in conds:
        curve = quasi_static_pour(g, cond, t)
        top = float(tilt(cond, np.array([duration_s]))[0])
        poured = g.density * max(g.liquid_mass / g.density - float(retained_volume(g, np.array([top]))[0]), 0.0)
        for k in range(trials_per_condition):
            scalars = {"onset_angle": float(np.degrees(onset_angle_static(g)) + noise_deg * rng.standard_normal()),
                       "poured_mass": poured + noise_kg * float(rng.standard_normal()),
                       "spill": noise_kg * float(rng.standard_normal())}
            series = {"t": t, "m_poured": curve + noise_kg * rng.standard_normal(len(t))}
            trials.append(RigTrial(cond.id, k, "calib" if k < calib_per_condition else "valid", series, scalars))
    card = RigCard("R7.pour", f"synthetic-qs-{seed}", "synthetic/1", g.to_objects(),
                   {"m_poured": {"u95": 2.0 * noise_kg}, "onset_angle": {"u95": 2.0 * noise_deg}}, {}, {})
    sha = digest_of({"law": "R7.pour", "geometry": g.to_objects(), "rates": list(rates), "seed": seed})
    return RigDataset(card, conds, tuple(trials), sha)


def write_result(name: str, payload: dict) -> None:
    """Write ``payload`` as ``$ALLOY_WP9_RESULTS/<name>.json`` when the variable is set (results rows cite them)."""
    import json
    import os
    from pathlib import Path

    target = os.environ.get("ALLOY_WP9_RESULTS")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / f"{name}.json").write_text(json.dumps(payload, indent=1, default=float))


def drain_tank(cells: int = 40, depth: int = 50, hole_diameters: float = 8.0, *, cfg: DFSPHConfig | None = None,
               device: str | None = None) -> SPHScene:
    """LIQ-6: a square tank (``cells`` x ``cells`` lattice cells) filled ``depth`` cells deep, draining through a
    centred square thin-plate orifice of side ``hole_diameters`` particle diameters; particles 0.03 m below the
    floor leave through the escape box (retired, counted as escaped)."""
    from alloy.solvers.sph.build import add_orifice_tank

    base = cfg or default_config()
    s = base.spacing
    side, hole = cells * s, hole_diameters * s
    bounds = ((-0.05, -0.05, -0.03), (side + 0.05, side + 0.05, (depth + 10) * s))
    cfg = dataclasses.replace(base, world_bounds=bounds)
    cfg.validate()
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    tank = add_orifice_tank(b, (0.0, 0.0, 0.0), (side, side, (depth + 8) * s), wall=cfg.h, hole=hole)
    add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (cells, cells, depth))
    b.end_world()
    model = b.finalize(device=device)
    return SPHScene(model, cfg, (0.0, 0.0, 0.0), (side, side, (depth + 8) * s), [(cells, cells, depth)], [tank],
                    {"hole": hole})


def time_steps(solver, model: newton.Model, dt: float, steps: int, warmup: int = 20) -> dict:
    """Wall time per step after ``warmup`` eager steps: CUDA-graph replay of a two-step ping-pong when capture works
    (falls back to eager stepping otherwise). Returns {"wall_per_step_s", "mode", "steps"}."""
    import time

    a, b = model.state(), model.state()
    solver.prime(a, dt)
    for _ in range(warmup):
        solver.step(a, b, None, None, dt)
        a, b = b, a
    wp.synchronize_device(model.device)
    graph, mode = None, "eager"
    if wp.get_device(model.device).is_cuda:
        try:
            with wp.ScopedCapture(device=model.device) as cap:
                solver.step(a, b, None, None, dt)
                solver.step(b, a, None, None, dt)
            graph, mode = cap.graph, "graph"
        except Exception as exc:  # noqa: BLE001 - timing falls back to eager; the reason is reported
            mode = f"eager (capture failed: {type(exc).__name__}: {exc})"[:200]
    pairs = max(1, steps // 2)
    t0 = time.perf_counter()
    for _ in range(pairs):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            solver.step(a, b, None, None, dt)
            solver.step(b, a, None, None, dt)
    wp.synchronize_device(model.device)
    wall = time.perf_counter() - t0
    return {"wall_per_step_s": wall / (2 * pairs), "mode": mode, "steps": 2 * pairs}


def _liq8_save(path: str, rows: list, es, next_gen: int) -> None:
    """Checkpoint after a batch: finished rows, the CMA-ES state and its RNG (so a resumed run continues exactly)."""
    import os
    import pickle

    st = es.state
    blob = {"rows": rows, "next_gen": next_gen, "rng": es._rng.bit_generator.state,
            "cma": {k: getattr(st, k) for k in ("mean", "sigma", "cov", "p_sigma", "p_c", "generation", "best_x",
                                                "best_f", "history")}}
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(blob, fh)
    os.replace(tmp, path)  # atomic: a preemption mid-write leaves the previous checkpoint


def _liq8_load(path: str, es) -> tuple[list, int]:
    import pickle

    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    for k, v in blob["cma"].items():
        setattr(es.state, k, v)
    es._update_eigen()
    es._rng.bit_generator.state = blob["rng"]
    return blob["rows"], int(blob["next_gen"])


def liq8_campaign(worlds: int = 256, generations: int = 40, *, horizon_s: float = 2.0, knots: int = 8,
                  sigma0: float = 0.3, device: str = "cuda:0", seed: int = 0, progress=None,
                  checkpoint: str | None = None) -> dict:
    """LIQ-8 on the B0 pour scene: one B0 batch (stress phases tiled over ``worlds``) and ``generations`` B1 CMA-ES
    batches (WP7's EpisodeRunner, legalised knot traces), with SolverDFSPH's leak monitor on. CMA-ES maximises the
    bully badness plus a leak term (deepest step-path reach into a wall, in radii, and 10 per leaked particle), so
    the search hunts leaks. With ``checkpoint`` (a file path) every finished batch is saved and a rerun resumes after
    the last one (spot preemption). Returns totals and per-batch rows; ``progress(rows)`` after each batch."""
    import os

    from alloy.bully.actions import knots_to_trace, legalize, tick_count, trace_to_knots
    from alloy.bully.cma import BatchedCMAES
    from alloy.bully.replay import EpisodeRunner
    from alloy.bully.stress import named_stress_phases, tile_phases
    from alloy.bully.targets import get_target
    from alloy.bully.types import BullyLimits

    target = get_target("sph_pour")
    runner = EpisodeRunner(target, worlds, device, BullyLimits())
    sph = runner.sim.solver.solver("sph")
    mon = sph.enable_leak_monitor()
    rows: list[dict] = []
    sp = target.actions
    batch = tile_phases([t for _, t in named_stress_phases(target, horizon_s)], worlds)
    es = BatchedCMAES(trace_to_knots(batch[0], sp.low, sp.high, knots).ravel(), sigma0, lam=worlds, seed=seed)
    first_gen = 0
    if checkpoint and os.path.exists(checkpoint):
        rows, first_gen = _liq8_load(checkpoint, es)
    else:
        score = _liq8_play(runner, mon, batch, "B0", rows, progress)
        es = BatchedCMAES(trace_to_knots(batch[int(np.argmax(score))], sp.low, sp.high, knots).ravel(), sigma0,
                          lam=worlds, seed=seed)
        if checkpoint:
            _liq8_save(checkpoint, rows, es, 0)
    ticks = tick_count(horizon_s, sp.rate_hz)
    for gen in range(first_gen, generations):
        u = es.ask().reshape(worlds, knots, sp.dim)
        traces = legalize(knots_to_trace(u, sp.low, sp.high, ticks), sp.low, sp.high, sp.max_delta_per_tick,
                          sp.initial).trace
        es.tell(-_liq8_play(runner, mon, traces, f"B1-{gen}", rows, progress))
        if checkpoint:
            _liq8_save(checkpoint, rows, es, gen + 1)
    return _liq8_summary(rows, worlds, generations, horizon_s, sph, device)


def _liq8_play(runner, mon, traces: np.ndarray, label: str, rows: list, progress) -> np.ndarray:
    """One batch of episodes; appends its row and returns the CMA-ES score per world (higher = worse)."""
    mon.reset()
    res = runner.run(traces)
    c = mon.counts()
    rows.append({"batch": label, "episodes": int(traces.shape[0]), "leaks": int(c["leaks"].sum()),
                 "escaped": int(c["escaped"].sum()), "worlds_with_leak": int(np.count_nonzero(c["leaks"] + c["escaped"])),
                 "exit_bottom": int(c["exit_bottom"].sum()), "exit_side": int(c["exit_side"].sum()),
                 "exit_top": int(c["exit_top"].sum()),
                 "max_depth_r": float(c["max_depth_r"].max()), "s1": int(np.sum(res.severity == 1)),
                 "s2": int(np.sum(res.severity == 2)), "max_speed": float(np.max(res.monitors.get("max_speed", [0])))})
    if progress is not None:
        progress(rows)
    return np.asarray(res.badness, np.float64) + c["max_depth_r"] + 10.0 * (c["leaks"] + c["escaped"])


def _liq8_summary(rows: list, worlds: int, generations: int, horizon_s: float, sph, device: str) -> dict:
    keys = ("episodes", "leaks", "escaped", "worlds_with_leak", "exit_bottom", "exit_side", "exit_top", "s1", "s2")
    totals = {k: int(sum(r[k] for r in rows)) for k in keys}
    totals["max_depth_r"] = max(r["max_depth_r"] for r in rows)
    totals["max_speed"] = max(r["max_speed"] for r in rows)
    return {"totals": totals, "rows": rows, "worlds": worlds, "generations": generations, "horizon_s": horizon_s,
            "particles_per_world": int(sph.layout.sph_count // max(worlds, 1)), "device": device}
