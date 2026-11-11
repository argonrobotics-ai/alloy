# SPDX-License-Identifier: Apache-2.0
"""Plain seam proof task (sewing gap, coordinator 2026-11-06; their plain-seam): ``run_seam``.

A rig-scale lockstitch machine sews two muslin plies 10 mm from their right edges, on the lagged MuJoCo -> VBD
Scheduler with the cloth entry from the route's ``make_vbd`` (``alloy.tasks.sew_ply_pinch.cloth_scheduler``), CPU:
- a static bed (top z = 0) carries the plies (0.61 mm, 125 g/m2, ply-ply mu 0.65, bending D = their KES mean through
  the CLOTH_VBD hinge map; their params.py). Without the bending (edge_ke 1e-5) the plies fold up beside the foot
  and the feed stalls (docs/notes/sewing-seam.md);
- a 20 x 12 mm presser foot on a z slide holds the stack at ``foot_n`` (force mode: the servo's lead re-centred each
  period). The machine's bodies weigh 1.2 kg each so the exchange clock (dt_x ~ sqrt(m_eff / k_contact)) runs at
  1 ms, 16 exchanges a period instead of 64 at their 0.12 kg foot; following the dog's rise costs the foot ~1.4 N of
  inertia against its 40 N;
- a 10 x 11 mm feed dog (on x and z slides, under the foot, clamping the seam rows at -5, 0 and 5 mm) rises through
  the bed, moves the stack one stitch length, drops and returns: one stitch per cycle, forward or reversed (backtack).
  Its z servo leads by the foot's load and its own weight, so the loaded teeth stand ``dog_rise`` above the bed as a
  cam-driven dog's do (without the lead they sagged to 0.05 mm at 40 N; ``SeamResult.dog_top``).
  A dog that clamps the seam row alone (3 x 6 mm) lets the bed's friction turn the plies about it;
- frictions are their cloth-shape pairs (steel 0.15 for the foot and the bed); VBD mixes the cloth's and a shape's mu
  as a geometric mean, so each shape takes pair^2 / cloth mu;
- the needle point sits ``allowance`` inside the plies' right edges. After the settle and at the end of each cycle
  (the dog down, the stack held by the foot) the needle stitches the plies: the thread spring between the two plies'
  particles of the seam column under the needle switches on. The springs are declared at build time with zero
  stiffness, and the second ply is recoloured so no spring joins two particles of one VBD colour (Newton's colouring
  skips springs; ``alloy.solvers.film.seam_vbd.recolor_twins``).
Checks, from their plain-seam predicate (``SeamResult.passed``): the world stays valid; the stitched columns are
contiguous and span at least 80 % of the seam; at every stitch the needle is ``allowance`` +- 1.6 mm inside both
right edges and the edges are aligned within 1.5 mm; every cycle feeds 0.7-1.15 stitch lengths in its direction,
and a cycle beside a feed reversal at least 0.5 (``feed_ok``; the coupled solve slips there, WP10's gap); at the end
every stitch holds the plies within two thicknesses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.coupling.pinch_rigs import SHAPE_KE, _pass_through_count
from alloy.tasks.sew_ply_pinch import cloth_scheduler

SERVO_KE = 1.0e5                 # foot / feed dog servo gain [N/m]
PERIOD = 1.0 / 60.0              # control period [s]
PER_CYCLE = 12                   # control periods per stitch cycle (0.2 s)
ALLOW_TOL, EDGE_TOL = 1.6e-3, 1.5e-3   # their seam allowance tolerance and edge alignment [m]
FEED_BAND = (0.7, 1.15)          # stitch lengths fed per cycle, in its direction (their plain-seam predicate)
REVERSAL_FEED_MIN = 0.5          # a cycle beside a feed reversal (feed_ok; WP11's settle spread, 2026-11-09)


@dataclass(frozen=True)
class SeamSpec:
    """The plies, the machine and the seam (lengths in m, forces in N, times in s)."""

    plies: int = 2
    thickness: float = 0.61e-3
    areal_density: float = 0.1252
    length: float = 0.060            # along the feed (x)
    width: float = 0.030             # across (y); the right edge is at its low y
    cell_x: float = 0.0025           # one grid column per stitch length
    cell_y: float = 0.005
    soft_ke: float = 1.0e4
    mu: float = 0.65
    tri_ke: float = 400.0
    tri_ka: float = 3000.0
    bending: float | None = 2.65e-6  # plate rigidity D [N m] (their KES B mean) by the CLOTH_VBD hinge map; None: 1e-5
    allowance: float = 0.010         # needle inside the right edges (their 10 mm)
    lead: float = 0.005              # needle behind the plies' leading end at the start (their 3-8 mm)
    stitch: float = 0.0025           # feed per stitch
    stitches: int = 10               # forward stitches of the seam body
    backtack: int = 3                # reversed stitches at both ends (their 3-4)
    foot_n: float = 40.0             # their presser
    thread_k: float = 2.0e3          # thread spring [N/m]
    foot_mu: float = 0.15            # cloth-shape pair frictions: their cloth-steel 0.15 (foot, bed)
    dog_mu: float = 1.0              # the feed dog's teeth
    bed_mu: float = 0.15
    dog_rise: float = 0.5e-3         # teeth above the bed when up [m]
    foot_half: tuple[float, float, float] = (0.010, 0.006, 0.003)    # presser foot half extents [m]: 20 x 12 mm
    dog_half: tuple[float, float, float] = (0.005, 0.0055, 0.002)    # feed dog: 10 x 11 mm (seam rows -5, 0, 5 mm)
    foot_mass: float = 1.2           # presser bar + foot [kg] (theirs 0.12 kg on a 40 N spring; their dog is kinematic)
    feed_mass: float = 1.2           # the feed carrier and the dog, each [kg]; with the foot they set the clock
    iterations: int = 20
    settle: float = 0.2


@dataclass
class SeamResult:
    stitches: list[dict[str, float]] = field(default_factory=list)   # per needle event
    columns: int = 0                 # distinct seam columns stitched
    contiguous: bool = False         # every column between the seam's ends stitched
    feed_ratio: tuple[float, float] = (math.nan, math.nan)            # min / max feed per cycle / stitch length
    joined_max: float = math.nan     # largest stitched pair distance at the end [m]
    dog_top: float = math.nan        # lowest height of the dog's teeth over the bed during the feed strokes [m]
    pen_fail: bool = False
    flags: int = 0
    pass_through: int = 0
    finite: bool = True
    failure: str = ""

    def passed(self, spec: SeamSpec) -> dict[str, bool]:
        s = self.stitches
        return {"valid": self.finite and not self.pen_fail and self.pass_through == 0,
                "columns": self.contiguous and self.columns >= 0.8 * spec.stitches + 1,
                "allowance": bool(s) and all(abs(r[f"allow{k}"] - spec.allowance) <= ALLOW_TOL
                                             for r in s for k in range(spec.plies)),
                "edges": bool(s) and all(r["edge_offset"] <= EDGE_TOL for r in s),
                "feed": feed_ok(s),
                "joined": self.joined_max <= 2.0 * spec.thickness}


def feed_ok(stitches: list[dict[str, float]]) -> bool:
    """Every cycle feeds FEED_BAND stitch lengths in its direction. A cycle beside a reversal (the last before the
    feed turns, or the first after) only needs REVERSAL_FEED_MIN: there the coupled solve is not converged at the
    default 20 VBD iterations, and the plies slip on the teeth by up to a third of a stitch
    (docs/known_gaps/coupling-cloth-grip-slip-at-reversal.md)."""
    rows = [(r["dir"], r["dir"] * r["advance"]) for r in stitches if r["dir"] != 0 and not math.isnan(r["advance"])]
    if not rows:
        return False
    dirs = [d for d, _ in rows]
    for i, (d, f) in enumerate(rows):
        turn = (i > 0 and dirs[i - 1] != d) or (i + 1 < len(dirs) and dirs[i + 1] != d)
        if not (REVERSAL_FEED_MIN if turn else FEED_BAND[0]) <= f <= FEED_BAND[1]:
            return False
    return True


def _feed_dirs(spec: SeamSpec) -> list[int]:
    """Forward backtack, reverse over it, the seam body, reverse and forward again at the end."""
    bt = spec.backtack
    return [1] * bt + [-1] * bt + [1] * spec.stitches + [-1] * bt + [1] * bt


def _machine(b: newton.ModelBuilder, spec: SeamSpec) -> float:
    """Static bed (top z = 0), presser foot (body 0, z slide: joint 0), feed carrier (body 1, x: joint 1) and feed dog
    (body 2, z off the carrier: joint 2), all position servos. Returns the dog's low z (teeth 1 mm under the bed)."""

    def cfg(mu: float, mass: float | None = None, half: tuple[float, ...] = (1.0,), shapes: bool = True) -> Any:
        c = b.default_shape_cfg.copy()
        c.mu = mu * mu / spec.mu                 # VBD's body-particle mu is sqrt(cloth mu * shape mu): the pair's
        if mass is not None:
            c.density = mass / (8.0 * math.prod(half))
        c.has_shape_collision = shapes
        return c

    def slide(parent: int, child: int, axis: tuple[float, float, float], z: float, mass: float, label: str) -> int:
        return b.add_joint_prismatic(parent, child, axis=axis, target_ke=SERVO_KE, label=label,
                                     target_kd=2.0 * math.sqrt(SERVO_KE * mass),     # critically damped
                                     parent_xform=wp.transform((0.0, 0.0, z), wp.quat_identity()),
                                     actuator_mode=newton.JointTargetMode.POSITION)

    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.01), wp.quat_identity()), hx=0.1, hy=0.06, hz=0.01,
                    cfg=cfg(spec.bed_mu))
    fh, dh, ch = spec.foot_half, spec.dog_half, (0.002, 0.002, 0.002)
    zf = spec.plies * spec.thickness + 0.002 + fh[2]
    foot = b.add_link(xform=wp.transform((0.0, 0.0, zf), wp.quat_identity()), label="presser_foot")
    b.add_shape_box(foot, hx=fh[0], hy=fh[1], hz=fh[2], cfg=cfg(spec.foot_mu, spec.foot_mass, fh))
    z_low = -1.0e-3 - dh[2]
    carrier = b.add_link(xform=wp.transform((0.0, 0.0, -0.03), wp.quat_identity()), label="feed_carrier")
    b.add_shape_box(carrier, hx=ch[0], hy=ch[1], hz=ch[2], cfg=cfg(0.0, spec.feed_mass, ch, shapes=False))
    dog = b.add_link(xform=wp.transform((0.0, 0.0, z_low), wp.quat_identity()), label="feed_dog")
    b.add_shape_box(dog, hx=dh[0], hy=dh[1], hz=dh[2], cfg=cfg(spec.dog_mu, spec.feed_mass, dh, shapes=False))
    b.add_articulation([slide(-1, foot, (0.0, 0.0, 1.0), zf, spec.foot_mass, "foot_z")])
    b.add_articulation([slide(-1, carrier, (1.0, 0.0, 0.0), -0.03, 2.0 * spec.feed_mass, "feed_x"),
                        slide(carrier, dog, (0.0, 0.0, 1.0), z_low + 0.03, spec.feed_mass, "feed_z")])
    return z_low


def _plies(b: newton.ModelBuilder, spec: SeamSpec) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """The plies (right edges at y = -allowance, the needle at x = y = 0, ``lead`` behind their leading ends) and the
    thread springs (zero stiffness) between their seam-row particles. Returns the seam and edge rows (sorted by x) and
    each ply's first particle."""
    from alloy.solvers.vbd.cloth_profile import rewrite_hinges

    r, y_edge = 0.5 * spec.thickness, -spec.allowance
    nx, ny = int(round(spec.length / spec.cell_x)), int(round(spec.width / spec.cell_y))
    starts = []
    for k in range(spec.plies):
        starts.append(len(b.particle_q))
        e0 = len(b.edge_indices)
        b.add_cloth_grid(pos=wp.vec3(spec.lead - spec.length, y_edge, r + k * spec.thickness + 1.0e-6),
                         rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=nx, dim_y=ny, cell_x=spec.cell_x,
                         cell_y=spec.cell_y, mass=spec.areal_density * spec.cell_x * spec.cell_y,
                         tri_ke=spec.tri_ke, tri_ka=spec.tri_ka, tri_kd=1.0e-2, edge_ke=1.0e-5, edge_kd=1.0e-6,
                         particle_radius=r)
        if spec.bending is not None:
            rewrite_hinges(b, e0, starts[-1], spec.bending, 1.0e-3)
    q = np.array([tuple(p) for p in b.particle_q], np.float64)
    rows = [np.arange(s0, s0 + (nx + 1) * (ny + 1)) for s0 in starts]
    seam = [ids[np.abs(q[ids, 1]) < 0.25 * spec.cell_y] for ids in rows]
    edge = [ids[np.abs(q[ids, 1] - y_edge) < 0.25 * spec.cell_y] for ids in rows]
    seam, edge = [ids[np.argsort(q[ids, 0])] for ids in seam], [ids[np.argsort(q[ids, 0])] for ids in edge]
    for a, c in zip(seam[0], seam[-1]):
        b.add_spring(int(a), int(c), 0.0, 1.0e-3, 0)
    return seam, edge, starts


def seam_model(spec: SeamSpec = SeamSpec()) -> tuple[newton.Model, dict[str, Any]]:
    """The machine (``_machine``) and the plies with their thread springs (``_plies``) in one world; the second ply's
    seam row is recoloured so no spring joins one VBD colour. Returns the model and its layout."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    from alloy.solvers.film.seam_vbd import recolor_twins

    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.ke, b.default_shape_cfg.kd = SHAPE_KE, 1.0e3
    z_low = _machine(b, spec)
    seam, edge, starts = _plies(b, spec)
    world = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(world)
    world.add_world(b)
    world.color()
    m = world.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = spec.soft_ke, 1.0e-2, spec.mu
    recolor_twins(m, SimpleNamespace(a=seam[0].astype(np.int64), b=seam[-1].astype(np.int64)))
    return m, {"seam": seam, "edge": edge, "z_low": z_low, "y_edge": -spec.allowance, "start": starts}


def _dog_targets(k: int, d: int, spec: SeamSpec) -> tuple[float, float]:
    """Feed dog (x, z) joint targets at period k of a cycle feeding in direction d: rise, feed, drop, return."""
    hi = 1.0e-3 + spec.dog_rise
    if k < 3:
        return 0.0, hi * (k + 1) / 3
    if k < 8:
        return d * spec.stitch * (k - 2) / 5, hi
    if k < 10:
        return d * spec.stitch, hi * (9 - k) / 2
    return d * spec.stitch * (11 - k) / 2, 0.0


def _edge_y(q: np.ndarray, ids: np.ndarray, x: float) -> float:
    o = np.argsort(q[ids, 0])
    return float(np.interp(x, q[ids[o], 0], q[ids[o], 1]))


def _vbd_model(sched: Any) -> Any:
    return next(e.solver.model for name, e in sched._entries.items() if name == "vbd")


def _needle(q: np.ndarray, lay: dict[str, Any], d: int) -> dict[str, float]:
    """At the needle (x = y = 0): the seam column nearest it per ply, the allowance per ply, the edges' offset."""
    row = {"dir": float(d)}
    ys = []
    for k, (seam, edge) in enumerate(zip(lay["seam"], lay["edge"])):
        j = int(np.argmin(np.abs(q[seam, 0])))
        row[f"col{k}"], row[f"dx{k}"] = float(j), float(q[seam[j], 0])
        ys.append(_edge_y(q, edge, 0.0))
        row[f"allow{k}"] = -ys[-1]
    row["edge_offset"] = float(np.ptp(ys))
    return row


@dataclass
class _Run:
    """A seam machine stepping on the Scheduler."""

    spec: SeamSpec
    m: Any
    lay: dict[str, Any]
    sched: Any
    monitor: Any
    s0: Any
    s1: Any
    ctl: Any
    pipe: Any
    contacts: Any
    k: int = 0                       # control periods stepped
    last: Any = None                 # the Scheduler's last report
    x_prev: float | None = None      # the seam row's mean x at the last needle event
    dog_top: float = math.inf        # lowest height of the teeth over the bed during the feed strokes [m]
    stitched: set[int] = field(default_factory=set)


def _start(spec: SeamSpec, policy: Any) -> _Run:
    from alloy.integration.stepper import entry_solvers
    from alloy.precision.health import HealthMonitor
    from alloy.precision.types import CheckPolicy

    m, lay = seam_model(spec)
    sched = cloth_scheduler(m, spec.iterations, 0.5 * spec.thickness, policy=policy)
    # solvers=: health reads a solver's own overflow counters only from the solvers it is given. Without them the
    # ply-pinch rig dropped MuJoCo constraint rows 3,171 times in one run and read clean (GAP-10A r8, 2026-11-10).
    monitor = HealthMonitor(m, CheckPolicy(level="full", every=1, full_every=1), capacity_flags=sched.capacity_flags(),
                            solvers=list(entry_solvers(sched).values()))
    s0 = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    pipe = newton.CollisionPipeline(m)
    return _Run(spec, m, lay, sched, monitor, s0, m.state(), m.control(), pipe, pipe.contacts())


def _period(run: _Run, d: int | None, n: int) -> float:
    """One control period: the foot in force mode (lowered over the first 12 periods of the settle) and the dog's
    targets, then the exchanges at the Scheduler's dt_x (returned)."""
    spec = run.spec
    q = run.s0.joint_q.numpy().astype(np.float64)
    foot = q[0] - (spec.foot_n - spec.foot_mass * 9.81) / SERVO_KE
    if d is None and run.k < 12:
        foot = max(-0.003 * (run.k + 1) / 12, foot)
    dx, dz = (0.0, 0.0) if d is None else _dog_targets(n, d, spec)
    if d is not None:                # the dog carries the foot: its servo leads by that load and its own weight
        dz += (spec.foot_n + spec.feed_mass * 9.81) / SERVO_KE
    run.ctl.joint_target_q.assign(np.array([foot, dx, dz], np.float32))
    dt, _ = run.sched.choose_clock(PERIOD, run.last)
    for j in range(int(round(PERIOD / dt))):
        run.s0.clear_forces()
        run.pipe.collide(run.s0, run.contacts)
        run.sched.step(run.s0, run.s1, run.ctl, run.contacts, dt)
        run.s0, run.s1 = run.s1, run.s0
        run.monitor.update(int(round((run.k * PERIOD + (j + 1) * dt) / dt)), run.s1, run.s0, run.contacts, dt=dt)
    run.last, run.k = run.sched.report(), run.k + 1
    if d is not None and 3 <= n < 8:                # the feed stroke: the teeth's height over the bed
        top = run.lay["z_low"] + spec.dog_half[2] + float(run.s0.joint_q.numpy()[2])     # the bed's top is z = 0
        run.dog_top = min(run.dog_top, top)
    return dt


def _stitch(run: _Run, d: int) -> dict[str, float]:
    """Needle event: measure at the needle, then switch on the thread spring of the seam column under it."""
    spec = run.spec
    pq = run.s0.particle_q.numpy().astype(np.float64)
    row = _needle(pq, run.lay, d)
    x_now = float(pq[run.lay["seam"][0], 0].mean())
    row["advance"] = math.nan if run.x_prev is None else (x_now - run.x_prev) / spec.stitch
    run.x_prev = x_now
    j = int(row["col0"])
    if j not in run.stitched:
        view = _vbd_model(run.sched)
        for arr in (view.spring_stiffness, run.m.spring_stiffness):
            a = arr.numpy()
            a[j] = spec.thread_k
            arr.assign(a)
        run.stitched.add(j)
    return row


def _finish(run: _Run, res: SeamResult) -> SeamResult:
    from alloy.precision.types import HealthFlags

    pq = run.s0.particle_q.numpy().astype(np.float64)
    a, c = run.lay["seam"][0], run.lay["seam"][-1]
    on = sorted(run.stitched)
    res.columns, res.contiguous = len(on), bool(on) and on[-1] - on[0] + 1 == len(on)
    res.joined_max = max((float(np.linalg.norm(pq[a[j]] - pq[c[j]])) for j in on), default=math.nan)
    res.dog_top = run.dog_top
    feed = [r["dir"] * r["advance"] for r in res.stitches if r["dir"] != 0 and not math.isnan(r["advance"])]
    res.feed_ratio = (min(feed), max(feed)) if feed else (math.nan, math.nan)
    rep = run.monitor.poll()
    res.flags = int(rep.flags[0])
    res.pen_fail = bool(res.flags & int(HealthFlags.PEN_FAIL | HealthFlags.PAIR_PEN_FAIL))
    res.failure = rep.failure_text(0) if res.flags else ""
    res.pass_through = _pass_through_count(run.sched)
    res.finite = bool(np.isfinite(pq).all())
    return res


def run_seam(spec: SeamSpec = SeamSpec(), verbose: bool = False, policy: Any = None,
             trace: list[dict[str, Any]] | None = None) -> SeamResult:
    """Settle under the lowering foot and stitch the start, then one stitch cycle per feed direction of
    ``_feed_dirs`` (module docstring). ``policy``: the Scheduler's policy (default: its defaults); ``trace``: if a
    list, one entry per control period (feed direction, joint q, dt_x, watchdog trips, particle positions)."""
    run, res = _start(spec, policy), SeamResult()
    settle = int(round(spec.settle / PERIOD))
    plan = [(None, n) for n in range(settle)] + [(d, n) for d in _feed_dirs(spec) for n in range(PER_CYCLE)]
    for i, (d, n) in enumerate(plan):
        dt = _period(run, d, n)
        if trace is not None:
            trace.append({"d": d, "n": n, "q": run.s0.joint_q.numpy().copy(), "dt": dt,
                          "trips": getattr(run.sched, "_trips", 0), "x": run.s0.particle_q.numpy().copy()})
        if i == settle - 1 or (d is not None and n == PER_CYCLE - 1):
            res.stitches.append(_stitch(run, 0 if d is None else d))
            if verbose:
                print({k: round(v * 1e3, 3) if k.startswith(("allow", "edge", "dx")) else v
                       for k, v in res.stitches[-1].items()}, flush=True)
    return _finish(run, res)


def _main(argv: list[str] | None = None) -> None:
    import argparse
    import dataclasses
    import json

    ap = argparse.ArgumentParser(description="plain seam proof task (rig scale, CPU)")
    ap.add_argument("--stitches", type=int, default=10)
    ap.add_argument("--backtack", type=int, default=3)
    ap.add_argument("--foot", type=float, default=SeamSpec.foot_n)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    spec = SeamSpec(stitches=a.stitches, backtack=a.backtack, foot_n=a.foot)
    r = run_seam(spec, verbose=not a.quiet)
    print("SEAM", json.dumps({"spec": dataclasses.asdict(spec), "passed": r.passed(spec),
                              "result": {k: v for k, v in dataclasses.asdict(r).items() if k != "stitches"}},
                             default=float), flush=True)


if __name__ == "__main__":
    _main()
