# SPDX-License-Identifier: Apache-2.0
"""task concrete_joints: Gym-Anything's tool-joints, a surface-flow concrete proof.

"Tool the joints of the new block concave. Fill anything short first." (``concrete_finishing/task.py:29``; spec
``[internal note]`` §2). A block infill, four courses of three CMU in running
bond, 3/8 in joints of Type N mortar struck flush; the courses were laid 45, 37, 28 and 20 min before, so their
mortar starts at 18.0, 10.4, 5.1 and 2.5 kPa. One head joint is 3 mm short over 60 mm; crumbs of squeeze-out sit on
the faces next to the joints. The start is theirs cell for cell (``concrete_joints_start``, 1.5 mm grid).

Model: the wall is laid flat (z out of the face, faces at 0): at 2.5-20 kPa the mortar's own weight moves nothing (a
heap would need 3.83 tau / (rho g) > 0.5 m), so gravity's direction does not matter, and the scripted carriage
presses the tools with their total force. One ``surface_flow`` plane chart over the window; its fixed ``base`` is the
block faces (0) and, in the joints, ``FILM_DEPTH`` below them (deeper than any tool reaches); the paste above is the
film at the bulk's yield. A privileged carriage (``concrete_tools``; no robot) works the BJCC12 jointer, a 12.7 mm
rod (a ``roller`` sole on WP15's implicit pad, sliding along its axis), along every joint at ``force`` (their fixture:
20 N): resting on the arrises, the rod's underside is a circular arc 2.1 mm deep at the joint centre, so the film
above it is pushed on along the joint and fills what is short. Courses are tooled in order of age while their
mortar is thumbprint hard (5-20 kPa): the three oldest at once, then a wait (the paste ages, the tools parked) until
the youngest reaches 5 kPa. The process layer (``concrete_paste``, masonry mortar) ages the mortar, compacts it under
the sole and records the bulk yield at last working. Verdict: their five criteria (``concrete_metrics.tool_joints``).

What the verdict judges (audit AB-gym-7; ``concrete_ports.TOOL_JOINTS``): four criteria as theirs and ``smear`` in
part: its off-wall part reads the film's books (``concrete_joints_books.off_wall``) and fails over 2e-8 m³, but a pass
is not counted (laid flat, the lifted trowel drops no gob). How the scene differs from theirs is in the verdict.

Run: ``python -m alloy.tasks.concrete_joints [--window x0 x1 z0 z1] [--device cuda:0] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, replace
import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store
from alloy.plugins.surface_flow.supply_spec import Drain
from alloy.tasks import concrete_joints_schedule as SCH
from alloy.tasks import concrete_joints_start as JS
from alloy.tasks import concrete_joints_strips as JSTR
from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_metrics as MX
from alloy.tasks import concrete_verdict as CV
from alloy.tasks.concrete_joints_books import OFF_BANDS, OFF_KEYS, off_wall, window_differs  # noqa: F401
from alloy.tasks.concrete_paste import ConcretePaste, SoleWork
from alloy.tasks.concrete_ports import TOOL_JOINTS
from alloy.tasks.concrete_tools import FORCE, HEIGHT, Stroke, ToolCarriage, ToolProgram

log = logging.getLogger("alloy.tasks.concrete_joints")

FILM_DEPTH = 0.005                      # m: joint paste above this depth below the faces is film (the rod reaches 2.1)
ROD_R, ROD_L, ROD_MASS = 0.00635, 0.150, 0.12          # BJCC12 jointer (tools.py:149-161; params.py:148)
JOINTER = SoleWork("jointer", a_tool=3.0e-5, dens=0.6, seal=0.0, half=(ROD_L / 2 - ROD_R, 0.98 * ROD_R))
TROWEL_HALF, TROWEL_MASS = (0.0635, 0.0255, 0.00075), 0.12   # margin trowel 52D blade (tools.py:136-146)
TROWEL_BODY = (0.0635, 0.0255, 0.02), 0.25   # its body: the blade with the handle's bulk (a 1.5 mm plate alone is so
                                              # light about its long axis that the film's capped drag kicks it unstable)
TROWEL = SoleWork("trowel", a_tool=1.0e-4, dens=0.5, seal=0.3, half=TROWEL_HALF[:2])
TROWEL_CAP = 1.5e-4                     # m³: what it carries (params.py:160, CARRY_CAP)
TROWEL_GAP = 5.0e-5                     # m: its underside over the faces while it takes the tags
TROWEL_DOWN = 0.3                       # s: it comes down onto (and lifts off) a strip's end
SIDE_STOP = ROD_L / 2 - ROD_R + 0.002   # m: the rod's bearing half length and 2 mm (ToolJoints.side_stop)
PARK = (0.0, 0.0, 0.10)                 # m: above the wall's centre
PARK_T = (0.0, 0.0, 0.16)
PERIOD = 0.02                           # s: one exchange


@dataclass(frozen=True)
class Lane:
    """One jointer stroke along a joint: wall coordinates (x, z) of its ends [m] and the course it tools; ``one_way``:
    every pass runs p0 to p1 (``concrete_joints_schedule.off_older``)."""

    p0: tuple[float, float]
    p1: tuple[float, float]
    course: int
    one_way: bool = False


@dataclass
class ToolJoints:
    """The task's settings: seed, site temperature, the wall window (x, z ranges [m]; None = the whole infill) and the
    jointer's strokes."""

    seed: int = 0
    temperature: float = 21.1
    window: tuple[float, float, float, float] | None = None
    force: float = 20.0                  # N total on the rod (their fixture)
    speed: float = 0.10                  # m/s along a joint
    trowel_speed: float = 0.10           # m/s along a trowel strip
    air_speed: float = 0.30              # m/s for moves off the work
    passes: int = 4                      # strokes per joint
    trowel: bool = True                  # take the tags off the faces with the margin trowel after the tooling
    run_in: float = 0.02                 # m: the rod sets down and lifts while moving, this far beyond the lane ends
    overrun: float = 0.10                # m: a trowel strip runs this far past its lane's ends where no crossing joint
                                         # stops it (the rod's leading end lays its roll up to ROD_L/2 + run_in,
                                         # 95 mm, past the lane)
    wait_margin: float = 30.0            # s of age past the youngest course's 5 kPa before tooling it
    hard_margin: float = 12.0            # s before a course's mortar passes 20 kPa that its last stroke ends
    speed_max: float = 0.20              # m/s: the jointer's speed is raised up to this to meet hard_margin
    side_stop: float | None = SIDE_STOP  # m: a side stroke's lower end above its lane's, off the older course below
                                         # (None: side joints are tooled like the others)
    omega_rot: float = 400.0             # rad/s: the carriage's attitude servo (the arris friction's pitch moment)
    extra: dict = field(default_factory=dict)


# --- geometry -------------------------------------------------------------------------------------------------------

def joint_lanes() -> list[Lane]:
    """Every joint's centre line (wall coordinates), in tooling order: per course, the head joints of its row (top to
    bottom) and then the bed under it, and last the top bed (course 3), as ``concrete_joints_start._joints`` lays
    them."""
    (x0, x1), j, (bl, _, bh), z0 = JS.INFILL_X, JS.JOINT_W, JS.BLOCK, JS.INFILL_Z0
    lanes = []
    for c in range(JS.COURSES):
        zb = z0 + c * (bh + j)
        heads = [x0, x1 - j]
        if c % 2 == 0:
            heads += [x0 + j + bl + k * (bl + j) for k in range(2)]
        else:
            half = (bl - j) / 2
            heads += [x0 + j + half, x0 + j + half + j + bl, x0 + j + half + j + 2 * bl + j]
        lanes += [Lane((hx + j / 2, zb + j + bh), (hx + j / 2, zb + j), c) for hx in sorted(heads)]
        lanes.append(Lane((x0, zb + j / 2), (x1, zb + j / 2), c))       # then the bed under them: the heads' rolls
    z1 = z0 + JS.COURSES * (bh + j) + j                                 # run down into it and on to its ends
    lanes.append(Lane((x0, z1 - j / 2), (x1, z1 - j / 2), JS.COURSES - 1))
    return lanes


def _clip(lane: Lane, win: tuple[float, float, float, float]) -> Lane | None:
    """The part of an axis-aligned lane inside the window (None if it misses)."""
    x0, x1, z0, z1 = win
    (a, b), (c, d) = lane.p0, lane.p1
    if a == c:                                              # vertical
        if not x0 <= a < x1:
            return None
        lo, hi = max(min(b, d), z0), min(max(b, d), z1)
        return None if hi <= lo else replace(lane, p0=(a, hi if d < b else lo), p1=(a, lo if d < b else hi))
    if not z0 <= b < z1:
        return None
    lo, hi = max(min(a, c), x0), min(max(a, c), x1)
    return None if hi <= lo else replace(lane, p0=(lo, b), p1=(hi, b))


@dataclass(frozen=True)
class Wall:
    """The run's wall: the start fields on the window, its size [m] and the map from wall (x, z) to world (x, y)."""

    f: JS.Infill
    size: tuple[float, float]

    @property
    def cells(self) -> tuple[int, int]:
        return self.f.joint.shape

    def world(self, x: float, z: float) -> tuple[float, float]:
        return (x - self.f.origin[0] - self.size[0] / 2, z - self.f.origin[1] - self.size[1] / 2)


def wall(task: ToolJoints, seed: int | None = None) -> Wall:
    full = JS.infill(task.seed if seed is None else seed, task.temperature)
    f = full if task.window is None else JS.crop(full, task.window[:2], task.window[2:])
    n = f.joint.shape
    return Wall(f, (n[0] * f.cell, n[1] * f.cell))


def split(f: JS.Infill) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(base, film, paste mask): the base is the faces and, in the joints, FILM_DEPTH below them; the film is the paste
    above it (the joint fill, the crumbs)."""
    base = np.where(f.joint, np.maximum(f.base, -FILM_DEPTH), 0.0)
    return base, np.maximum(f.top - base, 0.0), f.joint | f.crumbs


# --- scene, strokes, plug-ins ----------------------------------------------------------------------------------------

ROD_MU = 0.35                           # tool friction (tools.py:22)
RIGID_DT = 1.25e-3                      # s: the rigid step (the harness CPU route)


def _stiff(mu: float) -> newton.ModelBuilder.ShapeConfig:
    """Rod and face contact gains realising ~0.4 MN/m on the rod (alloy.scene.native_gains, stable at the rigid step):
    with Newton's default kd = 0 MuJoCo falls back to solref 0.02 s and 20 N sinks the rod 0.7 mm into the arrises."""
    from alloy.scene.native_gains import native_gains  # noqa: PLC0415

    ke, kd = native_gains(1.0e6, ROD_MASS, 0.0, dt=RIGID_DT)
    return newton.ModelBuilder.ShapeConfig(mu=mu, ke=ke, kd=kd)


def _add_rod(b: newton.ModelBuilder) -> None:
    """The jointer: a free body with the rod as a capsule along its x (BJCC12: 12.7 x 150 mm, 0.12 kg). It rests on the
    blocks' arrises (rigid contact), which set the tooled depth; the pad works the film."""
    body = b.add_body(xform=wp.transform(wp.vec3(*PARK), wp.quat_identity()), label="jointer")
    hh = ROD_L / 2 - ROD_R
    vol = math.pi * ROD_R**2 * 2 * hh + 4.0 / 3.0 * math.pi * ROD_R**3
    along_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)
    cfg = _stiff(ROD_MU)
    cfg.density = ROD_MASS / vol
    b.add_shape_capsule(body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), along_x), radius=ROD_R, half_height=hh,
                        cfg=cfg, label="jointer_rod")


def _add_trowel(b: newton.ModelBuilder) -> None:
    """The margin trowel's blade: a free body whose shape collides with nothing (the carriage holds it at a height)."""
    body = b.add_body(xform=wp.transform(wp.vec3(*PARK_T), wp.quat_identity()), label="trowel")
    (hx, hy, hz), mass = TROWEL_BODY
    b.add_shape_box(body, xform=wp.transform(wp.vec3(0.0, 0.0, hz - TROWEL_HALF[2]), wp.quat_identity()), hx=hx, hy=hy,
                    hz=hz, cfg=newton.ModelBuilder.ShapeConfig(density=mass / (8.0 * hx * hy * hz), collision_group=0,
                                                               has_shape_collision=False, has_particle_collision=False))


def _faces(w: Wall) -> list[tuple[float, float, float, float]]:
    """Every rigid face (all blocks and the old wall, also outside a light run's window: a rod reaching past the
    window rests on the wall as it would), in world (x0, x1, y0, y1)."""
    out = []
    for a, b, c, d in JS.blocks(old_wall=0.25):
        (xa, ya), (xb, yb) = w.world(a, c), w.world(b, d)
        out.append((xa, xb, ya, yb))
    return out


def build_scene(w: Wall):
    """The chart's wall (a box whose top is the face plane; it collides with nothing: the joints are open) and the
    blocks' faces as static boxes 20 mm thick with their tops at the face plane, gaps at the joints."""
    def build(b: newton.ModelBuilder) -> None:
        b.rigid_gap = 0.002                 # the rod meets the faces only at their arrises (Newton's 0.1 m default
                                            # makes contact candidates with every face box near it: overflow)
        ghost = newton.ModelBuilder.ShapeConfig(collision_group=0, has_shape_collision=False,
                                                has_particle_collision=False)
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.025), hx=w.size[0] / 2, hy=w.size[1] / 2, hz=0.025,
                        cfg=ghost, label="wall")
        face = _stiff(ROD_MU)
        for k, (xa, xb, ya, yb) in enumerate(_faces(w)):
            b.add_shape_box(-1, xform=body_xform((xa + xb) / 2, (ya + yb) / 2, -0.01), hx=(xb - xa) / 2,
                            hy=(yb - ya) / 2, hz=0.01, cfg=face, label=f"face_{k}")
        _add_rod(b)
        _add_trowel(b)
    return build


def _stroke(keys: list[Stroke], t: float, a: tuple[float, float], b: tuple[float, float], task: ToolJoints) -> float:
    """One working stroke from a to b (world x, y): fly to a run-in point short of a, come down while moving, work to
    b at ``force``, run out past b and lift while moving (a tool stopped on the paste pushes it out radially)."""
    d = math.dist(a, b)
    u = ((b[0] - a[0]) / d, (b[1] - a[1]) / d)
    yaw = math.atan2(u[1], u[0])
    r = task.run_in
    s = (a[0] - r * u[0], a[1] - r * u[1])
    e = (b[0] + r * u[0], b[1] + r * u[1])
    up = ROD_R + 0.02
    t += math.dist(keys[-1].pos[:2], s) / task.air_speed + 0.3
    keys.append(Stroke(t, (*s, up), yaw=yaw, mode=HEIGHT))
    t += 0.3
    keys.append(Stroke(t, (*s, ROD_R), yaw=yaw, mode=HEIGHT))
    t += r / task.speed
    keys.append(Stroke(t, (*a, ROD_R), yaw=yaw, force=task.force, mode=FORCE))
    t += d / task.speed
    keys.append(Stroke(t, (*b, ROD_R), yaw=yaw, force=task.force, mode=HEIGHT))
    t += r / task.speed
    keys.append(Stroke(t, (*e, ROD_R + 0.004), yaw=yaw, mode=HEIGHT))
    t += 0.2
    keys.append(Stroke(t, (*e, up), yaw=yaw, mode=HEIGHT))
    return t


def _sweep(keys: list[Stroke], t: float, a: tuple[float, float], b: tuple[float, float], yaw: float,
           task: ToolJoints) -> float:
    """One margin-trowel strip (``concrete_joints_strips``): its centre from a to b (world x, y), the blade's long side
    along ``yaw`` and its underside ``TROWEL_GAP`` over the faces (HEIGHT). It comes straight down at a and lifts
    straight up at b (a scraping blade cuts what stands above its underside as it lowers), so its whole footprint
    works the strip's ends: coming down while moving, it left the ``run_in`` at each end, the T corners' tags, uncut."""
    z = TROWEL_HALF[2] + TROWEL_GAP
    up = z + 0.02
    t += math.dist(keys[-1].pos[:2], a) / task.air_speed + 0.3
    keys.append(Stroke(t, (*a, up), yaw=yaw, mode=HEIGHT))
    t += TROWEL_DOWN
    keys.append(Stroke(t, (*a, z), yaw=yaw, mode=HEIGHT))
    t += max(math.dist(a, b) / task.trowel_speed, 0.1)
    keys.append(Stroke(t, (*b, z), yaw=yaw, mode=HEIGHT))
    t += TROWEL_DOWN
    keys.append(Stroke(t, (*b, up), yaw=yaw, mode=HEIGHT))
    return t


def _course(keys: list[Stroke], t: float, lanes: list[Lane], w: Wall, task: ToolJoints) -> float:
    """One course's strokes onto ``keys``: ``passes`` per lane, back and forth (one way for a ``one_way`` lane)."""
    for ln in lanes:
        for k in range(task.passes):
            a, b = (ln.p0, ln.p1) if k % 2 == 0 or ln.one_way else (ln.p1, ln.p0)
            t = _stroke(keys, t, w.world(*a), w.world(*b), task)
    return t


def youngest_wait(task: ToolJoints) -> float:
    """s of extra age the youngest course needs to reach the thumbprint window's 5 kPa (0 if it already has)."""
    mat = CM.MasonryMortar()
    age = JS.AGES_MIN[-1] * 60.0 * CM.masco_factor(task.temperature)
    need = mat.time_to_tau(CM.THUMBPRINT[0], task.temperature)
    return max(need - age, 0.0) + task.wait_margin


def program(task: ToolJoints, w: Wall) -> tuple[ToolProgram, ToolProgram, float, float]:
    """The jointer's program: the joints of courses 0-2 (oldest first), parked over the wall at ``t_wait`` while the
    paste ages, then course 3; then the margin trowel's strips beside every joint, off the tooled grooves (the
    jointer parked). A course is tooled fast enough to end ``hard_margin`` before its mortar leaves the thumbprint
    window, and a side joint's strokes stay off the older course below (``concrete_joints_schedule``). Returns
    (jointer, trowel, t_wait, duration)."""
    win = (w.f.origin[0], w.f.origin[0] + w.size[0], w.f.origin[1], w.f.origin[1] + w.size[1])
    lanes = [ln for ln in (_clip(x, win) for x in joint_lanes()) if ln is not None]
    work = [ln for ln in (_clip(SCH.off_older(x, task.side_stop), win) for x in joint_lanes()) if ln is not None]
    keys = [Stroke(0.0, PARK, mode=HEIGHT)]
    t = 0.0
    t_wait = None
    for course in range(JS.COURSES):
        own = [x for x in work if x.course == course]
        late = course == JS.COURSES - 1 and bool(own)
        if late:
            t += 1.0
            keys.append(Stroke(t, PARK, mode=HEIGHT))
            t_wait = t + 0.5
            t += 1.0
            keys.append(Stroke(t, PARK, mode=HEIGHT))
        by = SCH.hard_time(course, task.temperature, youngest_wait(task) if late else 0.0) - task.hard_margin
        v = SCH.course_speed(lambda v, t=t, own=own: _course(list(keys), t, own, w, replace(task, speed=v)), by,
                             task.speed, task.speed_max)
        if own and _course(list(keys), t, own, w, replace(task, speed=v)) > by:
            log.warning("tool-joints: course %d ends after its mortar's thumbprint window even at %.2f m/s", course, v)
        t = _course(keys, t, own, w, replace(task, speed=v))
    t += math.dist(keys[-1].pos[:2], PARK[:2]) / task.air_speed + 0.2
    keys.append(Stroke(t, PARK, mode=HEIGHT))
    tkeys = [Stroke(0.0, PARK_T, mode=HEIGHT), Stroke(t, PARK_T, mode=HEIGHT)]
    if task.trowel:
        sc = JSTR.scored(JS.infill(task.seed, task.temperature))       # the whole wall's scored joint cells
        for st in JSTR.plan(lanes, sc, TROWEL_HALF[:2], task.overrun, win):
            t = _sweep(tkeys, t, w.world(*st.a), w.world(*st.b), st.yaw, task)
        t += math.dist(tkeys[-1].pos[:2], PARK_T[:2]) / task.air_speed + 0.2
        tkeys.append(Stroke(t, PARK_T, mode=HEIGHT))
    return (ToolProgram("jointer", tuple(keys)), ToolProgram("trowel", tuple(tkeys)),
            (t_wait if t_wait is not None else t), t + 0.5)


BOARD_R, BOARD_OUT = 0.013, 0.016       # m: drain radius and its centre beyond the infill's side (in the 30 mm margin)


def boards(w: Wall) -> tuple[Drain, ...]:
    """Where the beds' rolls go: a free drain in the old wall's margin at each end of every bed joint (the excess
    struck off the wall and back onto the mortar board, their margin trowel's carry, ``CARRY_CAP``), in the window."""
    (x0, x1), out = JS.INFILL_X, []
    for ln in joint_lanes():
        if ln.p0[1] != ln.p1[1]:
            continue
        for k, x in enumerate((x0 - BOARD_OUT, x1 + BOARD_OUT)):
            xw, yw = w.world(x, ln.p0[1])
            if abs(xw) < w.size[0] / 2 and abs(yw) < w.size[1] / 2:
                out.append(Drain(f"board_{len(out)}", chart=0, center=(xw, yw, 0.0), radius=BOARD_R))
    return tuple(out)


def plugins(task: ToolJoints, w: Wall) -> tuple[ToolCarriage, SurfaceFlow, ConcretePaste, float, float]:
    mat = CM.MasonryMortar()
    tau0 = float(mat.tau_bulk(JS.AGES_MIN[0] * 60.0 * CM.masco_factor(task.temperature), task.temperature))
    film = FilmMaterial("masonry_mortar", density=mat.rho, viscosity=mat.mu, yield_stress=tau0, surface_tension=0.0)
    rod = Applicator(body="jointer", shape="roller", size=(ROD_R, ROD_L / 2 - ROD_R), center=(0.0, 0.0, 0.0),
                     axis=(1.0, 0.0, 0.0), stiffness=5.0e7, damping=0.0, implicit=True)
    # the margin trowel cuts what stands above its underside into its load and keeps it (WP15: a scraping store,
    # split 1; retained = held = 1: it never lays); the cut is geometric, so it lifts a tag whatever its stiffness
    trowel = Applicator(body="trowel", shape="rect", size=TROWEL_HALF[:2], center=(0.0, 0.0, -TROWEL_HALF[2]),
                        stiffness=1.0e5, damping=0.0,
                        store=Store(capacity=TROWEL_CAP, retained=1.0, held=1.0, split=1.0, pickup=0.0, drip=0.0,
                                    scrape=True))
    flow = SurfaceFlow(charts=(PlaneChart("wall", cells=w.cells),), material=film, inertia=False,
                       max_depth=FILM_DEPTH + 0.002, tools=(rod, trowel), tool_speed=0.3, per_cell_rheology=True,
                       drains=boards(w))
    jointer, trowel_p, t_wait, duration = program(task, w)
    carriage = ToolCarriage(programs=(jointer, trowel_p), omega_rot=task.omega_rot)
    layer = ConcretePaste(material="masonry_mortar", temperature=task.temperature, tools=(JOINTER, TROWEL), flow=flow,
                          carriage=carriage, film_yield="bulk")
    return carriage, flow, layer, t_wait, duration


# --- the run and the verdict -----------------------------------------------------------------------------------------

@dataclass
class JointsResult:
    verdicts: list[dict]
    success: int
    worlds: int
    sim_s: float
    wall_s: float
    wait_s: float
    ledger_rel: float
    spilled_l: list[float]
    board_l: list[float]
    trowel_ml: list[float]
    max_load_n: float
    cells: tuple[int, int]
    off_wall_ml: list[float] = field(default_factory=list)   # per world: mortar off the wall (``off_wall``)


def grids(w: Wall, flow: SurfaceFlow, layer: ConcretePaste) -> list[MX.PasteGrid]:
    """The verdict's grids; what left the wall other than into the drains is their off-region volume."""
    n = w.cells
    h = flow.h.numpy().reshape(-1, *n)
    base = flow.base.numpy().reshape(-1, *n)
    off = off_wall(flow.read())
    r = {k: v.reshape(-1, *n) for k, v in layer.read().items() if v.ndim == 2 and v.shape[1] == n[0] * n[1]}
    return [MX.PasteGrid(h=base[i] + h[i], t=np.where(w.f.joint, base[i] + h[i] - w.f.base, h[i]), cell=w.f.cell,
                         paste_mask=w.f.joint, C=r["C"][i], tau_w=r["tau_w"][i], worked=r["worked"][i],
                         off_m3=float(off[i]))
            for i in range(h.shape[0])]


def start_grid(w: Wall) -> MX.PasteGrid:
    z = np.zeros(w.cells)
    return MX.PasteGrid(h=w.f.top, t=w.f.top - np.where(w.f.joint, w.f.base, 0.0), cell=w.f.cell,
                        paste_mask=w.f.joint, C=z, tau_w=z, worked=z)


def run(task: ToolJoints | None = None, *, worlds: int = 1, device: str = "cpu", npz: str | None = None) -> JointsResult:
    task = task or ToolJoints()
    walls = [wall(task, task.seed + i) for i in range(worlds)]               # world i: seed + i
    w = walls[0]
    carriage, flow, layer, t_wait, duration = plugins(task, w)
    sim = build_simulation(TestScene(build=build_scene(w), ground=False, worlds=worlds, dt=2.0e-3),
                           [carriage, flow, layer], device)
    C = w.cells[0] * w.cells[1]
    parts = [split(x.f) for x in walls]
    flow.set_input("base", np.stack([p[0].reshape(C) for p in parts]).astype(np.float32))
    # the trowel's knife fill goes into the joints' cells, never the faces beside them (GA's paste_mask)
    flow.set_input("paste_mask", np.stack([x.f.joint.reshape(C) for x in walls]).astype(np.float32))
    flow.set_depth(np.stack([p[1].reshape(C) for p in parts]))
    layer.set_start(paste=np.stack([p[2].reshape(C) for p in parts]).astype(np.int32),
                    age=np.stack([x.f.age.reshape(C) for x in walls]), a=np.stack([x.f.a.reshape(C) for x in walls]))
    v0 = flow.read()["film_volume"]
    wait = youngest_wait(task)
    periods, i_wait = int(math.ceil(duration / PERIOD)), int(round(t_wait / PERIOD))
    max_load, t0 = 0.0, time.perf_counter()
    for i in range(periods):
        if i == i_wait and wait > 0.0:                    # the tools are parked: the paste ages
            layer.set_start(age=layer.read()["age"] + wait)
        sim.step_period()
        if i % 25 == 0:
            max_load = max(max_load, float(carriage.read()["load"].max()))
        if i % 500 == 0:
            log.info("period %d of %d (%.0f s sim, %.0f s wall)", i, periods, i * PERIOD, time.perf_counter() - t0)
    wall_s = time.perf_counter() - t0
    read = flow.read()
    ends = grids(w, flow, layer)
    verdicts = [MX.tool_joints(g, start_grid(x)) for g, x in zip(ends, walls, strict=True)]
    if npz:
        np.savez_compressed(npz, h=np.stack([g.h for g in ends]), t=np.stack([g.t for g in ends]),
                            C=np.stack([g.C for g in ends]), tau_w=np.stack([g.tau_w for g in ends]),
                            worked=np.stack([g.worked for g in ends]), joint=w.f.joint, h0=w.f.top, cell=w.f.cell)
    out = [CV.verdict_dict(v, TOOL_JOINTS, differs=window_differs(task.window)) for v in verdicts]
    return JointsResult(verdicts=out, success=sum(d["success"] for d in out), worlds=worlds,
                        sim_s=periods * PERIOD, wall_s=wall_s, wait_s=wait,
                        ledger_rel=float((np.abs(read["ledger"] - v0) / np.maximum(v0, 1e-12)).max()),
                        spilled_l=[float(x) * 1e3 for x in read["spilled_volume"]],
                        board_l=[float(x) * 1e3 for x in read["drained_total_volume"]],
                        trowel_ml=[float(x) * 1e6 for x in np.asarray(read.get("tool_load", np.zeros((worlds, 1))))
                                   .reshape(worlds, -1).sum(axis=1)], max_load_n=max_load,
                        cells=w.cells, off_wall_ml=[float(x) * 1e6 for x in off_wall(read)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=float, nargs=4, default=None, metavar=("X0", "X1", "Z0", "Z1"))
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--force", type=float, default=20.0)
    ap.add_argument("--no-trowel", action="store_true", help="tool only (no tag removal)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--npz", default=None, help="final fields per world and the start")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    task = ToolJoints(seed=args.seed, window=tuple(args.window) if args.window else None, passes=args.passes,
                      force=args.force, trowel=not args.no_trowel)
    res = run(task, worlds=args.worlds, device=args.device, npz=args.npz)
    out = json.dumps(asdict(res), indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(out)
    print(out)
    return 0 if res.success == res.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["JOINTER", "ToolJoints", "joint_lanes", "program", "run", "split", "wall"]
