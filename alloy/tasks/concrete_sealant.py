# SPDX-License-Identifier: Apache-2.0
"""task concrete_sealant: Gym-Anything's tool-sealant, a surface-flow concrete proof.

"Tool the sealant in the control joint between the tape marks, dry." (``concrete_finishing/task.py:30``; spec
``[internal note]`` §1). A 13 mm control joint, its backer 8.5 mm behind the
wall faces, holds a convex bead 1.5-2.7 mm proud with two gun stops and a 40 mm stretch sagged off the north wall; a
quarter of the cells beside the walls do not touch them. The start is theirs cell for cell (``concrete_sealant_start``,
1 mm grid).

Model: the wall is laid flat (z out of the face, faces at 0) with gravity along it, down the joint (-y): at 600 Pa
the 10 mm bead's own weight (137 Pa of wall shear) moves nothing, and a lifted nose's sole faces sideways, as on GA's
vertical wall. One ``surface_flow`` plane chart over the region (or a window along the joint): x across the
joint, y along it; its ``base`` is the backer in the joint and the faces outside. A privileged carriage
(``concrete_tools``; no robot) slides the tooling spatula's nose (a nylon capsule of r 11 mm, its 18 mm sole along
the joint) down the joint at ``force``: it rests on both arrises, so its underside is an arc 2.1 mm below the faces at
the centre. WP15's ``roller`` pad works the film under it, and its scraping store (15 ml carry) cuts what stands above
the arc and fills every joint cell under it up to the arc from its carry (``fill="profile"``, GA's nose fill,
``coupling.py:561-567``). The nose is wiped on the rag after every stroke (45 ml/m of bead is displaced,
``params.py:160-164``): it lets go of 1 - exp(-d / 5 cm) of its load over each slide d (``world.py:259-285``),
booked out of the film. A lifted nose keeps only what its sealant's yield holds on it (0.52 ml with its sole sideways)
and drops the rest off the region (WP15's lifted gob, GA ``coupling.py:922-997``: it falls to the floor and fails the
smear rule), so strokes are short and each is wiped; ``liftoff_ml`` reports the load at each lift-off.

What the verdict judges (audit AB-gym-7; ``concrete_ports.TOOL_SEALANT``): all five criteria from the sealant's
fields; ``smear``'s off-region part is the gobs the lifted nose dropped. The scene differs from Gym-Anything's: the
actor (their task interface is robot-only; their carriage is an ASSISTED probe fixture, ``probes/common.py:1-4``),
the wall as a flat chart with gravity along the joint, the rag (a scripted release on their wipe law, no rag body;
the script wipes after every stroke of at most ``max_stroke``, which no robot of theirs is held to), and the site
(one temperature for every seed; theirs varies). Each verdict lists them (``scene_differs``, ``summary``).

Run: ``python -m alloy.tasks.concrete_sealant [--window z0 z1] [--device cuda:0] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.core.timebase import snap_to_grid
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store
from alloy.plugins.surface_flow.stores_lift import gravity_dir, stick_share, stick_volume
from alloy.tasks import concrete_metrics as MX
from alloy.tasks import concrete_sealant_metrics as SM
from alloy.tasks import concrete_sealant_plan as SP
from alloy.tasks import concrete_sealant_start as SS
from alloy.tasks import concrete_verdict as CV
from alloy.tasks.concrete_paste import MATERIALS, ConcretePaste, SoleWork
from alloy.tasks.concrete_ports import TOOL_SEALANT
from alloy.tasks.concrete_sealant_layer import SealantContact
from alloy.tasks.concrete_tools import FORCE, HEIGHT, Stroke, ToolCarriage, ToolProgram

log = logging.getLogger("alloy.tasks.concrete_sealant")

NOSE_R, NOSE_HALF, NOSE_MASS = 0.011, 0.009, 0.10   # m, m, kg: the tip capsule (tools.py:166-169; params.py:149-150);
                                                    # the body carries the handle's bulk, as the jointer's does
NOSE_REST = math.sqrt(NOSE_R**2 - (SS.CJ_W / 2) ** 2)   # m: its axis over the faces resting on both arrises (8.9)
NOSE_W = 0.0159                         # m: the blade across the joint (5/8 in, params.py:149)
NOSE_WORK = SoleWork("nose", a_tool=2.0e-5, dens=0.3, seal=0.0, half=(NOSE_HALF, NOSE_W / 2))   # tools.py:164-173
NOSE_STICK = 4.0 * NOSE_HALF * NOSE_W / 2   # m²: the sole a lifted gob sticks to (GA: 4 hx hy, coupling.py:933)
NOSE_CAP = 1.5e-5                       # m³: what it carries (params.py:160)
RAG_WIPE = 0.05                         # m: the rag keeps 1 - exp(-d / RAG_WIPE) of the load (params.py:163)
TOOL_MU = 0.35                          # tools.py:22
SEALANT = dict(density=1400.0, viscosity=50.0, yield_stress=600.0)   # materials.py:126-142; params.py:119
MAX_DEPTH = 0.016                       # m: the bead (11.2 mm) with a gun stop (3 mm)
FACE_HALF = 0.25                        # m: the faces' extent beside and along the joint
PERIOD = 0.02                           # s: one control period
X_DT = 2.0e-3                           # s: the target exchange step (the stepper snaps it to PERIOD / 2^k)
GRAVITY = (0.0, -9.81, 0.0)             # down the wall: world +y is up the joint (Region.along)
RIGID_DT = 1.25e-3                      # s: the rigid step (the harness CPU route)
PARK = (0.0, 0.0, 0.06)                 # m: over the window's centre
RAG_X = 0.05                            # m: the rag beside the joint at the stroke's end (off the chart: the tool
                                        # touches nothing there), slid along x


@dataclass
class ToolSealant:
    """The task's settings: seed, site temperature, the fast-skinning grade, a window along the joint (z0, z1 [m];
    None = the whole region) and the strokes."""

    seed: int = 0
    temperature: float = 21.1
    fast: bool = False
    window: tuple[float, float] | None = None
    force: float = 5.0                   # N on the nose
    speed: float = 0.10                  # m/s along the joint
    air_speed: float = 0.50              # m/s off the work
    run_in: float = 0.005                # m: it comes down this far above a stroke's top, on tooled sealant
    run_out: float = 0.0                 # m: it runs out this far below a stroke's bottom while rising ``rise``,
    rise: float = 0.0                    # m  then lifts straight up. Cells that newly come under an advancing,
                                         # rising nose fill to its arc from its carry (the roll lands where it
                                         # lifts), and the next stroke cuts them again: so none
    budget: float = 0.30e-6              # m³: the most a planned stroke cuts (concrete_sealant_plan). A nose touching
                                         # no sealant keeps only what its yield holds on it, GA's 0.52 ml on a
                                         # vertical wall (coupling.py:922-938), and drops the rest off the region;
                                         # it lifts off with the cut plus the 13.5 % its last wipe left on it
    max_stroke: float = 0.03             # m: the longest stroke between wipes
    wipe: float = 0.10                   # m slid over the rag (86 % of the load)
    wipe_speed: float = 0.60             # m/s over the rag
    omega_rot: float = 400.0             # rad/s: the carriage's attitude servo


@dataclass(frozen=True)
class Region:
    """The run's joint region: the start fields on the window and the map from the grid to world (x, y)."""

    j: SS.Joint

    @property
    def cells(self) -> tuple[int, int]:
        return self.j.joint.shape

    @property
    def size(self) -> tuple[float, float]:
        n = self.cells
        return n[0] * self.j.cell, n[1] * self.j.cell

    def along(self, z: float) -> float:
        """World y of the wall height z [m]."""
        return z - float(self.j.z[0, 0]) + self.j.cell / 2 - self.size[1] / 2


def crop(j: SS.Joint, z0: float, z1: float) -> SS.Joint:
    """The rows whose centres lie in z0 <= z < z1 [m]."""
    k = (j.z[0] >= z0) & (j.z[0] < z1)
    return SS.Joint(**{f: (v[:, k] if isinstance(v, np.ndarray) else v) for f, v in vars(j).items()})


def region(task: ToolSealant, seed: int | None = None) -> Region:
    j = SS.joint_region(task.seed if seed is None else seed, task.temperature)
    return Region(j if task.window is None else crop(j, *task.window))


# --- scene, strokes, plug-ins ----------------------------------------------------------------------------------------

def _gains(mu: float, mass: float) -> newton.ModelBuilder.ShapeConfig:
    """Stiff, damped contact for the nose on the arrises (alloy.scene.native_gains, stable at the rigid step)."""
    from alloy.scene.native_gains import native_gains  # noqa: PLC0415

    ke, kd = native_gains(1.0e6, mass, 0.0, dt=RIGID_DT)
    return newton.ModelBuilder.ShapeConfig(mu=mu, ke=ke, kd=kd)


def build_scene(r: Region):
    """The chart's wall (a ghost box, top at the face plane), the two faces beside the joint as static boxes with
    their arrises at the joint's edges, and the nose: a free body with a capsule along its x."""
    def build(b: newton.ModelBuilder) -> None:
        b.rigid_gap = 0.002
        ghost = newton.ModelBuilder.ShapeConfig(collision_group=0, has_shape_collision=False,
                                                has_particle_collision=False)
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.025), hx=r.size[0] / 2, hy=r.size[1] / 2, hz=0.025,
                        cfg=ghost, label="wall")
        face, half = _gains(TOOL_MU, NOSE_MASS), (FACE_HALF - SS.CJ_W / 2) / 2
        for k, sx in enumerate((-1.0, 1.0)):
            b.add_shape_box(-1, xform=body_xform(sx * (SS.CJ_W / 2 + half), 0.0, -0.01), hx=half,
                            hy=r.size[1] / 2 + FACE_HALF, hz=0.01, cfg=face, label=f"face_{k}")
        body = b.add_body(xform=wp.transform(wp.vec3(*PARK), wp.quat_identity()), label="nose")
        cfg = _gains(TOOL_MU, NOSE_MASS)
        cfg.density = NOSE_MASS / (math.pi * NOSE_R**2 * 2 * NOSE_HALF + 4.0 / 3.0 * math.pi * NOSE_R**3)
        b.add_shape_capsule(body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0),
                                                     wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)),
                            radius=NOSE_R, half_height=NOSE_HALF, cfg=cfg, label="nose_tip")
    return build


UP = NOSE_REST + 0.012                  # m: the nose's axis between strokes, clear of the bead and gun stops
LIFT_S = 0.02                           # s: the lift off the work, quick to save time (since WP15's 3eb6b2228 a
                                        # lifting nose lays nothing at any speed)


def _stroke(keys: list[Stroke], t: float, a: float, b: float, task: ToolSealant) -> float:
    """One stroke down the joint from world y a to b: come down on tooled sealant ``run_in`` before a, slide to b at
    ``force``, run out ``run_out`` while rising ``rise`` (none by default), then lift straight up."""
    u = 1.0 if b > a else -1.0
    yaw = u * math.pi / 2
    s, e = a - u * task.run_in, b + u * task.run_out
    t += math.dist(keys[-1].pos[:2], (0.0, s)) / task.air_speed + 0.1
    keys.append(Stroke(t, (0.0, s, UP), yaw=yaw, mode=HEIGHT))
    t += 0.15
    keys.append(Stroke(t, (0.0, s, NOSE_REST), yaw=yaw, mode=HEIGHT))
    t += max(task.run_in / task.speed, 0.02)
    keys.append(Stroke(t, (0.0, a, NOSE_REST), yaw=yaw, force=task.force, mode=FORCE))
    t += abs(b - a) / task.speed
    keys.append(Stroke(t, (0.0, b, NOSE_REST), yaw=yaw, force=task.force, mode=HEIGHT))
    t += max(task.run_out / task.speed, 0.04)
    keys.append(Stroke(t, (0.0, e, NOSE_REST + task.rise), yaw=yaw, mode=HEIGHT))
    t += LIFT_S
    keys.append(Stroke(t, (0.0, e, UP), yaw=yaw, mode=HEIGHT))
    return t


def _wipe(keys: list[Stroke], t: float, task: ToolSealant) -> tuple[float, tuple[float, float]]:
    """Fly to the rag beside the stroke's end, slide ``wipe`` over it (away from the joint) and stay up; returns the
    time and the slide's (start, end) times."""
    y = keys[-1].pos[1]
    t += math.dist(keys[-1].pos[:2], (RAG_X, y)) / task.air_speed + 0.1
    keys.append(Stroke(t, (RAG_X, y, UP), yaw=math.pi / 2, mode=HEIGHT))
    t0 = t
    t += task.wipe / task.wipe_speed
    keys.append(Stroke(t, (RAG_X + task.wipe, y, UP), yaw=math.pi / 2, mode=HEIGHT))
    return t, (t0, t)


def stroke_edges(task: ToolSealant, r: Region) -> list[float]:
    """The strokes' edges down the window's segment [m along the wall] (concrete_sealant_plan)."""
    j = r.j
    z0, z1 = float(j.z[0, 0]), float(j.z[0, -1])
    # it first comes down past the bead's end; the ends fall on row boundaries, so no row's centre lies on the pad's
    # square end, where float rounding decides whether the stroke cut it (a row left standing there touches the
    # lifting nose)
    lo, hi = max(z0, SS.SEGMENT[0]), z1 + NOSE_HALF + 1.5 * j.cell
    f = SP.Field(surf=np.where(j.joint, j.top, -np.inf), x=(j.y[:, :1] - SS.CJ_Y), z=j.z[0].copy(), cell=j.cell)
    return SP.plan(f, lo, hi, task.budget, task.max_stroke, SP.Path(task.run_in, task.run_out, task.rise),
                   SP.Nose(NOSE_R, NOSE_HALF, NOSE_REST))


def program(task: ToolSealant, r: Region) -> tuple[ToolProgram, list[tuple[float, float]], float]:
    """Planned strokes down the window's segment (the joint between the tape marks), each followed by a wipe. Returns
    the nose's program, the wipes' slide intervals [s] and the duration."""
    keys, wipes, t = [Stroke(0.0, PARK, mode=HEIGHT)], [], 0.0
    edges = stroke_edges(task, r)                                       # top to bottom
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        t = _stroke(keys, t, r.along(float(a)), r.along(float(b)), task)
        t, w = _wipe(keys, t, task)
        wipes.append(w)
    t += math.dist(keys[-1].pos[:2], PARK[:2]) / task.air_speed + 0.2
    keys.append(Stroke(t, PARK, mode=HEIGHT))
    return ToolProgram("nose", tuple(keys)), wipes, t + 0.5


def material(task: ToolSealant) -> str:
    """The process layer's material: the 2 h sealant, or the 20 min fast-skinning grade (variants.py:33)."""
    return "sealant_fast" if task.fast else "sealant"


def skin(task: ToolSealant) -> float:
    """The sealant's skin age at the site [s] (the process layer's material)."""
    return MATERIALS[material(task)].skin_time(task.temperature)


def film() -> FilmMaterial:
    return FilmMaterial("sealant", surface_tension=0.0, **SEALANT)


def nose_tool() -> Applicator:
    """The nose on WP15's scraping store: a roller pad along the joint (body x), its sole facing -z."""
    return Applicator(body="nose", shape="roller", size=(NOSE_R, NOSE_HALF), center=(0.0, 0.0, 0.0),
                      axis=(1.0, 0.0, 0.0), stiffness=5.0e7, damping=0.0, implicit=True,
                      store=Store(capacity=NOSE_CAP, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0, bins=1,
                                  scrape=True, fill="profile", stick_area=NOSE_STICK))


def stick_ml(tool: Applicator, material: FilmMaterial, gravity: tuple[float, float, float]) -> float:
    """What a lifted nose keeps [ml], by the engine's rule (stores_lift): the stick volume times the share for its
    sole's normal along the scene's gravity. The carriage holds the nose level and turns it only about z, so its
    sole's normal in the world is its body normal."""
    share = float(stick_share(wp.vec3(*(float(v) for v in tool.normal)), gravity_dir(gravity)))
    return stick_volume(tool, material, gravity) * share * 1e6


def plugins(task: ToolSealant, regions: list[Region]) -> tuple[ToolCarriage, SurfaceFlow, ConcretePaste,
                                                               SealantContact, list[list[tuple[float, float]]], float]:
    """The plug-ins; each world's strokes are planned on its own start (its seed's gun stops). Returns them, each
    world's wipe intervals [s] and the longest program's duration."""
    flow = SurfaceFlow(charts=(PlaneChart("wall", cells=regions[0].cells),), material=film(), inertia=False,
                       max_depth=MAX_DEPTH, tools=(nose_tool(),), tool_speed=0.3)
    progs = [program(task, x) for x in regions]
    per_world = tuple((p[0],) for p in progs) if len(progs) > 1 else ()
    carriage = ToolCarriage(programs=(progs[0][0],), world_programs=per_world, omega_rot=task.omega_rot)
    layer = ConcretePaste(material=material(task), temperature=task.temperature, tools=(NOSE_WORK,), flow=flow,
                          carriage=carriage, film_yield="bulk")
    contact = SealantContact(skin=skin(task), paste=layer, flow=flow)
    return carriage, flow, layer, contact, [p[1] for p in progs], max(p[2] for p in progs)


# --- the run and the verdict -----------------------------------------------------------------------------------------

@dataclass
class SealantResult:
    verdicts: list[dict]
    success: int
    worlds: int
    sim_s: float
    wall_s: float
    ledger_rel: float
    wiped_ml: list[float]
    dropped_ml: list[float]
    carried_ml: list[float]
    max_load_n: float                    # the carriage's largest push on the work [N]
    cells: tuple[int, int]
    liftoff_ml: list[list[float]]        # per world and stroke: the load the nose lifted off with (at the rag, plus
                                         # what fell since the last wipe; exact when nothing fell mid-stroke)
    stick_ml: float                      # what a lifted nose keeps; more falls off the region
    rag_kept: list[list[float]]          # per world, the share of the load each wipe kept (exp(-wipe / RAG_WIPE))


def exchanges(dt: float = X_DT) -> int:
    """Exchanges per control period: the stepper runs PERIOD / 2^k <= dt."""
    return round(PERIOD / snap_to_grid(dt, PERIOD))


def release_schedule(wipes: list[tuple[float, float]], periods: int, task: ToolSealant, n_x: int) -> np.ndarray:
    """The rag's share of the free load let go in each exchange of each period [periods] (``tool_release`` acts per
    exchange): over the time d a period spends on the rag it keeps exp(-v·d / RAG_WIPE), split evenly over its n_x
    exchanges, so a whole wipe keeps exp(-wipe / RAG_WIPE)."""
    t = np.arange(periods + 1)[:, None] * PERIOD
    w = np.asarray(wipes, np.float64).reshape(-1, 2)
    d = np.clip(np.minimum(w[None, :, 1], t + PERIOD) - np.maximum(w[None, :, 0], t), 0.0, None).sum(axis=1)
    return np.where(d > 0.0, -np.expm1(-task.wipe_speed * d / (RAG_WIPE * n_x)), 0.0)


@dataclass
class Drive:
    """What the run loop saw: the carriage's largest push, the store's load at each lift-off and across each wipe."""

    max_load: float
    liftoff: list[list[float]]
    rag_kept: list[list[float]]
    wall_s: float


def _drive(sim, flow: SurfaceFlow, carriage: ToolCarriage, task: ToolSealant,
           wipes: list[list[tuple[float, float]]], periods: int) -> Drive:
    """Step every period, setting each world's rag release, and watch the nose's store at each wipe. Every stroke ends
    in a wipe, and in the air only a falling gob changes the load: so the load the nose lifted off with is the load it
    brings to the rag plus what it dropped since the last wipe."""
    worlds, t0 = len(wipes), time.perf_counter()
    rel = np.stack([release_schedule(w, periods, task, exchanges()) for w in wipes])    # [W, periods + 1]
    out = Drive(0.0, [[] for _ in range(worlds)], [[] for _ in range(worlds)], 0.0)
    loads = lambda: flow.tool_load.numpy().reshape(worlds, -1).sum(axis=1)           # noqa: E731 [W] m³
    drops = lambda: flow.tool_dropped.numpy().reshape(worlds, -1).sum(axis=1)        # noqa: E731
    before, dropped, share = np.zeros(worlds), np.zeros(worlds), np.full(worlds, -1.0)
    for i in range(periods):
        s = rel[:, i]
        if np.any(s != share):
            start = (share <= 0.0) & (s > 0.0)                  # at the rag: the stroke's lift-off load
            if start.any():
                ld, d = loads(), drops()
                for w in np.flatnonzero(start):
                    before[w] = ld[w]
                    out.liftoff[w].append(float(ld[w] + d[w] - dropped[w]) * 1e6)
                    dropped[w] = d[w]
            flow.tool_release.assign(s.reshape(worlds, 1))
            share = s.copy()
        sim.step_period()
        end = (s > 0.0) & (rel[:, i + 1] == 0.0)
        if end.any():
            after = loads()
            for w in np.flatnonzero(end & (before > 1e-12)):
                out.rag_kept[w].append(float(after[w] / before[w]))
        if i % 25 == 0:
            out.max_load = max(out.max_load, float(carriage.read()["load"].max()))
        if i % 500 == 0:
            log.info("period %d of %d (%.0f s sim, %.0f s wall)", i, periods, i * PERIOD, time.perf_counter() - t0)
    out.wall_s = time.perf_counter() - t0
    return out


def grids(r: Region, flow: SurfaceFlow, layer: ConcretePaste, contact: SealantContact, skin_s: float,
          off: np.ndarray) -> list[SM.SealantGrid]:
    """The end state as verdict grids, with ``off`` [m³] per world off the region (the gobs a lifted nose dropped: they
    fall clear of the wall, and their sealant fails the smear rule above 2e-8 m³, GA task.py:159-167)."""
    n = r.cells
    h = flow.h.numpy().reshape(-1, *n)
    base = flow.base.numpy().reshape(-1, *n)
    pl, cl = layer.read(), contact.read()
    f = {k: v.reshape(-1, *n) for k, v in {**pl, **cl}.items() if v.ndim == 2 and v.shape[1] == n[0] * n[1]}
    s0 = SM.start_grid(r.j, skin_s)
    return [SM.SealantGrid(g=MX.PasteGrid(h=base[i] + h[i], t=h[i], cell=r.j.cell, paste_mask=r.j.joint,
                                          worked=f["worked"][i], off_m3=float(off[i])),
                           across=s0.across, z=s0.z, edge_l=s0.edge_l, edge_r=s0.edge_r, contact_l=f["contact_l"][i],
                           contact_r=f["contact_r"][i], seal_age=f["seal_age"][i], skin_torn=f["skin_torn"][i],
                           skin=skin_s) for i in range(h.shape[0])]


def run(task: ToolSealant | None = None, *, worlds: int = 1, device: str = "cpu",
        npz: str | None = None) -> SealantResult:
    task = task or ToolSealant()
    regions = [region(task, task.seed + i) for i in range(worlds)]           # world i: seed + i
    r = regions[0]
    carriage, flow, layer, contact, wipes, duration = plugins(task, regions)
    scene = TestScene(build=build_scene(r), ground=False, worlds=worlds, dt=X_DT, gravity=GRAVITY)
    sim = build_simulation(scene, [carriage, flow, layer, contact], device)
    C = r.cells[0] * r.cells[1]
    flow.set_input("base", np.stack([np.where(x.j.joint, x.j.base, 0.0).reshape(C) for x in regions])
                   .astype(np.float32))
    flow.set_input("paste_mask", np.stack([x.j.joint.reshape(C) for x in regions]).astype(np.float32))
    flow.set_depth(np.stack([np.where(x.j.joint, x.j.top - x.j.base, 0.0).reshape(C) for x in regions]))
    stack = lambda f: np.stack([f(x.j).reshape(C) for x in regions])           # noqa: E731 - per world
    layer.set_start(paste=stack(lambda j: j.joint).astype(np.int32), age=stack(lambda j: np.full(j.joint.shape, j.age)),
                    a=stack(lambda j: j.a))
    contact.set_start(contact_l=stack(lambda j: j.contact_l), contact_r=stack(lambda j: j.contact_r),
                      edge_l=stack(lambda j: j.edge_l), edge_r=stack(lambda j: j.edge_r))
    v0 = flow.read()["film_volume"]
    periods = int(math.ceil(duration / PERIOD))
    drive = _drive(sim, flow, carriage, task, wipes, periods)
    read = flow.read()
    skin_s = skin(task)
    dropped = np.asarray(read["tool_dropped"]).reshape(worlds, -1).sum(axis=1)
    ends = grids(r, flow, layer, contact, skin_s, dropped)
    verdicts = [SM.tool_sealant(g, SM.start_grid(x.j, skin_s)) for g, x in zip(ends, regions, strict=True)]
    if npz:
        np.savez_compressed(npz, h=np.stack([g.g.h for g in ends]), t=np.stack([g.g.t for g in ends]),
                            joint=r.j.joint, h0=r.j.top, cell=r.j.cell)
    load = np.asarray(read.get("tool_load", np.zeros((worlds, 1)))).reshape(worlds, -1).sum(axis=1)
    out = [CV.verdict_dict(v, TOOL_SEALANT, differs=window_differs(task.window)) for v in verdicts]
    return SealantResult(verdicts=out, success=sum(d["success"] for d in out), worlds=worlds,
                         sim_s=periods * PERIOD, wall_s=drive.wall_s,
                         ledger_rel=float((np.abs(read["ledger"] - v0) / np.maximum(v0, 1e-12)).max()),
                         wiped_ml=[float(x) * 1e6 for x in np.asarray(read["tool_released"]).reshape(worlds, -1)
                                   .sum(axis=1)],
                         dropped_ml=[float(x) * 1e6 for x in dropped], carried_ml=[float(x) * 1e6 for x in load],
                         max_load_n=drive.max_load, cells=r.cells, liftoff_ml=drive.liftoff,
                         stick_ml=stick_ml(flow.config.tools[0], flow.config.material, scene.gravity),
                         rag_kept=drive.rag_kept)


def window_differs(window: tuple[float, float] | None) -> tuple[str, ...]:
    """A window's verdict says it is not the whole region's."""
    if window is None:
        return ()
    return (f"window: {window[0]:g}-{window[1]:g} m along the joint, not the whole region: the shares count the "
            f"window's part of the taped segment only",)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("Z0", "Z1"))
    ap.add_argument("--force", type=float, default=5.0)
    ap.add_argument("--fast", action="store_true", help="the fast-skinning grade (20 min)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--npz", default=None, help="final fields per world and the start")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    task = ToolSealant(seed=args.seed, fast=args.fast, force=args.force,
                       window=tuple(args.window) if args.window else None)
    res = run(task, worlds=args.worlds, device=args.device, npz=args.npz)
    out = json.dumps(asdict(res), indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(out)
    print(out)
    return 0 if res.success == res.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
