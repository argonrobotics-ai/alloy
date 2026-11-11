# SPDX-License-Identifier: Apache-2.0
"""task concrete_drywall: Gym-Anything's patch-drywall on surface flow (``concrete_finishing``, the concrete set's
entry task; SOFT-2's spec ``[internal note]`` §2).

"Fill the anchor holes and knife the patch flush" (``task.py:31``). The start is theirs cell for cell
(:mod:`alloy.tasks.concrete_drywall_start`), the verdict their five criteria and progress-v1
(:mod:`alloy.tasks.concrete_drywall_checks`).

Model:
- One ``surface_flow`` plane chart on the partition's face, standing as theirs: u along -x, v up, gravity along the
  face. The anchor holes and the mesh's fibres are the chart's ``base``; the joint compound is the film, a Bingham
  paste (700 Pa, 6 Pa·s, 1000 kg/m³: Gym-Anything's ``COMPOUND``, from 3M US 11,267,758; the density their
  estimate). At 700 Pa a 16 mm column in a hole stands (ρ·g·h = 160 Pa), so nothing slumps.
- The grid is theirs, 1 mm (``cell``). A coarser one is a smoke, labelled so in the result: a hole cell is read at
  1 mm depth (:func:`~alloy.tasks.concrete_drywall_checks.surface`), but ridges and smears average away.
- The board's face paper can take water from the compound (design note §3.5): ``uptake=True`` runs the process
  stage's ``Substrate``, Lucas-Washburn uptake S·sqrt(t) up to a capacity, with S = 2e-6 m/sqrt(s) and 150 µm, their
  painting set's paperboard ``cardboard`` (``painting/params.py:111-114``, estimates) as the nearest unprimed paper
  face. It is off by default: Gym-Anything models no uptake here (their compound is constant,
  ``materials.py:145-160``), and over their 15 min it would take about 60 µm, so compound knifed to the fibre tops
  (0.5 mm) would fall under ``mesh_covered``'s 10 µm in 25 s. Over the 19 s script it takes at most 9 µm. Without a
  solids species it takes compound volume; water is about 60 % of the compound (an estimate).
- Two knives, Gym-Anything's (``tools.py:176-189``): the 2 in putty knife (51 mm, carries 25 mL) and the 6 in joint
  knife (152 mm, 100 mL), each a scraping store on WP15's surface-flow stores (``Store(scrape=True)``,
  Gym-Anything's carrying knife, ``coupling.py:521-596``). The blade's edge band (10 mm deep) cuts what stands above
  it into the carry and fills the lows under it from the carry, up to the edge, filling only to the board where it
  bears on it (``coupling.py:571-583``). The knives never drip (``held`` 1); lifted, each keeps Gym-Anything's stick
  over its blade's face (``stick_area``: putty 10.6 mL, joint 31.6 mL) and drops the rest as a gob, off the board.
  A full blade stops cutting (Gym-Anything spills its excess at the blade's ends, ``coupling.py:599``); the script
  never fills one. Their hawk and the knife's flex are not modelled.
- Privileged scripted strokes (no robot; Gym-Anything's task interface is robot-only, ``env.py:1-8``, and it ships
  no policy for this task): the knives are kinematic bodies
  posed each control period with the velocity of their move (:func:`strokes`). A knife bearing on the board rests on
  it (``press`` 0; the store bears within Gym-Anything's 0.05 mm, ``coupling.py:478``), so the holes fill to the
  board. The mesh's fibre tops hold a blade at 0.5 mm, Gym-Anything's support height (``paste.py:128-135``): the
  flow's ``support`` input is 0.5 mm there (the base elsewhere), so a knife on the fibres bears and fills the mesh's
  openings (the chart's base there is the fibres' volume, 0.15 mm) up to them (``coupling.py:454-458, 581``), and any
  pose whose edge band overlaps the mesh is lifted to them (:func:`supported`).
  1. The joint knife clears the mound off the mesh, bearing on the board: up to the mesh's lower edge and down to its
     upper edge (its edge band stopping at the mesh's edge), then a lane each side, its edge at the mesh's.
  2. The putty knife rides the mesh on the fibre tops (``mesh_gap``, 0.5 mm) in a lane on each half, set down and
     lifted within the mesh, and between the lanes fills the four anchor holes, a stroke each bearing on the board.
     A blade spanning the mesh's edge would rest on the fibres and leave 0.5 mm on the board beside them, past
     ``flush``; a knife lifted with more than sticks to its face drops a gob (WP15's stores, GA's
     ``coupling.py:922-938``), so the putty knife never lifts more than one lane's cut.

Each world's books gate its verdict (score None when they are open, as the rinse tasks' ``no_simulation_fault``): the
film's ledger with the knives' carry, and the knives' store ledger, to 1e-9 of the start. Compound off the board is
measured (:func:`off_board`): a lifted knife's gob, drips and releases (the chart's edges are walls, so compound
leaves the board only from a knife); Gym-Anything's gob falls on the floor's catch tiles and counts past 2e-8 m³
(``coupling.py:937-938``, ``task.py:159-166``). With the uptake on, the visibility masks read the film plus what the
paper took (its solids stay in place). The score is progress-v1, the benchmark's task-set contract; Gym-Anything's
default, the share of criteria passed, is reported beside it.

What the verdict judges (audit AB-gym-7; ``concrete_ports.PATCH_DRYWALL``): all five of their criteria from the
compound's fields. The books gate is Alloy's own, not a Gym-Anything criterion (stricter; ``alloy_gates`` in the
verdict), and gates success too. The scene differs from theirs, and each verdict says so (``scene_differs``,
``summary``): the actor, no hawk or knife flex, one temperature for every seed (theirs varies), and a coarser grid
when one is set.

Run: ``python -m alloy.tasks.concrete_drywall [--worlds W] [--device cuda:0] [--cell 0.001] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.process_specs import Process, Substrate
from alloy.plugins.surface_flow.stores import Store
from alloy.tasks import concrete_drywall_start as ds
from alloy.tasks import concrete_verdict as CV
from alloy.tasks.concrete_drywall_checks import DrywallGrid, patch_drywall
from alloy.tasks.concrete_ports import BOOKS_GATE, PATCH_DRYWALL

log = logging.getLogger("alloy.tasks.concrete_drywall")

CONTROL_DT = 0.02                       # s
WALL = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)   # quat xyzw: local x -> -x, y -> +z, z -> +y (out of the wall)
PARK_GAP = 0.05                         # m off the face: clear of the work
TOOL_SPEED = 0.5                        # m/s: the fastest knife move (in the air)
COMPOUND = dict(density=1000.0, viscosity=6.0, yield_stress=700.0)  # Gym-Anything's COMPOUND (params.py:121-122)


@dataclass(frozen=True)
class Knife:
    """A knife's blade: its edge's half width [m], carry [m³] and length from the edge [m] (``tools.py:176-189``,
    ``params.py:152-153``); its face (2 half x length) holds GA's stick when lifted (``coupling.py:933-936``)."""

    name: str
    half: float
    carry: float
    park: tuple[float, float] = (0.0, 0.0)      # x, z on the wall where it waits [m]
    length: float = 0.070

    @property
    def face(self) -> float:
        return 2.0 * self.half * self.length


PUTTY = Knife("putty_knife", 0.0255, 2.5e-5, park=(1.50, 1.20))
JOINT = Knife("joint_knife", 0.076, 1.0e-4, park=(2.10, 1.30))


@dataclass
class PatchDrywall:
    """The task's settings: grid, seed, site temperature, the strokes and the paper's uptake."""

    cell: float = 0.001                  # m: theirs; 2, 5 or 10 mm is a smoke
    seed: int = 0
    temperature: float = 21.1
    speed: float = 0.10                  # m/s along a stroke
    air: float = 0.5                     # m/s off the wall
    mesh_gap: float = ds.MESH_T          # m: the putty knife's edge over the mesh, on the fibre tops (GA's)
    press: float = 0.0                   # m: a knife bearing on the board presses this far into it (the store
                                         # bears within GA's 0.05 mm, coupling.py:478, so 0 touches every exchange)
    band: float = 0.005                  # m: the blade's edge band, half its depth along the stroke
    reach: float = 0.11                  # m: strokes run this far each side of the patch centre
    uptake: bool = False                 # the face paper's uptake (an Alloy option; Gym-Anything has none)
    sorptivity: float = 2.0e-6           # m/sqrt(s): the face paper (painting/params.py:114 cardboard, estimate)
    capacity: float = 1.5e-4             # m

    @property
    def board(self) -> ds.Wallboard:
        return ds.on_grid(ds.wallboard(self.seed, self.temperature), self.cell)


# --- the strokes ------------------------------------------------------------------------------------------------------

@dataclass
class Path:
    """A knife's pose (x, gap, z) every control period [m]: x, z on the wall, gap off its face."""

    at: tuple[float, float, float]
    poses: list[tuple[float, float, float]] = field(default_factory=list)

    def to(self, x: float, gap: float, z: float, speed: float) -> None:
        """Move straight to (x, gap, z) at ``speed``, ending exactly there."""
        p0, p1 = np.asarray(self.at), np.array([x, gap, z])
        n = max(1, int(math.ceil(float(np.linalg.norm(p1 - p0)) / (speed * CONTROL_DT) - 1e-9)))
        self.poses += [tuple(p0 + (p1 - p0) * (k / n)) for k in range(1, n)] + [(x, gap, z)]
        self.at = (x, gap, z)

    def hold(self, periods: int) -> None:
        self.poses += [self.at] * periods

    def work(self, x: float, z0: float, z1: float, gap: float, task: PatchDrywall) -> None:
        """Come down at (x, z0) to ``gap`` (straight down from the air), work along the wall to z1, lift off."""
        self.to(x, PARK_GAP, z0, task.air)
        self.to(x, gap, z0, task.air)
        self.to(x, gap, z1, task.speed)
        self.to(x, PARK_GAP, z1, task.air)


def strokes(task: PatchDrywall) -> list[list[tuple[float, float, float]]]:
    """Per control period, each knife's pose (``PUTTY`` then ``JOINT``); each waits parked while the other works.

    1. The joint knife clears the mound off the mesh, bearing on the board: up to the mesh's lower edge, down to its
       upper edge (its edge band stopping at the mesh's), then a lane each side, its edge at the mesh's.
    2. The putty knife rides the mesh at ``mesh_gap`` in a lane on each half (set down and lifted within it), and
       between them fills the four holes, a stroke each bearing on the board. So it never lifts more than its first
       lane's cut less the holes plus its second's (GA's knife keeps what sticks to its face and drops the rest when
       lifted, coupling.py:922-938)."""
    (cx, cz), r, h, b = ds.PATCH, task.reach, ds.MESH_HALF, task.band
    lo, hi, bear = cz - h, cz + h, -task.press
    joint = Path((*JOINT.park[:1], PARK_GAP, JOINT.park[1]))
    joint.hold(1)
    joint.work(cx, cz - r, lo - b, bear, task)                # its band's leading edge at the mesh's lower edge
    joint.work(cx, cz + r, hi + b, bear, task)                # and at its upper edge
    for x in (cx - h - JOINT.half, cx + h + JOINT.half):
        joint.work(x, cz - r, cz + r, bear, task)
    joint.to(JOINT.park[0], PARK_GAP, JOINT.park[1], task.air)
    putty = Path((*PUTTY.park[:1], PARK_GAP, PUTTY.park[1]))
    putty.hold(len(joint.poses))
    putty.work(cx - h + PUTTY.half, lo + b, hi - b, task.mesh_gap, task)
    for hx, hz in ds.HOLES:
        putty.work(hx, hz - 0.03, hz + 0.03, bear, task)
    putty.work(cx + h - PUTTY.half, lo + b, hi - b, task.mesh_gap, task)
    putty.to(PUTTY.park[0], PARK_GAP, PUTTY.park[1], task.air)
    joint.hold(len(putty.poses) - len(joint.poses))
    return [putty.poses, joint.poses]


# --- the scene --------------------------------------------------------------------------------------------------------

def build_scene(task: PatchDrywall, init, worlds: int = 1) -> TestScene:  # noqa: ANN001
    """The partition's face (a box whose top face is the wall, ``WALL``), the knives (kinematic, no contacts) parked
    off it, and a resting dummy sphere (a scene needs a dynamic body)."""
    import warp as wp  # noqa: PLC0415

    (xa, xb, za, zb) = ds.BOX
    hx, hy, hz = 0.5 * (xb - xa), 0.5 * (zb - za), 0.06

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.5 * (xa + xb), -hz, 0.5 * (za + zb), WALL), hx=hx, hy=hy, hz=hz,
                        label="wallboard")
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        cfg.has_shape_collision = False
        for knife in (PUTTY, JOINT):
            body = b.add_body(xform=body_xform(knife.park[0], PARK_GAP, knife.park[1], WALL), is_kinematic=True,
                              label=knife.name)
            b.add_shape_box(body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()), hx=knife.half,
                            hy=0.035, hz=0.0005, cfg=cfg)
        dummy = b.add_body(xform=body_xform(1.82, 0.5, 0.01), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=init, worlds=worlds, periods=1, control_dt=CONTROL_DT, dt=CONTROL_DT,
                     solver="xpbd", ground=True)


def make_flow(task: PatchDrywall) -> SurfaceFlow:
    """The face's chart, the compound film, the knives' scraping stores and (``uptake``) the paper's uptake."""
    tools = tuple(Applicator(body=k.name, shape="rect", size=(k.half, task.band), stiffness=1.0e3,
                             store=Store(capacity=k.carry, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0,
                                         scrape=True, stick_area=k.face))
                  for k in (PUTTY, JOINT))
    film = FilmMaterial("joint_compound", surface_tension=0.0, **COMPOUND)
    process = Process(substrates={0: Substrate(sorptivity=task.sorptivity, capacity=task.capacity)}) if task.uptake \
        else None
    return SurfaceFlow(charts=(PlaneChart("wallboard", cells=task.board.base.shape),), material=film, inertia=False,
                       max_depth=0.02, tools=tools, tool_speed=TOOL_SPEED, process=process)


# --- the run ----------------------------------------------------------------------------------------------------------

BOOKS_REL = 1e-9                        # a world's film and store books, relative to its start (atol 0)


@dataclass
class DrywallResult:
    verdicts: list[dict]
    success: int
    worlds: int
    sim_s: float
    wall_s: float
    absorbed_ml: list[float]
    carried_ml: list[list[float]]
    books_rel: list[float]
    off_ml: list[float]
    cell_m: float
    grid: str


def supported(knife: Knife, pose: tuple[float, float, float], band: float) -> tuple[float, float, float]:
    """``pose`` with its gap no lower than the mesh's fibre tops where its edge band overlaps the mesh."""
    x, gap, z = pose
    (cx, cz), h = ds.PATCH, ds.MESH_HALF
    eps = 1e-9
    over = abs(x - cx) < h + knife.half - eps and abs(z - cz) < h + band - eps
    return (x, max(gap, ds.MESH_T), z) if over else pose


def _place(sim, idx: list[np.ndarray], poses: list[tuple[float, float, float]],  # noqa: ANN001
           before: list[tuple[float, float, float]]) -> None:
    """Pose each knife (``idx``: its body per world) at (x, gap, z), facing the wall, with the velocity of its move
    from ``before`` over the period (XPBD passes kinematic bodies through unchanged)."""
    q, qd = sim.state.body_q.numpy(), sim.state.body_qd.numpy()
    for bodies, (x, gap, z), (x0, gap0, z0) in zip(idx, poses, before, strict=True):
        q[bodies, :3] = (x, gap, z)
        q[bodies, 3:] = WALL
        qd[bodies, :3] = ((x - x0) / CONTROL_DT, (gap - gap0) / CONTROL_DT, (z - z0) / CONTROL_DT)
        qd[bodies, 3:] = 0.0
    sim.state.body_q.assign(q)
    sim.state.body_qd.assign(qd)


OFF_TOOL_KEYS = ("tool_dropped", "tool_dripped", "tool_released")     # [W, tools] m³ that left a knife


def off_board(read: dict) -> np.ndarray:
    """Compound off the board per world [m³]: what left the knives (a gob, drips, releases), and any film spilled or
    drained (none here: the chart's edges are walls and it has no drains)."""
    W = np.asarray(read["spilled_volume"]).shape[0]
    tools = sum(np.asarray(read.get(k, np.zeros((W, 1)))).sum(axis=1) for k in OFF_TOOL_KEYS)
    return tools + np.asarray(read["spilled_volume"]) + np.asarray(read["drained_total_volume"])


def _grid(board: ds.Wallboard, base: np.ndarray, film: np.ndarray, off_m3: float = 0.0,
          taken: np.ndarray | None = None) -> DrywallGrid:
    """A world's grid; ``taken`` [m]: what the paper took, read as compound by the visibility masks."""
    return DrywallGrid(h=base + film, t=film if taken is None else film + taken, board=board, off_m3=off_m3)


def inputs(boards: list[ds.Wallboard]) -> dict[str, np.ndarray]:
    """The flow's inputs per world [W, C] (float32): the chart's ``base`` and GA's support height ``support``, the
    fibre tops on the mesh (``paste.py:128-135``): a blade within 0.05 mm of it bears there and fills up to it."""
    return {"base": np.stack([b.base.reshape(-1) for b in boards]).astype(np.float32),
            "support": np.stack([b.support().reshape(-1) for b in boards]).astype(np.float32)}


def run(task: PatchDrywall | None = None, *, worlds: int = 1, device: str = "cpu",
        graph: bool = False) -> DrywallResult:
    """The scripted patch in ``worlds`` worlds (world w: seed + w), then each world's verdict against its start."""
    from alloy.tasks.paint_supply import bodies  # noqa: PLC0415

    task = task or PatchDrywall()
    boards = [ds.on_grid(ds.wallboard(task.seed + w, task.temperature), task.cell) for w in range(worlds)]
    shape, flow = boards[0].base.shape, make_flow(task)
    C = shape[0] * shape[1]

    def init(sim) -> None:  # noqa: ANN001
        for name, values in inputs(boards).items():
            flow.set_input(name, values)
        flow.set_depth(np.stack([b.film.reshape(C) for b in boards]))

    sim = build_simulation(build_scene(task, init, worlds), [flow], device, graph=graph)
    idx = [bodies(sim.model, k.name) for k in (PUTTY, JOINT)]
    v0 = flow.read()["film_volume"].copy()
    paths = [[supported(k, p, task.band) for p in path] for k, path in zip((PUTTY, JOINT), strokes(task), strict=True)]
    t0 = time.perf_counter()
    before = [path[0] for path in paths]
    for i, poses in enumerate(zip(*paths, strict=True)):
        _place(sim, idx, list(poses), before)
        before = list(poses)
        sim.step_period()
        if i % 250 == 0:
            log.info("period %d of %d (%.0f s wall)", i, len(paths[0]), time.perf_counter() - t0)
    wall = time.perf_counter() - t0
    read = flow.read()
    h = flow.h.numpy().reshape(worlds, *shape)
    base = flow.base.numpy().reshape(worlds, *shape)
    off = off_board(read)
    books = np.maximum(np.abs(read["ledger"] - v0), np.abs(read["store_ledger"])) / v0
    taken = read["absorbed_depth"].reshape(worlds, *shape) if "absorbed_depth" in read else np.zeros_like(h)
    verdicts = [_verdict_dict(patch_drywall(_grid(b, base[w], h[w], float(off[w]), taken[w]),
                                            _grid(b, b.base, b.film)), float(books[w]), task.cell)
                for w, b in enumerate(boards)]
    load = read["tool_load"]
    absorbed = read.get("absorbed_volume", np.zeros(worlds))
    return DrywallResult(verdicts=verdicts, success=sum(v["success"] for v in verdicts), worlds=worlds,
                         sim_s=len(paths[0]) * CONTROL_DT, wall_s=wall,
                         absorbed_ml=[float(x) * 1e6 for x in absorbed],
                         carried_ml=[[float(x) * 1e6 for x in row] for row in load],
                         books_rel=[float(x) for x in books], off_ml=[float(x) * 1e6 for x in off], cell_m=task.cell,
                         grid=grid_label(task.cell))


def grid_label(cell: float) -> str:
    """The result's ``grid``: their 1 mm grid gives the verdict; a coarser one is a smoke, lenient in every criterion
    (a hole reads exactly only with the board around it bare: a skim beside a short hole averages it away)."""
    if cell == ds.CELL:
        return "theirs (1 mm): the verdict"
    return f"smoke: {cell * 1e3:g} mm, coarser than their 1 mm (lenient)"


def _verdict_dict(v, books: float, cell: float = ds.CELL) -> dict:  # noqa: ANN001 - concrete_metrics.Verdict
    """One world's verdict by the AB-gym rule (``concrete_verdict``); open books (over ``BOOKS_REL``, Alloy's own
    gate) make it unverifiable: no success, score None. ``cell``: the grid it was read on (theirs by default)."""
    ok = bool(books <= BOOKS_REL)
    d = CV.verdict_dict(v, PATCH_DRYWALL, gates=CV.alloy_gates(BOOKS_GATE, books_closed=ok),
                        differs=CV.grid_differs(cell, ds.CELL, "all five criteria"))
    return {**d, "verifiable": ok, "books_rel": books, "score": v.score if ok else None, "contract": v.contract,
            "progress": None if v.progress is None or not ok else v.progress.value,
            "score_legacy": v.score_legacy if ok else None, "grid": grid_label(cell)}


def parser() -> argparse.ArgumentParser:
    """The command line (its defaults are the task's: their 1 mm grid, no uptake)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cell", type=float, default=ds.CELL, help="m: theirs 1 mm; 2, 5 or 10 mm is a smoke")
    ap.add_argument("--uptake", action="store_true", help="the face paper's uptake (an Alloy option)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph", action="store_true", help="replay each control period as a captured graph")
    ap.add_argument("--json", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run(PatchDrywall(cell=args.cell, seed=args.seed, uptake=args.uptake), worlds=args.worlds, device=args.device,
              graph=args.graph)
    out = json.dumps(asdict(res), indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(out)
    print(json.dumps({"success": res.success, "worlds": res.worlds, "grid": res.grid, "wall_s": round(res.wall_s, 1)}))
    return 0 if res.success == res.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BOOKS_REL", "COMPOUND", "JOINT", "OFF_TOOL_KEYS", "PUTTY", "DrywallResult", "Knife", "Path", "PatchDrywall",
           "build_scene", "main", "make_flow", "off_board", "run", "strokes", "supported"]
