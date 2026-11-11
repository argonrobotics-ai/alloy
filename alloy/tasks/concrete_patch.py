# SPDX-License-Identifier: Apache-2.0
"""task concrete_patch: Gym-Anything's patch-flush, the first surface-flow concrete proof.

"Strike the floor patch flush with the slab and float it" (``concrete_finishing/task.py``; spec
``[internal note]`` §4). A 400 x 250 mm recess chipped 25-38 mm into a slab,
filled with repair mortar 6 min old: overfilled by 3-7 mm, a void along the west short edge (top 6-9 mm below the
slab, 18 mm wide), 2 mm smears on the slab beside the north and east sides, texture 0.6 mm. The start state is drawn
cell for cell as theirs (``layout.py:369-413``, the same seeded noise on their 1 mm grid, block-averaged to ours).

Model: one ``surface_flow`` plane chart on the slab's top face, 200 mm beyond the cut (``PATCH_MARGIN``). The
recess floor and the fill below the slab plane are the chart's ``base`` (fixed: the tools' pressures stay well under
the paste's bearing capacity; the run reports both, ``max_pressure_pa`` sampled every period against ``bearing_pa``);
the paste above it (the overfill, the smears) is the film:
repair mortar as a Bingham film (Gym-Anything's two layers, ``paste.py:49-109``). A privileged scripted carriage
(``concrete_tools``; no robot: not their task's actor, whose interface is robot-only; the carriage is their ASSISTED
probe fixture, ``probes/common.py:1-4``) works two tools:
1. the margin trowel 52D strikes off, carrying: their carrying knife (``tools.py:136-146``), a scraping surface-flow
   ``Store`` at their CARRY_CAP 0.15 L with the lifted blade's stick limit (``coupling.py:922-938``). Strips east to
   west in two rows across the patch, its sole on the slab plane (the slab surrounding the cut is the screed rail):
   it cuts what stands above it into its carry and fills the lows below it from the carry, the west void last.
   Lifted, it flies to a bucket and lets go (``release``, standing in for their turning the blade over, which keeps
   2 %, ``coupling.py:930``; the carriage could pitch the blade instead, ``stores_lift``, ~1 mL left vs 0): the gob
   falls through DC5's supply into the bucket (a tank and its pool in a pit west of the slab, its floor 0.2 m below
   the slab top: their patch is cut into the floor slab). Their patch-flush has no bucket, tub, hawk or rag
   (``layout.py:162-163``): what leaves the slab region lands as smear or on the room floor, off-region
   (``coupling.py:940-985``, ``task.py:159-167``), and their 0.15 L blade has nowhere legal to empty the 0.46 L the
   start heaps over the slab (3-7 mm proud, ``layout.py:400-405``), a known gap of the source task (their benchmark
   makes no autonomous-solution claim). Alloy adds the bucket (coordinator, 2026-11-06: "DC5 adds the sink
   patch-flush lacked"), disclosed; it eases only ``no_feather_edge``, and a miss still counts off-region (over the
   chart's edges, drained, on the floor, caught on a shape, or a full blade's overflow past the chart's edge:
   ``off_region``). The strips are sized per world from its start surface (``world_programs``), what a mason or a
   depth camera sees of the heap, never the hidden recess floor (``_row``): each under STRIP_LOAD 40 mL or one blade
   width; each row's last strip runs from the void east until its excess covers the void's deficit 1.5 times and is
   a blade width wide, and takes in a remainder too short for a strip of its own, so on seeds 0-1 it takes 46-61 mL
   (2.9-3.7 times the 15.5-16.4 mL deficit, over 2.3-3.3 blade widths; WP15's review) and lifts 30-44 mL after
   filling the void, under the ~56 mL a lifted blade keeps;
2. the wood float 144D floats the patch: passes across it at 5 N, leaving its texture (``coupling.py:1038-1042``).
The tools have no rigid contacts: they meet the slab and the paste through the film's pad (implicit, 1e7 Pa/m),
so the CPU (MuJoCo C) and GPU (MuJoCo Warp) runs share one physics for them.
The process layer (``concrete_paste``) ages the mortar and keeps texture, densification and the stiffness at last
working, under every tool (its work fields: the trowel's store too). The verdict is their seven criteria
(``concrete_metrics.patch_flush``). Disclosed: ``finished_in_time`` passes with room (the scripted run takes 81 s, so
the stiffest working is ~1.2 kPa against 30 kPa), and no criterion needs the float: without it (float_passes=0, 10 mm,
seed 0; WP15's probe) all seven pass, texture 1.00 since the start's 0.6 mm sits on their band's edge, which counts
(``params.py:180``), and the trowel marks every cell worked (``task.py:618-622``); the float raises the lip from
0.0016 to 0.037 mm.

What the verdict judges (audit AB-gym-7; ``concrete_ports.PATCH_FLUSH``): all seven criteria from the paste's fields,
but on the default 5 mm cells, not their 1 mm: the lip, the cut's perimeter and the smear are read coarser (thin
smears and spikes average away), and each verdict's ``grid`` and ``scene_differs`` say so (``cell=0.001`` is their
grid: 800 x 650 cells, a period with the tools parked 53 times a 5 mm one's wall time on a CPU, about 3 days a
world: a GPU row, not run yet). The scene also differs by the actor, Alloy's bucket and the site (one temperature
for every seed; theirs varies).

Run: ``python -m alloy.tasks.concrete_patch [--worlds W] [--device cuda:0] [--cell 0.005] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store
from alloy.plugins.surface_flow.supply import Pool
from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_metrics as MX
from alloy.tasks import concrete_verdict as CV
from alloy.tasks.concrete_pad_start import smooth_noise
from alloy.tasks.concrete_paste import ConcretePaste, SoleWork
from alloy.tasks.concrete_ports import PATCH_FLUSH
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolCarriage, ToolProgram, add_tool_body, lane

log = logging.getLogger("alloy.tasks.concrete_patch")

L, W = 0.400, 0.250                     # the recess (params.py:38)
MARGIN = 0.20                           # slab modelled beyond the cut (params.py:40)
DEPTH = (0.025, 0.038)                  # chipped depth (params.py:39)
START_AGE = 6 * 60.0                    # s at 21 C (layout.py:211)
AUTHOR_CELL, AUTHOR_MARGIN = 0.001, 0.05     # their grid (layout.py:376-377)
# tools (tools.py:64-86, 136-146): blade half extents, mass, sole work (a_tool, dens, seal)
TROWEL = dict(half=(0.0635, 0.0254, 0.001), mass=0.12, work=(1.0e-4, 0.5, 0.3))      # margin trowel 52D (2 mm)
FLOAT = dict(half=(0.203, 0.04445, 0.0095), mass=0.28, work=(5.0e-4, 0.2, 0.05))     # wood float 144D
PARK = {"trowel": (0.0, 0.6, 0.15), "float": (0.0, -0.6, 0.15)}                       # held clear of the slab
STRIKE_GAP = 0.0                        # m: the trowel's underside on the slab plane while it scrapes (HEIGHT)
CARRY_CAP = 1.5e-4                      # m³ the margin trowel's blade carries (params.py:160)
STRIP_LOAD = 4.0e-5                     # m³ a strip may take: under the lifted blade's stick limit (~5.6e-5 at 6 min)
LOW_RESERVE = 0.5                       # a row's last strip cuts the west void's deficit x (1 + this) before it
ROWS = (-0.0625, 0.0625)                # y of the scraping rows: the 127 mm blade covers the 250 mm patch
WEST_END = -L / 2 - 0.005               # the blade's west edge at a row's end: 5 mm past the cut, over the void
FLOAT_ROLL_LIFT = 0.003                 # m: a float's roll lift (params.py:136), WP13's roll cap
RELEASE_S = 0.3                         # s over the bucket letting go
PAD_STIFFNESS = 1.0e7                   # Pa/m: the tools' pads, implicit (the float's 5 N sinks it 0.014 mm)
BUCKET_FLOOR, BUCKET_HALF = (-0.57, 0.0, -0.20), (0.12, 0.12)    # Alloy's bucket (their task has none), beside
                                                                    # the slab region's west edge, on the floor


@dataclass
class PatchFlush:
    """The task's settings: cell size, seed and site temperature, and the scripted strokes."""

    cell: float = 0.005                  # m: 1, 2, 5 or 10 mm (theirs 1 mm; a light CPU run uses 5 mm)
    seed: int = 0
    temperature: float = 21.1
    strike_speed: float = 0.1            # m/s along a strip
    float_force: float = 5.0             # N on the float (weight 2.7 N; the slab's soft contact gives 0.2 mm)
    float_passes: int = 2
    air_speed: float = 0.5               # m/s for moves off the work

    @property
    def size(self) -> tuple[float, float]:
        return (L + 2 * MARGIN, W + 2 * MARGIN)

    @property
    def cells(self) -> tuple[int, int]:
        return (int(round(self.size[0] / self.cell)), int(round(self.size[1] / self.cell)))


# --- start state (layout.py:369-413) --------------------------------------------------------------------------------

def start_fields(seed: int) -> dict[str, np.ndarray]:
    """Their patch on their 1 mm grid over our 200 mm margin: floor b, fill top, smear, recess (layout.py:383-410).
    Coordinates are relative to the patch centre; arrays are [nu, nv] along x, y."""
    rng = np.random.default_rng(3000 + seed)
    cell, m0, m = AUTHOR_CELL, AUTHOR_MARGIN, MARGIN
    k = int(round((m - m0) / cell))
    nu, nv = int(round((L + 2 * m) / cell)), int(round((W + 2 * m) / cell))
    u = (np.arange(nu) + 0.5) * cell - (L / 2 + m)
    v = (np.arange(nv) + 0.5) * cell - (W / 2 + m)
    X, Y = np.meshgrid(u, v, indexing="ij")
    recess = (np.abs(X) < L / 2) & (np.abs(Y) < W / 2)
    shape0 = (int(round((L + 2 * m0) / cell)), int(round((W + 2 * m0) / cell)))

    def drawn(field: np.ndarray) -> np.ndarray:
        out = np.zeros((nu, nv))
        out[k:k + shape0[0], k:k + shape0[1]] = field
        return out

    d0, d1 = DEPTH
    depth = d0 + (d1 - d0) * (0.5 + 0.35 * drawn(smooth_noise(rng, shape0, 12)).clip(-1.4, 1.4))
    floor = np.where(recess, -depth, 0.0)
    over = 0.003 + 0.004 * (0.5 + 0.5 * drawn(smooth_noise(rng, shape0, 10)).clip(-1, 1))
    top = np.where(recess, over, 0.0)
    void = recess & (X < -L / 2 + 0.018)
    top = np.where(void, -0.006 - 0.003 * (X + L / 2) / 0.018, top)
    smear = (~recess) & (((Y > W / 2) & (Y < W / 2 + 0.025) & (np.abs(X) < L / 2 - 0.05))
                         | ((X > L / 2) & (X < L / 2 + 0.025) & (np.abs(Y) < W / 2 - 0.03)))
    smear_t = np.where(smear, 0.002 * (0.6 + 0.4 * drawn(smooth_noise(rng, shape0, 3)).clip(-1, 1)), 0.0)
    return {"floor": floor, "top": top, "smear": smear_t, "recess": recess.astype(float),
            "smear_mask": smear.astype(float)}


def _block(field: np.ndarray, n: tuple[int, int]) -> np.ndarray:
    """Block-average a 1 mm field onto n cells (a whole number of 1 mm cells per side)."""
    fx, fy = field.shape[0] // n[0], field.shape[1] // n[1]
    if fx * n[0] != field.shape[0] or fy * n[1] != field.shape[1]:
        raise ValueError(f"cell must divide the {field.shape} mm grid evenly, got {n} cells")
    return field.reshape(n[0], fx, n[1], fy).mean(axis=(1, 3))


def start_state(task: PatchFlush, seed: int | None = None) -> dict[str, np.ndarray]:
    """On our grid: base (the fill below the slab plane, or the floor where the void is lower), film (paste above
    the slab plane: overfill and smears), recess and paste masks."""
    f = {k: _block(v, task.cells) for k, v in start_fields(task.seed if seed is None else seed).items()}
    recess = f["recess"] > 0.5
    base = np.where(recess, np.minimum(f["top"], 0.0), 0.0)
    film = np.where(recess, np.maximum(f["top"], 0.0), f["smear"])
    paste = recess | (f["smear_mask"] > 0.5)
    return {"base": base, "film": film, "floor": f["floor"], "recess": recess, "paste": paste}


# --- scene and strokes -----------------------------------------------------------------------------------------------

def build_scene(task: PatchFlush):
    sx, sy = task.size

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.025), hx=sx / 2, hy=sy / 2, hz=0.025, label="slab")
        add_tool_body(b, "trowel", TROWEL["half"], TROWEL["mass"], PARK["trowel"], collide=False)
        add_tool_body(b, "float", FLOAT["half"], FLOAT["mass"], PARK["float"], collide=False)
    return build


def _row(task: PatchFlush, st: dict, y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell x (east to west), and under the blade's width at each: the excess above the slab plane (``film``) and the
    deficit below it (the low void at the west cut, ``base`` < 0), both m³, for a row at ``y``."""
    x = (np.arange(task.cells[0]) + 0.5) * task.cell - task.size[0] / 2
    yc = (np.arange(task.cells[1]) + 0.5) * task.cell - task.size[1] / 2
    under = np.abs(yc - y) < TROWEL["half"][0]
    vol = st["film"][:, under].sum(axis=1) * task.cell ** 2
    low = -np.minimum(st["base"][:, under], 0.0).sum(axis=1) * task.cell ** 2
    order = np.argsort(-x)
    return x[order], vol[order], low[order]


def _last_strip_east(task: PatchFlush, x: np.ndarray, vol: np.ndarray, low: np.ndarray) -> float:
    """The east edge of a row's last strip: far enough east that the excess it cuts covers the void's deficit with
    LOW_RESERVE to spare (it fills the void from its carry, then lifts what is left), and one blade width at least."""
    need = (1.0 + LOW_RESERVE) * float(low.sum())
    west = WEST_END + 2 * TROWEL["half"][1]
    got = 0.0
    for xi, v in zip(x[::-1], vol[::-1], strict=True):            # west to east
        if xi < WEST_END:
            continue
        got += v
        edge = xi + 0.5 * task.cell
        if got >= need and edge >= west:
            return edge
    return math.inf                                                # the whole row is one strip


def strips(task: PatchFlush, st: dict) -> list[tuple[float, float, float]]:
    """The scraping strips of one start, as (y, blade centre from x, to x): each row east to west, cut where the
    excess the blade takes would pass STRIP_LOAD (one blade width at least), the last ending over the void and long
    enough that its carry fills the void's deficit (``_last_strip_east``); then the north smear in one strip. Sized
    from this world's start, as a mason sees the heap (``world_programs``)."""
    half = TROWEL["half"][1]
    out = []
    for y in ROWS:
        x, vol, low = _row(task, st, y)
        east, load = L / 2 + 0.025 + 0.01, 0.0          # the blade's east edge: past the east smear
        last = min(_last_strip_east(task, x, vol, low), east)
        start = east
        for xi, v in zip(x, vol, strict=True):
            if xi > east or xi < WEST_END:
                continue
            edge = xi + 0.5 * task.cell
            if edge <= last:
                break
            if load + v > STRIP_LOAD and start - edge >= 2 * half:
                if edge - last < 2 * half:              # too little left before the last strip: it starts here
                    last = edge
                    break
                out.append((y, start - half, edge + half))
                start, load = edge, 0.0
            load += v
        if start - last >= 2 * half:                    # the strip before the last (else the last takes it)
            out.append((y, start - half, last + half))
            start = last
        out.append((y, start - half, WEST_END + half))
    y_n = W / 2 + 0.0125
    out.append((y_n, L / 2 - 0.05 + half + 0.01, -(L / 2 - 0.05) - half - 0.01))
    return out


def _fly(keys: list[Stroke], t: float, p: tuple[float, float, float], yaw: float, speed: float,
         release: float = 0.0) -> float:
    t += math.dist(keys[-1].pos, p) / speed + 0.2
    keys.append(Stroke(t, p, yaw=yaw, mode=HEIGHT, release=release))
    return t


def trowel_program(task: PatchFlush, st: dict) -> ToolProgram:
    """Per strip: set the blade down on the slab plane at its east end, scrape west (the blade cuts what stands
    above it into its carry and fills the lows below it, the void, from the carry), lift, fly over the bucket and
    let go (``release``); then park."""
    across = math.pi / 2                                   # the blade's 127 mm across the row
    zt = TROWEL["half"][2] + STRIKE_GAP
    over = (BUCKET_FLOOR[0], BUCKET_FLOOR[1], 0.10)
    keys = [Stroke(0.0, PARK["trowel"], yaw=across, mode=HEIGHT)]
    t = 0.0
    for y, xa, xb in strips(task, st):
        t = lane(keys, t, (xa, y), (xb, y), zt, across, 0.0, task.strike_speed, task.air_speed, mode=HEIGHT)
        t = _fly(keys, t, over, across, task.air_speed, release=1.0)
        t = _fly(keys, t + RELEASE_S, over, across, task.air_speed)
    _fly(keys, t, PARK["trowel"], across, task.air_speed)
    return ToolProgram("trowel", tuple(keys))


def float_program(task: PatchFlush, t0: float) -> tuple[ToolProgram, float]:
    """After the trowel (``t0``): passes east to west across the patch at ``float_force``; then park."""
    across = math.pi / 2
    zf = FLOAT["half"][2]
    keys = [Stroke(0.0, PARK["float"], yaw=across, mode=HEIGHT), Stroke(t0, PARK["float"], yaw=across, mode=HEIGHT)]
    x0, x1 = L / 2 + FLOAT["half"][1] + 0.03, -(L / 2 + 0.03) - FLOAT["half"][1]
    t = t0
    for _ in range(task.float_passes):
        t = lane(keys, t, (x0, 0.0), (x1, 0.0), zf, across, task.float_force, task.strike_speed, task.air_speed)
    t += 3.0
    keys.append(Stroke(t, PARK["float"], yaw=across, mode=HEIGHT))
    return ToolProgram("float", tuple(keys)), t + 0.5


def bucket() -> dict:
    """Alloy's bucket (DC5's supply: a tank and its pool, off the slab region): what the trowel lets go over it
    falls in; a miss lands on the slab (smear) or the floor (off-region). Their patch-flush has no sink."""
    hx, hy = BUCKET_HALF
    return {"nodes": [{"name": "bucket", "medium": "tank", "volume": 0.0, "area": 4.0 * hx * hy}], "edges": [],
            "pools": [Pool("bucket", floor=BUCKET_FLOOR, half=BUCKET_HALF)]}


def plugins(task: PatchFlush, sts: list[dict]) -> tuple[ToolCarriage, SurfaceFlow, ConcretePaste, float]:
    mat = CM.RepairMortar()
    tau0 = float(mat.tau_bulk(START_AGE * CM.masco_factor(task.temperature), task.temperature))
    film = FilmMaterial("repair_mortar", density=mat.rho, viscosity=mat.mu, yield_stress=tau0, surface_tension=0.0)
    trowel = Applicator(body="trowel", shape="rect", size=TROWEL["half"][:2], center=(0.0, 0.0, -TROWEL["half"][2]),
                        stiffness=PAD_STIFFNESS, damping=0.0, implicit=True,
                        store=Store(capacity=CARRY_CAP, retained=0.0, held=1.0, split=1.0, drip=0.0, scrape=True,
                                    stick_area=4.0 * TROWEL["half"][0] * TROWEL["half"][1]))
    flt = Applicator(body="float", shape="rect", size=FLOAT["half"][:2], center=(0.0, 0.0, -FLOAT["half"][2]),
                     stiffness=PAD_STIFFNESS, damping=0.0, implicit=True, roll_lift=FLOAT_ROLL_LIFT)
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=task.cells),), material=film, inertia=False, max_depth=0.08,
                       tools=(trowel, flt), tool_speed=0.4, per_cell_rheology=True, network=bucket())
    sets = []
    for st in sts:
        tp = trowel_program(task, st)
        fp, _ = float_program(task, tp.duration + 0.5)
        sets.append((tp, fp))
    carriage = ToolCarriage(programs=sets[0], world_programs=tuple(sets), flow=flow)
    work = tuple(SoleWork(name, *spec["work"], half=spec["half"][:2]) for name, spec in (("trowel", TROWEL),
                                                                                        ("float", FLOAT)))
    layer = ConcretePaste(material="repair_mortar", temperature=task.temperature, tools=work, flow=flow,
                          carriage=carriage)
    return carriage, flow, layer, carriage.duration + 0.5


# --- the run and the verdict -----------------------------------------------------------------------------------------

@dataclass
class PatchResult:
    verdicts: list[dict]
    success: int
    worlds: int
    sim_s: float
    wall_s: float
    bucket_l: list[float]
    off_region_l: list[float]
    released_l: list[float]
    strips: list[int]
    spilled_l: list[float]
    cells_over_depth: int
    faces_at_cap: int
    ledger_rel: float
    store_ledger_m3: float
    max_pressure_pa: float
    bearing_pa: float
    cell_m: float


def _start_grid(task: PatchFlush, st: dict) -> MX.PasteGrid:
    z = np.zeros(task.cells)
    return MX.PasteGrid(h=st["base"] + st["film"], t=st["film"] + np.maximum(st["base"] - st["floor"], 0.0),
                        cell=task.cell, paste_mask=st["recess"], a=np.where(st["recess"], 6e-4, 0.0), D=z, tau_w=z,
                        worked=z)


def off_region(rd: dict) -> np.ndarray:
    """[W] m³ of paste that left the slab region other than into the bucket: over the chart's edges, drained, fallen
    on the floor (the supply's lost), caught on a shape, or a full blade's overflow past the chart's edge (WP13's
    end bands, ``tool_overflow_off``): their off-region catch (task.py:159-167)."""
    W = rd["spilled_volume"].shape[0]
    zero = np.zeros(W)
    over = np.asarray(rd.get("tool_overflow_off", np.zeros((W, 1)))).sum(axis=1)
    return (rd["spilled_volume"] + rd.get("drained_total_volume", zero) + rd.get("supply_lost", zero)
            + rd.get("supply_caught", zero) + over)


def grids(task: PatchFlush, flow: SurfaceFlow, layer: ConcretePaste, starts: list[dict]) -> list[MX.PasteGrid]:
    """The verdict's grids; what left the slab region, except into the bucket, is their off-region volume."""
    n = task.cells
    off = off_region(flow.read())
    h = flow.h.numpy().reshape(-1, *n)
    base = flow.base.numpy().reshape(-1, *n)
    r = {k: v.reshape(-1, *n) for k, v in layer.read().items() if v.ndim == 2 and v.shape[1] == n[0] * n[1]}
    out = []
    for w in range(h.shape[0]):
        st = starts[w]
        out.append(MX.PasteGrid(h=base[w] + h[w], t=h[w] + np.maximum(base[w] - st["floor"], 0.0), cell=task.cell,
                                paste_mask=st["recess"], a=r["a"][w], D=r["D"][w], tau_w=r["tau_w"][w],
                                worked=r["worked"][w], off_m3=float(off[w])))
    return out


def run(task: PatchFlush | None = None, *, worlds: int = 1, device: str = "cpu",
        npz: str | None = None) -> PatchResult:
    task = task or PatchFlush()
    sts = [start_state(task, task.seed + w) for w in range(worlds)]            # world w: seed + w
    carriage, flow, layer, duration = plugins(task, sts)
    sim = build_simulation(TestScene(build=build_scene(task), ground=False, worlds=worlds, dt=2.0e-3),
                           [carriage, flow, layer], device)
    C = task.cells[0] * task.cells[1]
    flow.set_input("base", np.stack([s["base"].reshape(C) for s in sts]).astype(np.float32))
    flow.set_depth(np.stack([s["film"].reshape(C) for s in sts]))
    age = START_AGE * CM.masco_factor(task.temperature)
    layer.set_start(paste=np.stack([s["paste"].reshape(C) for s in sts]).astype(np.int32), age=age,
                    a=np.stack([np.where(s["paste"], 6.0e-4, 0.0).reshape(C) for s in sts]))
    starts = [_start_grid(task, s) for s in sts]
    v0 = flow.read()["film_volume"]
    periods = int(math.ceil(duration / 0.02))
    max_p, over, cap, t0 = 0.0, 0, 0, time.perf_counter()
    for i in range(periods):
        sim.step_period()
        if i % 500 == 0:
            log.info("period %d of %d (%.0f s sim, %.0f s wall)", i, periods, i * 0.02, time.perf_counter() - t0)
        load = carriage.load.numpy().max(axis=0)          # every period: the largest pad pressure
        max_p = max(max_p, float(load[0] / (4 * TROWEL["half"][0] * TROWEL["half"][1])),
                    float(load[1] / (4 * FLOAT["half"][0] * FLOAT["half"][1])))
        if i % 25 == 0:
            hr = flow.read()
            over, cap = max(over, int(hr["cells_over_depth"].max())), max(cap, int(hr["faces_at_cap"].max()))
    wall = time.perf_counter() - t0
    read = flow.read()
    ends = grids(task, flow, layer, sts)
    verdicts = [MX.patch_flush(g, s) for g, s in zip(ends, starts, strict=True)]
    if npz:
        np.savez_compressed(npz, h=np.stack([g.h for g in ends]), t=np.stack([g.t for g in ends]),
                            a=np.stack([g.a for g in ends]), worked=np.stack([g.worked for g in ends]),
                            tau_w=np.stack([g.tau_w for g in ends]), recess=sts[0]["recess"],
                            h0=np.stack([g.h for g in starts]), cell=task.cell)
    tau_b = float(CM.RepairMortar().tau_bulk(age + duration, task.temperature))
    out = [verdict_dict(v, task.cell) for v in verdicts]
    return PatchResult(verdicts=out, success=sum(d["success"] for d in out), worlds=worlds, sim_s=periods * 0.02,
                       wall_s=wall,
                       bucket_l=[float(x) * 1e3 for x in read["supply_amount"][:, 0]],
                       off_region_l=[float(x) * 1e3 for x in off_region(read)],
                       released_l=[float(x) * 1e3 for x in read["tool_released"].sum(axis=1)],
                       strips=[len(strips(task, st)) for st in sts],
                       spilled_l=[float(x) * 1e3 for x in read["spilled_volume"]], cells_over_depth=over,
                       faces_at_cap=cap,
                       ledger_rel=float((np.abs(read["ledger"] - v0) / np.maximum(v0, 1e-12)).max()),
                       store_ledger_m3=float(np.abs(read["store_ledger"]).max()),
                       max_pressure_pa=max_p, bearing_pa=CM.NC * tau_b, cell_m=task.cell)


GRID_READS = ("no_lip", "no_hard_lip", "against_the_cut", "no_feather_edge")   # the criteria a coarse cell eases


def verdict_dict(v: MX.Verdict, cell: float) -> dict:
    """One world's verdict by the AB-gym rule (``concrete_verdict``), with the grid it was read on (theirs: 1 mm): on
    a coarser one the criteria of ``GRID_READS`` are judged by another measure, and say so."""
    differs = CV.grid_differs(cell, AUTHOR_CELL, ", ".join(GRID_READS))
    d = CV.verdict_dict(v, PATCH_FLUSH, differs=differs)
    for c in (d["criteria"][k] for k in GRID_READS if differs):
        c["note"] = "; ".join([f"read on {cell * 1e3:g} mm cells, not their 1 mm", *filter(None, [c.get("note")])])
    d["grid"] = f"{cell * 1e3:g} mm cells, coarser than their 1 mm (lenient)" if differs else "theirs (1 mm)"
    return CV.counted(d, PATCH_FLUSH, differs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cell", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--npz", default=None, help="final fields per world (h, t, a, worked, tau_w) and the start")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run(PatchFlush(cell=args.cell, seed=args.seed), worlds=args.worlds, device=args.device, npz=args.npz)
    out = json.dumps(asdict(res), indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(out)
    print(out)
    return 0 if res.success == res.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
