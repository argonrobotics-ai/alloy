# SPDX-License-Identifier: Apache-2.0
"""cut-in-trim: with the angled sash brush, cut in a band at least 40 mm wide along the left side of the window casing
and round the outlet plate, without paint on the casing or the plate (Gym-Anything's painting-tasks-v1 cut-in-trim,
``PT/layout.py:99-101``).

The wall (:mod:`alloy.tasks.trim_layout`) is two cropped charts at Gym-Anything's 2 mm: the casing band and the
outlet ring. The wall carries its patchy first coat of naval, dry, over the P-shade primer (seed 0); the task lays the
finish coat. The brush (:mod:`alloy.tasks.trim_sash`) is the sash on the surface-flow stores, loaded from the cut
pail, kinematic until a robot holds it. The store follows one chart, so each crop has its own brush body (v1, agreed
with WP15): the script loads one, cuts in the casing, then loads the other for the plate (the same brush, reloaded).
The film then settles ``SETTLE`` s, drying and soaking into the primed drywall (the process stage), and
:func:`alloy.tasks.trim_verdict.evaluate` judges it.

The script keeps the brush's head clear of the trim, as a hand must: its edge ``clearance`` off the casing's and the
plate's lines, and on the casing its head above the stool (the metering store lays under its whole contact as it
slides, so the band reaches the stool's line). It reloads after ``trim_sash.run_length`` of band, before the store
runs short; the metered film (100 µm wet) is over the 61 µm that hides where the first coat is thinnest
(:func:`trim_verdict.wet_to_hide`). The brushes' drips are split by where the brush was: into the pail, within 5 cm
of the wall (onto the trim below: counted in ``protected``) or elsewhere (the drop cloth).

Example::

    python -m alloy.tasks.trim_cut_in --casing-top 1.213 --out cut_in.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from alloy.tasks import paint_process as pp
from alloy.tasks import trim_sash as ts
from alloy.tasks import trim_verdict as tv
from alloy.tasks.paint_scene import chart_scene
from alloy.tasks.trim_layout import PAIL, TrimLayout

SETTLE = 60.0                   # s, as roll-strip (alloy.tasks.paint_roll_strip.SETTLE)
CLEARANCE = 5e-4                # m: the brush's edge this far off the line
OVERLAP = 0.02                  # m: a reloaded stroke starts this far back over the last one
PARK = (-0.4, -0.6, 1.2)        # m: where an idle brush waits (off both crops)
NEAR_WALL = 0.05                # m: a drip this close to the wall lands on the trim below
Stroke = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Plan:
    """The script's choices: the brush's edge off the line (< 0: over it, into the trim), and a pause [s] before each
    reload after the first (a long one lets the last stroke's edge set: a lap)."""

    clearance: float = CLEARANCE
    pause: float = 0.0


def casing_strokes(layout: TrimLayout, plan: Plan) -> list[Stroke]:
    """Up the casing band, its head above the stool, in segments of at most a load's run (:func:`ts.run_length`)."""
    b = layout.casing_band()
    x = layout.casing_x0() - plan.clearance - 0.5 * ts.WIDTH
    z0 = b.z0 + plan.clearance + 0.5 * ts.PATCH
    n = int(np.ceil((b.z1 - z0) / ts.run_length() - 1e-9))
    edges = np.linspace(z0, b.z1, n + 1)
    return [((x, float(lo) - (OVERLAP if k else 0.0)), (x, float(hi)))
            for k, (lo, hi) in enumerate(zip(edges, edges[1:]))]


def outlet_strokes(layout: TrimLayout, plan: Plan) -> list[Stroke]:
    """Round the plate: up its left side and down its right over the ring's height, then along its top and bottom
    between the sides (the head turned), each with the brush's edge ``plan.clearance`` off the plate."""
    p, r = layout.plate(), layout.outlet_band()
    d = plan.clearance + 0.5 * ts.WIDTH
    return [((p.x0 - d, r.z0), (p.x0 - d, r.z1)), ((p.x0, p.z1 + d), (p.x1, p.z1 + d)),
            ((p.x1 + d, r.z1), (p.x1 + d, r.z0)), ((p.x1, p.z0 - d), (p.x0, p.z0 - d))]


def loads(strokes: list[Stroke], run: float) -> list[list[Stroke]]:
    """The strokes in order, a new load whenever the next would take the brush past ``run`` [m] since its dip."""
    out: list[list[Stroke]] = []
    used = np.inf
    for a, b in strokes:
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if used + length > run + 1e-9:
            out.append([])
            used = 0.0
        out[-1].append((a, b))
        used += length
    return out


def job_poses(layout: TrimLayout, plan: Plan, kind: str) -> list[ts.Pose]:
    """One job's brush, from the park and back: a load (a pause first, if planned, after the first), its strokes."""
    strokes = casing_strokes(layout, plan) if kind == "casing" else outlet_strokes(layout, plan)
    level = PAIL["fill"] / (4.0 * layout.pail().half[0] ** 2)
    at, work = np.asarray(PARK, np.float64), []
    for k, group in enumerate(loads(strokes, ts.run_length())):
        if k and plan.pause > 0.0:
            work += ts.hold(at, plan.pause, ts.wall_quat(0.0))
        work += ts.load(at, layout.pail_floor(), level)
        for a, b in group:
            work += ts.stroke(work[-1].pos, a, b, layout.face_y)
        at = work[-1].pos
    return work + ts.line(at, PARK, ts.TRAVEL, ts.wall_quat(0.0))


def script(layout: TrimLayout, plan: Plan) -> Iterator[tuple[ts.Pose, ...]]:
    """Per control period, every job's brush pose (``layout.jobs`` order): one brush works while the others wait at
    the park, job after job."""
    park = ts.hold(PARK, ts.DT, ts.wall_quat(0.0))[0]
    for j, kind in enumerate(layout.jobs):
        for pose in job_poses(layout, plan, kind):
            yield tuple(pose if i == j else park for i in range(len(layout.jobs)))


def body_name(kind: str) -> str:
    return f"brush_{kind}"


def brush_scene(layout: TrimLayout, init):  # noqa: ANN001, ANN201 - a harness TestScene
    """``paint_scene.chart_scene`` of the wall crops, plus a kinematic brush per job at the park (its head at the
    sash's 0.10 kg, no collision)."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.harness.scene import body_xform  # noqa: PLC0415

    base = chart_scene([c.box(layout.face_y) for c in layout.crops()], init, ts.DT)
    head = ts.head_half()

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        base.build(b)
        for kind in layout.jobs:
            tool = b.add_body(xform=body_xform(*PARK), is_kinematic=True, label=body_name(kind))
            b.add_shape_box(tool, hx=head[0], hy=head[1], hz=head[2], xform=body_xform(0.0, 0.0, head[2]),
                            cfg=newton.ModelBuilder.ShapeConfig(density=ts.head_density(),
                                                                has_shape_collision=False))  # fmt: skip

    return dataclasses.replace(base, build=build)


def make_flow(layout: TrimLayout, tools: bool = True, initial: float = 0.0):  # noqa: ANN201 - surface_flow.SurfaceFlow
    """The film on the crops, a brush per crop (its store ``initial``-loaded) and the pail (``tools``). ``tool_speed``
    is the stroke speed: the brushes move faster only off the wall, where they touch no film (it bounds the film step:
    a cell a step)."""
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.plugins.surface_flow.process_specs import Substrate  # noqa: PLC0415

    settings = pp.film_settings(layout.paint, substrate="drywall_primed")
    s, cap = pp.SUBSTRATES["drywall_primed"]
    crops = layout.crops()
    settings["process"] = dataclasses.replace(settings["process"],
                                              substrates=dict.fromkeys(range(len(crops)), Substrate(s, cap)))
    extra = {}
    if tools:
        extra = {"tools": tuple(ts.sash(layout.paint, k, body_name(c.kind), initial) for k, c in enumerate(crops)),
                 "trays": (layout.pail(),), "tool_speed": ts.SPEED}
    return SurfaceFlow(charts=tuple(PlaneChart(c.name, cells=c.cells) for c in crops), inertia=False,
                       max_depth=4e-4, **extra, **settings)  # fmt: skip


def seal(flow, layout: TrimLayout) -> None:  # noqa: ANN001 - SurfaceFlow
    """Each crop's trim or plate footprint as a region of its own (no film crosses its edge)."""
    for k, crop in enumerate(layout.crops()):
        local = np.asarray(flow.cell_centres(k))
        if not np.allclose(local[..., :2], tv.crop_centres(crop), atol=1e-6):
            raise RuntimeError(f"cut-in-trim: crop {crop.name}'s cells disagree with the film's chart {k}")
        flow.set_regions(layout.regions(crop, crop.world_xz(local)), chart=k)


@dataclass
class Drips:
    """The brushes' drips [m³] by where the brush was: over the pail, within ``NEAR_WALL`` of the wall, elsewhere."""

    into_pail: float = 0.0
    near_wall: float = 0.0
    elsewhere: float = 0.0

    def add(self, volume: float, pos: np.ndarray, layout: TrimLayout) -> None:
        px, py, _ = layout.pail_floor()
        half = layout.pail().half[0]
        if abs(pos[0] - px) <= half and abs(pos[1] - py) <= half:
            self.into_pail += volume
        elif pos[1] > layout.face_y - NEAR_WALL:
            self.near_wall += volume
        else:
            self.elsewhere += volume


def run(layout: TrimLayout | None = None, plan: Plan | None = None, settle: float = SETTLE,
        device: str = "cpu") -> dict:
    """The kinematic cut-in on a wall-only scene, then ``settle`` s, then the verdict (plus the drips and wall time)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run

    layout, plan = layout or TrimLayout(), plan or Plan()
    flow = make_flow(layout)
    t0 = time.perf_counter()
    sim = build_simulation(brush_scene(layout, lambda sim: seal(flow, layout)), [flow], device)
    bodies = [list(sim.model.body_label).index(body_name(kind)) for kind in layout.jobs]
    drips, last, periods = Drips(), np.zeros(len(bodies)), 0
    for poses in script(layout, plan):
        for body, pose in zip(bodies, poses):
            ts.place(sim, body, pose)
        sim.step_period()
        now = flow.tool_dripped.numpy()[0].astype(np.float64)
        for k, pose in enumerate(poses):
            drips.add(float(now[k] - last[k]), pose.pos, layout)
        last, periods = now, periods + 1
    for _ in range(int(round(settle / ts.DT))):
        sim.step_period()
    read = flow.read()
    drips.elsewhere += float(read["tool_dripped"][0].sum()) - float(last.sum())
    v = tv.evaluate(layout, read, near_wall_drips=drips.near_wall)
    return {**v, "drips_mL": {k: round(x * 1e6, 4) for k, x in dataclasses.asdict(drips).items()},
            "picked_mL": round(float(read["tool_picked"][0].sum()) * 1e6, 3), "periods": periods,
            "wall_s": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--casing-top", type=float, default=None, help="cut the casing band at this height [m]")
    p.add_argument("--jobs", default="casing,outlet", help="the bands to cut in: casing, outlet or both")
    p.add_argument("--clearance", type=float, default=CLEARANCE, help="the brush's edge off the line [m]")
    p.add_argument("--pause", type=float, default=0.0, help="a pause before each reload after the first [s]")
    p.add_argument("--settle", type=float, default=SETTLE)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    layout = TrimLayout(casing_top=a.casing_top, jobs=tuple(a.jobs.split(",")))
    v = run(layout, Plan(clearance=a.clearance, pause=a.pause), a.settle, a.device)
    text = json.dumps(v, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)
    return 0 if v["succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLEARANCE", "Drips", "Plan", "SETTLE", "body_name", "brush_scene", "casing_strokes", "job_poses", "loads",
           "make_flow", "outlet_strokes", "run", "script", "seal"]
