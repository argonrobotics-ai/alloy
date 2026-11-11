# SPDX-License-Identifier: Apache-2.0
"""behind-radiator: "With the long-reach dog-leg brush (load it from the cut pail), paint the wall behind the
radiator -- the strip between the baseboard and the window apron -- to full hide, without getting paint on the radiator
or the trim." (Gym-Anything's painting wall preset, ``PT/layout.py:102-105``; its verdict in
:mod:`alloy.tasks.radiator_metrics`.)

The scene (:mod:`alloy.tasks.radiator_scene`): the wall behind the radiator as a surface-flow chart, the radiator's
top and the baseboard's ledge as protected charts, the radiator and the trim as static boxes. The paint is the latex
eggshell card in naval over grey primer; the wall already has Gym-Anything's first coat (43 µm dry, ±15 % patchy:
:func:`first_coat`), which hides nowhere on its own, so the brushed coat must bring the dry film to hide. The film
carries the paint's solids and dries through the process stage (PVA-primed drywall takes its solvent); its yield
stress is the paint's at rest (2 Pa), times Moore's structure, which a brush pass breaks and which rebuilds over 20 s
(Gym-Anything's ``t_r``, ``PT/params.py:36``): a fresh coat flows, then stands. Drying raises its viscosity
(Krieger-Dougherty, φ_m 0.64).

The brush (:mod:`alloy.tasks.radiator_brush`) is a store tool on a kinematic body following
:mod:`alloy.tasks.radiator_script`; it loads from the pail (a store tray). Disclosed:
- no robot holds it: no link marks the wet coat, and marks by the brush's ferrule or handle (Gym-Anything's
  ``no_marks`` counts those too) are not tracked either, so ``no_marks`` is not judged (None in the verdict, and
  nothing off the progress); the hand is a 35 mm clearance rule;
- an applicator lays only on its own chart, so the brush touching the radiator would lay nothing there; the script
  keeps every tool part clear of the radiator and the trim (tested), and paint reaches them only as drips;
- the store's drips fall through the supply's stage drips (DC5): over the pail back into it (a pool on its own
  tank, not the tray the brush loads from), onto the wall or the protected charts (judged), onto the radiator's or
  the trim's shapes (caught: protected's sinks, as Gym-Anything's), or onto the floor (booked lost, with a drip's
  cone past a chart's edge: protected's sinks too, worst case, since Gym-Anything's floor is protected;
  ``drips_mL`` says where they went). Drops still falling when the brush stops land during the settle.

Example::

    from alloy.tasks.radiator_task import run
    print(run()["succeeded"])
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from alloy.plugins.surface_flow.host_api import H_WET
from alloy.tasks import paint_metrics as pm
from alloy.tasks import paint_process as pp
from alloy.tasks import radiator_brush as rb
from alloy.tasks import radiator_metrics as rm
from alloy.tasks import radiator_scene as rs
from alloy.tasks import radiator_script as scr
from alloy.tasks.paint_materials import COLOURS, film_material, optics, paint
from alloy.tasks.paint_optics import srgb8_to_linear

PAINT, COLOUR, SUBSTRATE = "latex_eggshell", "naval", "primer_grey"
FIRST_COAT = 43e-6              # m dry, on the wall before the task (Gym-Anything's, patchy: first_coat)
SETTLE_MAX = 1800.0             # s: the settle runs until every wet band cell has set, at most 30 min of film time
SETTLE_CHUNK = 2.0              # s between checks (Gym-Anything: PT/task.py:4-7, PT/params.py:239, 249-250)
KRIEGER_PHI_M = 0.64
OWN_FLOW = ("judged by the port's own measure of flowed paint, stricter than Gym-Anything's: a cell's dry film less "
            "the solids the tools laid on it, net of those they took, so paint that crept in and was brushed back "
            "out still counts in full (theirs lowers it by the tool's take); equal when no tool takes film back")
OWN_SINKS = ("judged by the port's own measure, looser than Gym-Anything's for a thin smear in a sink: dry film over "
             "1 µm on its two protected charts (the radiator's top and the baseboard's ledge) as theirs, but film "
             "off a chart's open edge, drips caught by the radiator's or the trim's shapes and drips lost to the "
             "floor count only as their volume spread 100 µm thick (theirs: the larger of that and the sink's 4 mm "
             "voxels holding 1 µm)")
SCENE = ("the band, the pocket and the settle (until the film sets, up to 1800 s) are Gym-Anything's; the wall chart "
         "has 4 mm cells (theirs: 2 mm) and is cropped to the band's surroundings; a free-moving long-reach brush "
         "on a script paints, its grip point kept 35 mm outside the radiator's box as a hand would be while the "
         "head reaches into the gap, and no tool part touches the radiator or the trim (theirs: the FR3's hand "
         "holds the brush at the handle's end); the brush is a flat pad with one paint store, metered to their "
         "brushed film at a fitted 2 N press, with no bristle bending, and refilled to 5.6 mL at every dip; one "
         "start, the seed changes nothing")


def flow_spec(initial: float = 0.0, pail: float = rb.PAIL_PAINT, charts: tuple | None = None):  # noqa: ANN201
    """The film on the three charts (or ``charts``, the wall first), the brush (``initial`` share loaded) and the
    pail (``pail`` m³): a ``SurfaceFlow``."""
    from alloy.plugins.surface_flow import SurfaceFlow  # noqa: PLC0415
    from alloy.plugins.surface_flow.rheology import Thixotropy  # noqa: PLC0415

    p = paint(PAINT)
    settings = pp.film_settings(PAINT, substrate="drywall_primed", chart=rs.WALL)
    settings["material"] = dataclasses.replace(film_material(PAINT), yield_stress=p.rest_yield)
    return SurfaceFlow(charts=charts or rs.charts(), inertia=False, max_depth=4e-4,
                       tools=(rb.brush(PAINT, initial),),
                       trays=(rb.pail_tray(pail),), thixotropy=Thixotropy(rebuild=p.recovery), krieger=KRIEGER_PHI_M,
                       krieger_ref=p.volume_solids, tool_speed=scr.TRAVEL, network=rb.pail_network(),
                       **settings)  # fmt: skip


def scene(init, tcp: np.ndarray, alpha: float, boxes: list | None = None):  # noqa: ANN001, ANN201 - TestScene
    """The charts' boxes, the radiator and the trim (static; or ``boxes``), the kinematic brush at ``tcp``, a free
    dummy far away (a scene needs one dynamic body); XPBD, one world, one exchange a period. Nothing collides."""
    import warp as wp  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        for box in boxes if boxes is not None else rs.chart_boxes() + rs.obstacles():
            b.add_shape_box(-1, xform=wp.transform(wp.vec3(*box.pos), wp.quat(*box.quat)), hx=box.half[0],
                            hy=box.half[1], hz=box.half[2], label=box.name)  # fmt: skip
        body = b.add_body(xform=body_xform(*tcp, rb.quat_xyzw(rb.attitude(alpha))), is_kinematic=True, label="brush")
        hx, r = rb.HANDLE_BOX
        cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=rb.MASS / (8.0 * hx * r * r))
        b.add_shape_box(body, xform=body_xform(*rb.HANDLE_COM), hx=hx, hy=r, hz=r, cfg=cfg)
        dummy = b.add_body(xform=body_xform(2.0, -2.0, 1.0), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=init, worlds=1, periods=1, control_dt=scr.SUB, dt=scr.SUB, solver="xpbd",
                     ground=False)  # fmt: skip


def place(sim, body: int, tcp: np.ndarray, alpha: float, vel: np.ndarray, spin: float) -> None:  # noqa: ANN001
    """Put the kinematic brush at ``tcp`` and attitude ``alpha`` [deg] with its twist: ``vel`` is the TCP's velocity
    and ``spin`` alpha's rate [deg/s] (a rising alpha turns body x from +x toward +z: about -y); the body's twist is
    its COM's velocity and the angular velocity."""
    R = rb.attitude(alpha)
    w = np.array([0.0, -math.radians(spin), 0.0])
    q = sim.state.body_q.numpy().copy()
    q[body] = (*tcp, *rb.quat_xyzw(R))
    sim.state.body_q.assign(q)
    qd = sim.state.body_qd.numpy().copy()
    qd[body] = (*(np.asarray(vel) + np.cross(w, R @ np.asarray(rb.HANDLE_COM))), *w)
    sim.state.body_qd.assign(qd)


def path(tips: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """The TCP for each pose."""
    return np.stack([rb.tcp_for_tip(t, a) for t, a in zip(tips, alphas)])


def _slice(read: dict, key: str, chart: int, k: int | None = None) -> np.ndarray:
    """Chart ``chart``'s cells of a per-cell field [W, C(, K)] in world 0 (cells run chart by chart, i-major)."""
    sizes = [int(np.prod(c.cells)) for c in rs.charts()]
    lo = sum(sizes[:chart])
    v = np.asarray(read[key])[0, lo:lo + sizes[chart]]
    return v if k is None else v[:, k]


def evaluate(read: dict, wet: np.ndarray, xz: np.ndarray, areas: np.ndarray, marred: np.ndarray | None = None) -> dict:
    """The verdict of world 0 from the film's ``read()`` after the settle, the wall's depth [nx, ny], its cell
    centres (x, z) [nx, ny, 2] and every chart's cell areas [C] (the plug-in's ``tables.area_np``). Of Gym-Anything's
    four criteria three are judged (audit AB-gym-6): ``no_marks`` only from ``marred`` (None: not judged), ``no_runs``
    and ``protected`` by the port's own measures (their notes), on the scene ``SCENE`` describes."""
    shape, area = rs.wall_cells(), rs.CELL * rs.CELL
    dft = _slice(read, "dft", rs.WALL).reshape(shape)
    sup = _slice(read, "species_supplied", rs.WALL, 0).reshape(shape)
    out = _slice(read, "species_out", rs.WALL, 0).reshape(shape)
    m = rm.Masks(rs.band_mask(xz), rs.band_mask(xz) & ~rs.pocket_mask(xz), rs.edge_mask(xz, rm.RUN_EDGE))
    opt, r_g = optics(PAINT, COLOUR), srgb8_to_linear(COLOURS[SUBSTRATE])
    coat = first_coat(xz)
    hidden = opt.hidden(1e6 * (coat + dft), np.broadcast_to(r_g, shape + (3,)))
    hidden0 = opt.hidden(1e6 * coat, np.broadcast_to(r_g, shape + (3,)))
    marks = rm.marks(marred, m, area)
    marred = np.zeros(shape, bool) if marred is None else marred       # for the progress: no mark, nothing off it
    hide, hide_detail = rm.hide(hidden, m, area)
    settled = band_set(read, wet, m.band)
    runs, run_detail, run = rm.runs(rm.flowed_in(dft, 0.0, sup, out, area), wet, m, area, settled)
    prot = [(_slice(read, "dft", k), _slice({"a": areas[None]}, "a", k)) for k in rs.PROTECTED]
    # into sinks: film off a chart's open edge, drips caught by the radiator's or the trim's shapes, and drips the
    # supply books lost (hitting nothing, or a cone past a chart's edge: GA's floor is protected; worst case)
    lost = float(np.asarray(read["spilled"])[0].sum()) + sum(float(np.asarray(read[f"supply_{k}"])[0])
                                                             for k in ("caught", "lost"))
    protected = rm.protected(prot, lost)
    checks = [hide, runs.differs(OWN_FLOW), marks, protected.differs(OWN_SINKS), pm.ledger(read, rel=1e-12),
              pm.solids_ledger(read, rel=1e-12)]
    v = pm.verdict(checks, scene=SCENE)
    prog = rm.progress(hidden, m, area, rm.start_area(hidden0, m, area), marred, run, protected.value)
    lap = pm.chart_field(read, "laps", shape)
    return {**v, "score": 1.0 if v["succeeded"] else rm.FAIL_SCALE * prog["progress"], **prog,
            "hide": hide_detail, "runs": {**run_detail, "settled": settled},
            "laps_cm2": None if lap is None else round(float(((lap > 0) & m.band).sum()) * area * 1e4, 3),
            "dry_band": pm.dry_summary(dft, m.scored),
            "drips_mL": drips(read),
            "books": books(read)}  # fmt: skip


def first_coat(xz: np.ndarray, seed: int = 0) -> np.ndarray:
    """The wall's dry first coat [m] at wall points ``xz`` [..., 2]: Gym-Anything's, 43·(1 + 0.15 f) with ``f`` its
    smooth field (``PT/system.py:383-392``, seed 7000 + the episode's), read off its 2 mm wall chart in its room's
    frame (x + 0.55 m), as cut-in-trim reads it (``trim_layout``)."""
    from alloy.tasks.trim_layout import TrimLayout  # noqa: PLC0415

    ga = np.stack([np.asarray(xz)[..., 0] + rs.GA_X, np.asarray(xz)[..., 1]], axis=-1)
    return TrimLayout(first_coat_dft=FIRST_COAT, seed=seed).first_coat(ga)


def band_set(read: dict, wet: np.ndarray, band: np.ndarray) -> bool:
    """Every wet band cell has set: its solids at random close packing (the stage's φ_m), so it flows no more."""
    phi = _slice(read, "phi_s", rs.WALL).reshape(rs.wall_cells())
    return bool(np.all(phi[band & (wet > H_WET)] >= pp.DRYING[PAINT][0] - 1e-6))


def all_set(flow) -> bool:  # noqa: ANN001
    """Every wet cell of every chart (the wall, the radiator's top, the baseboard's ledge) has set."""
    read, phi_m = flow.read(), pp.DRYING[PAINT][0] - 1e-6
    return all(bool(np.all(_slice(read, "phi_s", k)[flow.depth(k)[0].ravel() > H_WET] >= phi_m))
               for k in (rs.WALL, *rs.PROTECTED))


def settle_until_set(sim, flow, cap: float = SETTLE_MAX) -> float:  # noqa: ANN001
    """Step with the brush still until every wet cell has set or ``cap`` s have passed; the film time it took [s].
    About Gym-Anything's settle, which runs until no chart holds wet film, at most 30 min (``PT/task.py:4-7``,
    ``:674-681``: the film's own future decides sag, curtains and runs): once every cell has set (solids at φ_m), the
    rest of it moves nothing here (the paint's 2 Pa yield exceeds ρgh, 1.6 Pa at the 126 µm wet maximum; WP15)."""
    n, t = int(round(SETTLE_CHUNK / scr.SUB)), 0.0
    while t < cap - 1e-9 and not all_set(flow):
        for _ in range(n):
            sim.step_period()
        t += n * scr.SUB
    return t


def drips(read: dict) -> dict:
    """Where the brush's drips went [mL]: all it dripped and staged; of what the supply cast (``sent``), back into
    the pail, caught by a shape, onto the floor, and onto a chart (the wall or a protected one: the rest); and what
    is still in flight (staged = sent + in flight: ``supply_stage``)."""
    d = {k: float(np.asarray(read[f"supply_{k}"])[0]) * 1e6 for k in ("staged", "sent", "to_tank", "caught", "lost")}
    d["on_charts"] = d["sent"] - d["to_tank"] - d["caught"] - d["lost"]
    d["falling"] = d["staged"] - d["sent"]
    return {"dripped": float(np.asarray(read["tool_dripped"])[0].sum()) * 1e6, **d}


def books(read: dict) -> dict:
    """The tools' and the pail's ledgers [m³] (0 and the pail's start, to rounding) and what moved."""
    return {"store_ledger": float(np.asarray(read["store_ledger"])[0]),
            "tray_ledger": float(np.asarray(read["tray_ledger"])[0]),
            "picked_mL": float(np.asarray(read["tool_picked"])[0].sum()) * 1e6,
            "laid_mL": -float(np.asarray(read["to_tools_volume"])[0]) * 1e6,
            "load_mL": float(np.asarray(read["tool_load"])[0].sum()) * 1e6}


def run(lanes_per_dip: int = scr.LANES_PER_DIP, pitch: float = scr.PITCH, settle: float = SETTLE_MAX,
        device: str = "cpu", moves: tuple | None = None, keep: dict | None = None) -> dict:
    """The scripted episode (``moves``: (start, [Move, ...]) instead of :func:`radiator_script.script`), the settle
    until every wet cell has set (at most ``settle`` s; an unset coat is judged by its wet film too), and the verdict;
    ``keep`` gets the final ``read``, ``wet`` and ``xz``."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run

    t0 = time.time()
    start, mv = moves if moves is not None else scr.script(lanes_per_dip, pitch)
    tips, alphas = scr.poses(start, mv)
    tcps = path(tips, alphas)
    flow, box = flow_spec(), {}

    def init(sim) -> None:  # noqa: ANN001
        box["xz"] = rs.wall_centres(flow.cell_centres(rs.WALL))

    sim = build_simulation(scene(init, tcps[0], float(alphas[0])), [flow], device)
    body = list(sim.model.body_label).index("brush")
    for k in range(len(tcps)):
        n = min(k + 1, len(tcps) - 1)
        place(sim, body, tcps[k], float(alphas[k]), (tcps[n] - tcps[k]) / scr.SUB, (alphas[n] - alphas[k]) / scr.SUB)
        sim.step_period()
    place(sim, body, tcps[-1], float(alphas[-1]), np.zeros(3), 0.0)
    settled_s = settle_until_set(sim, flow, settle)
    read, depth = flow.read(), flow.depth(rs.WALL)[0]
    if keep is not None:
        keep.update(read=read, wet=depth, xz=box["xz"])
    out = evaluate(read, depth, box["xz"], flow.tables.area_np)
    return {**out, "settle_s": settled_s, "sim_s": round(len(tcps) * scr.SUB + settled_s, 2),
            "wall_s": round(time.time() - t0, 1)}


__all__ = ["COLOUR", "FIRST_COAT", "PAINT", "SETTLE_MAX", "SUBSTRATE", "all_set", "band_set", "books", "drips",
           "evaluate", "first_coat", "flow_spec", "path", "place", "run", "scene", "settle_until_set"]
