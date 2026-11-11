# SPDX-License-Identifier: Apache-2.0
"""SurfaceFlow films in the camera (alloy.camera.films): the drawn coat / soil / wet area matches the plug-in's map.

Scenes: the brush-stroke task's stand-in (paint on paper, acrylic) and a rinse of a soiled plate with the floor-flush
task's soil law; both on the CPU, rendered by the fast tracer. The drawn masks come from the camera's albedo (each
pixel's nearest instance colour); the plug-in's maps are projected through the camera's own pinhole model (a ray per
pixel centre onto the chart plane), so the test does not reuse the film's geometry.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

import alloy.camera as ac
from alloy.camera.films import FilmLook, set_film_look
from alloy.camera.pose import image_extent, look_at_matrix

W_PX, H_PX = 320, 240         # 0.7 mm pixels on the chart from 15 cm: smaller than the 1-2 mm cells
HEIGHT = 0.15
SEEDS = 8                     # one-sample renders voted per pixel (an exact colour per sample, no mixing)
SUB = 4                       # the plug-in's map projected at SUB x SUB points per pixel
IOU_MIN = 0.9
DEVICE = os.environ.get("ALLOY_FILMS_TEST_DEVICE", "cpu")   # cuda:0 for a GPU check of the same tests
pytestmark = pytest.mark.compile_heavy  # each test may build the film, XPBD and camera kernels cold (~55 s on CPU)


def paint_sim(periods: int, coat: float = 60e-6, graph: bool = False):
    """The brush-stroke task's stand-in scene with the first ``periods`` of its stroke laid."""
    from alloy.plugins.harness.scene import build_simulation
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
    from alloy.tasks import paint_process as pp
    from alloy.tasks.paint_brush_stroke import BOARD_Z, CONTROL_DT, PAINT, PAPER, cells, stroke_rates
    from alloy.tasks.paint_scene import chart_scene

    flow = SurfaceFlow(charts=(PlaneChart("paper", cells=cells()),), inertia=False, max_depth=4e-4,
                       **pp.film_settings(PAINT, substrate="paper"))
    box: dict = {}

    def init(sim) -> None:
        c = np.asarray(flow.cell_centres(0))[..., :2].astype(np.float64)
        box["rates"] = stroke_rates(c, coat, 1.0)

    paper = ("paper", (0.0, 0.0, BOARD_Z - 0.005), (0.0, 0.0, 0.0, 1.0), (0.5 * PAPER[0], 0.5 * PAPER[1], 0.005))
    sim = build_simulation(chart_scene([paper], init, CONTROL_DT), [flow], DEVICE, graph=graph)
    return sim, flow, box["rates"][:periods], pp.source_species(PAINT)


def lay(sim, flow, rates, species) -> None:
    for r in rates:
        flow.set_source(r[None], chart=0, species=species)
        sim.step_period()
    flow.set_source(0.0)


def rinse_sim():
    """A 10 x 10 cm plate soiled in a band (0.2 kg/m², pre-soaked), rinsed by a water stream on one corner."""
    from alloy.plugins.harness.scene import build_simulation
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
    from alloy.plugins.surface_flow.process_specs import Soil
    from alloy.tasks.paint_scene import chart_scene

    soil = Soil(initial=0.0, charts=(0,), sigma_dry=4.0, sigma_wet=0.5, t_soak=2.0, k_e=0.05, k_i=2.0, k_wipe=0.0)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(50, 50)),), material="water", species=("soil",),
                       process={"soil": soil}, max_depth=0.01)
    plate = ("plate", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), (0.05, 0.05, 0.005))
    sim = build_simulation(chart_scene([plate], None, 0.02), [flow], DEVICE)
    x = np.asarray(flow.cell_centres(0))[..., 0]
    y = np.asarray(flow.cell_centres(0))[..., 1]
    flow.process.set_soil(np.where(np.abs(x + 0.5 * y) < 0.02, 0.2, 0.0), chart=0, hyd=0.9)
    flow.set_source(np.where((x > 0.02) & (y > 0.02), 2.0e-4, 0.0)[None], chart=0)
    for _ in range(25):
        sim.step_period()
    flow.set_source(0.0)
    return sim, flow


def view(z: float) -> tuple:
    return (0.0, -0.01, z + HEIGHT), (0.0, 0.0, z)


def top_camera(z: float) -> ac.Camera:
    return ac.Camera("top", look_at=view(z), width=W_PX, height=H_PX, spp=1)


def projected(cam_eye, target, flow, cell_mask: np.ndarray) -> np.ndarray:
    """(H, W) bool: pixels at least half covered by the plug-in's ``cell_mask`` (nx, ny) seen through the camera
    (SUB x SUB points per pixel, image row 0 on top)."""
    s = (np.arange(SUB) + 0.5) / SUB
    hits = [_projected_at(cam_eye, target, flow, cell_mask, dx, dy) for dx in s for dy in s]
    return np.mean(hits, axis=0) >= 0.5


def _projected_at(cam_eye, target, flow, cell_mask: np.ndarray, sx: float, sy: float) -> np.ndarray:
    R = look_at_matrix(cam_eye, target)
    ex, ey = image_extent(60.0, W_PX, H_PX)
    cols, rows = np.meshgrid(np.arange(W_PX) + sx, np.arange(H_PX) + sy)
    ndc_x, ndc_y = 2.0 * cols / W_PX - 1.0, 1.0 - 2.0 * rows / H_PX
    d = R[:, 0] * (ndc_x * ex)[..., None] + R[:, 1] * (ndc_y * ey)[..., None] - R[:, 2]
    X = flow.tables.clocal.numpy()[0, 0]
    org, rot = X[:3], quat_matrix(X[3:])
    o, dl = (np.asarray(cam_eye) - org) @ rot, d @ rot                  # into the chart frame
    t = -o[2] / dl[..., 2]
    p = o + t[..., None] * dl
    c = np.asarray(flow.cell_centres(0))
    nx, ny = cell_mask.shape
    dx, dy = c[1, 0, 0] - c[0, 0, 0], c[0, 1, 1] - c[0, 0, 1]
    i = np.floor((p[..., 0] - c[0, 0, 0]) / dx + 0.5).astype(int)
    j = np.floor((p[..., 1] - c[0, 0, 1]) / dy + 0.5).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (t > 0)
    return np.where(inside, cell_mask[np.clip(i, 0, nx - 1), np.clip(j, 0, ny - 1)], False)


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def drawn(cam: ac.Camera, sim, pred) -> np.ndarray:
    """(H, W) bool: pixels where most one-sample renders hit an instance whose name passes ``pred`` (each sample's
    albedo is its instance's colour exactly)."""
    sd = ac.camera_scene(sim).sd
    cols = np.asarray(sd.inst_color, np.float64)
    keep = np.array([pred(str(n)) for n in sd.inst_name])
    votes = np.zeros((H_PX, W_PX))
    for seed in range(SEEDS):
        alb = cam.render(sim, output="all", seed=seed).albedo.numpy()[0].reshape(H_PX, W_PX, 3)
        votes += keep[((alb[..., None, :] - cols[None, None]) ** 2).sum(-1).argmin(-1)]
    return votes / SEEDS >= 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def test_paint_coverage_matches_plugin():
    sim, flow, rates, sp = paint_sim(120)
    lay(sim, flow, rates, sp)
    set_film_look(flow, FilmLook(color=(0.6, 0.04, 0.03)))
    z = float(flow.tables.clocal.numpy()[0, 0][2])
    cam = top_camera(z)
    rd = flow.process.read()
    nx, ny = flow.tables.shapes[0]
    painted = (rd["dft"][0] > 1e-6).reshape(nx, ny)
    hidden = (rd["dft"][0] >= flow.process.spec.drying.hide_dft).reshape(nx, ny)
    eye, tgt = view(z)
    got = drawn(cam, sim, lambda n: ":paint" in n)
    assert painted.sum() > 200 and got.sum() > 200 and hidden.sum() > 50
    assert iou(got, projected(eye, tgt, flow, painted)) >= IOU_MIN
    top = drawn(cam, sim, lambda n: ":paint3" in n)
    assert iou(top, projected(eye, tgt, flow, hidden)) >= IOU_MIN


def test_rinse_soil_and_wet_match_plugin():
    sim, flow = rinse_sim()
    z = float(flow.tables.clocal.numpy()[0, 0][2])
    cam = top_camera(z)
    nx, ny = flow.tables.shapes[0]
    soiled = (flow.process.soil.numpy()[0] > 1e-6).reshape(nx, ny)
    wet = (flow.h.numpy()[0] > 2e-6).reshape(nx, ny)
    eye, tgt = view(z)
    assert soiled.sum() > 100 and wet.sum() > 50 and (soiled & ~wet).sum() > 50
    got_soil = drawn(cam, sim, lambda n: ":soil" in n and not n.split(":")[1].startswith("soil0"))
    got_wet = drawn(cam, sim, lambda n: ":soil" in n and "_wet" in n)
    assert iou(got_soil, projected(eye, tgt, flow, soiled)) >= IOU_MIN
    assert iou(got_wet, projected(eye, tgt, flow, wet)) >= IOU_MIN


def test_scene_without_films_renders_as_before():
    """A sim without the plug-in: the engine simulation renders exactly as its model + state did."""
    from alloy.plugins.harness.scene import build_simulation
    from alloy.tasks.paint_scene import chart_scene
    paper = ("paper", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), (0.1, 0.05, 0.005))
    sim = build_simulation(chart_scene([paper], None, 0.02), [], DEVICE)
    sim.step_period()
    cam = top_camera(0.505)
    a = cam.render(sim.model, state=sim.state).numpy().copy()
    b = cam.render(sim).numpy().copy()
    assert np.array_equal(a, b)
    assert getattr(ac.camera_scene(sim), "film", None) is None


def _snapshot(sim, flow) -> dict:
    out = {k: v.numpy().copy() for k, v in flow.state_arrays().items()} if hasattr(flow, "state_arrays") else {}
    out.update({f"sim.{k}": v.numpy().copy() for k, v in sim.state_arrays().items()})
    out["h"], out["species"] = flow.h.numpy().copy(), flow.sp.mass.numpy().copy()
    return out


@pytest.mark.parametrize("graph", [False, True])
def test_rendering_leaves_the_sim_bitwise_unchanged(graph, monkeypatch):
    """The same stroke with and without a render after every 10 periods: every state array is bit for bit the
    same; with ``graph`` (captured periods) the film update never runs while a graph is being captured."""
    from alloy.camera import films
    seen = []
    orig = films.FilmSet.update

    def update(self, state):
        seen.append(wp.get_device(DEVICE).is_capturing)
        orig(self, state)

    monkeypatch.setattr(films.FilmSet, "update", update)
    snaps = []
    for render in (False, True):
        sim, flow, rates, sp = paint_sim(40, graph=graph)
        cam = top_camera(float(flow.tables.clocal.numpy()[0, 0][2]))
        for k in range(0, len(rates), 10):
            lay(sim, flow, rates[k:k + 10], sp)
            if render:
                cam.render(sim)
        snaps.append(_snapshot(sim, flow))
    assert seen and not any(seen)
    assert snaps[0].keys() == snaps[1].keys()
    for k in snaps[0]:
        assert np.array_equal(snaps[0][k], snaps[1][k]), k


def test_paste_darkens_by_depth():
    """A concrete paste film (the pad task's material kind, held by its yield stress) in three depth bands: the wet
    look draws each band at its level, and the drawn wet area matches the plug-in's (h above the wet depth)."""
    from alloy.plugins.harness.scene import build_simulation
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
    from alloy.plugins.surface_flow.film import FilmMaterial
    from alloy.tasks.paint_scene import chart_scene

    paste = FilmMaterial("concrete_paste", density=2300.0, viscosity=30.0, yield_stress=400.0, surface_tension=0.0)
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=(60, 40)),), material=paste, inertia=False, max_depth=0.01)
    pad = ("pad", (0.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0), (0.06, 0.04, 0.01))
    sim = build_simulation(chart_scene([pad], None, 0.02), [flow], DEVICE)
    x = np.asarray(flow.cell_centres(0))[..., 0]
    flow.set_depth(np.select([x < -0.03, x < 0.0, x < 0.03], [1.0e-4, 5.0e-4, 2.0e-3], 0.0), chart=0)
    sim.step_period()
    z = float(flow.tables.clocal.numpy()[0, 0][2])
    cam = top_camera(z)
    eye, tgt = view(z)
    h = flow.h.numpy()[0].reshape(flow.tables.shapes[0])
    for lvl, band in enumerate([(h > 2e-6) & (h < 2e-4), (h >= 2e-4) & (h < 1e-3), h >= 1e-3]):
        assert band.sum() > 200
        got = drawn(cam, sim, lambda n, k=lvl: n.endswith(f":wet{k}") or f":wet{k}#" in n)
        assert iou(got, projected(eye, tgt, flow, band)) >= IOU_MIN, lvl


def test_full_material_path_draws_the_coat():
    """The full-material tracer draws the same coat: the pixels that change when the film is drawn (same seed) are
    the painted ones the plug-in's map projects to."""
    sim, flow, rates, sp = paint_sim(120)
    lay(sim, flow, rates, sp)
    z = float(flow.tables.clocal.numpy()[0, 0][2])
    eye, tgt = view(z)
    cam = ac.Camera("top", look_at=(eye, tgt), width=W_PX, height=H_PX, spp=4)
    imgs = []
    for look in (FilmLook(color=(0.6, 0.04, 0.03)), FilmLook(painted_dft=1.0)):  # drawn, then nothing drawn
        set_film_look(flow, look)
        sc = ac.camera_scene(sim, quality="full")
        imgs.append(cam.render(sc, state=sim.state, seed=0).numpy()[0].astype(np.int16))
        ac.camera._forget(sc)
    changed = np.abs(imgs[0] - imgs[1]).max(-1) > 24
    nx, ny = flow.tables.shapes[0]
    painted = projected(eye, tgt, flow, (flow.process.read()["dft"][0] > 1e-6).reshape(nx, ny))
    assert painted.sum() > 200
    assert iou(changed, painted) >= IOU_MIN


def test_view_follows_the_plugin_through_reset():
    """The camera keeps the plug-in's own arrays from bind: after sim.reset() it reads the restored (bare) paper."""
    sim, flow, rates, sp = paint_sim(40)
    lay(sim, flow, rates, sp)
    z = float(flow.tables.clocal.numpy()[0, 0][2])
    cam = top_camera(z)
    assert drawn(cam, sim, lambda n: ":paint" in n).sum() > 50
    view = ac.camera_scene(sim).film.layout.objects[0].view
    sim.reset()
    assert view.depth is flow.h and view.species_mass is flow.sp.mass and view.base is flow.base
    assert float(np.abs(flow.sp.mass.numpy()).max()) == 0.0
    assert drawn(cam, sim, lambda n: ":paint" in n).sum() == 0
