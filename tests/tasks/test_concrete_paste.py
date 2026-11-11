# SPDX-License-Identifier: Apache-2.0
"""The concrete process layer (alloy.tasks.concrete_paste) against closed forms, CPU: the setting curve with no tool,
bleed water into a sealed surface (the triangle's integral) and the blisters it lifts, one sole pass (texture and
densification per pass, the stiffness at last working), the water a pass works in (dwc), the pad pressure over the
bearing cells, and the sealant (its yield and skin, a torn skin)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store
from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_paste_work as PW
from alloy.tasks.concrete_paste import RHO_WATER, ConcretePaste, SoleWork, sole_axes
from alloy.tasks.concrete_tools import FORCE, HEIGHT, Stroke, ToolCarriage, ToolProgram

SLAB = (0.2, 0.1)                     # half extents [m]
HALF = (0.1, 0.04, 0.005)             # the float block
CELLS = (100, 50)                     # 4 mm
MORTAR = FilmMaterial("repair_mortar", density=2051.0, viscosity=20.0, yield_stress=2000.0, surface_tension=0.0)
WOOD = SoleWork("float", a_tool=5.0e-4, dens=0.2, seal=0.05, half=HALF[:2])     # wood float 144D (tools.py:64-86)
STROKE_S = 1.5                        # s: a pass, 0.3 m at 0.2 m/s
PASS_PERIODS = 80                     # the stroke and a 0.1 s hold, at 20 ms
SEALANT = FilmMaterial("sealant", density=1400.0, viscosity=50.0, yield_stress=600.0, surface_tension=0.0)


def _scene(z0: float, xy0: tuple[float, float]):
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=SLAB[0], hy=SLAB[1], hz=0.01, label="slab")
        tool = b.add_body(xform=body_xform(xy0[0], xy0[1], z0), label="float")
        b.add_shape_box(tool, hx=HALF[0], hy=HALF[1], hz=HALF[2], cfg=newton.ModelBuilder.ShapeConfig(density=1875.0))
        # 0.3 kg: a wood float's mass
    return build


def _sim(program: ToolProgram, material: str, z0: float = HALF[2] + 2e-4, worlds: int = 1, store: Store | None = None,
         cells: tuple[int, int] = CELLS, temperature: float = 21.1, **flow_kw):
    film = SEALANT if material.startswith("sealant") else MORTAR
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=cells),), material=film, inertia=False, max_depth=2e-3,
                       tools=(Applicator(body="float", shape="rect", size=HALF[:2], center=(0.0, 0.0, -HALF[2]),
                                         stiffness=5.0e5, damping=5.0e-4, store=store),), per_cell_rheology=True,
                       **flow_kw)
    carriage = ToolCarriage(programs=(program,))
    layer = ConcretePaste(material=material, tools=(WOOD,), flow=flow, carriage=carriage, temperature=temperature)
    xy0 = program.keys[0].pos[:2]                                      # built at its first keyframe
    sim = build_simulation(TestScene(build=_scene(z0, xy0), ground=False, worlds=worlds, dt=2.0e-3),
                           [carriage, flow, layer], "cpu")
    return sim, flow, layer


def _parked() -> ToolProgram:
    return ToolProgram("float", (Stroke(0.0, (-0.1, 0.0, 0.03), mode=HEIGHT),))       # held 25 mm clear


def test_with_no_tool_the_paste_ages_along_its_setting_curve():
    sim, flow, layer = _sim(_parked(), "repair_mortar")
    age0 = 6 * 60.0
    layer.set_start(paste=1, age=age0, D=0.5)
    for _ in range(25):
        sim.step_period()
    r = layer.read()
    t = 25 * 0.02
    assert r["age"].dtype == np.float64 and np.allclose(r["age"], age0 + t, rtol=1e-12, atol=0)
    tb, ts = layer.taus()
    mat = CM.RepairMortar()
    assert np.allclose(tb, mat.tau_bulk(age0 + t), rtol=1e-5, atol=0)
    written = flow.inputs()["yield_stress"].numpy()                    # the film's yield: the bulk's, as it sets
    assert np.allclose(written, mat.tau_bulk(age0 + t - 2.0e-3), rtol=1e-5, atol=0)
    assert np.allclose(ts, mat.tau_surf(mat.tau_bulk(age0 + t), 0.5), rtol=1e-5)
    assert r["worked"].max() == 0.0 and r["tau_w"].max() == 0.0


def test_a_sealed_surface_traps_the_bleed_triangle():
    """Concrete with S = 1: every bleed drop goes to q, so q = (1/rho_w) x the triangle's integral over the run."""
    sim, flow, layer = _sim(_parked(), "concrete")
    mat = CM.Concrete()
    a0 = 20 * 60.0                                                     # on the rising leg (t_end/3 ~ 36 min)
    layer.set_start(paste=1, age=a0, S=1.0)
    n = 50
    for _ in range(n):
        sim.step_period()
    a1 = a0 + n * 0.02
    te = mat.bleed_end_time()
    peak = 2.0 * mat.bleed_mean / 3600.0
    want = peak * (a1 * a1 - a0 * a0) / (2.0 * te / 3.0) / RHO_WATER
    r = layer.read()
    q = r["q"]
    assert np.allclose(q, want, rtol=1e-3, atol=0), (q.mean(), want)
    assert r["w"].max() == 0.0
    assert np.allclose(r["bled"], q, rtol=1e-6, atol=0) and r["evaporated"].max() == 0.0   # the books: all trapped


@pytest.mark.parametrize("load", [15.0])
def test_one_sole_pass_texture_and_densification_follow_their_per_pass_laws(load):
    """A wood float slid sideways (turned 90°: its 80 mm width along the motion, +x) at a set load crosses the
    middle cells once, after a lead-in that lets the low-passed advance settle: n = 1 pass, so
    a = a_tool + (a0 - a_tool) exp(-2.2 y) and D = 1 - exp(-m_i tau_s B dens / 112 J/m²)."""
    yaw = math.pi / 2
    prog = ToolProgram("float", (Stroke(0.0, (-0.15, 0.0, HALF[2]), yaw=yaw, force=load, mode=FORCE),
                                 Stroke(STROKE_S, (0.15, 0.0, HALF[2]), yaw=yaw, force=load, mode=FORCE)))
    sim, flow, layer = _sim(prog, "repair_mortar")
    age0, a0 = 8 * 60.0, 6.0e-4
    layer.set_start(paste=1, age=age0, a=a0, D=0.0)
    for _ in range(PASS_PERIODS):                                      # the stroke + 0.1 s hold
        sim.step_period()
    r = layer.read()
    end = sim.state.body_q.numpy()[0, :3]
    assert abs(end[0] - 0.15) < 3e-3, end
    xy = flow.cell_centres(0)[..., :2].reshape(-1, 2)
    mid = (np.abs(xy[:, 0]) < 0.05) & (np.abs(xy[:, 1]) < 0.08)       # crossed by the whole width, settled
    B = 2 * HALF[1]
    assert np.allclose(r["worked"][0, mid], B, rtol=0.08), r["worked"][0, mid]
    mat = CM.RepairMortar()
    tau_s = float(mat.tau_surf(mat.tau_bulk(age0 + STROKE_S / 2), 0.0))
    p = load / (4 * HALF[0] * HALF[1])
    mi = float(CM.interface_factor(tau_s))
    y = max(min(p / (CM.NC * tau_s), 1.0), mi)
    a_want = WOOD.a_tool + (a0 - WOOD.a_tool) * math.exp(-CM.TEX_PASS * y)
    assert np.allclose(r["a"][0, mid], a_want, rtol=0.02), (r["a"][0, mid].mean(), a_want)
    D_want = 1.0 - math.exp(-mi * tau_s * B * WOOD.dens / CM.DENS_ENERGY)
    assert np.allclose(r["D"][0, mid], D_want, rtol=0.1), (r["D"][0, mid].mean(), D_want)
    tau_w = r["tau_w"][0, mid]                                         # the surface's stiffness when worked, not
    assert np.allclose(tau_w, tau_s, rtol=0.02, atol=0), (tau_w.mean(), tau_s)          # the bulk's (2x)
    assert np.all(r["late"] == 0)


def test_a_flow_that_also_runs_the_setting_process_is_refused():
    """One writer of the paste's yield: the layer refuses a flow whose process stage sets it too."""
    from alloy.plugins.errors import PluginError
    from alloy.plugins.surface_flow.process_specs import Process, Setting

    with pytest.raises(PluginError, match="setting process"):
        _sim(_parked(), "repair_mortar", process=Process(setting=Setting()))


def _pass(load: float = 15.0) -> ToolProgram:
    """The float slid sideways across the slab at ``load`` (as in the one-pass test)."""
    yaw = math.pi / 2
    return ToolProgram("float", (Stroke(0.0, (-0.15, 0.0, HALF[2]), yaw=yaw, force=load, mode=FORCE),
                                 Stroke(STROKE_S, (0.15, 0.0, HALF[2]), yaw=yaw, force=load, mode=FORCE)))


def _mid(flow) -> np.ndarray:
    xy = flow.cell_centres(0)[..., :2].reshape(-1, 2)
    return (np.abs(xy[:, 0]) < 0.05) & (np.abs(xy[:, 1]) < 0.08)


def test_the_layer_states_are_readable_and_only_the_start_fields_settable():
    """SOFT-2 and the pad tasks read these by name; the derived ones cannot be started."""
    from alloy.plugins.errors import PluginError

    sim, flow, layer = _sim(_parked(), "concrete")
    sim.step_period()
    r = layer.read()
    W, C = 1, CELLS[0] * CELLS[1]
    for name in ("age", "dwc", "bl", "tau_s", "bear", "skin_torn", "D", "a", "S", "w", "q", "tau_w"):
        assert r[name].shape == (W, C), name
    assert r["pressure"].shape == (W,) and r["active"].shape == (W,) and r["advance"].shape == (W, 1, 3)
    assert r["bear"].dtype == np.int32 and r["skin_torn"].dtype == np.int32
    layer.set_start(dwc=0.1, bl=1e-4)
    assert np.all(layer.read()["dwc"] == np.float32(0.1)) and np.all(layer.read()["bl"] == np.float32(1e-4))
    for name in ("bear", "tau_s", "pressure", "skin_torn", "active"):
        with pytest.raises(PluginError, match="cannot start"):
            layer.set_start(**{name: 0})


@pytest.mark.parametrize("cells", [CELLS, (100, 25)], ids=["4mm", "4x8mm"])
def test_trapped_water_lifts_blisters_as_their_blur(cells):
    """Sealed concrete with a patch of trapped water: bl is their 4 (q - 2e-4)+, box-blurred +-7.5 mm along each axis
    (2 cells at 4 mm, 1 at 8 mm), capped at 3 mm; it never sinks (q only grows under S = 1). Their np.roll wraps,
    over their pad's margin, which holds no water; here the blur reads zeros past the chart's edges, as theirs with its
    margin, so a corner's blister stays in its corner at their height (not raised by a shorter window)."""
    sim, flow, layer = _sim(_parked(), "concrete", cells=cells)
    nx, ny = cells
    size = (2 * SLAB[0] / nx, 2 * SLAB[1] / ny)
    q = np.zeros((nx, ny))
    q[10:14, 20:24] = 6e-4                                             # 4 x 4 cells
    q[0, 0] = 1.0e-3                                                   # a corner
    q[60:70, 10:20] = 2.0e-3                                           # capped
    layer.set_start(paste=1, age=20 * 60.0, S=1.0, q=q.reshape(-1))
    for _ in range(5):
        sim.step_period()
    r = layer.read()
    q_now = r["q"][0].reshape(nx, ny).astype(np.float64)
    want = CM.blister_lift(q_now, size, wrap=False)
    assert np.allclose(r["bl"][0].reshape(nx, ny), want, rtol=1e-5, atol=1e-9)
    assert want.max() == pytest.approx(CM.BLISTER_CAP) and want[0, 0] > 0.0
    margin = np.zeros((nx + 20, ny + 20))                              # theirs: a dry margin around the paste
    margin[10:-10, 10:-10] = q_now
    assert np.allclose(want, CM.blister_lift(margin, size)[10:-10, 10:-10], rtol=1e-12, atol=0.0)
    assert r["bl"][0].reshape(nx, ny)[nx - 3:, ny - 3:].max() == 0.0 and r["bl"][0].reshape(nx, ny)[0, ny - 1] == 0.0
    assert np.all(r["bl"] >= 0.0)


def test_a_pass_over_a_wet_surface_works_its_water_in():
    """Bleed water standing on the paste when a sole works it goes into the top 6 mm: dwc += 1000 w / (350 x
    0.006), w -> 0, the cell marked wet-worked (coupling.py:1060-1063). Cells the sole never reached keep theirs."""
    sim, flow, layer = _sim(_pass(), "repair_mortar")
    w0, thin = 1.0e-4, 0.5 * CM.SHEEN_W                                # a sheen, and under one on the y > 0 half
    xy = flow.cell_centres(0)[..., :2].reshape(-1, 2)
    dry = xy[:, 1] > 0.0
    layer.set_start(paste=1, age=8 * 60.0, a=6e-4, D=0.0, w=np.where(dry, thin, w0))
    for _ in range(PASS_PERIODS):
        sim.step_period()
    r = layer.read()
    mid = _mid(flow)
    assert r["worked"][0, mid & dry].min() > 0.07                      # worked, but
    assert np.all(r["wet_worked"][0, mid & dry] == 0) and np.all(r["dwc"][0, mid & dry] == 0.0)   # no sheen
    assert r["w"][0, mid & dry].min() > 0.9 * thin
    mid = mid & ~dry
    assert np.all(r["wet_worked"][0, mid] == 1)
    assert np.allclose(r["dwc"][0, mid], CM.dwc_rise(w0), rtol=2e-3), (r["dwc"][0, mid].mean(), CM.dwc_rise(w0))
    assert r["w"][0, mid].max() < CM.SHEEN_W
    far = (xy[:, 0] > 0.19) & ~dry                                     # past the sole's stop at 0.15 + 0.04
    assert np.all(r["dwc"][0, far] == 0.0) and np.allclose(r["w"][0, far], w0, rtol=1e-3)   # bleed - evaporation
    start = np.where(dry, thin, w0)                                    # the books close per cell: what stands, was
    books = (r["w"][0].astype(float) + r["q"][0] + r["evaporated"][0] + r["worked_in"][0]   # trapped, evaporated or
             - r["bled"][0])
    assert np.allclose(books, start.astype(np.float32), rtol=1e-12, atol=0)                # worked in, less what bled
    assert np.allclose(r["worked_in"][0, mid], w0, rtol=1e-3) and r["worked_in"][0, far].max() == 0.0


def test_the_pad_pressure_is_the_load_over_the_bearing_cells():
    """A float held still at a set load, half over paste and half over cells with none: the pressure is its load over
    the half that bears (coupling.py:1003-1007), twice the whole sole's."""
    load = 12.0
    prog = ToolProgram("float", (Stroke(0.0, (0.0, 0.0, HALF[2]), force=load, mode=FORCE),))
    sim, flow, layer = _sim(prog, "repair_mortar")
    xy = flow.cell_centres(0)[..., :2].reshape(-1, 2)
    layer.set_start(paste=(xy[:, 0] < 0.0).astype(np.int32), age=8 * 60.0)
    for _ in range(25):
        sim.step_period()
    p = float(layer.read()["pressure"][0])
    sole = 4 * HALF[0] * HALF[1]                                       # 25 x 20 cells of 4 mm bear: half the sole
    assert p == pytest.approx(2.0 * load / sole, rel=1e-6, abs=0.0), (p, load / sole)
    assert np.all(layer.read()["bear"] == 0)                           # held still: no advance, no work


SKIN_S = {"sealant": 2 * 3600.0, "sealant_fast": 20 * 60.0}   # their 2 h skin and fast-sealant grade (variants.py:33)
NOSE = Store(capacity=1.5e-5, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0, scrape=True)  # sealant nose


@pytest.mark.parametrize("material, temp", [("sealant", 21.1), ("sealant_fast", 21.1), ("sealant", 15.0)])
def test_the_sealant_holds_its_yield_and_skins_at_two_hours(material, temp):
    """Their sealant: tau0 600 Pa at every age; the surface 3 x that once its skin forms at 2 h x MASCO (the fast
    grade: 20 min; at 15 °C MASCO puts the skin at 9880 s); no bleed."""
    sim, flow, layer = _sim(_parked(), material, temperature=temp)
    skin = SKIN_S[material] * CM.masco_factor(temp)
    xy = flow.cell_centres(0)[..., :2].reshape(-1, 2)
    age = np.where(xy[:, 0] < 0.0, skin - 60.0, skin + 60.0)
    layer.set_start(paste=1, age=age)
    for _ in range(10):
        sim.step_period()
    r = layer.read()
    tb, ts = layer.taus()
    assert np.all(tb == 600.0)
    assert np.array_equal(r["tau_s"][0], np.where(xy[:, 0] < 0.0, 600.0, 1800.0).astype(np.float32))
    assert np.allclose(ts[0], r["tau_s"][0], rtol=1e-12, atol=0)
    assert r["q"].max() == 0.0 and r["w"].max() == 0.0 and r["bl"].max() == 0.0


@pytest.mark.parametrize("material, store", [("sealant", None), ("sealant_fast", NOSE)], ids=["float", "nose"])
def test_working_a_skinned_sealant_tears_its_skin(material, store):
    """A pass over sealant skinned on one half: the skinned cells tear (a >= 5e-4, skin_torn); the stiffness at last
    working is the bulk's on both halves (coupling.py:1081-1088). A scraping-store nose (SOFT-2's tooling spatula, no
    film ceiling) tears it as the float does, on the fast grade's 20 min skin."""
    sim, flow, layer = _sim(_pass(), material, store=store)
    skin = SKIN_S[material] * CM.masco_factor(21.1)
    xy = flow.cell_centres(0)[..., :2].reshape(-1, 2)
    old = xy[:, 1] < 0.0
    layer.set_start(paste=1, age=np.where(old, skin + 600.0, 600.0), a=1e-4, D=0.0)
    for _ in range(PASS_PERIODS):
        sim.step_period()
    r = layer.read()
    mid = _mid(flow)
    assert np.all(r["skin_torn"][0, mid & old] == 1) and np.all(r["skin_torn"][0, mid & ~old] == 0)
    assert r["a"][0, mid & old].min() >= CM.TORN_A
    assert r["a"][0, mid & ~old].max() < CM.TORN_A
    assert np.allclose(r["tau_w"][0, mid], 600.0, rtol=1e-6)


def test_a_carrying_store_tool_works_the_paste_too():
    """A store tool (a carrying blade, as the sealant nose or a margin trowel) is no film ceiling, but the layer reads
    surface flow's work field for every tool (tf.wceil), so its pass works the paste as the float's does
    (GA's _work runs for any sole that touches and advances)."""
    sim, flow, layer = _sim(_pass(), "repair_mortar", store=Store(capacity=1e-4, retained=0.0, held=1.0, split=1.0,
                                                                   drip=0.0))
    age0, a0 = 8 * 60.0, 6.0e-4
    layer.set_start(paste=1, age=age0, a=a0, D=0.0)
    for _ in range(PASS_PERIODS):
        sim.step_period()
    r = layer.read()
    mid = _mid(flow)
    assert np.isinf(flow.read()["tool_ceiling"][0, mid]).all()           # not a film ceiling
    assert np.allclose(r["worked"][0, mid], 2 * HALF[1], rtol=0.08), r["worked"][0, mid]
    assert r["a"][0, mid].max() < a0 and np.all(r["D"][0, mid] > 0.0)


def test_a_stopped_sole_works_no_further():
    """After the stroke the float holds still while its low-passed advance decays through the 1 mm/s gate (~0.5 s):
    the slide is the sole's own (0 at rest, coupling.py:1009), so the cells under it gain no work; the advance only
    gates."""
    sim, flow, layer = _sim(_pass(), "repair_mortar")
    layer.set_start(paste=1, age=8 * 60.0, a=6e-4, D=0.0)
    for _ in range(PASS_PERIODS - 3):                                  # the stroke's end, + 2 periods
        sim.step_period()
    before, before_adv = layer.read()["worked"][0].copy(), layer.read()["advance"][0, 0].copy()
    for _ in range(30):                                                # 0.6 s: the gate shuts at ~0.5 s
        sim.step_period()
    r = layer.read()
    assert float(np.linalg.norm(before_adv)) > 0.05 > PW.MIN_ADVANCE   # it was still gating when held
    under = before > 0.0
    assert float((r["worked"][0] - before)[under].max()) < 1.0e-3, float((r["worked"][0] - before).max())


def test_a_sole_bears_only_where_its_footprint_covers_a_cell_centre():
    """GA's footprint (paste.py:572-600): three worlds, a float held at a load over one paste cell each (the float set
    0.3 cell off the grid in x, 0.05 in y). World 0: the cell's centre is inside the sole's corner, so it bears alone:
    load / cell² (their f / (n cell²)). World 1: the corner cell past both edges (0.2 and 0.45 cell, cover 0.3 x
    0.05, under their 1/4): no bearing, no pressure (a cover-weighted area there read ~1e8 Pa). World 2: 0.2 cell past
    the x edge only (cover 0.3, over 1/4): it bears, load / cell². The worlds do not mix."""
    load, cell = 12.0, 2 * SLAB[0] / CELLS[0]
    x0, y0 = -0.3 * cell, -0.05 * cell
    prog = ToolProgram("float", (Stroke(0.0, (x0, y0, HALF[2]), force=load, mode=FORCE),))
    sim, flow, layer = _sim(prog, "repair_mortar", worlds=3)
    xy = flow.cell_centres(0)[..., :2].reshape(-1, 2)
    corner = np.array([x0 - HALF[0], y0 - HALF[1]])
    at = [corner + d for d in ((0.8 * cell, 0.55 * cell), (-0.2 * cell, -0.45 * cell), (-0.2 * cell, 2.55 * cell))]
    one = [int(np.argmin(np.hypot(*(xy - p).T))) for p in at]
    assert all(np.allclose(xy[i], p, atol=1e-6) for i, p in zip(one, at, strict=True))
    paste = np.zeros((3, xy.shape[0]), np.int32)
    for w, i in enumerate(one):
        paste[w, i] = 1
    layer.set_start(paste=paste, age=8 * 60.0)
    for _ in range(25):
        sim.step_period()
    p = layer.read()["pressure"]
    assert p[0] == pytest.approx(load / cell ** 2, rel=1e-6, abs=0.0) and p[1] == 0.0, p
    assert p[2] == pytest.approx(load / cell ** 2, rel=1e-6, abs=0.0), p


def test_a_pass_is_the_bearing_patch_along_the_motion():
    """GA's passes (coupling.py:1015-1021): ds over the bearing patch's extent along the slide plus a cell. Over a
    16 mm strip of paste across its path (4 cells: 12 mm between the outer centres, + a cell) the float's 80 mm sole
    bears on the strip alone, so each strip cell, worked over ~82 mm, takes ~5.1 passes, a few more while the patch is
    partial at entry and exit (5.1-5.8 measured; without the cell 6.3-8.2; over the whole sole's extent about one).
    The passes are read from the texture's per-pass law at the strip's pressure."""
    sim, flow, layer = _sim(_pass(), "repair_mortar")
    xy = flow.cell_centres(0)[..., :2].reshape(-1, 2)
    strip = np.abs(xy[:, 0]) < 0.007                                   # 4 cells of 4 mm along the motion
    age0, a0 = 8 * 60.0, 6.0e-4
    layer.set_start(paste=strip.astype(np.int32), age=age0, a=a0, D=0.0)
    for _ in range(PASS_PERIODS):
        sim.step_period()
    r = layer.read()
    cells = strip & (np.abs(xy[:, 1]) < 0.03)
    mat = CM.RepairMortar()
    tau_s = float(mat.tau_surf(mat.tau_bulk(age0 + STROKE_S / 2), 0.0))
    p = 15.0 / (2 * HALF[0] * 0.016)                                   # the load over the strip it bears on
    y = max(min(p / (CM.NC * tau_s), 1.0), float(CM.interface_factor(tau_s)))
    a = r["a"][0, cells].astype(np.float64)
    passes = -np.log((a - WOOD.a_tool) / (a0 - WOOD.a_tool)) / (CM.TEX_PASS * y)
    assert 0.080 < r["worked"][0, cells].min() and r["worked"][0, cells].max() < 0.084
    assert 4.5 < passes.min() and passes.max() < 6.1, (passes.min(), passes.max())


@wp.kernel
def _slide_probe(body_q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3],
                 g: PW.Sole, out: wp.array[wp.vec3]):
    out[0] = PW.slide(body_q, qd[0], com[0], g, 0, 0, g.center[0])


@wp.kernel
def _inside_probe(xf: wp.array[wp.transform], g: PW.Sole, half: wp.vec2, pts: wp.array[wp.vec3],
                  out: wp.array[wp.int32]):
    i = wp.tid()
    out[i] = wp.where(PW.inside_sole(xf[0], g, 0, half, pts[i]), 1, 0)


def _sole(chart: wp.transform, app: Applicator) -> PW.Sole:
    g = PW.Sole()
    axes = np.array(sole_axes(app), np.float32)
    g.center, g.ex, g.ey, g.ez = (wp.array(axes[i:i + 1], dtype=wp.vec3, device="cpu") for i in range(4))
    g.n, g.cell = wp.vec3(0.0, 0.0, 1.0), 0.004
    g.cbody = wp.array(np.full((1, 1), -1, np.int32), dtype=wp.int32, device="cpu")
    g.clocal = wp.array([chart], dtype=wp.transform, device="cpu").reshape((1, 1))
    for name in ("pos",):
        setattr(g, name, wp.zeros(1, dtype=wp.vec3, device="cpu"))
    g.slid = wp.zeros((1, 1), dtype=wp.vec3, device="cpu")
    g.lo, g.hi = wp.zeros(1, dtype=wp.float32, device="cpu"), wp.zeros(1, dtype=wp.float32, device="cpu")
    return g


def test_the_slide_is_the_sole_centres_in_the_charts_plane():
    """coupling.py:479-485: the sole centre's velocity (the COM's plus omega x (centre - COM)), in the chart's frame,
    less its normal part; here a yawed, spinning float on a chart tilted and turned in world, its sole centre off its
    COM."""
    chart = wp.transform(wp.vec3(0.3, -0.2, 0.1), wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.2, 1.0)), 0.7))
    app = Applicator(body="float", shape="rect", size=HALF[:2], center=(0.05, 0.02, -0.005))
    g = _sole(chart, app)
    body = wp.transform(wp.vec3(1.0, 2.0, 0.3), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.6))
    lin, ang, com = np.array([0.3, -0.1, 0.05]), np.array([0.2, -0.4, 5.0]), np.array([0.01, 0.0, 0.02])
    out = wp.zeros(1, dtype=wp.vec3, device="cpu")
    wp.launch(_slide_probe, dim=1, inputs=[wp.array([body], dtype=wp.transform, device="cpu"),
                                           wp.array([wp.spatial_vector(*lin, *ang)], dtype=wp.spatial_vector,
                                                    device="cpu"),
                                           wp.array([com], dtype=wp.vec3, device="cpu"), g, out], device="cpu")
    Rb = Rotation.from_quat(np.array(body.q))
    Rc = Rotation.from_quat(np.array(chart.q))
    v = lin + np.cross(ang, Rb.apply(np.array(app.center)) - Rb.apply(com))
    vc = Rc.inv().apply(v)
    want = vc - np.array([0.0, 0.0, 1.0]) * vc[2]
    assert np.linalg.norm(np.cross(ang, Rb.apply(np.array(app.center)) - Rb.apply(com))) > 0.2   # the spin's share
    assert np.allclose(out.numpy()[0], want, rtol=0.0, atol=1e-6), (out.numpy()[0], want)


def test_the_footprint_follows_the_applicators_sole():
    """A sole facing its body's -y (normal), its first half extent along body z (axis): the body turned so that face
    meets the chart; GA's footprint (centre within the half extents, or cover over 1/4) against numpy, cell by cell."""
    app = Applicator(body="float", shape="rect", size=(0.03, 0.01), center=(0.0, -0.004, 0.0), normal=(0.0, -1.0, 0.0),
                     axis=(0.0, 0.0, 1.0))
    g = _sole(wp.transform_identity(), app)
    q = Rotation.from_euler("xz", [math.pi / 2, 0.4])                  # body -y -> world -z, then a yaw
    body = wp.transform(wp.vec3(0.01, -0.02, 0.006), wp.quat(*q.as_quat()))
    half, cell = wp.vec2(0.03, 0.01), 0.004
    u = (np.arange(-14, 14) + 0.5) * cell
    pts = np.stack(np.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2)
    pts = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)
    out = wp.zeros(len(pts), dtype=wp.int32, device="cpu")
    wp.launch(_inside_probe, dim=len(pts), inputs=[wp.array([body], dtype=wp.transform, device="cpu"), g, half,
                                                   wp.array(pts, dtype=wp.vec3, device="cpu"), out], device="cpu")
    o = np.array(body.p) + q.apply(np.array(app.center))
    loc = q.inv().apply(pts - o - np.outer((pts - o) @ q.apply([0, 1, 0]) / q.apply([0, 1, 0])[2], [0, 0, 1]))
    x, y = np.abs(loc[:, 2]), np.abs(loc[:, 0])                       # along axis (body z) and normal x axis (body x)
    cover = np.clip((0.03 - x) / cell + 0.5, 0, 1) * np.clip((0.01 - y) / cell + 0.5, 0, 1)
    want = ((x <= 0.03) & (y <= 0.01)) | (cover > 0.25)
    assert want.sum() > 50 and np.array_equal(out.numpy().astype(bool), want)
