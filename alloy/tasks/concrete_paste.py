# SPDX-License-Identifier: Apache-2.0
"""The concrete tasks' process layer: per-cell paste states on top of the ``surface_flow`` film (design note §5;
WP15's layer rule: a plug-in listed after ``surface_flow`` that reads its state and never writes it).

Per exchange, one thread per cell of every paste cell (the ``paste`` input is 1):
- **Age and setting.** ``age`` [s] advances; the bulk yield ``tau_b(age)`` follows the material's setting curve and
  the surface's ``tau_s`` its densification D (concrete, repair mortar) or compaction C (masonry mortar); a sealant
  holds tau0 and its surface is 3x that once its skin forms (2 h x MASCO; ``sealant_fast``, their fast-sealant
  variant, 20 min, ``variants.py:33``) (``concrete_materials``; Gym-Anything ``materials.py:23-142``). A sealant has
  no set and does not bleed (theirs neither); tack is not modelled.
- **Bleed water and blisters.** Concrete bleeds at the triangle rate until ``tau_b`` reaches 10 kPa; an open surface
  takes it as a film ``w``, a sealed one traps it as ``q`` (``paste.py:181-183``), and evaporation dries ``w``.
  Trapped water lifts blisters ``bl``: 4 (q - 2e-4)+, box-blurred +-7.5 mm along each axis of the flow's plane
  chart (zero past its edges, as their np.roll reads their pad's 40 mm margin with no water), capped at
  3 mm, never sinking (their ``paste.py`` _blisters; theirs skips rises under 1e-5 m). ``bl`` is a state: it does
  not lift the film (their surface is b + tb + ts + bl, ``paste.py:109``), so a metric that reads the surface adds it.
  Books per cell (float64, m): ``bled`` (risen into w and q), ``evaporated`` (off the film ``w``), ``worked_in``
  (into the top), each what the float32 states took, so w + q + evaporated + worked_in - bled is each cell's start
  water exactly. (A float32 film takes its small steps rounded: here 3e-5 of a 0.1 mm film over a 2 min pass.)
- **Sole work** on the cells under the working tool's sole (GA's footprint, ``paste.py:572-600``: a cell centre
  within the sole's half extents about its applicator's ``center``, or their cover over 1/4) where it bears on the
  paste (``bear``: the lowest tool underside there, surface flow's work field ``tf.wceil`` for every tool, a carrying
  store tool too, within ``BEAR_BAND`` of the surface; ``coupling.py:1001-1003``) while the tool advances faster
  than 1 mm/s, its slide low-passed twice (``coupling.py:57,60,641``; ``coupling.py:999-1094``): with ``ds`` the
  actual slide (the sole centre's in-plane speed, rotation included, x dt, ``coupling.py:479-485, 1009``; 0 at rest),
  ``n = ds/B`` the passes (B the bearing patch's extent along the slide plus a cell, ``coupling.py:1015-1021``) and
  ``p`` the pad pressure, aggregate exposure E, texture amplitude a, densification D, sealing S and compaction C
  follow their per-pass laws; the cell records the stiffness at that working (``tau_w``: ``tau_s`` for concrete
  and repair mortar, ``tau_b`` for masonry mortar and sealant, ``coupling.py:1087-1088``), ``late`` (worked past
  initial set), the distance ``worked``, and ``wet_worked`` (worked while a bleed sheen stood: the film goes into
  the top 6 mm, ``dwc`` += 1000 w / (350 kg/m³ x 6 mm), and ``w`` -> 0, ``coupling.py:1060-1063``). Working a
  skinned sealant tears its skin (``skin_torn``; its texture at least 0.5 mm, ``coupling.py:1081-1083``). Their
  bearing test also asks for paste thicker than 10 µm; here the ``paste`` input marks where paste is.

The pad ``pressure`` [W] is the working tool's vertical load (``concrete_tools.ToolCarriage.load``, the carriage's
commanded load) over its bearing cells' count x the cell area (``coupling.py:1003-1007``; an integer count, the same on
every run), so a sole half off the paste or pitched on one edge presses harder; the working tool is the one carrying the
largest load, or, with none loaded (held at a height), the lowest one. (The film's ``tf.press`` is the pad's own share;
where the slab's rigid contact carries a tool it reads near zero.) States companion layers read by name (``read()``):
``age``, ``tau_s``, ``bear``, ``pressure``, ``dwc``, ``bl``, ``skin_torn``, ``active`` (the working tool's index, -1
none), ``advance``, ``slide`` and the work fields above; the device pieces they share (footprint, touch, slide, advance,
bearing count, blisters) are in ``concrete_paste_work``. Each exchange the layer writes the paste's yield into the
film's declared per-cell inputs (WP15 phase 3), so the paste stiffens as it sets: ``yield_stress`` (with
``per_cell_rheology``) gets ``tau_b`` when the film is bulk paste (an overfill) or ``tau_s`` when it is a slab's surface
layer (``film_yield``), and ``bulk_yield`` (with a ``bulk``) gets ``tau_b``. The flow has one plane chart (its cells
share one area; refused otherwise); the sole and its slide are taken into its frame, which may ride on a body. It is the
only writer of the paste's yield: a flow that also runs surface flow's setting process (``process.setting``) is refused.
Age is float64 (a float32 clock drifts ~15 s over a 25 min task).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_paste_work as PW
from alloy.tasks.concrete_paste_work import BEAR_BAND, MIN_ADVANCE

ADVANCE_TAU = 0.1           # s: the sole's in-plane velocity is low-passed twice (coupling.py:57,60)
RHO_WATER = 1000.0          # kg/m³, as theirs
SETTING, MORTAR, SEALANT = 0, 1, 2


@dataclass(frozen=True)
class SoleWork:
    """A tool's working constants (Gym-Anything ``tools.py``): the texture amplitude it leaves ``a_tool`` [m],
    densification and sealing factors, and its sole's half extents along its body x and y [m]."""

    body: str
    a_tool: float
    dens: float
    seal: float
    half: tuple[float, float]


@wp.struct
class Paste:
    kind: wp.int32
    s: wp.float32               # MASCO factor at the site temperature
    tau0: wp.float32
    a_thix: wp.float32
    tau_is: wp.float32
    t_is: wp.float32
    t_late: wp.float32          # s: the initial set at the site (worked later = late)
    m: wp.float32
    r0: wp.float32
    r1: wp.float32
    tau_ref: wp.float32         # mortar
    t_ref: wp.float32
    harden: wp.float32
    bleed_end: wp.float32       # s
    bleed_mean: wp.float32      # kg/m²/h
    evap: wp.float32            # kg/m²/s
    skin: wp.float32            # s: a sealant's skin time at the site


@wp.struct
class Fields:
    age: wp.array2d[wp.float64]
    D: wp.array2d[wp.float32]
    a: wp.array2d[wp.float32]
    E: wp.array2d[wp.float32]
    S: wp.array2d[wp.float32]
    C: wp.array2d[wp.float32]
    w: wp.array2d[wp.float32]
    q: wp.array2d[wp.float32]
    tau_w: wp.array2d[wp.float32]
    dwc: wp.array2d[wp.float32]
    bl: wp.array2d[wp.float32]
    tau_s: wp.array2d[wp.float32]
    bear: wp.array2d[wp.int32]
    skin_torn: wp.array2d[wp.int32]
    late: wp.array2d[wp.int32]
    worked: wp.array2d[wp.float32]
    wet_worked: wp.array2d[wp.int32]
    bled: wp.array2d[wp.float64]          # m of bleed water risen (books: bled = w + q + evaporated + worked_in)
    evaporated: wp.array2d[wp.float64]    # m of the bleed film evaporated
    worked_in: wp.array2d[wp.float64]     # m of bleed film worked into the top (dwc)


@wp.func
def _taus(p: Paste, age: float, D: float, C: float):
    if p.kind == MORTAR:
        tb = CM.wp_tau_bulk_mortar(age, p.s, p.tau0, p.tau_ref, p.t_ref, p.m)
        return tb, CM.wp_tau_surf_mortar(tb, C, p.harden)
    if p.kind == SEALANT:                               # sealant: tau0; skinned, SKIN_FACTOR x (M:126-142)
        return p.tau0, wp.where(age >= p.skin, p.tau0 * CM.SKIN_FACTOR, p.tau0)
    tb = CM.wp_tau_bulk_setting(age, p.s, p.tau0, p.a_thix, p.tau_is, p.t_is, p.m)
    return tb, CM.wp_tau_surf_setting(tb, D, p.r0, p.r1)


@wp.kernel
def _active_kernel(load: wp.array2d[wp.float32], n_tools: int, tool: wp.array2d[wp.int32],
                   body_q: wp.array[wp.transform], g: PW.Sole, active: wp.array[wp.int32]):
    """Per world: the working tool. The one carrying the largest load over 0.5 N; with none loaded (a tool held at
    a height by the carriage), the lowest one along the chart's normal (parked tools are held high). The cells'
    touch test then decides where it works."""
    w = wp.tid()
    best = float(0.5)
    k_best = int(-1)
    for k in range(n_tools):
        if load[w, k] > best:
            best = load[w, k]
            k_best = k
    if k_best < 0:
        low = float(1.0e30)
        for k in range(n_tools):
            z = wp.dot(wp.transform_get_translation(PW.in_chart(body_q, g, w, tool[w, k])), g.n)
            if z < low:
                low = z
                k_best = k
    active[w] = k_best


@wp.kernel
def _advance_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                    body_com: wp.array[wp.vec3], tool: wp.array2d[wp.int32], g: PW.Sole, dt: float,
                    v1: wp.array2d[wp.vec3], v2: wp.array2d[wp.vec3]):
    """Each tool's slide (its sole centre's velocity in the chart's plane, ``coupling.py:479-485``) and its advance,
    the slide through two first-order low-passes (``coupling.py:57,60``), so a rocking or chattering sole works the
    paste by its stroke, not its jitter."""
    w, k = wp.tid()
    b = tool[w, k]
    a = 1.0 - wp.exp(-dt / ADVANCE_TAU)                 # their continuous filter (0.5 ms ticks), exact per exchange
    v = PW.slide(body_q, body_qd[b], body_com[b], g, w, b, g.center[k])
    g.slid[w, k] = v
    v1[w, k] = v1[w, k] + a * (v - v1[w, k])
    v2[w, k] = v2[w, k] + a * (v1[w, k] - v2[w, k])


@wp.kernel
def _paste_kernel(p: Paste, f: Fields, paste: wp.array2d[wp.int32], h: wp.array2d[wp.float64],
                  base: wp.array2d[wp.float32], ceil: wp.array2d[wp.float32], g: PW.Sole,
                  advance: wp.array2d[wp.vec3], active: wp.array[wp.int32], pressure: wp.array[wp.float32],
                  tool: wp.array2d[wp.int32], body_q: wp.array[wp.transform], a_tool: wp.array[wp.float32],
                  dens: wp.array[wp.float32], seal: wp.array[wp.float32], half: wp.array[wp.vec2], dt: float,
                  dt64: wp.float64, write_yield: int, film_surface: int, film_yield: wp.array2d[wp.float32],
                  write_bulk: int, bulk_yield: wp.array2d[wp.float32]):
    w, c = wp.tid()
    if paste[w, c] == 0:
        return
    age64 = f.age[w, c] + dt64
    f.age[w, c] = age64
    age = float(age64)
    tb, ts = _taus(p, age, f.D[w, c], f.C[w, c])
    f.tau_s[w, c] = ts
    f.bear[w, c] = 0
    if write_yield == 1:                                # the film's declared inputs (WP15 p3): the paste stiffens
        film_yield[w, c] = wp.where(film_surface == 1, ts, tb)
    if write_bulk == 1:
        bulk_yield[w, c] = tb
    if p.kind == SETTING:                               # bleed water: open -> film w, sealed -> trapped q
        r = CM.wp_bleed_rate(age, p.bleed_end, p.bleed_mean) * dt / RHO_WATER
        S = f.S[w, c]
        q0 = f.q[w, c]                                  # (no tuple assignment from struct fields: Warp codegen)
        w0 = f.w[w, c]
        q1 = q0 + S * r
        wo = w0 + (1.0 - S) * r
        wn = wp.max(wo - p.evap * dt / RHO_WATER, 0.0)
        f.q[w, c] = q1
        f.w[w, c] = wn
        # the books take what the float32 states took (exact as float64 differences), so they close exactly
        f.bled[w, c] = f.bled[w, c] + (wp.float64(q1) - wp.float64(q0)) + (wp.float64(wo) - wp.float64(w0))
        f.evaporated[w, c] = f.evaporated[w, c] + (wp.float64(wo) - wp.float64(wn))
    k = active[w]
    if k < 0:
        return
    if not PW.bearing(body_q, g, tool, half, w, k, c, ceil[w, c], base[w, c], h[w, c]):
        return                                          # not under the working tool's bearing sole
    if not PW.advancing(advance[w, k]):
        return
    f.bear[w, c] = 1
    v = g.slid[w, k]                                    # the sole centre's actual slide (coupling.py:1009)
    speed = wp.length(v)
    moving = speed > 1.0e-9
    m = advance[w, k] / wp.length(advance[w, k])        # at rest the advance gives the direction only (ds = 0)
    if moving:
        m = v / speed
    up = wp.transform_vector(PW.in_chart(body_q, g, w, tool[w, k]), g.ez[k])   # the sole's back, chart frame
    B = PW.patch_along(g.lo[w], g.hi[w], g.cell, half[k], moving)   # the bearing patch along the motion
    pres = pressure[w]                                  # over the bearing cells (coupling.py:1003-1007)
    ds = speed * dt
    n = ds / B
    pitch = wp.abs(wp.dot(up, m)) / wp.max(wp.dot(up, g.n), 1.0e-3)
    film = f.w[w, c]
    mi = CM.wp_interface_factor(ts, film)
    f.E[w, c] = CM.wp_embed_update(f.E[w, c], pres, n)
    f.a[w, c] = CM.wp_texture_update(f.a[w, c], a_tool[k], n, pres, ts, mi)
    f.D[w, c] = CM.wp_densify_update(f.D[w, c], ts, ds, mi, dens[k])
    f.S[w, c] = CM.wp_seal_update(f.S[w, c], ds, seal[k], pitch, pres)
    if film > CM.SHEEN_W:                               # a film worked in: the top 6 mm's w/c rises (C:1060-1063)
        f.wet_worked[w, c] = 1
        f.dwc[w, c] = f.dwc[w, c] + 1000.0 * film / (CM.CEMENT * CM.DWC_LAYER)
        f.worked_in[w, c] = f.worked_in[w, c] + wp.float64(film)
        f.w[w, c] = 0.0
    if p.kind == MORTAR:
        f.C[w, c] = CM.wp_compact_update(f.C[w, c], pres, ds, tb, CM.wp_interface_factor(ts, f.w[w, c]))
        # after the worked film is gone, as theirs (C:1069 reads it after C:1063)
    if p.kind == SEALANT and age >= p.skin:             # a skinned sealant's skin tears (C:1081-1083)
        f.skin_torn[w, c] = 1
        f.a[w, c] = wp.max(f.a[w, c], CM.TORN_A)
    f.tau_w[w, c] = wp.where(p.kind == SETTING, ts, tb) # the surface for concrete / repair mortar, else the bulk
                                                        # at last working (coupling.py:1087-1088)
    if age > p.t_late:
        f.late[w, c] = 1
    f.worked[w, c] = f.worked[w, c] + ds


class ConcretePaste(ForcePlugin):
    """Per-cell paste process states (age, setting, bleed, sole work) for a ``SurfaceFlow`` worked by a carriage."""

    name = "concrete_paste"
    summary = "concrete / mortar process layer: setting, bleed water, texture, densification, sealing, compaction"

    @dataclass
    class Config:
        material: str = "repair_mortar"                    # a MATERIALS key: "concrete", "repair_mortar",
                                                           # "masonry_mortar", "sealant" or "sealant_fast"
        temperature: float = 21.1                          # °C at the site
        air: str = "still"                                 # evaporation preset (concrete_materials.AIR)
        tools: tuple[Any, ...] = ()                         # SoleWork per carriage program, same order
        film_yield: str = "bulk"                           # the film's yield: "bulk" paste (an overfill) or the
                                                           # "surface" layer's (a slab's cream, tau_s)

        def __post_init__(self) -> None:
            if self.film_yield not in ("bulk", "surface"):
                raise ValueError(f"film_yield must be 'bulk' or 'surface', got {self.film_yield!r}")
            if self.material not in MATERIALS:
                raise ValueError(f"material must be one of {sorted(MATERIALS)}, got {self.material!r}")
            self.tools = tuple(t if isinstance(t, SoleWork) else SoleWork(**t) for t in self.tools)

    def __init__(self, config: Any = None, *, flow: Any = None, carriage: Any = None, label: str | None = None,
                 **params: Any) -> None:
        super().__init__(config, label=label, **params)
        if flow is None or carriage is None:
            raise PluginError("concrete_paste: pass flow= (the SurfaceFlow it reads) and carriage= (its tools)")
        self.flow, self.carriage = flow, carriage

    def setup(self, ctx: Any) -> None:
        c = self.config
        W, C = ctx.world_count, self.flow.tables.cell_count
        progs = self.carriage.config.programs
        if [t.body for t in c.tools] != [p.body for p in progs]:
            raise PluginError(f"{self.label}: tools {[t.body for t in c.tools]} must match the carriage's programs "
                              f"{[p.body for p in progs]} in order")
        self.mat = MATERIALS[c.material]
        self.p = _paste_struct(self.mat, c.temperature, c.air)
        self.f = Fields()
        self.f.age = self.add_state("age", ctx.zeros((W, C), wp.float64))
        for name in ("bled", "evaporated", "worked_in"):                  # the bleed water's books (per cell, m)
            setattr(self.f, name, self.add_state(name, ctx.zeros((W, C), wp.float64)))
        for name in ("D", "a", "E", "S", "C", "w", "q", "tau_w", "worked", "dwc", "bl", "tau_s"):
            setattr(self.f, name, self.add_state(name, ctx.zeros((W, C), wp.float32)))
        for name in ("late", "wet_worked", "bear", "skin_torn"):
            setattr(self.f, name, self.add_state(name, ctx.zeros((W, C), wp.int32)))
        self.pressure = self.add_state("pressure", ctx.zeros(W, wp.float32))
        self.count = ctx.zeros(W, wp.int32)                                  # the working sole's bearing cells
        self._blister = self._blister_grid(ctx, W, C)
        self.paste = self.add_input("paste", ctx.zeros((W, C), wp.int32))
        self.active = self.add_state("active", ctx.zeros(W, wp.int32))      # the working tool (-1: none)
        T = len(c.tools)
        self._sole = self._sole_geometry(ctx, W, T)
        self.v1 = self.add_state("advance_1", ctx.zeros((W, T), wp.vec3))
        self.advance = self.add_state("advance", ctx.zeros((W, T), wp.vec3))
        self.a_tool = ctx.array(np.array([t.a_tool for t in c.tools], np.float32), wp.float32)
        self.dens = ctx.array(np.array([t.dens for t in c.tools], np.float32), wp.float32)
        self.seal = ctx.array(np.array([t.seal for t in c.tools], np.float32), wp.float32)
        self.half = ctx.array(np.array([t.half for t in c.tools], np.float32).reshape(T, 2), wp.vec2)
        dummy = ctx.zeros((1, 1), wp.float32)
        fc = self.flow.config
        if getattr(fc.process, "setting", None) is not None:
            raise PluginError(f"{self.label}: the flow runs surface flow's setting process, which also writes the "
                              f"paste's yield; drop process.setting (this layer sets the paste by its material's "
                              f"curve) or drop this layer")
        ins = self.flow.inputs()
        self._yield = (1, ins["yield_stress"]) if fc.per_cell_rheology else (0, dummy)
        self._bulk = (1, ins["bulk_yield"]) if fc.bulk is not None else (0, dummy)

    def _sole_geometry(self, ctx: Any, W: int, T: int) -> PW.Sole:
        """GA's footprint inputs (``paste.py:572-600``): the cells' centres, the chart's one normal and cell (their
        bearing area is n x cell², ``coupling.py:1006``, so the cells must share one area), each tool's sole centre
        (its applicator's ``center``), its slide (state ``slide``) and the bearing patch's bounds."""
        tb, charts = self.flow.tables, self.flow.config.charts
        if len(charts) != 1 or not isinstance(charts[0], PlaneChart):
            raise PluginError(f"{self.label}: a sole works one plane (their region): the flow's one plane chart, got "
                              f"{len(charts)} charts")
        area, nrm = tb.area_np, tb.cnrm.numpy()
        if not np.allclose(area, area[0], rtol=1e-9, atol=0.0):
            raise PluginError(f"{self.label}: the bearing area is n x cell² (coupling.py:1006): the flow's cells must "
                              f"share one area, got {area.min():.4g} to {area.max():.4g} m²")
        apps = {a.body: a for a in self.flow.config.tools}
        missing = [t.body for t in self.config.tools if t.body not in apps]
        if missing:
            raise PluginError(f"{self.label}: tools {missing} have no applicator on the flow (their work fields and "
                              f"sole centre)")
        self._cell_area = float(area[0])
        g = PW.Sole()
        g.pos, g.n, g.cell = tb.cpos, wp.vec3(*(float(v) for v in nrm[0])), float(np.sqrt(area[0]))
        g.cbody, g.clocal = tb.cbody, tb.clocal
        axes = np.array([sole_axes(apps[t.body]) for t in self.config.tools], np.float32).reshape(T, 4, 3)
        g.center, g.ex, g.ey, g.ez = (ctx.array(np.ascontiguousarray(axes[:, i]), wp.vec3) for i in range(4))
        g.slid = self.slide = self.add_state("slide", ctx.zeros((W, T), wp.vec3))
        g.lo, g.hi = ctx.full(W, PW.FAR, wp.float32), ctx.full(W, -PW.FAR, wp.float32)
        return g

    def _blister_grid(self, ctx: Any, W: int, C: int) -> tuple | None:
        """(nx, ny, k along i, k along j, two [W, C] scratch arrays) for the blister blur on the flow's one plane
        chart, +-7.5 mm along each axis (their k = round(0.0075 / cell), per axis here); None when the paste does
        not bleed (no trapped water, no blisters)."""
        if self.p.kind != SETTING:
            return None
        charts = self.flow.config.charts
        if len(charts) != 1 or not isinstance(charts[0], PlaneChart):
            raise PluginError(f"{self.label}: blisters need the flow's one plane chart, got {len(charts)} charts")
        nx, ny = (int(v) for v in charts[0].cells)
        pos = self.flow.tables.cpos.numpy()
        di = float(np.linalg.norm(pos[ny] - pos[0])) if nx > 1 else 1.0      # cell c = i·ny + j
        dj = float(np.linalg.norm(pos[1] - pos[0])) if ny > 1 else 1.0
        ki, kj = (max(1, int(round(CM.BLISTER_BLUR_M / d))) for d in (di, dj))
        return nx, ny, ki, kj, ctx.zeros((W, C), wp.float32), ctx.zeros((W, C), wp.float32)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, fl = self.ctx, self.flow
        W, C = ctx.world_count, fl.tables.cell_count
        T = len(self.config.tools)
        wp.launch(_active_kernel, dim=W, inputs=[self.carriage.load, T, self.carriage.tool, state_out.body_q,
                                                 self._sole],
                  outputs=[self.active], device=ctx.device)
        g = self._sole
        wp.launch(_advance_kernel, dim=(W, T),
                  inputs=[state_out.body_q, state_out.body_qd, ctx.model.body_com, self.carriage.tool, g, float(dt)],
                  outputs=[self.v1, self.advance], device=ctx.device)
        # tf.wceil: surface flow's work field for every tool, store tools included ([W, C]; FREE_GAP = no tool down;
        # docs/notes/surface-flow-api.md, work fields): any sole that touches and advances works the paste, a
        # carrying tool too (GA's _work), on the cells under the working sole's footprint
        self.count.zero_()
        g.lo.fill_(PW.FAR)
        g.hi.fill_(-PW.FAR)
        wp.launch(PW.bear_count_kernel, dim=(W, C),
                  inputs=[self.paste, fl.h, fl.base, fl.tf.wceil, g, self.active, self.carriage.tool,
                          state_out.body_q, self.half],
                  outputs=[self.count], device=ctx.device)
        wp.launch(PW.pressure_kernel, dim=W, inputs=[self.carriage.load, self.active, self.count, self._cell_area],
                  outputs=[self.pressure], device=ctx.device)
        wp.launch(_paste_kernel, dim=(W, C),
                  inputs=[self.p, self.f, self.paste, fl.h, fl.base, fl.tf.wceil, g, self.advance,
                          self.active, self.pressure, self.carriage.tool, state_out.body_q,
                          self.a_tool, self.dens, self.seal, self.half, float(dt), wp.float64(dt), self._yield[0],
                          int(self.config.film_yield == "surface"), self._yield[1], self._bulk[0], self._bulk[1]],
                  device=ctx.device)
        if self._blister is not None:                   # trapped water lifts blisters (their paste.py _blisters)
            nx, ny, ki, kj, a, b = self._blister
            wp.launch(PW.lift_kernel, dim=(W, C), inputs=[self.f.q], outputs=[a], device=ctx.device)
            wp.launch(PW.blur_kernel, dim=(W, C), inputs=[a, nx, ny, ki, 1], outputs=[b], device=ctx.device)
            wp.launch(PW.blur_kernel, dim=(W, C), inputs=[b, nx, ny, kj, 0], outputs=[a], device=ctx.device)
            wp.launch(PW.blister_kernel, dim=(W, C), inputs=[a], outputs=[self.f.bl], device=ctx.device)

    def set_start(self, **fields: Any) -> None:
        """Start fields between periods: any of age, D, a, E, S, C, w, q, dwc, bl (scalar, [C] or [W, C]) and paste
        (the mask of paste cells)."""
        for name, values in fields.items():
            if name == "paste":
                self.set_input("paste", np.asarray(values, np.int32))
                continue
            arr = self._state.get(name)
            if arr is None or name in ("late", "wet_worked", "tau_w", "worked", "bear", "tau_s", "pressure",
                                       "skin_torn", "active", "advance", "advance_1", "slide", "bled", "evaporated",
                                       "worked_in"):
                raise PluginError(f"{self.label}: cannot start {name!r}; start fields are age, D, a, E, S, C, w, q, "
                                  f"dwc, bl")
            full = np.broadcast_to(np.asarray(values, wp.dtype_to_numpy(arr.dtype)), arr.shape).copy()
            if not np.isfinite(full).all():
                raise PluginError(f"{self.label}: start field {name} must be finite")
            arr.assign(full)

    def taus(self) -> tuple[np.ndarray, np.ndarray]:
        """Host: (tau_b, tau_s) [W, C] at the current ages (Pa), from the numpy curves."""
        r = self.read()
        tb = np.asarray(self.mat.tau_bulk(r["age"], self.config.temperature), np.float64)
        if isinstance(self.mat, CM.Sealant):
            return tb, np.asarray(self.mat.tau_surf(tb, r["age"], self.config.temperature), np.float64)
        ts = np.asarray(self.mat.tau_surf(tb, r["C"] if self.config.material == "masonry_mortar" else r["D"]))
        return tb, ts


def sole_axes(app: Any) -> list[tuple[float, float, float]]:
    """An applicator's sole in its body frame: centre, then unit axes along its first half extent (``axis``), its
    second (``normal`` x ``axis``) and its back (-``normal``)."""
    nrm = np.asarray(app.normal, float) / np.linalg.norm(app.normal)
    ax = np.asarray(app.axis, float) - np.dot(app.axis, nrm) * nrm
    if np.linalg.norm(ax) < 1e-9:
        raise PluginError(f"concrete_paste: applicator {app.body!r}: axis {app.axis} lies along its normal")
    ax /= np.linalg.norm(ax)
    return [tuple(app.center), tuple(ax), tuple(np.cross(nrm, ax)), tuple(-nrm)]


MATERIALS: dict[str, Any] = {"concrete": CM.Concrete(), "repair_mortar": CM.RepairMortar(),
                             "masonry_mortar": CM.MasonryMortar(), "sealant": CM.Sealant(),
                             "sealant_fast": CM.Sealant(skin_s=CM.FAST_SKIN)}   # their fast-sealant (variants.py:33)


def _paste_struct(mat: Any, temp_c: float, air: str) -> Paste:
    p = Paste()
    p.s = float(CM.masco_factor(temp_c))
    if isinstance(mat, CM.MasonryMortar):
        p.kind = MORTAR
        p.tau0, p.tau_ref, p.t_ref, p.m = float(mat.tau0), float(mat.tau_ref), float(mat.t_ref), float(mat.m)
        p.harden, p.t_is, p.t_late, p.bleed_end, p.bleed_mean = float(mat.harden), 1.0e30, 1.0e30, 0.0, 0.0
        p.evap = 0.0
        return p
    if isinstance(mat, CM.Sealant):                     # tau0 at every age, skinned after skin_time; no bleed
        p.kind = SEALANT
        p.tau0, p.skin = float(mat.tau0), float(mat.skin_time(temp_c))
        p.t_is, p.t_late, p.bleed_end, p.bleed_mean, p.evap = 1.0e30, 1.0e30, 0.0, 0.0, 0.0
        return p
    p.kind = SETTING
    p.tau0, p.a_thix, p.tau_is, p.t_is, p.m = (float(mat.tau0), float(mat.a_thix), float(mat.tau_is),
                                               float(mat.t_is), float(mat.m))
    p.r0, p.r1 = float(mat.surf_r[0]), float(mat.surf_r[1])
    p.bleed_end, p.bleed_mean = float(mat.bleed_end_time(temp_c)), float(mat.bleed_mean)
    p.t_late = float(mat.initial_set_time(temp_c))
    p.evap = float(CM.air_evaporation_kg_m2_h(air, temp_c)) / 3600.0    # the room's air shifted with the site
    if not math.isfinite(p.bleed_end):
        raise PluginError(f"concrete_paste: {mat} has no bleed end")
    return p


__all__ = ["ConcretePaste", "MATERIALS", "SoleWork", "sole_axes"]
