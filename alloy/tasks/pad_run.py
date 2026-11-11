# SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding of WP4's surface-flow pad tasks (``pad_edge``, ``pad_trowel``): Gym-Anything's pad
(``concrete_pad_start.pad_fields``) on one ``surface_flow`` plane chart at their 2 mm cells, WP6's process layer
(``concrete_paste``) and a scripted carriage (``concrete_tools``); their wait command on the material clock; the
verdict's grids; tools back on the cart.

Each task supplies its own chart split (``base`` fixed, ``film`` above it, ``paste`` mask), tools and program.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import body_xform
from alloy.plugins.surface_flow import FilmMaterial, SurfaceFlow
from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_ports as CP
from alloy.tasks import concrete_verdict as CV
from alloy.tasks.concrete_metrics import PasteGrid
from alloy.tasks.concrete_pad_start import FORM_H, INTERIOR, pad_fields
from alloy.tasks.concrete_paste import ConcretePaste
from alloy.tasks.concrete_tools import ToolCarriage, add_tool_body

SURFACE_VISCOSITY = 5.0                    # Pa·s, the surface layer (WP6's concrete_pad; design note §7, estimate)
PAD_STIFFNESS = 1.0e7                      # Pa/m: a tool's pad (implicit), as patch-flush
WAIT_STEP = 1.0                            # s: the material clock's step in a wait (P:188 WAIT_DT)
U_CAP = 0.2                                # m/s: the film's speed limit; the paste moves no faster than the tools
EXCHANGE_DT = 2.5e-3                       # s: target exchange step (two film substeps at 2 mm cells)
CARRIAGE_OMEGA = 250.0                     # rad/s: the carriage's servos (omega·dt 0.63, 1.4·omega·dt 0.88)
PERIOD = 0.02                              # s: one control period (sim.step_period)
ON_CART = 0.02                             # m: a tool counts as back on the cart within this of its rest
NOT_MEASURED = ("forms_undisturbed", "arms_clear")   # no forms as bodies, no robot: not judged (concrete_ports)
# the fields a verdict's PasteGrid takes from the layer when it publishes them (dwc, bl: WP6's paste, 11-08)
LAYER_FIELDS = ("D", "a", "E", "S", "w", "q", "late", "worked", "wet_worked", "tau_w", "dwc", "bl")


@dataclass(frozen=True)
class ToolBody:
    """A tool's rigid body: its label, half extents [m] (the sole is the box's underside), mass [kg] and its rest on
    the cart (pad frame) where every program starts and ends."""

    name: str
    half: tuple[float, float, float]
    mass: float
    rest: tuple[float, float, float]


def window(task: str, seed: int, temperature: float, air: str,
           extent: tuple[float, float] | None) -> dict[str, Any]:
    """Their start for ``task`` (2 mm), cut to ``extent`` (m east and north of the south-west inner corner) plus the
    forms beside it; None: the whole pad."""
    f = pad_fields(task, seed, temperature, air)
    if extent is None:
        return f
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    i = np.flatnonzero(f["x"] < -hx + extent[0])
    j = np.flatnonzero(f["y"] < -hy + extent[1])
    out = dict(f)
    for k, v in f.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            out[k] = v[i[0]:i[-1] + 1, j[0]:j[-1] + 1]
    out["x"], out["y"] = f["x"][i[0]:i[-1] + 1], f["y"][j[0]:j[-1] + 1]
    return out


def whole(f: dict[str, Any]) -> bool:
    """The chart holds the whole pad (every form), not a corner window."""
    return bool(f["y"][-1] > INTERIOR[1] / 2 and f["x"][-1] > INTERIOR[0] / 2)


def paste_faces(f: dict[str, Any]) -> tuple[float, float, float, float]:
    """The paste's west, east, south and north faces [m]: the outer cell edges of ``paste_mask``. The 2 mm grid meets
    the west, south and north forms on cell edges, but the interior's 1.219 m is 609.5 cells: the last paste cell on
    the east ends at x 0.6085, half a cell inside ``INTERIOR[0] / 2`` (0.6095, the middle of the form's first cell).
    A window's open edges are its last cells' outer edges."""
    m, x, y, half = np.asarray(f["paste_mask"], bool), f["x"], f["y"], 0.5 * float(f["cell"])
    i, j = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    return (float(x[i[0]]) - half, float(x[i[-1]]) + half, float(y[j[0]]) - half, float(y[j[-1]]) + half)


def extent(f: dict[str, Any]) -> tuple[float, float, float, float]:
    """The chart's centre and size (x, y, sx, sy) [m]."""
    x, y, cell = f["x"], f["y"], float(f["cell"])
    return 0.5 * (x[0] + x[-1]), 0.5 * (y[0] + y[-1]), len(x) * cell, len(y) * cell


def build_scene(f: dict[str, Any], tools: Sequence[ToolBody]) -> Callable[[newton.ModelBuilder], None]:
    """The chart's body (a static slab under the pad) and the tools' free bodies at their rests, without rigid
    contacts: the tools meet the work through the film's pads."""
    cx, cy, sx, sy = extent(f)

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(cx, cy, -0.025), hx=sx / 2, hy=sy / 2, hz=0.025, label="pad")
        for t in tools:
            add_tool_body(b, t.name, t.half, t.mass, t.rest, collide=False)
    return build


def surface_film(f: dict[str, Any], temperature: float) -> FilmMaterial:
    """The pad's surface layer as the film's material (its yield is the paste layer's to write)."""
    mat = CM.Concrete()
    tau_s = float(mat.tau_surf(mat.tau_bulk(float(f["age"]), temperature)))
    return FilmMaterial("concrete_surface", density=mat.rho, viscosity=SURFACE_VISCOSITY, yield_stress=tau_s,
                        surface_tension=0.0)


def set_start(flow: SurfaceFlow, layer: ConcretePaste, fs: list[dict[str, Any]],
              split: Callable[[dict[str, Any]], dict[str, np.ndarray]]) -> None:
    """Per world: the chart's fixed base and film from ``split(f)`` (keys base, film, paste) and the paste's start
    states (theirs on the paste cells)."""
    ch = [split(f) for f in fs]
    C = ch[0]["base"].size
    flow.set_input("base", np.stack([c["base"].reshape(C) for c in ch]).astype(np.float32))
    flow.set_depth(np.stack([c["film"].reshape(C) for c in ch]))
    pm = np.stack([c["paste"].reshape(C) for c in ch])
    layer.set_start(paste=pm.astype(np.int32), age=float(fs[0]["age"]),
                    **{k: np.stack([np.where(c["paste"], f[k], 0.0).reshape(C) for c, f in zip(ch, fs, strict=True)])
                       for k in ("E", "a", "D", "S", "w", "q")})


def tools_clear(flow: SurfaceFlow, carriage: ToolCarriage) -> bool:
    """No tool loaded and none down on the film (``tool_ceiling`` is +inf everywhere): their wait's refusal rule
    (``env.py:113-125``)."""
    return bool(np.isinf(flow.read()["tool_ceiling"]).all() and np.all(carriage.load.numpy() <= 0.05))


def rest_ratio(flow: SurfaceFlow, shape: tuple[int, int], cell: float) -> float:
    """An upper bound of every face's yield ratio, as their ``_bound`` (``paste.py:217-232``): below 1 no face of the
    film can move with no tool down. Per face, the paste that could flow is the higher side's (the donor): its depth h,
    its yield tau, and the drop of the surface s = base + h across the face, rho·g·h·|ds|/dx / tau."""
    h = flow.h.numpy().reshape(-1, *shape)
    tau = flow.inputs()["yield_stress"].numpy().reshape(-1, *shape).astype(float)
    s = flow.base.numpy().reshape(-1, *shape) + h
    rho, worst = flow.config.material.density, 0.0
    for ax in (1, 2):
        ds = np.diff(s, axis=ax)
        hi = np.where(ds < 0, 0, 1)                                     # the donor: the higher side of each face
        hd = np.where(hi, np.take(h, range(1, h.shape[ax]), axis=ax), np.take(h, range(h.shape[ax] - 1), axis=ax))
        td = np.where(hi, np.take(tau, range(1, h.shape[ax]), axis=ax), np.take(tau, range(h.shape[ax] - 1), axis=ax))
        r = rho * 9.81 * hd * np.abs(ds) / cell / np.maximum(td, 1e-12)
        worst = max(worst, float(np.where(hd > 1e-9, r, 0.0).max()))
    return worst


def material_wait(sim: Any, flow: SurfaceFlow, layer: ConcretePaste, carriage: ToolCarriage, seconds: float,
                  step: float = WAIT_STEP, rest: Callable[[SurfaceFlow], bool] | None = None) -> float:
    """Advance the paste's material clock alone by ``seconds`` (age, setting, bleed, evaporation) in ``step``s, the
    film unstepped: their wait command (``world.py:322-338``). Exact for a film at rest as a plug. Refused (nothing
    done) unless every tool is off the work and the film rests: ``rest(flow)`` when given (a task's own rule), else
    ``rest_ratio`` < 1 (their ``_bound``). Returns the seconds advanced."""
    if not tools_clear(flow, carriage):
        raise PluginError("material_wait: a tool is down on the paste or loaded; lift it clear first")
    if rest is not None and not rest(flow):
        raise PluginError("material_wait: the film does not rest by the task's rule; step until it rests")
    if rest is None:
        n_xy = tuple(int(n) for n in flow.config.charts[0].cells)
        ratio = rest_ratio(flow, n_xy, float(flow._dx))                     # the chart's cell [m]
        if ratio >= 1.0:
            raise PluginError(f"material_wait: the film can still flow (face yield ratio up to {ratio:.3g}); step "
                              f"until it rests")
    n, done = int(math.ceil(seconds / step - 1e-9)), 0.0
    for _ in range(n):
        h = min(step, seconds - done)
        layer.update(sim.state, sim.state, h)
        done += h
    return done


def layer_fields(layer: ConcretePaste, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """The layer's per-cell states as [W, nx, ny] arrays."""
    n = shape[0] * shape[1]
    return {k: np.asarray(v).reshape(-1, *shape) for k, v in layer.read().items()
            if np.ndim(v) == 2 and np.shape(v)[1] == n}


def surface(flow: SurfaceFlow, shape: tuple[int, int]) -> np.ndarray:
    """The paste's surface per world [W, nx, ny], relative to the form top."""
    return (flow.h.numpy() + flow.base.numpy()).reshape(-1, *shape) - FORM_H


def grids(flow: SurfaceFlow, layer: ConcretePaste, fs: list[dict[str, Any]], **extra: np.ndarray) -> list[PasteGrid]:
    """The verdict's grids per world: heights relative to the form top, film thickness as t, their pad_d as d_edge,
    every state of ``LAYER_FIELDS`` the layer publishes, and ``extra`` fields [W, nx, ny] (e.g. the lip's mark as C)."""
    n = fs[0]["h"].shape
    t = flow.h.numpy().reshape(-1, *n)
    h = surface(flow, n)
    r = layer_fields(layer, n)
    return [PasteGrid(h=h[w], t=t[w], cell=float(f["cell"]), paste_mask=f["paste_mask"], d_edge=f["d_edge"],
                      **{k: r[k][w] for k in LAYER_FIELDS if k in r}, **{k: v[w] for k, v in extra.items()})
            for w, f in enumerate(fs)]


def start_grid(f: dict[str, Any], **extra: np.ndarray) -> PasteGrid:
    """Their start as a verdict grid (nothing worked yet)."""
    z = np.zeros(f["h"].shape)
    return PasteGrid(h=f["h"] - FORM_H, t=f["tb"] + f["ts"], cell=float(f["cell"]), paste_mask=f["paste_mask"],
                     d_edge=f["d_edge"], D=f["D"], a=f["a"], E=f["E"], S=f["S"], w=f["w"], q=f["q"], bl=f["bl"],
                     wet_worked=z.astype(np.int32), late=z.astype(np.int32), worked=z, dwc=z, **extra)


def tools_back(sim: Any, carriage: ToolCarriage, tools: Sequence[ToolBody]) -> np.ndarray:
    """Per world: how many tools are not back at their rests (within ``ON_CART``), from the bodies' end poses."""
    q = sim.state.body_q.numpy()
    ids = carriage.tool.numpy()                                                    # [W, tools] body index
    off = np.zeros(ids.shape[0], int)
    for k, t in enumerate(tools):
        off += np.linalg.norm(q[ids[:, k], :3] - np.asarray(t.rest), axis=1) >= ON_CART
    return off


def verdict_dict(v: Any, port: CV.Port | None = None, **more: Any) -> dict:
    """A pad verdict as JSON by the AB-gym rule (``concrete_verdict.verdict_dict``; ``more``: its ``gates`` and
    ``differs``): success from the judged criteria, with forms and arms not judged (``passed`` None, never a pass by
    default). ``port``: the task's list (``concrete_ports``), by default the one for ``v.task``."""
    return CV.verdict_dict(v, port or CP.PORTS.get(getattr(v, "task", ""), CP.PAD), **more)


def ledger_rel(flow: SurfaceFlow, v0: np.ndarray) -> float:
    """The film's volume ledger against its start, relative, worst world."""
    r = flow.read()
    return float((np.abs(r["ledger"] - v0) / np.maximum(v0, 1e-12)).max())


def step_watch(sim: Any, flow: SurfaceFlow, periods: int, every: int = 25,
               hooks: Mapping[int, Callable[[], None]] | None = None) -> tuple[int, int]:
    """Step ``periods`` control periods, calling ``hooks[i]`` before period i; return the peak faces at the speed cap
    and cells over max_depth."""
    cap = over = 0
    for i in range(periods):
        if hooks and i in hooks:
            hooks[i]()
        sim.step_period()
        if i % every == 0:
            r = flow.read()
            cap, over = max(cap, int(r["faces_at_cap"].max())), max(over, int(r["cells_over_depth"].max()))
    return cap, over
