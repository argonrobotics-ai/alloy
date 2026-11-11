# SPDX-License-Identifier: Apache-2.0
"""pad-finish's stage plumbing (:mod:`alloy.tasks.pad_finish`): what one stage's simulation hands the next at a rest
point, read from the plug-ins' registered states (``state_arrays``, not a hand list), and how the next takes it, only
through the host API between periods (WP15's and DC5's conditions; :mod:`alloy.tasks.pad_finish_handoff` has the
field rules).

Each state of the film (``surface_flow``) and the paste layer (``concrete_paste``) is one of:
- **carried**: the film's depth (``set_depth``) and base (``set_input("base")``); the layer's start states and age
  (``set_start``);
- **a record** (the layer's ``late``, ``wet_worked``, ``worked``, ``tau_w``, ``skin_torn``): never started; ``late``
  and ``wet_worked`` ORed and ``worked`` summed across stages on the host (``Records``); ``tau_w`` and ``skin_torn``
  end with their stage (no pad-finish criterion reads them, T:398-596);
- **a book** (the film's running totals: supplied, spilled, drained, evaporated, ...; the layer's bleed water: bled,
  evaporated, worked in, as ``water_*`` m³): summed across stages on the host for the whole chain's books;
- **at rest**: what moves only while a tool works or the film flows (face velocity and flux, pending sources, the
  tools' advance, the carriage's load). They are not carried; the hand-off measures them (``Carry.rest``) and a rest
  point holds them at or near zero;
- **stage-local** (``STAGE_LOCAL``: a stage's tool stores, tub, supply network, drips and pools, e.g. WP6's margin
  trowel and tub): they end with the stage. The stores must be empty (``check_rest``) and no drip in flight
  (``src_drip_v`` in ``Carry.rest``); what the film gave the stores (``to_tools``) is a book.
A state of neither plug-in in these lists stops the hand-off (``unlisted``): a new state must be placed first. A
sole's roll (WP13's cap) is film on the cells ahead of it, not a state, so a lifted tool has laid it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks.concrete_tools import HEIGHT, Stroke
from alloy.tasks.pad_finish_handoff import START_STATES, Records, exact_split

LAYER_RECORDS = ("late", "wet_worked", "worked", "tau_w", "skin_torn")
LAYER_BOOKS = ("bled", "evaporated", "worked_in")        # the bleed water's books [m per cell] -> water_* [m³]
LAYER_REST = ("active", "advance", "advance_1", "slide", "pressure", "bear", "tau_s")   # transient or derived
FLOW_CARRIED = ("h",)
FLOW_BOOKS = ("supplied", "spilled", "drained", "drained_total", "evaporated", "absorbed", "dripped", "shed",
              "pooled", "pooled_total", "clipped", "travel", "tool_roll_spill", "tool_overflow_off",
              "tool_overflow_off_sp", "to_tools")
FLOW_REST = ("u", "q", "gface", "src_acc", "src_mom", "shed_mom", "cwet", "drain_cap", "flooded", "impact_p",
             "impact_share",
             "jet_cover", "jet_tau", "pool_jet_tau", "tile_act", "tick", "g_sum", "alpha_sum", "tool_drag_capped",
             "tool_drag_scale", "tool_pad_ratio")
CARRIAGE_REST = ("load", "release")
STAGE_LOCAL = ("tool_", "tray_", "net_", "supply_", "jet_hit_", "pool_", "src_drip_", "tank_", "nose_")   # prefixes
# ^ nose_low / nose_seen: a scraping store's nose, its lowest point this pass and the exchange it last covered
MOVING = {"flow": ("u", "q", "src_acc", "src_drip_v"), "layer": ("advance", "advance_1", "slide"),
          "carriage": ("load",)}
# ^ what must read ~0 at a rest point: the film's face velocity and flux, pending inflow; the tools' slide; the load


@dataclass
class Carry:
    """One rest point's hand-off, per world [W, C]: the film's depth and base, the layer's start states, the records
    and books so far, and the size of what was dropped at rest (max |value| per state)."""

    h: np.ndarray
    base: np.ndarray
    paste: np.ndarray
    layer: dict[str, np.ndarray]
    records: Records
    books: dict[str, np.ndarray] = field(default_factory=dict)
    rest: dict[str, float] = field(default_factory=dict)


def unlisted(flow: Any, layer: Any, carriage: Any) -> list[str]:
    """The registered states this hand-off does not place (a new plug-in state must be placed first)."""
    known = {"flow": FLOW_CARRIED + FLOW_BOOKS + FLOW_REST,
             "layer": ("age", *START_STATES) + LAYER_RECORDS + LAYER_BOOKS + LAYER_REST, "carriage": CARRIAGE_REST}
    out = []
    for name, p in (("flow", flow), ("layer", layer), ("carriage", carriage)):
        out += [f"{name}.{k}" for k in p.state_arrays()
                if k not in known[name] and not (name == "flow" and k.startswith(STAGE_LOCAL))]
    return out


def require_placed(flow: Any, layer: Any, carriage: Any, *others: tuple[Any, tuple[str, ...]]) -> None:
    """Refuse a stage before it steps when a plug-in registers a state this hand-off does not place (a new engine
    state must be placed first; the hand-off would refuse it after the whole stage). ``others``: the stage's other
    plug-ins, each with the states the stage places itself (the edger's lip: ``lip``, carried as ``compact``)."""
    extra = unlisted(flow, layer, carriage) + [f"{type(p).__name__}.{k}" for p, placed in others
                                               for k in p.state_arrays() if k not in placed]
    if extra:
        raise ValueError(f"pad-finish stage: unplaced states {extra} (place them in pad_finish_stage first)")


def carry(flow: Any, layer: Any, carriage: Any, records: Records | None = None,
          books: dict[str, np.ndarray] | None = None) -> Carry:
    """The hand-off at a rest point: carried states, records folded into ``records``, books added to ``books``."""
    if extra := unlisted(flow, layer, carriage):
        raise ValueError(f"pad-finish hand-off: unplaced states {extra}")
    fs, ls, cs = (p.state_arrays() for p in (flow, layer, carriage))
    rec = records or Records()
    rec.add({k: ls[k].numpy() for k in ("late", "wet_worked", "worked")})
    out_books = dict(books or {})
    for k in FLOW_BOOKS:
        if k in fs:
            per_world = fs[k].numpy().astype(np.float64).reshape(fs[k].shape[0], -1).sum(axis=1)
            out_books[k] = out_books.get(k, 0.0) + per_world
    area = np.asarray(flow.tables.area_np, np.float64)
    for k in LAYER_BOOKS:
        if k in ls:
            out_books[f"water_{k}"] = out_books.get(f"water_{k}", 0.0) + ls[k].numpy() @ area
    rest = {}
    for name, states in (("flow", fs), ("layer", ls), ("carriage", cs)):
        for k in MOVING[name]:
            if k in states:
                v = np.asarray(states[k].numpy(), np.float64)
                rest[f"{name}.{k}"] = float(np.abs(v).max()) if v.size else 0.0
    layer_start = {k: ls[k].numpy().copy() for k in ("age", *START_STATES)}
    return Carry(h=fs["h"].numpy().astype(np.float64).copy(), base=flow.inputs()["base"].numpy().copy(),
                 paste=layer.inputs()["paste"].numpy().copy(), layer=layer_start, records=rec, books=out_books,
                 rest=rest)


def seed_yields(flow: Any, layer: Any) -> None:
    """The film's per-cell yield inputs as the paste has them now (the layer writes them each exchange after the film
    steps, on the paste only, so a fresh simulation's first exchange would read the film material's build-time yield):
    on the paste, tau_s for a surface-layer film, else tau_b; off it, where film stands it is the forms' smear, which
    their region keeps in tb (``layout.py:263-269``) and so yields as the bulk, tau_b (a stage's own material, a
    surface layer's yield, let 20 mm of darby smear on a form top flow at 10 min), at the site's one clock
    (``site_taus``). ``bulk_yield`` tau_b everywhere, likewise."""
    ins, paste = flow.inputs(), layer.inputs()["paste"].numpy() > 0
    tb, ts = layer.taus()
    tb_site = site_taus(tb, paste)
    if "yield_stress" in ins:
        own = ts if layer.config.film_yield == "surface" else tb
        flow.set_input("yield_stress", np.where(paste, own, tb_site).astype(np.float32))
    if "bulk_yield" in ins:
        flow.set_input("bulk_yield", tb_site.astype(np.float32))


def site_taus(tb: np.ndarray, paste: np.ndarray) -> np.ndarray:
    """tau_b [W, C] with every cell off the paste at the paste's own (one clock, their region's): the layer ages only
    the paste's cells (``concrete_paste._paste_kernel``), so its ``taus`` off the paste are at the build age (the
    darby's 5 min, 1641 Pa, read at its 10.8 min hand-off: rest ratio 1.021 in seed 4's world). A world with no paste
    keeps its own."""
    tb, paste = np.asarray(tb, np.float64), np.asarray(paste, bool)
    at = np.where(paste, tb, -np.inf).max(axis=1, keepdims=True)
    return np.where(paste | ~paste.any(axis=1, keepdims=True), tb, at)


def resume(c: Carry, flow: Any, layer: Any) -> None:
    """Start the next stage's simulation (same chart) from ``c``, between periods, through the host API only."""
    flow.set_input("base", c.base)
    flow.set_depth(c.h)
    layer.set_start(paste=c.paste, **c.layer)
    seed_yields(flow, layer)


def start_stage(flow: Any, layer: Any, fs: list[dict[str, Any]], chart: Any) -> None:
    """A stage's start from its fields ``fs`` (one per world), through the host API: the chart split by its owner's
    rule ``chart(f)`` (base, film, paste) made exact for the surface (``exact_split``), the layer's start states that
    the fields hold, and each world's age."""
    parts = [exact_split(chart(f), f["h"]) for f in fs]
    C = parts[0]["base"].size
    flow.set_input("base", np.stack([p["base"].reshape(C) for p in parts]))
    flow.set_depth(np.stack([p["film"].reshape(C) for p in parts]))
    keys = [k for k in START_STATES if k in fs[0]]
    paste = [np.asarray(p["paste"]) > 0.5 for p in parts]
    layer.set_start(paste=np.stack([m.reshape(C) for m in paste]).astype(np.int32),
                    age=np.stack([np.full(C, float(f["age"])) for f in fs]),
                    **{k: np.stack([np.where(m, np.asarray(f[k], np.float64), 0.0).reshape(C)
                                    for f, m in zip(fs, paste, strict=True)]) for k in keys})
    seed_yields(flow, layer)


RESEAT = 10                                # periods: their RESEAT_S (0.2 s) of full ticks after a wait (env.py:23, 107)


def _ratios(flow: Any) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Per axis, every face's rest ratio (``pad_run.rest_ratio``'s formula: the donor, the higher side, its depth h
    and yield tau, rho g h |ds| / dx / tau) and the donor's fields, [W, faces along the axis]."""
    n_xy = tuple(int(n) for n in flow.config.charts[0].cells)
    h = flow.h.numpy().reshape(-1, *n_xy)
    tau = flow.inputs()["yield_stress"].numpy().reshape(-1, *n_xy).astype(float)
    s = flow.base.numpy().reshape(-1, *n_xy) + h
    cell, rho = float(np.sqrt(flow.tables.area_np[0])), flow.config.material.density
    out, donors = [], []
    for ax in (1, 2):
        ds = np.diff(s, axis=ax)
        hi = ds >= 0
        pick = [np.take(a, range(a.shape[ax] - 1), axis=ax) for a in (h, tau)] + \
               [np.take(a, range(1, a.shape[ax]), axis=ax) for a in (h, tau)]
        hd, td = np.where(hi, pick[2], pick[0]), np.where(hi, pick[3], pick[1])
        out.append(np.where(hd > 1e-9, rho * 9.81 * hd * np.abs(ds) / cell / np.maximum(td, 1e-12), 0.0))
        donors.append({"donor_h": hd, "drop": ds, "tau": td})
    return out, donors


def film_rest(flow: Any) -> dict[str, Any]:
    """The film's rest by one of two rules (WP15, DC5, 11-07): ``bound``, every face's rest ratio under 1
    (``pad_run.rest_ratio``: no face can yield), or ``held``, its own last film step moved no face (speed and flux 0
    on every face of every world: its yield law held each one). Under ``held`` the film stays held while the paste
    ages, as its yields only grow (their ``dirty`` flag, ``paste.py:87``); ``rest_after`` re-checks it after every wait.
    The bound over-reads where the face's yield is the stiffer neighbour's (the film's face law) or a feature is narrow
    and steep (their ``_bound`` caps a feature by the cut height 3.83 tau/(rho g), ``paste.py:217-232``): the edger's
    spill on a form top reads 1.1, held. Also the faces at a ratio of 1 or more (``over``)."""
    rs, _ = _ratios(flow)
    bound = max((float(r.max()) for r in rs if r.size), default=0.0)
    over = int(sum(int((r >= 1.0).sum()) for r in rs))
    face = float(max(np.abs(flow.u.numpy()).max(), np.abs(flow.q.numpy()).max()))
    rule = "bound" if bound < 1.0 else ("held" if face == 0.0 else "none")
    return {"bound": bound, "over": over, "face_max": face, "rule": rule, "rest": rule != "none"}


def _clear(flow: Any, carriage: Any) -> bool:
    from alloy.tasks.pad_run import tools_clear  # noqa: PLC0415

    return True if carriage is None and getattr(flow, "tools", None) is None else bool(tools_clear(flow, carriage))


def rest_after(sim: Any, flow: Any, carriage: Any, periods: int = RESEAT) -> dict[str, Any]:
    """``film_rest`` from ``periods`` full periods stepped with every tool clear (WP15: a ``held`` must come from a film
    step that ran after the tools were clear; their wait ends with RESEAT_S of full ticks): refused if a tool is down
    or loaded before or after."""
    if periods < 1 or not _clear(flow, carriage):
        raise ValueError("pad-finish rest: a tool is down on the paste or loaded (or no period to step)")
    for _ in range(periods):
        sim.step_period()
    if not _clear(flow, carriage):
        raise ValueError("pad-finish rest: a tool came down on the paste while the film was watched")
    return {**film_rest(flow), "periods": periods}


def check_opening(stepped_s: float, start: float) -> None:
    """Refuse a stage whose opening watch and wait stepped the device clock ``stepped_s`` to its program's first key
    ``start``: the carriage would leave the cart mid-watch, toward a target already part-way into its program (WP15,
    M1). The 0.2 s watch takes 10 periods; it steps on while the film moves, and a wait adds 10 per command."""
    if stepped_s >= start:
        raise ValueError(f"pad-finish: the opening watch and wait stepped {stepped_s:.2f} s, to the program's "
                         f"first key at {start:.2f} s")


def shifted_hooks(hooks: dict[int, Any], periods: int) -> dict[int, Any]:
    """The ride watch's hooks (by program period) on the stage's clock after its opening's ``periods``; refused when
    any falls inside the opening (its run would go unchecked: WP15, M1)."""
    early = sorted(k for k in hooks if k < periods)
    if early:
        raise ValueError(f"pad-finish: {len(early)} ride checks fall inside the opening's {periods} periods "
                         f"(first at period {early[0]})")
    return {k - periods: fn for k, fn in hooks.items()}


def check_rest(flow: Any, carriage: Any, rest: dict[str, Any], stores: bool = True) -> dict[str, Any]:
    """A hand-off's rest point (WP15, DC5; their wait's rule, env.py:113-126): no tool down or loaded, nothing in a
    tool's store (else it would vanish with the stage), the film at rest by ``rest`` (``rest_after``'s, the last thing
    stepped). Returns ``rest``."""
    read = flow.read()
    if not _clear(flow, carriage):
        raise ValueError("pad-finish hand-off: a tool is down on the paste or loaded")
    if stores and "tool_load" in read and float(np.abs(read["tool_load"]).max()) > 0.0:
        raise ValueError(f"pad-finish hand-off: {float(read['tool_load'].sum()) * 1e6:.3g} mL left on the tools")
    if int(rest.get("periods", 0)) < 1:
        raise ValueError("pad-finish hand-off: the film's rest must come from a period stepped with the tools clear")
    if not rest["rest"]:
        raise ValueError(f"pad-finish hand-off: the film can still flow (bound {rest['bound']:.3g}, faces moving "
                         f"{rest['face_max']:.3g}; worst {rest_worst(flow)})")
    return rest


def rest_worst(flow: Any) -> dict[str, float]:
    """Where the rest ratio is largest: world, cell, axis, the donor's depth, the surface drop, its yield, and the
    film's largest face speed (a ratio over 1 with no speed is the bound's slack, not flow)."""
    rs, ds = _ratios(flow)
    best: dict[str, float] = {"ratio": 0.0}
    for ax, (r, d) in enumerate(zip(rs, ds, strict=True), start=1):
        if not r.size:
            continue
        k = np.unravel_index(int(r.argmax()), r.shape)
        if float(r[k]) > best["ratio"]:
            best = {"ratio": float(r[k]), "world": int(k[0]), "i": int(k[1]), "j": int(k[2]), "axis": ax,
                    **{n: float(v[k]) for n, v in d.items()}}
    best["u_max"] = float(np.abs(flow.u.numpy()).max())
    return best


def end_surface(c: Carry) -> np.ndarray:
    """The surface at a hand-off [W, C], float64: the chart's base (float32) plus the film."""
    return c.base.astype(np.float64) + c.h


def run_keys(a: tuple[float, float], b: tuple[float, float], yaw: float, z: float, t: float, *, lift: float = 0.02,
             touch: float = 0.004, ramp: float = 0.02, speed: float = 0.2) -> tuple[list[Stroke], float]:
    """One run of a held (HEIGHT) sole from ``a`` to ``b`` with its body at ``z``, as WP4's trowel runs (hung ``lift``
    over the start, set down ``touch`` above, brought down over ``ramp``, run, lifted), from time ``t``; returns the
    keys and the time the lift ends."""
    k = ramp / math.dist(a, b)
    ramped = (a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1]))
    keys = [Stroke(t, (*a, z + lift), yaw=yaw, mode=HEIGHT)]
    t += 0.3
    keys.append(Stroke(t, (*a, z + touch), yaw=yaw, mode=HEIGHT))
    t += ramp / speed
    keys.append(Stroke(t, (*ramped, z), yaw=yaw, mode=HEIGHT))
    t += (math.dist(a, b) - ramp) / speed
    keys.append(Stroke(t, (*b, z), yaw=yaw, mode=HEIGHT))
    t += 0.3
    keys.append(Stroke(t, (*b, z + lift), yaw=yaw, mode=HEIGHT))
    return keys, t


__all__ = ["CARRIAGE_REST", "Carry", "FLOW_BOOKS", "FLOW_CARRIED", "FLOW_REST", "LAYER_BOOKS", "LAYER_RECORDS",
           "LAYER_REST", "MOVING", "RESEAT", "STAGE_LOCAL", "carry", "check_opening", "check_rest", "end_surface",
           "film_rest", "require_placed", "rest_after", "rest_worst", "resume", "run_keys", "seed_yields",
           "shifted_hooks", "site_taus", "start_stage", "unlisted"]
