# SPDX-License-Identifier: Apache-2.0
"""surface_rinsing: ware (their revolved material polygons), racks, sponge, fixtures (layout parts) and the articulated
faucet blades, swing spout, spray valve and hose-reel valve.

Wares come from ``ware.build`` (``Ware.solid`` is the (r, z) polygon of the material, revolved here: watertight, hollow
objects keep their walls). Fixtures are the ``Part`` boxes/cylinders/capsules their ``layout`` installs as colliders.
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import numpy as np
import trimesh
from _common import Exporter, box, capsule, cited, cylinder, dim, line_of, revolve, sphere, union, vis, where

from classroom_arrangement import scene as CA_SCENE
from surface_rinsing import layout as L
from surface_rinsing import params as P
from surface_rinsing import render as R
from surface_rinsing import ware as W

ENV = "surface_rinsing"
LABEL = {"search": "search", "estimate": "estimate", "datasheet": "datasheet", "literature": "literature",
         "authored": "authored", "derived": "derived", "regulation": "standard", "literature-read": "literature-read"}
SCENE_FILE = Path(inspect.getsourcefile(L)).parent / "scene.py"
INCH = P.INCH
SPECIAL_MU = 0.35


# ------------------------------------------------------------------------------------------------ citations
def rel(path: Path, n: int) -> str:
    return f"{ENV}/{Path(path).name}:{n}"


def scene_line(pattern: str, start: int = 0) -> str:
    """``surface_rinsing/scene.py:<line>`` of the first line (after ``start``) that contains ``pattern``."""
    for n, line in enumerate(SCENE_FILE.read_text().splitlines(), 1):
        if n > start and pattern in line:
            return rel(SCENE_FILE, n)
    raise KeyError(pattern)


def scene_line_no(cite: str) -> int:
    return int(cite.rsplit(":", 1)[1])


def fline(func, pattern: str) -> str:
    """``<sim>/<file>.py:<line>`` of the first line inside ``func``'s source that contains ``pattern``."""
    lines, first = inspect.getsourcelines(func)
    for k, line in enumerate(lines):
        if pattern in line:
            return rel(Path(inspect.getsourcefile(func)), first + k)
    raise KeyError(pattern)


def ledger_text(name: str, keyword: str) -> str:
    """The ';'-separated segment of a ledger entry's basis text that mentions ``keyword`` (their provenance)."""
    basis = next(e["basis"] for e in P.LEDGER if e["name"] == name)
    for seg in basis.split(";"):
        if keyword.lower() in seg.lower():
            return " ".join(seg.split())[:300]
    return " ".join(basis.split())[:300]


def items_cite(kind: str, field: str, keyword: str, basis: str) -> dict:
    value = P.ITEMS[kind][field]
    return cited(value, basis, line_of(P, "ITEMS = _p"), ledger_text("ITEMS", keyword))


def density(material: str, key: str) -> dict:
    return cited(P.DENSITY[key], "literature", line_of(P, "DENSITY = _p"), ledger_text("DENSITY", "soda-lime"))


def friction_of(kind: str, key: str) -> dict:
    return cited(P.FRICTION[key], "estimate", line_of(P, "FRICTION = _p"), ledger_text("FRICTION", "dry static"))


def part_mu(value: float = SPECIAL_MU) -> dict:
    return cited(value, "authored", scene_line("p.mu = 0.35"), "fixtures: mu 0.35 on steel and chrome, 0.5 otherwise")


def default_part_mu() -> dict:
    return cited(0.3, "authored", where(CA_SCENE.Scene.add_part), "a Part without its own mu uses 0.3")


# ------------------------------------------------------------------------------------------------ geometry
def part_mesh(p) -> trimesh.Trimesh:
    """A layout ``Part`` (box: half extents; cylinder/capsule: (radius, half height); sphere: radius) as a mesh."""
    if p.kind == "box":
        return box(p.size, p.pos, p.quat)
    if p.kind == "cylinder":
        return cylinder(p.size[0], p.size[1], p.pos, p.quat)
    if p.kind == "capsule":
        return capsule(p.size[0], p.size[1], p.pos, p.quat)
    raise ValueError(f"part kind {p.kind}")


def parts_mesh(parts) -> trimesh.Trimesh:
    return union([part_mesh(p) for p in parts])


def recentre(m: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[float]]:
    """Move a world-placed fixture so its footprint centre is the origin (z kept: the floor is z = 0)."""
    c = 0.5 * (m.bounds[0] + m.bounds[1])
    off = np.array([c[0], c[1], 0.0])
    return m.copy().apply_translation(-off), [round(float(v), 4) for v in off]


def joint(jtype: str, parent: str, axis, rng, cite: str, extra: dict | None = None) -> dict:
    return {"type": jtype, "parent": parent, "axis": axis, "range": rng, "cite": cite, **(extra or {})}


# ------------------------------------------------------------------------------------------------ render looks
def srgb(rgb) -> list[float]:
    """Their material tables hold linear diffuse values (``S.color`` takes linear); parts carry sRGB."""
    c = np.clip(np.asarray(rgb, float), 0.0, 1.0)
    return [float(v) for v in np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)]


def paint(geom: trimesh.Trimesh, material: str, what: str = "") -> dict:
    """A render part in their ``render.COLORS[material]`` (unknown names draw as ``steel``, as their ``_part`` does)."""
    if material == "glass":  # RinsingDielectric absorption (1.5, 1.0, 1.8) /m; their wall is not stated, 3 mm assumed
        tint = srgb(np.exp(-np.array([1.5, 1.0, 1.8]) * 0.003))
        return vis(geom, tint, line_of(R, 'mats["glass"] = _dielectric'), 0.0, 0.0, name="glass (3 mm assumed, linear->sRGB)")
    name = material if material in R.COLORS else "steel"
    rgb, rough, metal = R.COLORS[name]
    return vis(geom, srgb(rgb), line_of(R, f"{name}=(("), rough, metal, name=f"{what or name} (linear->sRGB)")


def looks_of(parts, offset=(0.0, 0.0, 0.0)) -> list[dict]:
    """One render part per material of a list of layout ``Part``s (meshes concatenated, shifted by ``-offset``)."""
    groups: dict[str, list] = {}
    for p in parts:
        groups.setdefault(p.material if p.material in R.COLORS else "steel", []).append(part_mesh(p))
    return [paint(trimesh.util.concatenate(ms).apply_translation(-np.asarray(offset, float)), m) for m, ms in groups.items()]


def ware_looks(kind: str, w, geom: trimesh.Trimesh) -> list[dict]:
    """The drawn solid in the ware's material; the colander's handles and the pan's handle in their own part materials."""
    if w.solid is None:
        return [paint(geom, "tomato" if kind == "tomato" else w.material)]
    out = [paint(revolve(w.solid), w.material, "body")]
    return out + (looks_of(w.prims) if kind in ("colander", "pan") else [])


# ------------------------------------------------------------------------------------------------ ware
def _ware_extra(w) -> dict:
    return {"cavity_capacity_m3": w.cavity["capacity"]} if getattr(w, "cavity", None) else {}


def _add_ware(x: Exporter, kind: str, key: str, name: str, geom, mass: dict, mat: dict, mu: dict, partner: str, dims,
              category: str = "container", **more) -> None:
    w = W.build(kind)
    x.add(key, name, category, geom, visual=ware_looks(kind, w, geom), material=mat, mass=mass, friction=mu,
          friction_partner=partner, dims=dims, source_ref=where(W.BUILDERS[kind]), notes=" ".join((w.notes or "").split())[:300] + " Revolved material polygon "
          "(Ware.solid) as drawn, not their collision hulls.", extra=_ware_extra(w), **more)


def _glass(x: Exporter, kind: str, name: str, basis: str, kw: str) -> None:
    d = P.ITEMS[kind]
    cite = line_of(P, "ITEMS = _p")
    w = W.build(kind)
    _add_ware(x, kind, kind, name, revolve(w.solid), items_cite(kind, "mass", kw, basis),
              {"name": "soda-lime glass", "density": density("glass", "glass")}, friction_of(kind, "glass_steel"),
              "stainless steel sink and drainboard",
              [dim("height", d["H"], "z", cite), dim("rim diameter", 2 * d["r_top"], "diameter_xy", cite)])


def _plate(x: Exporter) -> None:
    d, w = P.ITEMS["plate"], W.build("plate")
    cite = line_of(P, "ITEMS = _p")
    _add_ware(x, "plate", "plate", "China plate 9 in", revolve(w.solid), items_cite("plate", "mass", "china plate", "search"),
              {"name": "vitrified china", "density": density("glaze", "glaze")}, friction_of("plate", "china_steel"),
              "stainless steel", [dim("height", d["H"], "z", cite), dim("rim diameter", 2 * d["R"], "diameter_xy", cite)])


def _bowl(x: Exporter) -> None:
    d, w = P.ITEMS["bowl"], W.build("bowl")
    cite = line_of(P, "ITEMS = _p")
    _add_ware(x, "bowl", "bowl", "China soup bowl 6 in", revolve(w.solid), items_cite("bowl", "mass", "6 in bowl", "estimate"),
              {"name": "vitrified china", "density": density("glaze", "glaze")}, friction_of("bowl", "china_steel"),
              "stainless steel", [dim("height", d["H"], "z", cite), dim("rim diameter", 2 * d["R"], "diameter_xy", cite)])


def _steel_bowl(x: Exporter) -> None:
    w = W.build("steel_bowl")
    doc = " ".join(W.steel_bowl.__doc__.split())
    mass = cited(0.40, "authored", fline(W.steel_bowl, "0.40"), doc)
    mu = cited(0.30, "authored", fline(W.steel_bowl, "friction=0.30"), "authored 300 mm stainless bowl")
    _add_ware(x, "steel_bowl", "steel_bowl", "Stainless mixing bowl 300 mm", revolve(w.solid), mass,
              {"name": "stainless steel", "density": density("steel", "steel")}, mu, "stainless steel",
              [dim("rim diameter", 0.300, "diameter_xy", fline(W.steel_bowl, "R, H, rb, wall")),
               dim("height", 0.110, "z", fline(W.steel_bowl, "R, H, rb, wall"))])


def _colander(x: Exporter) -> None:
    d, w = P.ITEMS["colander"], W.build("colander")
    cite = line_of(P, "ITEMS = _p")
    geom = union([revolve(w.solid), *(part_mesh(p) for p in w.prims)])
    mu = cited(0.30, "authored", fline(W.colander, "friction=0.30"), "5 qt perforated colander (estimate)")
    _add_ware(x, "colander", "colander", "Perforated colander 5 qt", geom, items_cite("colander", "mass", "colander", "estimate"),
              {"name": "stainless steel", "density": density("steel", "steel")}, mu, "stainless steel",
              [dim("height", d["H"], "z", cite), dim("rim diameter across the handles' axis", 2 * d["R"], "y", cite)])


def _tomato(x: Exporter) -> None:
    d, w = P.ITEMS["tomato"], W.build("tomato")
    cite = line_of(P, "ITEMS = _p")
    geom = sphere(1.0).apply_scale((d["r"], d["r"], d["c"]))
    _add_ware(x, "tomato", "tomato", "Cherry tomato", geom, items_cite("tomato", "mass", "cherry tomato", "estimate"),
              {"name": "tomato flesh", "density": cited(P.DENSITY["tomato"], "literature", line_of(P, "DENSITY = _p"),
                                                         "ripe tomato flesh")},
              friction_of("tomato", "tomato_steel"), "stainless steel",
              [dim("height", 2 * d["c"], "z", cite), dim("diameter", 2 * d["r"], "diameter_xy", cite)], category="food")


def _can(x: Exporter) -> None:
    d, w = P.ITEMS["can"], W.build("can")
    cite = line_of(P, "ITEMS = _p")
    _add_ware(x, "can", "waste_bin", "Garbage can 32 gal (Rubbermaid BRUTE 2632)", revolve(w.solid),
              items_cite("can", "mass", "BRUTE", "search"),
              {"name": "polyethylene (LLDPE)", "density": cited(P.DENSITY["lldpe"], "literature", line_of(P, "DENSITY = _p"),
                                                                 "polyethylenes")},
              friction_of("can", "lldpe_tile"), "quarry tile floor",
              [dim("height", d["H"], "z", cite), dim("top diameter", 2 * d["R"], "diameter_xy", cite)])


def _pan(x: Exporter) -> None:
    d, w = P.ITEMS["pan"], W.build("pan")
    cite = line_of(P, "ITEMS = _p")
    geom = union([revolve(w.solid), *(part_mesh(p) for p in w.prims)])
    mu = cited(0.30, "authored", fline(W.pan, "friction=0.30"), "10 in aluminium fry pan")
    _add_ware(x, "pan", "frying_pan", "Aluminium fry pan 10 in (Vollrath Wear-Ever)", geom,
              items_cite("pan", "mass", "fry pan", "search"),
              {"name": "aluminium", "density": cited(P.DENSITY["aluminum"], "literature", line_of(P, "DENSITY = _p"),
                                                      "aluminium")}, mu, "stainless steel",
              [dim("rim diameter across the handle's axis", 2 * d["R"], "y", cite)])


def _cutting_board(x: Exporter) -> None:
    w = W.build("cutting_board")
    lw, ww, tt = 0.610, 0.457, 0.0127
    cite = fline(W.cutting_board, "L_, W_, T_")
    slab = box((lw / 2, ww / 2, tt / 2), (0, 0, tt / 2))
    x.add("cutting_board", "HDPE cutting board 18 x 24 x 1/2 in", "tool", slab, visual=[paint(slab, "hdpe")],
          material={"name": "HDPE", "density": cited(950.0, "literature", fline(W.cutting_board, "950.0"),
                                                      "HDPE 18 x 24 x 1/2 in (derived mass 3.4 kg)")},
          friction=cited(0.4, "authored", fline(W.cutting_board, "friction=0.4"), "cutting board on the steel table"),
          friction_partner="stainless steel work table",
          dims=[dim("length", lw, "x", cite), dim("width", ww, "y", cite), dim("thickness", tt, "z", cite)],
          source_ref=where(W.cutting_board), notes="Mass is their density 950 kg/m^3 x volume (3.4 kg), so it is left to the density.")


def _rack(x: Exporter, kind: str, key: str, name: str, mass: float, quote: str, size_pat: str) -> None:
    w = W.build(kind)
    builder = W.BUILDERS[kind]
    cite = fline(builder, f'Ware("{kind}"')
    x.add(key, name, "fixture", parts_mesh(w.prims), visual=looks_of(w.prims), motion="dynamic",
          material={"name": "polypropylene rack"},
          mass=cited(mass, "authored", cite, quote), friction=friction_of(kind, "rack_steel"),
          friction_partner="stainless steel drainboard",
          dims=[dim("side", 0.5, "x", fline(builder, size_pat)), dim("side", 0.5, "y", fline(builder, size_pat))],
          source_ref=where(builder), notes=" ".join(w.notes.split()) + " Their parts (bars, posts, walls, dividers) unioned.")


def _sponge(x: Exporter) -> None:
    d, w = P.ITEMS["sponge"], W.build("sponge")
    L_, W_, sc = d["L"], d["W"], d["scour"]
    cite = line_of(P, "ITEMS = _p")
    mu = cited(0.5, "authored", fline(W.sponge, 'friction=0.5'), "sponge friction 0.5 (scour pad 0.8)")
    base_h = sc + 0.006
    base_m = sum(p.mass for p in w.prims)
    quote = ledger_text("ITEMS", "Scotch-Brite")
    scour = [p for p in w.prims if p.name == "scour"][0]
    cell = [p for p in w.prims if p.name == "cell_lo"][0]
    soft = [paint(part_mesh(scour), "scour_green", "scour pad"), paint(part_mesh(cell), "sponge_yellow", "lower cellulose")]
    x.add("sponge__base", "Scrub sponge base: scour pad and lower cellulose (Scotch-Brite heavy duty)", "articulated",
          box((L_ / 2, W_ / 2, base_h / 2), (0, 0, base_h / 2)), visual=soft, material={"name": "cellulose foam with nylon scour pad"},
          mass=cited(base_m, "derived", fline(W.sponge, 'Part("scour"'), "scour 4 g + lower cellulose 2 g; " + quote),
          friction=mu, friction_partner="pan and ware", dims=[dim("length", L_, "x", cite), dim("width", W_, "y", cite)],
          source_ref=where(W.sponge), notes="Full 111 x 66 mm drawn block (their firm collision core is 6 mm smaller all round). "
          "Young's modulus 50 kPa is in their ledger but no Poisson ratio, so it is exported rigid.")
    layer = box((L_ / 2, W_ / 2, 0.003))
    x.add("sponge__top", "Scrub sponge upper cellulose layer", "articulated", layer, visual=[paint(layer, "sponge_yellow")],
          material={"name": "cellulose foam"}, mass=cited(0.002, "derived", fline(W.sponge, 'Part("cell_hi"'), "upper cellulose 2 g"),
          friction=mu, friction_partner="hand pad", source_ref=where(W.sponge),
          notes="Second body on the compression slide; its frame is the centre of the layer.",
          extra={"joint": joint("prismatic", "sponge__base", [0, 0, -1], [0.0, P.SPONGE["q_stop"]],
                                line_of(P, "q_stop"), {"anchor_in_parent_m": [0, 0, w.top_z], "damping_Ns_m": P.SPONGE["damping"]})})


# ------------------------------------------------------------------------------------------------ fixtures
def _fixture(x: Exporter, key: str, name: str, parts, dims: list, cite: str, notes: str, category: str = "fixture") -> None:
    geom, off = recentre(parts_mesh(parts))
    x.add(key, name, category, geom, visual=looks_of(parts, (off[0], off[1], 0.0)), motion="static", material={"name": "stainless steel"}, friction=part_mu(),
          friction_partner="ware and hands", dims=dims, source_ref=cite, notes=notes, extra={"world_offset_m": off})


def _by_name(parts, include=None, exclude=()) -> list:
    return [p for p in parts if (include is None or include(p.name)) and p.name not in exclude]


def _sinks(x: Exporter) -> None:
    ps = L.pot_sink_parts()
    keep = _by_name(ps, exclude=("wall_shelf", "sink_sign") + tuple(p.name for p in ps if p.name.startswith("dispenser")))
    _fixture(x, "pot_sink", "Three-compartment pot sink with drainboards (Advance Tabco FS-3-1824-24RL)", keep,
             [dim("depth", 30 * INCH, "y", line_of(L, "SINK_Y0 = Y"))], where(L.pot_sink_parts),
             "Bowls, partitions, aprons, drainboards, legs and the pre-rinse riser/arm; the handles and spout are separate parts.")
    dt = L.dish_table_parts()
    _fixture(x, "dish_table", "Soiled dish table with 20 x 20 x 5 in pre-rinse sink (Advance Tabco DTS-S70-72R)",
             _by_name(dt, exclude=("dish_machine", "clean_table")),
             [dim("length", 72 * INCH, "y", line_of(L, "TABLE_Y0, TABLE_Y1")), dim("depth", P.DISH_TABLE["depth"], "x",
                                                                                   line_of(L, "TABLE_X1 ="))],
             where(L.dish_table_parts), "Table top, sink pool walls, rails, legs and the pre-rinse riser; the dish machine is left out.")
    pp = L.prep_parts()
    _fixture(x, "prep_sink", "Prep sink with gooseneck faucet", _by_name(pp, lambda n: n.startswith("v")),
             [dim("depth", 30 * INCH, "x", line_of(L, "VSINK_X0 ="))], where(L.prep_parts),
             "Single bowl, drainboard, aprons, legs and the gooseneck spout (a static collider in their scene).")
    _fixture(x, "work_table", "Steel work table", _by_name(pp, lambda n: n.startswith(("work", "wleg"))),
             [dim("top height", L.VS["table_h"], "z", line_of(P, "PREP_SINK = _p")),
              dim("length", 48 * INCH, "y", line_of(L, "WORK = dict"))], where(L.prep_parts), "Top, lower shelf and four legs.")


def _bay(x: Exporter) -> None:
    bp = L.bay_parts()
    _fixture(x, "dunnage_shelf", "Dunnage shelf (clean storage, three wire levels)",
             _by_name(bp, lambda n: n.startswith("shelf")),
             [dim("width", L.SHELF["x1"] - L.SHELF["x0"], "x", line_of(L, "SHELF = dict"))], where(L.bay_parts),
             "Three wire shelves on four posts.")
    _fixture(x, "hose_reel_mount", "Open hose reel drum and bracket (T&S B-7132-01 class)",
             _by_name(bp, lambda n: n.startswith("reel")), [dim("drum diameter", 0.40, "x", line_of(L, "reel_drum"))],
             where(L.bay_parts), "Static drum and wall bracket; the hose and valve are separate parts.")
    _fixture(x, "valve_holster", "Wall holster for the hose valve", _by_name(bp, lambda n: n.startswith("holster")),
             [dim("depth", L.HOLSTER["depth"], "z", line_of(L, "HOLSTER = dict"))], where(L.bay_parts),
             "Octagonal sleeve with a ring floor and its bracket.")
    x.fail("bay_floor", "the sloped floor and 6 in drain are the ground plane and a drawn strainer (collide=False), not a built "
           "solid; slope and drain size are in params.CAN_BAY")


# ------------------------------------------------------------------------------------------------ articulated fixtures
def _blade(x: Exporter) -> None:
    F = P.FAUCET
    hub = cylinder(0.014, 0.018, (0, 0.018, 0), (0.7071067811865476, 0, 0, 0.7071067811865476))
    blade = box((0.007, 0.004, F["blade_len"] / 2), (0, 0.030, F["blade_len"] / 2))
    fl = line_of(P, "FAUCET = _p")
    q = ledger_text("FAUCET", "wrist-blade")
    x.add("pot_sink__faucet_blade", "Wrist-blade faucet handle (quarter turn)", "articulated", union([hub, blade]),
          visual=[paint(trimesh.util.concatenate([hub, blade]), "chrome")],
          material={"name": "chrome-plated brass"},
          mass=cited(0.04 + F["blade_mass"], "estimate", scene_line('Part("hub"'), "hub 0.04 kg + blade FAUCET.blade_mass; " + q),
          friction=default_part_mu(), friction_partner="hand pad",
          dims=[dim("blade length", F["blade_len"], "z", fl)], source_ref=where(L.pot_sink_parts),
          notes="Frame: origin at the stem pivot, y along the stem, z along the blade (closed = up). The hot, cold and "
                "faucet-valve handles are the same part (instances 3).",
          extra={"joint": joint("revolute", "pot_sink", [0, 1, 0], [0.0, math.radians(F["turn_deg"])], fl,
                                {"friction_Nm": F["friction_nm"], "instances": 3, "sign": "+-1 per handle"})})


def _spout(x: Exporter) -> None:
    length, yaw0 = 0.30, math.radians(40.0)
    q = (0.7071067811865476, 0, 0, 0.7071067811865476)
    swivel = cylinder(0.016, 0.025)
    tube = capsule(0.0095, (length - 0.03) / 2 - 0.0095, (0, -0.03 - (length - 0.03) / 2, 0), q)
    nose = capsule(0.011, 0.02, (0, -length, -0.025))
    cite = scene_line('Part("swivel"')
    x.add("pot_sink__swing_spout", "Swing faucet spout 12 in (add-on)", "articulated", union([swivel, tube, nose]),
          visual=[paint(trimesh.util.concatenate([swivel, tube, nose]), "chrome")],
          material={"name": "chrome-plated brass"},
          mass=cited(0.15 + 0.25 + 0.05, "estimate", cite, "swivel collar 0.15 + tube 0.25 + nose 0.05 kg"),
          friction=default_part_mu(), friction_partner="hand pad", dims=[dim("reach", length, "y", scene_line("def swing_spout"))],
          source_ref=scene_line("def swing_spout"), notes="Tube on a vertical swivel with O-ring friction; frame at the swivel axis.",
          extra={"joint": joint("revolute", "pot_sink", [0, 0, 1], [-math.radians(130) - yaw0, math.radians(130) - yaw0],
                                scene_line("limit_lower=-math.radians(130.0)"), {"friction_Nm": 0.15, "start_yaw_rad": yaw0})})


def _valve_parts(x: Exporter, prefix: str, def_name: str, name: str, dims_r: tuple, lever_half: tuple, masses: tuple,
                 spec_cite: str, quote: str) -> None:
    """Valve body (four cylinders) and its squeeze lever as two parts; ``dims_r`` = (body r, head r, bumper r, inlet r)."""
    br, hr, kr, ir = dims_r
    body = union([cylinder(br, 0.06, (0, 0, -0.015)), cylinder(hr, 0.012, (0, 0, -0.083)),
                  cylinder(kr, 0.006, (0, 0, -0.098)), cylinder(ir, 0.012, (0, 0, 0.050))])
    first = scene_line('Part("body"', scene_line_no(scene_line(f"def {def_name}")))
    lever_m = P.PRSV["lever_mass"]
    seen = [paint(trimesh.util.concatenate([cylinder(br, 0.06, (0, 0, -0.015)), cylinder(hr, 0.012, (0, 0, -0.083)),
                                            cylinder(ir, 0.012, (0, 0, 0.050))]), "chrome", "body, head, inlet"),
            paint(cylinder(kr, 0.006, (0, 0, -0.098)), "rubber_grey", "bumper")]
    x.add(f"{prefix}__valve", f"{name}: valve body", "articulated", body, visual=seen, material={"name": "chrome-plated brass, rubber bumper"},
          mass=cited(sum(masses), "estimate", first, quote), friction=default_part_mu(),
          friction_partner="ware and hands", dims=[dim("body length incl. bumper", 0.162, "z", first)],
          source_ref=first, notes="Four stacked cylinders (body, head, rubber bumper, inlet); hose inlet up, spray face down.",
          extra={"flow_spec_ref": line_of(P, "PRSV = _p")})
    arm = box(lever_half, (0, 0, -0.045))
    x.add(f"{prefix}__lever", f"{name}: squeeze lever", "articulated", arm, visual=[paint(arm, "chrome")],
          material={"name": "chrome-plated steel"}, mass=cited(lever_m, "estimate", line_of(P, "PRSV = _p"), "lever_mass 20 g"),
          friction=default_part_mu(), friction_partner="finger pad", source_ref=first, notes="Hinged at (0.03, 0, 0.03) of the valve.",
          extra={"joint": joint("revolute", f"{prefix}__valve", [0, 1, 0], [0.0, P.PRSV["travel"] / P.PRSV["lever_arm"]],
                                spec_cite)})


def _hose_link(x: Exporter, key: str, name: str, radius: float, link_len: float, mass: dict, cite: str, rng: float) -> None:
    link = capsule(radius, link_len / 2 - radius)
    hose = paint(link, "hose_green" if key.startswith("reel") else "hose_steel", "hose")
    x.add(key, name, "articulated", link, visual=[hose], material={"name": "stainless braided hose"},
          mass=mass, friction=default_part_mu(), friction_partner="floor and ware", dims=[dim("link length", link_len, "z", cite)],
          source_ref=cite, notes="One capsule link of the hose chain on a two-axis bending joint (d6, +-%.1f rad); the chain "
          "repeats it." % rng, extra={"joint": joint("bend_2axis", "previous link", [[1, 0, 0], [0, 1, 0]], [-rng, rng], cite,
                                                     {"bend_stiffness_EI_Nm2": 0.05, "cite_EI": scene_line("EI 0.05 N m2")})})


def _spray_valve(x: Exporter) -> None:
    quote = ledger_text("PRSV", "body 0.55")
    _valve_parts(x, "spray_valve", "spray_valve", "Pre-rinse spray valve (T&S B-0107 class)",
                 (0.016, P.PRSV["face_r"], P.PRSV["face_r"] + 0.003, 0.010), (0.004, 0.010, 0.045), (0.30, 0.18, 0.04, 0.03),
                 scene_line("limit_upper=spec[\"travel\"]/spec[\"lever_arm\"]"), quote)
    cite = scene_line("0.32*link_len")
    link_len = 0.07
    _hose_link(x, "spray_valve__hose_link", "Spray valve balancer hose link", 0.008, link_len,
               cited(0.32 * link_len, "estimate", cite, "link mass 0.32 kg/m x link length"), cite, 1.2)
    clamp = cylinder(0.012, 0.015)
    x.add("spray_valve__balancer", "Spring balancer clamp", "articulated", clamp, visual=[paint(clamp, "chrome")], material={"name": "chrome"},
          mass=cited(0.05, "estimate", scene_line('Part("clamp"'), "balancer clamp 50 g"), friction=default_part_mu(),
          friction_partner="none", source_ref=scene_line('Part("clamp"'), notes="Clamp on a 0.6 m vertical slide with a spring.",
          extra={"joint": joint("prismatic", "world", [0, 0, -1], [0.0, 0.60], scene_line("limit_upper=0.60"),
                                {"damping_Ns_m": 8.0})})


def _reel_valve(x: Exporter) -> None:
    quote = ledger_text("REEL", "35 ft")
    _valve_parts(x, "reel_valve", "reel_hose", "Hose-reel squeeze valve (T&S EB-0107 class)", (0.018, 0.030, 0.033, 0.011),
                 (0.004, 0.011, 0.045), (0.32, 0.22, 0.05, 0.04), scene_line("limit_upper=P.PRSV[\"travel\"]/P.PRSV[\"lever_arm\"]"), quote)
    spec = P.REEL
    cite = scene_line('m_link = spec["hose_lin_mass"]')
    link_len = 0.125
    _hose_link(x, "reel_valve__hose_link", "Reel hose link (3/8 in hose)", spec["hose_r"], link_len,
               cited(spec["hose_lin_mass"] * link_len, "estimate", line_of(P, "REEL = _p"), "0.22 kg/m with water (estimate)"),
               cite, 2.6)


def export(x: Exporter) -> None:
    steps = [lambda e: _glass(e, "rocks_glass", "Rocks glass 12 oz (Libbey 15587)", "search", "rocks glass"),
             lambda e: _glass(e, "mixing_glass", "Mixing glass 16 oz", "estimate", "mixing glass"),
             _plate, _bowl, _steel_bowl, _colander, _tomato, _can, _pan, _cutting_board, _sponge,
             lambda e: _rack(e, "glass_rack", "glass_rack", "Glass rack 25 compartments (Cambro 25S class)", 1.6,
                             W.glass_rack.__doc__, "S, Hh, t"),
             lambda e: _rack(e, "peg_rack", "peg_rack", "Peg rack for plates", 1.4, W.peg_rack.__doc__, "S = 0.500"),
             _sinks, _bay, _blade, _spout, _spray_valve, _reel_valve]
    for i, step in enumerate(steps):
        try:
            step(x)
        except Exception as exc:  # noqa: BLE001 - one object must not stop the others
            x.fail(f"step_{i}_{getattr(step, '__name__', 'ware')}", f"{type(exc).__name__}: {exc}")
