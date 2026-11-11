# SPDX-License-Identifier: Apache-2.0
"""painting: roller (frame + cover), tray, four brushes, pail, can, palette, aerosol can (+ button), stencil, spray
box, crate side, board, folding table and the cast-iron radiator.

Shapes are their ``tools.py`` Part lists (boxes, capsules, cylinders) unioned into one body; the radiator comes from
``layout.static_boxes``. Parts they only draw (``collide=False``) are kept because they carry mass.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from _common import Exporter, box, capsule, cited, cylinder, dim, line_of, union, vis, where

from classroom_arrangement.furniture import Part
from painting import layout as LY
from painting import params as P
from painting import render as RN
from painting import scene as SC
from painting import tools as T

BASIS = {"retailer": "search", "datasheet": "datasheet", "published": "literature", "search": "search",
         "estimate": "estimate", "authored": "authored", "derived": "derived"}


GROW = 1e-4     # m: parts are grown 0.1 mm so touching parts overlap (edge contacts leave pinched, non-manifold unions)


def _part_mesh(p, g=GROW):
    if p.kind == "box":
        return box(tuple(h + g for h in p.size), p.pos, p.quat)
    if p.kind == "capsule":
        return capsule(p.size[0] + g, p.size[1], p.pos, p.quat)
    return cylinder(p.size[0] + g, p.size[1] + g, p.pos, p.quat)


def _clean(m):
    """Snap to 1 um, drop sliver faces and zero-volume shells (the OBJ round trip merges near-equal vertices)."""
    m.vertices = np.round(m.vertices, 6)
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    keep = [c for c in m.split(only_watertight=False) if abs(c.volume) > 1e-12]
    return trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]


def _body_mesh(parts):
    solid = [p for p in parts if p.kind in ("box", "capsule", "cylinder")]
    m = union([_part_mesh(p, GROW if len(solid) > 1 else 0.0) for p in solid])
    return _clean(m) if len(solid) > 1 else m


def _srgb(lin) -> list[float]:
    lin = np.clip(np.asarray(lin, float), 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).tolist()


def _look(geom, mat: str, label: str = "") -> dict:
    """Render part from their ``COLORS`` table (name -> linear diffuse, roughness, metallic; linear->sRGB here)."""
    rgb, rough, metal = RN.COLORS[mat]
    return vis(geom, _srgb(rgb), line_of(RN, f"{mat}=(("), rough, metal, name=f"{label or mat} (linear->sRGB)")


def _visual(parts, over: dict | None = None) -> list[dict]:
    """One part per render material of a Part list (``Part.material`` is a ``COLORS`` key); ``over`` maps a material
    to ``(sRGB 0-255, roughness, metallic, cite, label)`` where their renderer draws that part differently."""
    out = []
    for mat in dict.fromkeys(p.material for p in parts if p.kind in ("box", "capsule", "cylinder")):
        geom = trimesh.util.concatenate([_part_mesh(p, 0.0) for p in parts if p.material == mat and p.kind in
                                         ("box", "capsule", "cylinder")])
        if over and mat in over:
            col, rough, metal, cite, label = over[mat]
            out.append(vis(geom, col, cite, rough, metal, name=label))
        else:
            out.append(_look(geom, mat))
    return out


def _mu(kind: str, quote: str = "") -> dict:
    return cited(SC.MU[kind], "authored", line_of(SC, "MU = dict"),
                 quote or "contact friction their scene gives this tool's parts")


def _est(value: float, cite: str, quote: str, basis: str = "estimate") -> dict:
    return cited(value, basis, cite, quote)


def _add(x: Exporter, key, name, cat, parts, mu_kind, partner, dims, **kw) -> None:
    if mu_kind in SC.MU:
        kw["friction"] = _mu(mu_kind)
    kw.setdefault("visual", _visual(parts, kw.pop("over", None)))
    x.add(key, name, cat, _body_mesh(parts), friction_partner=partner, dims=dims, **kw)


def _roller(x: Exporter) -> None:
    R = P.ROLLER
    ref = line_of(P, "ROLLER = _p(")
    grip = Part("grip", "cylinder", (R["grip_r"], R["grip_len"] / 2), (0., 0., -R["wire_len"] - R["grip_len"] / 2),
                material="grip")
    frame = [q for q in T.roller_frame() if not q.name.startswith("grip_")] + [grip]
    _add(x, "roller__frame", "Paint roller frame (Sherlock R017-9 class)", "articulated", frame, "roller",
         "the hand and the wall",
         [dim("handle + wire length", R["wire_len"] + R["grip_len"], "z", ref)],
         material={"name": "steel cage with plastic grip"},
         mass=_est(R["frame_mass"], ref, "Sherlock R017-9 frame 0.34 kg [Wooster frame listings]", "search"),
         source_ref=where(T.roller_frame),
         notes="Wire, grip (their six thin boxes drawn as one cylinder), collar and axle; the axle stub is drawn-only in their sim but carries "
               "mass, so it is kept.")
    r_out = R["core_od"] / 2 + R["nap"]
    cover = cylinder(r_out, R["cover_len"] / 2).apply_transform(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.]]))
    x.add("roller__cover", "Roller cover 9 in x 3/8 in nap", "articulated", cover,
          visual=[_look(cover, "nap", "nap (dry base colour; their render streams a dynamic wet texture)")],
          material={"name": "microfibre nap on a 1.5 in core"},
          mass=_est(R["cover_mass"], ref, "cover 60 g dry (estimate)"),
          friction=cited(R["mu_dry"], "estimate", ref, "nap-wall friction 0.6 dry / 0.4 wet (estimates)"),
          friction_partner="painted wall",
          dims=[dim("cover length", R["cover_len"], "y", ref), dim("outer diameter", 2 * r_out, "x", ref)],
          source_ref=where(T.roller_cover),
          notes="Full nap cylinder (their collider is the densified core at 85 % nap compression); axis along y, "
                "centred on the cover.",
          extra={"joint": {"type": "revolute", "parent": "roller__frame", "axis": [0, 1, 0], "range": None,
                           "friction_nm": R["bearing"], "cite": where(SC.PaintingScene.roller)}})


def _tray(x: Exporter) -> None:
    ref = line_of(P, "TRAY = _p(")
    L, W, H = P.TRAY["size"]
    _add(x, "paint_tray", "Roller paint tray (Wooster R404 class)", "container", T.tray(), "tray", "floor and table",
         [dim("length", L, "x", ref), dim("width", W, "y", ref), dim("height", H, "z", ref)],
         material={"name": "polypropylene"},
         mass=_est(P.TRAY["mass"], ref, "0.25 kg PP mass are estimates"), source_ref=where(T.tray),
         notes="Well floor, ramp, lip, walls, feet and legs as boxes; their ramp ribs are not modelled.")


def _brushes(x: Exporter) -> None:
    names = {"sash": "Angled sash brush (Purdy XL Glide 2.5 in class)", "stencil": "Stencil brush 0.5 in (hog)",
             "round8": "Round brush no. 8", "longreach": "Dog-leg long-reach radiator brush (ProDec PDLB50 class)"}
    ref = line_of(P, "BRUSHES = _p(")
    for kind, name in names.items():
        B = P.BRUSHES[kind]
        parts = T.brush(kind)
        tall = max(0.0, B["length"])
        over = {"alder": (_srgb((.03, .03, .035)), .25, 0., line_of(RN, 'S.color("lacquer_black"'),
                          "black lacquer handle (linear->sRGB)")} if kind == "round8" else None
        _add(x, f"brush_{kind}", name, "tool", parts, kind, "paint film and wall",
             [dim("bristle width", B["width"], "x", ref)] if not B.get("round") else [], over=over, material={"name": "wood handle, steel ferrule, synthetic "
                                                                       "filaments"},
             mass=_est(B["mass"], ref, "brush total mass " + ("0.12 kg [retailer listings]" if kind == "longreach"
                                                              else "(estimate)"),
                       "search" if kind == "longreach" else "estimate"),
             source_ref=where(T.brush),
             notes="Ferrule, handle and the bristle pack as one box (their tufts are a separate model); origin at "
                   f"the ferrule's bristle face. Pack length {tall * 1e3:.1f} mm.")


def _containers(x: Exporter) -> None:
    for key, name, parts, kind, mat, ent, dims in (
            ("paint_pail", "Paint pail 1 qt (Bercom)", T.pail(), "pail", "plastic", "PAIL",
             lambda G: [dim("height", G["height"], "z", line_of(P, "PAIL = _p("))]),
            ("paint_can", "Paint can 1 gal", T.can(), "can", "steel", "CAN",
             lambda G: [dim("height", G["height"], "z", line_of(P, "CAN = _p("))])):
        G = getattr(P, ent)
        _add(x, key, name, "container", parts, kind, "floor and table", dims(G), material={"name": mat},
             mass=_est(G["mass"], line_of(P, f"{ent} = _p("), "total mass; wall staves are boxes (estimate)"),
             source_ref=where(getattr(T, kind)), notes="Hollow: bottom disc and a ring of overlapping staves.")


def _small_tools(x: Exporter) -> None:
    pal = T.palette()
    _add(x, "palette", "Palette plate", "tool", pal, "palette", "table", [dim("length", .25, "x", where(T.palette))],
         material={"name": "ceramic or plastic plate"},
         mass=_est(pal[0].mass, where(T.palette), "palette plate 0.15 kg (authored in tools.py)", "authored"),
         source_ref=where(T.palette), notes="Plain 250 x 170 x 12 mm plate; the six wells are not cut.")
    S = P.STENCIL
    sheet = T.stencil()
    _add(x, "stencil", "Stencil sheet 12 mil PET", "tool", sheet, "stencil", "the panel",
         [dim("length", S["size"][0], "x", line_of(P, "STENCIL = _p(")),
          dim("thickness", S["thick"], "z", line_of(P, "STENCIL = _p("))],
         material={"name": "PET", "density": _est(S["rho"], line_of(P, "STENCIL = _p("),
                                                  "PET density 1,390 kg/m3", "search")},
         source_ref=where(T.stencil), notes="Plain sheet; the motif cut-outs are not modelled. Mass from density.")


def _props(x: Exporter) -> None:
    sb = T.spray_box()
    _add(x, "spray_box", "Spray box (open carton)", "fixture", sb, "spray_box", "floor",
         [dim("width", T.SPRAY_BOX["size"][0], "x", line_of(T, "SPRAY_BOX ="))],
         material={"name": "cardboard"}, source_ref=where(T.spray_box),
         notes="Floor and three walls of a 600 x 450 x 450 mm carton, 4 mm board.")
    _add(x, "crate_side", "Crate side panel (plywood)", "fixture", T.crate_side(), "crate", "floor",
         [dim("length", .45, "x", where(T.crate_side))], material={"name": "plywood"},
         over={"plywood": (P.COLORS["plywood"], .6, 0., line_of(P, "plywood=(214"),
                           "plywood (their chart colour for the plywood substrate; the veneer photo is not exported)")},
         mass=_est(T.crate_side()[0].mass, where(T.crate_side), "1.1 kg in tools.py", "authored"),
         source_ref=where(T.crate_side), notes="450 x 300 x 12 mm panel.")
    _add(x, "work_board", "Work board (MDF)", "fixture", T.board(), "board", "table",
         [dim("length", .6, "x", where(T.board))], material={"name": "MDF"},
         mass=_est(T.board()[0].mass, where(T.board), "2.4 kg in tools.py", "authored"), source_ref=where(T.board),
         notes="600 x 450 x 12 mm board.")
    Tb = P.TABLE
    _add(x, "folding_table", "Folding work table (4 ft)", "furniture", T.table(), "table", "tools and floor",
         [dim("length", Tb["size"][0], "x", line_of(P, "TABLE = _p(")),
          dim("width", Tb["size"][1], "y", line_of(P, "TABLE = _p(")),
          dim("height", Tb["height"], "z", line_of(P, "TABLE = _p("))], material={"name": "laminate top, steel legs"},
         mass=_est(Tb["mass"], line_of(P, "TABLE = _p("), "Lifetime-class 4 ft folding table (estimate)", "authored"),
         source_ref=where(T.table), notes="Top slab and four capsule legs.")


def _aerosol(x: Exporter) -> None:
    A = P.AEROSOL
    ref = line_of(P, "AEROSOL = _p(")
    can_look = {"aerosol": (P.COLORS["gloss_red"], .3, .4, line_of(RN, 'pc = env.spec.get("paint", {}).get("color", "gloss_red")'),
                            "can body in the default paint colour gloss_red (label texture not exported)")}
    _add(x, "spray_can__can", "Aerosol spray paint can, 12 oz (empty body)", "articulated", T.aerosol_can(), "aerosol",
         "the hand", [dim("can diameter", 2 * A["r"], "diameter_xy", ref), dim("height", A["height"] - .026 + .02, "z", ref)],
         over=can_look, material={"name": "tinplate"}, mass=_est(A["can_mass"], ref, "12 oz can 66 x 196 mm (estimate)"),
         source_ref=where(T.aerosol_can),
         notes="Body (six boxes as a round grip), shoulder and valve cup; the 0.34 kg liquid fill is excluded.")
    btn = T.aerosol_button()
    _add(x, "spray_can__button", "Aerosol actuator button", "articulated", btn, None, "the fingertip",
         [dim("diameter", 2 * A["button_r"], "diameter_xy", where(T.aerosol_button))], material={"name": "plastic"},
         mass=_est(btn[0].mass, where(T.aerosol_button), "button mass 3 g in tools.py", "authored"),
         friction=cited(0.6, "authored", line_of(SC, "bshapes"), "button contact friction 0.6 in their scene"),
         source_ref=where(T.aerosol_button),
         notes="Cylinder on a sprung slide along the can axis.",
         extra={"joint": {"type": "prismatic", "parent": "spray_can__can", "axis": [0, 0, -1],
                          "range": [0.0, A["stroke"]], "cite": where(SC.PaintingScene.aerosol)}})


def _radiator(x: Exporter) -> None:
    items = [b for b in LY.static_boxes() if b[0].startswith("rad_") and b[0] != "rad_pipe"]
    parts = [box(h, np.asarray(c) - np.array([LY.WIN["x"], items[0][1][1], 0.])) for _n, c, h, _m in items]
    R = P.RADIATOR
    ref = line_of(P, "RADIATOR = _p(")
    whole = union(parts)
    x.add("radiator", "Cast-iron radiator (4-column, 10 sections)", "fixture", whole, motion="static",
          visual=[_look(whole, "radiator")],
          material={"name": "cast iron"},
          friction=cited(0.45, "authored", line_of(SC, "mu=0.45"), "static box friction in their room"),
          friction_partner="brush and tools",
          dims=[dim("length", R["len"], "x", ref), dim("height", R["height"], "z", ref)],
          source_ref=where(LY.static_boxes), notes="Section boxes, hubs and feet from layout.static_boxes; the supply "
                                                   "pipe is separate.")


def export(x: Exporter) -> None:
    for step in (_roller, _tray, _brushes, _containers, _small_tools, _props, _aerosol, _radiator):
        try:
            step(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(step.__name__, f"{type(exc).__name__}: {exc}")
