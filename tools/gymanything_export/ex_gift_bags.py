# SPDX-License-Identifier: Apache-2.0
"""gift_bags: Euro tote, tote plates, boards, candle carton, set-up gift box, photo frame, wrap, tape, supply carton,
tea cartons, jar, rope handle and the furniture their scene fixes in place.

Shapes come from their constants and builders (``giftbox.tray_boxes``, ``carton.body_parts`` / ``flap_specs`` /
``content_specs``, ``scene.prism_mesh``, ``wrapmap.wrap_mesh``). Their sheets (tote, wrap, tape) are open triangulated
membranes, so each is exported as a thin watertight slab/shell of its own caliper with their shell constants.
"""

from __future__ import annotations

import logging
import math
import re
import tempfile

import numpy as np
import trimesh

from _common import Exporter, box, cited, cylinder, dim, line_of, mesh, union, vis, where

from gift_bags import artwork as AW
from gift_bags import bagnet as B
from gift_bags import carton as CT
from gift_bags import giftbox as GB
from gift_bags import params as P
from gift_bags import render as R
from gift_bags import scene as S
from gift_bags import wrapmap as WM

log = logging.getLogger("gar.export.gift_bags")


def _block(mod, pattern: str, limit: int = 380) -> str:
    """The ``#`` comment lines directly above (and the trailing comment on) the line holding ``pattern``."""
    import inspect
    lines = open(inspect.getsourcefile(mod)).read().splitlines()
    i = next(n for n, line in enumerate(lines) if pattern in line)
    j = i
    while j > 0 and lines[j - 1].lstrip().startswith("#"):
        j -= 1
    text = " ".join(re.sub(r"^\s*#\s?", "", ln) for ln in lines[j:i])
    tail = lines[i].split("#", 1)[1] if "#" in lines[i] else ""
    return re.sub(r"\s+", " ", f"{text} {tail}").strip()[:limit]


# ------------------------------------------------------------------------------------------------ render parts
_UNIT_UV = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
_FACES = {"front": ((0, -1, 0), (1, 0, 0), (0, 0, 1)), "back": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
          "right": ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "left": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
          "top": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "bottom": ((0, 0, -1), (1, 0, 0), (0, -1, 0))}   # normal, u, v


def _srgb(rgb) -> list[float]:
    """Their render COLORS are USD diffuseColor values (linear); parts are sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in rgb]


def _paint(geom, name: str, label: str = "", glass: bool = False) -> dict:
    """Part painted with their ``render.COLORS[name]`` = (linear colour, roughness, metallic)."""
    rgb, rough, metal = R.COLORS[name]
    return vis(geom, _srgb(rgb), line_of(R, f"{name}=(("), rough, metal,
               name="glass" if glass else f"{label or name} (linear->sRGB)")


def _cabinet(geom) -> dict:
    """The counter cabinet: their render draws a downloaded grey-oak veneer scan (not exported) with no colour of its
    own; a grey-oak estimate at their roughness stands in, and the part name says so."""
    from gift_bags import dressing as DR  # noqa: PLC0415
    return vis(geom, (0.56, 0.52, 0.47), line_of(DR, 'oak = d.scan("grey_oak_veneer_01", 1.5, roughness=.55)'), 0.55,
               0.0, name="cabinet (ESTIMATE: grey oak, the colour of their veneer scan; scan not exported)")


def _quad(c, u, v, size) -> trimesh.Trimesh:
    """Their ``usd_scene.mesh`` quad: corners -u-v, +u-v, +u+v, -u+v about centre ``c`` (normal u x v)."""
    c, u, v = (np.asarray(a, float) for a in (c, u, v))
    a, b = size[0] / 2, size[1] / 2
    return trimesh.Trimesh([c - a * u - b * v, c + a * u - b * v, c + a * u + b * v, c - a * u + b * v],
                           [[0, 1, 2], [0, 2, 3]], process=False)


def _face_quad(face: str, half, centre=(0, 0, 0), inward: bool = False, grow: float = 0.0) -> trimesh.Trimesh:
    """Quad on one face of a box (``half`` extents), normal out (or in)."""
    n, u, v = (np.asarray(a, float) for a in _FACES[face])
    ax = int(np.argmax(np.abs(n)))
    uax, vax = int(np.argmax(np.abs(u))), int(np.argmax(np.abs(v)))
    c = np.asarray(centre, float) + n * (half[ax] + grow)
    return _quad(c, -u if inward else u, v, (2 * half[uax], 2 * half[vax]))


def _tote_uv(q: trimesh.Trimesh, face: str, uvspan) -> np.ndarray:
    """Atlas coordinates of a tote face's corners (their ``render._sheet_uv``: front, right, back, left, bottom)."""
    Wd, G = B.W_BAG, B.G_BAG
    us, vs, voff = uvspan
    p = np.asarray(q.vertices)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    u, v = {"front": (x + Wd / 2, z), "right": (Wd + (y + G / 2), z), "back": (Wd + G + (Wd / 2 - x), z),
            "left": (2 * Wd + G + (G / 2 - y), z), "bottom": (x + Wd / 2, -(y + G / 2))}[face]
    return np.c_[u / us, (v + voff) / vs]


def _tote_visual() -> list[dict]:
    """Outside: their burgundy tote print atlas; inside: the uncoated reverse of the art paper (flat colour)."""
    from PIL import Image  # noqa: PLC0415
    w, g, h = B.W_BAG, B.G_BAG, B.H_BAG
    half, centre = (w / 2, g / 2, h / 2), (0, 0, h / 2)
    with tempfile.TemporaryDirectory() as tmp:
        path, span = AW.tote_print(tmp, w, g, h, B.TURN_TOP)
        atlas = Image.open(path).convert("RGB").copy()
    cite = line_of(R, 'sheet_{k}_out", None, .55')
    outer = [vis(q, [1.0, 1.0, 1.0], cite, .55, 0.0, image=atlas, uv=_tote_uv(q, f, span), name=f"print {f}")
             for f in ("front", "back", "left", "right", "bottom") for q in [_face_quad(f, half, centre)]]
    inner = trimesh.util.concatenate([_face_quad(f, half, centre, inward=True, grow=-P.TOTE.t)
                                      for f in ("front", "back", "left", "right", "bottom")])
    paper = vis(inner, [.93, .91, .87], line_of(AW, "[.93, .91, .87]"), .8, 0.0, name="inside, uncoated paper reverse")
    return outer + [paper]


def _wrap_visual(geom: trimesh.Trimesh) -> list[dict]:
    """Printed outer face (their pattern, uv = rest coordinate / 0.20 m) over the plain back (``wrap_back``)."""
    from PIL import Image  # noqa: PLC0415
    X2, _, _ = WM.wrap_mesh()
    X2 = np.asarray(X2)
    centre = (X2.max(0) + X2.min(0))[:2] / 2
    half = geom.extents / 2
    top = _face_quad("top", half)
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(AW.wrap_pattern(tmp)).convert("RGB").copy()
    uv = (np.asarray(top.vertices)[:, :2] + centre) / .20
    printed = vis(top, [1.0, 1.0, 1.0], line_of(R, 'sheet_{k}_out", None, .25'), .25, 0.0, image=img, uv=uv, name="printed outer face")
    return [printed, _paint(_face_quad("bottom", half), "wrap_back", "back face")]


def _frame_visual(half) -> list[dict]:
    """Photo frame as their ``_photo_frame``: moulding, gilt fillet, hardboard back, print under glass."""
    from PIL import Image  # noqa: PLC0415
    hx, hy, hz = map(float, half)
    mw = R.FRAME_MOULDING
    rails = [box((hx, mw / 2, hz), (0, sy * (hy - mw / 2), 0)) for sy in (-1, 1)]
    stiles = [box((mw / 2, hy - mw + .0002, hz), (sx * (hx - mw / 2), 0, 0)) for sx in (-1, 1)]
    fillets = [box((hx - mw, .0009, hz - .0015), (0, sy * (hy - mw - .0009), -.0015)) for sy in (-1, 1)]
    fillets += [box((.0009, hy - mw, hz - .0015), (sx * (hx - mw - .0009), 0, -.0015)) for sx in (-1, 1)]
    back = box((hx - .002, hy - .002, .002), (0, 0, -hz + .002))
    ox, oy = hx - mw - .0018, hy - mw - .0018
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(AW.framed_print(tmp, 2 * ox, 2 * oy, (2 * (hx - mw) - .026, 2 * (hy - mw) - .026))).convert("RGB").copy()
    plane = lambda zc: _quad((0, 0, zc), (1, 0, 0), (0, 1, 0), (2 * ox, 2 * oy))
    ink = lambda name, geom: vis(geom, _srgb(_color_of(name)[0]), line_of(R, f'"{name}", ('), _color_of(name)[1],
                                 _color_of(name)[2], name=f"{name} (linear->sRGB)")
    return [ink("frame_moulding", union(rails + stiles)), ink("frame_fillet", union(fillets)), ink("frame_back", back),
            vis(plane(hz - .0045), [1.0, 1.0, 1.0], line_of(R, '"frame_print", None, .75'), .75, 0.0, image=img, uv=_UNIT_UV,
                name="print under glass"),
            vis(plane(hz - .0032), _srgb((.95, .97, .97)), line_of(R, '"frame_glass", (.95, .97, .97)'), .03, 0.0, name="glass")]


def _color_of(name: str) -> tuple:
    """Their ``_material(stage, mats, name, (rgb), rough, metal)`` literal for the frame materials."""
    import inspect  # noqa: PLC0415
    m = re.search(rf'"{name}", (\([^)]*\)), ([0-9.]+), ([0-9.]+)\)', inspect.getsource(R))
    return tuple(float(v) for v in m.group(1).strip("()").split(",")), float(m.group(2)), float(m.group(3))


def _candle_visual() -> list[dict]:
    """White carton with the printed panel on its front and back (their ``_body_visual``)."""
    from PIL import Image  # noqa: PLC0415
    hx, hy, hz = S.CANDLE_HALF
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(AW.candle_carton(tmp)).convert("RGB").copy()
    cite = line_of(R, '"candle_print", None, .45')
    parts = [vis(box(S.CANDLE_HALF), _srgb((.80, .76, .68)), line_of(R, '"carton_white", (.80, .76, .68)'), .45, 0.0,
                 name="carton board (linear->sRGB)")]
    for face, flip in (("front", 1), ("back", -1)):
        q = _quad((0, flip * (hy + 2e-4) * -1, 0), (flip, 0, 0), (0, 0, 1), (2 * hx * .97, 2 * hz * .97))
        parts.append(vis(q, [1.0, 1.0, 1.0], cite, .45, 0.0, image=img, uv=_UNIT_UV, name=f"print {face}"))
    return parts


def _carton_visual(parts_geom) -> list[dict]:
    """Kraft corrugated body with the one-colour print panel on the outside of both long side walls
    (their ``carton.draw``: the +y outside face of a wall wider than 0.1 m, uv 0..1); flaps plain kraft."""
    from PIL import Image  # noqa: PLC0415
    from gift_bags import carton as C2  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(C2.textures(tmp)["print"]).convert("RGB").copy()
    cite = line_of(R, 'f"carton_{k_}", None, .9')
    body = [_paint(union([box(h, c) for h, c in parts_geom]), "corrugated", "kraft corrugated")]
    for h, c in parts_geom:
        if int(np.argmin(h)) != 1 or h[0] <= .1:
            continue
        sgn = 1.0 if c[1] > 0 else -1.0
        q = _quad(np.add(c, (0, sgn * h[1], 0)), np.cross((0, 0, 1), (0, sgn, 0)), (0, 0, 1), (2 * h[0], 2 * h[2]))
        body.append(vis(q, [1.0, 1.0, 1.0], cite, .9, 0.0, image=img, uv=_UNIT_UV, name="flexo print"))
    return body


def _mu(a: str, b: str) -> dict:
    """Their pair table's coefficient for classes ``a`` / ``b`` with the line it is on."""
    val = P.FRICTION[frozenset((a, b))]
    for pat in (f'("{a}", "{b}")', f'("{b}", "{a}")', f'(("{a}", _m), .30)'):
        try:
            cite = line_of(P, pat)
            break
        except KeyError:
            continue
    else:
        raise KeyError(f"no friction pair line for {a}/{b}")
    return cited(val, "estimate", cite, f"friction pair ({a}, {b}) of params._MU_LIST; values inside paper-friction ranges are [E]")


def _rigid(x: Exporter, key: str, name: str, cat: str, geom, **kw) -> None:
    x.add(key, name, cat, geom, **kw)


# ------------------------------------------------------------------------------------------------ sheets
def _cloth(mat: P.ShellMat, pattern: str, basis_e: str, quote: str) -> dict:
    cite = line_of(P, pattern)
    nu = mat.nu12
    return {"thickness": cited(mat.t, "derived" if basis_e == "literature" else "estimate", cite, quote),
            "stretch_warp": cited(mat.Et_md, basis_e, cite, quote), "stretch_weft": cited(mat.Et_cd, basis_e, cite, quote),
            "shear": cited(mat.Gt12, "derived", cite, quote + " (their shear membrane Gt12)"),
            "bend_warp": cited(mat.D_md, "derived", cite, quote + " (E t^3 / 12(1-nu^2), their D_md)"),
            "bend_weft": cited(mat.D_cd, "derived", cite, quote + " (E t^3 / 12(1-nu^2), their D_cd)")} | {"_nu": nu}


def _clean(c: dict) -> dict:
    return {k: v for k, v in c.items() if not k.startswith("_")}


def _tote_shell() -> trimesh.Trimesh:
    """Open-top 203 x 102 x 254 mm tote as a thin wall (laminate caliper): outer box minus inner box."""
    w, g, h = B.W_BAG, B.G_BAG, B.H_BAG
    t = P.TOTE.t
    outer = box((w / 2, g / 2, h / 2), (0, 0, h / 2))
    inner = box((w / 2 - t, g / 2 - t, h / 2), (0, 0, h / 2 + t))
    return outer.difference(inner, engine="manifold")


def _tote(x: Exporter) -> None:
    cite = line_of(B, "W_BAG, G_BAG, H_BAG =")
    cl = _clean(_cloth(P.TOTE, '"tote laminate', "literature", _block(P, "_T_PAPER, _T_BOPP =", 330)))
    x.add("gift_tote", "Rope-handle Euro tote (laminated paper, 203 x 102 x 254 mm)", "container", _tote_shell(), kind="cloth", visual=_tote_visual(),
          material={"name": "BOPP-laminated coated paper with a turn-top board band"},
          mass=cited(S.BAG_MASS, "derived", line_of(S, "BAG_MASS ="), "kg per tote [D]"),
          friction=_mu("laminate", "laminate"), friction_partner="another tote / the counter top",
          thickness=cl["thickness"], cloth=cl,
          dims=[dim("width", B.W_BAG, "x", cite), dim("gusset depth", B.G_BAG, "y", cite), dim("height", B.H_BAG, "z", cite)],
          source_ref=where(B.bag_mesh),
          notes="Their tote is an open triangulated sheet; here its rest box (front, back, gussets) as a thin closed "
                "shell of the laminate caliper, open top. Turn-top board band, creases and rope holes are not modelled.")


def _wrap_slab() -> trimesh.Trimesh:
    X2, _, _ = WM.wrap_mesh()
    ext = X2.max(0) - X2.min(0)
    return box((ext[0] / 2, ext[1] / 2, P.WRAP.t / 2))


def _wrap(x: Exporter) -> None:
    cl = _clean(_cloth(P.WRAP, '"gift wrap 81', "estimate", _block(P, "_T_WRAP =", 330)))
    geom = _wrap_slab()
    x.add("gift_wrap", "Gift wrap sheet (81 g/m2 white gloss, Uline S-8551 class)", "cloth", geom, kind="cloth",
          visual=_wrap_visual(geom), material={"name": "coated gift-wrap paper"}, thickness=cl["thickness"], cloth=cl,
          friction=_mu("wrap", "wrap"), friction_partner="wrap on wrap",
          dims=[dim("net length", float(geom.extents[0]), "x", where(WM.wrap_mesh)),
                dim("caliper", P.WRAP.t, "z", line_of(P, "_T_WRAP ="))],
          source_ref=where(WM.wrap_mesh),
          notes="Flat net of the authored 180 x 130 x 55 mm present as a slab of the paper caliper; their sheet is an "
                "open triangulated net. Areal mass 81 g/m2 (not given as a total mass).")


def _tape(x: Exporter) -> None:
    cl = _clean(_cloth(P.MAGIC_TAPE, '"Scotch Magic 810', "estimate", _block(P, "MAGIC_TAPE = ShellMat", 330)))
    x.add("magic_tape_strip", "Scotch Magic 810 tape strip (19 mm x 60 mm seam piece)", "other",
          box((0.030, WM.TAPE_W / 2, P.MAGIC_TAPE.t / 2)), kind="cloth", material={"name": "matte acetate tape"},
          visual=[_paint(box((0.030, WM.TAPE_W / 2, P.MAGIC_TAPE.t / 2)), "tape_clear")],
          thickness=cl["thickness"], cloth=cl, friction=_mu("tape", "tape"), friction_partner="tape on tape",
          dims=[dim("width", WM.TAPE_W, "y", line_of(WM, "TAPE_W =")), dim("caliper", P.MAGIC_TAPE.t, "z", cl["thickness"]["cite"])],
          source_ref=where(WM.tape_sheets),
          notes="The 60 mm seam strip (the two end strips are 50 mm); 56 um total, areal mass 65 g/m2 (E).")


# ------------------------------------------------------------------------------------------------ rigid small items
def _small_items(x: Exporter) -> None:
    pl_outline = [(-B.W_BAG / 2, -B.H_BAG / 2), (B.W_BAG / 2, -B.H_BAG / 2), (B.W_BAG / 2, B.H_BAG / 2), (-B.W_BAG / 2, B.H_BAG / 2)]
    v, f = S.prism_mesh(pl_outline, S.BAG_PLATE_T)
    _rigid(x, "tote_plate", "Flat tote of the lower stack (rigid plate)", "container", mesh(v, f),
           visual=[_paint(mesh(v, f), "bag_plate")], material={"name": "folded laminate tote, 4 layers + turn-top board"},
           mass=cited(S.BAG_MASS, "derived", line_of(S, "BAG_MASS ="), "kg per tote [D]"),
           friction=_mu("laminate", "laminate"), friction_partner="another tote",
           dims=[dim("thickness", S.BAG_PLATE_T, "z", line_of(S, "BAG_PLATE_T =")), dim("length", B.H_BAG, "y", line_of(B, "W_BAG, G_BAG, H_BAG ="))],
           source_ref=where(S._counter_base), notes="Rigid plate with the shipped flat tote's outline (flap folded up).")
    _rigid(x, "bottom_board", "Tote bottom board (200 x 99 x 0.5 mm chipboard)", "other", box(S.BOARD_HALF),
           visual=[vis(box(S.BOARD_HALF), [.56, .53, .48], line_of(R, '"chipboard_tex", None, .85'), .85, 0.0,
                       name="chipboard (grain texture base colour)")],
           material={"name": "350 g/m2 chipboard"},
           mass=cited(S.BOARD_MASS, "derived", line_of(S, "BOARD_MASS ="), _block(S, "BOARD_HALF =")),
           friction=_mu("chipboard", "chipboard"), friction_partner="another board",
           dims=[dim("length", 2 * S.BOARD_HALF[0], "x", line_of(S, "BOARD_HALF ="))], source_ref=where(S.counter),
           notes="One of the three boards in the stack.")
    _rigid(x, "candle_carton", "Boxed candle carton (SBS folding carton 90 x 90 x 110 mm)", "container", box(S.CANDLE_HALF),
           visual=_candle_visual(), material={"name": "SBS folding carton with candle"},
           mass=cited(S.CANDLE_MASS, "estimate", line_of(S, "CANDLE_MASS ="), "SBS folding carton 90 x 90 x 110 mm [E]"),
           friction=_mu("carton", "counter"), friction_partner="the counter top",
           dims=[dim("height", 2 * S.CANDLE_HALF[2], "z", line_of(S, "CANDLE_HALF =")),
                 dim("width", 2 * S.CANDLE_HALF[0], "x", line_of(S, "CANDLE_HALF ="))], source_ref=where(S.counter),
           notes="Solid box as in their sim (printed artwork has no geometry).")
    fh = S.FRAME_HALF
    _rigid(x, "photo_frame", "Photo frame 170 x 120 x 15 mm", "other", box(fh),
           visual=_frame_visual(fh), material={"name": "wood and glass frame"},
           mass=cited(S.FRAME_MASS, "estimate", line_of(S, "FRAME_MASS ="), _block(S, "FRAME_HALF =")),
           friction=_mu("frame", "greyboard"), friction_partner="the gift box floor",
           dims=[dim("length", 2 * fh[0], "x", line_of(S, "FRAME_HALF =")), dim("thickness", 2 * fh[2], "z", line_of(S, "FRAME_HALF ="))],
           source_ref=where(S.home), notes="Solid box as their collider.")


def _gift_box(x: Exporter) -> None:
    b = GB.box_spec()
    cite = line_of(GB, "BOX_HALF =")
    base = union([box(h, c) for h, c in GB.tray_boxes(b.hx, b.hy, b.h, b.wall, b.wall)])
    lhx, lhy = b.hx + b.clearance + b.wall, b.hy + b.clearance + b.wall
    lid = union([box(h, c) for h, c in GB.tray_boxes(lhx, lhy, b.lid_h, b.wall, b.wall, open_up=False)])
    q = "kg [D: areas x (0.975 + 0.128 kg/m2)]"
    common = dict(friction=_mu("greyboard", "greyboard"), friction_partner="the other box part / table",
                  material={"name": "paper-covered greyboard, 1.5 mm"}, thickness=cited(b.wall, "datasheet", line_of(GB, "WALL ="), "greyboard 1.5 mm [V]"))
    x.add("gift_box__base", "Telescoping gift box base (180 x 130 x 55 mm)", "articulated", base,
          visual=[_paint(base, "giftbox_base")],
          mass=cited(b.base_mass, "derived", line_of(GB, "BASE_MASS, LID_MASS ="), q),
          dims=[dim("length", 2 * b.hx, "x", cite), dim("width", 2 * b.hy, "y", cite), dim("height", b.h, "z", cite)],
          source_ref=where(GB.make_box), notes="Open tray from their tray_boxes (floor + four walls).", **common)
    x.add("gift_box__lid", "Telescoping gift box lid", "articulated", lid,
          visual=[_paint(lid, "giftbox_lid")],
          mass=cited(b.lid_mass, "derived", line_of(GB, "BASE_MASS, LID_MASS ="), q),
          dims=[dim("length", 2 * lhx, "x", cite), dim("skirt depth", b.lid_h, "z", line_of(GB, "LID_H ="))],
          source_ref=where(GB.make_box), notes="Telescoping lid, opening down; slides on an air film over the base.",
          extra={"joint": {"type": "free", "parent": "gift_box__base", "axis": [0, 0, 1], "range": [0.0, b.lid_h - b.wall],
                           "cite": where(GB.AirFilm)}}, **common)


# ------------------------------------------------------------------------------------------------ supply carton
def _carton(x: Exporter) -> None:
    m = CT.masses()
    cite_m = line_of(CT, "def masses")
    q = "Board masses from areas x RHO_A [D]; RHO_A 0.55 kg/m2 C-flute 32 ECT [E]"
    mat = {"name": "C-flute corrugated board, 4 mm"}
    thick = cited(CT.T_B, "datasheet", line_of(CT, "T_B ="), "C-flute caliper 3.8-4.1 mm [V]")
    mu = _mu("corrugated", "corrugated")
    x.add("rsc_carton", "Supply carton RSC 14 x 10 x 8 in (C-flute)", "container", union([box(h, c) for h, c in CT.body_parts()]),
          visual=_carton_visual(CT.body_parts()),
          material=mat, thickness=thick, mass=cited(m["body"], "derived", cite_m, q), friction=mu,
          friction_partner="another carton / stainless table",
          dims=[dim("outer length", CT.L_OUT, "x", line_of(CT, "L_OUT, W_OUT ="))], source_ref=where(CT.body_parts),
          notes="Walls and two-layer floor as solid board boxes (their body); top flaps are separate parts; seam tape omitted.")
    for kind, key in (("major", "flap_major"), ("minor", "flap_minor")):
        spec = next(s for s in CT.flap_specs().values() if s["kind"] == kind)
        x.add(f"rsc_carton__{key}", f"RSC carton {kind} top flap", "articulated", box(spec["half"]), material=mat,
              visual=[_paint(box(spec["half"]), "corrugated", "kraft corrugated")],
              thickness=thick, mass=cited(m[kind], "derived", cite_m, q), friction=mu, friction_partner="another carton",
              source_ref=where(CT.flap_specs), notes="Rigid 4 mm plate; hinged along its score line (axis through its outer edge).",
              extra={"joint": {"type": "revolute", "parent": "rsc_carton", "axis": [float(v) for v in
                                np.subtract(spec["axis"][1], spec["axis"][0]) / spec["length"]],
                               "range": [0.0, math.pi], "cite": where(CT.flap_specs),
                               "crease_moment_90deg_Nm_per_m": 0.8, "crease_cite": line_of(P, "FLAP_CREASE =")}})
    _contents(x)


def _contents(x: Exporter) -> None:
    th, tm = CT.TEA_HALF, CT.TEA_MASS
    cite = line_of(CT, "TEA_HALF =")
    x.add("tea_carton", "Tea carton 150 x 80 x 70 mm (SBS folding carton)", "food", box(th),
          visual=[_paint(box(th), "tea_carton")], material={"name": "SBS folding carton with tea"}, mass=cited(tm, "estimate", line_of(CT, "TEA_MASS ="), "contents [E]"),
          friction=_mu("carton", "carton"), friction_partner="another carton",
          dims=[dim("length", 2 * th[1], "y", cite), dim("height", 2 * th[2], "z", cite)], source_ref=where(CT.content_specs),
          notes="Solid box as their rigid content.")
    x.add("goods_jar", "Glass jar of goods (85 x 110 mm, 450 g)", "container", cylinder(CT.JAR_R, CT.JAR_H / 2),
          visual=[_paint(cylinder(CT.JAR_R, CT.JAR_H / 2), "glass_jar", glass=True)], material={"name": "glass jar with contents"}, mass=cited(CT.JAR_MASS, "estimate", line_of(CT, "JAR_R, JAR_H, JAR_MASS"), "contents [E]"),
          friction=_mu("glass", "carton"), friction_partner="carton floor",
          dims=[dim("diameter", 2 * CT.JAR_R, "diameter_xy", line_of(CT, "JAR_R, JAR_H, JAR_MASS")),
                dim("height", CT.JAR_H, "z", line_of(CT, "JAR_R, JAR_H, JAR_MASS"))], source_ref=where(CT.content_specs),
          notes="Filled jar as one solid cylinder, as their rigid prism body; the loose kraft wrap is not built.")


def _rope(x: Exporter) -> None:
    from gift_bags import handles as H
    r = P.PP_ROPE
    cite = line_of(P, "PP_ROPE =")
    quote = _block(P, "PP_ROPE =")
    tube = trimesh.creation.cylinder(radius=r.radius, segment=[(0, 0, 0), (H.LOOP_LEN, 0, 0)], sections=16)
    x.add("rope_handle", "Braided PP rope handle (5 mm, 400 mm loop)", "other", None, kind="cable",
          visual=[_paint(tube, "rope")],
          material={"name": "braided polypropylene rope"},
          mass=cited(r.lin_rho * H.LOOP_LEN, "derived", cite, quote),
          friction=_mu("rope", "laminate"), friction_partner="tote laminate",
          cable={"points": [[0.0, 0.0, 0.0], [H.LOOP_LEN, 0.0, 0.0]], "radius": cited(r.radius, "datasheet", cite, quote),
                 "bend_stiffness": cited(r.EI, "estimate", cite, quote)},
          source_ref=where(H.author),
          notes="Centre line given straight over the 400 mm loop length; their rod hangs as a loop between the two holes.")


# ------------------------------------------------------------------------------------------------ furniture
def _furniture(x: Exporter) -> None:
    z = S.COUNTER_Z
    top = box((.76, .30, .02), (0, 0, z - .02))
    cab = box((.76 - .02, .30 - .04, (z - .04) / 2), (0, .02, (z - .04) / 2))
    cite = line_of(S, "COUNTER_Z =")
    _static(x, "counter", "Cash-wrap counter with cabinet (36 in top)", union([top, cab]), "laminate-topped counter on a box cabinet",
            _mu("laminate", "counter"), "wrapping supplies", [dim("top height", z, "z", cite)], where(S._counter_base),
            look=[_paint(top, "counter_laminate", "laminate top"), _cabinet(cab)])
    mat = box(S.MAT_HALF, (0, 0, S.MAT_HALF[2]))
    _static(x, "counter_mat", "Counter mat 400 x 300 x 3 mm", mat, "rubber mat", cited(.6, "estimate", where(S._counter_base), "mat mu=.6"),
            "the tote", [dim("width", 2 * S.MAT_HALF[0], "x", line_of(S, "MAT_HALF ="))], where(S._counter_base),
            look=[_paint(mat, "counter_mat")])
    zt = S.TABLE_Z
    legs = [box((.03, .03, (zt - .04) / 2), (sx * (.75 - .06), sy * (.45 - .06), (zt - .04) / 2)) for sx in (-1, 1) for sy in (-1, 1)]
    _static(x, "dining_table", "Dining table (0.75 m)", union([box((.75, .45, .02), (0, 0, zt - .02))] + legs), "oak table",
            _mu("oak", "greyboard"), "gift box", [dim("top height", zt, "z", line_of(S, "TABLE_Z ="))], where(S.home),
            look=[_paint(union([box((.75, .45, .02), (0, 0, zt - .02))] + legs), "table_wood", "veneer fallback colour")])
    zr = S.RECEIVING_Z
    hx, hy, _ = S.RECEIVING_TABLE_HALF
    legs = [box((.019, .019, (zr - .04) / 2), (sx * (hx - .04), sy * (hy - .04), (zr - .04) / 2)) for sx in (-1, 1) for sy in (-1, 1)]
    shelf = box((hx - .03, hy - .03, .006), (0, 0, .18))
    _static(x, "receiving_table", "Stainless receiving table with undershelf", union([box(S.RECEIVING_TABLE_HALF, (0, 0, zr - .02)), shelf] + legs),
            "stainless steel table", _mu("carton", "steel"), "the supply carton",
            [dim("top height", zr, "z", line_of(S, "RECEIVING_Z ="))], where(S.receiving),
            look=[_paint(union([box(S.RECEIVING_TABLE_HALF, (0, 0, zr - .02)), shelf] + legs), "steel_top", "stainless")])
    x.fail("shelving", "display and steel shelving are Poly Haven model bounding boxes (CC0 render dressing), excluded")


def _static(x: Exporter, key: str, name: str, geom, mat: str, mu: dict, partner: str, dims: list, cite: str,
            look: list[dict] | None = None) -> None:
    x.add(key, name, "furniture", geom, motion="static", material={"name": mat}, friction=mu, friction_partner=partner,
          visual=look,
          dims=dims, source_ref=cite, notes="Fixed collision boxes of their scene, unioned and centred on the top's x/y.")


def export(x: Exporter) -> None:
    steps = [_tote, _wrap, _tape, _small_items, _gift_box, _carton, _rope, _furniture]
    for fn in steps:
        try:
            fn(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__.strip("_"), f"{type(exc).__name__}: {exc}")
            log.exception("%s failed", fn.__name__)
