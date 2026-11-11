# SPDX-License-Identifier: Apache-2.0
"""whiteboard: the 8 x 4 ft porcelain board with its tray, a marker and its cap, the magnetic eraser, a student desk
and chair and the teacher's desk, from their ``geometry.py`` scene description (one list of boxes, cylinders and a
curved shell per object) and the ``config.py`` priors.

Each object is the union of its collision parts in its own frame (the object's world placement is dropped).
"""

from __future__ import annotations

import inspect
import re
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from _common import Exporter, box, cited, cylinder, dim, line_of, mesh, union, vis, where

from whiteboard import config as CFG
from whiteboard import geometry as G
from whiteboard import tool_appearance as TA
from whiteboard import usd_scene as US

REPORT = "whiteboard REPORT p.3"
MARKER_QUOTE = ("STAEDTLER 351 B capped envelope 138 x 17 x 17 mm, total mass 19.6 g; split into a 16.2 g marker and "
                "3.4 g cap, an estimate (" + REPORT + ")")


def part_mesh(p: G.Part) -> trimesh.Trimesh:
    """One part as a closed mesh (their ``size`` is the full box extents / cylinder radius and length)."""
    if p.kind == "box":
        return box(np.asarray(p.size) / 2, p.position, p.rotation)
    if p.kind == "cylinder":
        return cylinder(p.size[0], p.size[1] / 2, p.position, p.rotation)
    m = mesh(p.vertices, p.faces)
    return m.apply_translation(p.position)


def _srgb(lin) -> list[float]:
    """Their PALETTE is UsdPreviewSurface diffuseColor (linear); the part colour is sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]


def _texture_base(name: str):
    """(sRGB 0-255, cite) of the base colour their ``textures()`` paints the ``name`` surface from, or None."""
    text = Path(inspect.getsourcefile(US)).read_text()
    m = re.search(rf'\("{name}", \((\d+), (\d+), (\d+)\)\)', text)
    return (tuple(int(g) for g in m.groups()), line_of(US, f'("{name}", (')) if m else None


def _look(geom, material: str) -> dict:
    """Part with their PALETTE row for ``material``; wood and laminate are drawn from a generated texture whose base
    colour is used instead of the palette's (flat; the grain pattern is not exported)."""
    rgb, rough, metal = US.PALETTE[material]
    base = _texture_base(material) if material in ("wood", "laminate") else None
    if base:
        return vis(geom, base[0], base[1], rough, metal, name=f"{material} (texture base colour; grain not exported)")
    row = f'"{material}": ('
    if material in CFG.COLORS:       # marker colours: config.COLORS, drawn at roughness 0.29 by the PALETTE comprehension
        cite = line_of(CFG, row) + "; roughness " + line_of(US, "**{name: (rgb, 0.29, 0)")
    else:
        cite = line_of(US, row)
    return vis(geom, _srgb(rgb), cite, rough, metal, name=f"{material} (linear->sRGB)")


def object_visual(obj: G.Object) -> list[dict]:
    """One part per ``Part.material`` of the object's collision parts (their render binds PALETTE[material])."""
    groups: dict[str, list] = {}
    for p in obj.parts:
        if p.collision:
            groups.setdefault(p.material, []).append(part_mesh(p))
    return [_look(trimesh.util.concatenate(ms), name) for name, ms in groups.items()]


def _barrel_label_part(obj: G.Object) -> dict:
    """The marker barrel with their generated 'Lumocolor' label (``marker_label`` + ``cylindrical_uv``)."""
    part = next(p for p in obj.parts if p.name == "barrel")
    height = part.size[1]
    shape = trimesh.creation.cylinder(radius=part.size[0], height=height, sections=64)
    uv = TA.cylindrical_uv(shape.vertices, shape.faces, height)
    flat = trimesh.Trimesh(shape.vertices[shape.faces].reshape(-1, 3), np.arange(3 * len(shape.faces)).reshape(-1, 3),
                           process=False)
    flat.apply_translation(part.position)
    from PIL import Image  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        image = Image.open(TA.marker_label(Path(tmp))).convert("RGB")
        image.load()
    rgb, rough, metal = US.PALETTE["marker_white"]
    return vis(flat, _srgb(rgb), line_of(US, '"marker_white": ('), rough, metal, image=image, uv=uv,
               name="barrel (printed label from marker_label)")


def marker_visual(obj: G.Object) -> list[dict]:
    rest = G.Object(obj.name, parts=[p for p in obj.parts if p.name != "barrel"])
    return [_barrel_label_part(obj), *object_visual(rest)]


def object_mesh(obj: G.Object) -> trimesh.Trimesh:
    parts = [part_mesh(p) for p in obj.parts if p.collision]
    return union(parts)


def _objects(cfg) -> dict[str, G.Object]:
    objs = {o.name: o for o in G.classroom(cfg)}
    objs.update({o.name: o for o in G.tools(cfg)})
    return objs


def _board(x: Exporter, o: G.Object, b) -> None:
    x.add("whiteboard", "Classroom whiteboard (8 x 4 ft porcelain on steel, with tray)", "fixture", object_mesh(o),
          visual=object_visual(o),
          motion="static", material={"name": "porcelain-enamel steel with aluminium trim and tray"},
          friction=cited(0.35, "estimate", line_of(G, "friction=0.35,"), "board/nib friction 0.35, E contact-pair "
                         "estimate (" + REPORT + ")"), friction_partner="marker nib",
          dims=[dim("width", b.width, "x", line_of(CFG, "width: float")), dim("height", b.height, "z",
                                                                                line_of(CFG, "height: float"))],
          source_ref=where(G.classroom),
          notes="Backer, porcelain face, trims, tray, tray end caps, brackets and mount spacers unioned; the board is a "
                "fixed boundary (no mass in their code). Dimensions are the MooreCo 8 x 4 ft family.")


def _tools(x: Exporter, objs: dict, cfg) -> None:
    cite_m = where(G.marker)
    barrel, cap = objs["marker_black"], objs["cap_black"]
    mu = cited(0.45, "estimate", line_of(G, "friction=0.45"), "finger/tool friction 0.45, E estimate (" + REPORT + ")")
    x.add("marker_pen", "Whiteboard marker (STAEDTLER 351 B class), without cap", "tool", object_mesh(barrel),
          material={"name": "polypropylene barrel with felt nib"},
          mass=cited(barrel.mass, "estimate", cite_m, MARKER_QUOTE), friction=mu, friction_partner="gripper fingers",
          source_ref=cite_m, visual=marker_visual(barrel),
          notes="Barrel, collar, tail and chisel nib as rigid parts (their elastic nib is a separate tetrahedral model); "
                "axis along body z, nib at +z. Six colours exist; black is exported.")
    x.add("marker_cap", "Whiteboard marker cap (hollow)", "container", object_mesh(cap),
          visual=object_visual(cap),
          material={"name": "polypropylene"}, mass=cited(cap.mass, "estimate", cite_m, MARKER_QUOTE),
          friction=mu, friction_partner="marker barrel and fingers", source_ref=cite_m,
          notes="Twelve wall boxes in a ring (their rigid proxy for finger contact), end disc and clip; the real wall "
                "is an elastic shell with a bore and three retention ribs.")
    e = objs["eraser"]
    x.add("board_eraser", "Magnetic board eraser (652 class)", "tool", object_mesh(e),
          visual=object_visual(e),
          material={"name": "plastic holder with felt pad"},
          mass=cited(cfg.materials.eraser_mass_kg, "estimate", line_of(CFG, "eraser_mass_kg"),
                     "30 g total mass and magnetic force law are declared estimates (" + REPORT + ")"),
          friction=cited(0.55, "estimate", line_of(G, "friction=0.55,"), "pad/board friction 0.55, E estimate"),
          friction_partner="the board", dims=[dim("length", 0.107, "x", line_of(G, '"housing"')),
                                              dim("width", 0.057, "y", line_of(G, '"housing"'))],
          source_ref=where(G.tools), notes="Holder box and 103 x 53 x 5 mm pad; the magnet is a force law in their code.")


def _furniture(x: Exporter, objs: dict, cfg) -> None:
    pri = cfg.materials
    for key, oname, name, mass, mpat, mu_p, dims in (
            ("student_desk", "desk_0_0", "Student desk (24 x 18 in top)", pri.desk_mass_kg, "desk_mass_kg", .30,
             [dim("top width", 0.6096, "x", line_of(G, "0.6096, 0.4572"))]),
            ("student_chair", "chair_0_0", "Student chair (blue shell, tube frame)", pri.chair_mass_kg, "chair_mass_kg",
             .28, [])):
        o = objs[oname]
        x.add(key, name, "furniture", object_mesh(o), visual=object_visual(o), material={"name": "laminate / polyethylene on steel tube"},
              mass=cited(mass, "estimate", line_of(CFG, mpat), "engineering prior, NOT a manufacturer measurement"),
              friction=cited(mu_p, "estimate", line_of(G, f"friction={mu_p:.2f},"), "glide friction on linoleum"),
              friction_partner="linoleum floor", dims=dims, source_ref=where(G.classroom),
              notes="Union of their collision parts (top, book box, tube legs, glides; the chair has a seat, curved back "
                    "shell, rails and supports).")
    t = objs["teacher_desk"]
    x.add("teacher_desk", "Teacher's desk (wood top, modesty panel)", "furniture", object_mesh(t),
          visual=object_visual(t),
          material={"name": "wood top on steel legs"},
          mass=cited(28.0, "estimate", line_of(G, "mass=28.0"), "teacher desk mass 28 kg (authored)"),
          friction=cited(0.4, "estimate", line_of(G, "friction: float = 0.4"), "Part default friction"),
          friction_partner="linoleum floor", dims=[dim("top length", 1.2, "x", line_of(G, '"top", (1.20, 0.65'))],
          source_ref=where(G.classroom), notes="Top, modesty panel and four legs of their teacher desk.")


def export(x: Exporter) -> None:
    cfg = CFG.Config()
    objs = _objects(cfg)
    for fn, args in ((_board, (objs["board"], cfg.board)), (_tools, (objs, cfg)), (_furniture, (objs, cfg))):
        try:
            fn(x, *args)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
