# SPDX-License-Identifier: Apache-2.0
"""pencap: pen, cap (hollow revolved solid), notebook, capping fixture and the work table, from their ``newton15``
geometry and scene code (no engine needed: only the profile functions and constants are used)."""

from __future__ import annotations

import inspect

import trimesh
from _common import Exporter, box, cited, dim, line_of, mesh, union, vis, where

from pencap import task_presets as TP
from pencap.newton15 import build as B
from pencap.newton15 import config as CFG
from pencap.newton15 import geometry as G
from pencap.newton15 import render as RN
from pencap.newton15 import scene as S

REPORT_QUOTE = ("Pen: 140 mm, 5.8 g. Cap: 58 mm, 5.0 g. These are inherited authored parameters, not a measured "
                "commercial product pair (pencap REPORT p.2)")


def _revolved(profile) -> trimesh.Trimesh:
    """Their own revolve() of a profile (axis = local y), as a trimesh."""
    v, f = G.revolve(profile)
    return mesh(v, f)


def _srgb(lin) -> list[float]:
    """Their render palette is a UsdPreviewSurface diffuseColor (linear); the part colour is sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]


def _look(geom, material: str, name: str = "", cite_key: str | None = None) -> dict:
    """One render part with the PALETTE entry ``material`` (colour, roughness, metallic) of their ``render.py``."""
    rgb, rough, metal = RN.PALETTE[material]
    return vis(geom, _srgb(rgb), line_of(RN, f'"{cite_key or material}": (('), rough, metal,
               name=f"{name or material} (linear->sRGB)")


def _pen_parts(pen: trimesh.Trimesh) -> list[dict]:
    """Aluminium writing tip (faces within 4 mm of the tip, their finish subset) and the orange shell."""
    tip = pen.vertices[pen.faces].mean(axis=1)[:, 1] < .004
    parts = []
    for mask, material, name in ((~tip, "pen", "pen shell"), (tip, "aluminum", "pen tip")):
        if mask.any():
            parts.append(_look(trimesh.Trimesh(pen.vertices, pen.faces[mask], process=False), material, name))
    return parts


def _plastic_mu(c) -> dict:
    return cited(c.plastic_friction, "authored", line_of(CFG, "plastic_friction:"),
                 "plastic_friction = 0.25 (pen and cap contact material)")


def _pen_cap(x: Exporter, c) -> None:
    pen = _revolved(G.pen_profile(c))
    x.add("pen", "Pen (revolved shell)", "tool", pen, material={"name": "ABS-like plastic"},
          mass=cited(c.pen_mass, "authored", line_of(CFG, "pen_mass:"), REPORT_QUOTE),
          friction=_plastic_mu(c), friction_partner="cap and table",
          dims=[dim("length", c.pen_length, "y", line_of(CFG, "pen_length:"))],
          source_ref=where(G.pen_profile), visual=_pen_parts(pen),
          notes="Solid revolved profile (axis = body y, tip at y = 0); the four click nubs are separate sphere "
                "colliders in their sim and are not part of the mesh.")
    cap = _revolved(G.cap_profile(c))
    x.add("pen_cap", "Pen cap (hollow, with bore)", "container", cap, material={"name": "ABS-like plastic"},
          mass=cited(c.cap_mass, "authored", line_of(CFG, "cap_mass:"), REPORT_QUOTE),
          friction=_plastic_mu(c), friction_partner="pen and table",
          dims=[dim("length", c.cap_length, "y", line_of(CFG, "cap_length:")),
                dim("outer diameter", 2 * c.cap_radius, "x", line_of(CFG, "cap_radius:"))],
          source_ref=where(G.cap_profile), visual=[_look(cap, "cap")],
          notes="Hollow revolved wall with the open bore (axis = body y, mouth at y = 0); wall volume only. "
                "The four contact nubs are not meshed.")


def _notebook(x: Exporter) -> None:
    src = inspect.getsource(S.studio)
    mu = float(src.split("notebook_contact = ")[1].split("mu=")[1].split(",")[0])
    parts = [box((.074, .105, .007)), box((.076, .107, .0005), (0, 0, -.0075)), box((.002, .107, .008), (-.074, 0, 0))]
    look = [_look(union(parts[:1]), "paper", "pages"), _look(union(parts[1:]), "cap", "cover and spine")]
    x.add("notebook", "Bound notebook", "other", union(parts), visual=look, material={"name": "paper and card"},
          mass=cited(.22, "authored", line_of(S, "mass=.22"), "A bound notebook is a movable rigid object, 220 g"),
          friction=cited(mu, "authored", line_of(S, "notebook_contact ="), "notebook contact mu .4"),
          friction_partner="table",
          dims=[dim("cover width", 0.152, "x", line_of(S, "notebook_cover")),
                dim("cover length", 0.214, "y", line_of(S, "notebook_cover"))],
          source_ref=where(S.studio),
          notes="Pages, cover and spine boxes of their scene unioned in the body frame (origin at the page-block "
                "centre).")


def _fixture(x: Exporter, c) -> None:
    boxes = TP.fixture_boxes(c)
    cx, cy = TP.FIXTURE_XY
    parts = [box(h, (p[0] - cx, p[1] - cy, p[2] - c.table_top)) for _, p, h in boxes]
    mu = float(inspect.getsource(TP.apply_start).split("mu=")[1].split(",")[0])
    x.add("capping_fixture", "Capping fixture (60 mm block, 14 mm bore)", "fixture", union(parts), motion="static",
          visual=[_look(union(parts), "aluminum")],
          material={"name": "aluminium"},
          friction=cited(mu, "authored", line_of(TP, "mu=.35"), "fixture contact mu .35"),
          friction_partner="pen and cap",
          dims=[dim("height", TP.FIXTURE_HEIGHT, "z", line_of(TP, "FIXTURE_HEIGHT =")),
                dim("outside width", 2 * TP.FIXTURE_HALF, "x", line_of(TP, "FIXTURE_HALF ="))],
          source_ref=where(TP.fixture_boxes),
          notes="Five non-overlapping boxes (floor and four walls) around a 14 mm square bore 50 mm deep, unioned; "
                "body frame origin at the base centre on the table top. No mass: it is fixed.")


def _table(x: Exporter, c) -> None:
    mu = 0.35  # box() default in build()
    x.add("work_table", "Work table top", "furniture", box((.55, .45, .015)), motion="static",
          visual=[_look(box((.55, .45, .015)), "wood", "oak veneer top (fallback colour; texture not exported)")],
          material={"name": "wood"},
          friction=cited(mu, "authored", line_of(B, "mu=.35"), "box() default mu .35"), friction_partner="pen and cap",
          dims=[dim("thickness", 0.03, "z", line_of(B, '"table"')), dim("length", 1.1, "x", line_of(B, '"table"'))],
          source_ref=line_of(B, '"table"'), notes="Single 1.1 x 0.9 x 0.03 m box collider; top at z = 0.75 m.")


def export(x: Exporter) -> None:
    c = CFG.Config()
    for fn in (_pen_cap, _fixture, _table):
        try:
            fn(x, c)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
    try:
        _notebook(x)
    except Exception as exc:  # noqa: BLE001
        x.fail("notebook", f"{type(exc).__name__}: {exc}")
