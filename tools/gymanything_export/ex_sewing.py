# SPDX-License-Identifier: Apache-2.0
"""sewing: the muslin ply (cloth), the sewing table and the lockstitch head as rigid parts with their joints.

Cloth constants, layout dimensions and friction come from their ``params.py`` ledger, ``layout.py`` and ``scene.py``;
only those modules are imported (the cloth solver needs a GPU stack and is not touched).
"""

from __future__ import annotations

import math
from pathlib import Path

import trimesh
from _common import Exporter, _rel, box, cited, cylinder, dim, line_of, union, vis, where

from sewing import layout as LY
from sewing import params as P
from sewing import render as RN
from sewing import scene as SC

FAB = P.FABRICS["muslin_125"]
MUSLIN_LINE = '"muslin_125": dict('


def _srgb(lin) -> list[float]:
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]


def _pbr(geom, srgb255, rough: float, cite_pat: str, metal: float | None = None, name: str = "") -> dict:
    """A part with a ``A.pbr(name, (r, g, b), rough=, metal=)`` call of their ``render.py`` (sRGB 0-255)."""
    return vis(geom, srgb255, line_of(RN, cite_pat), rough, metal, name=name)


def _machine_look(geom, usd: str, material: str, name: str) -> dict:
    """Colour, roughness and metallic of a Principled material authored in their machine USD (linear diffuse)."""
    from pxr import Usd, UsdShade  # noqa: PLC0415 - their own USD reader, as in their render code

    path = Path(RN.MACHINE) / usd
    stage = Usd.Stage.Open(str(path))
    sh = next(UsdShade.Shader(p) for p in stage.Traverse()
              if p.GetPath().pathString.endswith(f"/_materials/{material}/Principled_BSDF"))
    vals = {i.GetBaseName(): i.Get() for i in sh.GetInputs()}
    cite = f"{_rel(path)}:{sh.GetPath().GetParentPath()}"
    return vis(geom, _srgb(tuple(vals["diffuseColor"])), cite, float(vals["roughness"]), float(vals["metallic"]),
               name=f"{name} (linear->sRGB)")


def _cl(value: float, basis: str, pattern: str, quote: str, module=P) -> dict:
    return cited(value, basis, line_of(module, pattern), quote)


def _ply_cloth(x: Exporter) -> None:
    w, length, free = 0.200, 0.420, FAB["thickness_free"]
    task = "sewing/task.py:" + str(next(i for i, ln in enumerate(open(_task_path()).read().splitlines(), 1)
                                        if '"plain-seam": dict' in ln))
    th = _cl(FAB["thickness"], "derived", "areal_density=0.1252", "Ramaiah KES: T0 0.610 mm, WC 0.181, LC 0.242 -> Tm "
             "0.311 mm (thickness of the cloth under the KES 50 gf/cm2 pressure)")
    k0 = "derived from KES EMT/LT (Ramaiah) with a 10 % toe-load fraction (estimate)"
    sh = _cl(FAB["G"], "literature-read", MUSLIN_LINE, "G 0.52 gf/cm deg (Ramaiah) converted to N/m per rad")
    bend = "read mean B 2.65e-6 N m split by Pal's warp/weft ratio 0.0609/0.0319 (estimate)"
    cloth = {"thickness": th, "stretch_warp": _cl(FAB["K0w"], "derived", MUSLIN_LINE, k0),
             "stretch_weft": _cl(FAB["K0f"], "derived", MUSLIN_LINE, k0), "shear": sh,
             "bend_warp": _cl(FAB["Bw"], "derived", MUSLIN_LINE, bend),
             "bend_weft": _cl(FAB["Bf"], "derived", MUSLIN_LINE, bend)}
    area_mass = FAB["areal_density"] * w * length
    x.add("fabric_sheet", "Muslin ply (unbleached cotton, 125 g/m2)", "cloth", box((w / 2, length / 2, free / 2)),
          kind="cloth", cloth=cloth, thickness=th,
          visual=[_pbr(box((w / 2, length / 2, free / 2)), RN.MUSLIN_SRGB, .85, "muslin = A.pbr(",
                       name="muslin (flat ecru colour; generated weave texture not exported)")], material={"name": "unbleached cotton plain weave (muslin)"},
          mass=_cl(area_mass, "derived", MUSLIN_LINE, "areal density 0.1252 kg/m2 (read, Ramaiah KES) x 0.200 x "
                   "0.420 m ply"),
          friction=_cl(P.CONTACT["table_mu"], "estimate", "table_mu=P(", "cloth on laminate, estimate 0.25-0.35"),
          friction_partner="the laminate table top",
          dims=[dim("ply width", w, "x", task), dim("ply length", length, "y", task),
                dim("free thickness", free, "z", line_of(P, "thickness_free=0.61e-3"))],
          source_ref=line_of(P, MUSLIN_LINE),
          extra={"areal_density_kg_m2": FAB["areal_density"], "stretch_after_toe_N_m": [FAB["K1w"], FAB["K1f"]],
                 "toe_strain": [FAB["etw"], FAB["etf"]], "friction_cloth_cloth": P.CONTACT["ply_ply_mu"],
                 "friction_cloth_steel": FAB["mu_steel"]},
          notes="Slab of the free thickness stands in for the shell their solver uses; stretch is the low-load toe "
                "stiffness (K0), the steeper stiffness after the toe is in extra.")


def _task_path() -> str:
    import inspect

    import sewing.task as T
    return inspect.getsourcefile(T)


def _table(x: Exporter) -> None:
    t = LY.TABLE
    (x0, x1), (y0, y1), th = t["x"], t["y"], t["thickness"]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    parts = [box((0.5 * (x1 - x0), 0.5 * (y1 - y0), th / 2), (0, 0, -th / 2))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            px = (x1 - .05 if sx > 0 else x0 + .05) - cx
            py = (y1 - .05 if sy > 0 else y0 + .05) - cy
            parts.append(box((.02, .02, (LY.TABLE_H - th) / 2), (px, py, -(LY.TABLE_H + th) / 2)))
    look = [_pbr(parts[0], (200, 180, 150), .45, 'A.pbr("table_top"',
                 name="wood-grain melamine top (fallback colour; veneer texture not exported)"),
            _pbr(union(parts[1:]), (70, 72, 76), .45, 'A.pbr("k_leg_steel"', .6, "steel legs")]
    x.add("sewing_table", "Sewing table (laminate top, four legs)", "furniture", union(parts), motion="static",
          visual=look,
          material={"name": "laminate top on steel legs"},
          friction=cited(SC.MU_TABLE, "estimate", line_of(SC, "MU_TABLE ="), "rigid-rigid (fingers on laminate/cast "
                         "iron) [estimate]"), friction_partner="robot fingers",
          dims=[dim("top length", P.TABLE_TOP[0], "x", line_of(P, "TABLE_TOP =")),
                dim("top depth", P.TABLE_TOP[1], "y", line_of(P, "TABLE_TOP =")),
                dim("height", P.TABLE_H, "z", line_of(P, "TABLE_H ="))],
          source_ref=where(SC.Scene.room),
          notes="Top slab 1.2 x 0.533 x 0.03 m plus four 40 mm legs (their scene boxes), unioned; origin at the top "
                "surface centre. The machine bed set flush in the top is not meshed here.")


def _foot_mesh() -> trimesh.Trimesh:
    length, back, thick = P.FOOT_LEN, P.FOOT_BACK, 3.0e-3
    yc = back - 0.5 * (length - 5e-3)
    parts = []
    for s, wt in ((-1, P.FOOT_W_LEFT), (1, P.FOOT_W_RIGHT)):
        parts.append(box((0.5 * wt, 0.5 * (length - 5e-3), thick / 2), (s * (0.5 * P.FOOT_SLOT + 0.5 * wt), yc, thick / 2)))
    k, b = LY.FOOT_KEEP_OUT, LY.PRESSER_BAR
    zs = (0.002, b["z"][0] + 1e-3)       # their shank starts at 3.5 mm; run it into the toes and the bar to be one solid
    parts.append(box((0.5 * (k["x"][1] - k["x"][0]), 0.5 * (k["y"][1] - k["y"][0]), 0.5 * (zs[1] - zs[0])),
                     (0.5 * sum(k["x"]), 0.5 * sum(k["y"]), 0.5 * sum(zs))))
    parts.append(box((0.5 * (b["x"][1] - b["x"][0]), 0.5 * (b["y"][1] - b["y"][0]), 0.5 * (b["z"][1] - b["z"][0])),
                     (0.5 * sum(b["x"]), 0.5 * sum(b["y"]), 0.5 * sum(b["z"]))))
    return union(parts)


def _span_box(d: dict) -> trimesh.Trimesh:
    half = [0.5 * (d[a][1] - d[a][0]) for a in "xyz"]
    return box(half, [0.5 * sum(d[a]) for a in "xyz"])


def _machine_head(x: Exporter) -> None:
    mesh = union([_span_box(LY.PILLAR), _span_box(LY.ARM), _span_box(LY.HEAD)])
    note = "estimate from the manufacturer's product photographs and the class's catalogue drawings"
    x.add("sewing_machine__head", "Lockstitch sewing head (pillar, arm, head casting)", "articulated", mesh,
          visual=[_machine_look(mesh, "machine_static.usd", "juki_white", "head casting")],
          motion="static", material={"name": "painted cast iron / aluminium casting"},
          friction=_cl(P.CASTING_MU, "estimate", "CASTING_MU =", "machine.casting_mu"), friction_partner="cloth",
          dims=[dim("height", LY.PILLAR["z"][1], "z", line_of(LY, "PILLAR =")),
                dim("needle to far pillar face, plus head", LY.PILLAR["x"][1] - LY.HEAD["x"][0], "x", line_of(LY, "PILLAR ="))],
          source_ref=where(SC.Scene.machine), extra={"frame": "origin at needle axis on the throat plate top"},
          notes=f"Pillar, arm and head boxes unioned (their static colliders, {note}); the base part of the "
                "articulated machine; no mass given in their code.")


def _moving_parts(x: Exporter) -> None:
    cite_l = line_of(LY, "NEEDLE_BAR =")
    r, zb = LY.NEEDLE_BAR["radius"], LY.NEEDLE_BAR["z_bottom_up"]
    bar = box((r, r, 0.5 * (LY.HEAD["z"][0] + 1e-3 - zb)), (0, 0, 0.5 * (LY.HEAD["z"][0] + 1e-3 + zb)))
    x.add("sewing_machine__needle_bar", "Needle bar", "articulated", bar,
          visual=[_machine_look(bar, "needle_bar.usd", "chrome", "needle bar")],
          material={"name": "steel"}, source_ref=where(SC.Scene.machine),
          dims=[dim("width", 2 * r, "x", cite_l)],
          extra={"joint": {"type": "prismatic", "parent": "sewing_machine__head", "axis": [0, 0, 1],
                           "range": [-P.STROKE, 0.0],
                           "cite": line_of(P, "STROKE = P(\"machine.needle_bar_stroke\"")}},
          notes="Square 8 mm bar as their static keep-out collider (the needle rides the slider-crank in their cloth "
                "solver, stroke 30.7 mm); no mass in their code.")
    foot = _foot_mesh()
    x.add("sewing_machine__presser_bar", "Presser bar with foot", "articulated", foot,
          visual=[_machine_look(foot, "presser_foot.usd", "chrome", "presser bar and foot")],
          material={"name": "steel"},
          mass=cited(P.PRESSER_MASS, "estimate", line_of(P, "PRESSER_MASS ="), "presser bar + foot (open)"),
          friction=cited(FAB["mu_steel"], "literature-read", line_of(P, "friction.cloth_steel"),
                         "cotton-steel 0.10-0.21 (tribology.rs 2021); KES MIU"), friction_partner="cotton cloth",
          dims=[dim("bar top", LY.PRESSER_BAR["z"][1], "z", line_of(LY, "PRESSER_BAR ="))],
          source_ref=where(SC.Scene.machine),
          extra={"joint": {"type": "prismatic", "parent": "sewing_machine__head", "axis": [0, 0, 1],
                           "range": [0.0, P.FOOT_LIFT_AUTO2], "cite": line_of(P, "FOOT_LIFT_AUTO2 =")}},
          notes="Bar box plus the two 3 mm toes (foot sole sizes from their params) and the shank, unioned; the "
                "shank is run 1.5 mm into the toes and the bar so the parts form one solid. Lift 0 / 5.0 / 8.5 mm.")
    h = LY.HANDWHEEL
    wheel = cylinder(h["radius"], 0.5 * h["width"], h["center"], (0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)))
    x.add("sewing_machine__handwheel", "Handwheel", "articulated", wheel,
          visual=[_machine_look(wheel, "handwheel.usd", "panel_black", "handwheel rim")], material={"name": "cast iron"},
          dims=[dim("diameter", 2 * h["radius"], "max", line_of(LY, "HANDWHEEL =")),
                dim("width", h["width"], "x", line_of(LY, "HANDWHEEL ="))],
          source_ref=where(SC.Scene.machine),
          extra={"joint": {"type": "revolute", "parent": "sewing_machine__head", "axis": [1, 0, 0], "range": None,
                           "cite": line_of(LY, "HANDWHEEL =")}},
          notes="Cylinder of their envelope (they collide it as a box); continuous rotation about x; no mass given.")


def export(x: Exporter) -> None:
    for fn in (_ply_cloth, _table, _machine_head, _moving_parts):
        try:
            fn(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
