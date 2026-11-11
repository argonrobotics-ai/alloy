# SPDX-License-Identifier: Apache-2.0
"""filleting: the fillet knife, bench vessels (GN pans, bowl), board, knife block, finger inserts, tweezers and the
three workpieces (trout, salmon half side, pork tenderloin).

Knife and pans use their ``dressing`` geometry; the blade is re-lofted here because their blade/bevel meshes are open
shells. Workpiece surfaces are meshed from their signed-distance functions at 3 mm (soft solids).
"""

from __future__ import annotations

import logging
import math
import tempfile

import numpy as np
import trimesh
from _common import Exporter, _revolve_polygon, box, cited, dim, line_of, mesh, union, vis, where

from filleting import dressing as D
from filleting import layout as LY
from filleting import params as PR
from filleting import render as RN
from filleting import scene as SC

log = logging.getLogger("gar.export")


def _p(name: str, scale: float = 1.0, basis: str | None = None, extra: str = "", value: float | None = None) -> dict:
    """A ``params.py`` entry as a cited number; their label (read/estimate/open/...) maps to a ledger basis."""
    ent = getattr(PR, name)
    label = ent.label.split("+")[0].split("(")[0]
    mapped = basis or {"read": "literature-read", "open": "estimate", "repo": "estimate", "prior": "estimate",
                       "abstract": "literature-read", "search": "search", "literature": "literature",
                       "derived": "derived", "measured": "measured", "authored": "authored"}.get(label, "estimate")
    return cited(float(ent.value) * scale if value is None else value, mapped, line_of(PR, f"{name} = P("), (ent.basis or ent.label)[:300] + extra)


def _steel(geom, pattern: str, rough: float, srgb=(195, 197, 198), label: str = "stainless") -> dict:
    """Brushed stainless as their ``Dressing.steel`` / ``omnipbr`` call sets it (sRGB colour, metallic 1)."""
    return vis(geom, srgb, line_of(D, pattern), rough, 1.0, name=label)


def _loft_solid(rings: list[np.ndarray]) -> trimesh.Trimesh:
    """Closed solid from consecutive same-size rings with fan caps at both ends."""
    V, F = D.loft(rings, cap0=True, cap1=True)
    return mesh(V, F)


def _blade_mesh() -> trimesh.Trimesh:
    from filleting.knife import EDGE_HALF_SHOULDER, BladeSpec  # noqa: PLC0415
    spec = BladeSpec("fillet")
    e, H = EDGE_HALF_SHOULDER, spec.H
    hb = e / math.tan(spec.bevel)
    rings = []
    for x in np.linspace(0, spec.L, 60):
        s = spec.spine_at(x / spec.L) / 2
        rings.append([(x, e, hb), (x, s, H), (x, -s, H), (x, -e, hb), (x, 0., 0.)])
    return _loft_solid(rings)


def _knife(x: Exporter) -> None:
    g = D.knife_geometry("fillet")
    blade, handle = _blade_mesh(), mesh(*g["handle"])
    d = PR.FILLET_BLADE.value
    h = PR.HANDLE.value
    ref = line_of(PR, "FILLET_BLADE = P(")
    looks = [vis(blade, (196, 198, 199), line_of(D, 'blade=d.omnipbr("knife_blade"'), .30, 1.0, name="blade steel"),
             vis(handle, (24, 25, 26), line_of(D, 'handle=d.preview("fibrox_black"'), .62, 0.0, name="Fibrox handle")]
    x.add("knife", "Fillet knife (Victorinox Fibrox 5.3703.18)", "tool", trimesh.util.concatenate([blade, handle]),
          visual=looks,
          material={"name": "stainless blade steel, TPE handle"},
          mass=cited(d["mass"], "literature-read", ref, PR.FILLET_BLADE.basis),
          friction=_p("MU_BLADE_TISSUE"), friction_partner="wet tissue (blade)",
          dims=[dim("blade + handle length", d["length"] + h["length"], "x", ref),
                dim("handle width", h["width"], "y", line_of(PR, "HANDLE = P(")),
                ],
          source_ref=where(D.knife_geometry),
          notes="Blade re-lofted from their flat-grind/bevel section and spine taper (their bevel mesh is an open "
                "shell); handle is their lofted mesh. Origin at the heel on the cutting edge, x to the tip. In their "
                "sim the blade is five hinged segments; here it is one rigid body.",
          extra={"blade_segments": 5, "spine_thickness_m": d["spine"], "bevel_deg": d["bevel_deg"]})


def _pan(x: Exporter, key: str, name: str, size: tuple, depth: float, spec_ref: str, notes: str) -> None:
    V, F, _ = D.gn_pan_geometry(size, depth)
    geom = mesh(V, F)
    x.add(key, name, "container", geom, visual=[_steel(geom, 'd.steel("pan_steel", .22)', .22)], motion="static",
          material={"name": "stainless steel"},
          friction=cited(0.45, "authored", line_of(SC, "mu=.45"), "default shape friction in their scene"),
          friction_partner="food and tools",
          dims=[dim("length", size[0], "x", spec_ref), dim("width", size[1], "y", spec_ref),
                dim("depth", depth, "z", spec_ref)], source_ref=where(D.gn_pan_geometry), notes=notes)


def _bowl(x: Exporter) -> None:
    from shapely.geometry import Polygon  # noqa: PLC0415
    from shapely.geometry.polygon import orient  # noqa: PLC0415
    W = LY.BOWL
    R, depth, t, rb = W["diameter"] / 2, W["depth"], .0007, .065
    outer = []
    for s in np.linspace(0, 1, 24):
        a = s * math.pi / 2
        outer.append((rb + (R - .003 - rb) * math.sin(a) ** .9, (depth - .003) * (1 - math.cos(a)) ** 1.05))
    bead = [(R - .003 + .003 * math.sin(a), depth - .003 + .003 * (1 - math.cos(a))) for a in np.linspace(0, math.pi, 9)]
    poly = [(0., t), (rb * .6, t)] + [(r - t, z + t) for r, z in outer[1:]] + bead[::-1] + \
        [(r, z) for r, z in outer[::-1][1:]] + [(rb * .6, 0.), (0., 0.)]
    ref = line_of(LY, "BOWL = dict")
    shell = _revolve_polygon(orient(Polygon(poly).simplify(1e-6)), 96)
    x.add("bowl", "Stainless mixing bowl 240 mm", "container", shell, motion="static",
          visual=[vis(shell, (196, 198, 200), line_of(D, 'd.steel("bowl_steel", .12'), .12, 1.0, name="bowl steel")],
          material={"name": "stainless steel"},
          friction=cited(0.45, "authored", line_of(SC, "mu=.45"), "default shape friction in their scene"),
          friction_partner="food", dims=[dim("rim diameter", W["diameter"], "diameter_xy", ref),
                                         dim("depth", depth, "z", ref)],
          source_ref=where(D.bowl_geometry),
          notes="Wall revolved from their dressing.bowl_geometry profile (their lathe mesh is an open shell); in "
                "their physics scene the bowl is 16 wall slabs on a floor.")


def _board_look(slab: trimesh.Trimesh, sx: float, sy: float) -> dict:
    """Their blue (fish) board: the generated scored-HDPE texture, spanning the top face once (``build_board``)."""
    from PIL import Image  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        path = D.make_textures(tmp, "A")["board_blue"]
        img = Image.open(path).convert("RGB")
        img.load()
    uv = np.c_[slab.vertices[:, 0] / sx + .5, slab.vertices[:, 1] / sy + .5]
    return vis(slab, (38, 92, 168), line_of(D, '("board_blue", (38, 92, 168))'), .55, 0.0, image=img, uv=uv,
               name="blue HDPE board (generated texture; red variant for meat tasks)")


def _board(x: Exporter) -> None:
    sx, sy, sz = LY.BOARD["size"]
    ref = line_of(LY, "BOARD = dict")
    slab = box((sx / 2, sy / 2, sz / 2))
    x.add("board", "Cutting board (NSF HDPE)", "fixture", slab, motion="static", visual=[_board_look(slab, sx, sy)],
          material={"name": "HDPE"}, friction=_p("MU_FLESH_HDPE"), friction_partner="wet fish or pork flesh",
          dims=[dim("length", sx, "x", ref), dim("width", sy, "y", ref), dim("thickness", sz, "z", ref)],
          source_ref=ref, notes="610 x 457 x 12.7 mm slab on a damp cloth (static in their scene).")


def _block(x: Exporter) -> None:
    from shapely.geometry import Polygon  # noqa: PLC0415
    K = LY.KNIFE_BLOCK
    sx, sy, sz = K["size"]
    c = D.BLOCK_CHAMFER
    poly = [(-sy / 2, 0.), (sy / 2, 0.), (sy / 2, sz), (-sy / 2 + c, sz), (-sy / 2, sz - c)]
    m = trimesh.creation.extrude_polygon(Polygon(poly), sx)        # (y, z) profile, extruded along local z
    m.apply_transform(np.array([[0, 0, 1, -sx / 2], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.]]))
    ref = line_of(LY, "KNIFE_BLOCK = dict")
    x.add("knife_block", "Knife block 100 x 160 x 200 mm", "fixture", m, motion="static",
          visual=[vis(m, (70, 72, 74), line_of(D, 'pp = d.preview("block_pp"'), .55, 0.0, name="grey polypropylene")],
          material={"name": "wood"}, friction=cited(0.45, "authored", line_of(SC, "mu=.45"), "default shape friction"),
          friction_partner="knives",
          dims=[dim("width", sx, "x", ref), dim("depth", sy, "y", ref), dim("height", sz, "z", ref)],
          source_ref=where(SC.Scene.knife_block),
          notes="Solid block with the 40 mm front chamfer; their blade slots (3.6 mm gaps) are not cut.")


def _inserts(x: Exporter) -> None:
    I, HH = SC.INSERT, SC.HANDLE_H
    tip_face_y = 7.58e-3 - 15.2e-3 / 2
    yb, yl = tip_face_y - I["back"] / 2, tip_face_y - I["back"] - I["lip"] / 2
    lh = I["lip_height"]
    parts = [box((I["length"] / 2, I["back"] / 2, HH / 2 + .0005 + lh), (0, yb, 0)),
             box((I["length"] / 2, I["lip"] / 2, lh / 2), (0, yl, HH / 2 + .0005 + lh / 2)),
             box((I["length"] / 2, I["lip"] / 2, lh / 2), (0, yl, -HH / 2 - .0005 - lh / 2))]
    ref = line_of(SC, "INSERT = dict")
    whole = union(parts)
    x.add("handle_insert", "Franka finger knife-handle insert (TPU)", "fixture", whole,
          visual=[vis(whole, (48, 50, 54), line_of(RN, '"insert": lambda'), .6, 0.0, name="TPU insert")],
          material={"name": "TPU with 2 mm liner"},
          mass=cited(I["mass"], "authored", ref, "Each insert ~25 g (authored hardware)"),
          friction=cited(I["mu"], "authored", ref, "2 mm TPU liner (mu 0.8, the repo's finger pad friction)"),
          friction_partner="knife handle",
          dims=[dim("length", I["length"], "x", ref)], source_ref=where(SC.Scene._insert),
          notes="Back plate and two lips wrapping the 32 mm knife handle; channel 32 mm + 2 x 0.5 mm clearance.")


def _tweezers(x: Exporter) -> None:
    ref = line_of(D, "def tweezers_geometry")
    tw = mesh(*D.tweezers_geometry())
    x.add("tweezers", "Fish-bone tweezers (Victorinox 7.6120 class)", "tool", tw,
          visual=[_steel(tw, 'd.steel("tweezer_steel", .18)', .18)],
          material={"name": "stainless steel"}, friction=_p("MU_TWEEZER_BONE"), friction_partner="fish bone",
          dims=[dim("length", .12, "x", ref)], source_ref=where(D.tweezers_geometry),
          notes="Two 1.2 mm arms joined at the back; dressing geometry (their scene uses a box collider). No mass "
                "given.")


def _level_set(wp, edge: float = 0.003) -> trimesh.Trimesh:
    """Watertight surface of ``wp.sdf < 0`` (manifold3d level set; marched meshes of it come out non-manifold)."""
    import manifold3d as mf  # noqa: PLC0415
    lo, hi = np.asarray(wp.bounds[0], float) - 0.002, np.asarray(wp.bounds[1], float) + 0.002
    inside = lambda x, y, z: -float(wp.sdf(np.array([[x, y, z]], float))[0])  # noqa: E731
    out = mf.Manifold.level_set(inside, [*lo, *hi], edge).to_mesh()
    return mesh(out.vert_properties[:, :3], out.tri_verts)


def _flesh_look(m: trimesh.Trimesh, wp) -> dict:
    """Mean of their anatomy's per-vertex flesh colour (``tissue_appearance``, sRGB) over the surface; roughness is the
    flesh material's .30 (``tissue_look``). The per-vertex colour field and skin texture are not exported."""
    col = np.asarray(RN.tissue_appearance(wp, m.vertices)["flesh_color"], float)
    return vis(m, col.mean(axis=0), where(RN.tissue_appearance), float(RN.TISSUE_LOOK["C"]["flesh"]["roughness"]),
               0.0, name="flesh (mean of the per-vertex anatomy colour)")


def _workpiece(x: Exporter, kind: str, key: str, name: str, mass: dict, dims: list, notes: str, nu: dict) -> None:
    from filleting.anatomy import load  # noqa: PLC0415
    wp = load(kind)
    m = _level_set(wp)
    flesh = _flesh_look(m, wp)
    m.apply_translation(-m.bounds[0] * np.array([0, 0, 1.]))
    x.add(key, name, "food", m, kind="soft", visual=[flesh],
          material={"name": "raw " + ("pork muscle" if kind == "tenderloin" else "fish flesh"),
                    "density": _p("PORK_DENSITY" if kind == "tenderloin" else "FISH_DENSITY"),
                    "youngs": _p("PORK_E" if kind == "tenderloin" else "FISH_E"), "poisson": nu},
          mass=mass, friction=_p("MU_FLESH_HDPE"), friction_partner="HDPE board", dims=dims,
          source_ref=where(load), notes=notes)


def _workpieces(x: Exporter) -> None:
    nu = _p("FISH_NU", extra=" (their only Poisson value; porcine nu_LT 0.47 read, fish unmeasured)")
    T, S, P_ = PR.TENDERLOIN.value, PR.SALMON_SIDE.value, PR.TROUT.value
    soft = "Level-set surface of their signed-distance function at 3 mm; flesh only (bones, skin and fins are separate parts of their sim)."
    _workpiece(x, "trout", "trout_gutted", "Trout, gutted (farmed, 31.5 cm)", _p("TROUT", value=P_["gutted_kg"]),
               [], soft + " Flesh only: its 234 mm length is shorter than the 315 mm whole fish (head, fins and tail are separate).", nu)
    _workpiece(x, "salmon", "salmon_half_side", "Salmon half side (front half of a side)",
               _p("SALMON_SIDE", value=S["kg"]), [dim("length", S["length"], "x", line_of(PR, "SALMON_SIDE = P("))],
               soft, nu)
    _workpiece(x, "tenderloin", "pork_tenderloin", "Pork tenderloin (untrimmed)", _p("TENDERLOIN", value=T["kg"]),
               [dim("length", T["length"], "x", line_of(PR, "TENDERLOIN = P("))], soft, nu)


def export(x: Exporter) -> None:
    steps = (_knife, _bowl, _board, _block, _inserts, _tweezers,
             lambda e: _pan(e, "fillet_tray", "Perforated GN 1/2 fillet tray", LY.FILLET_TRAY["size"][:2],
                            LY.FILLET_TRAY["size"][2], line_of(LY, "FILLET_TRAY = dict"),
                            "Solid GN pan shell (their dressing geometry); perforations not modelled."),
             lambda e: _pan(e, "ice_pan", "GN 1/1 ice pan", LY.ICE_PAN["size"][:2], LY.ICE_PAN["size"][2],
                            line_of(LY, "ICE_PAN = dict"), "GN 1/1 pan shell (EN 631); the ice fill is not part of it."),
             lambda e: _pan(e, "frame_pan", "GN 1/2 frame pan, 100 mm", LY.FRAME_PAN["size"][:2],
                            LY.FRAME_PAN["size"][2], line_of(LY, "FRAME_PAN = dict"), "Solid GN pan for frame and head."),
             _workpieces)
    for i, step in enumerate(steps):
        try:
            step(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(f"step_{i}", f"{type(exc).__name__}: {exc}")
