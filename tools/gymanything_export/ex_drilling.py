# SPDX-License-Identifier: Apache-2.0
"""drilling: hand tools, drill press, vise, fasteners, shop vacuum, workpieces and the concrete wall.

Their tools are Newton links built by ``drilling.tools`` / ``drilling.parts`` / ``drilling.vacuum`` into a scene
builder. This exporter runs those same constructors on a bare builder (no robots, no GPU, no solver), then writes one
spec per rigid link: the union of the link's collision shapes in the link's own frame, the link's mass as their
densities give it, and the joint that holds it to its parent (``extra.joint``). The tool's ledger totals (maker's
weights) are quoted from ``params.py``; links are never meshed from render-only shapes.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from alloy._vendor import newton
import warp as wp
from scipy.spatial.transform import Rotation

from _common import Exporter, box, cited, cylinder, line_of, mesh, revolve, sphere, transformed, union, vis, where

from drilling import params as P
from drilling import parts as PT
from drilling import render as RN
from drilling import scene as S
from drilling import stations as ST
from drilling import tools as T
from drilling import vacuum as V
from drilling import visual_models as VM

wp.config.quiet = True
log = logging.getLogger("gar.export.drilling")
COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)
GT = newton.GeoType
BASIS = {"read": "literature-read", "derived": "derived", "estimate": "estimate", "open": "estimate",
         "authored": "authored", "repo": "authored", "literature": "literature", "measured": "measured",
         "engine": "authored"}
IDENT = (np.zeros(3), Rotation.identity())


class _Bare(S.Scene):
    """Their ``Scene`` with only the builder and its registries (no room, robots or cameras)."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips Scene.__init__
        self.newton, self.wp = newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0.0, 0.0, -P.GRAVITY))
        from alloy._vendor.newton.solvers import SolverMuJoCo  # noqa: PLC0415
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.visual, self.layout = {}, set(), {}
        self.bodies, self.parts = {}, {}
        self.static_boxes, self.static_cyls, self.filter_groups, self.filter_pairs = [], [], [], []
        self.tool_bodies, self.tools, self.drillables, self.items, self.item_bodies = set(), {}, {}, {}, set()
        self.stiff_shapes, self.stiff_solids, self.clamp_shapes = set(), set(), set()
        self.bearing_shapes, self.stack_shapes, self.loose_shapes = set(), set(), set()


# ------------------------------------------------------------------------------------------------ ledger
def ledger(name: str) -> dict:
    """Their ``_p`` ledger row as a cited number (value, their label, params.py line, their basis text)."""
    row = next(r for r in P.LEDGER if r["name"] == name)
    return cited(row["value"], BASIS[row["label"]], line_of(P, f'_p("{name}"'), " ".join(str(row["basis"]).split())[:400])


def mu_cited(key: str) -> dict:
    """Their friction table entry ``P.MU[key]``, cited at the table and quoting its ledger text."""
    row = next(r for r in P.LEDGER if r["name"] == "MU")
    return cited(P.MU[key], "estimate", line_of(P, "MU = dict("), " ".join(row["basis"].split())[:300])


# ------------------------------------------------------------------------------------------------ geometry
def _capsule(radius: float, half_height: float, n_arc: int = 12) -> trimesh.Trimesh:
    """Watertight capsule along z (trimesh's own has an open seam): a revolved cylinder with hemispherical ends."""
    a = np.linspace(0.0, 0.5 * np.pi, n_arc)
    top = [(radius * np.cos(t), half_height + radius * np.sin(t)) for t in a]
    bottom = [(r, -z) for r, z in reversed(top)]
    return revolve(np.array([(0.0, -half_height - radius), *bottom[1:], *top[:-1], (0.0, half_height + radius)]))


def shape_geom(b, s: int) -> trimesh.Trimesh:
    """One Newton shape (box, cylinder, capsule, sphere, convex hull) as a mesh in its link's frame."""
    t, sc_ = b.shape_type[s], np.asarray(b.shape_scale[s], float)
    tf = np.asarray(b.shape_transform[s], float)
    pos, quat = tf[:3], tf[3:7]
    if t == GT.BOX:
        return box(sc_, pos, quat)
    if t == GT.CYLINDER:
        return cylinder(sc_[0], sc_[1], pos, quat)
    if t == GT.CAPSULE:
        return transformed(_capsule(sc_[0], sc_[1]), pos, quat)
    if t == GT.SPHERE:
        return transformed(sphere(sc_[0]), pos, quat)
    if t == GT.CONVEX_MESH:
        src = b.shape_source[s]
        return transformed(mesh(np.asarray(src.vertices, float) * sc_, np.asarray(src.indices).reshape(-1, 3)), pos, quat)
    raise ValueError(f"unsupported shape type {t}")


SOLID_VISUALS = ("wrench_socket_barrel",)   # drawn-only shapes that are the part's real body (the socket's barrel)


def body_shapes(b, body: int) -> list[int]:
    """Collision shapes of a link, plus the drawn-only shapes listed in ``SOLID_VISUALS``."""
    return [s for s in range(b.shape_count) if int(b.shape_body[s]) == body and b.shape_type[s] != GT.PLANE
            and (int(b.shape_flags[s]) & COLLIDE or b.shape_label[s] in SOLID_VISUALS)]


SOLID_VISUALS = ("wrench_socket_barrel",)   # drawn-only shapes that are the part's real body (the socket's barrel)


def body_shapes(b, body: int) -> list[int]:
    """Collision shapes of a link, plus the drawn-only shapes listed in ``SOLID_VISUALS``."""
    return [s for s in range(b.shape_count) if int(b.shape_body[s]) == body and b.shape_type[s] != GT.PLANE
            and (int(b.shape_flags[s]) & COLLIDE or b.shape_label[s] in SOLID_VISUALS)]


def main_shape(b, body: int) -> int:
    """The link's largest collision shape (its material and friction stand for the link)."""
    return max(body_shapes(b, body), key=lambda s: abs(float(shape_geom(b, s).volume)))


def tidy(m: trimesh.Trimesh, grid: float | None) -> trimesh.Trimesh:
    """Snap vertices to ``grid`` (m; None = leave) and drop the degenerate and duplicate faces that leaves."""
    v = m.vertices if grid is None else np.round(m.vertices / grid) * grid
    out = trimesh.Trimesh(np.round(v, 8), m.faces, process=False)
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(out)
    return out


def importer_closed(m: trimesh.Trimesh, tol: float = 1e-7) -> bool:
    """The importer's own closedness test: weld at ``tol``, drop degenerate/duplicate faces, then every edge shared by
    exactly two faces with no directed edge used twice (alloy.cards.geometry.is_closed)."""
    keys = np.round(m.vertices / tol).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    f = inv.reshape(-1)[m.faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    f = f[np.sort(np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1])]
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    _, dcounts = np.unique(directed, axis=0, return_counts=True)
    return len(f) > 0 and bool((counts == 2).all()) and not bool((dcounts > 1).any())


def _swell(m: trimesh.Trimesh, factor: float) -> trimesh.Trimesh:
    """A shape scaled about its centre: shapes that only touch (a shared face or edge) then overlap, so that the
    boolean union is one manifold surface instead of one pinched along the contact edge."""
    out = m.copy()
    c = out.bounds.mean(axis=0)
    out.vertices = (out.vertices - c) * factor + c
    return out


DROPPED: dict[str, list[str]] = {}
THIN_SHAPE_M = 1.5e-3


def _closed_union(shapes: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    """First closed union over a few swell factors and vertex grids, or None."""
    for factor in ((1.0,) if len(shapes) == 1 else (1.004, 1.0, 1.01, 1.03)):
        raw = union([_swell(m, factor) for m in shapes]) if len(shapes) > 1 else shapes[0]
        for grid in (None, 1e-7, 5e-7, 2e-6, 1e-5):
            out = tidy(raw, grid)
            if importer_closed(out):
                return out
    return None


def body_geom(b, body: int) -> trimesh.Trimesh | None:
    """Union of the link's collision shapes as a closed surface the importer accepts. When no union closes (slivers
    between very thin lead-in shapes), shapes thinner than 1.5 mm are left out and listed in ``DROPPED``."""
    ids = body_shapes(b, body)
    if not ids:
        return None
    shapes = [shape_geom(b, s) for s in ids]
    out = _closed_union(shapes)
    if out is not None:
        return out
    keep = [s for s, m in zip(ids, shapes) if m.extents.min() > THIN_SHAPE_M]
    DROPPED[b.body_label[body]] = [b.shape_label[s] for s in ids if s not in keep]
    out = _closed_union([shape_geom(b, s) for s in keep]) if keep else None
    if out is None:
        log.warning("body %s: no closed union found", b.body_label[body])
        out = tidy(union(shapes), None)
    return out


def joint_info(b, body: int, cite: str) -> dict | None:
    """The joint whose child is ``body`` (type, parent link, axis, limits) or None for a free or world-fixed root."""
    for j in range(b.joint_count):
        if int(b.joint_child[j]) != body:
            continue
        kind = newton.JointType(int(b.joint_type[j])).name.lower()
        parent = int(b.joint_parent[j])
        if kind == "free" or parent < 0:
            return None
        d = int(b.joint_qd_start[j])
        info = {"type": kind, "parent": b.body_label[parent], "cite": cite}
        if kind in ("revolute", "prismatic"):
            info.update(axis=[float(v) for v in b.joint_axis[d]],
                        range=[float(b.joint_limit_lower[d]), float(b.joint_limit_upper[d])])
        return info
    return None


# ------------------------------------------------------------------------------------------------ render looks
HIDDEN_LOOKS = {"press_hidden", "collision_hidden", "proxy_hidden"}


def _srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _look_cite(name: str) -> str:
    return line_of(RN, f" {name}=((")


def _rep_material(sc: _Bare, s: int) -> str | None:
    """Dominant material (by drawn area) of the replacement model their renderer draws for shape ``s``, if any."""
    b = sc.b
    rep = VM.shape_visual(sc, s, b.shape_label[s], sc.materials.get(s), np.asarray(b.shape_scale[s], float),
                          np.asarray(b.shape_transform[s], float))
    area: dict[str, float] = {}
    for part in rep or []:
        v, f = np.asarray(part[1], float), np.asarray(part[2])
        tri = v[f]
        area[part[3]] = area.get(part[3], 0.0) + 0.5 * float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    return max(area, key=area.get) if area else None


def shape_look(sc: _Bare, s: int) -> str | None:
    """Name of the render material of shape ``s``: its collision material, or, for the hidden collision proxies, the
    material of the model their renderer draws in its place."""
    name = sc.materials.get(s)
    if name in HIDDEN_LOOKS or name not in RN.COLORS:
        name = _rep_material(sc, s) or name
    return name if name in RN.COLORS else None


def link_visual(sc: _Bare, body: int) -> list[dict] | None:
    """One render part per distinct material among the link's collision shapes (colour, roughness, metallic from their
    ``render.COLORS``, linear -> sRGB); None when no shape has a colour in their table."""
    groups: dict[str, list] = {}
    for s in body_shapes(sc.b, body):
        name = shape_look(sc, s)
        if name:
            groups.setdefault(name, []).append(shape_geom(sc.b, s))
    parts = []
    for name, geoms in groups.items():
        rgb, rough, metal = RN.COLORS[name]
        parts.append(vis(union(geoms) if len(geoms) > 1 else geoms[0], [_srgb(c) for c in rgb], _look_cite(name),
                         roughness=float(rough), metallic=float(metal), name=f"{name} (linear->sRGB)"))
    return parts or None


def fallback_visual(sc: _Bare, body: int) -> list[dict]:
    """A link none of whose shapes has a drawn colour (hidden proxies, parts of a downloaded model): one part in the
    colour of its workpiece's or shape's material in their ``render.COLORS``, else their default steel."""
    label = sc.b.body_label[body]
    part = next((d.get("material") for k, d in sc.parts.items() if d.get("body") == body or label.startswith(k + "_")),
                None)
    mat = part if part in RN.COLORS else sc.materials.get(main_shape(sc.b, body))
    name = mat if mat in RN.COLORS else "steel"
    rgb, rough, metal = RN.COLORS[name]
    return [vis(body_geom(sc.b, body), [_srgb(c) for c in rgb], _look_cite(name), roughness=float(rough),
                metallic=float(metal), name=f"{name} (the link's material; their drawn model is not exported; "
                                            f"linear->sRGB)")]


# ------------------------------------------------------------------------------------------------ links
def export_links(x: Exporter, sc: _Bare, prefix: str, key: str, title: str, cls: object, ledger_names: list[str],
                 skip: tuple[str, ...] = (), category: str = "articulated", motion: str = "dynamic") -> None:
    """One spec ``<key>__<part>`` per link of the tool just built whose label is ``<prefix>_<part>``."""
    b = sc.b
    quote = " | ".join(f"{n}: {ledger(n)['quote'][:160]}" for n in ledger_names)
    for body in range(b.body_count):
        label = b.body_label[body]
        if not (label == prefix or label.startswith(prefix + "_")) or label.endswith(skip) or not body_shapes(b, body):
            continue
        part = label[len(prefix) + 1:] or "body"
        try:
            _link(x, sc, body, f"{key}__{part}", f"{title}: {part.replace('_', ' ')}", cls, quote, category, motion)
        except Exception as exc:  # noqa: BLE001 - one bad link must not stop the tool
            x.fail(f"{key}__{part}", f"{type(exc).__name__}: {exc}")


def _link(x: Exporter, sc: _Bare, body: int, key: str, name: str, cls: object, quote: str, category: str,
          motion: str) -> None:
    b = sc.b
    main = main_shape(b, body)
    cite = where(cls)
    q = np.asarray(b.body_q[body], float)
    joint = joint_info(b, body, cite)
    x.add(key, name, category, body_geom(b, body), motion=motion,
          visual=link_visual(sc, body) or fallback_visual(sc, body),
          material={"name": sc.materials.get(main, "steel").replace("_", " ")},
          mass=cited(float(b.body_mass[body]), "derived", cite, quote),
          friction=mu_cited_value(float(b.shape_material_mu[main])), friction_partner="any surface (their default pair rule)",
          source_ref=cite,
          notes="Union of the link's collision shapes in the link frame; mass from their shape densities scaled to the "
                "maker's weight in the ledger. Render-only shapes are not meshed."
                + (f" Left out as thinner than 1.5 mm (the union would not close): {DROPPED[b.body_label[body]]}."
                   if b.body_label[body] in DROPPED else ""),
          extra={"joint": joint, "pose_in_tool_xyz_xyzw": [float(v) for v in q],
                 "tool_ledger": quote[:600]})


def mu_cited_value(mu: float) -> dict:
    """Friction read off the built shape; their table is ``P.MU`` and every row of it is labelled an estimate."""
    key = next((k for k, v in P.MU.items() if abs(v - mu) < 1e-9), "default")
    return mu_cited(key) | {"value": mu}


TOOLS = [
    ("cordless_drill", "Cordless drill/driver (DeWalt DCD800 class)", T.CordlessDrill, "drill", ["DRILL_BARE_KG", "BATTERY_2AH_KG"],
     lambda sc: sc.add_tool(T.CordlessDrill(sc, "drill", IDENT, 6.0, 0.1))),
    ("rotary_hammer", "SDS-plus rotary hammer (DeWalt DCH273 class)", T.RotaryHammer, "hammer",
     ["HAMMER_BARE_KG", "BATTERY_5AH_KG"], lambda sc: sc.add_tool(T.RotaryHammer(sc, "hammer", IDENT, 10.0, 0.1))),
    ("vise", "4 in drill-press vise", T.Vise, "vise", ["VISE_PRECLAMP_NM"], lambda sc: sc.add_tool(T.Vise(sc, "vise", IDENT, 0.1))),
    ("centre_punch", "Automatic centre punch (Rennsteig 430 130)", T.CentrePunch, "punch", ["PUNCH_MASS_KG"],
     lambda sc: sc.add_tool(T.CentrePunch(sc, "punch", IDENT))),
    ("impact_wrench", "Impact wrench (Hilti SIW 6AT-A22 class) with socket", T.ImpactWrench, "wrench",
     ["IMPACT_TOOL_KG", "IMPACT_BATTERY_KG", "AT_MODULE_KG", "SOCKET_KG"], lambda sc: sc.add_tool(T.ImpactWrench(sc, "wrench", IDENT))),
    ("drill_press", "Drill press (Grizzly G7943 class)", T.DrillPress, "press", ["PRESS_QUILL_KG", "PRESS_CHUCK_KG"],
     lambda sc: sc.add_tool(T.DrillPress(sc, "press", (0, 0), 0.0, 1000, 6.0, 0.1))),
]


def _tool(x: Exporter, row: tuple) -> None:
    key, title, cls, prefix, names, make = row
    sc = _Bare()
    make(sc)
    skip = ("press_quill",) if key == "drill_press" else ()
    export_links(x, sc, prefix, key, title, cls, names, skip=skip)
    if key == "drill_press":
        x.fail("drill_press__quill", "their quill is two thin tube shapes carrying a 1.4 kg ledger mass (PRESS_QUILL_KG "
               "spindle/chuck mass set on the link), so mass over the modelled volume is 97,000 kg/m^3: no honest solid")
        _press_frame(x, sc)


def _frame_visual(sc: _Bare, ids: list[int]) -> list[dict] | None:
    """Render parts of the press frame's world-body shapes, grouped by material like ``link_visual``."""
    groups: dict[str, list] = {}
    for s in ids:
        name = shape_look(sc, s) or "steel*"          # drawn by a downloaded model: their default steel stands in
        groups.setdefault(name, []).append(shape_geom(sc.b, s))
    return [vis(union(g), [_srgb(c) for c in RN.COLORS[n.rstrip("*")][0]], _look_cite(n.rstrip("*")),
                roughness=float(RN.COLORS[n.rstrip("*")][1]), metallic=float(RN.COLORS[n.rstrip("*")][2]),
                name=f"{n.rstrip('*')} ({'their drawn model is not exported; ' if n.endswith('*') else ''}linear->sRGB)")
            for n, g in groups.items()] or None


def _press_frame(x: Exporter, sc: _Bare) -> None:
    """The press's fixed stand, base, column, table and head (shapes on the world body)."""
    b = sc.b
    ids = [s for s in range(b.shape_count) if int(b.shape_body[s]) == -1 and b.shape_label[s].startswith("press_")]
    cite = where(T.DrillPress)
    shapes = [shape_geom(b, s) for s in ids]
    x.add("drill_press__frame", "Drill press (Grizzly G7943 class): stand, column, table and head", "articulated",
          _closed_union(shapes) or tidy(union(shapes), None), motion="static", visual=_frame_visual(sc, ids),
          material={"name": "painted steel and cast iron"}, friction=mu_cited("default"),
          friction_partner="any surface (their default pair rule)", source_ref=cite,
          notes="Union of their static collision boxes and cylinders (no mass: they are fixed to the world; the moving "
                "links carry the ledger masses).", extra={"joint": None, "shapes": [b.shape_label[s] for s in ids]})


PARTS = [
    ("ballpeen_hammer", "Ball-peen hammer 32 oz", PT.ballpeen, "ballpeen", [], dict(name="ballpeen")),
    ("hole_brush", "Anchor-hole cleaning brush (HIT-RB class)", PT.brush, "brush", [], dict(name="brush")),
    ("vacuum_nozzle", "HEPA vacuum wand with narrow nozzle", PT.nozzle, "nozzle", [], dict(name="nozzle")),
    ("hex_bolt_m8", "M8 x 30 hex bolt", PT.bolt, "bolt", [], dict(name="bolt", d=0.008, length=0.03)),
    ("carriage_bolt", "1/2 x 4 in carriage bolt", lambda sc, sp: PT.bolt(sc, sp, True), "cbolt", [],
     dict(name="cbolt", d=0.0127, length=0.1016)),
    ("flat_bar", "S235 flat bar 50 x 150 x 6 mm", PT.workpiece, "flat_bar", [],
     dict(name="flat_bar", kind="bar", material="steel", half=(0.025, 0.075, 0.003))),
    ("alu_block", "6061-T6 aluminium block 50 x 50 x 25 mm", PT.workpiece, "alu_block", [],
     dict(name="alu_block", kind="block", material="aluminium", half=(0.025, 0.025, 0.0125))),
    ("ply_backer", "Plywood backer 220 x 280 x 18 mm", PT.workpiece, "ply_backer", [],
     dict(name="ply_backer", kind="panel", material="plywood", half=(0.11, 0.14, 0.009))),
    ("fir_board", "Douglas-fir board 140 x 300 x 38 mm", PT.workpiece, "board", [],
     dict(name="board", kind="board", material="douglas_fir", half=(0.070, 0.150, 0.019))),
]


def _simple(x: Exporter, row: tuple) -> None:
    key, title, fn, prefix, names, spec = row
    sc = _Bare()
    fn(sc, dict(spec, xyz=(0.0, 0.0, 0.0)))
    export_links(x, sc, prefix, key, title, PT.bolt if "bolt" in key else fn, names, category="tool" if "bolt" not in key else "other")


def _vacuum(x: Exporter) -> None:
    sc = _Bare()
    V.build(sc, {"xyz": (0.0, 0.0, 0.0), "yaw": 0.0})
    export_links(x, sc, "vacuum", "shop_vacuum", "Shop vacuum (RIDGID RV2400HF class)", V.VacuumUnit, [], category="articulated")


def export(x: Exporter) -> None:
    jobs = [(r[0], lambda r=r: _tool(x, r)) for r in TOOLS]
    jobs += [(r[0], lambda r=r: _simple(x, r)) for r in PARTS]
    jobs += [("shop_vacuum", lambda: _vacuum(x))]
    for key, fn in jobs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            x.fail(key, f"{type(exc).__name__}: {exc}")
