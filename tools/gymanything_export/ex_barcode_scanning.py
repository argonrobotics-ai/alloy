# SPDX-License-Identifier: Apache-2.0
"""barcode_scanning: handheld scanner, its cradle, the stock SKUs, shelving, counter, basket/tote, books, keypad.

Their ``Scene`` builds Newton colliders; this exporter subclasses it with a recorder that keeps each box/hull the
scene would add (``Recorder``), so the geometry is their own code's output, with no physics engine involved. The
scanner and SKU shapes use their render/hull builders (``housing.py``, ``catalog.neck``).
"""

from __future__ import annotations

import logging
import math
import tempfile
from types import SimpleNamespace

import numpy as np
import trimesh

from _common import Exporter, box, cited, comment_of, cylinder, dim, line_of, transformed, union, vis, where

from barcode_scanning import catalog as C
from barcode_scanning import housing as H
from barcode_scanning import packaging as PK
from barcode_scanning import params as P
from barcode_scanning import render as R
from barcode_scanning import scene as S
from barcode_scanning import stock as K

log = logging.getLogger("gar.export.barcode_scanning")
SRC_DOC = " ".join(C.__doc__.split())[:300]


class _Stub:
    """Stands in for ``newton`` / ``wp`` / the model builder: keeps the shapes the scene adds."""

    def __init__(self) -> None:
        self.hulls: list[tuple[trimesh.Trimesh, tuple, tuple]] = []

    def Mesh(self, v, f):  # noqa: N802
        return trimesh.Trimesh(np.asarray(v, float), np.asarray(f).reshape(-1, 3), process=False)

    def transform(self, pos=(0, 0, 0), quat=(0, 0, 0, 1)):
        return tuple(pos), tuple(quat)

    def quat_identity(self):
        return (0.0, 0.0, 0.0, 1.0)

    def add_shape_convex_hull(self, body, mesh=None, xform=None, **_kw):
        pos, quat = xform if xform else ((0, 0, 0), (0, 0, 0, 1))
        self.hulls.append((mesh, pos, quat))
        return len(self.hulls)


class Recorder(S.Scene):
    """Their ``Scene`` with ``box`` recording instead of adding to a physics model (no engine needed)."""

    def __init__(self, preset: str = "counter") -> None:  # noqa: super init needs the engine
        self.stub = _Stub()
        self.newton = self.wp = self.b = self.stub
        self.c = SimpleNamespace(preset=preset, task="", seed=0)
        self.g1, self.cart_pose = False, None
        self.materials, self.fixtures, self.items, self.slots = {}, {}, {}, {}
        self.rng = np.random.default_rng(0)
        self.boxes: list[dict] = []
        self._n = 0

    def cfg(self, **kw):
        return kw

    def box(self, name, pos, half, material, body=-1, rot=None, collide=True, hull=False, **kw):
        self.boxes.append(dict(name=name, pos=tuple(pos), half=tuple(half), rot=tuple(rot or (0, 0, 0, 1)),
                               body=body, mu=kw.get("mu"), material=material))
        return len(self.boxes)

    def free_body(self, name, pos, rot=None):
        self._n += 1
        return self._n

    def set_mass(self, body, mass):
        return None

    def meshes(self, only: str | None = None) -> list[trimesh.Trimesh]:
        return [box(b["half"], b["pos"], b["rot"]) for b in self.boxes if only is None or only in b["name"]]

    def meshes_by_material(self) -> dict[str, list[trimesh.Trimesh]]:
        out: dict[str, list[trimesh.Trimesh]] = {}
        for b in self.boxes:
            out.setdefault(b["material"], []).append(box(b["half"], b["pos"], b["rot"]))
        return out


def _scene_for_scanner() -> S.Scene:
    return S.Scene.__new__(S.Scene)


# ------------------------------------------------------------------------------------------------ render parts
def _srgb(rgb) -> list[float]:
    """Their COLORS table is linear (render.py ledger note); the part colour is sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in rgb]


_INLINE = {"imager_glass": ((.02, .02, .02), .05, 0.)}   # render.py: S.color("imager_glass", (.02, .02, .02), .05, 0.)


def _paint(geom: trimesh.Trimesh, name: str, label: str = "") -> dict:
    """Part painted with their ``render.COLORS[name]`` = (linear colour, roughness, metallic)."""
    if name in R.COLORS:
        (rgb, rough, metal), cite = R.COLORS[name], line_of(R, f"{name}=((")
    else:   # colours their ``build`` creates inline with ``S.color``
        rgb, rough, metal = _INLINE[name]
        cite = line_of(R, f'S.color("{name}"')
    return vis(geom, _srgb(rgb), cite, rough, metal, name=f"{label or name} (linear->sRGB)")


def _flat(geom: trimesh.Trimesh, rgb255, cite: str, finish: str, label: str) -> dict:
    """Part with a generated sRGB colour printed through their ``printed`` material of the given finish."""
    f = R.FINISH[finish]
    return vis(geom, [c / 255 for c in rgb255], cite, f["coat_rough"] if f else 0.85, 0.0, name=label)


def _quad(c, u, v, size) -> trimesh.Trimesh:
    """Their ``Scene.quad``: rectangle spanning size along unit u, v; uv (0,0)..(1,1)."""
    c, u, v = (np.asarray(x, float) for x in (c, u, v))
    a, b = size[0] / 2, size[1] / 2
    m = trimesh.Trimesh([c - a * u - b * v, c + a * u - b * v, c + a * u + b * v, c - a * u + b * v],
                        [[0, 1, 2], [0, 2, 3]], process=False)
    return m


_QUAD_UV = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)


def _print_part(geom, image, uv, finish: str, label: str) -> dict:
    """Printed texture on a substrate with the item's finish (their ``printed``: reflectance PRINT_WHITE)."""
    f = R.FINISH.get(finish)
    metal = f.get("metal") if f else None
    return vis(geom, [R.PRINT_WHITE] * 3, line_of(R, "PRINT_WHITE ="), f["coat_rough"] if f else 0.85, metal or 0.0,
               image=image, uv=uv, name=label)


def _face_parts(faces: dict, centre, size, finish: str, skip=(), book=False) -> list[dict]:
    """One textured quad per box face, laid out as their ``_package`` / ``_hinged_book`` (front is -y)."""
    from PIL import Image  # noqa: PLC0415
    (cx, cy, cz), (W, D, H) = centre, size
    spec = {"front": ((0, -D / 2, 0), (1, 0, 0), (0, 0, 1), (W, H)), "back": ((0, D / 2, 0), (-1, 0, 0), (0, 0, 1), (W, H)),
            "right": ((W / 2, 0, 0), (0, 1, 0), (0, 0, 1), (D, H)), "left": ((-W / 2, 0, 0), (0, -1, 0), (0, 0, 1), (D, H)),
            "top": ((0, 0, H / 2), (1, 0, 0), (0, 1, 0), (W, D)), "bottom": ((0, 0, -H / 2), (1, 0, 0), (0, -1, 0), (W, D))}
    out = []
    for face, (c, u, v, sz) in spec.items():
        if face in skip or face not in faces:
            continue
        q = _quad(np.add(c, (cx, cy, cz)), u, v, sz)
        fin = "matte" if book and face in ("right", "top", "bottom") else finish
        out.append(_print_part(q, Image.open(faces[face][0]).copy(), _QUAD_UV, fin, f"print {face}"))
    return out


def _side(r0: float, r1: float, z0: float, z1: float, n: int = 48, uv: bool = False) -> trimesh.Trimesh:
    """Open frustum side (axis z, azimuth 0 = -y towards +x, as their ``cylinder_side``); uv u = azimuth, v = height."""
    v, t = [], []
    for k in range(n + 1):
        a = 2 * math.pi * k / n
        d = np.array([math.sin(a), -math.cos(a), 0.0])
        v += [d * r0 + [0, 0, z0], d * r1 + [0, 0, z1]]
        t += [(k / n, 0.0), (k / n, 1.0)]
    f = [t_ for i in range(n) for t_ in ([2 * i, 2 * i + 2, 2 * i + 3], [2 * i, 2 * i + 3, 2 * i + 1])]
    m = trimesh.Trimesh(v, f, process=False)
    if uv:
        m.visual = trimesh.visual.TextureVisuals(uv=np.array(t))
    return m


def _disc(r: float, z: float, n: int = 48) -> trimesh.Trimesh:
    ring = [(r * math.sin(a), -r * math.cos(a), z) for a in np.linspace(0, 2 * math.pi, n, False)]
    return trimesh.Trimesh([(0, 0, z)] + ring, [[0, 1 + (k + 1) % n, 1 + k] for k in range(n)], process=False)


def _wrap_uv(m: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(m.visual.uv)


def _cylinder_pkg(s: C.SKU, wrap: str) -> list[dict]:
    """Can / jar / bottle as their ``render._can``: printed wrap, body/shoulder/cap materials, lids."""
    from PIL import Image  # noqa: PLC0415
    r, h = s.dims
    g = C.neck(s)
    img = Image.open(wrap).copy()
    z0, z1 = g["panel"] if g else (-h / 2, h / 2)
    parts = [(lambda m: _print_part(m, img, _wrap_uv(m), s.finish, "printed wrap"))(_side(r, r, z0, z1, uv=True))]
    if not g:
        top = {"K05": "cap_blue", "K04": "lid_gold", "K10": "cap_blue"}.get(s.id, "lid_metal")
        bottom = "lid_metal" if top == "lid_metal" else "tote_grey"
        return parts + [_paint(_disc(r * .98, h / 2 - .0025), top, "lid"), _paint(_disc(r * .98, -h / 2), bottom, "base")]
    (b0, b1), (s0, s1), (c0, c1) = g["body"], g["shoulder"], g["cap"]
    glass = g["body_mat"] in ("pet_water", "jar_glass")
    body = [_paint(_side(r, r, b0, z0), g["body_mat"]), _paint(_side(r, r, z1, b1), g["body_mat"])]
    if glass:
        for p in body:
            p["name"] = "glass"
    sh = _paint(_side(r, g["cap_r"], s0, s1), g["shoulder_mat"], "shoulder")
    if g["shoulder_mat"] in ("pet_water", "jar_glass"):
        sh["name"] = "glass"
    cap = [_paint(_side(g["cap_r"], g["cap_r"], c0, c1), g["cap_mat"], "cap"), _paint(_disc(g["cap_r"], c1), g["cap_mat"], "cap top")]
    return parts + body + [sh] + cap


def _sku_visual(s: C.SKU) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        if s.shape == "box":
            return _face_parts(PK.box_faces(s, tmp), (0, 0, 0), s.dims, s.finish, book=s.category == "library")
        wrap = PK.wrap_texture(s, tmp)[0]
        return _cylinder_pkg(s, str(wrap))


def _planar_box(half, pos) -> trimesh.Trimesh:
    """Box with per-face vertices and planar uv at 1 cycle per metre (their ``box_uv`` for decks)."""
    m = box(half, pos)
    verts, faces, uv = [], [], []
    for tri in m.faces:
        p = m.vertices[tri]
        ax = int(np.argmax(np.abs(np.cross(p[1] - p[0], p[2] - p[0]))))
        a, b = [i for i in range(3) if i != ax]
        verts += list(p)
        uv += [(q[a], q[b]) for q in p]
        faces.append([len(verts) - 3, len(verts) - 2, len(verts) - 1])
    uv = np.array(uv)
    out = trimesh.Trimesh(verts, faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uv - uv.min(axis=0))
    return out


def _by_material(r: Recorder) -> list[dict]:
    return [_paint(union(ms), name) for name, ms in r.meshes_by_material().items()]


# ------------------------------------------------------------------------------------------------ scanner, cradle
def _scanner(x: Exporter) -> None:
    s = _scene_for_scanner()
    hc, hq = s.handle_frame()
    parts = H.scanner_meshes(s.HEAD_HALF, hc, hq, s.HANDLE_HALF)
    geom = union([parts["head"], parts["handle"]])
    env = P.SCANNER_ENVELOPE_M
    cite = line_of(P, "SCANNER_ENVELOPE_M =")
    x.add("scanner", "Handheld barcode scanner (Zebra DS2278 class)", "tool", geom,
          material={"name": "moulded ABS housing"},
          visual=[_paint(parts["head"], "scanner_body", "head shell"), _paint(parts["lining"], "imager_glass", "window lining"),
                  _paint(parts["handle"], "scanner_grip", "handle")],
          mass=cited(P.SCANNER_MASS_KG, "datasheet", line_of(P, "SCANNER_MASS_KG ="),
                     "scanner body (Zebra DS2278 class); includes the 3 g trigger blade"),
          friction=cited(P.MU["scanner_cradle"], "estimate", line_of(P, "MU = dict"), "Coulomb, estimates"),
          friction_partner="the cradle socket",
          dims=[dim("height", env[0], "z", cite), dim("width", env[1], "y", cite), dim("depth", env[2], "x", cite)],
          source_ref=where(S.Scene.scanner_and_cradle),
          notes="Housing: their render-only rounded head + handle (housing.scanner_meshes), window tunnel cut; the "
                "trigger blade is the separate scanner__trigger part.")
    _trigger(x, s)


def _trigger(x: Exporter, s: S.Scene) -> None:
    arm = P.TRIGGER_ARM_M
    cite = line_of(P, "TRIGGER_ARM_M =")
    _, hq = s.handle_frame()
    x.add("scanner__trigger", "Scanner trigger blade", "articulated", H.trigger_mesh(arm),
          material={"name": "black plastic"}, visual=[_paint(H.trigger_mesh(arm), "trigger_black")],
          mass=cited(P.TRIGGER_MASS_KG, "estimate", line_of(P, "TRIGGER_MASS_KG ="), "trigger"),
          friction=cited(0.6, "estimate", where(S.Scene.scanner_and_cradle), "trigger_blade box mu=.6"),
          friction_partner="the fingertip", motion="dynamic",
          source_ref=where(S.Scene.scanner_and_cradle),
          notes="Blade in its body frame (pivot at the origin); the spring is a preloaded joint target.",
          extra={"joint": {"type": "revolute", "parent": "scanner", "axis": [0, 1, 0],
                           "range": [0.0, P.TRIGGER_STOP_M / arm], "cite": cite,
                           "pivot_frame_quat_xyzw": [float(q) for q in hq],
                           "stiffness_N_per_m": P.TRIGGER_RATE_N_PER_M, "preload_N": P.TRIGGER_PRELOAD_N}})


def _cradle_geometry() -> trimesh.Trimesh:
    """Everything ``build_cradle`` adds, in the cradle frame (base bottom at z = 0, +x towards the window)."""
    r = Recorder("counter")
    pose = r.cradle_pose()
    r._seated_scanner_pose(pose)
    r.build_cradle()
    (px, py, pz), q = pose
    parts = r.meshes()
    for mesh, pos, quat in r.stub.hulls:
        parts.append(transformed(mesh, pos, quat))
    whole = union(parts)
    inv = np.linalg.inv(trimesh.transformations.quaternion_matrix([q[3], q[0], q[1], q[2]]) @ np.eye(4))
    inv[:3, 3] = -inv[:3, :3] @ np.array([px, py, pz])
    return whole.apply_transform(inv)


def _cradle(x: Exporter) -> None:
    env = P.CRADLE_ENVELOPE_M
    cite = line_of(P, "CRADLE_ENVELOPE_M =")
    geom = _cradle_geometry()
    x.add("scanner_cradle", "Scanner presentation cradle (Zebra CR2278-PC class)", "fixture", geom,
          motion="static", visual=[_paint(geom, "cradle_black")], material={"name": "black moulded plastic"},
          mass=cited(P.CRADLE_MASS_KG, "datasheet", line_of(P, "CRADLE_MASS_KG ="), "cradle, 151 g, fastened"),
          friction=cited(P.MU["scanner_cradle"], "estimate", line_of(P, "MU = dict"), "Coulomb, estimates"),
          friction_partner="the scanner handle",
          dims=[dim("width", env[1], "y", cite), dim("height", env[0], "z", cite)],
          source_ref=where(S.Scene.build_cradle),
          notes="Teardrop base hull plus the socket floor, lip, rear and side walls and pedestal, as their "
                "build_cradle adds them (recorded, then unioned); datasheet height 72 mm vs their 74 mm top.")


# ------------------------------------------------------------------------------------------------ stock SKUs
def _sku_mesh(s: C.SKU) -> trimesh.Trimesh:
    if s.shape == "box":
        w, d, h = s.dims
        return box((w / 2, d / 2, h / 2))
    r, h = s.dims
    n = C.neck(s)
    if not n:
        return cylinder(r, h / 2)
    ring = lambda rad, z: [(rad * math.cos(a), rad * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 32, False)]
    pts = np.array(ring(n["r"], n["body"][0]) + ring(n["r"], n["body"][1]) + ring(n["cap_r"], n["shoulder"][1]))
    z2, z3 = n["shoulder"][1], n["cap"][1]
    cap = cylinder(n["cap_r"], (z3 - z2) / 2, (0, 0, (z2 + z3) / 2))
    return union([trimesh.convex.convex_hull(pts), cap])


def _sku_dims(s: C.SKU, cite: str) -> list[dict]:
    if s.shape == "box":
        w, d, h = s.dims
        return [dim("width", w, "x", cite), dim("depth", d, "y", cite), dim("height", h, "z", cite)]
    return [dim("diameter", 2 * s.dims[0], "diameter_xy", cite), dim("height", s.dims[1], "z", cite)]


def _sku(x: Exporter, s: C.SKU) -> None:
    cite = line_of(C, f'SKU("{s.id}"')
    basis = "derived" if s.id == "K08" else "estimate"
    cat = "food" if s.category == "grocery" else "other"
    x.add(f"sku_{s.id.lower()}", f"{s.name} ({s.brand}), {s.size_text}".strip(), cat if s.id < "K07" else "other",
          _sku_mesh(s), material={"name": f"packaged retail goods, {s.finish} finish"}, visual=_sku_visual(s),
          mass=cited(s.mass, basis, cite, SRC_DOC), friction=cited(s.mu, "estimate", cite, "per-SKU Coulomb estimate"),
          friction_partner="shelf deck or counter top (MuJoCo takes the larger of the two shapes' values)",
          dims=_sku_dims(s, cite), source_ref=where(S.Scene.add_item),
          notes="Solid pack/can as their collider (box, cylinder, or necked body hull + cap); printed artwork has "
                "no geometry. Mass is the filled retail unit.")


def _book_visual(s: C.SKU, t: float) -> tuple[list[dict], list[dict]]:
    """(cover parts, text-block parts) as their ``_hinged_book``: printed boards, endpaper, board-coloured edges."""
    w, d, h = s.dims
    with tempfile.TemporaryDirectory() as tmp:
        faces = PK.box_faces(s, tmp)
        block = _face_parts(faces, (0, t / 2, 0), (w, d - t, h), s.finish, skip=("front",), book=True)
        cover = _face_parts(faces, (w / 2, 0, 0), (w, t, h), s.finish, skip=("back", "left", "right", "top", "bottom"))
    edge = tuple(c / 255 * .8 for c in s.palette[0])
    edge_part = vis(box((w / 2, t / 2, h / 2), (w / 2, 0, 0)), _srgb(edge), line_of(R, "board_edge_"), .6, 0.0,
                    name="board edges (linear->sRGB)")
    endpaper = vis(_quad((0, t - d / 2, 0), (-1, 0, 0), (0, 0, 1), (w, h)), _srgb((.86, .84, .78)),
                   line_of(R, 'S.color("endpaper"'), .8, 0.0, name="endpaper (linear->sRGB)")
    return [edge_part] + cover, block + [endpaper]


def _books(x: Exporter) -> None:
    for sid in ("K12", "K13"):
        s = C.SKUS[sid]
        w, d, h = s.dims
        t, m_cover, hinge = P.book_cover(s)
        cite = line_of(C, f'SKU("{sid}"')
        block = box((w / 2, (d - t) / 2, h / 2), (0, t / 2, 0))
        key = f"book_{s.size_text.lower()}"
        cov, blk = _book_visual(s, t)
        x.add(key, f"{s.name} ({s.size_text}) text block and back cover", "other", block, visual=blk,
              material={"name": "paper text block with board back cover"},
              mass=cited(s.mass - m_cover, "estimate", cite, SRC_DOC),
              friction=cited(s.mu, "estimate", cite, "per-SKU Coulomb estimate"), friction_partner="counter top",
              dims=[dim("width", w, "x", cite), dim("height", h, "z", cite)], source_ref=where(S.Scene.add_book),
              notes="Total book mass minus the front cover's (P.book_cover); cover is the __cover part.")
        x.add(f"{key}__cover", f"{s.size_text} book front cover", "articulated",
              box((w / 2, t / 2, h / 2), (w / 2, 0, 0)), material={"name": "greyboard / card cover"}, visual=cov,
              mass=cited(m_cover, "estimate", line_of(P, "def book_cover"), comment_of(P, "BOOK_COVER_MASS_KG =")),
              friction=cited(s.mu, "estimate", cite, "per-SKU Coulomb estimate"), friction_partner="counter top",
              source_ref=where(S.Scene.add_book), notes="Hinge at the cover's -x edge (body origin).",
              extra={"joint": {"type": "revolute", "parent": key, "axis": [0, 0, -1], "range": [0.0, 1.5 * math.pi],
                               "friction_Nm": hinge, "cite": where(S.Scene.add_book)}})


# ------------------------------------------------------------------------------------------------ fixtures
def _shelf_visual(r: Recorder) -> list[dict]:
    """Steel posts and beams, and the decks printed with their generated particle-board texture (planar uv)."""
    from PIL import Image  # noqa: PLC0415
    steel = [box(b["half"], b["pos"]) for b in r.boxes if b["material"] == "shelf_steel"]
    decks = [_planar_box(b["half"], b["pos"]) for b in r.boxes if b["material"] == "particle_board"]
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(PK.particle_board(tmp)).copy()
    parts = [_print_part(d, img, np.asarray(d.visual.uv), "matte", "particle-board deck") for d in decks]
    return [_paint(union(steel), "shelf_steel")] + parts


def _shelving(x: Exporter) -> None:
    r = Recorder("stockroom")
    r.shelving(bays=(0,))
    ctr = r.boxes[0]
    geom = union(r.meshes())
    cite = line_of(P, "SHELF_BAY_M =")
    x.add("shelving_bay", "Boltless steel shelving bay (Uline H-7115 class)", "furniture", geom, motion="static",
          visual=_shelf_visual(r),
          material={"name": "steel angle posts and beams with particle-board decks"},
          friction=cited(0.30, "estimate", where(S.Scene.shelving), "deck mu=.30 (particle board, Uline H-7115)"),
          friction_partner="stock on the deck",
          dims=[dim("bay width", P.SHELF_BAY_M[0], "x", cite), dim("height", P.SHELF_BAY_M[2], "z", cite)],
          source_ref=where(S.Scene.shelving), notes=f"One bay: 4 posts, 5 decks with front/back beams. {ctr['name']}")


def _counter(x: Exporter) -> None:
    r = Recorder("counter")
    r.counter_body("store")
    geom = union(r.meshes())
    cite = line_of(P, "COUNTER_TOP_M =")
    x.add("counter", "Service counter (ADA 2010 904.4 height)", "furniture", geom, motion="static",
          visual=_by_material(r),
          material={"name": "laminate-topped panel counter"},
          friction=cited(0.35, "estimate", where(S.Scene.counter_body), "counter_top mu=.35"),
          friction_partner="items on the counter",
          dims=[dim("top height", P.COUNTER_TOP_M, "z", cite)], source_ref=where(S.Scene.counter_body),
          notes="Top slab, front and back panels and two end panels (their boxes, unioned).")


def _open_box(x: Exporter, key: str, name: str, size: tuple, mass: float, mat: str) -> None:
    r = Recorder("counter")
    K._open_box(r, key, (0, 0, 0), size, mat, wall=0.004, mass=mass)
    geom = union(r.meshes())
    cite = line_of(K, f'_open_box(scene, "{key}"') if key != "returns_bin" else line_of(K, '"returns_bin"')
    rgb, tag = ((30, 88, 190), "(30, 88, 190)") if key == "basket" else ((58, 62, 66), "(58, 62, 66)")
    look = _flat(geom, rgb, line_of(R, tag), "semigloss", "moulded polypropylene (print tile base colour, semi-gloss)")
    x.add(key, name, "container", geom, material={"name": "rigid moulded plastic, 4 mm walls"}, visual=[look],
          mass=cited(mass, "authored", cite, "authored hand basket / tote / bin"),
          friction=cited(0.4, "estimate", where(K._open_box), "wall mu=.4"), friction_partner="counter top and stock",
          dims=[dim("length", size[0], "x", cite), dim("width", size[1], "y", cite), dim("height", size[2], "z", cite)],
          source_ref=where(K._open_box), notes="Five walls (floor + four sides), open top, as in their _open_box.")


def _keypad(x: Exporter) -> None:
    cite = where(S.Scene.keypad)
    x.add("keypad_case", "17-key numeric keypad case (Cherry MX Red class)", "fixture",
          box((0.045, 0.07, 0.012)), motion="static", material={"name": "black plastic"},
          visual=[_paint(box((0.045, 0.07, 0.012)), "keypad_black")],
          friction=cited(0.5, "estimate", cite, "keypad_case mu=.5"), friction_partner="the fingertip",
          source_ref=cite, notes="Case only; the 17 keycaps are separate parts (keypad_case__key_1u).")
    half = ((P.KEY_PITCH_M - 0.0025) / 2, (P.KEY_PITCH_M - 0.0025) / 2, 0.004)
    x.add("keypad_case__key_1u", "Keypad keycap (1u)", "articulated", box(half), material={"name": "ABS keycap"},
          visual=[_paint(box(half), "keycap")],
          mass=cited(P.KEY_MASS_KG, "estimate", line_of(P, "KEY_MASS_KG ="), comment_of(P, "KEY_MASS_KG =")),
          friction=cited(0.5, "estimate", cite, "keycap mu=.5"), friction_partner="the fingertip",
          dims=[dim("key pitch less gap", P.KEY_PITCH_M - 0.0025, "x", line_of(P, "KEY_PITCH_M ="))],
          source_ref=cite, notes="2 g per 1u key; larger keys scale mass with area (not exported separately).",
          extra={"joint": {"type": "prismatic", "parent": "keypad_case", "axis": [0, 0, -1],
                           "range": [0.0, P.KEY_TRAVEL_M], "cite": line_of(P, "KEY_OPERATE_M, KEY_TRAVEL_M"),
                           "force_initial_N": P.KEY_FORCE_INITIAL_N, "force_operate_N": P.KEY_FORCE_OPERATE_N}})


def export(x: Exporter) -> None:
    steps = [("scanner", lambda: _scanner(x)), ("scanner_cradle", lambda: _cradle(x))]
    steps += [(f"sku_{sid.lower()}", (lambda s=s: _sku(x, s))) for sid, s in C.SKUS.items() if sid not in ("K12", "K13")]
    steps += [("books", lambda: _books(x)), ("shelving_bay", lambda: _shelving(x)), ("counter", lambda: _counter(x)),
              ("basket", lambda: _open_box(x, "basket", "Customer hand basket", (.43, .30, .20), .6, "basket_blue")),
              ("bag_tote", lambda: _open_box(x, "bag_tote", "Bagging tote", (.40, .30, .18), .9, "tote_grey")),
              ("keypad", lambda: _keypad(x))]
    for key, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            x.fail(key, f"{type(exc).__name__}: {exc}")
            log.exception("%s failed", key)
