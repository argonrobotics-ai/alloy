# SPDX-License-Identifier: Apache-2.0
"""label_application: packing bench, shipping carton, label printers, specimen tube, HDPE canister, squeegee card, labels.

Shapes are the boxes/cylinders their ``Scene`` adds (constants in ``scene.py`` / ``materials.py``); the scene needs a
physics engine, so the printer, carton and tube are rebuilt here from the same numbers (cited by line). Labels are
thin films with their stock constants (cloth block).
"""

from __future__ import annotations

import logging
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from _common import Exporter, box, cited, cylinder, dim, transformed, union, vis

from label_application import artwork as AW
from label_application import layouts as LY
from label_application import materials as M
from label_application import render as R
from label_application import scene as S
from label_application import config as CFG

log = logging.getLogger("gar.export.label_application")
SRC = Path(S.__file__).parent
IN = S.IN


def _line(fname: str, pattern: str) -> str:
    for n, line in enumerate((SRC / fname).read_text().splitlines(), 1):
        if pattern in line:
            return f"label_application/{fname}:{n}"
    raise KeyError(f"{pattern!r} not in {fname}")


def _comment(fname: str, pattern: str) -> str:
    for line in (SRC / fname).read_text().splitlines():
        if pattern in line:
            tail = line.split("#", 1)[1] if "#" in line else line
            return " ".join(tail.split())[:300]
    return ""


def _mu(value: float, pattern: str, partner: str) -> tuple[dict, str]:
    return cited(value, "estimate", _line("scene.py", pattern), "Coulomb coefficient of the shape in their scene"), partner


# ------------------------------------------------------------------------------------------------ render parts
def _srgb(rgb) -> list[float]:
    """Their render COLORS are linear diffuse colours; parts are sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in rgb]


def _paint(geom: trimesh.Trimesh, name: str, label: str = "", glass: bool = False) -> dict:
    """Part painted with their ``render.COLORS[name]`` = (linear colour, roughness, metallic)."""
    rgb, rough, metal = R.COLORS[name]
    return vis(geom, _srgb(rgb), _line("render.py", f"{name}=(("), rough, metal,
               name="glass" if glass else f"{label or name} (linear->sRGB)")


def _dressed(geom: trimesh.Trimesh, name: str, label: str = "") -> dict:
    """Part painted with a colour their ``dressing.py`` creates inline: ``d.color("name", (rgb), roughness)``."""
    text = (SRC / "dressing.py").read_text()
    m = re.search(rf'd\.color\("{name}", \(([^)]*)\), ([0-9.]+)', text)
    rgb = [float(v) for v in m.group(1).split(",")]
    return vis(geom, _srgb(rgb), _line("dressing.py", f'd.color("{name}"'), float(m.group(2)), 0.0,
               name=f"{label or name} (linear->sRGB)")


def _slice(m: trimesh.Trimesh, z: float, keep_above: bool) -> trimesh.Trimesh:
    return trimesh.intersections.slice_mesh_plane(m, [0, 0, -1.0 if keep_above else 1.0], [0, 0, z], cap=True)


def _planar_box(half, pos=(0, 0, 0), per_m: float = 1 / .4) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Box with per-face vertices and planar uv (their ``_box_uv``: the texture spans 0.4 m)."""
    m = box(half, pos)
    verts, faces, uv = [], [], []
    for tri in m.faces:
        p = m.vertices[tri]
        ax = int(np.argmax(np.abs(np.cross(p[1] - p[0], p[2] - p[0]))))
        a, b = [i for i in range(3) if i != ax]
        verts += list(p)
        uv += [(q[a] * per_m, q[b] * per_m) for q in p]
        faces.append([len(verts) - 3, len(verts) - 2, len(verts) - 1])
    out = trimesh.Trimesh(verts, faces, process=False)
    return out, np.array(uv)


def _label_visual(fmt: str, md: float, cd: float, t: float) -> list[dict]:
    """Printed front (their label art, uv from the rest coordinates) over the white back (``label_back``)."""
    from PIL import Image  # noqa: PLC0415
    task, make = {"4x6": ("carrier-top", AW.shipping_label), "2x1": ("tube-wrap", AW.specimen_label),
                  "2x4": ("part-id-hdpe", AW.part_label)}[fmt]
    with tempfile.TemporaryDirectory() as tmp:
        art = make(tmp, LY.PAYLOADS[task])
        img = Image.open(art["path"]).convert("RGB").copy()
    pet = M.STOCKS[S.LABELS[fmt]["stock"]].name and "polyester" in M.STOCKS[S.LABELS[fmt]["stock"]].name.lower()
    xs, ys, z = [-md / 2, md / 2], [-cd / 2, cd / 2], t / 2
    corners = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
    top = trimesh.Trimesh([(x, y, z) for x, y in corners], [[0, 1, 2], [0, 2, 3]], process=False)
    uv = np.array([(1 - (y + cd / 2) / cd, (x + md / 2) / md) for x, y in corners])
    front = vis(top, [1.0, 1.0, 1.0], _line("render.py", 'material("label_print"'), .42 if pet else .55, .55 if pet else 0.0,
                image=img, uv=uv, name="printed facestock")
    bottom = trimesh.Trimesh([(x, y, -z) for x, y in corners], [[0, 2, 1], [0, 3, 2]], process=False)
    return [front, _paint(bottom, "label_back", "adhesive side")]


def _bench(x: Exporter) -> None:
    z = S.TABLE
    parts = [box((.915, .38, .02), (0, 0, z - .02)), box((.88, .34, .01), (0, 0, .18))]
    parts += [box((.02, .02, (z - .04) / 2), (sx, sy, (z - .04) / 2)) for sx in (-.86, .86) for sy in (-.33, .33)]
    mu, partner = _mu(.4, 'self.box("bench_top"', "cartons and tools")
    look = [_paint(parts[0], "laminate", "top"), _paint(parts[1], "laminate_dark", "lower shelf"),
            _paint(union(parts[2:]), "steel_frame", "legs")]
    x.add("packing_bench", "Packing bench 1.83 x 0.76 m with lower shelf", "furniture", union(parts), motion="static", visual=look,
          material={"name": "laminate top on a steel frame"}, friction=mu, friction_partner=partner,
          dims=[dim("top height", z, "z", _line("scene.py", "TABLE = ")), dim("length", 1.83, "x", _line("scene.py", 'self.box("bench_top"'))],
          source_ref=_line("scene.py", "def room"), notes="40 mm top, four legs and the lower shelf, as their room().")


def _carton(x: Exporter) -> None:
    L, W, H = 16 * IN, 12 * IN, 8 * IN
    mass = .55 + CFG.Config().carton_content_kg
    cite = _line("scene.py", "L, W, H = 16*IN")
    mu, partner = _mu(.4, 'carton_shell', "the bench top")
    kraft, kuv = _planar_box((L / 2, W / 2, H / 2))
    with tempfile.TemporaryDirectory() as tmp:
        from PIL import Image  # noqa: PLC0415
        kimg = Image.open(AW.kraft_texture(tmp)).convert("RGB").copy()
    look = vis(kraft, [1.0, 1.0, 1.0], _line("render.py", 'material("kraft"'), .85, 0.0, image=kimg, uv=kuv, name="kraft liner")
    x.add("shipping_carton", "Shipping carton 16 x 12 x 8 in RSC with contents", "container", box((L / 2, W / 2, H / 2)),
          visual=[look],
          material={"name": "kraft corrugated board"},
          mass=cited(mass, "authored", _line("scene.py", "self.set_mass(body, .55"), "board ~0.55 kg + contents (carton_content_kg 2.0)"),
          friction=mu, friction_partner=partner,
          dims=[dim("length", L, "x", cite), dim("width", W, "y", cite), dim("height", H, "z", cite)], source_ref=cite,
          notes="Solid box as their collider (mass includes 2 kg contents); seam tape is a 20 um visual film.")


def _printer(x: Exporter, key: str, name: str, head_width: float, lip: float, hd: float, ht: float) -> None:
    riser = .28
    hw = head_width / 2 + .06
    parts = [box((.12, .16, .006), (-.04, 0, riser - .006))]
    parts += [box((.01, .16, (riser - .012) / 2), (-.04 + sx * .11, 0, (riser - .012) / 2)) for sx in (-1, 1)]
    parts += [box((hd, hw, ht / 2), (0, 0, riser + ht / 2)),
              box(((lip + .006) / 2, hw - .01, .002), (hd - .002 + (lip + .006) / 2, 0, riser + .058 - .002)),
              box((.003, hw - .004, .025), (hd + .002, 0, riser + .058 - .025))]
    cite = _line("scene.py", "def printer")
    exit_z = riser + .058
    housing = _dressed(_slice(parts[3], exit_z, keep_above=False), "printer_shell", "housing base")
    lid = _dressed(_slice(parts[3], exit_z, keep_above=True), "printer_lid", "housing lid")
    look = [_paint(union(parts[:3]), "stand_grey", "shelf riser"), housing, lid,
            _dressed(parts[4], "printer_trim", "peel bar"), _dressed(parts[5], "printer_shell", "door")]
    x.add(key, name, "fixture", union(parts), motion="static", visual=look, material={"name": "ABS printer housing on a steel shelf riser"},
          friction=cited(.45, "estimate", _line("scene.py", "v = dict(mu=.45"), "default shape mu"), friction_partner="labels and parts",
          dims=[dim("riser plus housing height", riser + ht, "z", cite)], source_ref=cite,
          notes="Housing, riser top and sides, peel bar and door with the authored presenter lip; authored boxes at the "
                "footprint class of the Zebra ZD421 (221 x 254 x 170 mm); the exit slot is left closed. No mass given.")


def _tube(x: Exporter) -> None:
    r, Lt = .0065, .100
    wall = cylinder(r, Lt / 2, sections=32)
    cap = cylinder(.0081, .012, (0, 0, Lt / 2 + .010), sections=32)
    cite = _line("scene.py", "r, Lt = .0065")
    look = [_paint(wall, "pet_clear", glass=True), _paint(cap, "cap_lavender", "cap")]
    x.add("specimen_tube", "Specimen tube with cap (BD Vacutainer class, 13 x 100 mm)", "container", union([wall, cap]), visual=look,
          material={"name": "PET tube, polypropylene cap",
                    "density": cited(1380., "estimate", _line("scene.py", 'tube_wall'), "tube_wall density 1380 kg/m3")},
          mass=cited(.0105, "estimate", _line("scene.py", "self.set_mass(body, .0105)"), "PET tube + closure + dry fill (estimate)"),
          friction=cited(.4, "estimate", _line("scene.py", 'tube_wall'), "tube wall mu=.4"), friction_partner="POM tube rest",
          dims=[dim("overall height", Lt / 2 + .022 + Lt / 2, "z", cite)],
          source_ref=cite, notes="Solid cylinder and cap hull (their collider; contents lumped in the mass), axis +z, "
                                 "origin at the tube-wall centre; cap at +z.")


def _canister(x: Exporter) -> None:
    cite = _line("scene.py", 'jug_wall')
    x.add("hdpe_canister", "HDPE canister 4 L with contents (150 x 250 mm)", "container", cylinder(.075, .125, sections=64),
          visual=[_paint(cylinder(.075, .125, sections=64), "hdpe_white")],
          material={"name": "white HDPE", "density": cited(950., "estimate", cite, "jug_wall density 950 kg/m3")},
          mass=cited(3.2, "authored", _line("scene.py", "self.set_mass(body, 3.2)"), "4 L round screw-lid HDPE canister with contents (authored)"),
          friction=cited(.3, "estimate", cite, "jug_wall mu=.3"), friction_partner="the bench top",
          dims=[dim("diameter", .15, "diameter_xy", cite), dim("height", .25, "z", cite)], source_ref=cite,
          notes="Solid cylinder as their collider (contents lumped in the mass).")


def _squeegee(x: Exporter) -> None:
    cite = _line("scene.py", 'squeegee_card')
    x.add("squeegee_card", "PP squeegee card 100 x 70 x 1.5 mm", "tool", box((.05, .035, .00075)), visual=[_paint(box((.05, .035, .00075)), "squeegee_pp")],
          material={"name": "polypropylene", "density": cited(905., "estimate", cite, "squeegee_card density 905 kg/m3")},
          friction=cited(.3, "estimate", cite, "squeegee_card mu=.3"), friction_partner="label",
          dims=[dim("length", .1, "x", cite), dim("thickness", .0015, "z", cite)], source_ref=cite,
          thickness=cited(.0015, "authored", cite, "PP squeegee card, 1.5 mm"),
          notes="Thin plate; mass from density x volume.")


def _label(x: Exporter, fmt: str) -> None:
    spec = S.LABELS[fmt]
    st = M.STOCKS[spec["stock"]]
    cite = _line("materials.py", "PAPER_DT = " if spec["stock"] == "paper" else "PET_7818 = ")
    note = _comment("materials.py", "Direct-thermal paper shipping") if spec["stock"] == "paper" else _comment("materials.py", "3M 7818-class")
    q = lambda what: f"{what} ({st.name})"
    t = st.thickness
    cl = {"thickness": cited(t, "datasheet", cite, q("facestock + adhesive caliper")),
          "stretch_warp": cited(st.E_md * st.t_face, "estimate", cite, q("E_md x facestock caliper")),
          "stretch_weft": cited(st.E_cd * st.t_face, "estimate", cite, q("E_cd x facestock caliper")),
          "shear": cited(math.sqrt(st.E_md * st.E_cd) / (2 * (1 + st.nu)) * st.t_face, "derived", cite, q("G12 x caliper")),
          "bend_warp": cited(st.D_md, "derived", cite, q("E t^3 / 12(1-nu^2)")),
          "bend_weft": cited(st.D_cd, "derived", cite, q("E t^3 / 12(1-nu^2)"))}
    md, cd = spec["md"], spec["cd"]
    x.add(f"label_{fmt}", f"Label {fmt} in ({spec['stock']} stock)", "cloth", box((md / 2, cd / 2, t / 2)), kind="cloth", visual=_label_visual(fmt, md, cd, t),
          material={"name": st.name}, thickness=cl["thickness"], cloth=cl,
          mass=cited(st.rho_s * md * cd, "derived", cite, q("areal mass incl. adhesive x area")),
          friction=cited(M.SURFACES["paper_label"].mu if hasattr(M.SURFACES.get("paper_label"), "mu") else .4, "estimate", cite, "facestock Coulomb"),
          friction_partner="bench and tools",
          dims=[dim("length (feed)", md, "x", _line("scene.py", f'"{fmt}": dict(md=')), dim("width", cd, "y", _line("scene.py", f'"{fmt}": dict(md='))],
          source_ref=_line("scene.py", "LABELS = {"), notes="Flat slab of the stock caliper; " + note)


def export(x: Exporter) -> None:
    steps = [("bench", _bench), ("carton", _carton), ("printer_4in", lambda x: _printer(x, "label_printer_4in", "Label printer, 4-inch head (ZD421 class)", 4 * IN, .035, .11, .17)),
             ("printer_2in", lambda x: _printer(x, "label_printer_2in", "Label printer, 2-inch head (ZD421 class)", 2 * IN, 0., .095, .15)),
             ("tube", _tube), ("canister", _canister), ("squeegee", _squeegee)]
    steps += [(f"label_{f}", (lambda x, f=f: _label(x, f))) for f in S.LABELS]
    for key, fn in steps:
        try:
            fn(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(key, f"{type(exc).__name__}: {exc}")
            log.exception("%s failed", key)
