# SPDX-License-Identifier: Apache-2.0
"""syringe_handling: syringes (1/3/10 mL, prefilled flush), five needles, 10R vial, injection pad, saline-lock
connector and clamp, sharps container, procedure tray, basin and the counter.

Their scene builds engine shapes from ``params.py`` dictionaries (``SYRINGES``, ``NEEDLES``, ``VIAL``, ``TRAY`` ...);
this exporter rebuilds the same shapes from those dictionaries (the builders need a physics engine). Articulated
objects are one spec per rigid part (``<object>__<part>``) with the joint in ``extra.joint``.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from _common import Exporter, box, cited, cylinder, dim, revolve, transformed, union, vis

from syringe_handling import params as P
from syringe_handling import render as R

log = logging.getLogger("gar.export.syringe_handling")
Y90 = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))      # rotates local z onto +x
SRC = Path(P.__file__).parent


def _line(fname: str, pattern: str) -> str:
    """``syringe_handling/<file>:<line>`` of the first line containing ``pattern``."""
    for n, line in enumerate((SRC / fname).read_text().splitlines(), 1):
        if pattern in line:
            return f"syringe_handling/{fname}:{n}"
    raise KeyError(f"{pattern!r} not in {fname}")


def _note(fname: str, pattern: str) -> str:
    for line in (SRC / fname).read_text().splitlines():
        if pattern in line:
            return " ".join(line.strip().split())[:300]
    return ""


def _mu(key: str, partner: str) -> tuple[dict, str]:
    pat = f'{key}=_p("MU_{key.upper()}"' if key != "default" else 'default=_p("MU_DEFAULT"'
    return cited(P.MU[key], "estimate", _line("params.py", pat), _note("params.py", pat)), partner


def _hollow(r_out: float, r_in: float, z0: float, z1: float) -> trimesh.Trimesh:
    """Tube about +z between z0 and z1."""
    ring = [(r_in, z0), (r_out, z0), (r_out, z1), (r_in, z1)]
    return revolve(ring, 64)


def _along_x(m: trimesh.Trimesh) -> trimesh.Trimesh:
    return transformed(m, quat_xyzw=Y90)


# ------------------------------------------------------------------------------------------------ render parts
def _srgb(rgb) -> list[float]:
    """Their render COLORS are linear diffuse colours; parts are sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in rgb]


def _paint(geom: trimesh.Trimesh, name: str, label: str = "") -> dict:
    """Part painted with their ``render.COLORS[name]`` = (linear colour, roughness, metallic)."""
    rgb, rough, metal = R.COLORS[name]
    return vis(geom, _srgb(rgb), _line("render.py", f"{name}=(("), rough, metal, name=f"{label or name} (linear->sRGB)")


def _glass(geom: trimesh.Trimesh, key: str, thickness: float) -> dict:
    """Clear dielectric (their ``GLASS`` table: absorption per metre, index): tint = exp(-absorption x wall)."""
    absorption, _ = R.GLASS[key]
    tint = [math.exp(-a * thickness) for a in absorption]
    return vis(geom, tint, _line("render.py", f'"{key}": (('), None, 0.0, name="glass")


def _decal(geom: trimesh.Trimesh, image, uv, cite: str, label: str) -> dict:
    """Printed label / scale (their OmniPBR decal, roughness 0.6) on a patch laid over the object."""
    return vis(geom, [1.0, 1.0, 1.0], cite, 0.6, 0.0, image=image, uv=uv, name=label)


def _patch(radius: float, length: float, arc: float, x0: float, centre: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Their ``wrap_patch`` (cylinder patch about +x) as a mesh with its uv."""
    v, t, uv, _ = R.wrap_patch(radius, length, arc, x0=x0, centre=centre)
    return trimesh.Trimesh(v, t, process=False), uv


def _load(path) -> "object":
    from PIL import Image  # noqa: PLC0415
    return Image.open(path).copy()


def _barrel_visual(kind: str, sp: dict, barrel: trimesh.Trimesh) -> list[dict]:
    """Clear polypropylene barrel with its printed scale (and the flush syringe's label)."""
    ro = 0.5 * sp["od"]
    parts = [_glass(barrel, "barrel_clear", 0.5 * (sp["od"] - sp["bore"]))]
    with tempfile.TemporaryDirectory() as tmp:
        path, width, length = R.scale_map(tmp, kind)
        m, uv = _patch(ro + 0.00004, length, width / ro, -0.003, math.pi / 2)
        parts.append(_decal(m, _load(path), uv, _line("render.py", "S.mesh(base + \"_scale\""), "printed scale"))
        if kind == "flush":
            img = R.label_map(tmp, "flush_label", (0.030, 0.016), [("0.9% Sodium Chloride", 2.6, True), ("Flush  10 mL", 2.6, False),
                                                                   ("training use only", 1.6, False)], band=(30, 90, 170))
            m, uv = _patch(ro + 0.00006, 0.016, 0.030 / ro, sp["barrel_len"] - 0.019, -math.pi / 2)
            parts.append(_decal(m, _load(img), uv, _line("render.py", "S.mesh(base + \"_label\", v, t, uv=uv, normals=nr, mat=_decal(S, \"flush_label\""), "flush label"))
    return parts


def _plunger_visual(kind: str, sp: dict) -> list[dict]:
    """Rod, disc and thumb rest in ``syringe_rod``; the stopper in rubber (grey on the flush syringe)."""
    rod_len, rw = sp["barrel_len"] + sp["flange"][2] + sp["proj"], 0.5 * sp["rod_w"]
    rod = union([box((0.5 * rod_len, rw, rw), (0.5 * rod_len, 0, 0)),
                 _along_x(cylinder(0.5 * sp["thumb_d"], 0.5 * sp["thumb_t"], (0, 0, rod_len + 0.5 * sp["thumb_t"])))])
    stopper = _along_x(cylinder(0.5 * sp["bore"], 0.5 * sp["stopper_len"], (0, 0, 0.5 * sp["stopper_len"])))
    return [_paint(rod, "syringe_rod", "rod and thumb rest"), _paint(stopper, "stopper_grey" if kind == "flush" else "stopper_rubber", "stopper")]


# ------------------------------------------------------------------------------------------------ syringes
def _barrel(s: dict) -> trimesh.Trimesh:
    r, L, lu, fl = 0.5 * s["od"], s["barrel_len"], P.LUER, s["flange"]
    parts = [_along_x(_hollow(r, 0.5 * s["bore"], 0.0, L)),
             _along_x(cylinder(min(0.5 * lu["collar_od"], r), 0.5 * lu["nozzle_len"], (0, 0, -0.5 * lu["nozzle_len"] + 0.0005))),
             box((0.5 * fl[2], 0.5 * fl[0], 0.5 * fl[1]), (L + 0.5 * fl[2], 0, 0))]
    return union(parts)


def _plunger(s: dict) -> trimesh.Trimesh:
    rod_len, rw = s["barrel_len"] + s["flange"][2] + s["proj"], 0.5 * s["rod_w"]
    return union([box((0.5 * rod_len, rw, rw), (0.5 * rod_len, 0, 0)),
                  _along_x(cylinder(0.5 * s["thumb_d"], 0.5 * s["thumb_t"], (0, 0, rod_len + 0.5 * s["thumb_t"]))),
                  _along_x(cylinder(0.5 * s["bore"], 0.5 * s["stopper_len"], (0, 0, 0.5 * s["stopper_len"])))])


def _syringe(x: Exporter, kind: str) -> None:
    s = P.SYRINGES[kind]
    pat = f'{kind}=dict(label=' if kind != "flush" else 'SYRINGES["flush"]'
    cite = _line("params.py", pat)
    mu, partner = _mu("pp", "polypropylene or laminate")
    base = f"syringe_{kind}"
    name = s["label"]
    barrel = _barrel(s)
    x.add(f"{base}__barrel", f"{name} syringe barrel", "tool", barrel, material={"name": "polypropylene"},
          visual=_barrel_visual(kind, s, barrel),
          mass=cited(s["mass_barrel"], "estimate", _line("params.py", "SYRINGE_MASS"), _note("params.py", "_p(\"SYRINGE_MASS\"")),
          friction=mu, friction_partner=partner,
          dims=[dim("finger flange width", s["flange"][0], "y", cite)], source_ref=cite,
          notes="Tube, Luer collar and finger flange (their syringe frame: +x from nozzle to thumb rest). Outline anchored on "
                "measured Terumo syringes and ISO 7886-1; bore from makers' tables.",
          extra={"bore_m": s["bore"], "nominal_ml": s["nominal_ml"], "dead_space_ml": s["dead_ml"]})
    x.add(f"{base}__plunger", f"{name} syringe plunger", "articulated", _plunger(s), material={"name": "polypropylene rod with rubber stopper"},
          visual=_plunger_visual(kind, s),
          mass=cited(s["mass_plunger"], "estimate", _line("params.py", "SYRINGE_MASS"), _note("params.py", "_p(\"SYRINGE_MASS\"")),
          friction=mu, friction_partner=partner, source_ref=cite,
          notes="Rod, thumb rest and stopper; origin at the stopper's front face.",
          extra={"joint": {"type": "prismatic", "parent": f"{base}__barrel", "axis": [1, 0, 0], "range": [0.0, s["stroke_max"]],
                           "cite": cite, "seal_break_N": s["seal_break"], "seal_glide_N": s["seal_glide"]}})


# ------------------------------------------------------------------------------------------------ needles
def _needle(x: Exporter, kind: str) -> None:
    n, hub, sh = P.NEEDLES[kind], P.NEEDLE_HUB, P.NEEDLE_SHEATH
    cite = _line("params.py", f"{kind}=dict(label=")
    Lh, Ln = hub["length"], n["length"]
    tube = _along_x(_hollow(0.5 * n["od"], 0.5 * n["bore"], 0.0, Ln))
    tube = transformed(tube, (-Lh - Ln, 0, 0))
    hubm = _along_x(cylinder(0.5 * hub["od"], 0.5 * Lh, (0, 0, -0.5 * Lh)))
    m_tube = 7900.0 * math.pi * ((0.5 * n["od"]) ** 2 - (0.5 * n["bore"]) ** 2) * Ln
    mu, partner = _mu("steel", "plastic or laminate")
    hub_look = vis(hubm, n["colour"], _line("render.py", 'COLORS["hub_" + _k]'), 0.40, 0.0,
                   name="hub (ISO 6009 colour, sRGB; the render linearises it with 2.2)")
    x.add(f"needle_{kind}", f"Hypodermic needle {n['label']} (hub and cannula)", "tool", union([hubm, tube]),
          visual=[hub_look, _paint(tube, "steel_needle", "cannula")],
          material={"name": "stainless steel cannula in a polypropylene hub"},
          mass=cited(hub["mass"] + m_tube, "derived", _line("devices.py", "m_tube = 7900."),
                     "hub mass 0.25 g (NEEDLE_HUB) + tube 7900 kg/m3 x annulus x length"),
          friction=mu, friction_partner=partner, thickness=cited(0.5 * (n["od"] - n["bore"]), "datasheet", cite,
                                                                 "Hamilton needle gauge chart (nominal od, bore)"),
          dims=[dim("overall length", Lh + Ln, "x", cite), dim("hub outside diameter", hub["od"], "min", _line("params.py", "NEEDLE_HUB = dict"))],
          source_ref=cite, notes="Hub at x<=0, hollow cannula pointing -x (their needle frame: mouth plane at the origin).")
    shl = Ln + sh["overlap"] + 0.004
    sheath = transformed(_along_x(_hollow(0.5 * sh["od"], 0.5 * sh["od"] - sh["wall"], 0.0, shl)), (0, 0, 0))
    rgb = n["colour"] if n["blunt"] else _srgb((.80, .82, .83))
    sheath_look = vis(sheath, rgb, _line("render.py", "rgb = COLORS[\"hub_\" + n.kind][0]"), 0.35, 0.0,
                      name="translucent sheath (55 % opaque, hub colour if blunt)")
    x.add(f"needle_{kind}__sheath", f"Needle sheath for {n['label']}", "tool", sheath, visual=[sheath_look],
          material={"name": "polypropylene"}, mass=cited(sh["mass"], "estimate", _line("params.py", "NEEDLE_SHEATH = dict"), "sheath mass"),
          friction=_mu("pp", "polypropylene")[0], friction_partner="polypropylene",
          thickness=cited(sh["wall"], "estimate", _line("params.py", "NEEDLE_SHEATH = dict"), "sheath wall"),
          source_ref=cite, notes="Hollow cap tube pressed over the hub; 10 mm overlap.")


# ------------------------------------------------------------------------------------------------ vial, pad, lock
def _vial(x: Exporter) -> None:
    v = P.VIAL
    r, H = 0.5 * v["od"], v["height"]
    body_h = H - v["neck_h"] - v["cap_h"]
    outer = revolve([(0, 0), (r, 0), (r, body_h), (0.5 * v["flange_d"] - 0.002, body_h),
                     (0.5 * v["flange_d"] - 0.002, body_h + v["neck_h"]), (0.5 * v["flange_d"] + 0.0003, body_h + v["neck_h"]),
                     (0.5 * v["flange_d"] + 0.0003, H), (0, H)], 64)
    cav = revolve([(0, 0.0012), (r - v["wall"], 0.0012), (r - v["wall"], body_h), (0, body_h)], 64)
    cite = _line("params.py", "VIAL = dict")
    shell = outer.difference(cav, engine="manifold")
    x.add("vial_10r", "Glass vial 10 mL (ISO 8362-1 10R) with stopper and cap", "container", shell,
          visual=_vial_visual(shell, v, body_h),
          material={"name": "borosilicate glass with rubber stopper and plastic cap"},
          mass=cited(v["mass"], "standard", cite, _note("params.py", "_p(\"VIAL_10R\"")),
          friction=cited(P.MU["glass"], "estimate", _line("params.py", 'glass=_p("MU_GLASS"'), "glass vial on laminate or polypropylene"),
          friction_partner="laminate counter or polypropylene", thickness=cited(v["wall"], "standard", cite, "ISO 8362-1 10R, 1 mm wall"),
          dims=[dim("height", H, "z", cite), dim("outside diameter", v["od"], "diameter_xy", cite)], source_ref=cite,
          notes="Body shell, solid neck and cap; the stopper is inside the cap. Mass is their 13 g with closure.")


def _vial_visual(shell: trimesh.Trimesh, v: dict, body_h: float) -> list[dict]:
    """Glass body and neck, crimp cap in ``vial_cap`` above the neck, and the printed label on the body."""
    H, r = v["height"], 0.5 * v["od"]
    top = H - v["cap_h"]
    glass = _slice(shell, top, keep_above=False)
    cap = _slice(shell, top, keep_above=True)
    with tempfile.TemporaryDirectory() as tmp:
        img = R.label_map(tmp, "vial_label", (0.050, 0.022), [("Simulated medication", 3.0, True), ("for training only", 2.4, False),
                                                              ("10 mL  multi-dose", 2.4, False)], band=(25, 70, 160))
        m, uv = _patch(r + 0.00008, 0.022, 0.050 / r, 0.005, -math.pi / 2)
        m.vertices = m.vertices @ np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]).T     # their turn: x -> z, y -> x, z -> y
        label = _decal(m, _load(img), uv, _line("render.py", "S.mesh(base + \"_label\", vv @ turn.T"), "printed label")
    return [_glass(glass, "glass", v["wall"]), _paint(cap, "vial_cap", "crimp cap"), label]


def _slice(m: trimesh.Trimesh, z: float, keep_above: bool) -> trimesh.Trimesh:
    sign = -1.0 if keep_above else 1.0
    return trimesh.intersections.slice_mesh_plane(m, [0, 0, sign], [0, 0, z], cap=True)


def _pad(x: Exporter) -> None:
    p = P.PAD
    thick = sum(P.LAYERS[k]["thick"] for k in ("skin", "fat", "muscle")) + p["board"]
    cite = _line("params.py", "PAD = dict")
    mu, partner = _mu("skin", "plastic")
    x.add("injection_pad", "Injection trainer pad (skin, fat, muscle on a board)", "other",
          box((p["size"][0] / 2, p["size"][1] / 2, thick / 2), (0, 0, thick / 2)), material={"name": "layered silicone trainer pad"},
          visual=_pad_visual(p),
          mass=cited(p["mass"], "estimate", cite, "trainer makers' figures"), friction=mu, friction_partner=partner,
          dims=[dim("side", p["size"][0], "x", cite), dim("thickness", thick, "z", _line("params.py", "LAYERS = dict"))],
          source_ref=cite, notes="One solid block (their pad_block); the layers are stacked 2 + 8 + 30 mm on a 5 mm board.")


def _pad_visual(p: dict) -> list[dict]:
    """Board, muscle, fat and skin layers stacked as their ``render._pad``."""
    hx, hy, z, out = p["size"][0] / 2, p["size"][1] / 2, 0.0, []
    for name, th in (("board", p["board"]), ("muscle", P.LAYERS["muscle"]["thick"]), ("fat", P.LAYERS["fat"]["thick"]),
                     ("skin", P.LAYERS["skin"]["thick"])):
        out.append(_paint(box((hx, hy, th / 2), (0, 0, z + th / 2)), "pad_" + name))
        z += th
    return out


def _lock(x: Exporter) -> None:
    c = P.CONNECTOR
    cite = _line("params.py", "CONNECTOR = dict")
    mu, partner = _mu("pp", "polypropylene")
    body = _along_x(cylinder(0.5 * c["od"], 0.5 * c["length"], (0, 0, -0.5 * c["length"])))
    ring = _along_x(cylinder(0.5 * c["od"] + 0.0005, 0.0025, (0, 0, -0.008)))
    x.add("iv_connector", "Needle-free connector of the saline lock", "other", body,
          visual=[_paint(body, "connector"), _paint(ring, "connector_ring", "luer ring")],
          material={"name": "polycarbonate connector body"}, mass=cited(c["mass"], "estimate", cite, _note("params.py", "_p(\"CONNECTOR\"")),
          friction=mu, friction_partner=partner,
          dims=[dim("length", c["length"], "x", cite), dim("outside diameter", c["od"], "min", cite)], source_ref=cite,
          notes="Solid cylinder drawn as their connector (collider is a capsule of the same length); taped to the skin.")
    clamp = box((0.007, 0.0045, 0.005))
    x.add("iv_clamp", "Slide clamp on the extension tube", "other", clamp, visual=[_paint(clamp, "clamp")], material={"name": "polypropylene", "density":
          cited(1100.0, "estimate", _line("devices.py", "density=1100., mu=P.MU[\"pp\"])"), "clamp body density 1100 kg/m3")},
          friction=mu, friction_partner=partner, source_ref=_line("devices.py", "_clamp_body"),
          notes="Box 14 x 9 x 10 mm; slides 8 mm along the tube against 4 N friction.",
          extra={"joint": {"type": "prismatic", "parent": "world", "axis": [1, 0, 0], "range": [0.0, P.EXTENSION["clamp_travel"]],
                           "cite": _line("params.py", "EXTENSION = dict"), "friction_N": P.EXTENSION["clamp_n"]}})


# ------------------------------------------------------------------------------------------------ furniture
def _open_tray(size: tuple, wall: float) -> trimesh.Trimesh:
    sx, sy, sz = size
    parts = [box((sx / 2, sy / 2, wall / 2), (0, 0, wall / 2))]
    parts += [box((sx / 2, wall / 2, (sz - wall) / 2), (0, s * (sy / 2 - wall / 2), (sz + wall) / 2)) for s in (-1, 1)]
    parts += [box((wall / 2, sy / 2 - wall, (sz - wall) / 2), (s * (sx / 2 - wall / 2), 0, (sz + wall) / 2)) for s in (-1, 1)]
    return union(parts)


def _tray_basin(x: Exporter) -> None:
    mu, partner = _mu("pp", "polypropylene")
    for key, d, label, extra in (("procedure_tray", P.TRAY, "Procedure tray", True), ("basin", P.BASIN, "Basin (a box, not a kidney dish)", False)):
        name = "TRAY" if extra else "BASIN"
        cite = _line("params.py", f"{name} = dict")
        geom = _open_tray(d["size"], d["wall"])
        if extra:
            n = int((d["size"][1] - 2 * d["wall"]) // d["rib_pitch"])
            ribs = [box((0.5 * d["size"][0] - d["wall"], 0.001, 0.5 * d["rib_h"]),
                        (0, (k - 0.5 * (n - 1)) * d["rib_pitch"], d["wall"] + 0.5 * d["rib_h"])) for k in range(n)]
            geom = union([geom] + ribs)
        x.add(key, label, "container", geom, material={"name": "polypropylene"}, visual=[_paint(geom, "tray" if extra else "basin")],
              mass=cited(d["mass"], "authored" if extra else "estimate", cite, _note("params.py", f'_p("{name}"') or "authored prop"),
              friction=mu, friction_partner=partner, thickness=cited(d["wall"], "authored", cite, "wall"),
              dims=[dim("length", d["size"][0], "x", cite), dim("height", d["size"][2], "z", cite)] , source_ref=cite,
              notes="Five walls (floor + four sides)" + (" and 2 mm ribs every 15 mm" if extra else "") + " as their scene.tray / basin.")


def _sharps(x: Exporter) -> None:
    s = P.SHARPS
    (sx, sy, sz), wl = s["size"], s["wall"]
    slx, sly = s["slot"]
    parts = [box((sx / 2, sy / 2, wl / 2), (0, 0, wl / 2))]
    parts += [box((sx / 2, wl / 2, (sz - 2 * wl) / 2), (0, t * (sy / 2 - wl / 2), sz / 2)) for t in (-1, 1)]
    parts += [box((wl / 2, sy / 2 - wl, (sz - 2 * wl) / 2), (t * (sx / 2 - wl / 2), 0, sz / 2)) for t in (-1, 1)]
    sy0 = -sy / 2 + 0.020
    sy1 = sy0 + sly
    top = sz - wl / 2
    parts += [box((sx / 2, (sy0 + sy / 2) / 2, wl / 2), (0, (sy0 - sy / 2) / 2, top)),
              box((sx / 2, (sy / 2 - sy1) / 2, wl / 2), (0, (sy1 + sy / 2) / 2, top)),
              box(((sx - slx) / 4, sly / 2, wl / 2), (-(sx + slx) / 4, (sy0 + sy1) / 2, top)),
              box(((sx - slx) / 4, sly / 2, wl / 2), ((sx + slx) / 4, (sy0 + sy1) / 2, top))]
    cite = _line("params.py", "SHARPS = dict")
    lid = vis(union(parts[5:]), _srgb((.86, .86, .83)), _line("render.py", 'S.mats["sharps_lid"]'), 0.25, 0.0,
              name="lid (translucent, 88 % opaque; linear->sRGB)")
    x.add("sharps_container", "Counter-standing sharps container with drop slot", "container", union(parts), motion="static",
          visual=[_paint(union(parts[:5]), "sharps_red", "body"), lid],
          material={"name": "red polypropylene, 3 mm walls"}, mass=cited(s["mass"], "estimate", cite, _note("params.py", '_p("SHARPS"')),
          friction=_mu("pp", "polypropylene")[0], friction_partner="syringes",
          thickness=cited(wl, "estimate", cite, "container makers' figures"),
          dims=[dim("height", sz, "z", cite), dim("length", sx, "x", cite)], source_ref=cite,
          notes="Floor, four walls and the four-piece slotted lid; the counterweighted flap is not exported.")


def _counter(x: Exporter) -> None:
    cl, cd, ct, cz = P.COUNTER_LENGTH, P.COUNTER_DEPTH, P.COUNTER_TOP_THICK, P.COUNTER_Z
    parts = [box((cl / 2, cd / 2, ct / 2), (0, 0, -ct / 2)), box((cl / 2, 0.01, 0.05), (0, cd / 2 - 0.01, 0.05)),
             box((cl / 2, (cd - 0.025) / 2, (cz - ct - 0.10) / 2), (0, 0.0125, -ct - (cz - ct - 0.10) / 2)),
             box((cl / 2, (cd - 0.075) / 2, 0.05), (0, 0.0375, -cz + 0.05))]
    cite = _line("params.py", "COUNTER_Z")
    mu = cited(P.MU["laminate"], "estimate", _line("params.py", 'laminate=_p("MU_LAMINATE"'), "counter top against plastic props")
    look = [_paint(union(parts[:2]), "solid_surface", "top and upstand"), _paint(parts[2], "casework"), _paint(parts[3], "toe_kick")]
    x.add("counter", "Ward counter with casework (solid-surface top)", "furniture", union(parts), motion="static", visual=look,
          material={"name": "solid-surface top on casework"}, friction=mu, friction_partner="plastic props",
          dims=[dim("top thickness", ct, "z", _line("params.py", "COUNTER_TOP_THICK")) if False else dim("depth", cd, "y", _line("params.py", "COUNTER_DEPTH"))],
          source_ref=_line("scene.py", '"counter_top"'),
          notes="Top slab, upstand, casework and toe kick as their scene.room spans; origin at the top surface, back edge.")


def export(x: Exporter) -> None:
    steps = [(f"syringe_{k}", (lambda k=k: _syringe(x, k))) for k in ("s1", "s3", "s10", "flush")]
    steps += [(f"needle_{k}", (lambda k=k: _needle(x, k))) for k in P.NEEDLES]
    steps += [("vial", lambda: _vial(x)), ("pad", lambda: _pad(x)), ("lock", lambda: _lock(x)),
              ("tray_basin", lambda: _tray_basin(x)), ("sharps", lambda: _sharps(x)), ("counter", lambda: _counter(x))]
    for key, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            x.fail(key, f"{type(exc).__name__}: {exc}")
            log.exception("%s failed", key)
    x.fail("training_arm", "static board, hand and forearm capsules of the IV training arm are a prop rig; not exported")
