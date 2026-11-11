# SPDX-License-Identifier: Apache-2.0
"""ingredient_mixing: vessels (revolved walls), hand tools (articulated parts), mixer, hob knob, packs and the counter.

Geometry follows their ``geometry.py`` (``wall_polygon`` revolved: the wall, not the closed outline) and the shapes
their ``tools.py`` / ``scene.py`` place; masses, frictions and sizes come from the parameter ledger ``params.LEDGER``
(each entry's own label and basis text is the citation).
"""

from __future__ import annotations

import inspect
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from _common import Exporter, box, capsule, cited, cylinder, dim, line_of, mesh, transformed, union, vis, where

from ingredient_mixing import geometry as G
from ingredient_mixing import dressing as DR
from ingredient_mixing import kitchen as KIT
from ingredient_mixing import params as P
from ingredient_mixing import render as R
from ingredient_mixing import scene as S
from ingredient_mixing import tools as T

# their ledger labels -> the spec's basis names
LABEL = {"read": "literature-read", "search": "search", "graph": "literature-read", "derived": "derived",
         "estimate": "estimate", "open": "estimate", "authored": "authored", "repo": "authored", "literature": "literature",
         "measured": "measured", "engine": "authored"}
LEDGER = {e["name"]: e for e in P.LEDGER}
GEN_Z = (0.0, 0.0, 1.0)


# ------------------------------------------------------------------------------------------------ citations
def led(name: str, value: float | None = None) -> dict:
    """A ledger parameter as a cited number: their label, their basis text, the line of ``_p("<name>"``."""
    e = LEDGER[name]
    quote = " ".join(str(e["basis"]).split())[:300]
    return cited(e["value"] if value is None else value, LABEL[e["label"]], line_of(P, f'_p("{name}"'), quote)


def literal(value: float, pattern: str, basis: str = "authored", quote: str = "", module=P) -> dict:
    """A bare constant of their code (no ledger entry): cited by the line that holds ``pattern``."""
    return cited(value, basis, line_of(module, pattern), quote)


def comment_block(pattern: str) -> str:
    """The ``#`` comment lines just above the first line of params.py that contains ``pattern``."""
    lines = Path(inspect.getsourcefile(P)).read_text().splitlines()
    idx = next(i for i, ln in enumerate(lines) if pattern in ln)
    out = []
    while idx > 0 and lines[idx - 1].lstrip().startswith("#"):
        idx -= 1
        out.insert(0, lines[idx].lstrip("# ").strip())
    return re.sub(r"\s+", " ", " ".join(out))[:300]


def mu_cited(name: str) -> dict:
    return led(f"MU_{name.upper()}")


# ------------------------------------------------------------------------------------------------ render looks
def lin2srgb(c) -> list[float]:
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return [float(v) for v in np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)]


def paint(geom: trimesh.Trimesh, material: str, what: str = "") -> dict:
    """A render part in their ``render.COLORS[material]`` (linear diffuse, roughness, metallic) or the glass dielectric."""
    if material == "glass":  # Beer-Lambert tint through the 4 mm wall their comment quotes
        tint = lin2srgb(np.exp(-np.asarray(R.GLASS["glass"][0]) * 0.004))
        return vis(geom, tint, line_of(R, "GLASS = {"), 0.0, 0.0, name=f"glass {what} (4 mm transmittance, linear->sRGB)".replace("  ", " "))
    rgb, rough, metal = R.COLORS[material]
    return vis(geom, lin2srgb(rgb), line_of(R, f"{material}=(("), rough, metal, name=f"{what or material} (linear->sRGB)")


def uv_plane(geom: trimesh.Trimesh, tile_m: float) -> np.ndarray:
    """Per-vertex uv of a world-planar mapping (x, y over the tile), as ``S.box(..., uv_per_m=1.)`` with a 0.40 m tile."""
    return np.asarray(geom.vertices)[:, :2] / tile_m


def quartz_image():
    """Their generated pale-quartz diffuse map (``dressing._quartz_map``), loaded back from a scratch folder."""
    from PIL import Image  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        path = DR._quartz_map(tmp)["diff"]
        return Image.open(path).convert("RGB").copy()


def label_part(name: str, title: str, lines, band, bg, hx: float, hy: float, centre_z: float, w: float, h: float) -> dict:
    """The printed front label of a pack (``kitchen.label_map`` artwork) as a quad on the -y face, as ``kitchen.panel``."""
    from PIL import Image  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(KIT.label_map(tmp, name, title, lines, bg=bg, band=band)).convert("RGB").copy()
    y, z = -hy - 0.0008, centre_z
    quad = trimesh.Trimesh([[-w / 2, y, z - h / 2], [w / 2, y, z - h / 2], [w / 2, y, z + h / 2], [-w / 2, y, z + h / 2]],
                           [[0, 2, 1], [0, 3, 2]], process=False)
    return vis(quad, (1.0, 1.0, 1.0), line_of(KIT, "def label_map"), 0.5, 0.0, image=img, uv=[(0, 0), (1, 0), (1, 1), (0, 1)],
               name=f"printed label ({title.lower()})")


# ------------------------------------------------------------------------------------------------ mesh helpers
def wall_solid(profile: np.ndarray, wall: float) -> trimesh.Trimesh:
    """Their wall of a vessel: the inner profile offset outward by ``wall``, revolved (watertight)."""
    v, f = G.revolve_mesh(G.wall_polygon(profile, wall), segments=64)
    return mesh(v, f)


def tube(points: np.ndarray, radius: float) -> list[trimesh.Trimesh]:
    """A polyline as a chain of capsules (one per segment)."""
    out = []
    for a, b in zip(points[:-1], points[1:]):
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        rot = T.rot_z_to(d / n).as_quat()
        out.append(transformed(trimesh.creation.capsule(n, radius, count=[6, 6]), 0.5 * (a + b), tuple(rot)))
    return out


def sweep_tube(points: np.ndarray, radius: float, sides: int = 10) -> trimesh.Trimesh:
    """A closed tube of ``radius`` swept along a smooth polyline (parallel-transport frames, flat end caps)."""
    pts = np.asarray(points, float)
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    n = np.cross(tan[0], [1.0, 0.0, 0.0] if abs(tan[0][0]) < 0.9 else [0.0, 1.0, 0.0])
    n /= np.linalg.norm(n)
    rings = []
    for p, t in zip(pts, tan):
        n = n - t * float(n @ t)
        n /= np.linalg.norm(n)
        b = np.cross(t, n)
        th = np.linspace(0, 2 * math.pi, sides, endpoint=False)
        rings.append(p + radius * (np.outer(np.cos(th), n) + np.outer(np.sin(th), b)))
    verts = np.vstack([*rings, pts[:1], pts[-1:]])
    faces = []
    for i in range(len(pts) - 1):
        for j in range(sides):
            a, b2, c, d = i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j
            faces += [[a, b2, c], [a, c, d]]
    c0, c1 = len(pts) * sides, len(pts) * sides + 1
    for j in range(sides):
        faces += [[c0, (j + 1) % sides, j], [c1, (len(pts) - 1) * sides + j, (len(pts) - 1) * sides + (j + 1) % sides]]
    return mesh(verts, faces)


def wire_cage(profile_rz: np.ndarray, angles: list[float], radius: float, n: int = 24, r_min: float = 0.0025) -> trimesh.Trimesh:
    """Separate wires along an (r, z) profile, one per angle about z, each a closed tube. Wires stop where the
    profile narrows to ``r_min`` so that neighbours do not merge (the loops close on the axis in the real whisk)."""
    zs = np.linspace(profile_rz[0, 1], profile_rz[-1, 1], n)
    rs = np.interp(zs, profile_rz[:, 1], profile_rz[:, 0])
    keep = rs >= r_min
    zs, rs = zs[keep], rs[keep]
    tubes = [sweep_tube(np.column_stack([rs * math.cos(a), rs * math.sin(a), zs]), radius) for a in angles]
    return trimesh.util.concatenate(tubes)


def handle_capsule(length: float, radius: float) -> trimesh.Trimesh:
    """Their ``tools.handle``: a capsule that spans z = 0 .. length along the tool axis."""
    return capsule(radius, 0.5 * (length - 2 * radius), pos=(0, 0, 0.5 * length))


def round_prism(radius_flat: float, height: float) -> trimesh.Trimesh:
    """Their ``disc`` / round prop: twelve flats at ``radius_flat`` from the axis, z = 0 .. height."""
    return cylinder(radius_flat / math.cos(math.pi / 12), 0.5 * height, pos=(0, 0, 0.5 * height), sections=12)


def joint(jtype: str, parent: str, axis, rng, cite: str, anchor=None) -> dict:
    out = {"type": jtype, "parent": parent, "axis": axis, "range": rng, "cite": cite}
    if anchor is not None:
        out["anchor_in_parent_m"] = [float(v) for v in anchor]
    return out


def cap_quat_y(angle: float) -> tuple:
    return (0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle))


# ------------------------------------------------------------------------------------------------ vessels
def _vessel(x: Exporter, key: str, name: str, spec: dict, prof: np.ndarray, wall: float, tag: str, mass: dict,
            mat: dict, mu: dict, partner: str, extra_parts=None, dims=None, notes: str = "",
            paints: tuple = ("bowl_steel", None)) -> None:
    body = wall_solid(prof, wall)
    seen = [paint(body, paints[0], "wall")]
    if extra_parts:
        extras = trimesh.util.concatenate(extra_parts)
        body = union([body, *extra_parts])
        seen = [paint(wall_solid(prof, wall), paints[0], "wall"), paint(extras, paints[1] or paints[0], "added parts")]
    x.add(key, name, "container", body, visual=seen, material=mat, mass=mass, friction=mu, friction_partner=partner,
          dims=dims or [], source_ref=where(S.Scene.station), notes=notes,
          extra={"inner_profile_rz_m": np.round(prof, 5).tolist(), "wall_m": wall})


def _bowl5(x: Exporter) -> None:
    s = P.BOWL5
    prof = G.bowl_profile(s["rim_r"], s["depth"], s["base_r"])
    cite = line_of(P, "BOWL5 = dict(")
    _vessel(x, "bowl", "Steel mixing bowl 5 qt (Vollrath 69050 class)", s, prof, 0.0012, "bowl5", led("BOWL5_MASS"),
            {"name": "stainless steel", "density": literal(P.STEEL, "STEEL = ", "estimate", "stainless steel density")},
            led("MU_STEEL_STEEL"), "whisk wire or spoon on the bowl",
            dims=[dim("rim inside diameter", 2 * s["rim_r"], "diameter_xy", cite), dim("height", s["depth"], "z", cite)],
            notes="Wall 1.2 mm (their render wall, scene.py render_wall) revolved from the inner profile; the rigid sim "
                  "uses 3 mm boxes and a 5 mm rim bead that are not drawn here.")


def _bowl2(x: Exporter) -> None:
    s = P.BOWL2
    prof = G.bowl_profile(s["rim_r"], s["depth"], s["base_r"])
    cite = line_of(P, "BOWL2 = dict(")
    mass = led("BOWL2_MASS")
    _vessel(x, "glass_bowl", "Glass mixing bowl 1.5 L", s, prof, s["wall"], "bowl2", mass,
            {"name": "pressed glass", "density": literal(2500.0, '_p("BOWL2_MASS"', "estimate",
                                                          "glass volume x 2500 kg/m3 of this shape")},
            led("MU_GLASS"), "glass on stone, steel on glass",
            dims=[dim("rim inside diameter", 2 * s["rim_r"], "diameter_xy", cite), dim("height with base wall", s["depth"] + s["wall"], "z", cite)],
            paints=("glass", None), notes="Wall 4 mm (BOWL2_WALL). Their ledger calls the 0.6 kg mass 'open': no maker weight was found.")


def _pan_handle() -> trimesh.Trimesh:
    s = P.PAN
    a = math.radians(215.0)
    r0 = s["radius"] + s["wall"] + s["handle_r"]
    p0 = np.array([r0 * math.cos(a), r0 * math.sin(a), s["depth"] - 0.012])
    p1 = np.array([(r0 + s["handle_len"]) * math.cos(a), (r0 + s["handle_len"]) * math.sin(a), s["depth"] + 0.030])
    return tube(np.array([p0, p1]), s["handle_r"])[0]


def _saucepan(x: Exporter) -> None:
    s = P.PAN
    prof = G.cylinder_profile(s["radius"], s["depth"], corner=s["corner"])
    _vessel(x, "pot", "Saucepan 1.5 qt (All-Clad D3 class)", s, prof, s["wall"], "pan", led("PAN_MASS"),
            {"name": "tri-ply stainless steel over aluminium core", "density": literal(P.STEEL, "STEEL = ", "estimate",
                                                                                      "steel density")},
            led("MU_STEEL_STEEL"), "whisk wire or spoon on the pan", extra_parts=[_pan_handle()], paints=("bowl_steel", "steel_brushed"),
            notes="Wall revolved from the inner profile plus the handle capsule (scene.py:321-330); mass includes the "
                  "0.12 kg handle. No inside-diameter dim: the handle is in the same mesh.")


def _crock(x: Exporter) -> None:
    s = P.POT
    prof = G.cylinder_profile(s["radius"], s["depth"], corner=0.006)
    walls = [box((s["radius"] * 0.5, 0.0015, 0.5 * s["depth"]),
                 tuple(np.array([math.cos(a), math.sin(a), 0.0]) * 0.5 * s["radius"] * 0.5 + (0, 0, 0.5 * s["depth"])),
                 (0, 0, math.sin(0.5 * a), math.cos(0.5 * a)))
             for a in (2 * math.pi * k / 3 + math.pi / 2 for k in range(3))]
    _vessel(x, "crock", "Ceramic utensil crock", s, prof, s["wall"], "pot", led("POT_MASS"),
            {"name": "stoneware"}, literal(P.MU["default"], "MU = dict(", "estimate", "unlisted pairs"), "tools and counter",
            extra_parts=walls, paints=("ceramic", None),
            dims=[dim("outside diameter (inside 180 mm + walls)", 2 * (s["radius"] + s["wall"]), "diameter_xy", line_of(P, "POT = dict("))],
            notes="Revolved 6 mm wall plus the three-way radial divider (tools.py Vessel divider); cylinder inside "
                  "diameter 180 mm is an estimate in their ledger.")


# ------------------------------------------------------------------------------------------------ hand tools
def _whisk(x: Exporter) -> None:
    s = P.WHISK
    hl, cl = s["handle_len"], s["cage_len"]
    m_cage = 0.3 * s["mass"]
    q = "mass split: cage = 0.3 x WHISK_MASS (tools.py Whisk), handle the rest"
    tl = line_of(T, "m_cage = 0.3")
    mu = literal(P.MU["default"], "MU = dict(", "estimate", "unlisted pairs")
    grip = handle_capsule(hl, s["handle_r"])
    x.add("whisk__handle", "Balloon whisk handle (OXO 11.5 in)", "articulated", grip, visual=[paint(grip, "tool_handle", "handle")],
          material={"name": "soft-grip handle over steel"}, mass=cited(s["mass"] - m_cage, "derived", tl, q),
          friction=mu, friction_partner="gripper pads, counter",
          dims=[dim("handle length", hl, "z", line_of(P, '_p("WHISK_HANDLE_LEN"'))], source_ref=where(T.Whisk),
          notes="Handle only; the sprung wire cage is the part whisk__cage.", extra={"tool_frame": "origin at the butt, +z to the cage"})
    cage = np.array([(r * s["cage_r"] / 0.032, z * cl / 0.175) for r, z in T.Whisk.CAGE])
    angles = [math.pi * k / int(s["loops"]) for k in range(2 * int(s["loops"]))]
    wires = wire_cage(cage, angles, 0.5 * s["wire"])
    x.add("whisk__cage", "Balloon whisk wire cage", "articulated", wires, visual=[paint(wires, "steel_wire", "wires")],
          material={"name": "stainless steel wire"}, mass=cited(m_cage, "derived", tl, q),
          friction=literal(P.MU["steel_steel"], '_p("MU_STEEL_STEEL"', "estimate", "whisk wire or spoon on a steel bowl"),
          friction_partner="steel bowl or pan",
          dims=[dim("cage length", cl, "z", line_of(P, '_p("WHISK_CAGE_LEN"')),
                dim("cage diameter", 2 * s["cage_r"], "diameter_xy", line_of(P, '_p("WHISK_CAGE_R"'))],
          source_ref=where(T.Whisk), notes="Ten 1.4 mm wires on the cage profile (tools.py Whisk.CAGE), as capsule chains.",
          extra={"joint": joint("bend_2axis", "whisk__handle", [[1, 0, 0], [0, 1, 0]], [-0.6, 0.6], line_of(T, "limit_lower=-0.6"),
                                (0, 0, hl))})


def _spatula(x: Exporter) -> None:
    s = P.SPATULA
    bw, bl = s["blade"]
    t, hr = s["blade_t"], s["handle_r"]
    hl = s["length"] - bl
    m_tip = 0.015
    v_root = bw * t * 0.5 * bl
    dens = (s["mass"] - m_tip) / (math.pi * hr * hr * hl + v_root)
    ml = line_of(P, '_p("SPATULA_MASS"')
    mu = literal(P.MU["silicone"] * 0.5, "mu=P.MU[\"silicone\"]*0.5", "authored", "silicone blade, half of MU_SILICONE 0.9", T)
    handle = union([handle_capsule(hl, hr), box((0.5 * bw, 0.5 * t, 0.25 * bl), (0, 0, hl + 0.25 * bl))])
    x.add("spatula__handle", "Silicone spatula handle and blade root (GIR 11 in)", "articulated", handle,
          visual=[paint(handle, "silicone_red", "silicone")],
          material={"name": "silicone over nylon core"},
          mass=cited(dens * (math.pi * hr * hr * hl + v_root), "derived", ml, "SPATULA_MASS less the 15 g tip, by volume"),
          friction=mu, friction_partner="bowl or pan", dims=[dim("length to hinge", hl + 0.5 * bl, "z", ml)],
          source_ref=where(T.Spatula), notes="Handle plus the fixed half of the blade; the sprung tip is spatula__tip.")
    tip = box((0.5 * bw, 0.4 * t, 0.25 * bl), (0, 0, 0.25 * bl))
    x.add("spatula__tip", "Silicone spatula blade tip", "articulated", tip, visual=[paint(tip, "silicone_red", "silicone")], material={"name": "silicone"},
          mass=literal(m_tip, "m_tip = 0.015", "authored", "tip mass", T), friction=mu, friction_partner="bowl or pan",
          source_ref=where(T.Spatula), notes="Tip half of the blade on a sprung hinge across its width.",
          extra={"joint": joint("revolute", "spatula__handle", [1, 0, 0], [-0.9, 0.9], line_of(T, "limit_lower=-0.9"),
                                (0, 0, hl + 0.5 * bl))})


def _paddle(x: Exporter, key: str, name: str, spec: dict, end: trimesh.Trimesh, ledger: str, notes: str) -> None:
    hl = spec["length"] - (spec["bowl"] if "bowl" in spec else spec["blade"])[1]
    body = union([handle_capsule(hl, spec["handle_r"]), end])
    x.add(key, name, "tool", body, visual=[paint(body, "wood", "wood (fallback colour of the veneer texture)")],
          material={"name": "wood"}, mass=led(ledger),
          friction=led("MU_WOOD_STEEL"), friction_partner="steel bowl or pan",
          dims=[dim("length", spec["length"], "z", line_of(P, f'_p("{ledger.split("_")[0]}_LENGTH"'))],
          source_ref=where(T.Paddle), notes=notes)


def _spoon(x: Exporter) -> None:
    s = P.SPOON
    bw, bl = s["bowl"]
    hl = s["length"] - bl
    parts = [box(h, (c[0], c[1], hl + 0.5 * bl + c[2]), cap_quat_y(a))
             for _, c, h, a, _ in T.dish_boxes(bw, bl, s["bowl_t"], s["bowl_depth"], s["bowl_rim"])]
    _paddle(x, "wooden_spoon", "Wooden spoon (Choice 12 in)", s, union(parts), "SPOON_MASS",
            "Round shank plus the octagonal dish of boxes (their dish_boxes) with the 7 mm hollow toward +y.")


def _stirrer(x: Exporter) -> None:
    s = P.STIRRER
    bw, bl = s["blade"]
    end = box((0.5 * bw, 0.5 * s["blade_t"], 0.5 * bl), (0, 0, s["length"] - bl + 0.5 * bl))
    _paddle(x, "wooden_stirrer", "Flat wooden stirrer", s, end, "STIRRER_MASS", "Round shank plus a flat 40x90x6 mm blade.")


def _thermometer(x: Exporter) -> None:
    s = P.THERMOMETER
    hl = s["handle_len"]
    grip, stem = handle_capsule(hl, s["handle_r"]), tube(np.array([[0, 0, hl], [0, 0, s["length"] - 0.002]]), 0.002)[0]
    body = union([grip, stem])
    x.add("thermometer", "Probe thermometer", "tool", body,
          visual=[paint(grip, "thermo_body", "body"), paint(stem, "steel_wire", "stem")], material={"name": "plastic body, steel stem"},
          mass=led("THERMOMETER_MASS"), friction=literal(P.MU["default"], "MU = dict(", "estimate", "unlisted pairs"),
          friction_partner="counter", dims=[dim("length", s["length"], "z", line_of(P, '_p("THERMOMETER_LENGTH"'))],
          source_ref=where(T.Thermometer), notes="Round body and 4 mm stem (tools.py Thermometer); the sensor is not modelled.")


# ------------------------------------------------------------------------------------------------ mixer and hob
def _beater() -> trimesh.Trimesh:
    s = P.MIXER
    n = 12
    u = (np.arange(n) + 0.5) / n
    zz = -s["shaft_len"] - u * s["beater_len"]
    rr = s["beater_r"] * np.sin(math.pi * np.clip(0.08 + 0.84 * u, 0, 1)) ** 0.6
    prof = np.vstack([[0.0, -s["shaft_len"]], np.column_stack([rr, zz]), [0.0, -s["shaft_len"] - s["beater_len"]]])
    strips = wire_cage(prof[::-1], [2 * math.pi * q / int(s["strips"]) for q in range(int(s["strips"]))], 0.5 * s["strip"])
    shaft = tube(np.array([[0, 0, -0.004], [0, 0, -s["shaft_len"]]]), 0.003)[0]
    return trimesh.util.concatenate([strips, shaft])


def _mixer(x: Exporter) -> None:
    s = P.MIXER
    hx, hy, hz = (0.5 * v for v in s["body"])
    hzh = hz + 0.045
    house = union([box((hx, hy, hz)), *tube(np.array([[-0.072, 0, hzh], [0.062, 0, hzh]]), s["handle_r"]),
                   box((0.012, 0.012, 0.024), (-0.072, 0, hz + 0.022)), box((0.012, 0.012, 0.024), (0.062, 0, hz + 0.022))])
    ml = line_of(P, '_p("MIXER_MASS"')
    mu = literal(P.MU["default"], "MU = dict(", "estimate", "unlisted pairs")
    tl = line_of(T, "m_house = ")
    x.add("hand_mixer__housing", "Hand mixer housing and handle (KitchenAid KHM5AP class)", "articulated", house,
          visual=[paint(house, "mixer_white", "housing")],
          material={"name": "ABS plastic housing"},
          mass=cited(s["mass"] - 2 * 0.04 - 0.01, "derived", tl, "KHM5AP net weight 2.3 lb less beaters and switch (tools.py)"),
          friction=mu, friction_partner="counter",
          source_ref=where(T.Mixer), notes="Housing box 188x88x110 mm, loop handle and two struts; beaters and switch are separate parts.",
          extra={"total_mass_kg": s["mass"], "frame": "origin housing centre, +x beater end, +z handle"})
    beater = _beater()
    x.add("hand_mixer__beater", "Hand mixer beater", "articulated", beater, visual=[paint(beater, "steel_wire", "strips")], material={"name": "stainless steel strips"},
          mass=literal(0.04, "m_beater = 0.04", "authored", "beater mass", T), friction=literal(
              P.MU["steel_steel"], '_p("MU_STEEL_STEEL"', "estimate", "whisk wire or spoon on a steel bowl"),
          friction_partner="steel bowl", dims=[dim("beater head diameter plus strip", 2 * s["beater_r"] + s["strip"], "diameter_xy",
                                                    line_of(P, '_p("MIXER_BEATER_R"'))],
          source_ref=where(T.Mixer), notes="Four bowed strips and a shaft; the second beater is the same part at y = -0.02 m.",
          extra={"joint": joint("revolute", "hand_mixer__housing", [0, 0, 1], [], line_of(T, "add_joint_revolute(H, Bk"),
                                (0.045, 0.5 * s["beater_gap"], -hz)), "instances": 2})
    tab = box((0.006, 0.009, 0.004))
    x.add("hand_mixer__switch", "Hand mixer speed switch", "articulated", tab, visual=[paint(tab, "mixer_grey", "tab")],
          material={"name": "plastic"}, mass=literal(0.004, "mass=0.004", "authored", "switch tab mass", T),
          friction=led("MU_PAD"), friction_partner="finger pad", source_ref=where(T.Mixer),
          notes="Slide tab with OFF and five detents.",
          extra={"joint": joint("prismatic", "hand_mixer__housing", [1, 0, 0], [-0.0005, 5 * s["detent_pitch"] + 0.0005],
                                line_of(T, "add_joint_prismatic"), (0.050, 0, hzh + s["handle_r"] + 0.004))})


def _hob(x: Exporter) -> None:
    h = P.HOB
    plate = box((0.16, 0.26, 0.002))
    x.add("hob__top", "Induction hob glass top (Duxtop class)", "fixture", plate, visual=[paint(plate, "hob_glass", "hob glass")],
          motion="static",
          material={"name": "ceramic glass"}, source_ref=where(S.Scene.station),
          dims=[dim("width", 0.32, "x", line_of(S, "hob_glass"))],
          notes="Visual-only plate in their scene (collide=False); size from scene.py. No mass or friction in their code.")
    pitch = math.radians(h["detent_deg"])
    cap = cylinder(h["knob_r"], 0.5 * (h["knob_h"] - 0.0015), pos=(0, 0, 0.0015 + 0.5 * (h["knob_h"] - 0.0015)), sections=24)
    grip = box((0.020, 0.005, 0.007), (0, 0, h["knob_h"] + 0.007))
    x.add("hob__knob", "Hob rotary knob", "articulated", union([cap, grip]), visual=[paint(trimesh.util.concatenate([cap, grip]), "knob_black", "knob")],
          material={"name": "black plastic"},
          mass=literal(0.06, "mass=0.06", "authored", "knob mass", T), friction=led("MU_PAD"), friction_partner="finger pad",
          dims=[dim("knob diameter", 2 * h["knob_r"], "diameter_xy", line_of(P, "knob_r=0.022"))], source_ref=where(T.HobKnob),
          notes="Cap stands 1.5 mm clear of the counter; detent every 15 degrees (21 positions).",
          extra={"joint": joint("revolute", "world", [0, 0, -1], [-0.02, 20 * pitch + 0.02], line_of(T, "limit_lower=-0.02"))})


# ------------------------------------------------------------------------------------------------ packs, counter
PROPS = {"flour_bag": ("flour_pack", "Flour pack (1 kg paper bag)"), "sugar_jar": ("sugar_pack", "Sugar jar (600 g)"),
         "milk_carton": ("milk_carton", "Milk carton 1 L"), "egg_carton": ("egg_carton", "Egg carton (six eggs)"),
         "cake_box": ("cake_mix_pack", "Cake mix box (432 g)")}


# kind -> (kitchen.props material name, linear rgb, roughness, label (name, title, lines, band, bg) or None)
PACK_LOOK = {
    "flour_bag": ("flour_paper", (.86, .83, .76), .9, ("flour", "FLOUR", ("plain wheat flour", "all-purpose", "1 kg"), (36, 78, 150),
                                                       (245, 242, 232))),
    "sugar_jar": ("sugar_goods", (.93, .93, .91), .55, None),
    "milk_carton": ("carton_white", (.90, .90, .88), .45, ("milk", "MILK", ("whole", "3.5 % fat", "1 litre"), (30, 100, 170),
                                                          (250, 250, 248))),
    "egg_carton": ("egg_pulp", (.52, .50, .45), .95, None),
    "cake_box": ("cakebox", (.86, .68, .25), .55, ("cake", "CAKE MIX", ("yellow sponge", "add water, oil, 3 eggs", "432 g"),
                                                    (150, 60, 20), (250, 236, 190))),
}


def _pack_look(kind: str, geom: trimesh.Trimesh, hx: float, hy: float, h: float) -> list[dict]:
    """The pack colour their ``kitchen.props`` gives it, plus its printed front label (their artwork) where it has one."""
    mat, rgb, rough, label = PACK_LOOK[kind]
    parts = [vis(geom, lin2srgb(rgb), line_of(KIT, f'"{mat}"'), rough, 0.0, name=f"{mat} (linear->sRGB)")]
    if label is not None:
        w, hh, cz = (1.75 * hx, 0.72 * (h - 0.03), 0.47 * (h - 0.03)) if kind == "flour_bag" else (
            (1.9 * hx, 0.9 * (h - 0.05), 0.5 * (h - 0.05)) if kind == "milk_carton" else (1.9 * hx, 0.92 * h, 0.5 * h))
        parts.append(label_part(*label[:2], label[2], label[3], label[4], hx, hy, cz, w, hh))
    return parts


def _prop(x: Exporter, kind: str, key: str, name: str) -> None:
    d = P.PROP[kind]
    (hx, hy), h, m = d["half"], d["h"], d["mass"]
    if d.get("round"):
        geom, dims = round_prism(hx, h), [dim("height", h, "z", line_of(P, f"{kind}=dict("))]
    else:
        geom = box((hx, hy, 0.5 * h), (0, 0, 0.5 * h))
        dims = [dim("height", h, "z", line_of(P, f"{kind}=dict(")), dim("width", 2 * hx, "x", line_of(P, f"{kind}=dict("))]
    mu = literal(P.MU["default"], "MU = dict(", "estimate", "unlisted pairs")
    x.add(key, name, "food", geom, visual=_pack_look(kind, geom, hx, hy, h), material={"name": "packaged food and its container"},
          mass=cited(m, "estimate", line_of(P, f"{kind}=dict("), comment_block("footprint half sizes")),
          friction=mu, friction_partner="quartz counter top", dims=dims, source_ref=where(S.Scene.props),
          notes="Their collision body: a box (or twelve-sided prism for round packs) of the pack outline; contents are "
                "inside the mass, not modelled.")


def _counter(x: Exporter) -> None:
    cl, cd, ct, cz = P.COUNTER_LENGTH, P.COUNTER_DEPTH, P.COUNTER_TOP_THICK, P.COUNTER_Z
    parts = [box((0.5 * cl, 0.5 * cd, 0.5 * ct), (0, 0.5 * cd, cz - 0.5 * ct)),
             box((0.5 * cl, 0.5 * cd - 0.0125, 0.5 * (cz - ct) - 0.05), (0, 0.5 * cd + 0.0125, 0.5 * (cz - ct) + 0.05)),
             box((0.5 * cl, 0.5 * cd - 0.05, 0.05), (0, 0.5 * cd + 0.05, 0.05))]
    top_uv = uv_plane(parts[0], 0.40)
    seen = [vis(parts[0], (0.845, 0.845 * 0.985, 0.845 * 0.955), line_of(DR, "img = 0.845"), 0.22, 0.0, image=quartz_image(), uv=top_uv,
                name="quartz top (generated speckle texture)"),
            paint(parts[1], "cabinet", "carcass"), paint(parts[2], "toe_kick", "toe kick")]
    x.add("counter", "Teaching-kitchen counter run", "furniture", union(parts), visual=seen, motion="static",
          material={"name": "engineered quartz top on a cabinet carcass"},
          friction=led("MU_STEEL_QUARTZ"), friction_partner="polished steel",
          dims=[dim("height", cz, "z", line_of(P, '_p("COUNTER_Z"')), dim("depth", cd, "y", line_of(P, '_p("COUNTER_DEPTH"')),
                dim("length", cl, "x", line_of(P, '_p("COUNTER_LENGTH"'))],
          source_ref=where(S.Scene.room), notes="Top, carcass and toe kick as one solid; origin on the floor at the front edge, centred in x.")


def _cart(x: Exporter) -> None:
    z, dt = P.ARM_BASE_Z, 0.012
    parts = [box((0.44, 0.24, 0.5 * dt), (0, 0, z - 0.5 * dt)), box((0.42, 0.22, 0.008), (0, 0, 0.20))]
    parts += [box((0.02, 0.02, 0.5 * (z - dt)), (ix * 0.41, iy * 0.21, 0.5 * (z - dt))) for ix in (-1, 1) for iy in (-1, 1)]
    parts += [box((0.16, 0.18, 0.09), (ix * 0.20, 0, 0.208 + 0.09)) for ix in (-1, 1)]
    cat = trimesh.util.concatenate
    seen = [paint(cat([parts[0], parts[1]]), "cart_plate", "deck and shelf"), paint(cat(parts[2:6]), "cart_frame", "posts"),
            paint(cat(parts[6:]), "controller", "controllers")]
    x.add("cart", "Robot cart (deck, posts, shelf, controllers)", "furniture", union(parts), visual=seen, motion="static",
          material={"name": "steel frame and plate"}, friction=literal(P.MU["default"], "MU = dict(", "estimate", "unlisted pairs"),
          friction_partner="floor", dims=[dim("deck height", z, "z", line_of(P, '_p("ARM_BASE_Z"'))],
          source_ref=where(S.Scene.room), notes="Static collision boxes of the docked cart; origin on the floor at the cart centre.")


def export(x: Exporter) -> None:
    steps = [_bowl5, _bowl2, _saucepan, _crock, _whisk, _spatula, _spoon, _stirrer, _thermometer, _mixer, _hob, _counter, _cart]
    steps += [lambda e, k=k, v=v: _prop(e, k, *v) for k, v in PROPS.items()]
    for step in steps:
        try:
            step(x)
        except Exception as exc:  # noqa: BLE001 - one object must not stop the others
            x.fail(getattr(step, "__name__", "prop"), f"{type(exc).__name__}: {exc}")
