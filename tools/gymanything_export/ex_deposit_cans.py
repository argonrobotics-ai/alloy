# SPDX-License-Identifier: Apache-2.0
"""deposit_cans: recycling station, film bag, bussing tub, three aluminium can types, tumbler, paper cup, bins.

Meshes come from their builders (``tub_geometry.geometry``, ``props.cup_geometry`` / ``rounded_shell``,
``station.lid_mesh``). Their cans and film bag are open triangulated membranes (not solids), so those two are
rebuilt here as thin watertight walls of their own thickness around the same surface/profile.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import LineString, box as sbox

from _common import Exporter, _rel, box, cited, dim, line_of, mesh, revolve, union, vis, where

from deposit_cans import assembly as A
from deposit_cans import config as CFG
from deposit_cans import geometry as G
from deposit_cans import graphics as GR
from deposit_cans import props as PR
from deposit_cans import render as RN
from deposit_cans import station as ST
from deposit_cans import tub_geometry as TG

log = logging.getLogger("gar.export.deposit_cans")
CONFIG_MU = "friction: float"
KINDS = ("standard", "slim", "tall")
PC_MDL = Path(RN.__file__).parents[1] / "wiping" / "polycarbonate.mdl"


def _mu(x_cite_mod=CFG) -> dict:
    return cited(CFG.Config().friction, "estimate", line_of(CFG, CONFIG_MU), "default contact friction of every shape")


# ------------------------------------------------------------------------------------------------ render parts
def _srgb(rgb) -> list[float]:
    """Their render COLORS are written as USD diffuseColor values (linear); parts are sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in rgb]


def _paint(geom: trimesh.Trimesh, name: str, label: str = "") -> dict:
    """Part painted with their ``render.COLORS[name]`` = (linear colour, roughness, metallic)."""
    rgb, rough, metal = RN.COLORS[name]
    return vis(geom, _srgb(rgb), line_of(RN, f"{name}=(("), rough, metal, name=f"{label or name} (linear->sRGB)")


def _mdl_line(path: Path, pattern: str) -> str:
    """``<sim>/<file>:<line>`` of the first line of a shader file containing ``pattern``."""
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if pattern in line:
            return f"{_rel(path)}:{n}"
    raise KeyError(f"{pattern!r} not in {path}")


def _clear(geom: trimesh.Trimesh, mdl: Path, tint_pat: str, rough_pat: str) -> dict:
    """Clear thin-wall plastic from its MDL shader: transmission colour as tint, specular roughness, no metal."""
    text = mdl.read_text()
    tint = [float(v) for v in re.search(tint_pat, text).group(1).split(",")]
    rough = float(re.search(rough_pat, text).group(1))
    return vis(geom, tint * 3 if len(tint) == 1 else tint, _mdl_line(mdl, "specular_transmission_color"), rough, 0.0,
               name="glass")


# ------------------------------------------------------------------------------------------------ station
def _station_parts() -> dict[str, list[trimesh.Trimesh]]:
    """Crossed-U frame tubes, pivots, feet and bag-lock loops (their ``station.build``) plus the molded lid,
    grouped by their render material: steel frame, rubber feet, blue_plastic lid with clips and ribs."""
    lid, feet, parts = [mesh(*ST.lid_mesh(7)[:2])], [], []
    tube = lambda p, q, r: trimesh.creation.cylinder(radius=r, segment=[p, q], sections=16)
    r = 0.0047625
    for frame in (0, 1):
        hx, top_y = 0.2965 + frame * 0.0098, (-1 if frame else 1) * 0.2126
        parts.append(tube((-hx, top_y, .989), (hx, top_y, .989), r))
        for s in (-1, 1):
            x = s * hx
            parts.append(tube((x, -top_y, .011), (x, top_y, .989), r))
            feet.append(box((.007, .011, .006), (x, -top_y, .006)))
            lx = s * (hx - .019)
            pts = [(lx, top_y, .989), (lx, top_y * .91, 1.002), (lx - s * .075, top_y * .91, 1.002),
                   (lx - s * .075, top_y, .989)]
            parts += [tube(p, q, .0032) for p, q in zip(pts, pts[1:])]
    parts += [tube((s * .288, 0, .5), (s * .314, 0, .5), .0075) for s in (-1, 1)]
    for y in (-.211, .211):
        for xx in (-.19, .19):
            lid += [box((.009, .002, .011), (xx, y + o, .989)) for o in (-.0068, .0068)]
    lid += [box((.252, .0015, .006), (0, y, 1.005)) for y in (-.177, .177)]
    return {"blue_plastic": lid, "rubber": feet, "steel": parts}


def _station_mesh() -> trimesh.Trimesh:
    return union([m for ms in _station_parts().values() for m in ms])


def _station(x: Exporter) -> None:
    cite = line_of(ST, "Published:")
    look = [_paint(union(ms), name) for name, ms in _station_parts().items()]
    x.add("station", "Recycling station, folding frame and slotted lid (ClearStream SingleStream Blue class)", "fixture",
          _station_mesh(), motion="static", visual=look, material={"name": "steel rod frame with molded blue plastic lid"},
          mass=cited(3.62874, "datasheet", line_of(A, "mass_kg=3.62874"),
                     "Published: overall 40x25x18 in, 3/8 in rod, 8 lb total, 1.75 lb lid, 16x4.5 in slot"),
          friction=cited(.65, "estimate", where(ST.build), "rubber feet mu=.65"), friction_partner="the floor",
          dims=[dim("height", 1.016, "z", cite), dim("lid length", .635, "x", cite), dim("lid width", .4572, "y", cite)],
          source_ref=where(ST.build),
          notes="Frame tubes, feet, bag-lock loops, clips, ribs and the slotted lid rebuilt from their station.build "
                "arithmetic (scene revision 7 lid); frame and lid 3.62874 kg together.")


# ------------------------------------------------------------------------------------------------ film bag
def _bag_shell() -> trimesh.Trimesh:
    """Open-mouth bag as a thin watertight wall: their bag surface (revision-1 shape) +- the film thickness."""
    v, _, _, _ = G.bag_mesh(64, 36)[:4]
    t = CFG.Config().film_thickness
    outer = trimesh.convex.convex_hull(v)
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    hx, hy = np.abs(v[:, 0]).max(), np.abs(v[:, 1]).max()
    inner_pts = v * np.array([1 - t / hx, 1 - t / hy, 1.0])
    inner_pts[:, 2] = np.clip(inner_pts[:, 2], z0 + t, z1)
    top = inner_pts[inner_pts[:, 2] >= z1 - 1e-9].copy()
    top[:, 2] = z1 + 0.02
    inner = trimesh.convex.convex_hull(np.vstack([inner_pts, top]))
    return union([outer]).difference(inner, engine="manifold")


def _bag(x: Exporter) -> None:
    c = CFG.Config()
    t, e, nu = c.film_thickness, c.film_young, c.film_poisson
    q = "estimated LLDPE film (38.1 um)"
    cf = lambda v, name: cited(v, "estimate", line_of(CFG, name), q)
    shell = _bag_shell()
    film = _clear(shell, Path(RN.__file__).with_name("film.mdl"), r"specular_transmission_color: color\(([^)]*)\)",
                  r"specular_reflection_roughness: ([0-9.]+)")
    x.add("film_bag", "Film recycling bag (40 x 45 in lay-flat, 38.1 um)", "cloth", shell, kind="cloth", visual=[film],
          material={"name": "LLDPE film", "density": cf(c.film_density, "film_density: float"),
                    "youngs": cf(e, "film_young: float"), "poisson": cf(nu, "film_poisson: float")},
          friction=_mu(), friction_partner="cans and the stand",
          thickness=cf(t, "film_thickness: float"),
          cloth={"thickness": cf(t, "film_thickness: float"),
                 "stretch_warp": cited(e * t, "derived", line_of(A, "tri_ke=mu*"), "E*t, isotropic film (their E, t)"),
                 "stretch_weft": cited(e * t, "derived", line_of(A, "tri_ke=mu*"), "E*t, isotropic film (their E, t)"),
                 "shear": cited(e * t / (2 * (1 + nu)), "derived", line_of(A, "mu=E/(2*(1+nu))"), "mu*t, their surface law"),
                 "bend_warp": cited(e * t ** 3 / (12 * (1 - nu * nu)), "derived", line_of(A, "rigidity=E*thick"),
                                    "plate rigidity E t^3 / 12(1-nu^2), their discrete-shell bending"),
                 "bend_weft": cited(e * t ** 3 / (12 * (1 - nu * nu)), "derived", line_of(A, "rigidity=E*thick"),
                                    "plate rigidity E t^3 / 12(1-nu^2), their discrete-shell bending")},
          dims=[dim("height", .9985 - .022, "z", where(G.bag_mesh))], source_ref=where(G.bag_mesh),
          notes="Their membrane is an open triangulated sheet; here the revision-1 bag outline (hull of their rows) is "
                "given its film thickness as a closed thin shell, mouth open. Isotropic film: warp = weft.")


# ------------------------------------------------------------------------------------------------ cans
def _can_profile(kind: int) -> np.ndarray:
    """(r, z) outline of ``can_mesh``'s body rows: bottom centre, bead, wall, neck, to the inner top rim."""
    h = G.CAN_SPECS[kind][1]
    n = 24
    v, _, _ = G.can_mesh(kind, n)
    rows = 3 + max(3, round(h / .017)) + 3
    rings = v[: rows * n].reshape(rows, n, 3)
    return np.array([(0.0, -h / 2 + .008)] + [(float(np.hypot(*r[0, :2])), float(r[0, 2])) for r in rings])


def _can_shell(kind: int) -> trimesh.Trimesh:
    d, h, t = G.CAN_SPECS[kind]
    prof = _can_profile(kind)
    # run the line through the axis and one point past it, so clipping at r = 0 leaves a straight cut
    prof = np.vstack([[-prof[1, 0], prof[1, 1]], prof, [0.0, prof[-1, 1]], [-prof[-1, 0], prof[-1, 1]]])
    band = LineString(prof).buffer(t / 2, cap_style=2, join_style=2).intersection(sbox(0, -1, 1, 1))
    poly = np.asarray(band.geoms[0].exterior.coords if hasattr(band, "geoms") else band.exterior.coords)
    return revolve(poly, 48)


def _can_visual(kind: int, shell: trimesh.Trimesh) -> list[dict]:
    """Aluminium can with the printed label band as their render: uv u = azimuth/(2 pi) + .5, v = height fraction,
    on the outer wall between 10 mm above the base and 8 mm below the rim (``assembly`` label_faces)."""
    from PIL import Image  # noqa: PLC0415
    _, h, _ = G.CAN_SPECS[kind]
    tri = shell.vertices[shell.faces]                       # (F, 3, 3), unmerged
    ctr = tri.mean(axis=1)
    rc = np.linalg.norm(ctr[:, :2], axis=1)
    wall = (np.abs(shell.face_normals[:, 2]) < .5) & (tri[:, :, 2].min(1) >= -h / 2 + .010 - 1e-9) \
        & (tri[:, :, 2].max(1) <= h / 2 - .008 + 1e-9)
    band = wall & (rc > 0.5 * (rc[wall].min() + rc[wall].max()))      # the outer of the two walls (the shell is thin)
    uv = np.stack((np.arctan2(tri[..., 1], tri[..., 0]) / (2 * np.pi) + .5, (tri[..., 2] + h / 2) / h), axis=2)
    seam = np.ptp(uv[..., 0], axis=1) > .5
    uv[seam, :, 0] += uv[seam, :, 0] < .5
    printed = tri[band].reshape(-1, 3)
    pm = trimesh.Trimesh(printed, np.arange(len(printed)).reshape(-1, 3), process=False)
    rest = trimesh.Trimesh(tri[~band].reshape(-1, 3), np.arange(3 * int((~band).sum())).reshape(-1, 3), process=False)
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.open(GR.artwork(tmp)[f"can_{kind}"]).convert("RGB").copy()
    cite = line_of(GR, "metallic_constant")
    look = vis(pm, [1.0, 1.0, 1.0], cite, .3, .15, image=img, uv=uv[band].reshape(-1, 2), name="printed label")
    return [_paint(rest, "aluminum", "bare aluminium"), look]


def _can(x: Exporter, kind: int) -> None:
    d, h, t = G.CAN_SPECS[kind]
    cite = line_of(G, "CAN_SPECS=")
    q = "Standard 66x115 mm; slim 57x146 mm; tall 66x167 mm. Open ends, thin walls (report p.3); " \
        "mass lumped from triangle area x wall x 2700 kg/m3"
    c = CFG.Config()
    shell = _can_shell(kind)
    x.add(f"can_{KINDS[kind]}", f"Aluminium beverage can, {KINDS[kind]} ({d * 1000:.0f} x {h * 1000:.0f} mm)", "container",
          shell, visual=_can_visual(kind, shell),
          material={"name": "aluminium alloy", "density": cited(2700., "literature", line_of(A, "2700.,plastic"), q),
                    "youngs": cited(c.aluminum_young, "estimate", line_of(CFG, "aluminum_young"), "E = 69 GPa (researched alloy estimate)"),
                    "poisson": cited(c.aluminum_poisson, "estimate", line_of(CFG, "aluminum_poisson"), "nu = 0.33")},
          friction=_mu(), friction_partner="any surface (their default contact)",
          dims=[dim("diameter", d, "diameter_xy", cite), dim("height", h, "z", cite)], source_ref=where(G.can_mesh),
          notes=f"Thin watertight wall ({t * 1e6:.0f} um) revolved from their can_mesh body profile; the opened-tab "
                "hole is omitted (lid closed) and the 220 um end thickening is not modelled. Mass = density x wall volume.")


# ------------------------------------------------------------------------------------------------ props
def _shell_obj(x: Exporter, key: str, name: str, geom: trimesh.Trimesh, mass: dict, dims: list[dict], mat: str,
               cite: str, notes: str, wall: dict | None = None, look: list[dict] | None = None) -> None:
    x.add(key, name, "container", geom, material={"name": mat}, mass=mass, friction=_mu(), visual=look,
          friction_partner="any surface (their default contact)", dims=dims, source_ref=cite, notes=notes,
          thickness=wall)


def _tub(x: Exporter) -> None:
    v, f = TG.geometry()
    cite = where(TG.geometry)
    _shell_obj(x, "bussing_tub", "Bussing tub (Cambro Poly Cambox class)", mesh(v, f),
               cited(.70, "estimate", line_of(A, "set_mass(body,.70)"),
                     "0.70 kg is a case-mass-derived estimate, not measured net mass (report p.3)"),
               [dim("length", .506, "y", cite), dim("width", .389, "x", cite), dim("height", .125, "z", cite)],
               "molded polypropylene tub, 1.7 mm skin", cite,
               "Their watertight molded surface: rim return, grip recesses and bottom ribs are estimates.",
               wall=cited(.0017, "estimate", line_of(TG, "thickness=.0017"), "1.7mm skin (estimated features)"),
               look=[_paint(mesh(v, f), "gray_plastic")])


def _cups(x: Exporter) -> None:
    cite = where(PR.cup_geometry)
    v, f = PR.cup_geometry()
    _shell_obj(x, "tumbler_cup", "Reusable tumbler cup (Cambro Camwear 1200CW)", mesh(v, f),
               cited(.077, "estimate", line_of(PR, "polycarbonate',.077"), "77 g and wall/taper reconstruction are estimates"),
               [dim("height", .132, "z", cite), dim("rim diameter", .075, "diameter_xy", cite)],
               "polycarbonate", cite, "Open hollow tumbler: outer 75 x 132 mm, tapered walls (their cup_geometry).",
               look=[_clear(mesh(v, f), PC_MDL, r"specular_transmission_color: color\(([^)]*)\)",
                            r"specular_reflection_roughness: ([0-9.]+)")])
    v, f = PR.cup_geometry(rings_rz=PR.PAPER_CUP_RINGS)
    _shell_obj(x, "paper_cup", "Single-use paper hot cup, 12 oz", mesh(v, f),
               cited(PR.PAPER_CUP_MASS, "authored", line_of(PR, "PAPER_CUP_MASS="),
                     "12 oz class: rim 90 mm, base 60 mm, height 110 mm outer; 1.5 mm wall and 12 g mass are authored estimates"),
               [dim("height", .110, "z", cite), dim("rim diameter", .090, "diameter_xy", cite)],
               "paper cup, 1.5 mm wall", cite, "Open hollow frustum (their PAPER_CUP_RINGS).",
               wall=cited(.0015, "authored", line_of(PR, "PAPER_CUP_MASS="), "1.5 mm wall and floor authored estimates"),
               look=[_paint(mesh(v, f), "paper_cup")])


def _bins(x: Exporter) -> None:
    v, f = PR.rounded_shell((.508, .2794, .762), wall=.004, inset=.030, radius=.045)
    shell = mesh(v, f)
    handles = [box((.0127, .075, .012), (s * .2667, 0, .714)) for s in (-1, 1)]
    cite = where(PR.build)
    _shell_obj(x, "landfill_container", "Waste container (Rubbermaid FG354060 class)", union([shell] + handles),
               cited(3.54, "datasheet", line_of(PR, "'charcoal',3.54"), "Rubbermaid FG354060: body+end handles reach 558.8x279.4x762mm"),
               [dim("length with handles", .5588, "x", cite), dim("height", .762, "z", cite)],
               "molded polyethylene", cite, "Open rounded shell with the two end handles unioned.",
               look=[_paint(union([shell] + handles), "charcoal")])
    v, f = PR.rounded_shell((.40, .18, .020), wall=.002, inset=.004, radius=.015)
    _shell_obj(x, "return_tray", "Reusable-item return tray", mesh(v, f),
               cited(.23, "authored", line_of(PR, "'gray_plastic',.23"), "authored prop: wall profile and mass are estimates"),
               [dim("length", .40, "x", cite), dim("height", .020, "z", cite)], "gray plastic", cite,
               "Open shallow tray, 2 mm walls (their rounded_shell).", look=[_paint(mesh(v, f), "gray_plastic")])


def export(x: Exporter) -> None:
    steps = [("station", lambda: _station(x)), ("film_bag", lambda: _bag(x)), ("bussing_tub", lambda: _tub(x)),
             ("cans", lambda: [_can(x, k) for k in range(3)]), ("tumbler_cup", lambda: _cups(x)),
             ("landfill_container", lambda: _bins(x))]
    for key, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            x.fail(key, f"{type(exc).__name__}: {exc}")
            log.exception("%s failed", key)
