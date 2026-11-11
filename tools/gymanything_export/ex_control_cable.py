# SPDX-License-Identifier: Apache-2.0
"""control_cable: the four harness cables, the prepared ferrule, the 0.4 N m torque driver, the terminal operating
tool and the RJ45 plug with its elastic latch.

Their scene code needs a GPU stack, so geometry is rebuilt here from the same parameters; every cite is a line found
in their source by text search. Cable centre-lines are their planar coil layout (task revision 1, the default).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import trimesh
from _common import Exporter, box, cited, cylinder, dim, line_of, union, vis
from shapely.geometry import Point
from shapely.geometry import box as sbox

from control_cable import render as RN

PKG = Path(importlib.util.find_spec("control_cable").submodule_search_locations[0])


def at(file: str, pattern: str) -> str:
    """``control_cable/<file>:<line>`` of the first line containing ``pattern``."""
    for n, line in enumerate((PKG / file).read_text().splitlines(), 1):
        if pattern in line:
            return f"control_cable/{file}:{n}"
    raise KeyError(f"{pattern!r} not in {file}")


def _srgb(lin) -> list[float]:
    """Their COLORS table holds UsdPreviewSurface diffuseColor values (linear); the part colour is sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]


def _look(geom, material: str, name: str, bound_at: str = "") -> dict:
    """Part with the ``RN.COLORS[material]`` row (colour, roughness, metallic); ``bound_at`` = where their code
    assigns that material to the object."""
    rgb, rough, metal = RN.COLORS[material]
    cite = line_of(RN, f'"{material}":((') + (f"; assigned at {bound_at}" if bound_at else "")
    return vis(geom, _srgb(rgb), cite, rough, metal, name=f"{name} ({material}, linear->sRGB)")


def _tube(points, radius: float) -> trimesh.Trimesh:
    """Open render surface of a cable: one cylinder per centre-line segment."""
    pts = np.asarray(points, float)
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = trimesh.creation.cylinder(radius=radius, segment=np.array([a, b]), sections=8)
        segs.append(seg)
    return trimesh.util.concatenate(segs)


# cable key -> COLORS material their scene assigns, and the line that assigns it
CABLE_MATERIAL = {"supply_brown": ("cable_brown", "for name,y,color in"),
                  "supply_blue": ("cable_blue", "for name,y,color in"),
                  "sensor_cable": ("cable_black", '"cable_black" if name=="sensor"'),
                  "ethernet_cable": ("cable_green", '"cable_black" if name=="sensor" else "cable_green"')}


def coil_points(length: float, r0: float, r1: float, segments: int, rotation: float = 0.0) -> list[list[float]]:
    """Their uniform-arclength planar coil (scene.add_loose_harness, task revision 1) around the origin."""
    turns = length / (math.pi * (r0 + r1))
    theta = np.linspace(0, 2 * math.pi * turns, 2001)
    radii = np.linspace(r0, r1, len(theta))
    pts = np.column_stack((radii * np.cos(theta), radii * np.sin(theta), np.zeros(len(theta))))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    pts *= length / arc[-1]
    arc *= length / arc[-1]
    out = np.column_stack([np.interp(np.linspace(0, length, segments + 1), arc, pts[:, i]) for i in range(3)])
    cs, sn = math.cos(rotation), math.sin(rotation)
    return (out @ np.array([[cs, sn, 0], [-sn, cs, 0], [0, 0, 1]])).tolist()


# key, name, call pattern, length, diameter, line mass, r0, r1, kind, segments, EI, basis of mass, quote
CABLES = [
    ("supply_brown", "Supply lead, brown (H05V-K 1 mm2, ferrule ends)", '"supply_brown"', .6, .0026, .015, .055, .085,
     40, .00030, "datasheet", "LAPP H05V-K 1 mm2: 2.4..2.8 mm OD, 15 kg/km"),
    ("supply_blue", "Supply lead, blue (H05V-K 1 mm2, ferrule ends)", '"supply_blue"', .6, .0026, .015, .055, .085,
     40, .00030, "datasheet", "LAPP H05V-K 1 mm2: 2.4..2.8 mm OD, 15 kg/km"),
    ("sensor_cable", "Sensor cable with M12 plugs (1.5 m)", '"sensor",(.25', 1.5, .0047, .04, .135, .195, 100, .00070,
     "authored", "add_loose_harness line mass 0.04 kg/m, diameter 4.7 mm"),
    ("ethernet_cable", "RJ45 patch cable (1 m)", '"network",(.25', 1.0, .0053, .036, .09, .14, 80, .00070,
     "authored", "add_loose_harness line mass 0.036 kg/m, diameter 5.3 mm"),
]


def _cable(x: Exporter, row: tuple) -> None:
    key, name, pat, length, dia, lm, r0, r1, segs, ei, basis, quote = row
    call = at("scene.py", "self.add_loose_harness(" + pat) if "(.25" in pat else at("scene.py", "for name,y,color in")
    ei_cite = at("scene.py", "stranded-wire estimates")
    mu = cited(.45, "authored", at("scene.py", "cfg=self.shape_config(mu=.45,density=1200)"),
               "rod shape config mu .45 (CableSpec.friction 0.45)")
    pts = coil_points(length, r0, r1, segs)
    material, line_pat = CABLE_MATERIAL[key]
    look = [_look(_tube(pts, dia / 2), material, "insulation", at("scene.py", line_pat))]
    x.add(key, name, "cable", None, kind="cable", motion="dynamic", visual=look,
          cable={"points": pts,
                 "radius": cited(dia / 2, basis, call, quote),
                 "bend_stiffness": cited(ei, "estimate", ei_cite, "EI/GJ are documented stranded-wire estimates, "
                                         "not the stiffness of a solid copper rod")},
          material={"name": "insulated stranded copper conductor"},
          mass=cited(lm * length, basis, call, quote + f"; mass = line mass {lm} kg/m x {length} m"),
          friction=mu, friction_partner="table and connectors", source_ref=at("scene.py", "def add_loose_harness"),
          extra={"length_m": length, "segments": segs, "torsional_rigidity_nm2": .00030 if ei > .0005 else .00015,
                 "axial_rigidity_n": 100000.0, "layout": "planar coil, task revision 1"},
          notes=f"Centre-line is their planar coil of {length} m around the origin (revision 1 layout); the connectors "
                "at the two ends are separate objects.")


def _ferrule(x: Exporter) -> None:
    metal = cylinder(.000725, .006, (0, 0, .006))
    collar = cylinder(.0013, .00225 + .0005 / 2, (0, 0, .01425 - .0005 / 2))
    bound = at("terminal.py", 'material="cable_blue"')
    look = [_look(metal, "copper", "sleeve", at("terminal.py", 'a._cylinder(name+"_metal"')),
            _look(collar, "cable_blue", "plastic collar", bound)]
    x.add("ferrule_wire", "Prepared wire-end ferrule (1 mm2, plastic collar)", "other", union([metal, collar]),
          visual=look,
          material={"name": "tinned copper ferrule with plastic collar"},
          mass=cited(.00024, "authored", at("terminal.py", "a._mass(body,.00024)"), "mass .24 g (Phoenix 1 mm2 ferrule)"),
          friction=cited(.40, "authored", at("terminal.py", "mu=.40,gap=.00002,density=8800"), "ferrule metal mu .40"),
          friction_partner="the terminal clamp", dims=[dim("length", .0165, "z", at("terminal.py", "cable_anchor=[0,0,.0165]"))],
          source_ref=at("terminal.py", "def add_ferrule"),
          notes="1.45 mm outer metal sleeve 12 mm long plus a 2.6 mm collar (their cylinders, collar run 0.5 mm into "
                "the sleeve end to make one solid); the wire it terminates is the supply lead.")


def _terminal_tool(x: Exporter) -> None:
    parts = [cylinder(.007, .0375, (0, 0, .060)), cylinder(.0015, .014, (0, 0, .013)), box((.0012, .0003, .004), (0, 0, -.003))]
    ts = "terminal.py"
    look = [_look(parts[0], "safety_orange", "handle", at(ts, 'name+"_handle",body,.007,.075')),
            _look(union(parts[1:]), "nickel", "shaft and tip", at(ts, 'name+"_shaft",body,.0015'))]
    x.add("terminal_tool", "Terminal operating tool (screwdriver-style)", "tool", union(parts), visual=look,
          material={"name": "plastic handle, steel shaft"},
          mass=cited(.022, "authored", at("terminal.py", "a._mass(body,.022)"), "operating tool mass 22 g"),
          friction=cited(.7, "authored", at("terminal.py", "mu=.7,density=1200"), "handle mu .7"),
          friction_partner="gripper fingers",
          dims=[dim("handle diameter", .014, "x", at("terminal.py", "_handle")), dim("length", .0975, "z",
                                                                                      at("terminal.py", "_handle"))],
          source_ref=at("terminal.py", "def add_operating_tool"),
          notes="Handle cylinder, shaft and blade tip of their tool unioned; the shaft overlaps the handle.")


def _driver_handle_parts() -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    levels = [(0, .009), (.014, .013), (.030, .0175), (.060, .0185), (.090, .0165), (.105, .012)]
    ang = np.arange(12) * 2 * math.pi / 12
    parts = []
    for (z0, r0), (z1, r1) in zip(levels, levels[1:]):
        v = [(r * math.cos(t), r * math.sin(t), z) for z, r in ((z0, r0), (z1, r1)) for t in ang]
        parts.append(trimesh.convex.convex_hull(v))
    bands = [cylinder(.0177, .002, (0, 0, z)) for z in (.023, .081)]
    return union(parts), union(bands)


def _driver_handle() -> trimesh.Trimesh:
    return union(list(_driver_handle_parts()))


def _driver_rotor() -> trimesh.Trimesh:
    radius = .002 / math.cos(math.pi / 6)
    v = [(radius * math.cos(t), radius * math.sin(t), z) for z in (-.100, .004) for t in np.arange(6) * math.pi / 3]
    blade = trimesh.convex.convex_hull(v)
    c = (-.017, 0)
    prof = Point(*c).buffer(.014, quad_segs=32).difference(Point(*c).buffer(.0075, quad_segs=32))
    prof = prof.difference(sbox(-.05, -.0076, -.017, .0076)).union(sbox(-.006, -.0035, .003, .0035))
    jaw = trimesh.creation.extrude_polygon(prof, .006, engine="earcut")
    jaw.apply_translation((0, 0, -.100))
    return union([blade, jaw])


def _driver(x: Exporter) -> None:
    src = "torque_tool.py"
    quote = "Phoenix 1208429 + 1208432 envelope and masses; internal mass split is an estimate"
    grip, bands = _driver_handle_parts()
    look = [_look(grip, "cable_black", "grip sections", at(src, 'material="cable_black"')),
            _look(bands, "paint_blue", "grip bands", at(src, 'material="paint_blue"'))]
    x.add("torque_driver__handle", "Torque driver handle (0.4 N m, friction clutch)", "articulated", _driver_handle(),
          visual=look,
          material={"name": "plastic grip"},
          mass=cited(.0799, "datasheet", at(src, "a._mass(handle,.0799)"), quote),
          friction=cited(.75, "estimate", at(src, "mu=.75,density=1200"), "handle contact mu .75 (numerical prior)"),
          friction_partner="table and gripper fingers",
          dims=[dim("handle length", .105, "z", at(src, "105 mm handle")),
                dim("handle diameter", .037, "x", at(src, "driver_width_m=.037"))],
          source_ref=at(src, "def add_driver"),
          notes="Six convex grip sections and two grip bands unioned; axis along body z, handle from z = 0 to 0.105 m.")
    rotor = _driver_rotor()
    x.add("torque_driver__shaft_and_bit", "Torque driver shaft and D15 side-entry bit", "articulated", rotor,
          visual=[_look(rotor, "nickel", "blade and bit", at(src, 'material="nickel"'))],
          material={"name": "steel"}, mass=cited(.0290, "datasheet", at(src, "a._mass(rotor,.0290)"),
                                                 "11.0 g driver blade + 18.0 g bit; total 108.9 g"),
          friction=cited(.3, "estimate", at(src, "mu=.3,gap=.000025,density=7900"), "steel contact mu .3"),
          friction_partner="the terminal nut", source_ref=at(src, "def add_driver"),
          extra={"joint": {"type": "revolute", "parent": "torque_driver__handle", "axis": [0, 0, 1], "range": None,
                           "cite": at(src, "_passive_torque_clutch"), "friction_torque_nm": .4}},
          notes="Hex blade (4 mm across flats) and the estimated open-crescent D15 jaw (extruded 6 mm); passive "
                "Coulomb friction clutch capped at 0.4 N m, continuous rotation.")


def _rj45_plug(x: Exporter) -> None:
    src = "mechanisms.py"
    body = [box((.00584, .0039, .00925), (0, 0, .00925)), box((.00685, .00655, .0071), (0, 0, .0256))]
    look = [_look(body[0], "polycarbonate", "nose", at(src, 'name+"_nose"')),
            _look(body[1], "cable_black", "boot", at(src, 'material="cable_black"'))]
    x.add("rj45_plug__body", "RJ45 plug (nose and boot)", "articulated", union(body), visual=look,
          material={"name": "polycarbonate nose, soft boot"},
          mass=cited(.0064, "authored", at(src, "self._mass(body,.0064)"), "plug body 6.4 g"),
          friction=cited(.28, "authored", at(src, "ke=150000,kd=.4,mu=.28,gap=.00003,density=1200"), "plug nose mu .28"),
          friction_partner="the RJ45 socket", dims=[dim("length", .0327, "z", at(src, '"rj45":.0327'))],
          source_ref=at(src, "def rj45_plug"),
          notes="Nose and boot boxes only; the eight contact blades (0.12 mm thick) are omitted.")
    E, w, t, ln = 2.35e9, .0057, .0007, .012
    tooth = trimesh.convex.convex_hull(np.array([[-.00285, 0, .008], [.00285, 0, .008], [-.00285, 0, .010],
                                                 [.00285, 0, .010], [-.00285, .0003, .008], [.00285, .0003, .008],
                                                 [-.00285, .0010, .010], [.00285, .0010, .010]]))
    beam = box((w / 2, t / 2, ln / 2), (0, 0, ln / 2))
    latch = union([beam, tooth])
    x.add("rj45_plug__latch", "RJ45 plug elastic latch", "articulated", latch,
          visual=[_look(latch, "polycarbonate", "latch", at(src, 'name+"_latch_beam"'))],
          material={"name": "polycarbonate"},
          mass=cited(.000196, "authored", at(src, "self._mass(latch,.000196)"), "latch 0.196 g"),
          friction=cited(.28, "authored", at(src, "ke=150000,kd=.4,mu=.28,gap=.00003,density=1200"), "latch mu .28"),
          friction_partner="the RJ45 socket", source_ref=at(src, "def rj45_plug"),
          extra={"joint": {"type": "revolute", "parent": "rj45_plug__body", "axis": [1, 0, 0],
                           "range": [-.25, .15], "cite": at(src, "_latch_spring"), "anchor_in_parent": [0, .0035, .002],
                           "rest_angle_rad": -.14, "stiffness_nm_rad": 3 * E * (w * t ** 3 / 12) / ln}},
          notes="Cantilever beam 12 x 5.7 x 0.7 mm with the sloped retaining tooth, as built in their latch; "
                "hinge stiffness 3EI/L with E = 2.35 GPa.")


def export(x: Exporter) -> None:
    for row in CABLES:
        try:
            _cable(x, row)
        except Exception as exc:  # noqa: BLE001
            x.fail(row[0], f"{type(exc).__name__}: {exc}")
    for fn in (_ferrule, _terminal_tool, _driver, _rj45_plug):
        try:
            fn(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
    x.fail("terminal_mechanism", "spring-clamp terminals, M12 plug/nut and sockets are many multi-body parts with "
           "internal estimate-only dimensions; not exported (the RJ45 plug latch is)")
