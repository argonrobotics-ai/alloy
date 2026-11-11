# SPDX-License-Identifier: Apache-2.0
"""titrate: Erlenmeyer flask, burette with stopcock, volumetric pipette with its manual filler, hotplate stirrer, stir bar.

Revolved parts use their own ``geometry.revolve`` on the same meridian polygons; boxes/cylinders follow the numbers in
``laboratory.py`` (whose ``lab.box`` sizes are full lengths). The stopcock plug and socket are cut with boolean
operations from their stated radii (their meshes come from a 0.125 mm marching-cubes grid).
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from _common import Exporter, box, capsule, cited, cylinder, dim, line_of, mesh, transformed, union, vis, where

from titrate.newton15 import apparatus as AP
from titrate.newton15 import geometry as G
from titrate.newton15 import lab_config as LC
from titrate.newton15 import laboratory as LAB
from titrate.newton15 import pipette_filler as PF
from titrate.newton15 import pipette_geometry as PG
from titrate.newton15 import usd_lab as UL

C = LC.LabConfig()
Z_TO_Y = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))     # rotation about x by +90 deg: local z -> -y
Z_TO_X = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))     # rotation about y by +90 deg: local z -> +x
GLASS = "borosilicate glass"


def solid(profile, segments: int = 96) -> trimesh.Trimesh:
    """Their ``geometry.revolve`` of a closed (r, z) polygon as a trimesh."""
    m = G.revolve(profile, segments)
    return mesh(m.vertices, m.triangles)


def lab_box(size, pos, quat=(0, 0, 0, 1)) -> trimesh.Trimesh:
    """Their ``Laboratory.box`` convention: ``size`` is the full edge length."""
    return box(0.5 * np.asarray(size, float), pos, quat)


def lab_cyl(radius: float, height: float, pos, quat=(0, 0, 0, 1)) -> trimesh.Trimesh:
    """Their ``Laboratory.cylinder`` convention: ``height`` is the full length."""
    return cylinder(radius, 0.5 * height, pos, quat)


def _srgb(rgb) -> list[float]:
    """Their material tables hold linear diffuse values (S.color takes linear); parts carry sRGB."""
    c = np.clip(np.asarray(rgb, float), 0.0, 1.0)
    return [float(v) for v in np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)]


def look(geom: trimesh.Trimesh, material: str, name: str = "") -> dict:
    """One render part in their ``usd_lab.PALETTE`` material (diffuse colour, roughness, metallic)."""
    color, rough, metal = UL.PALETTE[material]
    return vis(geom, _srgb(color), line_of(UL, f'"{material}": ('), rough, metal,
               name=f"{name or material} (linear->sRGB)")


def glass_look(geom: trimesh.Trimesh, what: str) -> list[dict]:
    """Their 'glass' material is the OmniGlass tint with frosting roughness: a clear-glass part."""
    return [look(geom, "glass", f"glass ({what})")]


def glass_density() -> dict:
    return cited(2230.0, "estimate", line_of(LAB, "density=2230. if movable"), "glass shape density 2230 kg/m^3 (hollow())")


def mu_cited(value: float, cite: str, quote: str) -> dict:
    return cited(value, "authored", cite, quote)


def volume_mass(parts: list[tuple[trimesh.Trimesh, float]], cite: str, quote: str) -> dict:
    """Mass of a multi-material body: sum of (density x volume) as their sim assigns shape densities."""
    return cited(sum(m.volume * d for m, d in parts), "derived", cite, quote)


# ------------------------------------------------------------------------------------------------ glassware
def _flask(x: Exporter) -> None:
    cite = line_of(G, "Corning 4980-125")
    flask = solid(G.FLASK_PROFILE)
    x.add("erlenmeyer_flask", "Erlenmeyer flask 125 mL (Corning 4980-125)", "container", flask, visual=glass_look(flask, "flask"),
          material={"name": GLASS, "density": glass_density()},
          friction=mu_cited(0.25, line_of(AP, "density=2230., mu=.25"), "flask glass mu 0.25"),
          friction_partner="hotplate ceramic and bench",
          dims=[dim("height", 0.114, "z", cite), dim("maximum outer diameter", 0.067, "diameter_xy", cite)],
          source_ref=where(AP.Apparatus.add_flask),
          notes="Hollow reconstruction (their FLASK_PROFILE: 1.5 mm side wall, 2 mm bottom); the 67 mm OD and 114 mm height are "
                "published, wall/shoulder/rim are their engineering reconstruction. Mass left to the glass density.")


def _stopcock_socket() -> trimesh.Trimesh:
    sleeve = lab_cyl(0.010, 0.020, (0, 0, 0), Z_TO_Y)
    cavity = lab_cyl(C.valve_plug_radius, 0.024, (0, 0, 0), Z_TO_Y)
    port = lab_cyl(C.valve_bore_radius, 0.030, (0, 0, 0))
    return trimesh.boolean.difference([sleeve, cavity, port], engine="manifold")


def _burette(x: Exporter) -> None:
    ri, ro, h = C.burette_inner_radius, C.burette_outer_radius, C.burette_height
    col = solid([(C.valve_bore_radius, .008), (.003, .008), (ro, .038), (ro, h - .055), (ri, h - .055), (ri, .040),
                 (C.valve_bore_radius, .014)])
    tip = solid([(C.tip_inner_radius, -.055), (C.tip_outer_radius, -.055), (.003, -.013), (.003, -.008),
                 (C.valve_bore_radius, -.008), (C.valve_bore_radius, -.013)])
    cite = line_of(LC, "Corning 2103-50")
    socket = _stopcock_socket()
    whole = union([col, tip, socket])
    x.add("burette", "Burette 50 mL (Corning 2103-50) with drawn tip and ported socket", "container",
          whole, visual=glass_look(whole, "column, tip, socket"), motion="static", material={"name": GLASS, "density": glass_density()},
          friction=mu_cited(0.1, line_of(LAB, "ported_glass_socket"), "ported glass socket mu 0.1"), friction_partner="PTFE plug",
          dims=[dim("overall height", C.burette_height, "z", cite)], source_ref=line_of(LAB, "burette_column"),
          notes="Column (inner wall reconstructed), tip and the ported glass sleeve, clamped in their scene (fixed). Frame: the "
                "stopcock centre. Tip and wall radii are labelled estimates in lab_config.",
          extra={"valve_position_in_lab": list(C.valve_position)})
    plug = trimesh.boolean.difference([lab_cyl(C.valve_plug_radius, 2 * C.valve_plug_half_length, (0, 0, 0), Z_TO_Y),
                                       lab_cyl(C.valve_bore_radius, 0.040, (0, 0, 0))], engine="manifold")
    handle = lab_box((.043, .007, .009), (0, -.020, 0))
    pcite = line_of(LAB, "bored_ptfe_plug")
    both = union([plug, handle])
    x.add("burette__stopcock", "Burette PTFE stopcock plug with handle", "articulated", both,
          visual=[look(both, "ptfe", "ptfe plug and lever")],
          material={"name": "PTFE", "density": cited(2200.0, "estimate", pcite, "bored PTFE plug density 2200 kg/m^3")},
          friction=mu_cited(0.10, pcite, "plug mu 0.10"), friction_partner="glass socket",
          dims=[dim("lever length", .043, "x", line_of(LAB, "valve_handle"))],
          source_ref=pcite, notes="Plug with its 2 mm bore across z, plus the lever; the lever box sits 0.5 mm clear of the plug.",
          extra={"joint": {"type": "revolute", "parent": "burette", "axis": [0, 1, 0], "range": [0.0, math.pi / 2],
                           "cite": line_of(LAB, "stopcock_rotation"), "closed_at_rad": math.pi / 2,
                           "friction_Nm": 0.002}})


def _pipette(x: Exporter) -> None:
    cite = line_of(PG, "Corning 7100-10")
    body = solid(PG.profile())
    x.add("pipette", "Volumetric pipette 10 mL (Corning 7100-10)", "container", body, visual=glass_look(body, "pipette"),
          material={"name": GLASS, "density": glass_density()},
          friction=mu_cited(0.25, line_of(LAB, "density=2230. if movable"), "hollow vessel mu 0.25"), friction_partner="rack saddles",
          dims=[dim("overall length", PG.LENGTH_M, "z", line_of(PG, "LENGTH_M ="))], source_ref=where(PG.profile),
          notes="Manufacturer gives capacity, tolerance and length; the bulb radius is solved from the 10 mL mark and the 0.8 mm wall "
                "is a reconstruction. Frame: tip at z = 0, axis +z (horizontal in their scene).")


# ------------------------------------------------------------------------------------------------ pipette filler
def _filler(x: Exporter) -> None:
    chuck = solid([(.006, .419), (.006, .439), (.0028, .439), (.0028, .419)])
    barrel = solid([(.016, .439), (.016, .505), (.010, .505), (.010, .443), (.002, .443), (.002, .439)])
    cite = line_of(PF, "pipette_silicone_chuck")
    x.add("pipette_filler", "Manual pipette filler (Kartell 198 class): barrel and silicone chuck", "tool",
          union([chuck, barrel]), visual=[look(chuck, "rubber", "silicone chuck (rubber)"),
                                          look(barrel, "instrument_blue", "ABS barrel (instrument_blue)")], material={"name": "ABS barrel with silicone chuck"},
          mass=volume_mass([(chuck, 1100.0), (barrel, 1040.0)], cite, "silicone chuck 1100 and ABS barrel 1040 kg/m^3 times volume"),
          friction=mu_cited(0.5, cite, "filler shapes mu 0.5"), friction_partner="gripper pads and rack",
          dims=[dim("barrel and chuck length", 0.505 - 0.419, "z", cite), dim("barrel diameter", 0.032, "diameter_xy", cite)],
          source_ref=where(PF.add_filler),
          notes="Fixed to the pipette body (pipette frame). The 40 mm diameter and 100 mm body are the Kartell article's envelope; "
                "their interior, dead volume and seal friction are disclosed estimates.")
    sh = .519 - (PF.PISTON_FACE_Z_M + .002)
    parts = [(lab_cyl(.00995, .002, (0, 0, 0)), 950.0), (lab_cyl(.003, sh, (0, 0, .001 + sh / 2)), 950.0),
             (lab_cyl(.020, .006, (0, 0, .520 - PF.PISTON_FACE_Z_M - .001)), 1040.0)]
    pc = line_of(PF, "filler_piston")
    x.add("pipette_filler__plunger", "Pipette filler plunger (piston, rod, thumb disc)", "articulated", union([p for p, _ in parts]),
          visual=[look(trimesh.util.concatenate([parts[0][0], parts[1][0]]), "polypropylene", "piston and rod (polypropylene)"),
                  look(parts[2][0], "instrument_blue", "thumb disc (instrument_blue)")],
          material={"name": "polypropylene rod with ABS thumb disc"},
          mass=volume_mass(parts, pc, "polypropylene 950 and instrument-blue 1040 kg/m^3 times volume"),
          dims=[dim("thumb disc diameter", 0.040, "x", line_of(PF, "filler_thumb_disc"))], source_ref=pc,
          notes="Three cylinders on the prismatic slide; frame at the piston face + 1 mm.",
          extra={"joint": {"type": "prismatic", "parent": "pipette_filler", "axis": [0, 0, 1], "range": [0.0, PF.STROKE_M],
                           "cite": line_of(PF, "pipette_plunger_slide"), "anchor_in_parent_m": [0, 0, PF.PISTON_FACE_Z_M + .001]}})
    vc = line_of(PF, "pipette_vent_button")
    x.add("pipette_filler__vent_button", "Pipette filler vent button", "articulated", lab_box((.006, .010, .012), (0, 0, 0)),
          visual=[look(lab_box((.006, .010, .012), (0, 0, 0)), "polypropylene")],
          material={"name": "polypropylene", "density": cited(950.0, "estimate", vc, "button density 950 kg/m^3")},
          dims=[dim("height", .012, "z", vc)], source_ref=vc, notes="Slide button on the barrel.",
          extra={"joint": {"type": "prismatic", "parent": "pipette_filler", "axis": [1, 0, 0], "range": [0.0, PF.VENT_TRAVEL_M],
                           "cite": line_of(PF, "pipette_vent_travel"), "anchor_in_parent_m": [-.019, 0, .478]}})


# ------------------------------------------------------------------------------------------------ stirrer, bar, bench
def _stirrer(x: Exporter) -> None:
    parts = [lab_box((.150, .260, .091), (0, -.055, .0525)), lab_box((.100, .100, .007), (0, 0, .1015)),
             lab_box((.137, .002, .072), (0, -.1855, .046))]
    parts += [lab_cyl(.008, .008, (dx, dy, .004)) for dx in (-.060, .060) for dy in (-.15, .04)]
    cite = line_of(LAB, "mass, dimensions = 2.716")
    cat = trimesh.util.concatenate
    seen = [look(parts[0], "instrument_blue", "housing (instrument_blue)"), look(parts[1], "ceramic", "plate (ceramic)"),
            look(parts[2], "cabinet", "control panel (cabinet)"), look(cat(parts[3:]), "rubber", "feet (rubber)")]
    x.add("stirrer", "Magnetic stirrer IKA C-MAG MS 4", "fixture", union(parts), visual=seen, material={"name": "painted metal housing, ceramic plate"},
          mass=cited(2.716, "datasheet", cite, "Manufacturer mass and outer envelope"),
          dims=[dim("width", .150, "x", line_of(LC, "IKA C-MAG MS 4")), dim("depth", .260, "y", line_of(LC, "IKA C-MAG MS 4")),
                dim("height", .105, "z", line_of(LC, "IKA C-MAG MS 4"))], source_ref=cite,
          notes="Housing, 100 mm ceramic plate, control panel and four feet as one solid; origin on the bench under the plate "
                "(plate centre). The speed knob is a separate part.")
    dial, mark = lab_cyl(.013, .015, (0, 0, 0), Z_TO_Y), lab_box((.001, .0004, .006), (0, -.0076, .007))
    knob = union([dial, mark])
    kc = line_of(LAB, "stirrer_speed_knob")
    x.add("stirrer__knob", "Stirrer speed knob", "articulated", knob,
          visual=[look(dial, "robot_dark", "knob (robot_dark)"), look(mark, "ptfe", "index mark (ptfe)")],
          material={"name": "dark plastic", "density": cited(1100.0, "estimate", kc, "knob density 1100 kg/m^3")},
          dims=[dim("diameter", .026, "x", kc)], source_ref=kc, notes="Cylinder with a PTFE index mark (zero-density in their model).",
          extra={"joint": {"type": "revolute", "parent": "stirrer", "axis": [0, 1, 0], "range": [0.0, 5 * math.pi / 3],
                           "cite": line_of(LAB, "stirrer_speed_dial"), "anchor_in_parent_m": [0, -.195, .038], "friction_Nm": .001}})


def _stir_bar(x: Exporter) -> None:
    cite = line_of(AP, "add_shape_capsule(body, radius=.003")
    bar = capsule(.003, .007, (0, 0, 0), Z_TO_X)
    x.add("stir_bar", "PTFE stir bar 20 x 6 mm", "tool", bar, visual=[look(bar, "ptfe", "ptfe")],
          material={"name": "PTFE-sheathed magnetic bar",
                    "density": cited(3200.0, "estimate", line_of(AP, "Density 3200"), "estimates the mixed PTFE + magnetic core bulk density")},
          friction=mu_cited(0.15, cite, "stir bar mu 0.15"), friction_partner="flask glass",
          dims=[dim("length", .020, "x", line_of(AP, "PTFE-sheathed 20 x 6 mm bar")), dim("diameter", .006, "z", cite)],
          source_ref=where(AP.Apparatus.add_stir_bar), notes="Capsule along x; mass is the 3200 kg/m^3 times the capsule volume.")


def _bench(x: Exporter) -> None:
    cite = line_of(LAB, "bench_epoxy")
    top = lab_box((C.bench_width, C.bench_depth, .03), (0, 0, .015))
    x.add("bench", "Epoxy-resin laboratory bench top", "furniture", top, visual=[look(top, "epoxy")], motion="static", material={"name": "epoxy resin"},
          friction=mu_cited(0.4, line_of(LAB, "def box"), "Laboratory.box default mu 0.4"), friction_partner="glassware",
          dims=[dim("length", C.bench_width, "x", line_of(LC, "bench_width")), dim("depth", C.bench_depth, "y", line_of(LC, "bench_depth"))],
          source_ref=cite, notes="Top slab only (origin on its underside centre); height above floor is bench_height 0.9 m. Cabinets, "
          "room, stand and clamps are not exported.")


def export(x: Exporter) -> None:
    for step in (_flask, _burette, _pipette, _filler, _stirrer, _stir_bar, _bench):
        try:
            step(x)
        except Exception as exc:  # noqa: BLE001 - one object must not stop the others
            x.fail(step.__name__.strip("_"), f"{type(exc).__name__}: {exc}")
