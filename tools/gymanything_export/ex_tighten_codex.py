# SPDX-License-Identifier: Apache-2.0
"""tighten_codex: the W4F sensor, its BEF-W4-A bracket, the 280 mm HFS5-2020 rail, the 15 g corner bracket, M3/M4
socket-head screws, the two torque drivers (handle + clutch rotor), the bench, the reference flag, the driver
cradles and the camera stands.

Their ``scene.build`` is one long function that needs Newton, so the shapes are rebuilt here from the same numbers
(each cited by line). Where parts move relative to each other (driver handle and bit rotor) they are separate specs
with ``extra.joint``.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from _common import Exporter, box, cited, cylinder, dim, line_of, mesh, transformed, union, vis

from tighten_codex import config as CF
from tighten_codex import render as RN
from tighten_codex import scene as SC

QY = lambda a: (0, math.sin(a / 2), 0, math.cos(a / 2))  # noqa: E731 - xyzw about y
QX = lambda a: (math.sin(a / 2), 0, 0, math.cos(a / 2))  # noqa: E731 - xyzw about x
BORE_QUOTE = "reference values, none measured on these parts (their config.py)"


def _mf():
    import manifold3d as mf  # noqa: PLC0415
    return mf


def _from_manifold(m) -> trimesh.Trimesh:
    out = m.to_mesh()
    return mesh(out.vert_properties[:, :3], out.tri_verts)


def _boxes(items, origin=(0., 0., 0.)) -> trimesh.Trimesh:
    """Union of ``(centre, half[, quat_xyzw])`` boxes given in world coordinates, moved by -origin."""
    o = np.asarray(origin, float)
    return union([box(h, np.asarray(c, float) - o, *(q or [])) for c, h, *q in items])


def _plate_xz(cs, thickness: float, y0: float) -> trimesh.Trimesh:
    """A 2-D (x, z) cross-section extruded ``thickness`` along +y from y0."""
    m = _from_manifold(_mf().Manifold.extrude(cs, thickness))
    return m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, y0 + thickness], [0, 1, 0, 0], [0, 0, 0, 1.]]))


def _look(geom: trimesh.Trimesh, mat: str, label: str = "") -> dict:
    """Render part from their ``PALETTE`` (UsdPreviewSurface diffuse, roughness, metallic). The diffuse is linear (the
    same file raises robot colours with ``**2.2``), so it is converted to sRGB here."""
    rgb, rough, metal = RN.PALETTE[mat]
    lin = np.clip(np.asarray(rgb, float), 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return vis(geom, srgb.tolist(), line_of(RN, f'"{mat}":(('), rough, metal,
               name=f"{label or mat} (linear->sRGB)")


def _box_looks(items, mats, origin=(0., 0., 0.)) -> list[dict]:
    """One part per distinct material: the boxes of ``items`` (as ``_boxes``) grouped by ``mats[i]``."""
    out = []
    for mat in dict.fromkeys(mats):
        out.append(_look(_boxes([it for it, m in zip(items, mats) if m == mat], origin), mat))
    return out


def _mu(name: str, quote: str = BORE_QUOTE, basis: str = "literature") -> dict:
    return cited(CF.Config.__dataclass_fields__[name].default, basis, line_of(CF, f"{name}:"), quote)


# ---------------------------------------------------------------------------------------------- sensor, bracket
def _sensor(x: Exporter) -> None:
    ref = line_of(SC, "sensor_box=np.array")
    parts = [box((.008, .00605, .014), (0, 0, .0046))]
    mf = _mf()
    cable = []
    for dx in (-.0045, .0045):
        ring = mf.CrossSection.circle(.0035, 32) - mf.CrossSection.circle(.0017, 32)
        lobe = _from_manifold(mf.Manifold.extrude(ring, .0121)).apply_transform(
            np.array([[1, 0, 0, dx], [0, 0, -1, .00605], [0, 1, 0, -.014], [0, 0, 0, 1.]]))
        parts.append(lobe)
    strip = box((.008, .00605, .001), (0, 0, -.010))
    parts.append(strip)
    for a, c in zip(((0., .001, -.0186), (0., .003, -.030), (0., .003, -.038)),
                    ((0., .003, -.030), (0., .003, -.038), (0., .003, -.044))):
        d = np.subtract(c, a)
        pig = trimesh.creation.cylinder(.0017, float(np.linalg.norm(d)), sections=16)
        pig.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], d / np.linalg.norm(d)))
        cable.append(pig.apply_translation((np.add(a, c) / 2)))
    cable.append(cylinder(.004, .006, (0., .003, -.050)))
    looks = [_look(union([parts[0], parts[1], parts[2]]), "sensor_blue", "housing and M3 lobes"),
             _look(strip, "steel", "M3 clamping strip"), _look(union(cable), "cable_black", "pigtail and M8 plug")]
    parts += cable
    x.add("sick_w4f_sensor", "SICK W4F photoelectric sensor (WTB4FP)", "other", union(parts), visual=looks,
          material={"name": "glass-fibre plastic housing (VISTAL) with M8 pigtail"},
          mass=cited(.030, "datasheet", line_of(SC, "housing_m,cable_m"), "SICK weight 'approx. 30 g' includes the "
                     "77 mm pigtail and M8 plug (24 g housing + 6 g pigtail)"),
          friction=_mu("sensor_face_mu", "W4F plastic face on the stainless BEF-W4-A; " + BORE_QUOTE),
          friction_partner="stainless bracket plate",
          dims=[dim("housing width", .016, "x", ref), dim("housing depth", .0121, "y", ref)],
          source_ref=ref,
          notes="Housing box, two bored M3 lobes (a 1 mm bridge joins them to the housing), pigtail and M8 plug as "
                "in their scene; the 28 mm housing collider is shorter than the 37.2 mm drawing envelope.")


def _bef_front() -> trimesh.Trimesh:
    mf = _mf()
    left, right, bottom, top = .248, .282, 1.067, 1.1064
    pivot, slot_r = np.array((.261, 1.074)), .009
    plate = mf.CrossSection.square((right - left, top - bottom)).translate((left, bottom))
    hole = mf.CrossSection.circle(.0017, 48).translate(tuple(pivot))
    ang = np.radians(np.linspace(-15, 15, 25))
    outer = [(pivot[0] + (slot_r + .0017) * math.cos(t), pivot[1] + (slot_r + .0017) * math.sin(t)) for t in ang]
    inner = [(pivot[0] + (slot_r - .0017) * math.cos(t), pivot[1] + (slot_r - .0017) * math.sin(t)) for t in ang[::-1]]
    slot = mf.CrossSection([np.array(outer + inner)])
    for t in (ang[0], ang[-1]):
        slot = slot + mf.CrossSection.circle(.0017, 24).translate((pivot[0] + slot_r * math.cos(t),
                                                                   pivot[1] + slot_r * math.sin(t)))
    return _plate_xz(plate - hole - slot, .002, .086)


def _bef_bracket(x: Exporter) -> None:
    ref = line_of(SC, "bef_left,bef_right")
    end_leg = box((.001, .007, (1.1064 - 1.067) / 2), (.281, .095, (1.067 + 1.1064) / 2))
    m = union([_bef_front(), end_leg])
    m.apply_translation(-m.bounds.mean(axis=0))
    x.add("bef_w4_a_bracket", "SICK BEF-W4-A mounting bracket (stainless 1.4571, 2 mm)", "fixture", m,
          visual=[_look(m, "steel", "stainless plate")],
          material={"name": "stainless steel 1.4571", "density": cited(7950., "literature", line_of(SC, "rail_mass,bef_mass"),
                                                                     "~1890 mm2 x 2 mm x 7.95 g/cm3 = 0.030 kg")},
          mass=cited(.030, "derived", line_of(SC, "rail_mass,bef_mass"), "2 mm stainless 1.4571: 14 mm leg plus "
                     "34 x 39.4 mm front leg, ~1890 mm2 x 2 mm x 7.95 g/cm3 = 0.030 kg"),
          friction=_mu("sensor_face_mu", "W4F plastic face on the stainless BEF-W4-A; " + BORE_QUOTE),
          friction_partner="W4F plastic housing",
          dims=[dim("front leg width", .034, "x", ref), dim("leg height", 1.1064 - 1.067, "z", ref),
                dim("overall depth (2 mm plate + 14 mm leg)", .016, "y", ref)],
          source_ref=line_of(SC, "def bef_front_mesh"),
          notes="Front leg with the 3.4 mm pivot hole and 30 deg arc slot plus the 14 mm rail-end leg; in their sim "
                "it is one body with the rail. Frame origin at the bracket's bounding-box centre.")


# ---------------------------------------------------------------------------------------------- frame parts
def _rail(x: Exporter) -> None:
    items = [((0, 0, .0085), (.14, .010, .0015)), ((0, 0, 0), (.14, .004, .004))]
    for s in (-1, 1):
        items += [((0, s * .0085, 0), (.14, .0015, .007)), ((0, s * .007, -.0085), (.14, .003, .0015))]
    ref = line_of(SC, 'box("horizontal_profile_top"')
    x.add("hfs5_2020_rail_280mm", "T-slot profile HFS5-2020, 280 mm", "fixture", _boxes(items),
          visual=[_look(_boxes(items), "aluminum")], material={"name": "anodised aluminium"},
          mass=cited(.14, "datasheet", line_of(SC, "rail_mass,bef_mass"), "The rail body carries the 280 mm HFS5-2020 "
                     "(0.14 kg); report: 'The 280 mm HFS5-2020 rail (0.14 kg)'"),
          friction=_mu("clamp_mu", "anodised aluminium on anodised aluminium (rail/bracket/post faces); "
                       + BORE_QUOTE),
          friction_partner="anodised aluminium bracket leg",
          dims=[dim("length", .28, "x", ref), dim("width", .02, "y", ref), dim("height", .02, "z", ref)],
          source_ref=ref, notes="Top plate, web, two side walls and two slot lips as boxes (their collision "
                                "profile of the 20 x 20 extrusion); frame origin at the profile centre.")


def _corner_bracket(x: Exporter) -> None:
    o = (.013, .095, 1.075)
    items = [((-.005625, .0875, 1.065), (.003375, .0025, .0155)), ((.018875, .0875, 1.065), (.016625, .0025, .0155)),
             ((0., .0875, 1.057125), (.00225, .0025, .007625)), ((0., .0875, 1.074875), (.00225, .0025, .005625)),
             ((.004, .10125, 1.083), (.013, .01625, .0025)), ((.03125, .10125, 1.083), (.00425, .01625, .0025)),
             ((.022, .091375, 1.083), (.005, .006375, .0025)), ((.022, .109875, 1.083), (.005, .007625, .0025)),
             ((.008, .100, 1.0855 + .00075), (.005, .0038, .00075))]
    ref = line_of(SC, "bracket = b.add_link")
    x.add("corner_bracket_hblts5", "Corner angle bracket (MISUMI HBLTS5-4 class, aluminium)", "fixture",
          _boxes(items, o), visual=[_look(_boxes(items, o), "aluminum")], material={"name": "anodised aluminium"},
          mass=cited(.015, "datasheet", ref, "Corner angle bracket (MISUMI HBLTS5-4 class, aluminium, 15 g)"),
          friction=_mu("clamp_mu", "anodised aluminium on anodised aluminium; " + BORE_QUOTE),
          friction_partner="aluminium post and rail",
          dims=[dim("horizontal leg length", .0445, "x", ref)], source_ref=ref,
          notes="Vertical leg strips, horizontal leg boxes with the 4.5 mm hole and 10 mm slot left open, and the "
                "key tab; boxes as in their scene. Frame origin at their bracket body origin.")


# ---------------------------------------------------------------------------------------------- screws, drivers
def _screw_mesh(d: float, head_d: float, head_h: float, af: float, shank: float) -> trimesh.Trimesh:
    mf = _mf()
    ri = af / 2 / math.cos(math.pi / 6)
    hexagon = mf.CrossSection([np.array([(ri * math.cos(math.pi / 6 + k * math.pi / 3),
                                          ri * math.sin(math.pi / 6 + k * math.pi / 3)) for k in range(6)])])
    ring = mf.CrossSection.circle(head_d / 2, 48) - hexagon
    head = _from_manifold(mf.Manifold.extrude(ring, head_h)).apply_translation((0, 0, -head_h))
    return union([head, cylinder(d / 2, shank / 2, (0, 0, shank / 2))])


def _screws(x: Exporter) -> None:
    rows = (("screw_m3_socket_head", "M3 socket-head screw, 16 mm shank", .003, .0055, .003, .0025, .016, .001),
            ("screw_m4_socket_head", "M4 socket-head screw, 10 mm shank", .004, .007, .004, .003, .010, .002))
    ref = line_of(SC, "def screw(")
    for key, name, d, hd, hh, af, shank, mass in rows:
        screw = _screw_mesh(d, hd, hh, af, shank)
        x.add(key, name, "fixture", screw, visual=[_look(screw, "nickel")], material={"name": "nickel-plated steel"},
              mass=cited(mass, "authored", line_of(SC, "mass=.002 if diameter>.003"), "screw body link mass in their "
                         "scene (.002 kg M4, .001 kg M3)"),
              friction=_mu("thread_mu", "thread (steel screw in steel nut, electro-zinc): Bossard low end; " + BORE_QUOTE),
              friction_partner="zinc-plated steel nut or clamping strip",
              dims=[dim("head diameter", hd, "diameter_xy", ref), dim("total length", shank + hh, "z", ref)],
              source_ref=ref, notes="Socket head ring with the internal hex and the shank, along +z with the head "
                                    "below z = 0. The M3 body mass is authored (their 1 g).",
              extra={"pitch_m": .0005 if d < .004 else .0007})


def _bit_rotor(af: float) -> trimesh.Trimesh:
    radius = (af - .00005) / (2 * math.cos(math.pi / 6))
    az = math.pi / 6 + np.arange(6) * math.pi / 3
    verts = [(s * radius * math.cos(t), s * radius * math.sin(t), z)
             for z, s in ((-.040, .80), (-.0395, 1.), (-.022, 1.)) for t in az]
    bit = trimesh.Trimesh(np.array(verts)).convex_hull
    return union([cylinder(.00317, .020, (0, 0, -.011)), bit])


def _handle_groups(slim: bool, limit: float) -> dict[str, list[trimesh.Trimesh]]:
    """Handle cylinders by their render material (scene.py: outer sections dark, middle blue, rating band by limit)."""
    segs = (((.022, .030, .0063), (.052, .055, .0069), (.087, .015, .0063)) if slim else
            ((.030, .035, .011), (.065, .055, .018), (.105, .020, .014)))
    band = (.072, .006, .007) if slim else (.087, .006, .0185)
    cyl = lambda z, length, r: cylinder(r, length / 2, (0, 0, z))  # noqa: E731
    g = {"robot_dark": [cyl(*segs[0]), cyl(*segs[2])], "tool_blue": [cyl(*segs[1])],
         "tool_green" if limit < .5 else "tool_yellow": [cyl(*band)]}
    if slim:
        g["robot_dark"].append(cylinder(.0115, .015, (0, 0, .065)))
    return g


def _handle(slim: bool) -> trimesh.Trimesh:
    segs = (((.022, .030, .0063), (.052, .055, .0069), (.087, .015, .0063)) if slim else
            ((.030, .035, .011), (.065, .055, .018), (.105, .020, .014)))
    band = (.072, .006, .007) if slim else (.087, .006, .0185)
    parts = [cylinder(r, length / 2, (0, 0, z)) for z, length, r in (*segs, band)]
    if slim:
        parts.append(cylinder(.0115, .015, (0, 0, .065)))
    return union(parts)


def _driver(x: Exporter, key: str, name: str, slim: bool, af: float, total: float, limit: float) -> None:
    hlooks = [_look(union(v), k) for k, v in _handle_groups(slim, limit).items()]
    ref = line_of(SC, "def driver(")
    rotor = _bit_rotor(af)
    rho = 7850.
    m_rotor = rho * rotor.volume
    dens = cited(rho, "estimate", line_of(SC, "steel_cfg = cfg(mu=.35"), "steel shaft and hex bit, density 7850")
    clutch = line_of(SC, "limit_lower=-1.,limit_upper=1.")
    quote = ("Product masses: Wera 7440 180 g, HansTool 0176 32 g + 6 g collar. The steel shaft/bit rotor is ~10.5 g "
             "from its own shapes; handle densities give the remainder")
    x.add(f"{key}__handle", f"{name}: handle", "articulated", _handle(slim),
          visual=hlooks, material={"name": "plastic handle" + (" with grip collar" if slim else "")},
          mass=cited(total - m_rotor, "derived", line_of(SC, "slim=name.startswith"), quote),
          friction=cited(.75, "estimate", line_of(SC, "handle_cfg = cfg(mu=.75"), "handle contact friction in their scene"),
          friction_partner="robot finger pads and rubber V-groove holder",
          dims=[dim("grip diameter" if slim else "band diameter", 2 * (.0115 if slim else .0185), "diameter_xy", ref)],
          source_ref=ref, extra={"total_tool_mass_kg": total, "torque_limit_nm": limit},
          notes="Handle cylinders (and the 23 mm grip collar on the HansTool) as their collision shapes; handle mass "
                "is the datasheet product mass minus the steel rotor. The M3 handle's upper section has a 2.5 mm gap "
                "to the lower ones in their shapes." if not slim else
                "Handle cylinders and the authored 23 mm grip collar as their collision shapes; handle mass is the "
                "datasheet product mass (incl. 6 g collar) minus the steel rotor.")
    x.add(f"{key}__bit", f"{name}: shaft and hex bit", "articulated", rotor,
          visual=[_look(rotor, "nickel", "shaft and bit")], material={"name": "steel", "density": dens},
          friction=cited(.35, "estimate", line_of(SC, "steel_cfg = cfg(mu=.35"), "steel shaft and bit contact friction"),
          friction_partner="screw socket (nickel)",
          dims=[dim("shaft diameter", .00634, "diameter_xy", line_of(SC, '"_steel_shaft"'))],
          source_ref=ref, notes=f"Steel shaft and chamfered hex bit ({af * 1e3:.1f} mm AF less 50 um fit clearance); "
                                "a one-way detent clutch lets it turn against the handle.",
          extra={"joint": {"type": "revolute", "parent": f"{key}__handle", "axis": [0, 0, 1], "range": [-1., 1.],
                           "cite": clutch}})


def _drivers(x: Exporter) -> None:
    _driver(x, "driver_m3_wera_7440", "Torque driver M3 (Wera 7440 class, 2.5 mm hex, 0.35 N m)", False, .0025, .180,
            CF.Config.__dataclass_fields__["m3_tool_limit_nm"].default)
    _driver(x, "driver_m4_hanstool_0176", "Torque driver M4 (HansTool 0176 class, 3 mm hex, 0.60 N m)", True, .003,
            .038, CF.Config.__dataclass_fields__["m4_tool_limit_nm"].default)


# ---------------------------------------------------------------------------------------------- bench and stands
def _bench(x: Exporter) -> None:
    items = [((0, 0, .755), (.60, .375, .025))] + [((px, py, .365), (.023, .023, .365))
                                                   for px in (-.52, .52) for py in (-.31, .31)]
    ref = line_of(SC, 'box("worktop"')
    looks = _box_looks(items, ["laminate"] + ["steel"] * 4)
    looks[0]["name"] = "laminate top (their palette colour under the oak veneer texture, linear->sRGB)"
    x.add("workbench", "Workbench with laminate top", "furniture", _boxes(items), visual=looks, motion="static",
          material={"name": "laminate top on steel legs"},
          friction=cited(.45, "authored", line_of(SC, "cfg = lambda mu=.45"), "default shape friction in their scene"),
          friction_partner="parts on the bench",
          dims=[dim("top length", 1.2, "x", ref), dim("top depth", .75, "y", ref), dim("height", .78, "z", ref)],
          source_ref=ref, notes="1.2 x 0.75 m top (50 mm thick) and four square legs; authored, no datasheet.")


def _flag(x: Exporter) -> None:
    items = [((0, 0, 0), (.004, .004, .1535)), ((0, 0, -.149), (.020, .025, .005)), ((0, .005, .149), (.005, .001, .022))]
    ref = line_of(SC, "reference_flag_carriage")
    x.add("reference_flag", "Sliding white reference flag", "articulated", _boxes(items),
          visual=_box_looks(items, ["steel", "steel", "paper"]),
          material={"name": "steel stand with paper flag"},
          mass=cited(.050, "authored", ref, "flag carriage mass in their scene"),
          friction=cited(.45, "authored", line_of(SC, "cfg = lambda mu=.45"), "default shape friction"),
          friction_partner="the track", dims=[dim("height", .3135, "z", ref)], source_ref=ref,
          notes="Stand, foot and white flag of the carriage; slides along x on a static track.",
          extra={"joint": {"type": "prismatic", "parent": "world", "axis": [1, 0, 0], "range": [-.006, .10],
                           "cite": line_of(SC, "limit_lower=-.006,limit_upper=.10")}})


def _cradle(x: Exporter, key: str, name: str, R: float) -> None:
    apex = .797
    axis_z = apex + R * math.sqrt(2)
    items = [((0, .20, .788), (.028, .13, .008))]
    for cy0, cy1, ap in ((.245, .28, apex), (.165, .205, axis_z - .00317 * math.sqrt(2))):
        half, t = .012, .0015
        for side in (-1, 1):
            items.append(((side * (half + t) / math.sqrt(2), (cy0 + cy1) / 2, ap + (half - t) / math.sqrt(2)),
                          (half, (cy1 - cy0) / 2, t), QY(-side * math.pi / 4)))
    items += [((0, .20 + s * .132, .801), (.028, .003, .005)) for s in (-1, 1)]
    ref = line_of(SC, "def v_saddle")
    x.add(key, name, "fixture", _boxes(items), motion="static", material={"name": "rubber"},
          visual=[_look(_boxes(items), "robot_dark", "rubber holder")],
          friction=cited(.8, "authored", line_of(SC, "rubber=cfg(mu=.8"), "rubber V-groove tool holder in their scene"),
          friction_partner="driver handle", source_ref=ref,
          notes="Base plate, two 90 deg V saddles and two end lips; the shaft saddle height follows the handle radius.")


def _stands(x: Exporter) -> None:
    cam = [((.17, -.285, .786), (.035, .030, .006)), ((.17, -.285, .974), (.010, .010, .182)),
           ((.17, -.2735, 1.163), (.014, .0015, .014)), ((.17, -.260, 1.17), (.012, .012, .011))]
    ov = [((0, .372, .786), (.045, .030, .006)), ((0, .372, 1.496), (.015, .015, .704)),
          ((0, .352, 2.205), (.012, .006, .010)), ((0, .34, 2.22), (.018, .019, .014), QX(1.17))]
    fr = cited(.45, "authored", line_of(SC, "cfg = lambda mu=.45"), "default shape friction in their scene")
    x.add("inspection_camera_stand", "Inspection camera on a bench-clamped 20 x 20 mast", "fixture",
          _boxes(cam, (.17, -.285, .78)), motion="static",
          visual=_box_looks(cam, ["steel", "aluminum", "robot_dark", "robot_dark"], (.17, -.285, .78)), material={"name": "aluminium mast, plastic camera"},
          friction=fr, friction_partner="bench top", source_ref=line_of(SC, "inspection_camera_mast"),
          notes="Foot, mast, clamp and camera body boxes; authored.")
    x.add("overview_camera_stand", "Overview camera on a 30 x 30 profile post", "fixture",
          _boxes(ov, (0, .372, .78)), motion="static",
          visual=_box_looks(ov, ["steel", "aluminum", "steel", "robot_dark"], (0, .372, .78)), material={"name": "aluminium post, plastic camera"},
          friction=fr, friction_partner="bench top", source_ref=line_of(SC, "overview_camera_post"),
          notes="Base plate, post, bracket and tilted camera body boxes; authored.")


def export(x: Exporter) -> None:
    for step in (_sensor, _bef_bracket, _rail, _corner_bracket, _screws, _drivers, _bench, _flag,
                 lambda e: _cradle(e, "driver_cradle_m3", "V-groove cradle for the M3 driver", .018),
                 lambda e: _cradle(e, "driver_cradle_m4", "V-groove cradle for the M4 driver", .0115), _stands):
        try:
            step(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(getattr(step, "__name__", "step"), f"{type(exc).__name__}: {exc}")
    x.fail("conveyor_and_cabinets", "built only as render-only scenery in render.py (no collision body in scene.py)")
