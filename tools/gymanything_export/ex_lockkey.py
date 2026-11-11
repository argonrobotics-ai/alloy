# SPDX-License-Identifier: Apache-2.0
"""lockkey: the five-pin padlock (casing, plug, driver and key pins), shackle, key and hinged hasp box, from their
``lock_mesh`` builders (via the ``newton15`` geometry adapter) and the scene code in ``newton15/assembly.py``.

Articulated objects are exported as one spec per rigid part, ``<object>__<part>``, with the joint in ``extra.joint``.
"""

from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import trimesh
from _common import Exporter, box, cited, cylinder, dim, line_of, union, vis, where

from lockkey import lock_specs as LK
from pencap.newton15 import render as PR
from lockkey.newton15 import assembly as AS
from lockkey.newton15 import config as CFG
from lockkey.newton15 import geometry as NG
from lockkey.newton15 import mechanism as MECH
from lockkey.newton15 import render as RN
from lockkey.newton15 import shackle as SH

BRASS_QUOTE = "brass plug/shell/pins ~8.5 g/cc (WEB)"


def _srgb(lin) -> list[float]:
    """Their palette entries are UsdPreviewSurface diffuseColor values (linear); the part colour is sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]


def _extra_palette() -> dict:
    """The keyword entries of the ``dict(PALETTE, brass=..., nickel=..., paint_blue=...)`` call in ``render.export``."""
    for node in ast.walk(ast.parse(inspect.getsource(RN.export).lstrip())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "dict" and node.keywords:
            return {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    raise KeyError("palette dict call")


def _look(geom, material: str, name: str) -> dict:
    """Part with the palette entry ``material`` their lock scene binds (their own, else the shared bench palette)."""
    own = _extra_palette()
    if material in own:
        (rgb, rough, metal), cite = own[material], line_of(RN, f"{material}=((")
    else:
        (rgb, rough, metal), cite = PR.PALETTE[material], line_of(PR, f'"{material}": ((')
    return vis(geom, _srgb(rgb), cite, rough, metal, name=f"{name} ({material}, linear->sRGB)")


def _brass(c) -> dict:
    return {"name": "brass", "density": cited(LK.RHO_BRASS, "search", line_of(LK, "RHO_BRASS ="), BRASS_QUOTE)}


def _brass_mu(c) -> tuple[dict, str]:
    return cited(c.brass_friction, "authored", line_of(CFG, "brass_friction:"),
                 "brass_friction = 0.10 (lubricated brass on brass; calibration target)"), "brass parts"


def _joint(kind: str, parent: str, axis: list, rng, cite: str) -> dict:
    return {"joint": {"type": kind, "parent": parent, "axis": axis, "range": rng, "cite": cite}}


def _hull(m: trimesh.Trimesh) -> trimesh.Trimesh:
    return m.convex_hull


def _lock_body_parts(x: Exporter, c) -> None:
    lm = NG.legacy_mesh()
    casing, _ = lm.make_padlock_body(c.plug_length, c.chambers_x, LK.PIN_R + c.bore_clearance,
                                     (2 * c.keyway_half_width - c.blade_clearance + .0003) / 2,
                                     c.spine_z + LK.BLADE_HEIGHT / 2, LK.BLADE_HEIGHT + .001,
                                     align_shackle_bores=c.shackle_swivel)
    low, high = casing.bounds.copy()
    high[0] = c.face_x
    clip = box((high - low + np.array([0., .002, .002])) / 2, (low + high) / 2)
    casing = casing.intersection(clip)
    mu, partner = _brass_mu(c)
    x.add("pin_tumbler_lock__casing", "Padlock casing (rounded brass body with plug bore)", "articulated", casing,
          visual=[_look(casing, "brass", "casing")], motion="static", material=_brass(c), friction=mu, friction_partner=partner,
          source_ref=where(AS.build), extra={"frame": "origin on the plug axis at the key face reference"},
          notes="The body of their padlock cut for the plug bore, five chambers, keyway and shackle holes, trimmed at "
                "the key face as in their scene; the fixed base part of the lock. No mass given.")
    plug = NG.complete_plug(c)
    x.add("pin_tumbler_lock__plug", "Lock plug (5 chambers, keyway)", "articulated", plug,
          visual=[_look(plug, "brass", "plug")],
          material=_brass(c), friction=mu, friction_partner=partner,
          dims=[dim("diameter", LK.PLUG_DIA, "y", line_of(LK, "PLUG_DIA ")),
                dim("length", c.plug_length, "x", line_of(CFG, "plug_length:"))],
          source_ref=where(NG.complete_plug),
          extra=_joint("revolute", "pin_tumbler_lock__casing", [1, 0, 0], [math.radians(-.5), math.pi / 2],
                       line_of(MECH, 'label="plug_turn"')),
          notes="Cylinder with five blind bores and an open-bottom keyway (their complete_plug); mass is brass "
                "density times this volume in their sim.")


def _pin(x: Exporter, c, key: str, name: str, length: float, top: float, bot: float, joint: dict) -> None:
    lm = NG.legacy_mesh()
    tm = _hull(lm.make_pin_mesh(length, tip_top=top, tip_bot=bot))
    mu, partner = _brass_mu(c)
    finish = "nickel" if "driver" in key else "brass"          # their render.py: driver pins nickel, key pins brass
    x.add(key, name, "articulated", tm, visual=[_look(tm, finish, "pin")], material=_brass(c), friction=mu, friction_partner=partner,
          dims=[dim("length", length, "z", line_of(LK, "DRIVER_LEN =" if "driver" in key else "KEY_PIN_LEN_IN =")),
                dim("diameter", LK.PIN_DIA, "diameter_xy", line_of(LK, "PIN_DIA ="))],
          source_ref=where(MECH.build), extra=joint,
          notes="Chamfered pin cylinder (convex hull of their make_pin_mesh, as their collider); mass is brass "
                "density times volume (LK.pin_mass) in their sim.")


def _pins(x: Exporter, c) -> None:
    _pin(x, c, "pin_tumbler_lock__driver_pin", "Driver (top) pin", LK.DRIVER_LEN, 0.0, c.pin_chamfer,
         _joint("prismatic", "pin_tumbler_lock__casing", [0, 0, 1], None, line_of(MECH, "add_joint_prismatic(-1, body")))
    jr = [c.ledge_z - c.keyway_top - .0003, .006]
    for d in range(1, 8):
        _pin(x, c, f"pin_tumbler_lock__key_pin_d{d}", f"Key (bottom) pin, depth {d}", LK.KEY_PIN_LEN[d],
             c.pin_chamfer, c.bottom_pin_tip,
             _joint("prismatic", "pin_tumbler_lock__plug", [0, 0, 1], jr, line_of(MECH, "limit_lower=c.ledge_z")))


def _key(x: Exporter, c) -> None:
    parts = list(AS.key_parts(c, c.bitting))
    bc = (c.face_x + c.insertion_travel + .012, 0., c.spine_z + LK.BLADE_HEIGHT / 2 + .012)
    head = trimesh.creation.cylinder(radius=.011, height=.005, sections=44)
    head.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    neck = box((.004, .0025, .005), (-.0063, 0., -.008))
    bow = trimesh.util.concatenate([head, neck]).convex_hull
    bow.apply_translation(bc)
    mass = max(LK.RHO_BRASS * sum(p.volume for p in parts), .005)
    key = union([*parts, bow])
    x.add("padlock_key", "Kwikset KW1 key (bitting 31542)", "tool", key, visual=[_look(key, "nickel", "key")],
          material={"name": "nickel-silver / brass"},
          mass=cited(mass, "derived", line_of(AS, "set_mass(b, bodies[key]"),
                     "legacy scene mass: brass density x blade volume, at least 5 g (RHO_BRASS 8.5 g/cc, WEB)"),
          friction=cited(c.brass_friction, "authored", line_of(CFG, "brass_friction:"), "brass_friction = 0.10"),
          friction_partner="the plug and pins",
          source_ref=where(AS.key_parts),
          notes="KW1 blade with the 31542 bitting (prism segments of their make_key_blade_prisms) unioned with their "
                "simplified solid bow; body frame as in their scene (blade along -x).")


def _shackle(x: Exporter, c) -> None:
    parts = [cylinder(.003, length / 2, (-.0015, y, .060 - length / 2)) for y, length in ((-.010, .044), (.010, .030))]
    for i in range(16):
        a, h = (i + .5) * math.pi / 16, math.pi / 16
        pos = (-.0015, .010 * math.cos(a), .060 + .010 * math.sin(a))
        tangent = np.array([0., -math.sin(a), math.cos(a)])
        quat = trimesh.transformations.quaternion_from_matrix(trimesh.geometry.align_vectors([0, 0, 1], tangent))
        parts.append(cylinder(.003, (.010 * h + .0002) / 2, pos, (quat[1], quat[2], quat[3], quat[0])))
    cite_m = line_of(AS, "set_mass(b, shackle")
    shackle = union(parts)
    x.add("padlock_shackle", "Padlock shackle (6 mm, unequal legs)", "articulated", shackle,
          visual=[_look(shackle, "steel", "shackle")],
          material={"name": "steel"},
          mass=cited(.030, "authored", cite_m, "shackle body mass 30 g"),
          friction=cited(.4, "authored", line_of(AS, "cfg = newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=.4"),
                         "scene contact mu .4"), friction_partner="casing and hasp",
          dims=[dim("leg diameter", 2 * SH.LEG_RADIUS, "x", line_of(SH, "LEG_RADIUS ="))],
          source_ref=where(AS.build),
          extra=_joint("prismatic", "pin_tumbler_lock__casing", [0, 0, 1], [-.0005, c.shackle_travel],
                       line_of(SH, "add_guide")) | {"swivel_axis": [0, 0, 1]},
          notes="Two cylinder legs (44 and 30 mm) and a 16-segment arch, unioned. Slides 18 mm and swivels about z "
                "once the short leg is out (their d6 guide).")


def _hasp_box(x: Exporter, c) -> None:
    z, centre = c.table_height, -.102
    wall = .160
    lid_top = wall + .024
    pieces = [box((.15, .21, .0075), (0, 0, .0075))]
    for dx in (-.1425, .1425):
        pieces.append(box((.0075, .21, wall / 2), (dx, 0, wall / 2)))
    for y in (-.2025, .2025):
        pieces.append(box((.15, .0075, wall / 2), (0, y, wall / 2)))
    slot_z = z + .10 + .045 - z
    for y in (-.0195, -.0005):
        pieces.append(box(((.1095 - .048) / 2, .002, .002), ((.048 + .1095) / 2 - centre, y, slot_z)))
    pieces.append(box((.002, .0095, .002), (.1115 - centre, -.010, slot_z)))
    mu = cited(.4, "authored", line_of(AS, "cfg = newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=.4"),
               "scene contact mu .4")
    look = [_look(union(pieces[:5]), "paint_blue", "floor and walls"), _look(union(pieces[5:]), "steel", "staple")]
    x.add("hinged_box__body", "Hasp chest (open box with staple)", "articulated", union(pieces), motion="static",
          visual=look,
          material={"name": "painted steel"}, friction=mu, friction_partner="lid and shackle",
          dims=[dim("height", wall, "z", line_of(AS, "wall_height = ")), dim("width", .42, "y", line_of(AS, "box_side_"))],
          source_ref=where(AS.build),
          notes="Floor, four 15 mm-thick walls and the fixed staple (their boxes) unioned; origin at the floor centre. "
                "No mass given (fixed).")
    lc = wall + .012
    lid = [box((.16, .22, .012))]
    lid += [box((.14, .003, .002), (0, y, .014)) for y in (-.15, .15)]
    sy_z, sw, sh = .045 + .10 - lc + 0.0, .030, .014
    plate_x = .048 + .004 + .0015 - centre
    slot_zr = (.10 + .045) - lc
    above = lid_top - (slot_zr + lc + sh / 2)
    lid.append(box((.0015, .025, above / 2), (plate_x, -.010, slot_zr + sh / 2 + above / 2)))
    lid.append(box((.0015, .025, .008), (plate_x, -.010, slot_zr - sh / 2 - .008)))
    for sy in (-1, 1):
        lid.append(box((.0015, .005, sh / 2), (plate_x, -.010 + sy * .020, slot_zr)))
    lid.append(box((.025, .025, .0015), (.052 - centre, -.010, lid_top + .0015 - lc)))
    look = [_look(lid[0], "paint_blue", "lid slab"), _look(union(lid[1:]), "steel", "ribs and hasp")]
    x.add("hinged_box__lid", "Hasp chest lid with slotted hasp", "articulated", union(lid), visual=look,
          material={"name": "painted steel"},
          mass=cited(1.8, "authored", line_of(AS, "set_mass(b, lid"), "lid mass 1.8 kg"),
          friction=mu, friction_partner="chest and shackle",
          dims=[dim("length", .32, "x", line_of(AS, 'box("lid"')), dim("width", .44, "y", line_of(AS, 'box("lid"'))],
          source_ref=where(AS.build),
          extra=_joint("revolute", "hinged_box__body", [0, 1, 0], [math.radians(-95), math.radians(.5)],
                       line_of(AS, 'label="lid_hinge"')) | {"anchor_in_parent": [-.16, 0.0, lid_top],
                                                           "frame": "origin at the lid centre"},
          notes="Lid slab, two ribs, slotted hasp and its bridge (their boxes) unioned in the lid frame.")


def export(x: Exporter) -> None:
    c = CFG.Config()
    for fn in (_lock_body_parts, _pins, _key, _shackle, _hasp_box):
        try:
            fn(x, c)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
