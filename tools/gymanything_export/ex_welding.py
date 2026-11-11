# SPDX-License-Identifier: Apache-2.0
"""welding: the two MIG guns (``torch.py`` part tables), the coupon plates and the fixture-table cell equipment.

Gun geometry is their ``GunSpec.parts()`` (mount, body, neck capsule, tapered nozzle) in the flange frame, unioned
because the parts are rigidly fixed to each other. Fixture boxes come from ``layout.py`` / ``scene.py`` constants.
"""

from __future__ import annotations

import numpy as np
from _common import Exporter, box, capsule, cited, cylinder, dim, line_of, mesh, revolve, union, vis, where

from welding import layout as LY
from welding import params as PR
from welding import render as RN
from welding import scene as SC
from welding import shop as SH
from welding import torch as TO
from welding.process import tee_coupon


def _matrix(centre, rot) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(rot, float)
    T[:3, 3] = centre
    return T


def _gun_parts(spec: TO.GunSpec) -> list[tuple[str, object]]:
    parts = []
    for name, kind, c, R, d, _mass in spec.parts():
        if kind == "cylinder":
            m = cylinder(d[0], d[1])
        elif kind == "capsule":
            m = capsule(d[0], d[1])
        else:
            rb, rt, h = d
            m = revolve([(0, -h), (rb, -h), (rt, h), (0, h)])
        parts.append((name, m.apply_transform(_matrix(c, R))))
    return parts


def _gun_mesh(spec: TO.GunSpec):
    return union([m for _n, m in _gun_parts(spec)])


def _gun_visual(spec: TO.GunSpec) -> list[dict]:
    """Their gun materials: grey mount, black jacket (body, neck), copper nozzle (render.py ``gun_grey`` ...)."""
    looks = {"gun_mount": ((70, 72, 76), .35, .7, "gun_grey = A.pbr", "mount"),
             "gun_body": ((20, 20, 22), .6, 0., "black = A.pbr", "jacket"),
             "gun_neck": ((20, 20, 22), .6, 0., "black = A.pbr", "jacket"),
             "gun_nozzle": (RN.NOZZLE_COPPER_SRGB, .35, 1., "NOZZLE_COPPER_SRGB =", "copper nozzle")}
    out = []
    for name, m in _gun_parts(spec):
        col, rough, metal, pat, label = looks[name]
        out.append(vis(m, col, line_of(RN, pat), rough, metal, name=label))
    return out


def _steel_friction() -> tuple[dict, str]:
    return cited(SC.MU_STEEL, "literature", line_of(SC, "MU_STEEL ="), "mild steel on mild steel, dry (static ~0.74, "
                 "kinetic ~0.57) [literature, rounded]"), "mild steel"


def _steel_density() -> dict:
    return cited(SC.STEEL_RHO, "literature-read", line_of(PR, "STEEL_RHO = P("), PR.STEEL_RHO.basis)


def _guns(x: Exporter) -> None:
    rows = (("mig_gun_ba1", TO.BA1, "MIG gun BA1 (Tregaskiss class cobot gun, UR10e)", "derived",
             [("work point out along flange axis", 0.350, "z")]),
            ("mig_light_torch_rm2", TO.RM2, "Light MIG torch RM2 on KS-2 collision sensor (FR3, Panda, G1)",
             "derived", []))
    mu, partner = _steel_friction()
    for key, g, name, basis, dims in rows:
        try:
            ref = line_of(TO, f"{'BA1' if g is TO.BA1 else 'RM2'} = GunSpec")
            mass, _ = g.mass_properties()
            x.add(key, name, "tool", _gun_mesh(g),
                  material={"name": "steel, copper and insulator parts (part masses authored/read)"},
                  visual=_gun_visual(g), mass=cited(mass, basis, ref, g.source[:300]), friction=mu, friction_partner=partner,
                  dims=[dim(n, v, a, ref) for n, v, a in dims], source_ref=where(TO.GunSpec.parts),
                  notes="Four rigid parts (mount, body, neck capsule, nozzle cone) unioned in the flange frame; "
                        "geometry authored by them from maker datasheet values, no CAD.",
                  extra={"tcp_flange_m": [float(v) for v in g.tcp], "neck_deg": g.neck_deg})
        except Exception as exc:  # noqa: BLE001
            x.fail(key, f"{type(exc).__name__}: {exc}")


def _coupon(x: Exporter) -> None:
    c = tee_coupon()
    mu, partner = _steel_friction()
    dens = _steel_density()
    ref = where(tee_coupon)
    bx, by, bz = c.base
    wx, wt, wh = c.web
    scale = [vis(box((bx / 2, by / 2, bz / 2)), RN.MILL_SCALE_SRGB, line_of(RN, "MILL_SCALE_SRGB ="), name="mill scale")]
    x.add("steel_base_plate", "A36 steel base plate", "tool", box((bx / 2, by / 2, bz / 2)), visual=scale,
          material={"name": "A36 mild steel", "density": dens}, friction=mu, friction_partner=partner,
          dims=[dim("length", bx, "max", ref), dim("thickness", bz, "min", ref)], source_ref=ref,
          notes="Rectangular 200 x 150 x 10 mm plate (their tee coupon default); mass from density x volume.")
    scale = [vis(box((wx / 2, wt / 2, wh / 2)), RN.MILL_SCALE_SRGB, line_of(RN, "MILL_SCALE_SRGB ="), name="mill scale")]
    x.add("steel_bracket", "A36 steel bracket (tee web)", "tool", box((wx / 2, wt / 2, wh / 2)), visual=scale,
          material={"name": "A36 mild steel", "density": dens}, friction=mu, friction_partner=partner,
          dims=[dim("height", wh, "z", ref), dim("thickness", wt, "min", ref)], source_ref=ref,
          notes="Rectangular 150 x 8 x 75 mm web of the tee coupon; mass from density x volume.")


def _table(x: Exporter) -> None:
    T = LY.TABLE
    z, (sx, sy), t = T["top_z"], T["size"], T["top_thickness"]
    parts = [box((sx / 2, sy / 2, t / 2), (0, 0, z - t / 2))]
    for px in (-sx / 2 + .06, sx / 2 - .06):
        for py in (-sy / 2 + .06, sy / 2 - .06):
            parts.append(box((T["leg"] / 2,) * 2 + ((z - t) / 2,), (px, py, (z - t) / 2)))
    ref = line_of(LY, "TABLE = dict")
    look = [vis(parts[0], (58, 60, 63), line_of(RN, "base = (np.array([58, 60, 63]"), .42, .85,
                name="nitrided top (base grey of their hole-pattern texture, not exported)"),
            vis(union(parts[1:]), (52, 54, 58), line_of(RN, 'legs = A.pbr("table_legs"'), .5, .6, name="legs")]
    x.add("fixture_table", "Fixture welding table (Siegmund System 16 class)", "furniture", union(parts),
          visual=look, motion="static", material={"name": "steel"}, friction=cited(.6, "estimate", line_of(SC, "mu=.6"),
          "floor ground plane mu .6; the table shares the default steel cfg"), friction_partner="steel workpiece",
          dims=[dim("working height", z, "z", ref), dim("length", sx, "x", ref), dim("depth", sy, "y", ref)],
          source_ref=where(SC.Scene.room), notes="Top plate and four square-tube legs unioned; fixed in place.")


def _clamps(x: Exporter) -> None:
    H, R = LY.CLAMP_PAD["height"], LY.CLAMP_PAD["radius"]
    zp = LY.WELD_ORIGIN[2]
    ref = where(SC.Scene.clamps)
    body = union([box((.022, .03, .025), (0, -.085, LY.TABLE["top_z"] + .025)),
                  box((.009, .055, .008), (0, -.045, zp + .035))])
    zinc = [vis(body, (168, 170, 172), line_of(RN, 'zinc = A.pbr("zinc"'), .35, .9, name="zinc-plated steel")]
    x.add("toggle_clamp__body", "Toggle hold-down clamp, base and arm (DESTACO 225-U class)", "articulated", body,
          visual=zinc, motion="static", material={"name": "painted steel"}, source_ref=ref,
          notes="Static base box and locked arm box of their clamp; the spindle pad is the moving part.",
          extra={"clamp_force_n": LY.CLAMP_FORCE,
                 "holding_capacity_n": {"value": 2200., "cite": line_of(LY, "DESTACO 225-U")}})
    x.add("toggle_clamp__pad", "Toggle clamp spindle and neoprene pad", "articulated", cylinder(R, .030),
          visual=[vis(cylinder(R, .030), (22, 22, 22), line_of(RN, 'neoprene = A.pbr("neoprene"'), .8, name="neoprene")],
          material={"name": "neoprene"}, mass=cited(.3, "authored", ref, "spindle + pad: 0.3 kg as a 24 mm x 60 mm "
                                                    "cylinder"),
          friction=cited(SC.MU_PAD, "estimate", line_of(SC, "MU_PAD ="), "neoprene spindle cap on steel [estimate]"),
          friction_partner="the steel plate", dims=[dim("pad diameter", 2 * R, "diameter_xy",
                                                        line_of(LY, "CLAMP_PAD ="))],
          source_ref=ref, notes="Spindle + pad drawn as their 24 mm x 60 mm inertia cylinder (0.3 kg); the collider is only its 10 mm pad.",
          extra={"joint": {"type": "prismatic", "parent": "toggle_clamp__body", "axis": [0, 0, 1],
                           "range": [-.02, .03], "cite": ref}})


def _work_clamp_and_source(x: Exporter) -> None:
    W, S = LY.WORK_CLAMP, LY.SOURCE
    sx, sy, sz = W["size"]
    x.add("ground_clamp", "Magnetic ground (work) clamp (Strong Hand GM205 class)", "tool",
          box((sx / 2, sy / 2, sz / 2)),
          visual=[vis(box((sx / 2, sy / 2, sz / 2)), (150, 28, 22), line_of(RN, 'housing = A.pbr("magnet_housing"'), .5,
                      .2, name="red housing")], material={"name": "copper"},
          mass=cited(W["mass"], "datasheet", line_of(LY, "WORK_CLAMP ="), "magnetic ground clamp, Strong Hand GM205 "
                     "class [read: 500 A, 490 N pull]"),
          friction=cited(SC.MU_STEEL, "literature", line_of(SC, "MU_STEEL ="), "default steel contact"),
          friction_partner="the steel table top", dims=[dim("length", sx, "x", line_of(LY, "WORK_CLAMP ="))],
          source_ref=where(SC.Scene.work_clamp), notes="Box of their size; 1.3 kg as read for the class.")
    ref = line_of(LY, "SOURCE = dict")
    w, l, h = S["case"]
    paint = (66, 71, 77)
    pcite = line_of(SH, 'case = A.pbr("source_paint"')
    x.add("power_source", "Welding power source (Miller Auto-Continuum 350)", "fixture",
          box((w / 2, l / 2, h / 2)), visual=[vis(box((w / 2, l / 2, h / 2)), paint, pcite, .45, .3, name="case paint")],
          motion="static", material={"name": "sheet steel case"},
          mass=cited(59.4, "derived", ref, "Miller spec sheet AU/10.0, Mar. 2022, p. 8: 130 lb = 59.4 kg"),
          dims=[dim("height", h, "z", ref), dim("depth", l, "y", ref)], source_ref=where(LY.source_boxes),
          notes="Case box only (running gear, feeder, cylinder are separate objects).")
    fw, fl, fh = S["feeder"]
    x.add("wire_feeder", "Wire feeder (Continuum class)", "fixture", box((fw / 2, fl / 2, fh / 2)),
          visual=[vis(box((fw / 2, fl / 2, fh / 2)), paint, pcite, .45, .3, name="case paint")], motion="static",
          material={"name": "sheet steel case"}, dims=[dim("height", fh, "z", ref), dim("width", fw, "x", ref)],
          source_ref=where(LY.source_boxes), notes="Box; length (600 mm) is their estimate, no mass given.")
    cy = S["cylinder"]
    x.add("gas_cylinder", "Shielding-gas cylinder (size 300)", "fixture", cylinder(cy["d"] / 2, cy["h"] / 2),
          visual=_cylinder_visual(cy["d"] / 2, cy["h"]), motion="static", material={"name": "painted steel"},
          dims=[dim("height", cy["h"], "z", ref), dim("diameter", cy["d"], "diameter_xy", ref)],
          source_ref=where(LY.source_boxes), notes="Solid cylinder outline (their collider); no mass given.")


def _cylinder_visual(r: float, h: float) -> list[dict]:
    """Grey body and bright-green shoulder (their equipment(): shoulder = top 0.12 m, tapering to 0.045 m)."""
    zb, zs, zt = -h / 2, h / 2 - .12, h / 2
    body = revolve([(0, zb), (r, zb), (r, zs), (0, zs)])
    shoulder = revolve([(0, zs), (r, zs), (.045, zt), (0, zt)])
    return [vis(body, (128, 132, 132), line_of(SH, 'body = A.pbr("cylinder_paint"'), .4, .2, name="body paint"),
            vis(shoulder, (40, 160, 70), line_of(SH, 'green = A.pbr("cylinder_shoulder"'), .4, .1,
                name="shoulder (EN 1089-3 green)")]


def _mount_look(box_name: str) -> tuple:
    """Their camera-mount boxes: black camera bodies, anodised aluminium extrusions (shop.py)."""
    if "camera" in box_name:
        return (25, 26, 28), line_of(SH, 'cam_m = A.pbr("camera_body"'), .4, .3, "camera body"
    return (160, 164, 168), line_of(SH, 'alu = A.pbr("extrusion"'), .35, .85, "anodised extrusion"


def _screens_and_mounts(x: Exporter) -> None:
    sname, c, hh, _yaw = LY.screen_boxes()[0]
    ref = line_of(LY, "SCREENS = [")
    x.add("welding_screen", "Welding screen, 6 x 6 ft (EN ISO 25980 class)", "fixture", box(hh),
          visual=[vis(box(hh), (255, 120, 20), line_of(SH, 'vinyl = A.glass("screen_vinyl"'), name="glass orange vinyl")],
          motion="static",
          material={"name": "vinyl curtain in a frame"}, dims=[dim("width", 2 * hh[0], "x", ref),
                                                               dim("height", 2 * hh[2], "z", ref)],
          source_ref=where(LY.screen_boxes), notes="One 1.83 m x 1.83 m x 20 mm slab (three in the cell).")
    mounts = {n: (cc, h) for n, cc, h in LY.CAMERA_MOUNTS}
    for key, names, label in (("bench_camera_mount", ("bench_drop_rod", "bench_camera_body"), "Bench camera drop rod"),
                              ("side_camera_mount", ("side_base", "side_post", "side_camera_body"),
                               "Side camera base, post and body")):
        parts = [box(mounts[n][1], mounts[n][0]) for n in names]
        m = union(parts)
        looks = [vis(p, col, cite, r, mt, name=lab) for p, n in zip(parts, names)
                 for col, cite, r, mt, lab in [_mount_look(n)]]
        x.add(key, label, "fixture", m, visual=looks, motion="static", material={"name": "steel and plastic"},
              source_ref=line_of(LY, "CAMERA_MOUNTS ="), notes="Authored collision boxes of the mount (unioned).")


def export(x: Exporter) -> None:
    _guns(x)
    _coupon(x)
    for step in (_table, _clamps, _work_clamp_and_source, _screens_and_mounts):
        try:
            step(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(step.__name__, f"{type(exc).__name__}: {exc}")
