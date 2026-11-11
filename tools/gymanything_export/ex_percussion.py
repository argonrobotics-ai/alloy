# SPDX-License-Identifier: Apache-2.0
"""percussion: 13 undercut rosewood xylophone bars, the xylophone frame, two resonator-bar boxes, the 14 in drum
on its stand, the 5A drumsticks, an Orff mallet and the felt beater.

Bar and stick profiles come from their ``modes.bar`` / ``modes.stick_geometry`` tables; the other shapes are rebuilt
from the same ``params``/``scene`` constants their ``Scene`` uses for its colliders.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from _common import Exporter, box, capsule, cited, comment_of, cylinder, dim, line_of, mesh, revolve, sphere, union, vis, where

from percussion import dressing as DR
from percussion import layout as LY
from percussion import modal, modes
from percussion import params as P
from percussion import render as RN
from percussion import scene as SC


def _look(geom, name: str, cite_pat: str, label: str = "", tint=None):
    """Render part from their ``COLORS`` table (name -> linear diffuse, roughness, metallic; OmniPBR takes the colour
    as linear, so it is converted to sRGB here). ``tint`` replaces the colour for their textured materials."""
    rgb, rough, metal = RN.COLORS[name]
    lin = np.clip(np.asarray(tint if tint is not None else rgb, float), 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return vis(geom, srgb.tolist(), line_of(RN, cite_pat), rough, metal, name=(label or name) + " (linear->sRGB)")


def _textured_look(geom, mat: str, rough: float, tint, label: str):
    """Render part for their ``S.textured`` veneers: the diffuse photo is not exported, the material's ``tint`` is."""
    lin = np.clip(np.asarray(tint, float), 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return vis(geom, srgb.tolist(), line_of(RN, f'S.textured("{mat}"'), rough, 0.0,
               name=f"{label} (tint of the veneer texture, linear->sRGB; photo not exported)")


def _rosewood_look(geom):
    """The bars' veneer photo is not exported and carries no tint; their rosewood clave colour stands in, with the
    veneer material's roughness (``S.textured("rosewood", ..., .38)``)."""
    lin = np.array([0.38, 0.15, 0.08])
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return vis(geom, srgb.tolist(), line_of(DR, 'S.color("rosewood_clave", (.38, .15, .08)'), 0.38, 0.0,
               name="rosewood (their rosewood clave colour; veneer photo not exported; linear->sRGB)")


MAPLE_TINT = (0.88, 0.72, 0.52)
WALNUT_TINT = (0.62, 0.40, 0.24)


def _mf():
    import manifold3d as mf  # noqa: PLC0415
    return mf


def _extrude_xz(poly_xz: np.ndarray, width: float) -> trimesh.Trimesh:
    """Closed (x, z) polygon extruded symmetrically along y by ``width`` (watertight, via manifold3d)."""
    mf = _mf()
    solid = mf.Manifold.extrude(mf.CrossSection([np.asarray(poly_xz, float)]), width).to_mesh()
    m = mesh(solid.vert_properties[:, :3], solid.tri_verts)
    R = np.array([[1, 0, 0, 0], [0, 0, -1, width / 2], [0, 1, 0, 0], [0, 0, 0, 1.]])
    return m.apply_transform(R)


def _difference(a: trimesh.Trimesh, cuts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.boolean.difference([a, *cuts], engine="manifold")


def _between(p0, p1, r: float) -> trimesh.Trimesh:
    """Capsule of radius r from p0 to p1 (hemispherical caps beyond the ends by r, as their capsule shapes)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    m = trimesh.creation.capsule(length, r, count=[16, 16])
    T = trimesh.geometry.align_vectors([0, 0, 1], d / length)
    T[:3, 3] = 0.5 * (p0 + p1)
    return m.apply_transform(T)


def _z_to_x(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate a z-axis solid so its axis lies along +x (the tool frame of their tools)."""
    return m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))


def _bar_mesh(note: str) -> trimesh.Trimesh:
    bd = modes.bar(note)
    x, h = np.asarray(bd["x"], float), np.asarray(bd["h"], float)
    h_node = np.r_[h[0], 0.5 * (h[1:] + h[:-1]), h[-1]]
    top = P.BAR_THICK_M / 2
    poly = np.vstack([np.c_[x, np.full_like(x, top)][::-1], np.c_[x, top - h_node]])
    blank = _extrude_xz(poly, P.BAR_WIDTH_M)
    holes = [cylinder(P.BAR_HOLE_R_M, 0.02, (float(xn), 0, 0)) for xn in bd["nodes"]]
    return _difference(blank, holes)


def _bars(x: Exporter) -> None:
    dens = cited(P.BAR_RHO, "literature-read", line_of(P, "BAR_RHO ="), comment_of(P, "BAR_RHO ="))
    mu = cited(0.45, "authored", line_of(SC, "mu=0.45"), "bar collision boxes in their scene (MuJoCo contact)")
    for note in P.XYLO_NOTES:
        key = "bar_" + note.lower()
        try:
            L = P.bar_length(note)
            cite = line_of(P, "def bar_length")
            geom = _bar_mesh(note)
            x.add(key, f"Rosewood xylophone bar {note} (Sonor Orff soprano type)", "other", geom,
                  material={"name": "rosewood", "density": dens}, friction=mu, visual=[_rosewood_look(geom)],
                  friction_partner="felt pads and pins of the frame",
                  dims=[dim("length", L, "x", cite), dim("width", P.BAR_WIDTH_M, "y", line_of(P, "BAR_WIDTH_M =")),
                        dim("thickness", P.BAR_THICK_M, "z", line_of(P, "BAR_THICK_M ="))],
                  source_ref=where(modes.build_bar),
                  notes="Undercut profile solved by them to Rucz's measured partials, 6.4 mm node holes; mass left to "
                        "density x volume (their rigid mass sums the same profile without the holes).")
        except Exception as exc:  # noqa: BLE001
            x.fail(key, f"{type(exc).__name__}: {exc}")


def _frame(x: Exporter) -> None:
    X, foot = P.XYLO_BOX, 0.004
    L, H = X["length"], X["height"]
    parts = []
    for k in range(3):
        w = X["width_low"] + (X["width_high"] - X["width_low"]) * (k + 0.5) / 3
        parts.append(box((L / 6, w / 2, H / 2), (-L / 2 + L * (k + 0.5) / 3, 0., H / 2)))
    for sx in (-L / 2 + 0.04, L / 2 - 0.04):
        for sy in (-0.1, 0.1):
            parts.append(box((0.015, 0.015, foot / 2), (sx, sy * (0.8 if sx < 0 else 0.55), -foot / 2)))
    ref = line_of(P, "XYLO_BOX = dict")
    looks = [_textured_look(union(parts[:3]), "maple_tex", .45, MAPLE_TINT, "maple frame"),
             _look(union(parts[3:]), "rubber", "rubber=((", "rubber feet")]
    x.add("xylophone_frame", "Orff xylophone frame (Studio 49 AX 1000 class)", "other", union(parts), visual=looks,
          material={"name": "wood resonator frame"},
          mass=cited(X["mass"], "datasheet", ref, "length, height, mass: Studio 49 AX 1000 datasheet [S26]"),
          friction=cited(0.9, "authored", line_of(SC, "xylo_foot_"), "rubber feet on the table, mu 0.9 in their scene"),
          friction_partner="table top",
          dims=[dim("length", L, "x", ref), dim("height", H + foot, "z", ref)], source_ref=where(SC.Scene.xylophone),
          notes="Solid trapezoid hull (three boxes) with four rubber feet; the real frame is hollow, so the card "
                "density is low (4.4 kg over the hull volume).")


def _resonator_box(x: Exporter, note: str) -> None:
    B = modal.resonator_bar_box(note)
    W, L, H, t = B["width"], B["length"], B["height"], B["wall"]
    outer = box((W / 2, L / 2, H / 2), (0, 0, H / 2))
    inner = box((W / 2 - t, L / 2 - t, H / 2 - t), (0, 0, H / 2))
    slot = box((B["slot_w"] / 2, B["slot_l"] / 2, t), (0, 0, H - t / 2))
    parts = [_difference(outer, [inner, slot])]
    for sy in (-1, 1):
        for sx in (-1, 1):
            parts.append(box((0.006, 0.008, 0.0015), (sx * (W / 2 - 0.008), sy * (L / 2 - 0.02), -0.0015)))
    ds = line_of(P, "RESONATOR_BAR_DATASHEET =")
    whole = union(parts)
    x.add(f"resonator_box_{note.lower()}", f"Resonator bar box {note} (Studio 49 KB/AX class)", "container",
          whole, visual=[_textured_look(whole, "maple_tex", .45, MAPLE_TINT, "maple box")], material={"name": "maple", "density": cited(P.MAPLE_RHO, "literature", line_of(P, "MAPLE_RHO ="),
                                                                    comment_of(P, "MAPLE_RHO ="))},
          mass=cited(B["mass"], "derived", where(modal._box_impl), "listed 0.40 kg minus the bar (dealer datasheet "
                     "[S27b]); wall derived from it"),
          friction=cited(0.9, "authored", line_of(SC, "resbox_{note}_foot"), "rubber feet, mu 0.9 in their scene"),
          friction_partner="table top",
          dims=[dim("length", L, "y", ds), dim("width", W, "x", ds), dim("height", H, "z", ds)],
          source_ref=where(SC.Scene.resonator_bar),
          notes="Hollow maple box of the datasheet outer size, wall derived by them from the listed mass, slot "
                "under the bar tuned to its first partial, four rubber feet.",
          extra={"slot_m": [B["slot_w"], B["slot_l"]]})


def _drum_groups() -> dict[str, list[trimesh.Trimesh]]:
    """The drum and stand pieces grouped by their render material (shell, head, hoop, chrome stand, rubber)."""
    a, depth, zh = P.DRUM_RADIUS_M, P.DRUM_SHELL_DEPTH_M, 0.74
    r0, r1 = P.DRUM_HOOP_R_M
    ring = lambda ri, ro, z0, z1: revolve([(ri, z0), (ro, z0), (ro, z1), (ri, z1)])  # noqa: E731
    g = {"shell": [ring(a - 0.003, a + 0.003, -depth - 0.004, 0.0)], "hoop": [ring(r0, r1, -0.012, P.DRUM_HOOP_H_M)],
         "head": [cylinder(a - 0.004, 0.001, (0, 0, -SC.HEAD_RECESS_M - 0.001))], "chrome": [], "rubber": []}
    zf = -zh
    col_top, col_base = -depth - 0.03, zf + 0.30
    g["chrome"].append(_between((0, 0, col_base), (0, 0, col_top), 0.011))
    for k in range(3):
        d = np.array([math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3), 0.])
        elbow = (a + 0.009) * d + np.array([0, 0, -depth - 0.016])
        foot = 0.27 * d + np.array([0, 0, zf + 0.015])
        g["chrome"] += [_between((0, 0, col_top), elbow, 0.006), _between((0, 0, col_base), foot, 0.009)]
        g["rubber"] += [_between(elbow, elbow + [0, 0, 0.045], 0.005), sphere(0.015, foot)]
    return g


def _drum_visual(g: dict[str, list[trimesh.Trimesh]]) -> list[dict]:
    u = {k: union(v) for k, v in g.items()}
    return [_textured_look(u["shell"], "walnut_shell", .25, WALNUT_TINT, "walnut shell"),
            _look(u["head"], "head_coated", "head_coated=((", "coated head"),
            _look(u["hoop"], "chrome", "chrome=((", "chrome hoop"),
            _look(u["chrome"], "chrome", "chrome=((", "chrome stand"),
            _look(u["rubber"], "rubber", "rubber=((", "rubber hooks and feet")]


def _drum(x: Exporter) -> None:
    a, depth = P.DRUM_RADIUS_M, P.DRUM_SHELL_DEPTH_M
    groups = _drum_groups()
    parts = [m for v in groups.values() for m in v]
    total = P.DRUM_MASS_KG + P.STAND_MASS_KG + 0.05
    x.add("drum_14in_with_stand", "Drum 14 x 5 in, coated head, on a snare stand", "other", union(parts),
          visual=_drum_visual(groups),
          material={"name": "wood shell, steel hoop, Mylar head, steel stand"},
          mass=cited(total, "estimate", line_of(P, "DRUM_MASS_KG ="), comment_of(P, "DRUM_MASS_KG =") +
                     "; stand 3.2 kg (catalogue 2.8-4 kg); head 0.05 kg authored"),
          friction=cited(0.9, "authored", line_of(SC, "stand_foot_"), "rubber stand feet, mu 0.9 in their scene"),
          friction_partner="carpet floor",
          source_ref=where(SC.Scene.drum),
          notes="Shell, hoop, head disc and stand are one rigid body in their sim and are unioned here; shell and hoop "
                "as revolved rings (their colliders are 16 and 24 boxes).",
          extra={"head_radius_m": a, "head_film_thickness_m": P.DRUM_FILM_M})


def _stick_mesh() -> trimesh.Trimesh:
    xs, _xm, d = modes.stick_geometry()
    xm = 0.5 * (xs[1:] + xs[:-1])
    prof = [(0., 0.), *[(float(dd) / 2, float(xx)) for xx, dd in zip(xm, d)], (0., float(xs[-1]))]
    return _z_to_x(revolve(prof))


def _sticks(x: Exporter) -> None:
    S = P.STICK
    ref = line_of(P, "STICK = dict")
    fr = line_of(P, '"hickory_tip": {"mylar"')
    stick = _stick_mesh()
    x.add("drumstick_5a", "Drumstick Vic Firth American Classic 5A", "tool", stick,
          visual=[_look(stick, "hickory", "hickory=((")],
          material={"name": "hickory", "density": cited(modes.stick()["rho"], "derived", where(modes.build_stick),
                                                         "density that gives the datasheet mass over their profile")},
          mass=cited(S["mass"], "datasheet", ref, comment_of(P, "STICK = dict") or "maker datasheet [S29]"),
          friction=cited(P.FRICTION["hickory_tip"]["rosewood"], "estimate", fr,
                         comment_of(P, '"hickory_tip": {"mylar"')), friction_partner="rosewood bar",
          dims=[dim("length", S["L"], "x", ref), dim("butt diameter", S["D"], "y", ref)],
          source_ref=where(modes.stick_geometry),
          notes="Revolved diameter profile of their stick (shaft, taper, oval tip), butt at x = 0.")


def _mallet(x: Exporter, key: str, name: str, spec: dict, head_note: str) -> None:
    L, rh, rr = spec["L"], spec["handle_r"], spec["head_r"]
    Lh = L - 2 * rr
    handle = _z_to_x(capsule(rh, (Lh + 0.002 - rh) / 2, (0, 0, (Lh + 0.002 + rh) / 2)))
    head = sphere(rr, (L - rr, 0, 0))
    mass = spec["handle_rho"] * math.pi * rh ** 2 * Lh + spec["head_mass"]
    var = "MALLET" if key == "mallet" else "BEATER"
    ref = line_of(P, f"{var} = dict")
    mat = "rubber_mallet" if key == "mallet" else "felt"
    head_look = "mallet_rubber" if key == "mallet" else "beater_felt"
    x.add(key, name, "tool", union([handle, head]),
          visual=[_look(handle, "maple", "maple=((", "handle"),
                  _look(head, head_look, f"{head_look}=((", "head")],
          material={"name": "hardwood handle (700 kg/m3), " + head_note},
          mass=cited(mass, "estimate", ref, "dimensions are estimates; handle mass = handle_rho x handle volume + "
                     f"head_mass {spec['head_mass']} kg"),
          friction=cited(P.FRICTION[mat]["rosewood"], "estimate", line_of(P, f'"{mat}": {{"mylar"'),
                         comment_of(P, f'"{mat}": {{"mylar"')), friction_partner="rosewood bar",
          dims=[dim("length", L, "x", ref), dim("head diameter", 2 * rr, "y", ref)],
          source_ref=where(SC.Scene._mallet),
          notes="Capsule handle and spherical head along +x from the butt; all dimensions are their estimates.")


def export(x: Exporter) -> None:
    _bars(x)
    steps = (_frame, lambda e: _resonator_box(e, "C4"), lambda e: _resonator_box(e, "G4"), _drum, _sticks,
             lambda e: _mallet(e, "mallet", "Orff rubber-head mallet", P.MALLET, "rubber head"),
             lambda e: _mallet(e, "felt_beater", "Felt beater", P.BEATER, "felt head"))
    for i, step in enumerate(steps):
        try:
            step(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(f"step_{i}", f"{type(exc).__name__}: {exc}")
