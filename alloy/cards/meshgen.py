# SPDX-License-Identifier: Apache-2.0
"""Procedural, watertight, outward-wound meshes for the starter library (``library/meshes/*.obj``).

Shapes are built from four constructions: ``lathe`` (a profile from the axis back to the axis revolved about Z),
``extrude`` (a simple polygon, ear-clipped), ``extrude_ring`` (an outer polygon with one hole, both star-shaped
about the origin) and ``box_pocket`` (a box with a rectangular pocket open on +Z). Multi-part objects are several
closed shells in one file, one OBJ group each (mass adds per group; ``part_materials`` keys are group names).
Run ``python -m alloy.cards.meshgen library/meshes`` to regenerate; output is byte-for-byte deterministic.
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from alloy.cards.geometry import MeshData, write_obj

log = logging.getLogger("alloy.cards")

Poly = list[tuple[float, float]]


def _mesh(verts: list[Sequence[float]], faces: list[Sequence[int]]) -> MeshData:
    return MeshData(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64))


def merge(parts: dict[str, MeshData]) -> MeshData:
    """Concatenate closed shells into one mesh with one group per shell."""
    verts, faces, groups, nv, nf = [], [], {}, 0, 0
    for name, m in parts.items():
        verts.append(m.vertices)
        faces.append(m.faces + nv)
        groups[name] = np.arange(nf, nf + len(m.faces), dtype=np.int64)
        nv += len(m.vertices)
        nf += len(m.faces)
    return MeshData(np.concatenate(verts), np.concatenate(faces), groups)


def translate(m: MeshData, d: Sequence[float]) -> MeshData:
    return MeshData(m.vertices + np.asarray(d, dtype=np.float64), m.faces, m.groups)


def lathe(profile: Poly, n: int = 64) -> MeshData:
    """Revolve an (r, z) polyline that starts and ends on the axis (r = 0) about Z; walk it bottom-outward so the
    solid lies to the right of the walking direction in the (r, z) plane."""
    if profile[0][0] != 0.0 or profile[-1][0] != 0.0:
        raise ValueError("lathe profile must start and end on the axis")
    ang = [2 * math.pi * k / n for k in range(n)]
    verts: list[Sequence[float]] = [(0.0, 0.0, profile[0][1])]
    rings: list[int] = []
    for r, z in profile[1:-1]:
        rings.append(len(verts))
        verts += [(r * math.cos(a), r * math.sin(a), z) for a in ang]
    top = len(verts)
    verts.append((0.0, 0.0, profile[-1][1]))
    faces: list[Sequence[int]] = [(0, rings[0] + (k + 1) % n, rings[0] + k) for k in range(n)]
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(n):
            k1 = (k + 1) % n
            faces += [(a + k, a + k1, b + k1), (a + k, b + k1, b + k)]
    faces += [(top, rings[-1] + k, rings[-1] + (k + 1) % n) for k in range(n)]
    return oriented(_mesh(verts, faces))


def _area(poly: Poly) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]))


def _inside(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    def side(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    return side(a, b, p) >= 0 and side(b, c, p) >= 0 and side(c, a, p) >= 0


def ear_clip(poly: Poly) -> list[tuple[int, int, int]]:
    """Triangulate a simple counter-clockwise polygon (O(n^2) ear clipping)."""
    idx = list(range(len(poly)))
    tris: list[tuple[int, int, int]] = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue
            if any(_inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            raise ValueError("ear_clip: polygon is not simple")
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def extrude(poly: Poly, z0: float, z1: float) -> MeshData:
    """Prism of a simple polygon between z0 < z1."""
    if _area(poly) < 0:
        poly = poly[::-1]
    n = len(poly)
    verts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    tris = ear_clip(poly)
    faces = [(a, c, b) for a, b, c in tris] + [(a + n, b + n, c + n) for a, b, c in tris]
    for k in range(n):
        k1 = (k + 1) % n
        faces += [(k, k1, k1 + n), (k, k1 + n, k + n)]
    return oriented(_mesh(verts, faces))


def ray_angles(n: int, extra: Sequence[float], tol: float = 1e-9) -> list[float]:
    """n evenly spaced angles in [0, 2pi) plus ``extra`` ones, with near-duplicates (and 2pi ~ 0) merged, so no
    zero-length edges appear (they would break the watertight check after vertex welding)."""
    two_pi = 2 * math.pi
    out: list[float] = []
    for a in sorted([two_pi * k / n for k in range(n)] + [e % two_pi for e in extra]):
        if (not out or a - out[-1] > tol) and two_pi - a > tol:
            out.append(a)
    return out


def _resample_star(poly: Poly, angles: Sequence[float]) -> Poly:
    """Points where rays from the origin at ``angles`` leave a star-shaped polygon."""
    out = []
    for a in angles:
        d = (math.cos(a), math.sin(a))
        best = math.inf
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
            ex, ey = x1 - x0, y1 - y0
            den = d[0] * ey - d[1] * ex
            if abs(den) < 1e-15:
                continue
            t = (x0 * ey - y0 * ex) / den
            s = (x0 * d[1] - y0 * d[0]) / den
            if t > 0 and -1e-12 <= s <= 1 + 1e-12:
                best = min(best, t)
        out.append((best * d[0], best * d[1]))
    return out


def extrude_ring(outer: Poly, inner: Poly, z0: float, z1: float, n: int = 96) -> MeshData:
    """Prism of ``outer`` minus ``inner`` (both star-shaped about the origin, inner inside outer); both loops are
    resampled at the same n ray angles plus their own corners, so the faces between them are quads."""
    corners = [math.atan2(y, x) % (2 * math.pi) for x, y in outer + inner]
    angles = ray_angles(n, corners)
    o, i = _resample_star(outer, angles), _resample_star(inner, angles)
    m = len(angles)
    verts = [(x, y, z) for z in (z0, z1) for x, y in o] + [(x, y, z) for z in (z0, z1) for x, y in i]
    ob, ot, ib, it = 0, m, 2 * m, 3 * m
    faces: list[Sequence[int]] = []
    for k in range(m):
        k1 = (k + 1) % m
        faces += [(ob + k, ob + k1, ot + k1), (ob + k, ot + k1, ot + k)]          # outer wall
        faces += [(ib + k, it + k1, ib + k1), (ib + k, it + k, it + k1)]          # inner wall (faces inward)
        faces += [(ot + k, ot + k1, it + k1), (ot + k, it + k1, it + k)]          # top annulus
        faces += [(ob + k, ib + k1, ob + k1), (ob + k, ib + k, ib + k1)]          # bottom annulus
    return oriented(_mesh(verts, faces))


def box_pocket(outer: tuple[float, float, float], pocket: tuple[float, float, float]) -> MeshData:
    """One closed shell: a box of full size ``outer`` on z ∈ [0, outer_z] with a centred pocket of full size
    ``pocket`` open on +Z (its floor at outer_z − pocket_z)."""
    ox, oy, oz = outer[0] / 2, outer[1] / 2, outer[2]
    px, py, zf = pocket[0] / 2, pocket[1] / 2, outer[2] - pocket[2]
    if not (0 < px < ox and 0 < py < oy and 0 < zf < oz):
        raise ValueError("pocket must fit inside the box with a floor")
    o = [(-ox, -oy), (ox, -oy), (ox, oy), (-ox, oy)]
    i = [(-px, -py), (px, -py), (px, py), (-px, py)]
    verts = [(x, y, 0.0) for x, y in o] + [(x, y, oz) for x, y in o] + [(x, y, oz) for x, y in i] + \
            [(x, y, zf) for x, y in i]
    ob, ot, it, fl = 0, 4, 8, 12
    faces: list[Sequence[int]] = [(ob + 0, ob + 2, ob + 1), (ob + 0, ob + 3, ob + 2), (fl + 0, fl + 1, fl + 2),
                                  (fl + 0, fl + 2, fl + 3)]
    for k in range(4):
        k1 = (k + 1) % 4
        faces += [(ob + k, ob + k1, ot + k1), (ob + k, ot + k1, ot + k)]            # outer walls
        faces += [(ot + k, ot + k1, it + k1), (ot + k, it + k1, it + k)]            # rim
        faces += [(it + k, it + k1, fl + k1), (it + k, fl + k1, fl + k)]            # pocket walls
    return oriented(_mesh(verts, faces))


def hex_bolt(s: float, k: float, d: float, length: float, n: int = 64) -> MeshData:
    """One closed shell: a hex head (across flats ``s``, height ``k``) on z ∈ [0, k] and a round shank of
    diameter ``d`` and ``length`` on z ∈ [−length, 0]."""
    angles = ray_angles(n, [math.pi / 3 * j for j in range(6)])
    hexl = _resample_star(hex_poly(s), angles)
    circ = [(d / 2 * math.cos(a), d / 2 * math.sin(a)) for a in angles]
    m = len(angles)
    verts = [(x, y, k) for x, y in hexl] + [(x, y, 0.0) for x, y in hexl] + [(x, y, 0.0) for x, y in circ] + \
            [(x, y, -length) for x, y in circ] + [(0.0, 0.0, k), (0.0, 0.0, -length)]
    ht, hb, cs, ce, top, bot = 0, m, 2 * m, 3 * m, 4 * m, 4 * m + 1
    faces: list[Sequence[int]] = []
    for j in range(m):
        j1 = (j + 1) % m
        faces += [(top, ht + j, ht + j1), (hb + j, hb + j1, ht + j1), (hb + j, ht + j1, ht + j)]
        faces += [(hb + j, cs + j1, hb + j1), (hb + j, cs + j, cs + j1)]            # head underside annulus
        faces += [(cs + j, ce + j1, cs + j1), (cs + j, ce + j, ce + j1), (bot, ce + j1, ce + j)]
    return oriented(_mesh(verts, faces))


def oriented(m: MeshData) -> MeshData:
    """Flip the winding if the closed mesh has negative signed volume (outward normals afterwards)."""
    from alloy.cards.geometry import signed_volume  # noqa: PLC0415

    if signed_volume(m) < 0:
        return MeshData(m.vertices, m.faces[:, ::-1].copy(), m.groups)
    return m


def hex_poly(s: float) -> Poly:
    """Regular hexagon with across-flats ``s`` (flats normal to ±Y)."""
    rc = s / math.sqrt(3.0)
    return [(rc * math.cos(math.pi / 3 * k), rc * math.sin(math.pi / 3 * k)) for k in range(6)]


def circle(r: float, n: int = 64) -> Poly:
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def rect(w: float, h: float) -> Poly:
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


def _usb_a_plug() -> MeshData:
    """USB-A plug: steel shell 12.0 x 4.5 x 11.75 mm, sheet 0.315 mm, open toward +Z; plastic insert leaving a
    1.95 mm tongue cavity (10 um clear of the shell); overmold 15.3 x 7.3 x 17 mm below."""
    t = 0.315e-3
    shell = box_pocket((12.0e-3, 4.5e-3, 11.75e-3), (12.0e-3 - 2 * t, 4.5e-3 - 2 * t, 11.75e-3 - t))
    gap = 10e-6                          # keeps the insert's vertices off the shell's (separate closed shells)
    ins_h = (4.5e-3 - 2 * t) - 1.95e-3 - gap
    insert = translate(extrude(rect(12.0e-3 - 2 * t - 2 * gap, ins_h), t + gap, 11.75e-3),
                       (0.0, -(4.5e-3 - 2 * t) / 2 + gap + ins_h / 2, 0.0))
    overmold = extrude(rect(15.3e-3, 7.3e-3), -17.0e-3, 0.0)
    return merge({"shell": shell, "insert": insert, "overmold": overmold})


def _usb_a_receptacle() -> MeshData:
    """One USB-A port: opening 12.5 x 5.12 mm, depth 8.88 mm, tongue 11.1 x 1.84 mm on the upper half; ABS housing
    20 x 12 x 12 mm (housing size is ours, a single port)."""
    housing = box_pocket((20.0e-3, 12.0e-3, 12.0e-3), (12.5e-3, 5.12e-3, 8.88e-3))
    floor = 12.0e-3 - 8.88e-3
    tongue = translate(extrude(rect(11.1e-3, 1.84e-3), floor, 12.0e-3 - 0.5e-3), (0.0, 5.12e-3 / 2 - 1.84e-3 / 2 - 0.9e-3, 0.0))
    return merge({"housing": housing, "tongue": tongue})


def _cup(r_bot: float, r_top: float, h: float, wall: float, floor: float, n: int = 96) -> MeshData:
    """Open cup of revolution (outer radii at bottom and top), wall measured horizontally."""
    return lathe([(0.0, 0.0), (r_bot, 0.0), (r_top, h), (r_top - wall, h), (r_bot - wall + (r_top - r_bot) * floor / h, floor),
                  (0.0, floor)], n)


def _hole_circle(r: float, n: int = 96) -> Poly:
    """A hole polygon whose flats sit on the nominal radius (vertices at r / cos(pi/n)), so the meshed hole is
    never tighter than nominal ([internal reference])."""
    return circle(r / math.cos(math.pi / n), n)


def _spoon_outline() -> Poly:
    """A flat 190 mm table spoon outline: 60 x 40 mm elliptical bowl and a 10 mm handle."""
    bowl = [(0.03 * math.cos(a), 0.02 * math.sin(a)) for a in np.linspace(0.25, 2 * math.pi - 0.25, 40)]
    x0 = 0.03 * math.cos(0.25)
    return bowl + [(x0, -0.005), (0.16, -0.005), (0.16, 0.005), (x0, 0.005)]


def _screwdriver() -> MeshData:
    shaft = lathe([(0.0, 0.0), (3e-3, 0.0), (3e-3, 100e-3), (0.0, 100e-3)], 48)
    grip = lathe([(0.0, 100e-3), (10e-3, 100e-3), (14e-3, 115e-3), (14e-3, 195e-3), (11e-3, 204e-3), (0.0, 204e-3)], 64)
    return merge({"shaft": shaft, "grip": grip})


LIBRARY_MESHES: dict[str, tuple[Callable[[], MeshData], str]] = {
    "nut_m10_din934": (lambda: extrude_ring(hex_poly(17e-3), circle(9.026e-3 / 2, 96), -4e-3, 4e-3),
                       "DIN 934 M10: s 17 mm, m 8 mm; bore at the basic pitch diameter D2 9.026 mm (ISO 724), "
                       "mass-equivalent for the thread; the thread itself is the iso_thread analytic feature"),
    "nut_m6_din934": (lambda: extrude_ring(hex_poly(10e-3), circle(5.350e-3 / 2, 96), -2.5e-3, 2.5e-3),
                      "DIN 934 M6: s 10 mm, m 5 mm; bore at D2 5.350 mm (ISO 724)"),
    "washer_m10_din125a": (lambda: extrude_ring(circle(10e-3, 96), _hole_circle(5.25e-3), -1e-3, 1e-3),
                           "DIN 125-A M10: d1 10.5 mm, d2 20 mm, h 2 mm"),
    "bolt_m10x60_din933": (lambda: hex_bolt(17e-3, 6.4e-3, 8.928e-3, 60e-3, 96),
                           "DIN 933 M10x60: s 17 mm, k 6.4 mm, full thread as a d2 cylinder at the 6g mid-tolerance "
                           "8.928 mm (ISO 965-1: d2 9.026, es -32 um, Td2 132 um)"),
    "bolt_m10x40_din933": (lambda: hex_bolt(16.865e-3, 6.4e-3, 8.928e-3, 40e-3, 96),
                           "M10x40: head AF 16.865 mm, k 6.4 mm, "
                           "d2 cylinder 8.928 mm"),
    "bolt_m6x30_din933": (lambda: hex_bolt(10e-3, 4e-3, 5.268e-3, 30e-3, 96),
                          "DIN 933 M6x30: s 10 mm, k 4 mm, d2 cylinder at the 6g mid-tolerance 5.268 mm "
                          "(ISO 965-1: d2 5.350, es -26 um, Td2 112 um)"),
    "block_bore_d8_1": (lambda: extrude_ring(rect(40e-3, 40e-3), _hole_circle(8.1e-3 / 2), -4.5e-3, 4.5e-3),
                        "40 x 40 x 9 mm acrylic block, 8.1 mm through-bore"),
    "block_bore_d12_2": (lambda: extrude_ring(rect(40e-3, 40e-3), _hole_circle(12.2e-3 / 2), -4.5e-3, 4.5e-3),
                         "40 x 40 x 9 mm acrylic block, 12.2 mm through-bore"),
    "block_bore_d16_5": (lambda: extrude_ring(rect(40e-3, 40e-3), _hole_circle(16.5e-3 / 2), -4.5e-3, 4.5e-3),
                         "40 x 40 x 9 mm acrylic block, 16.5 mm through-bore"),
    "block_rect_hole_12x8": (lambda: extrude_ring(rect(40e-3, 40e-3), rect(12.2e-3, 8.2e-3), -4.5e-3, 4.5e-3, n=4),
                             "40 x 40 x 9 mm acrylic block, 12.2 x 8.2 mm through-hole"),
    "usb_a_plug": (_usb_a_plug, "USB-A plug: shell, insert, overmold groups"),
    "usb_a_receptacle": (_usb_a_receptacle, "single USB-A port: housing, tongue groups"),
    "cup_pp_disposable": (lambda: _cup(26e-3, 36.5e-3, 80e-3, 0.35e-3, 0.5e-3),
                          "7 oz disposable PP cup: 52 mm base, 73 mm rim, 80 mm tall, 0.35 mm wall, 0.5 mm floor"),
    "tumbler_pp": (lambda: _cup(40e-3, 40e-3, 91e-3, 1.2e-3, 1.5e-3),
                   "PP tumbler 80 x 91 mm, floor 1.5 mm, wall 1.2 mm"),
    "mug_stoneware_body": (lambda: _cup(44e-3, 44e-3, 96e-3, 5e-3, 6e-3),
                           "12 oz stoneware mug body without handle: r 44 mm, h 96 mm, wall 5 mm, floor 6 mm"),
    "tray_aluminium": (lambda: box_pocket((0.30, 0.22, 0.02), (0.296, 0.216, 0.018)),
                       "300 x 220 x 20 mm sheet tray, 2 mm walls and floor"),
    "bin_pp_small": (lambda: box_pocket((0.124, 0.090, 0.0762), (0.121, 0.087, 0.0747)),
                     "124 x 90 x 76.2 mm PP bin (Uline S-12413 footprint), 1.5 mm walls"),
    "enclosure_abs_base": (lambda: box_pocket((0.100, 0.070, 0.030), (0.096, 0.066, 0.028)),
                           "ABS enclosure base 100 x 70 x 30 mm, 2 mm wall"),
    "spoon_steel_flat": (lambda: extrude(_spoon_outline(), -0.75e-3, 0.75e-3),
                         "flat 190 mm table spoon, 1.5 mm stainless sheet"),
    "jar_glass_8oz": (lambda: lathe([(0.0, 0.0), (34.5e-3, 0.0), (34.5e-3, 80e-3), (33.5e-3, 95e-3), (30.5e-3, 95e-3),
                                     (31.5e-3, 80e-3), (31.5e-3, 5e-3), (0.0, 5e-3)], 96),
                      "8 oz jar body, 69 mm OD (70-400 finish T 69.05 mm), "
                      "95 mm tall, 3 mm wall, 5 mm base (wall and base are ours, not catalogue); finish thread not modelled"),
    "screwdriver_pz2_100": (_screwdriver, "PZ2 x 100 screwdriver: 6 mm shaft 100 mm, 28 mm x 104 mm grip of "
                            "revolution (grip shape is ours)"),
    "pet_bottle_500ml": (lambda: lathe([(0.0, 0.0), (32e-3, 0.0), (32e-3, 180e-3), (12.5e-3, 210e-3), (12.5e-3, 230e-3),
                                        (12.2e-3, 230e-3), (12.2e-3, 209.9e-3), (31.7e-3, 179.8e-3), (31.7e-3, 0.3e-3),
                                        (0.0, 0.3e-3)], 96),
                         "500 ml PET water bottle without cap: 64 mm body, 25 mm neck, 230 mm tall, "
                         "0.3 mm wall (ours)"),
}


def write_library_meshes(outdir: Path) -> dict[str, str]:
    """Write every library mesh as ``<name>.obj``; returns name -> sha256."""
    from alloy.cards.geometry import sha256_file  # noqa: PLC0415

    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (build, desc) in LIBRARY_MESHES.items():
        path = outdir / f"{name}.obj"
        write_obj(path, build(), header=f"alloy.cards.meshgen {name}: {desc}\nunits: m")
        out[name] = sha256_file(path)
    return out


def main(argv: Sequence[str]) -> int:
    logging.basicConfig(level=logging.INFO)
    outdir = Path(argv[0]) if argv else Path("library/meshes")
    for name, digest in write_library_meshes(outdir).items():
        log.info("%s %s", digest, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
