# SPDX-License-Identifier: Apache-2.0
"""wiping: the microfibre cloth (thin-sheet constants), the jug of the blender and the formed GN prep trays.

Geometry: the jug and its lid are their ``kitchen.jug_geometry(hollow=True)`` and lid boxes; the trays are their
``formed_pan.geometry`` (one watertight shell shared by collision and render); the cloth is a flat sheet of the
cloth side and contact thickness (their cloth is a folded Style3D mesh, so the rest sheet is the equivalent).
"""

from __future__ import annotations

import numpy as np
import trimesh
from _common import Exporter, box, cited, comment_of, dim, line_of, mesh, union, vis, where

from wiping import assembly as A
from wiping import config as CFG
from wiping import formed_pan as FP
from wiping import kitchen as K
from wiping import render as R

C = CFG.Config()


def _srgb(rgb) -> list[float]:
    """Their material tables hold linear diffuse values (S.color takes linear); parts carry sRGB."""
    c = np.clip(np.asarray(rgb, float), 0.0, 1.0)
    return [float(v) for v in np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)]


def _cloth(x: Exporter) -> None:
    side, thick = C.cloth_side, 2 * C.cloth_radius
    add = where(A.Assembly.add_cloth)
    aniso = line_of(A, "tri_aniso_ke")
    edge = line_of(A, "edge_aniso_ke")
    q_k = "style3d tri_aniso_ke=(120, 120, 25) N/m: warp, weft, shear; the file calls its defaults qualification candidates"
    q_b = "style3d edge_aniso_ke=(2e-5, 2e-5, 1e-5) bending, as passed to add_cloth_mesh"
    cloth = {"thickness": cited(thick, "estimate", line_of(CFG, "cloth_radius:"), "contact (pile) thickness = 2 x cloth_radius"),
             "stretch_warp": cited(120.0, "estimate", aniso, q_k), "stretch_weft": cited(120.0, "estimate", aniso, q_k),
             "shear": cited(25.0, "estimate", aniso, q_k),
             "bend_warp": cited(2e-5, "estimate", edge, q_b), "bend_weft": cited(2e-5, "estimate", edge, q_b)}
    sheet = box((side / 2, side / 2, thick / 2), (0, 0, thick / 2))
    dye, rough, metal = R.COLORS["cloth_0"]
    look = [vis(sheet, _srgb(dye), line_of(R, "'cloth_0':"), rough, metal, name="dry microfibre dye (cloth_0, linear->sRGB)")]
    x.add("towel", "Microfibre wiping cloth 40 x 40 cm", "cloth", sheet, visual=look,
          kind="cloth", material={"name": "terry microfibre"}, thickness=cloth["thickness"], cloth=cloth,
          mass=cited(C.cloth_dry_mass, "estimate", line_of(CFG, "cloth_dry_mass:"), "dry mass of one cloth"),
          friction=cited(C.contact_mu, "estimate", line_of(CFG, "contact_mu:"), "cloth-shape friction (geometric mean with the shape's)"),
          friction_partner="table top and tools",
          dims=[dim("side", side, "x", line_of(CFG, "cloth_side:")), dim("side", side, "y", line_of(CFG, "cloth_side:"))],
          source_ref=add, notes="Flat rest sheet equivalent to their folded Style3D cloth (assembly.folded_grid); mass is the dry "
          "mass, water is a separate state. Bending is flat-rest; the VBD variant uses 2e-5 too.")


def _jug(x: Exporter) -> None:
    v, f = K.jug_geometry(hollow=True)
    f = np.asarray(f).reshape(-1, 3)
    shell = mesh(v, f)
    lid = box((0.086, 0.084, 0.020), (0, 0, 0.464))
    plug = box((0.025, 0.025, 0.012), (0, 0, 0.496))
    geom = union([shell, lid, plug])
    cite = where(K.jug_geometry)
    # render.py sets the jug's frosted plastic after the COLORS table (diffuse, roughness; opacity 0.22), lid and plug 'charcoal'
    cover = trimesh.util.concatenate([lid, plug])
    pc = R.COLORS["charcoal"]
    look = [vis(shell, _srgb((.78, .85, .86)), line_of(R, "diffuseColor').Set(Gf.Vec3f(.78"), .12, 0.0, name="frosted jug plastic (linear->sRGB)"),
            vis(cover, _srgb(pc[0]), line_of(R, "'charcoal':"), pc[1], pc[2], name="lid and plug (charcoal, linear->sRGB)")]
    x.add("jug", "Blender jug with lid (frosted, empty)", "container", geom, visual=look,
          material={"name": "frosted jug plastic", "density": cited(350.0, "estimate", line_of(K, "density=350."),
                                                                      "blender_jug shape density 350 kg/m^3")},
          friction=cited(0.5, "estimate", line_of(K, "density=350."), "blender_jug mu 0.5"), friction_partner="table top",
          dims=[dim("height with lid", 0.508 - 0.228, "z", line_of(K, "height=.508"))], source_ref=cite,
          notes="12-sided tapered shell with 3 mm walls, the lid and plug; the jug has no mass of its own in their code (only the "
                "5.53 kg blender), so mass is left to the density. Frame: the blender's body origin.")


def _tray(x: Exporter, key: str, name: str, size: tuple, mass: float, pattern: str, paint: str) -> None:
    v, f, _ = FP.geometry(size)
    cite = line_of(K, pattern)
    geom = mesh(v, np.asarray(f).reshape(-1, 3))
    rgb, rough, metal = R.COLORS[paint]
    look = [vis(geom, _srgb(rgb), line_of(R, f"'{paint}':"), rough, metal, name=f"{paint} (linear->sRGB)")]
    x.add(key, name, "container", geom, visual=look, material={"name": "formed food pan, 3 mm wall"},
          mass=cited(mass, "estimate", cite, "mass given to pan() for this tray"),
          friction=cited(0.45, "estimate", line_of(A, "mu=.45"), "shape_config default mu"), friction_partner="table top",
          dims=[dim("length", size[0], "x", cite), dim("width", size[1], "y", cite), dim("height", size[2], "z", cite)],
          source_ref=where(FP.geometry), notes=" ".join(FP.__doc__.split())[:250])


def export(x: Exporter) -> None:
    _cloth(x)
    _jug(x)
    _tray(x, "prep_tray", "Formed GN prep tray 265 x 325 x 65 mm", (.265, .325, .065), .42, "pan(a,'prep_tray'", "pan_white")
    _tray(x, "clean_staging_tray", "Formed GN staging tray 176 x 325 x 65 mm", (.176, .325, .065), .30, "pan(a,'clean_staging_tray'", "pan_green")
