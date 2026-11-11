# SPDX-License-Identifier: Apache-2.0
"""laundry: the five canonical garments (T-shirt, button-down shirt, jeans, bath towel, draped top) from their
``garment_patterns`` constructions and ``specs`` material profiles.

The mesh is the set of cut panels laid out side by side (their pattern outlines extruded to each panel's unloaded
thickness); their solver sews the panels into a shell, which is not a closed solid. All values are their
"engineering prior" numbers, exported with the ``estimate`` basis.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import trimesh
from _common import Exporter, _rel, cited, dim, line_of, vis, where
from shapely.geometry import Polygon

from laundry import garment_patterns as GP
from laundry import garment_showcase_scene as GS
from laundry import specs as SP

GAP = 0.02
# key -> (construction id, builder, display name)
GARMENTS = {
    "tshirt": ("cotton_jersey_half_sleeve_tee", GP.build_cotton_tee, "T-shirt (cotton jersey, half sleeve)"),
    "shirt": ("woven_button_down_shirt", GP.build_woven_shirt, "Button-down shirt (woven cotton)"),
    "jeans": ("denim_five_pocket_jeans", GP.build_denim_jeans, "Five-pocket denim jeans"),
    "towel": ("terry_cotton_bath_towel", GP.build_terry_towel, "Bath towel (terry cotton)"),
    "top": ("asymmetric_draped_one_sleeve_top", GP.build_asymmetric_draped_top, "Asymmetric draped top (viscose crepe)"),
}


def _spec_line(module, profile: str, field: str) -> str:
    """``laundry/specs.py:<line>`` of ``field`` inside the profile ``profile`` (first match after its name line)."""
    path = Path(inspect.getsourcefile(module))
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if f'name="{profile}"' in ln)
    for i in range(start, len(lines)):
        if lines[i].strip().startswith(field + "="):
            return f"laundry/{path.name}:{i + 1}"
    raise KeyError(f"{profile}.{field}")


def _panel_mesh(panel) -> trimesh.Trimesh:
    m = trimesh.creation.extrude_polygon(Polygon(panel.outline_polygon), panel.unloaded_thickness_m)
    m.apply_translation(-m.bounds[0] * [1, 1, 0] - [0, 0, 0])
    return m


def _layout_parts(panels) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
    """Panels in a row along x (gap 2 cm), bottom edges on y = 0, centred on the origin; the whole and each panel."""
    meshes, x = [], 0.0
    for p in panels:
        m = _panel_mesh(p)
        m.apply_translation([x, 0, 0])
        x += m.extents[0] + GAP
        meshes.append(m)
    whole = trimesh.util.concatenate(meshes)
    shift = -whole.bounds[0] * [1, 1, 0] - whole.extents * [.5, .5, 0]
    for m in (whole, *meshes):
        m.apply_translation(shift)
    return whole, meshes


def _colour_line(builder, rgb) -> str:
    """``laundry/garment_patterns.py:<line>`` of the colour tuple equal to ``rgb``: inside ``builder``'s source if it
    is there, else the first such tuple of the module (a panel helper below or above the builder)."""
    path = Path(inspect.getsourcefile(builder))
    _, first = inspect.getsourcelines(builder)
    last = first + len(inspect.getsourcelines(builder)[0])
    hits = []
    for n, text in enumerate(path.read_text().splitlines(), 1):
        for m in re.finditer(r"\(([\d.]+), ([\d.]+), ([\d.]+)\)", text):
            if all(abs(float(a) - b) < 1e-9 for a, b in zip(m.groups(), rgb)):
                hits.append(n)
    inside = [n for n in hits if first <= n < last]
    return f"{_rel(path)}:{(inside or hits or [first])[0]}"


def _dye_parts(panels, meshes, builder) -> list[dict]:
    """One part per distinct dye: the panel's ``display_color_rgb`` (their OmniPBR diffuse constant, used as sRGB)
    with the roughness their ``_TEXTILE_ROUGHNESS`` gives the panel's fabric."""
    groups: dict[tuple, list] = {}
    for p, m in zip(panels, meshes):
        groups.setdefault((tuple(p.display_color_rgb), p.material_profile), []).append(m)
    parts = []
    for (rgb, profile), ms in groups.items():
        rough = GS._TEXTILE_ROUGHNESS[profile]
        cite = f"{_colour_line(builder, rgb)}; roughness {line_of(GS, f'{chr(34)}{profile}{chr(34)}: {rough}')}"
        parts.append(vis(trimesh.util.concatenate(ms), rgb, cite, rough, 0.0,
                         name=f"{profile.replace('_prior', '')} dye (flat colour; weave texture not exported)"))
    return parts


def _cloth_block(profile_name: str, thickness: float, builder_cite: str) -> dict:
    p = SP.MATERIAL_PROFILES[profile_name]
    q = f"{profile_name}: {p.calibration_status}"

    def c(value: float, field: str) -> dict:
        return cited(value, "estimate", _spec_line(SP, profile_name, field), q)
    return {"thickness": cited(thickness, "estimate", builder_cite, "panel unloaded_thickness_m (engineering prior)"),
            "stretch_warp": c(p.warp_membrane_n_m, "warp_membrane_n_m"),
            "stretch_weft": c(p.weft_membrane_n_m, "weft_membrane_n_m"), "shear": c(p.shear_n_m, "shear_n_m"),
            "bend_warp": c(p.warp_bend_nm, "warp_bend_nm"), "bend_weft": c(p.weft_bend_nm, "weft_bend_nm")}


def _garment(x: Exporter, key: str) -> None:
    cid, builder, name = GARMENTS[key]
    con = builder()
    main = max(con.panels, key=lambda p: p.area_m2)
    prof = SP.MATERIAL_PROFILES[main.material_profile]
    mass = sum(p.mass_kg for p in con.panels)
    b = where(builder)
    block = _cloth_block(main.material_profile, main.unloaded_thickness_m, b)
    mu = cited(prof.table_kinetic_friction, "estimate", _spec_line(SP, prof.name, "table_kinetic_friction"),
               f"{prof.name} table kinetic friction (the workcell's single cloth/shape friction)")
    dims = []
    if key == "towel":
        p = con.panels[0]
        dims = [dim("width", 0.45, "x", b), dim("length", 0.70, "y", b), dim("thickness", p.unloaded_thickness_m, "z", b)]
    whole, panel_meshes = _layout_parts(con.panels)
    x.add(key, name, "cloth", whole, visual=_dye_parts(con.panels, panel_meshes, builder), kind="cloth", cloth=block, thickness=block["thickness"],
          material={"name": f"{main.material_profile.replace('_prior', '').replace('_', ' ')} (main panels)"},
          mass=cited(mass, "derived", b, f"sum over {len(con.panels)} panels of area x areal density "
                     f"({con.calibration_status})"),
          friction=mu, friction_partner="the folding table", dims=dims, source_ref=b,
          extra={"construction": cid, "panels": {p.id: {"material": p.material_profile, "areal_density_kg_m2":
                                                        p.areal_density_kg_m2, "thickness_m": p.unloaded_thickness_m,
                                                        "area_m2": round(p.area_m2, 5)} for p in con.panels},
                 "friction_cloth_cloth_kinetic": prof.cloth_kinetic_friction,
                 "friction_finger_kinetic": prof.finger_kinetic_friction,
                 "calibration_status": con.calibration_status},
          notes="Cut panels laid out side by side as extruded slabs (the sewn shell is not a closed solid); cloth "
                "constants are the dominant panel's material profile (prior-only, uncalibrated); the stiffnesses are "
                "numerical priors, not measurements.")


def export(x: Exporter) -> None:
    for key in GARMENTS:
        try:
            _garment(x, key)
        except Exception as exc:  # noqa: BLE001
            x.fail(key, f"{type(exc).__name__}: {exc}")
