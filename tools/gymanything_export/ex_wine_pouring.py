# SPDX-License-Identifier: Apache-2.0
"""wine_pouring: ten glass/stainless vessels built by their ``geometry.py`` (revolved wall profiles) and the bar top.

Geometry is their ``Container.render_mesh`` (the wall polygon revolved, watertight); masses are their datasheet
overrides where they have one, else their material density times the same wall volume (as their sim does).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from _common import Exporter, box, cited, comment_of, dim, line_of, mesh, vis, where

from wine_pouring import geometry as G
from wine_pouring import params as P
from wine_pouring import render as R

WALL_M = 0.003  # their comment on render.GLASS: transmittances are quoted through a 3 mm wall

# kind -> (key, name, category, cited dims [(name, value m, measure)], mass basis or None = density x volume)
VESSELS = {
    "bottle_leonor": ("bottle_bordeaux_750", "Wine bottle 75 cl (Saverglass LEONOR)", "container",
                      [("height", G.LEONOR["height"], "z"), ("shoulder diameter", G.LEONOR["shoulder_d"], "diameter_xy")],
                      "datasheet"),
    "bottle_burgundy": ("bottle_burgundy_750", "Wine bottle 75 cl (Saverglass BG CLASSIQUE 2)", "container",
                        [("height", G.BG_CLASSIQUE2["height"], "z"), ("body diameter", G.BG_CLASSIQUE2["body_d"],
                                                                      "diameter_xy")], "datasheet"),
    "glass_pure": ("glass_bordeaux", "Bordeaux wine glass (Zwiesel PURE 8545/130)", "container",
                   [("height", 0.267, "z"), ("bowl diameter", 0.094, "diameter_xy")], "datasheet"),
    "glass_iso": ("glass_iso_tasting", "ISO 3591 tasting glass", "container",
                  [("height", 0.155, "z"), ("widest diameter", 0.065, "diameter_xy")], None),
    "glass_lined": ("glass_lined_goblet", "Lined wine goblet 175 mL line", "container", [], "estimate"),
    "tumbler": ("tumbler", "Water tumbler", "container", [("height", 0.105, "z"), ("rim diameter", 0.080, "diameter_xy")],
                None),
    "carafe_1l": ("carafe_1l", "Water carafe 1 L", "container",
                  [("height", 0.235, "z"), ("body diameter", 0.100, "diameter_xy")], None),
    "carafe_500": ("carafe_500ml", "Lined wine carafe 500 mL", "container",
                   [("height", 0.190, "z"), ("body diameter", 0.080, "diameter_xy")], None),
    "dump_bucket": ("dump_bucket", "Stainless dump bucket", "container",
                    [("height", 0.150, "z"), ("rim diameter", 0.120, "diameter_xy")], "estimate"),
    "coaster": ("bottle_coaster", "Stainless bottle coaster", "fixture",
                [("dish diameter", 0.105, "diameter_xy"), ("rim height", 0.015, "z")], "estimate"),
}
GLASS = {"glass_green": "soda-lime glass", "glass_clear": "soda-lime glass", "crystal": "Tritan crystal",
         "stainless": "stainless steel"}


def _density(c: G.Container) -> dict:
    if c.density == P.SODA_LIME_DENSITY:
        return cited(c.density, "literature", line_of(P, "SODA_LIME_DENSITY ="), "soda-lime glass")
    if c.density == P.TRITAN_CRYSTAL_DENSITY:
        return cited(c.density, "estimate", line_of(P, "TRITAN_CRYSTAL_DENSITY ="),
                     comment_of(P, "TRITAN_CRYSTAL_DENSITY ="))
    return cited(c.density, "estimate", where(G.dump_bucket), "stainless 7900 kg/m^3")


def _friction(kind: str) -> tuple[dict, str]:
    if kind == "coaster":
        return cited(P.MU["bottle_coaster"], "literature", line_of(P, "MU = dict")), "a glass bottle"
    if kind == "dump_bucket":
        return cited(P.MU["default"], "estimate", line_of(P, "MU = dict")), "any surface (their default)"
    quote = "glass on glass, quartz aggregate or the stainless coaster: 0.4 (Engineering ToolBox friction table)"
    return cited(P.MU["glass_quartz"], "literature", line_of(P, "MU = dict"), quote), "the quartz bar top"


def _srgb(rgb) -> list[float]:
    """Their material tables hold linear diffuse values (S.color takes linear); parts carry sRGB."""
    c = np.clip(np.asarray(rgb, float), 0.0, 1.0)
    return [float(v) for v in np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)]


def _visual(c: G.Container, geom) -> list[dict]:
    """Their render material for the wall: OmniPBR stainless, or a dielectric whose tint is Beer-Lambert absorption."""
    if c.material == "stainless":
        rgb, rough, metal = R.COLORS["stainless"]
        return [vis(geom, _srgb(rgb), line_of(R, "stainless=("), rough, metal, name="stainless (linear->sRGB)")]
    absorb, _ior = R.GLASS[c.material]
    tint = _srgb(np.exp(-np.asarray(absorb) * WALL_M))
    return [vis(geom, tint, line_of(R, f'"{c.material}": (('), 0.0, 0.0,
                name=f"glass {c.material} (3 mm transmittance, linear->sRGB)")]


def _vessel(x: Exporter, kind: str) -> None:
    key, name, category, dims, mass_basis = VESSELS[kind]
    c = G.get(kind)
    builder = G.BUILDERS[kind]
    v, f = c.render_mesh(sectors=96)
    doc = " ".join((builder.__doc__ or "").split())
    mass = cited(c.mass, mass_basis, where(builder), doc[:300]) if mass_basis and c.mass_override else None
    mu, partner = _friction(kind)
    geom = mesh(v, f)
    x.add(key, name, category, geom, visual=_visual(c, geom),
          material={"name": GLASS.get(c.material, c.material), "density": _density(c)},
          mass=mass, friction=mu, friction_partner=partner,
          dims=[dim(n, val, m, where(builder)) for n, val, m in dims],
          source_ref=where(builder),
          notes=f"wall profile revolved (their Container.render_mesh); capacity {c.capacity * 1e6:.0f} mL. {c.notes}",
          extra={"capacity_m3": c.capacity, "rim_z_m": c.rim_z, "fill_lines_z_m": dict(c.lines)})


def _bar_image(dressing):
    """Their generated bar-top speckle (``dressing._dark_stone``), run against a scratch folder and a stand-in stage."""
    from types import SimpleNamespace  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        dressing._dark_stone(SimpleNamespace(tex=Path(tmp)), SimpleNamespace(textured=lambda *a, **k: None))
        return Image.open(Path(tmp) / "bar_quartz_diff.png").convert("RGB").copy()


def _bar_top(x: Exporter) -> None:
    from wine_pouring import dressing as D  # noqa: PLC0415 - heavy import, only the source lines are needed
    hx, hy, hz = 0.5 * P.BAR_LENGTH_M, 0.5 * P.BAR_DEPTH_M, 0.5 * P.BAR_TOP_THICK_M
    slab = box((hx, hy, hz))
    # their render textures the slab with a generated speckle (mean grey 0.30, tint 1.0/0.98/0.95), satin roughness 0.30
    grey = (0.30, 0.30 * 0.98, 0.30 * 0.95)
    top = [vis(slab, grey, line_of(D, "img = 0.30"), 0.30, 0.0, image=_bar_image(D), uv=np.asarray(slab.vertices)[:, :2] / 0.35,
               name="quartz (generated speckle texture, 0.35 m tile)")]
    x.add("bar_top_quartz", "Engineered-quartz bar top", "furniture", slab, motion="static", visual=top,
          material={"name": "engineered quartz"},
          friction=cited(P.MU["glass_quartz"], "literature", line_of(P, "MU = dict")), friction_partner="glassware",
          dims=[dim("thickness", P.BAR_TOP_THICK_M, "z", line_of(P, "BAR_TOP_THICK_M =")),
                dim("depth", P.BAR_DEPTH_M, "mid", line_of(P, "BAR_DEPTH_M ="))],
          source_ref=line_of(P, "BAR_TOP_THICK_M ="), notes="3 cm quartz slab, 26 in deep, 3 m long (authored length)")


def export(x: Exporter) -> None:
    for kind in VESSELS:
        try:
            _vessel(x, kind)
        except Exception as exc:  # noqa: BLE001
            x.fail(VESSELS[kind][0], f"{type(exc).__name__}: {exc}")
    _bar_top(x)
