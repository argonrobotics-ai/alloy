# SPDX-License-Identifier: Apache-2.0
"""forklift: the GMA 48 x 40 in pallet, the OCC bale, its dunnage strip and the loose-stack carton, from their
``loads.py`` builders and ``params.py`` ledger (the truck, warehouse and steel coil are not exported)."""

from __future__ import annotations

from _common import Exporter, box, cited, dim, line_of, union, vis, where

from forklift import loads as LD
from forklift import params as P
from forklift import render as RN

BASIS = {"read/estimate": "estimate", "estimate": "estimate", "search": "search", "authored": "authored",
         "read": "literature-read"}


def _led(name: str, value=None, quote: str = "", basis: str | None = None) -> dict:
    row = P.LEDGER[name]
    return cited(row["value"] if value is None else value, basis or BASIS[row["label"]], line_of(P, f'{name} = P("{name}"'),
                 quote or row["source"][:300])


def _srgb(lin) -> list[float]:
    """Their COLORS table is linear; the part colour is sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]


def _look(geom, key: str, name: str) -> list[dict]:
    """The ``COLORS[key]`` entry (linear colour, roughness, metallic) of their ``render.py`` as one render part."""
    rgb, rough, metal = RN.COLORS[key]
    return [vis(geom, _srgb(rgb), line_of(RN, f"{key}=(("), rough, metal, name=f"{name} (linear->sRGB)")]


def _dim(label: str, value: float, measure: str, name: str) -> dict:
    return dim(label, value, measure, line_of(P, f'{name} = P("{name}"'))


def _pallet(x: Exporter) -> None:
    g = P.GMA_PALLET
    parts = [box(h, c) for c, h, _ in LD.gma_boxes()]
    geom = union(parts)
    x.add("gma_pallet", "GMA 48 x 40 in stringer pallet", "other", geom,
          visual=_look(geom, "pallet_wood", "softwood pallet (fallback colour; wood texture not exported)"),
          material={"name": "softwood boards and stringers"}, mass=_led("GMA_PALLET", g["mass"]),
          friction=_led("MU_PALLET_FLOOR"), friction_partner="smooth concrete floor",
          dims=[_dim("length", g["L"], "x", "GMA_PALLET"), _dim("width", g["W"], "y", "GMA_PALLET"),
                _dim("height", g["H"], "z", "GMA_PALLET")], source_ref=where(LD.gma_boxes),
          notes="Three stringers, seven top and five bottom deck boards (their boxes) unioned; the 20 kg is their "
                "total, spread over the boards by volume.")


def _boxes(x: Exporter) -> None:
    bale = P.BALE
    lo, hi = bale["mass"]
    bale_box = box([s / 2 for s in bale["size"]])
    x.add("occ_bale", "OCC (cardboard) bale 60 x 30 x 48 in", "other", bale_box,
          visual=_look(bale_box, "bale", "OCC bale (flat colour; generated strata texture not exported)"),
          material={"name": "baled old corrugated cardboard"},
          mass={**_led("BALE", (lo + hi) / 2, "OCC bale 60x30x48 in, 1,000-1,300 lb (search); mid of 450-590 kg"),
                "unc": [lo, hi]},
          friction=_led("BALE", bale["mu"], "bale friction 0.40, estimate", "estimate"), friction_partner="dunnage",
          dims=[_dim("length", bale["size"][0], "x", "BALE"), _dim("height", bale["size"][2], "z", "BALE")],
          source_ref=line_of(P, 'BALE = P("BALE"'),
          notes="Solid box; their sim draws the mass within the 450-590 kg range, the midpoint is exported.")
    d = P.DUNNAGE
    strip = box([s / 2 for s in d["size"]])
    x.add("dunnage_strip", "Dunnage strip 4 x 4 in", "other", strip,
          visual=_look(strip, "dunnage", "timber dunnage"),
          material={"name": "timber"}, mass=_led("DUNNAGE", d["mass"]), friction=_led("MU_CARTON"),
          friction_partner="bale", dims=[_dim("section", d["size"][0], "x", "DUNNAGE")],
          source_ref=line_of(P, 'DUNNAGE = P("DUNNAGE"'), notes="4 x 4 in strip 48 in long (their size tuple).")
    lo, hi = P.CARTON_MASS
    cbox = box([s / 2 for s in P.CARTON_SIZE])
    x.add("carton", "Loose-stack carton 0.40 x 0.50 x 0.25 m", "container", cbox,
          visual=_look(cbox, "carton", "corrugated carton (flat colour; print decal not exported)"),
          material={"name": "corrugated cardboard, filled"},
          mass={**_led("CARTON_MASS", (lo + hi) / 2, "carton mass drawn uniformly in 12-20 kg; midpoint"), "unc": [lo, hi]},
          friction=_led("MU_CARTON"), friction_partner="pallet and cartons",
          dims=[_dim("length", P.CARTON_SIZE[1], "y", "CARTON_SIZE"), _dim("height", P.CARTON_SIZE[2], "z", "CARTON_SIZE")],
          source_ref=where(LD.add_carton_stack), notes="Filled solid box, as their sim treats it.")


def export(x: Exporter) -> None:
    for fn in (_pallet, _boxes):
        try:
            fn(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
    x.fail("steel_coil_unit", "coil load is built inline in add_pallet as a composite with its pallet; not separate")
