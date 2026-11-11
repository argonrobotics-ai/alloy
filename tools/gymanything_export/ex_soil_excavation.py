# SPDX-License-Identifier: Apache-2.0
"""soil_excavation: the digging tools, drill and auger, tamper, post, root ball and pail of their ``tools.py`` / ``scene.py``.

Geometry is the union of their rigid ``Part`` primitives (boxes, cylinders, capsules) in each tool's body frame; the
render/MPM-only meshes are not used, except the auger's helical flight (their ``auger_flight_mesh``), which is the
auger's real shape. Masses are their ``params.py`` catalogue masses (the parts' own masses are shares of it).
"""

from __future__ import annotations

import numpy as np

from _common import Exporter, box, cited, cylinder, dim, line_of, mesh, union, vis, where

from soil_excavation import params as P
from soil_excavation import render as RN
from soil_excavation import scene as SC
from soil_excavation import tools as T
from ex_sweeping import parts_mesh

SOLID = ("box", "cylinder", "capsule")
MU_PARTNER = "soil and yard surfaces"


def _solid(parts) -> list:
    return [p for p in parts if p.kind in SOLID and p.collide and p.mass >= 0]


def _srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def look(name: str, geom) -> dict:
    """One render part in the colour, roughness and metallic of their ``render.COLORS`` entry ``name`` (linear -> sRGB)."""
    rgb, rough, metal = RN.COLORS[name]
    return vis(geom, [_srgb(c) for c in rgb], line_of(RN, f" {name}=(("), roughness=float(rough), metallic=float(metal),
               name=f"{name} (linear->sRGB)")


def parts_visual(parts) -> list[dict] | None:
    """One render part per material among the solid parts their renderer draws (not ``mass_only`` / ``no_render``)."""
    drawn = [p for p in _solid(parts) if p.material != "mass_only" and "no_render" not in p.role and p.name != "flight_envelope"]
    names = list(dict.fromkeys(p.material for p in drawn if p.material in RN.COLORS))
    keep = lambda m: {p.name for p in drawn if p.material != m}  # noqa: E731
    return [look(m, parts_mesh(drawn, exclude=keep(m))) for m in names] or None


def _mass(ledger: str, value: float, derived: bool = False) -> dict:
    row = next(r for r in P.LEDGER if r["name"] == ledger)
    basis = "datasheet" if str(row["label"]).startswith("datasheet") else "estimate" if "estimate" in row["label"] else "literature"
    if derived:
        basis = "derived"
    return cited(value, basis, line_of(P, f'{ledger} = _p('), " ".join(str(row["basis"]).split())[:400])


def _mu(kind: str) -> dict:
    return cited(SC.MU_TOOL[kind], "estimate", line_of(SC, "MU_TOOL = dict"),
                 "Coulomb friction their scene gives the tool's shapes (MU_TOOL)")


def _ledger_dims(ledger: str, rows: list[tuple[str, float, str]]) -> list[dict]:
    cite = line_of(P, f"{ledger} = _p(")
    return [dim(n, v, m, cite) for n, v, m in rows]


# key -> (builder, ledger, name, category, mass fn, dims, friction kind, material)
def _simple(x: Exporter, key: str, name: str, parts, ledger: str, mass: float, dims: list, kind: str, material: str,
            builder, category: str = "tool", derived: bool = False, **extra) -> None:
    x.add(key, name, category, parts_mesh(_solid(parts)), material={"name": material},
          visual=parts_visual(parts), mass=_mass(ledger, mass, derived),
          friction=_mu(kind), friction_partner=MU_PARTNER, dims=_ledger_dims(ledger, dims), source_ref=where(builder),
          notes="Union of their collision primitives in the tool body frame; render-only and soil-collider-only meshes "
                "are not included.", extra=extra)


def _hand_tools(x: Exporter) -> None:
    S, V, R, M = P.SPADE, P.TRENCH_SHOVEL, P.TROWEL, P.MATTOCK
    _simple(x, "spade", "Digging spade (Spear & Jackson Traditional Stainless)", T.spade(), "SPADE", S["mass"],
            [("overall length", S["length"], "z")], "spade", "stainless-steel blade, ash shaft", T.spade)
    _simple(x, "trench_shovel", "Trench shovel (Ames Razor-Back 47171)", T.trench_shovel(), "TRENCH_SHOVEL", V["mass"],
            [("handle length", V["length"], "max")], "trench_shovel", "red-painted steel blade, ash shaft", T.trench_shovel)
    _simple(x, "trowel", "Hand trowel (Fiskars Ergo)", T.trowel(), "TROWEL", R["mass"],
            [("length", R["length"], "z")], "trowel", "cast aluminium blade, rubber grip", T.trowel)
    _simple(x, "mattock", "Garden mattock (INTERTOOL)", T.mattock(), "MATTOCK", M["mass"],
            [("handle length", M["handle"], "z")], "mattock", "forged steel head, fibreglass handle", T.mattock)


def _digger(x: Exporter) -> None:
    D = P.DIGGER
    for side, other in (("a", "b"), ("b", "a")):
        pin = D["pin_above_tip"]
        joint = {"type": "revolute", "parent": f"post_hole_digger__{other}", "axis": [0.0, 1.0, 0.0], "range": None,
                 "cite": where(T.digger_half)}
        _simple(x, f"post_hole_digger__{side}", f"Clamshell post-hole digger, half {side}", T.digger_half(side), "DIGGER",
                D["mass"] / 2, [], "digger_" + side, "red-painted steel blades, ash handle", T.digger_half,
                category="articulated", derived=True, joint=joint, pin_above_tip_m=pin)


def _drill(x: Exporter) -> None:
    S = P.DRILL
    _simple(x, "cordless_drill__body", "Cordless drill (DeWalt DCD996 class), body", T.drill_body(), "DRILL",
            S["mass"] + S["side_handle"], [], "drill", "plastic housing, battery, rubber grips", T.drill_body,
            category="articulated", derived=True, joint=None)
    for key, name, parts, mass in (("chuck", "chuck", T.drill_chuck(), 0.35), ("trigger", "trigger", T.drill_trigger(), 0.01)):
        _simple(x, f"cordless_drill__{key}", f"Cordless drill, {name}", parts, "DRILL", mass, [], "drill" if key == "chuck"
                else "trigger", "steel" if key == "chuck" else "plastic", T.drill_chuck, category="articulated", derived=True,
                joint={"type": "revolute" if key == "chuck" else "prismatic", "parent": "cordless_drill__body",
                       "axis": [1.0, 0.0, 0.0] if key == "chuck" else [0.0, 0.0, 1.0], "range": None, "cite": where(T.drill_body)})


def _auger(x: Exporter) -> None:
    A = P.AUGER
    verts, faces = T.auger_flight_mesh()
    flight = mesh(verts.astype(float), faces)
    shaft = cylinder(A["shaft_r"], A["length"] / 2, (0.0, 0.0, -A["length"] / 2))
    x.add("auger", "Auger bit 3 x 12 in (Power Planter 312)", "tool", union([shaft, flight]),
          visual=[look("steel_black", union([shaft, flight]))],
          material={"name": "steel shaft with 10-gauge helical flight"}, mass=_mass("AUGER", A["mass"]),
          friction=_mu("auger"), friction_partner=MU_PARTNER,
          dims=_ledger_dims("AUGER", [("flight diameter", A["diameter"], "diameter_xy"), ("length", A["length"], "z")]),
          source_ref=where(T.auger_flight_mesh),
          notes="Shaft cylinder unioned with their helical flight sheet (the soil collider's shape); the hex end and tip "
                "capsule are omitted.", extra={})


def _timber(x: Exporter) -> None:
    Tm, Po, Rb = P.TAMPER, P.POST, P.ROOTBALL
    _simple(x, "tamper", "Tamper (2x4 SPF, 1.2 m)", T.tamper(), "TAMPER", T.tamper()[0].mass,
            [("length", Tm["length"], "z")], "tamper", "SPF softwood", T.tamper, derived=True)
    _simple(x, "cedar_post", "4x4 cedar post, 6 ft", T.post(), "POST", T.post()[0].mass,
            [("length", Po["length"], "z")], "post", "western red cedar", T.post, category="other", derived=True)
    ball = [p for p in T.rootball() if p.name == "ball"]
    _simple(x, "plant_pot", "Nursery container root ball (#1, 6.5 in)", ball, "ROOTBALL", Rb["mass"],
            [("diameter", Rb["diameter"], "diameter_xy"), ("height", Rb["height"], "z")], "rootball",
            "wet peat-bark mix", T.rootball, category="other")


def _pail(x: Exporter) -> None:
    Pa = P.PAIL
    r, h, w = Pa["r"], Pa["height"], Pa["wall"]
    parts = [box((r, r, 0.003), (0.0, 0.0, 0.003))]
    hw = r * np.tan(np.pi / 16) + 0.002
    for k in range(16):
        a = 2 * np.pi * k / 16
        c = ((r + w / 2) * np.cos(a), (r + w / 2) * np.sin(a), h / 2)
        parts.append(box((w / 2, hw, h / 2), c, (0.0, 0.0, np.sin(a / 2), np.cos(a / 2))))
    cite = line_of(P, "PAIL = _p(")
    pail = union(parts)
    x.add("bucket", "2-gallon HDPE pail", "container", pail, material={"name": "HDPE"}, visual=[look("pail", pail)],
          friction=cited(P.INTERFACE["plastic"][0], "estimate", line_of(P, "INTERFACE = _p("), "plastic interface friction"),
          friction_partner=MU_PARTNER,
          dims=[dim("height", h, "z", cite), dim("width across", 2 * r, "x", cite)],
          source_ref=where(SC.SoilScene), notes="Ring of 16 wall boxes and a square bottom as their scene builds it "
          "(static in their sim). No pail mass is given in their code, so the mass is left to the importer default.",
          motion="static", extra={"fill_sand_depth_m": Pa["fill"]})


def export(x: Exporter) -> None:
    for fn in (_hand_tools, _digger, _drill, _auger, _timber, _pail):
        try:
            fn(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
    x.fail("pvc_pipe", "a plain cylinder along x with outer radius only; no wall thickness or mass is given in their code")
    x.fail("poly_tarp", "a 1.83 x 2.44 m, 0.40 kg sheet with no stiffness constants in their code (cloth block needs them)")
    x.fail("fieldstone", "ellipsoids of per-episode random radii (tools.stone), not a fixed object")
