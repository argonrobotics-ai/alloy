# SPDX-License-Identifier: Apache-2.0
"""concrete_finishing: the eleven hand tools of ``tools.py`` (floats, trowels, darby, edger, jointer, knives, spatula,
penetrometer), each the union of their collision ``Part`` primitives in the tool's body frame (origin at the centre of
its working sole). Masses and sizes are their ``params.TOOLS`` ledger row (maker's sizes, masses an estimate).
Room statics, carts, tub, hawk and mortar board are layout boxes inside their task layout, not objects with a builder.
"""

from __future__ import annotations

from _common import Exporter, cited, dim, line_of, vis, where

from concrete_finishing import params as P
from concrete_finishing import render as RN
from concrete_finishing import tools as T
from ex_sweeping import parts_mesh

# builder name -> (ledger key, display name, [(dim name, value m, measure, mesh-relevant)])
TOOLS = {
    "mag_float": ("float", "Magnesium hand float (Marshalltown 145D)", [("blade length", "blade", 0, "x")]),
    "wood_float": ("wood_float", "Wood float (Marshalltown 144D)", [("blade length", "blade", 0, "x")]),
    "trowel": ("trowel", "Finishing trowel (Marshalltown PB64D)", [("blade length", "blade", 0, "x")]),
    "darby": ("darby", "Magnesium darby (Marshalltown 4626D)", [("blade length", "blade", 0, "x")]),
    "edger": ("edger", "Hand edger (Marshalltown 136D)", [("blade length", "blade", 0, "x")]),
    "margin_trowel": ("margin", "Margin trowel (Marshalltown 52D)", []),
    "jointer": ("jointer", "Concave jointer (Marshalltown BJCC12)", [("rod length", "rod", 1, "x")]),
    "sealant_tool": ("sealant_tool", "Sealant tooling spatula (5/8 in)", []),
    "joint_knife": ("joint_knife", "Joint knife (Marshalltown JK886D)", [("blade width", "blade", 0, "x")]),
    "putty_knife": ("putty_knife", "Putty knife (Marshalltown PK878D)", [("blade width", "blade", 0, "x")]),
    "penetrometer": ("penetrometer", "Pocket penetrometer (Humboldt H-4134)", []),
}
QUOTE = ("sizes from the maker's pages; masses = shipping weight less 10-25 % (estimate); the penetrometer from the "
         "Humboldt H-4134 manual")


def _dims(builder: str, ledger: dict, rows: list) -> list[dict]:
    cite = line_of(P, "TOOLS = _p(")
    return [dim(n, ledger[k][i], m, cite) for n, k, i, m in rows]


def _srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def tool_visual(parts) -> list[dict] | None:
    """One render part per material among the tool's collision parts; colour, roughness, metallic from their
    ``render.COLORS`` (linear), the table their ``tool_visual`` looks each ``Part.material`` up in."""
    out = []
    for name in dict.fromkeys(p.material for p in parts if p.collide):
        rgb, rough, metal = RN.COLORS[name]
        geom = parts_mesh(parts, exclude={p.name for p in parts if p.material != name})
        out.append(vis(geom, [_srgb(c) for c in rgb], line_of(RN, f" {name}=(("), roughness=float(rough),
                       metallic=float(metal), name=f"{name} (linear->sRGB)"))
    return out or None


def _tool(x: Exporter, builder: str) -> None:
    key, name, rows = TOOLS[builder]
    spec = P.TOOLS[key]
    tool = T.make(builder)
    cite = line_of(P, "TOOLS = _p(")
    basis = "measured" if builder == "penetrometer" else "estimate"
    x.add(builder, name, "tool", parts_mesh(tool.parts), visual=tool_visual(tool.parts),
          material={"name": "metal blade with plastic or wood handle"},
          mass=cited(spec["mass"], basis, cite, QUOTE),
          friction=cited(T.MU_TOOL, "estimate", line_of(T, "MU_TOOL ="), "painted / bare metal and wood on cart tops and forms (estimate)"),
          friction_partner="cart tops and forms", dims=_dims(builder, spec, rows),
          source_ref=where(getattr(T, {"mag_float": "hand_float", "trowel": "finishing_trowel", "sealant_tool": "sealant_tool",
                                         "joint_knife": "knife", "putty_knife": "knife"}.get(builder, builder))),
          notes="Union of their collision parts (blade, posts or ferrule, handle); the paste model's sole is not geometry.",
          extra={"maker_part": spec["part"], "grasp": [list(map(float, tool.grasp[0])), list(map(float, tool.grasp[1]))]})


def export(x: Exporter) -> None:
    for builder in TOOLS:
        try:
            _tool(x, builder)
        except Exception as exc:  # noqa: BLE001
            x.fail(builder, f"{type(exc).__name__}: {exc}")
    for k in ("carts_tub_hawk_board_rag", "pump_room", "cmu_infill", "wallboard_patch", "floor_repair_patch", "pump_pad"):
        x.fail(k, "built from layout statics in the task layout (room boxes, parked plates, a paste field), not as a "
                  "separate object with its own builder")
