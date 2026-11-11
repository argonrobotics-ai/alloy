# SPDX-License-Identifier: Apache-2.0
"""sweeping: brooms, brush, dustpans and containers built by their ``toolkit.py`` from collision primitives.

Geometry is the union of their rigid ``Part`` primitives (boxes, capsules, cylinders) in the tool's body frame, the
shapes their rigid solver collides with. Bristle banks are a separate filament solver in their code and are not
meshed; they are described under ``extra.bristles``. The lobby pan is three kinds of rigid body (pan, handle,
wheels) joined by hinges, so it is exported as one spec per body.
"""

from __future__ import annotations

from _common import Exporter, box, capsule, cited, cylinder, dim, line_of, union, vis, where

from sweeping import params as P
from sweeping import render as RN
from sweeping import toolkit as T
from sweeping import visuals as V

MU_LINE = "TOOL_MU ="
FLOOR = "the vinyl floor"


def parts_mesh(parts, exclude=()):
    """Union of the primitives of ``parts`` (their ``Part`` records: kind, size, pos, quat) as one closed mesh."""
    out = []
    for p in parts:
        if p.name in exclude or not p.collide:
            continue
        if p.kind == "box":
            out.append(box(p.size, p.pos, p.quat))
        elif p.kind == "capsule":
            out.append(capsule(p.size[0], p.size[1], p.pos, p.quat))
        elif p.kind == "cylinder":
            out.append(cylinder(p.size[0], p.size[1], p.pos, p.quat))
        else:
            raise ValueError(f"unsupported primitive {p.kind!r} in part {p.name}")
    return union(out)


def look(name: str, label: str = "") -> dict:
    """Render look ``name`` from their tables (``visuals.PLASTIC`` / ``visuals.PLAIN`` as linear colour + roughness
    [+ metallic], else ``render.COLORS``) as a dict of ``vis`` keywords; linear values go through their own ``srgb``."""
    for module, table in ((V, V.PLASTIC), (V, V.PLAIN), (RN, RN.COLORS)):
        if name not in table:
            continue
        row = table[name]
        return {"base_color": [c / 255.0 for c in V.srgb(row[0])], "roughness": float(row[1]),
                "metallic": float(row[2]) if len(row) > 2 else None,
                "cite": _table_line(module, name), "name": f"{label or name} (linear->sRGB)"}
    raise KeyError(f"no render look {name!r}")


def _table_line(module, name: str) -> str:
    """Line of ``name``'s entry in one of their colour tables (``dict(a=((...`` keyword or ``"a": (`` literal)."""
    for pat in (f" {name}=((", f"({name}=((", f'"{name}": ('):
        try:
            return line_of(module, pat)
        except KeyError:
            continue
    raise KeyError(f"no table line for {name!r} in {module.__name__}")


def look_parts(parts, groups) -> list[dict]:
    """One render part per ``(look name, label, [part names])`` group: the union of those collision primitives."""
    every = {p.name for p in parts}
    out = []
    for name, label, names in groups:
        geom = parts_mesh(parts, exclude=every - set(names))
        out.append(vis(geom, **look(name, label)))
    return out


def _mass(spec: dict, basis: str, pattern: str, value=None) -> dict:
    return cited(spec["mass"] if value is None else value, basis, line_of(P, pattern), spec["source"][:400])


def _mu(value: float, pattern: str, quote: str, partner_basis: str = "estimate") -> dict:
    return cited(value, partner_basis, line_of(P, pattern), quote)


# key -> (toolkit builder, spec dict, name, mass basis, line pattern, mass-line pattern, dims [(name, value, measure)])
BROOMS = {
    "angle_broom": ("angle_broom", P.ANGLE_BROOM, "Angle broom (Libman Commercial 201)", "estimate", "ANGLE_BROOM = dict",
                    "block=(0.279", [("sweep width", 0.279, "x")]),
    "push_broom": ("push_broom", P.PUSH_BROOM, "Push broom (Rubbermaid FG9B0600)", "estimate", "PUSH_BROOM = dict",
                   "block=(0.457", [("block length", 0.457, "x")]),
    "lobby_broom": ("lobby_broom", P.LOBBY_BROOM, "Lobby broom (Rubbermaid FG637400)", "datasheet", "LOBBY_BROOM = dict",
                    "handle_len=0.734", [("block length", 0.190, "x")]),
    "counter_brush": ("counter_brush", P.COUNTER_BRUSH, "Counter brush (Rubbermaid FG634200)", "derived",
                      "COUNTER_BRUSH = dict", "block=(0.203", [("overall length", 0.318, "x")]),
}


# broom -> render looks of (head block, handle); every hanger cap is vinyl_black. Their push broom and counter brush
# draw wood_beech (a downloaded Poly Haven veneer), so the colour of their physics-view wood stands in.
BROOM_LOOKS = {"angle_broom": ("pl_red", "handle_steel"), "lobby_broom": ("pl_black", "handle_steel"),
               "push_broom": ("handle_wood", "handle_wood"), "counter_brush": ("brush_wood", "brush_wood")}


def _broom_visual(key: str, tool: dict) -> list[dict]:
    head, handle = BROOM_LOOKS[key]
    groups = [(head, "head block", ["block"]), (handle, "handle", ["handle"])]
    if any(p.name == "cap" for p in tool["parts"]):
        groups.append(("vinyl_black", "hanger cap", ["cap"]))
    return look_parts(tool["parts"], groups)


def _bristles(spec: dict) -> dict:
    return {"filaments": spec["filaments"], "clusters": spec["clusters"], "filament_diameter_m": spec["filament_d"],
            "material": spec["material"], "free_length_m": list(spec["bristle_len"])}


def _broom(x: Exporter, key: str) -> None:
    builder, spec, name, basis, pat, mass_pat, dims = BROOMS[key]
    tool = getattr(T, builder)()
    x.add(key, name, "tool", parts_mesh(tool["parts"]),
          material={"name": "plastic block, steel or wood handle, polymer bristles (bristles not meshed)"},
          mass=_mass(spec, basis, mass_pat),
          friction=_mu(P.TOOL_MU, MU_LINE, "tool plastics and painted steel on vinyl and on each other (estimate)"),
          friction_partner=FLOOR, dims=[dim(n, v, m, line_of(P, mass_pat)) for n, v, m in dims],
          source_ref=where(getattr(T, builder)), visual=_broom_visual(key, tool),
          notes="Rigid parts only (block, handle, hanger cap); the bristle bank is a separate filament model, so the "
                "mesh has no bristles and the mass is their tool mass.",
          extra={"bristles": _bristles(spec), "bristle_mass_kg": float(tool["bristle_mass"]),
                 "bristle_material_E_Pa": (P.PET if spec["material"] == "PET" else P.PP)["E"]})


def _pan_extra(tool: dict) -> dict:
    return {"mouth": {k: float(v) for k, v in tool["mouth"].items()}}


def _hand_pan(x: Exporter) -> None:
    spec, tool = P.HAND_PAN, T.hand_pan()
    x.add("hand_pan", "Hand pan (Rubbermaid FG200500)", "tool", parts_mesh(tool["parts"]),
          material={"name": "injection-moulded plastic"}, mass=_mass(spec, "datasheet", "pan=(0.311"),
          friction=_mu(spec["mu"], "pan=(0.311", "pan plastic on the floor (their pan friction 0.30)"),
          friction_partner=FLOOR,
          dims=[dim("pan width", 0.311, "x", line_of(P, "pan=(0.311"))],
          visual=look_parts(tool["parts"], [("pl_black", "pan", [p.name for p in tool["parts"]])]),
          source_ref=where(T.hand_pan), notes="Open scoop from their collision boxes: floor slab, lip, walls, handle.",
          extra=_pan_extra(tool))


def _lobby_pan(x: Exporter) -> None:
    spec, tool = P.LOBBY_PAN, T.lobby_pan()
    pat = "pan=(0.286"
    cite_src = spec["source"][:400]
    joint = {"cite": where(T.lobby_pan)}
    x.add("lobby_pan__pan", "Lobby dustpan, pan (Rubbermaid FG253100)", "articulated", parts_mesh(tool["parts"]),
          material={"name": "injection-moulded plastic"},
          mass=cited(spec["pan_mass"] - 2 * spec["wheel_mass"], "derived", line_of(P, "pan_mass=0.63"),
                     "pan_mass less the two wheels (toolkit.lobby_pan): " + cite_src),
          friction=_mu(spec["mu"], "wheel_mu=0.5", "pan plastic on the floor (their pan friction 0.30)"),
          friction_partner=FLOOR, dims=[dim("pan width", 0.286, "x", line_of(P, pat))],
          visual=look_parts(tool["parts"], [("pl_black", "pan", [p.name for p in tool["parts"]])]),
          source_ref=where(T.lobby_pan),
          notes="Pan shell only (floor slab, lip, walls, hood). Maker weight 0.84 kg is split by them into pan, handle "
                "and wheels; the shell is their pan_mass 0.63 kg less the two wheels, which are exported separately.",
          extra={"mouth": _pan_extra(tool)["mouth"], "total_mass_kg": spec["mass"], "joint": None})
    x.add("lobby_pan__handle", "Lobby dustpan, handle and yoke", "articulated", parts_mesh(tool["handle_parts"]),
          material={"name": "vinyl-coated steel tube, plastic yoke"},
          mass=cited(spec["handle_mass"], "estimate", line_of(P, "pan_mass=0.63"), cite_src),
          friction=_mu(P.HANDLE_MU, "HANDLE_MU =", "painted steel handle elsewhere than on a finger pad"),
          friction_partner="a gripper pad or other tools",
          visual=look_parts(tool["handle_parts"], [("handle_steel", "steel tube", ["handle"]),
                                                   ("pl_black", "plastic yoke", ["yoke_l", "yoke_r"])]),
          source_ref=where(T.lobby_pan),
          notes="Geometry in the pan frame at the rest pose (hinge angle 0); their body frame is at the pivot.",
          extra={"joint": {"type": "revolute", "parent": "lobby_pan__pan", "axis": list(tool["pivot_axis"]),
                           "pivot": list(tool["pivot"]), "range": [float(v) for v in tool["pivot_limits"]], **joint}})
    _wheels(x, tool, spec)


def _wheels(x: Exporter, tool: dict, spec: dict) -> None:
    for wh, side in zip(tool["wheels"], ("l", "r")):
        part = wh["parts"][0]
        x.add(f"lobby_pan__wheel_{side}", f"Lobby dustpan, rear wheel ({side})", "articulated", parts_mesh([_centred(part)]),
              material={"name": "plastic"}, mass=cited(spec["wheel_mass"], "estimate", line_of(P, "pan_mass=0.63"),
                                                       spec["source"][:300]),
              friction=cited(spec["wheel_mu"], "estimate", line_of(P, "wheel_mu=0.5"), "wheel on the floor"),
              visual=look_parts([_centred(part)], [("pl_grey", "wheel", [part.name])]),
              friction_partner=FLOOR, dims=[dim("wheel diameter", 2 * spec["wheel_r"], "max", line_of(P, "wheel_mu=0.5"))],
              source_ref=where(T.lobby_pan), notes="Wheel in its own body frame (axle along x).",
              extra={"joint": {"type": "revolute", "parent": "lobby_pan__pan", "axis": list(wh["axis"]),
                               "pivot": [float(v) for v in wh["pos"]], "range": None, "cite": where(T.lobby_pan)}})


def _centred(part):
    import dataclasses  # noqa: PLC0415
    return dataclasses.replace(part, pos=(0.0, 0.0, 0.0))


CONTAINER_LOOKS = {"barrel": "pl_barrel", "pail": "pl_pail", "wastebasket": "pl_basket"}
CONTAINERS = {
    "barrel": ("Barrel 20 gal (Rubbermaid BRUTE 2620)", T.barrel, P.BARREL, "datasheet", "BARREL = dict",
               [("height", "h", "z"), ("diameter", "d", "diameter_xy")]),
    "pail": ("Utility pail 14 qt (Rubbermaid FG261400)", T.pail, P.PAIL, "derived", "PAIL = dict",
             [("height", "h", "z"), ("diameter", "d", "diameter_xy")]),
    "wastebasket": ("Wastebasket 28 qt (Rubbermaid FG295600)", T.wastebasket, P.WASTEBASKET, "datasheet",
                    "WASTEBASKET = dict", []),
}


def _container(x: Exporter, key: str) -> None:
    name, builder, spec, basis, pat, dims = CONTAINERS[key]
    tool = builder()
    cites = [dim(n, spec[k], m, line_of(P, pat)) for n, k, m in dims]
    if key == "wastebasket":
        cites = [dim("length", spec["size"][0], "x", line_of(P, pat)), dim("width", spec["size"][1], "y", line_of(P, pat)),
                 dim("height", spec["size"][2], "z", line_of(P, pat))]
    x.add(key, name, "container", parts_mesh(tool["parts"]), material={"name": "moulded plastic"},
          mass=_mass(spec, basis, pat), friction=_mu(P.TOOL_MU, MU_LINE, "tool plastics on vinyl (estimate)"),
          friction_partner=FLOOR, dims=cites, source_ref=where(builder),
          visual=look_parts(tool["parts"], [(CONTAINER_LOOKS[key], "moulded plastic", [p.name for p in tool["parts"]])]),
          notes="Walls and bottom from their collision primitives (flat staves for the round ones), open at the top.",
          extra={"inner": {k: (float(v) if not isinstance(v, str) else v) for k, v in tool["inner"].items()}})


def export(x: Exporter) -> None:
    steps = [(k, lambda k=k: _broom(x, k)) for k in BROOMS]
    steps += [("hand_pan", lambda: _hand_pan(x)), ("lobby_pan", lambda: _lobby_pan(x))]
    steps += [(k, lambda k=k: _container(x, k)) for k in CONTAINERS]
    for key, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - one bad object must not stop the sim
            x.fail(key, f"{type(exc).__name__}: {exc}")
