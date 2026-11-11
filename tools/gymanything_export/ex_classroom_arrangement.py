# SPDX-License-Identifier: Apache-2.0
"""classroom_arrangement: student desk, stack chair, tutoring table, teacher's desk, book tote with textbooks,
wastebasket and a textbook, from their ``furniture.py`` part lists (boxes, tube capsules and glide cylinders, with
the item's catalogue mass shared out over the parts) and the ``params.py`` ledger.

Geometry is the union of the item's collision parts, so hollow parts (book box, tote walls) keep their walls.
"""

from __future__ import annotations

import trimesh
from _common import Exporter, box, capsule, cited, cylinder, dim, line_of, union, vis, where

from classroom_arrangement import furniture as F
from classroom_arrangement import params as P
from classroom_arrangement import render as RN

# ledger label -> spec basis
LABEL = {"datasheet": "datasheet", "retailer": "search", "estimate": "estimate", "authored": "authored",
         "published": "literature", "repo": "authored"}


def _ledger(name: str) -> dict:
    return next(r for r in P.LEDGER if r["name"] == name)


def _c(name: str, value: float | None = None, quote: str = "") -> dict:
    """A cited ledger value; ``value`` overrides when the ledger entry is a tuple."""
    row = _ledger(name)
    v = row["value"] if value is None else value
    return cited(v, LABEL[row["label"]], line_of(P, f"{name} = _p("), quote or row["basis"])


def part_mesh(p: F.Part):
    if p.kind == "box":
        return box(p.size, p.pos, p.quat)
    if p.kind == "capsule":
        return capsule(p.size[0], p.size[1], p.pos, p.quat)
    return cylinder(p.size[0], p.size[1], p.pos, p.quat)


def _item(parts) -> object:
    """Boolean union of the collision parts; the chair's three angled back panels are kept as separate closed solids
    (their mutual overlap would leave 0.2 mm slivers in a boolean union)."""
    back = [part_mesh(p) for p in parts if p.collide and p.role == "back"]
    body = union([part_mesh(p) for p in parts if p.collide and p.role != "back"])
    return trimesh.util.concatenate([body, *back]) if back else body


def _srgb(lin) -> list[float]:
    """Their COLORS table is UsdPreviewSurface diffuseColor (linear); the part colour is sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]


# textured materials of their export(): (colour used as sRGB, roughness, line pattern, what the colour is)
TEXTURED = {"desk_top": ((.70, .70, .68), .45, "base = np.array([.70, .70, .68])", "base colour of the generated top"),
            "teacher_top": ((.38, .26, .18), .45, "tint=(.38, .26, .18)", "walnut tint of the oak veneer"),
            "table_top": ((1.0, .98, .88), .45, "tint=(1.05, 0.98, 0.88)", "maple tint of the oak veneer, clipped")}


def _material_part(geom, material: str) -> dict:
    if material in TEXTURED:
        rgb, rough, pattern, what = TEXTURED[material]
        return vis(geom, rgb, line_of(RN, pattern), rough, 0.0,
                   name=f"{material} (flat colour: {what}; texture not exported)")
    rgb, rough, metal = RN.COLORS[material]
    return vis(geom, _srgb(rgb), line_of(RN, f"{material}=(("), rough, metal, name=f"{material} (linear->sRGB)")


def _visual(parts) -> list[dict]:
    """One render part per ``Part.material`` of the item's collision parts (their furniture_visual draws each part
    with ``mats[p.material]``)."""
    groups: dict[str, list] = {}
    for p in parts:
        if p.collide:
            groups.setdefault(p.material, []).append(part_mesh(p))
    return [_material_part(trimesh.util.concatenate(ms), name) for name, ms in groups.items()]


def _glide_mu() -> tuple[dict, str]:
    row = _ledger("GLIDE_MU")
    return _c("GLIDE_MU", P.GLIDE_MU["nylon"][0], "nylon glide static mu 0.235; " + row["basis"][:160]), \
        "vinyl composition tile floor"


def _furniture(x: Exporter, key: str, name: str, parts, mass_name: str, material: str, dims: list, builder) -> None:
    mu, partner = _glide_mu()
    x.add(key, name, "furniture", _item(parts), visual=_visual(parts), material={"name": material}, mass=_c(mass_name),
          friction=mu, friction_partner=partner, dims=dims, source_ref=where(builder),
          notes="Union of their collision parts (top, frame tubes, glides); catalogue mass is their total for the "
                "item, shared over the parts in their code.")


def _dl(label: str, value: float, measure: str, ledger: str) -> dict:
    return dim(label, value, measure, line_of(P, f"{ledger} = _p("))


def _desks(x: Exporter) -> None:
    _furniture(x, "student_desk", "Student desk (Virco 785)", F.desk(), "DESK_MASS",
               "hard-plastic top, steel book box and legs",
               [_dl("top width", P.DESK_TOP[0], "x", "DESK_TOP"), _dl("top depth", P.DESK_TOP[1], "y", "DESK_TOP"),
                _dl("top height", P.DESK_HEIGHT, "z", "DESK_HEIGHT")], F.desk)
    _furniture(x, "tutoring_table", "Tutoring table (Virco 4000, 30 x 60 in)", F.table(), "TABLE_MASS",
               "particleboard top on steel legs",
               [_dl("top length", P.TABLE_TOP[0], "x", "TABLE_TOP"), _dl("top width", P.TABLE_TOP[1], "y", "TABLE_TOP"),
                _dl("height", P.DESK_HEIGHT, "z", "DESK_HEIGHT")], F.table)
    _furniture(x, "teacher_desk", "Teacher's desk (Virco 543, 48 x 30 in)", F.teacher_desk(), "TEACHER_MASS",
               "particleboard top, steel pedestal",
               [_dl("top length", P.TEACHER_TOP[0], "x", "TEACHER_TOP"), _dl("top width", P.TEACHER_TOP[1], "y", "TEACHER_TOP"),
                _dl("height", P.TEACHER_HEIGHT, "z", "TEACHER_HEIGHT")], F.teacher_desk)


def _chair(x: Exporter) -> None:
    _furniture(x, "student_chair", "Stack chair (Virco 9018)", F.chair(), "CHAIR_MASS",
               "polyethylene shell on chrome tube frame",
               [_dl("overall width", P.CHAIR_OVERALL[0], "x", "CHAIR_OVERALL"),
                _dl("overall height", P.CHAIR_OVERALL[2], "z", "CHAIR_OVERALL")], F.chair)


def _loose(x: Exporter) -> None:
    mu = _c("MU_TOTE")
    tote = F.tote()
    x.add("book_tote", "Plastic book tote with four textbooks", "other", _item(tote), visual=_visual(tote),
          material={"name": "polypropylene tote, paper textbooks"}, mass=_c("TOTE", P.TOTE[3]),
          friction=mu, friction_partner="vinyl composition tile floor",
          dims=[_dl("length", P.TOTE[0], "x", "TOTE"), _dl("width", P.TOTE[1], "y", "TOTE")], source_ref=where(F.tote),
          notes="Open tote (floor and four 4 mm walls) with its four upright textbooks as one rigid body.")
    r, h, m = P.BASKET
    basket = F.basket()
    x.add("wastebasket", "Plastic wastebasket", "container", _item(basket), visual=_visual(basket),
          material={"name": "polypropylene"}, mass=_c("BASKET", m), friction=mu,
          friction_partner="vinyl composition tile floor",
          dims=[_dl("diameter", 2 * r, "diameter_xy", "BASKET"), _dl("height", h, "z", "BASKET")],
          source_ref=where(F.basket), notes="A closed solid cylinder in their sim (the hull the robot pushes); real "
          "baskets are hollow.")
    book = F.book()
    x.add("textbook", "Hardcover textbook", "other", _item(book), visual=_visual(book), material={"name": "paper and board"},
          mass=_c("BOOK", P.BOOK[3]), friction=_c("MU_BOOK"), friction_partner="painted steel book box",
          dims=[_dl("height", P.BOOK[0], "x", "BOOK"), _dl("width", P.BOOK[1], "y", "BOOK"),
                _dl("thickness", P.BOOK[2], "z", "BOOK")], source_ref=where(F.book))


def export(x: Exporter) -> None:
    for fn in (_desks, _chair, _loose):
        try:
            fn(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
