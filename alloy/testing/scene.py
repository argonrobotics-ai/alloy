# SPDX-License-Identifier: Apache-2.0
"""WP2 scene fixtures: an in-memory card library and the mixed scene cards used by the assembler tests (CARD-3,
CARD-6, the arm + box + cloth acceptance scene). Values are test fixtures (source "model")."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from alloy.cards.types import (Crease, CurveSource, ObjectCard, PairCard, ParticleSource, Placement, Primitive,
                               Randomization, ResolvedObject, RobotRef, SceneCard)
from alloy.core.ids import digest_of
from alloy.testing import route as T
from alloy.testing.fakes import FakeCardLibrary

UP = (0.0, 0.0, 0.0, 1.0)


class TestLibrary(FakeCardLibrary):
    """FakeCardLibrary whose `resolve` uses alloy.testing.route.extents_of (curves and particle regions too) and
    the routing-worst end of each randomization: `lo` for stiffness-like fields, `hi` otherwise."""

    __test__ = False
    LOW_IS_WORST = ("youngs", "contact_stiffness")

    def resolve(self, card: ObjectCard, placement: Placement | None = None) -> ResolvedObject:
        worst = {}
        for r in (placement.randomize if placement else ()):
            worst[r.field] = r.lo if r.field.endswith(self.LOW_IS_WORST) else r.hi
        mat = self.materials[card.material]
        return ResolvedObject(card, mat, {k: self.materials[v] for k, v in card.part_materials.items()},
                              digest_of((card, mat)), T.extents_of(card.shape), worst, ())


def library(objects: Iterable[ResolvedObject], pairs: Sequence[PairCard] = ()) -> TestLibrary:
    objs = list(objects)
    return TestLibrary(materials={o.material.name: o.material for o in objs},
                       objects={o.card.name: o.card for o in objs},
                       pairs={(p.a, p.b): p for p in pairs})


def place(name: str, x: float = 0.0, y: float = 0.0, z: float = 0.0, randomize=()) -> Placement:
    return Placement(name, (x, y, z, *UP), tuple(randomize))


def pair(a: str, b: str, mu_s: float, mu_d: float, e: float | None = None) -> PairCard:
    return PairCard(a=a, b=b, mu_static=T.q(mu_s), mu_dynamic=T.q(mu_d), restitution=T.q(e) if e is not None else None)


# --- objects -----------------------------------------------------------------------------------------------------

def table() -> ResolvedObject:
    return T.box("table", T.material("wood", **T.WOOD), 0.4, 0.4, 0.02, motion="static")


def box(name: str = "box", hx: float = 0.03) -> ResolvedObject:
    return T.box(name, T.material("abs", density=1050.0, E=2.3e9, nu=0.35, mu=0.4, e=0.3, k=2e6), hx, hx, hx)


def rod() -> ResolvedObject:
    return T.rod("rod", T.material("steel", **T.STEEL), 0.005, 0.2)


def pad() -> ResolvedObject:
    return T.box("pad", T.material("rubber", **{**T.RUBBER, "E": 1e8}), 0.02, 0.02, 0.01)


def cable() -> ResolvedObject:
    pts = tuple((0.025 * i - 0.25, 0.0, 0.0) for i in range(21))
    return T.obj("cable", CurveSource(points=pts, radius=0.003), T.material("pvc", **T.CABLE_PVC))


def cloth() -> ResolvedObject:
    return T.box("towel", T.material("cotton", **T.COTTON, beta=1e-3), 0.1, 0.1, 2.5e-4,
                 budget=__import__("alloy.cards.types", fromlist=["Budget"]).Budget(particles=121))


def sponge() -> ResolvedObject:
    from alloy.cards.types import Budget

    return T.box("sponge", T.material("foam", **T.PU_FOAM, beta=1e-3), 0.04, 0.03, 0.02, budget=Budget(tets=200))


def sand() -> ResolvedObject:
    return T.obj("sand", ParticleSource(((-0.03, -0.03, 0.0), (0.03, 0.03, 0.03)), 0.002, grain_diameter=3e-4),
                 T.material("sand", density=1600.0, E=1e7, nu=0.3, friction_angle=math.radians(33)))


def lid() -> ResolvedObject:
    crease = Crease(a=(0.0, -0.1, 0.0), b=(0.0, 0.1, 0.0), stiffness=T.q(0.05))
    return T.box("lid", T.material("board", density=700.0, E=3e9, nu=0.3), 0.15, 0.1, 2.5e-4,
                 features=T.features(creases=(crease,)))


# --- scenes ------------------------------------------------------------------------------------------------------

def scene(objs: Sequence[ResolvedObject], placements: Sequence[Placement], *, pairs=(), worlds: int = 2,
          robots: Sequence[RobotRef] = ()) -> tuple[SceneCard, TestLibrary]:
    lib = library(objs)
    return SceneCard(robots=tuple(robots), objects=tuple(placements), pairs=tuple(pairs), worlds=worlds), lib


def box_on_table(worlds: int = 2):
    return scene([table(), box()], [place("table"), place("box", z=0.051)], worlds=worlds)


def mixed_scenes() -> list[tuple[str, SceneCard, TestLibrary]]:
    """Ten mixed scenes for CARD-3 readback (rigid, compliant, chain, cloth, soft, MPM, panels, randomized)."""
    out = [("box_on_table", *box_on_table())]
    out.append(("stack_pair_card", *scene([table(), box("a"), box("b")],
                                          [place("table"), place("a", z=0.051), place("b", z=0.112)],
                                          pairs=[pair("abs", "wood", 0.6, 0.5, 0.2)])))
    out.append(("rod_and_pad", *scene([table(), rod(), pad()], [place("table"), place("rod", z=0.03),
                                                                 place("pad", x=0.1, z=0.031)])))
    out.append(("cable_on_table", *scene([table(), cable()], [place("table"), place("cable", z=0.03)])))
    out.append(("cloth_on_table", *scene([table(), cloth()], [place("table"), place("towel", z=0.03)])))
    out.append(("sponge_on_table", *scene([table(), sponge()], [place("table"), place("sponge", z=0.045)])))
    out.append(("sand_heap", *scene([table(), sand()], [place("table"), place("sand", z=0.021)])))
    out.append(("creased_lid", *scene([table(), lid()], [place("table"), place("lid", z=0.03)])))
    out.append(("box_cloth_cable", *scene([table(), box(), cloth(), cable()],
                                          [place("table"), place("box", z=0.051), place("towel", z=0.12),
                                           place("cable", y=0.15, z=0.03)])))
    rnd = [Randomization("material.density", 900.0, 1200.0), Randomization("material.surface.mu_dynamic", 0.3, 0.5),
           Randomization("pose.x", -0.05, 0.05)]
    out.append(("randomized", *scene([table(), box(), cloth()],
                                     [place("table"), place("box", z=0.051, randomize=rnd),
                                      place("towel", z=0.12, randomize=[Randomization("material.elastic.youngs",
                                                                                      0.8e6, 1.2e6)])],
                                     worlds=4)))
    return out


def record(metric: str, value: float | None, status: str, **extra) -> None:
    """Append one result row to $ALLOY_RESULTS_DIR/wp2.jsonl when the variable is set (docs/notes/wp2-results.md)."""
    import json
    import os
    from pathlib import Path

    root = os.environ.get("ALLOY_RESULTS_DIR")
    if not root:
        return
    path = Path(root) / "wp2.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"metric": metric, "value": value, "status": status, **extra}
    with path.open("a") as f:
        f.write(json.dumps(row, default=float, sort_keys=True) + "\n")
