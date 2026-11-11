# SPDX-License-Identifier: Apache-2.0
"""CARD-2: router route = expert label on library/labels.yaml (WP1), target ≥ 95 % agreement.

Each object is routed alone (one free body, no partners) with every builtin route plus `liquid_sph` registered (the
labels assume the full system). Rod self-tests are stubbed: they decide the CHAIN integrator only, never the route.
The "router route fails a rig the label passes" half of CARD-2 is not_run until rig data exists (§8.10)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alloy.core.errors import AlloyError
from alloy.core.registry import isolated_registry, registered_routes
from alloy.route.router import Router
from alloy.route.types import SceneContext
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.testing import scene as S

LABELS = Path(__file__).resolve().parents[2] / "library" / "labels.yaml"
PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}
TARGET = 0.95


def _library():
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    return CardLibrary()


def route_all() -> tuple[list[tuple[str, str, str, str]], int]:
    labels = yaml.safe_load(LABELS.read_text())["labels"]
    lib = _library()
    router = Router(rod_self_test=lambda rod, h: dict(PASS))
    rows = []
    with isolated_registry(empty=True):
        register_builtin_solvers()
        routes = registered_routes() | {"liquid_sph"}
        for lab in labels:
            obj = lib.resolve(lib.object(lab["object"]))
            ctx = SceneContext(rigid_partners={obj.card.name: ()}, free_bodies_per_world=1, registered_routes=routes)
            try:
                got = str(router.route(obj, ctx).route)
            except AlloyError as exc:
                got = f"error:{type(exc).__name__}"
            rows.append((lab["object"], lab["route"], got, lab.get("reason", "")))
    return rows, len(labels)


@pytest.mark.skipif(not LABELS.exists(), reason="library/labels.yaml (WP1) not present")
def test_card2_label_agreement():
    rows, n = route_all()
    agree = sum(1 for _, want, got, _ in rows if want == got)
    rate = agree / n
    misses = [(o, want, got) for o, want, got, _ in rows if want != got]
    S.record("CARD-2.agreement", rate, "pass" if rate >= TARGET else "fail", n=n, agree=agree, misses=misses)
    assert n >= 60
    assert rate >= TARGET, f"{agree}/{n} agree; misses: {misses}"
