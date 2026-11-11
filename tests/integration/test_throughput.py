# SPDX-License-Identifier: Apache-2.0
"""WP12 throughput harness: argument parsing, entry kwarg overrides and the cost-model fit (CPU, no scene)."""

from __future__ import annotations

import dataclasses

import pytest

from alloy.integration import throughput as T
from alloy.scene.types import EntrySpec


def test_parse_values():
    assert T.parse_sets(["a=1", "b=1e-6", "c=none", "d=true", "e=cg", "f=(50,1e-10)"]) == {
        "a": 1, "b": 1e-6, "c": None, "d": True, "e": "cg", "f": (50, 1e-10)}
    with pytest.raises(ValueError):
        T.parse_sets(["novalue"])


def test_entry_kwargs_merge_and_drop():
    mj = EntrySpec("mujoco", "mujoco", (0,), (), (0,), (0,), 1, 1e-3,
                   solver_kwargs={"solver": "cg", "alloy:noslip": (50, 1e-10)})
    vbd = EntrySpec("vbd", "vbd", (), (0,), (), (), 1, 1e-3, solver_kwargs={"iterations": 5})
    built = dataclasses.make_dataclass("B", ["entries"])(entries=[mj, vbd])
    out = T.with_entry_kwargs(built, {"solver": "newton", "alloy:noslip": None, "tolerance": 1e-6})
    assert out.entries[0].solver_kwargs == {"solver": "newton", "tolerance": 1e-6}
    assert out.entries[1].solver_kwargs == {"iterations": 5}
    assert mj.solver_kwargs["solver"] == "cg"          # the input is not mutated
    assert T.with_entry_kwargs(built, {}) is built


def test_fit_recovers_floor_and_per_world_cost():
    pts = [T.Point("x", "s", n, 5, 10.0 + 0.5 * n, 0.0, 0, 64, {}, True, 0) for n in (1, 64, 256)]
    f = T.fit(pts, 0.02)
    assert f["t_floor_ms"] == pytest.approx(10.0, rel=1e-6)
    assert f["c_w_ms"] == pytest.approx(0.5, rel=1e-6)
    assert f["ceiling_rt"] == pytest.approx(40.0, rel=1e-6)
    assert T.fit(pts[:1], 0.02) is None


def test_route_profile_reaches_the_assembler_and_the_caveat_is_reported(monkeypatch):
    import alloy.scene.assembler as A
    from alloy.scene.fast_profile import FAST_CAVEAT

    seen = {}

    class FakeAssembler:
        def __init__(self, lib, router=None, *, route_profile="default"):
            seen["route_profile"] = route_profile

        def assemble(self, card, device, *, world_count=None):
            return "built"

    monkeypatch.setattr(A, "SceneAssembler", FakeAssembler)
    assert T.assemble("task_t09", "cpu", 4, "fast") == "built"
    assert seen["route_profile"] == "fast"

    def sim_with(warnings):
        report = dataclasses.make_dataclass("R", ["warnings"])(warnings=warnings)
        built = dataclasses.make_dataclass("Bt", ["report"])(report=report)
        return dataclasses.make_dataclass("S", ["parts"])(parts=dataclasses.make_dataclass("P", ["built"])(built))

    assert T.route_caveat(sim_with(["x", FAST_CAVEAT])) == FAST_CAVEAT
    assert T.route_caveat(sim_with(["x"])) is None
    assert T.route_caveat(object()) is None
