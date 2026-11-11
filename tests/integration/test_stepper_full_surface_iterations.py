# SPDX-License-Identifier: Apache-2.0
"""G4 (alloy-sc4, 2026-11-06): a film entry that receives full-surface proxy contacts runs at least
``FULL_SURFACE_MIN_ITERATIONS["film"]`` VBD iterations. At 10 the clamp's normal/friction balance does not converge at
1/960 s and the film creeps out of the pinch; at 40 the full-surface film pinch law holds. Per-vertex film entries and
other destinations keep their iterations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import alloy.integration.stepper as st
from alloy.coupling.proxy_contacts import ProxyContactPipeline


def _pair(dst: str, full: bool | None):
    pipe = None if full is None else ProxyContactPipeline(rigid_shapes=[], full_surface=full,
                                                          full_surface_shapes=[] if full else None)
    return SimpleNamespace(destination=dst, proxy=SimpleNamespace(collision_pipeline=pipe))


ENTRIES = {"film": SimpleNamespace(name="film", capability="film"), "vbd": SimpleNamespace(name="vbd", capability="vbd"),
           "sph": SimpleNamespace(name="sph", capability="sph")}


def test_floor_only_for_full_surface_film_destinations():
    floor = st.FULL_SURFACE_MIN_ITERATIONS["film"]
    assert floor == 40
    assert st.full_surface_iterations([_pair("film", True)], ENTRIES) == {"film": floor}
    assert st.full_surface_iterations([_pair("film", False)], ENTRIES) == {}         # per-vertex: unchanged
    assert st.full_surface_iterations([_pair("vbd", True)], ENTRIES) == {}           # cloth VBD: unchanged
    assert st.full_surface_iterations([_pair("sph", None)], ENTRIES) == {}           # contact-free: no pipeline


@pytest.mark.parametrize("configured, floor, want", [(10, 40, 40), (60, 40, 60), (10, 0, 10)])
def test_entry_iterations_raised_to_the_floor(monkeypatch, configured, floor, want):
    seen = {}
    cap = SimpleNamespace(factory=lambda view, kw: seen.update(kw) or "solver")
    monkeypatch.setattr(st, "capability", lambda name: cap)
    monkeypatch.setattr(st, "solver_kwargs", lambda spec, native: {"iterations": configured})
    spec = SimpleNamespace(name="film", capability="film", max_substeps=4, substeps=1, shapes=(), bodies=(),
                           particles=(0, 1, 2), joints=())
    ae = st._alloy_entry(spec, "mujoco", floor)
    assert ae.entry.solver(None) == "solver" and seen["iterations"] == want
