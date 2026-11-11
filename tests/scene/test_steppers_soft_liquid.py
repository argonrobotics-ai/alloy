# SPDX-License-Identifier: Apache-2.0
"""Soft source -> SPH proxies (SPH-VBD P1, WP9): the Scheduler proxies only the surface vertices (SolverDFSPH's soft
faces are triangles with three proxied vertices); the stock proxy, which maps bodies only and runs lagged, refuses
the pair instead of letting the liquid pass through."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.core.errors import CouplingError
from alloy.integration.stepper import _proxy_particles
from alloy.scene.steppers import StockProxyStepper


def _entry(name: str, capability: str, particles=(), bodies=()):
    return SimpleNamespace(name=name, capability=capability, particles=list(particles), bodies=list(bodies))


def _model(tris: list[tuple[int, int, int]]):
    arr = wp.array(np.asarray(tris, dtype=np.int32), dtype=wp.int32, device="cpu")
    return SimpleNamespace(tri_indices=arr, tri_count=len(tris))


def test_into_sph_only_surface_vertices_are_proxied() -> None:
    vbd = _entry("vbd", "vbd", particles=range(6))           # 0-3 on the surface, 4-5 interior
    model = _model([(0, 1, 2), (0, 2, 3)])
    assert _proxy_particles(vbd, _entry("sph", "sph"), model) == [0, 1, 2, 3]
    assert _proxy_particles(vbd, _entry("mpm", "mpm"), model) == list(range(6))    # other destinations: all


def test_stock_proxy_refuses_a_soft_source_into_sph() -> None:
    pair = SimpleNamespace(source="vbd", destination="sph")
    built = SimpleNamespace(entries=[_entry("vbd", "vbd", particles=range(4)), _entry("sph", "sph", particles=[9])],
                            coupler=SimpleNamespace(pairs=(pair,), strong_unrouted=()), world_count=1)
    with pytest.raises(CouplingError, match="soft source into SPH"):
        StockProxyStepper(built)
