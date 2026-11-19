# SPDX-License-Identifier: Apache-2.0
"""The film entry's VBD iterations as object cards ask (``film.vbd_iterations``; alloy/solvers/film/plugin.py): the
entry runs the largest count any of its objects asks for, and a model whose objects ask nothing asks 0."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from alloy.solvers.film.config import VBD_ITERATIONS
from alloy.solvers.film.plugin import asked_iterations


def _model(per_particle) -> SimpleNamespace:
    arr = wp.array(np.asarray(per_particle, np.int32), dtype=wp.int32, device="cpu")
    return SimpleNamespace(film=SimpleNamespace(vbd_iterations=arr))


def test_the_entry_takes_the_largest_count_asked() -> None:
    assert asked_iterations(_model([0, 0, 40, 40, 0, 25])) == 40
    assert asked_iterations(_model([0, 0, 0])) == 0


def test_a_model_without_the_attribute_asks_nothing() -> None:
    assert asked_iterations(SimpleNamespace()) == 0
    assert asked_iterations(SimpleNamespace(film=SimpleNamespace())) == 0
    assert asked_iterations(_model([])) == 0


def test_the_attribute_is_per_particle_in_the_film_namespace() -> None:
    assert VBD_ITERATIONS == "film:vbd_iterations"
