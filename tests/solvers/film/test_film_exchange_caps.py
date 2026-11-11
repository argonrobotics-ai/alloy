# SPDX-License-Identifier: Apache-2.0
"""A film can cap the lagged contact force of the rigid bodies coupled to it: ``FilmConfig.exchange_caps`` (opt-in;
the open-carton task's (c 4, c_d 0.5)) is the coupler's default for a SolverFilm destination, a SchedulerPolicy that
sets either cap wins, and the default None installs none (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.coupling.stiffness_cap import install_stiffness_caps
from alloy.solvers.film import FilmConfig
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.materials import HDPE_15UM
from alloy.solvers.film.rigs import FilmRig, grid_mesh

pytestmark = pytest.mark.compile_heavy
CAPS = (4.0, 0.5)                                # the open-carton task's caps


def _film_solver(**film):
    v, t = grid_mesh(4, 4, 5e-3)

    def build(b):
        add_film_mesh(b, v, t, HDPE_15UM, pos=(0.0, 0.0, 0.5))
        return {}

    return FilmRig(build, worlds=1, film=FilmConfig(**film)).solver


def _sched(solver):
    """The parts of a Scheduler install_stiffness_caps reads: one coupled body-proxy mapping onto ``solver``."""
    ids = SimpleNamespace(numpy=lambda: np.zeros(0, np.int64))
    mapping = SimpleNamespace(src_name="mujoco", dst_name="film", proxy_ids_local=ids)
    return SimpleNamespace(_proxy_mappings=[mapping], _one_way_routes=frozenset(),
                           _entries={"film": SimpleNamespace(solver=solver)})


def test_film_config_defaults_and_checks_its_exchange_caps() -> None:
    assert FilmConfig().exchange_caps is None                                 # opt-in (2026-11-09)
    assert FilmConfig(exchange_caps=CAPS).exchange_caps == CAPS
    with pytest.raises(ValueError, match="exchange_caps"):
        FilmConfig(exchange_caps=(0.0, 0.5))


def test_coupler_takes_the_films_caps_unless_the_policy_sets_its_own() -> None:
    solver = _film_solver(exchange_caps=CAPS)
    assert solver.alloy_exchange_caps == CAPS
    (cap,) = install_stiffness_caps(_sched(solver), None, None)               # the policy's default: None, None
    assert (cap.c, cap.c_d, cap.destination) == (4.0, 0.5, "film")
    (cap,) = install_stiffness_caps(_sched(_film_solver(exchange_caps=CAPS)), 2.0, None)   # a policy's cap wins
    assert (cap.c, cap.c_d) == (2.0, None)
    assert install_stiffness_caps(_sched(_film_solver()), None, None) == []   # the default: none


def test_routes_name_the_caps_on_a_film_with_a_body_on_it(monkeypatch) -> None:
    import functools

    import alloy
    from alloy.solvers.film import plugin

    monkeypatch.setattr(plugin, "FilmConfig", functools.partial(FilmConfig, exchange_caps=CAPS))   # the opt-in

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_object("sheet", "film_hdpe_sheet", pos=(0.0, 0.0, 0.002))
    scene.add_box("block", size=(0.03, 0.03, 0.03), pos=(0.0, 0.0, 0.02), mass=0.05)
    sim = alloy.build(scene, device="cpu")
    caps = [(c.destination, c.c, c.c_d) for c in sim.engine.stepper._caps]
    assert caps == [("film", 4.0, 0.5)]
    r = sim.routes()
    assert "exchange_caps" in r["sheet"] and "exchange_caps" not in r["block"]
