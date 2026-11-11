# SPDX-License-Identifier: Apache-2.0
"""SolverFilm starts VBD's iterations at x + h·v on film entries (Newton's ``particle_pinned_piece_start``, WP11).

With a finite iteration count, VBD's inertial start leaves an h² sag bias: FILM-1's cantilever read 62 deg at
0.625 ms on the stock start (target 41.5) and 42 deg with the flag (item 8 G1, scr/WP4.md). SolverFilm asks for the
flag only when the model has film items and the pinned Newton takes the keyword, so a model without film items stays
bitwise stock VBD (test_film_default_off.py) and older pins keep the stock start.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.film import SolverFilm, laws
from alloy.solvers.film import solver as film_solver
from alloy.solvers.film.config import FilmConfig, register_custom_attributes
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.materials import HDPE_15UM
from alloy.solvers.film.rigs import grid_mesh

FLAG = film_solver.PINNED_PIECE_START


def _strip(film: bool) -> newton.Model:
    """A 6 x 2 cell strip at 5 mm: an HDPE film item, or the same mesh as plain cloth (no ``film:*`` values)."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    v, t = grid_mesh(6, 2, 5.0e-3)
    if film:
        add_film_mesh(b, v, t, HDPE_15UM, pos=(0.0, 0.0, 0.5))
    else:
        b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.5), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(),
                         vertices=v.astype(np.float32), indices=t.reshape(-1).tolist(), density=0.3, tri_ke=660.0,
                         tri_ka=0.0, tri_kd=1e-3, edge_ke=2e-5, edge_kd=1e-8, particle_radius=3e-4)
    b.color()
    return b.finalize(device="cpu")


@pytest.fixture
def vbd_kwargs(monkeypatch) -> list[dict]:
    """Records the keywords SolverFilm passes to SolverVBD, as if the pinned Newton took the flag; the flag is
    dropped before the real constructor runs, so this works on pins without it."""
    seen: list[dict] = []
    original = newton.solvers.SolverVBD.__init__

    def init(self, model, **kwargs):
        seen.append(dict(kwargs))
        kwargs.pop(FLAG, None)
        original(self, model, **kwargs)

    monkeypatch.setattr(newton.solvers.SolverVBD, "__init__", init)
    monkeypatch.setattr(film_solver, "_vbd_accepts", lambda name: True)
    return seen


def test_film_items_start_pinned_pieces_at_x_plus_hv(vbd_kwargs) -> None:
    solver = SolverFilm(_strip(film=True), iterations=2)
    assert solver.film_topology.any
    assert vbd_kwargs[-1].get(FLAG) is True


def test_no_film_items_keep_the_stock_start(vbd_kwargs) -> None:
    solver = SolverFilm(_strip(film=False), iterations=2)
    assert not solver.film_topology.any
    assert FLAG not in vbd_kwargs[-1]


def test_film_config_can_turn_the_pinned_start_off(vbd_kwargs) -> None:
    SolverFilm(_strip(film=True), iterations=2, film=FilmConfig(pinned_piece_start=False))
    assert FLAG not in vbd_kwargs[-1]


def test_caller_keyword_wins(vbd_kwargs) -> None:
    SolverFilm(_strip(film=True), iterations=2, **{FLAG: False})
    assert vbd_kwargs[-1][FLAG] is False


@pytest.mark.skipif(not film_solver._vbd_accepts(FLAG), reason="pinned Newton lacks particle_pinned_piece_start")
@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_film1_cantilever_at_the_clip_step_on_cpu() -> None:
    """FILM-1 at 0.625 ms, 10 iterations (the C16 clip step without its hand-set 4 substeps): within 5 deg."""
    r = laws.cantilever(HDPE_15UM, device="cpu", dt=6.25e-4, iterations=10, seconds=2.5)
    assert abs(r["angle_deg"] - r["target_deg"]) <= 5.0, r


def test_momentum_step_only_on_film_models() -> None:
    """Item 8 G2 (WP11): SolverFilm builds on SolverVBDMomentum. Free film components take A17's translation step;
    a model without film items, or FilmConfig(momentum=False), has no step (stock iterations, bitwise default-off)."""
    assert SolverFilm(_strip(film=True), iterations=2).coarse is not None
    assert SolverFilm(_strip(film=False), iterations=2).coarse is None
    assert SolverFilm(_strip(film=True), iterations=2, film=FilmConfig(momentum=False)).coarse is None


def test_no_film_model_has_no_static_dat() -> None:
    """Without film items SolverFilm is stock SolverVBD exactly, under self-contact too (no static-shape DAT)."""
    solver = SolverFilm(_strip(film=False), iterations=2, particle_enable_self_contact=True)
    assert solver._static_dat_t is None and not solver._static_dat_active(None)
    film = SolverFilm(_strip(film=True), iterations=2, particle_enable_self_contact=True)
    assert film._static_dat_t is not None
