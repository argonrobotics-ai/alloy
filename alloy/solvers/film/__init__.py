# SPDX-License-Identifier: Apache-2.0
"""Thin-film plug-in (SC8): ``SolverFilm`` = Newton's SolverVBD + strain limiting, an elastoplastic hinge (crease
memory) and per-triangle air drag, switched on per object by ``film:*`` custom attributes (opt-in; a model without
them steps exactly as SolverVBD). Importing the package registers capability ``film`` (route ``cloth_film``).
Design: docs/notes/sc8-film-design.md; laws: alloy.solvers.film.laws (FILM-1..6).
"""

from alloy.solvers.film.config import FilmConfig
from alloy.solvers.film.solver import SolverFilm
from alloy.solvers.film import plugin as _plugin  # noqa: F401  (registers the capability)

__all__ = ["FilmConfig", "SolverFilm"]
