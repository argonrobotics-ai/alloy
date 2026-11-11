# SPDX-License-Identifier: Apache-2.0
"""Registers SolverFilm as the ``film`` solver capability, route ``cloth_film`` (SC8; ruling [internal note]).

Imported by ``alloy.solvers.film`` (``load_plugins``); idempotent. Besides the capability it registers, through the
public registry calls:
- a health adapter: iterations = the projection sweeps, residual per world = the largest (s1 − 1)/cap over the
  world's limited triangles after the last step (> 1: the cap is not met);
- a state adapter: SolverVBD's arrays (alloy.coupling.adapters) plus the film's rest angles, plastic work and
  ledger, with item→world maps for the per-edge arrays.
The stability estimator comes from SolverVBD's registration by MRO (``ParticleEstimator``).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping
from typing import Any

import warp as wp

from alloy.core.protocols import DetCapability
from alloy.core.registry import (
    HealthSample,
    SolverCapability,
    register_health_adapter,
    register_solver,
    register_state_adapter,
)

from . import strain
from .config import FilmConfig, register_custom_attributes
from .emit import FilmEmitter
from .solver import SolverFilm

log = logging.getLogger("alloy.film")

ROUTE = "cloth_film"
_FIELDS = frozenset(f.name for f in dataclasses.fields(FilmConfig))


BODY_PARTICLE_MIN = 256                 # Newton's per-body soft-contact list
BODY_PARTICLE_BYTES = 256 << 20         # memory the per-body lists may take, all bodies and worlds together
J2_ITERATIONS = 20                      # VBD iterations of a film with J2 metal sheet hinges (_factory)


def body_particle_buffer(model: Any) -> int:
    """Per-body soft-contact list size: 1.25x one world's particles as a power of two, within BODY_PARTICLE_BYTES.
    A sheet wrapped round a body (the wrapped present) puts every particle in its list; past Newton's 256 the list
    overflows into a scan of every stored soft contact each iteration (correct, slower, and a warning per step)."""
    per_world = int(model.particle_count) // max(1, int(getattr(model, "world_count", 1) or 1))
    want = 1 << max(0, math.ceil(1.25 * per_world) - 1).bit_length()
    room = BODY_PARTICLE_BYTES // (4 * max(1, int(getattr(model, "body_count", 1) or 1)))
    return max(BODY_PARTICLE_MIN, min(want, 1 << max(0, room.bit_length() - 1)))


def _factory(view: Any, kwargs: Mapping[str, Any]) -> SolverFilm:
    """Entry factory: ``film`` (a FilmConfig) and/or FilmConfig fields, the rest to SolverVBD. A film with J2 metal
    sheet hinges runs J2_ITERATIONS (20) VBD iterations from the inertial start (CPU, a 6 cm, 0.25 mm 6061 sheet):
    - at VBD's 10 a stiff sheet sags far past its statics and the sag yields: hung over a 20 mm bar it droops 9 mm,
      ebar 9.6e-4, a dent nothing pressed; dropped 0.3 mm onto round supports, ebar 1.2e-3. At 20 the bar
      droop is 5.1 mm, and neither yields;
    - not VBD's adaptive start (b650fd2fd, OSS-17): it keeps gravity in a vertex's guess only where the vertex
      accelerated downward on the last step, so on an unconverged stiff sheet gravity acts one step late. A dented
      sheet released from a press then walked off its round supports and climbed 3 cm (adaptive at 40 iterations:
      the same). At 20 from the inertial start it stays on round and box supports, slow or fast release.
    A higher count asked for is kept; a lower one (the capability's default is 10) is raised, logged at info (SC8).
    Per entry: a paper film in the same entry as a metal sheet runs 20 too (cost only; more accurate)."""
    kw = dict(kwargs)
    kw.setdefault("rigid_body_particle_contact_buffer_size", body_particle_buffer(view))
    if has_j2_sheets(view):
        asked = int(kw.get("iterations", J2_ITERATIONS))
        kw["iterations"] = max(asked, J2_ITERATIONS)
        if asked < J2_ITERATIONS:
            log.info("film entry with J2 sheet hinges: %d VBD iterations, not the %d asked (J2_ITERATIONS)",
                     J2_ITERATIONS, asked)
    film = kw.pop("film", None)
    fields = {k: kw.pop(k) for k in list(kw) if k in _FIELDS}
    film = FilmConfig(**fields) if film is None else dataclasses.replace(film, **fields)
    return SolverFilm(view, film=film, **kw)


def has_j2_sheets(model: Any) -> bool:
    """Whether ``model`` has J2 metal sheet hinges (``film:j2_yield`` > 0 on some edge).
    Example: ``has_j2_sheets(sim.engine.model)`` -> ``True`` for a scene with ``add_cloth(material="steel_carbon")``."""
    ns = getattr(model, "film", None)
    arr = getattr(ns, "j2_yield", None) if ns is not None else None
    return arr is not None and int(getattr(model, "edge_count", 0) or 0) > 0 and float(arr.numpy().max()) > 0.0


@wp.kernel(enable_backward=False)
def _world_strain_kernel(tri_world_start: wp.array[wp.int32], tris: wp.array[wp.int32], ratio: wp.array[float],
                         iters: int, offset: float, out_iters: wp.array[wp.int32],
                         out_residual: wp.array[wp.float32]):
    w = wp.tid()
    r = float(0.0)
    for n in range(tri_world_start[w], tri_world_start[w + 1]):
        r = wp.max(r, ratio[tris[n]] - offset)
    out_iters[w] = iters
    out_residual[w] = wp.float32(r)


def _health_adapter(solver: SolverFilm, sample: HealthSample) -> None:
    """Per world: projection sweeps and the largest strain ratio of the last stepped state (device only). With
    ``FilmConfig.limiter="stress"`` the film stretches elastically past its cap, so the residual is that stretch as a
    share of the cap (> 1: past twice the cap) and the iterations are VBD's."""
    hv = solver.health_view()
    if hv is None:
        sample.residual.fill_(float("nan"))
        return
    ratio, tri_world_start, tris = hv
    m = solver.model
    stressed = solver.film.limiter == "stress"
    iters = int(solver.iterations) if stressed else int(solver.film.strain_iterations)
    wp.launch(strain.tri_max_strain, dim=m.tri_count,
              inputs=[solver.last_particle_q, m.tri_indices, m.tri_poses, m.film.strain_cap, ratio], device=m.device)
    wp.launch(_world_strain_kernel, dim=tri_world_start.shape[0] - 1,
              inputs=[tri_world_start, tris, ratio, iters, 1.0 if stressed else 0.0], outputs=[sample.iters,
                                                                                             sample.residual],
              device=m.device)


def _state_arrays(solver: SolverFilm) -> Mapping[str, wp.array]:
    from alloy.coupling.adapters import vbd_state_arrays

    return {**vbd_state_arrays(solver), **solver.film_state_arrays()}


def _world_maps(solver: SolverFilm) -> Mapping[str, wp.array]:
    from alloy.coupling.adapters import vbd_world_maps

    out = dict(vbd_world_maps(solver))
    edge_world = solver.edge_world()
    for name, arr in solver.film_state_arrays().items():
        if name == "film_seam_force_avg":
            out[name] = solver._seams.world              # per seam (seams.py)
        elif edge_world is not None and arr.shape[0] == solver.model.edge_count:
            out[name] = edge_world
    return out


CAPABILITY = SolverCapability(
    name="film",
    factory=_factory,
    routes=frozenset({ROUTE}),
    owns=frozenset({"particle", "tri", "edge"}),
    coupling=True,
    uniform_worlds=False,
    per_contact_friction=False,
    determinism=DetCapability.WARP_LOWERING,
    friction_mix="geometric",
    emitter=FilmEmitter(),
    register_attributes=register_custom_attributes,
    default_kwargs={"iterations": 10},
    solver_cls=SolverFilm,
)


def register() -> None:
    register_solver(CAPABILITY)
    register_health_adapter(SolverFilm, _health_adapter)
    register_state_adapter(SolverFilm, _state_arrays, world_map=_world_maps)


register()
__all__ = ["CAPABILITY", "ROUTE", "register"]
