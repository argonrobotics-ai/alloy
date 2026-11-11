# SPDX-License-Identifier: Apache-2.0
"""Pours: liquid that enters the scene from a nozzle (an SPH emitter) instead of starting as a block in the air.

A pour is a liquid object whose particles start as the SPH solver's inactive spares (its "reservoir", parked at the
nozzle) and are released layer by layer by an emitter at the pour's rate and speed. Why not drop a block: a flat
block hitting a flat floor makes real pressure jets (corner and wall flip-through jets up to ~5x the impact speed),
which throw several percent of it out of a cup; the same volume poured as a stream stays in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("alloy.api")


def at_spacing(scene: Any, spacing: float, made_at: float = 4.0e-3) -> Any:
    """``scene`` with every pour re-sized for particles ``spacing`` apart (a copy; the same scene when nothing
    changes). ``add_pour`` sizes the reservoir and nozzle in whole particles of the default spacing ``made_at``; at
    another spacing (``liquid_accuracy="super"``) the reservoir holds the same volume in whole particles of that size
    and the nozzle keeps its width, area and exit speed, so the rate packs whole layers (WP2's review of
    SCR-WP9-LIQRES: a super pour of 8 ml held 10.65 ml and its layers over-packed)."""
    pours = [n for n, it in scene.items.items() if it.get("pour")]
    if not pours or abs(spacing - made_at) < 1e-12:
        return scene
    out = type(scene).from_dict(scene.to_dict())
    for name in pours:
        it, p = out.items[name], out.items[name]["pour"]
        width = 2.0 * float(p["half_extents"][0]) - 0.5 * made_at          # k x made_at: the nozzle's width
        k = max(1, round(width / spacing))
        v = float(p["volume"])
        n = max(1, round((v / spacing ** 3) ** (1.0 / 3.0)))
        m = max(1, round(v / (n * n * spacing ** 3)))
        half = [0.5 * (n + 0.5) * spacing, 0.5 * (n + 0.5) * spacing, 0.5 * (m + 0.5) * spacing]
        it["shape"] = {"primitive": {"kind": "box", "size": half}}
        it["volume"] = 8.0 * half[0] * half[1] * half[2]
        p.update(volume=n * n * m * spacing ** 3, velocity=[0.0, 0.0, -float(p["rate"]) / (k * spacing) ** 2],
                 half_extents=[(k + 0.5) * spacing / 2, (k + 0.5) * spacing / 2, spacing / 2])
        log.info("%s: pour re-sized for %.3g mm particles (nozzle %d x %d, reservoir %.4g ml)", name, spacing * 1e3, k,
                 k, p["volume"] * 1e6)
    return out


def make_spares(built: Any, index: Any, names: list[str]) -> None:
    """Turn the pours' particles into inactive SPH spares in every world (before the solver is built)."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.solvers.sph.config import PHASE_SPARE  # noqa: PLC0415

    m = built.model
    flags, phase = m.particle_flags.numpy(), m.sph.phase.numpy()
    active = int(newton.ParticleFlags.ACTIVE)
    for name in names:
        r = index.particle_range(name)
        for start in index.per_world(r.start, "particle"):
            ids = np.arange(start, start + len(r))
            flags[ids] &= ~active
            phase[ids] = PHASE_SPARE
    m.particle_flags.assign(flags)
    m.sph.phase.assign(phase)


def start(engine: Any, pours: dict[str, dict[str, Any]]) -> None:
    """Register one emitter per pour and world on the SPH solver, book the poured energy for the health monitor
    (:class:`FluxEnergy`), and rewind the emitters at every reset of their worlds (``sim.reset``, an env's reset), so
    a reset pour starts over from its full reservoir (``emitters.rewind``)."""
    import warp as wp  # noqa: PLC0415

    from alloy.api._liquid import liquid_solvers  # noqa: PLC0415
    from alloy.api.names import NameIndex  # noqa: PLC0415
    from alloy.solvers.sph.emitters import rewind  # noqa: PLC0415

    solver = next(iter(liquid_solvers(engine)), None)
    if solver is None:
        from alloy.api.errors import BuildError  # noqa: PLC0415

        raise BuildError("the scene has a pour but no liquid solver", "pour a fluid material (alloy.materials())")
    index = NameIndex.from_built(engine.built)
    for name, p in pours.items():
        n, starts = len(index.particle_range(name)), index.per_world(index.particle_range(name).start, "particle")
        x, y, z = p["pos"]
        region = wp.transform(wp.vec3(x, y, z), wp.quat_identity())
        half = wp.vec3(*p["half_extents"])
        vel = wp.vec3(*p["velocity"])
        for w in range(engine.built.world_count):
            solver.add_emitter(w, region, half, float(p["rate"]), vel, pool=np.arange(starts[w], starts[w] + n))
        log.info("%s: pouring %.4g ml at %.4g ml/s from %s", name, p["volume"] * 1e6, p["rate"] * 1e6, p["pos"])
    export_energy(engine, solver)
    engine.reset_hooks.append(lambda world_mask: rewind(solver.emitters.dev, world_mask, engine.built.world_count))


@dataclass
class FluxEnergy:
    """Step observer: after each physics step, before the health check, books the energy the pours brought in minus
    the energy of retired particles as external work (the SPH solver's ``export_flux_energy``), so the health
    monitor's energy check stays on and a pour does not read as an energy jump. Device only, captured in CUDA graphs."""

    solver: Any
    health: Any

    def __call__(self, s0: Any, s1: Any, control: Any, contacts: Any, dt: float) -> None:
        self.solver.export_flux_energy(self.health)


def export_energy(engine: Any, solver: Any) -> FluxEnergy | None:
    """Attach :class:`FluxEnergy` to ``engine`` once (None without a health monitor). Raises BuildError when the health monitor's worlds are not
    the liquid solver's worlds. Example: ``export_energy(sim.engine, sph_solver)``."""
    health = getattr(engine, "health", None)
    existing = next((o for o in engine._observers if isinstance(o, FluxEnergy)), None)
    if existing is not None or health is None:
        return existing
    worlds = getattr(getattr(getattr(solver, "emitters", None), "layout", None), "world_count", None)
    if worlds is not None and worlds != health.world_count:
        from alloy.api.errors import BuildError  # noqa: PLC0415

        raise BuildError(f"the liquid solver has {worlds} worlds but the health monitor {health.world_count}",
                         "build the pour in a scene whose worlds all hold the liquid (one Scene(worlds=...))")
    obs = FluxEnergy(solver, health)
    engine.add_step_observer(obs)
    return obs
