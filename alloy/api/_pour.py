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


def reservoir(volume: float, spacing: float) -> tuple[tuple[int, int, int], int]:
    """The whole particles a pour of ``volume`` m^3 holds at ``spacing``, and the block they wait in at the nozzle.
    The count is ``round(volume / spacing^3)`` (at least 1), so the pour delivers the asked volume within half a
    particle; a block alone rounds up to 11 % off (physics ruling 2026-11-11). The block is the a x b x m of at least
    that many with the fewest left over, its sides within 2x of each other; ``make_spares`` retires the leftovers, so
    they are never poured. The ratio is rounded to 1e-6 of a particle first, so float noise never decides a tie: an
    ask saved to cards and loaded back (rate x stop_s) keeps its count (AUDIT-STACK, 2026-11-11: 50.4 ml came back
    788, not 787). Example: ``reservoir(20e-6, 4e-3)`` -> ``((5, 7, 9), 312)``."""
    count = max(1, round(round(volume / spacing ** 3, 6)))
    n = max(1, round(count ** (1.0 / 3.0)))
    best = (n, n, -(-count // (n * n)))                     # n x n x enough layers, whatever its shape
    for a in range(max(1, n - 2), n + 3):
        for b in range(a, 2 * a + 1):
            m = -(-count // (a * b))
            if max(a, b, m) <= 2 * min(a, b, m) and a * b * m < best[0] * best[1] * best[2]:
                best = (a, b, m)
    return best, count


def at_spacing(scene: Any, spacing: float, made_at: float = 4.0e-3) -> Any:
    """``scene`` with every pour re-sized for particles ``spacing`` apart (a copy; the same scene when nothing
    changes). ``add_pour`` sizes the reservoir and nozzle in whole particles of the default spacing ``made_at``; at
    another spacing (``liquid_accuracy="super"``) the reservoir holds the asked volume (``reservoir``, from the ask,
    not from the default block) in whole particles of that size and the nozzle keeps its width, area and exit speed,
    so the rate packs whole layers (WP2's review of SCR-WP9-LIQRES: a super pour of 8 ml held 10.65 ml and its layers
    over-packed)."""
    pours = [n for n, it in scene.items.items() if it.get("pour")]
    if not pours or abs(spacing - made_at) < 1e-12:
        return scene
    out = type(scene).from_dict(scene.to_dict())
    for name in pours:
        it, p = out.items[name], out.items[name]["pour"]
        width = 2.0 * float(p["half_extents"][0]) - 0.5 * made_at          # k x made_at: the nozzle's width
        k = max(1, round(width / spacing))
        (a, b, m), count = reservoir(float(p.get("asked", p["volume"])), spacing)     # scenes saved before: volume
        half = [0.5 * (a + 0.5) * spacing, 0.5 * (b + 0.5) * spacing, 0.5 * (m + 0.5) * spacing]
        it["shape"] = {"primitive": {"kind": "box", "size": half}}
        it["volume"] = 8.0 * half[0] * half[1] * half[2]
        p.update(volume=count * spacing ** 3, count=count, velocity=[0.0, 0.0, -float(p["rate"]) / (k * spacing) ** 2],
                 half_extents=[(k + 0.5) * spacing / 2, (k + 0.5) * spacing / 2, spacing / 2])
        log.info("%s: pour re-sized for %.3g mm particles (nozzle %d x %d, reservoir %.4g ml)", name, spacing * 1e3, k,
                 k, p["volume"] * 1e6)
    return out


def save_as_emitters(scene: Any, mat: Any) -> Any:
    """``Scene.save``'s scene card with every pour written as a card emitter: a downward stream of its rate, speed and
    asked volume, which ``alloy.load`` reads back as ``add_pour``. Before, the cards held only the pour's reservoir
    block, which loaded as liquid falling from the nozzle at t 0 (API-19, AUDIT-STACK 2026-11-11). A card emitter
    has no name or temperature, so the pour loads as "<material>_pour_<k>" without heat, and this says so. Returns
    the scene file. Example: ``save_as_emitters(scene, materialize(scene, root))``."""
    pours = {n: it for n, it in scene.items.items() if it.get("pour")}
    if not pours:
        return mat.scene_file
    from alloy.cards.units_yaml import yaml_dump, yaml_load  # noqa: PLC0415

    path = mat.scene_file
    card = yaml_load(path.read_text(encoding="utf-8"), path.stem)
    card["objects"] = [pl for pl in card.get("objects") or [] if str(pl.get("object")) not in pours]
    card["emitters"] = list(card.get("emitters") or []) + [_card_emitter(it, mat.materials.get(n)) for n, it in
                                                            pours.items()]
    path.write_text(yaml_dump(card, header="written by alloy.api (alloy.Scene)"), encoding="utf-8")
    heated = sorted(set(pours) & set(getattr(scene, "thermal", {}).get("objects", ())))
    log.warning("scene.save(): the cards keep each pour as a stream (card emitter), so %s load%s as "
                "<material>_pour_<k>%s. Keep names and heat with scene.to_json() / Scene.from_dict.",
                ", ".join(sorted(pours)), "s" if len(pours) == 1 else "",
                f", and {', '.join(heated)} without temperature" if heated else "")
    return path


def _card_emitter(item: dict[str, Any], material: str | None) -> dict[str, Any]:
    """One pour item as a scene card ``emitters`` entry (``alloy.cards.types.EmitterSpec``)."""
    p = item["pour"]
    rate = float(p["rate"])
    return {"kind": "emitter", "material": material or str(item.get("material")),
            "pose": [*(float(c) for c in p["pos"]), 0.0, 0.0, 0.0, 1.0],
            "half_extents": [float(h) for h in p["half_extents"]], "rate": rate,
            "velocity": [0.0, 0.0, float(p["velocity"][2])], "start_s": 0.0,
            "stop_s": float(p.get("asked", p["volume"])) / rate}


def asked_volume(item: dict[str, Any]) -> float | None:
    """The volume (m^3) a liquid item asked for: a pour's ask (scenes saved before the exact count: its rounded
    volume), else the box's. ``sim``'s build-time check compares the simulated volume with it, so a tiny pour that
    rounds far off still warns (0.1 ml is 2 particles, +28 %; AUDIT-STACK's nit). Example: ``asked_volume(item)``."""
    p = item.get("pour") or {}
    return p.get("asked") or p.get("volume") or item.get("volume")


def make_spares(built: Any, index: Any, pours: dict[str, dict[str, Any]]) -> None:
    """Turn the pours' particles into inactive SPH spares in every world (before the solver is built): the first
    ``count`` of each pour's block (``reservoir``); the rest are retired, never poured and not counted as escaped.
    The pour's handle then covers its ``count`` particles only, so everything built after this (the sim's names,
    ``"<name>.points"`` / ``.active``, ``sim.liquid_volume``, heat) reads exactly the poured liquid, and a finished
    pour reads all of it active (AUDIT-BENCH, 2026-11-11)."""
    import dataclasses  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.solvers.sph.config import PHASE_RETIRED, PHASE_SPARE  # noqa: PLC0415

    m = built.model
    flags, phase = m.particle_flags.numpy(), m.sph.phase.numpy()
    active = int(newton.ParticleFlags.ACTIVE)
    for name, p in pours.items():
        r = index.particle_range(name)
        n = min(len(r), int(p.get("count", len(r))))
        for start in index.per_world(r.start, "particle"):
            ids = np.arange(start, start + len(r))
            flags[ids] &= ~active
            phase[ids[:n]], phase[ids[n:]] = PHASE_SPARE, PHASE_RETIRED
        h = built.handles[name]                         # the leftovers (the block's tail) belong to no name
        built.handles[name] = dataclasses.replace(h, template=dataclasses.replace(h.template, particle=r[:n]))
    m.particle_flags.assign(flags)
    m.sph.phase.assign(phase)


def start(engine: Any, pours: dict[str, dict[str, Any]]) -> None:
    """Register one emitter per pour and world on the SPH solver, book the poured energy for the health monitor
    (:class:`FluxEnergy`), and put the emitters' layer clocks and cursors in the start a reset restores
    (:func:`remake_start`), so a reset pour starts over from its full reservoir, and a pour reset after
    ``sim.settle`` goes on from the settled pour."""
    import warp as wp  # noqa: PLC0415

    from alloy.api._liquid import liquid_solvers  # noqa: PLC0415
    from alloy.api.names import NameIndex  # noqa: PLC0415

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
    remake_start(engine)
    refresh_snapshots(engine)


def remake_start(engine: Any) -> None:
    """Remake the engine's start snapshot, what ``reset`` restores, with the pours' emitters in it. The engine made it
    at build, before ``start`` added them, so a reset put the particles back but not the emitters' layer clocks and
    cursors. Rewinding those to 0 at every reset did not do: after ``sim.settle`` the start holds the settled liquid,
    and a rewound emitter released those particles again from the nozzle (and counted their mass in twice). Now a
    reset restores the emitters with the particles, from the build or from ``settle``. Arrays the old snapshot held
    keep the values it saved; the emitters' are taken now, as a fresh build's (0). Example: ``remake_start(engine)``."""
    import warp as wp  # noqa: PLC0415

    old = engine._initial
    new = engine._snapshot()
    new.save(engine.state)
    held = {slot.name: slot.store for slot in old.slots}
    for slot in new.slots:
        saved = held.get(slot.name)
        if saved is not None and saved.size == slot.store.size and saved.dtype == slot.store.dtype:
            wp.copy(slot.store, saved)
    engine._initial = new
    log.debug("start snapshot remade with the pour emitters (%d arrays)", len(new.slots))


def refresh_snapshots(engine: Any) -> None:
    """Remake the engine's rollback snapshots now that the pours' emitters exist. The engine made them at build,
    before ``start`` added the emitters, and a snapshot keeps the arrays it was made with: a rolled-back period put
    the particles' flags back but not the emitter's cursor, so each re-run that had released a layer skipped it for
    good (GAP-LC's 20 ml pour onto cloth, 2026-11-11: three rollbacks as the stream landed, 244 of 294 particles
    poured). ``make_period_snapshots`` remakes them, the one helper ``IntegratedSimulation.__init__`` uses too; before
    the first step only (a captured graph holds the snapshots' arrays). Example: ``refresh_snapshots(sim.engine)``."""
    if not hasattr(engine, "_per_world_snapshot"):  # never: both API build paths make an IntegratedSimulation
        return                                       # (wiring.build_simulation, _plugins); bare Simulations never pour
    if getattr(engine, "steps", 0):
        log.warning("pour started after %d steps: the rollback snapshots keep no emitter state, so a rolled-back "
                    "period can skip a layer", engine.steps)
        return
    from alloy.integration.period_snapshots import make_period_snapshots  # noqa: PLC0415

    make_period_snapshots(engine, good=True)


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
