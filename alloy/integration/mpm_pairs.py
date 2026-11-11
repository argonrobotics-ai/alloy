# SPDX-License-Identifier: Apache-2.0
"""Pair friction into implicit MPM's colliders (``scene.set_friction`` and sourced pair cards; coordinator GO
2026-11-08 14:59Z, known gap mpm-grain-slope-resolution).

Newton's MPM solver gives each collider shape one Coulomb friction, read once, when the solver is built, from that
shape's own ``shape_material_mu`` (the material's dynamic friction). The pair table (D13) never reached it, so
``set_friction(table, sand, 0.8)`` changed nothing for the sand. ``apply_pair_friction`` runs after the solvers are
built and before any step or graph capture. For every shape an MPM entry collides with, if the pair of the entry's
material and the shape's material is sourced (a scene pair from ``set_friction`` or a library pair card), the
collider takes that pair's dynamic friction, per world when the table is randomized. Unsourced pairs keep the
shape's own friction, as before: a scene without such pairs is unchanged, bit for bit.

Newton's collider has one friction per shape for all of an entry's particles. A shape whose sourced pairs with the
entry's materials disagree, or that one material pairs with and another does not, cannot be represented and is
refused at build (``AssemblyError`` naming the materials and the fix). So is a shape shared by every world (the
ground) whose randomized pair friction differs between worlds.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.scene.types import BuiltScene, PairTable

log = logging.getLogger("alloy.integration")

__all__ = ["apply_pair_friction", "collider_shapes"]


def apply_pair_friction(built: BuiltScene, solvers: Mapping[str, Any]) -> list[str]:
    """Write sourced pair friction into each MPM entry's collider shapes (module docstring); one note per entry
    that changed, for the build notes."""
    table = built.pair_table
    if table is None:
        return []
    sourced = _sourced(table, built.report.unsourced_pairs)
    notes = []
    for spec in built.entries:
        solver = solvers.get(spec.name)
        if spec.capability != "mpm" or solver is None or not hasattr(solver, "_mpm_model"):
            continue
        note = _apply_entry(spec.name, solver, table, sourced, np.asarray(spec.particles, dtype=np.int64))
        if note:
            notes.append(note)
    return notes


def _sourced(table: PairTable, unsourced: Any) -> np.ndarray:
    """[M, M] bool: the pair of table materials i and j has a pair card (scene or library)."""
    cards = [k.split("@", 1)[0] for k in table.materials]
    off = {tuple(sorted(p)) for p in unsourced}
    m = len(cards)
    return np.array([[tuple(sorted((cards[i], cards[j]))) not in off for j in range(m)] for i in range(m)], bool)


def _apply_entry(name: str, solver: Any, table: PairTable, sourced: np.ndarray, particles: np.ndarray) -> str:
    pmat = table.particle_material.numpy()[particles] if particles.size else np.zeros(0, np.int32)
    mats = sorted({int(m) for m in np.unique(pmat) if m >= 0})
    model = solver.model
    smat = table.shape_material.numpy()
    flagged = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_PARTICLES)) > 0
    if not mats or not any(smat[s] >= 0 and sourced[mats, smat[s]].any() for s in np.flatnonzero(flagged)):
        return ""                                       # nothing sourced: Newton's values stand, untouched
    shapes = collider_shapes(solver)
    mu = _pair_mu(table, mats)
    friction = solver._mpm_model.collider.material_friction
    old = friction.numpy().copy()                   # a CPU array's numpy() is a live view
    new = old.copy()
    world = model.shape_world.numpy() if model.shape_world is not None else np.full(smat.size, -1)
    for mid, s in enumerate(shapes):
        if s >= 0 and smat[s] >= 0 and sourced[mats, smat[s]].any():
            new[mid] = _one_value(name, table, mats, int(smat[s]), sourced, mu, int(world[s]))
    friction.assign(new)
    return _note(name, table, smat, shapes, old, new)


def collider_shapes(solver: Any) -> np.ndarray:
    """Shape id behind each of ``solver``'s collider materials (-1: Newton's default material 0), in Newton's
    default-discovery order (``ImplicitMPMModel.setup_collider``): material 0, then each collider's collision
    shapes. Checked against Newton's own per-material friction and thickness, so a change in Newton's order raises
    instead of writing the wrong shape. Public Model fields only, besides the pinned ``_mpm_model`` (WP4's review):
    with separate worlds a static collider is one world's static shapes, and only then does a collider carry a
    world id >= 0 (where every static shape is global, both readings give the same shapes)."""
    model, col = solver.model, solver._mpm_model.collider
    bodies, worlds = col.collider_body_index.numpy(), col.collider_world.numpy()
    shape_world = model.shape_world.numpy() if model.shape_world is not None else None
    per_world = shape_world is not None and bool((worlds >= 0).any())
    flags = model.shape_flags.numpy()
    out = [-1]
    for body, world in zip(bodies, worlds, strict=True):
        ids = np.asarray(model.body_shapes[int(body)], dtype=np.int64)
        ids = ids[(flags[ids] & int(newton.ShapeFlags.COLLIDE_PARTICLES)) > 0]
        if int(body) == -1 and per_world:
            ids = ids[shape_world[ids] == int(world)]
        out.extend(int(s) for s in ids)
    shapes = np.asarray(out, dtype=np.int64)
    _check_order(shapes, col, model)
    return shapes


def _check_order(shapes: np.ndarray, col: Any, model: Any) -> None:
    friction, thickness = col.material_friction.numpy(), col.material_thickness.numpy()
    own = shapes[1:]
    ok = (friction.size == shapes.size and np.array_equal(friction[1:], model.shape_material_mu.numpy()[own]
                                                          .astype(friction.dtype))
          and np.array_equal(thickness[1:], model.shape_margin.numpy()[own].astype(thickness.dtype)))
    if not ok:
        raise AssemblyError(f"MPM collider materials ({friction.size}) do not follow their shapes ({shapes.size} "
                            "with Newton's default material) in Newton's discovery order, so pair friction cannot "
                            "be written. Fix: report it to WP12; meanwhile put the friction on the shape "
                            "(friction= or its material) instead of set_friction")


def _pair_mu(table: PairTable, mats: list[int]) -> np.ndarray:
    """[W or 1, len(mats), M] dynamic pair friction of each entry material against every table material."""
    wt = table.world_table
    full = wt.numpy()[..., 1] if wt is not None else table.table.numpy()[None, ..., 1]
    return full[:, mats, :]


def _one_value(name: str, table: PairTable, mats: list[int], sm: int, sourced: np.ndarray, mu: np.ndarray,
               world: int) -> float:
    """The one friction the collider shape of material ``sm`` can take for every material of the entry."""
    rows = mu[:, :, sm] if world < 0 or mu.shape[0] == 1 else mu[world:world + 1, :, sm]
    values = {float(v) for v in np.unique(rows)}
    if sourced[mats, sm].all() and len(values) == 1:
        return values.pop()
    who = ", ".join(f"{table.materials[m]} " + ("/".join(f"{v:g}" for v in sorted({float(x) for x in rows[:, i]}))
                                                if sourced[m, sm] else "(no pair)") for i, m in enumerate(mats))
    raise AssemblyError(f"MPM entry {name!r}: shape material {table.materials[sm]!r} pairs with {who} (pair "
                        "friction per world); one MPM solver gives a shape one friction for all its materials and "
                        "worlds. Fix: set_friction the same value for every material on that shape, or put the "
                        "friction on the shape (friction= or its material)")


def _note(name: str, table: PairTable, smat: np.ndarray, shapes: np.ndarray, old: np.ndarray,
          new: np.ndarray) -> str:
    moved = [i for i in range(shapes.size) if new[i] != old[i]]
    parts = []
    for sm in sorted({int(smat[shapes[i]]) for i in moved}):
        ids = [i for i in moved if int(smat[shapes[i]]) == sm]
        was, now = sorted({float(old[i]) for i in ids}), sorted({float(new[i]) for i in ids})
        parts.append(f"{table.materials[sm]} ({len(ids)} shapes) mu {'/'.join(f'{v:.3g}' for v in was)} -> "
                     f"{'/'.join(f'{v:.3g}' for v in now)}")
    line = (f"mpm entry {name}: collider pair friction (set_friction / pair cards): " + "; ".join(parts)
            if parts else f"mpm entry {name}: collider pair friction equals the shapes' own")
    log.info("%s", line)
    return line
