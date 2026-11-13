# SPDX-License-Identifier: Apache-2.0
"""Per-world, per-object friction for VBD's particle-shape contacts.

Newton's SolverVBD mixes one scalar for every particle (``model.soft_contact_mu``) with the shape's own friction:
mu = sqrt(soft_contact_mu * shape_material_mu[shape]), cached per contact record at each collision detection. The
shape side is per world already (a randomized rigid object's shapes carry their draw). The particle side was not: a
cloth or soft body with ``Scene.randomize(name, "friction", lo, hi)`` slid at ``lo`` in every world (the scalar held
the card's value, capped at the lowest draw), a pair friction from ``Scene.set_friction`` never reached the contact,
and a second deformable took the heaviest one's value.

``PairMu`` holds what the scalar cannot: each material's friction per world [W, M] (a deformable's own card value,
or its world's draw) and the scene's own pair frictions [M, M] (-1: none). ``write`` runs after the solver's contact
refresh and rewrites the cached value of every record: the scene pair's friction when there is one, else
sqrt(mu[world, particle material] * shape_material_mu[shape]), Newton's rule with the particle's own value.

It is installed only for a scene that needs it (``wanted``): a deformable with a randomized surface value, or a scene
pair that names a deformable's material. Every other scene keeps the stock path, bit for bit.

Not covered: friction between two cloth layers and between a sheet and itself still reads the one scalar
(docs/known_gaps/cloth-friction-per-world.md).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def write_contact_mu(count: wp.array[wp.int32], indices: wp.array[wp.vec3i], shape: wp.array[wp.int32],
                     particle_world: wp.array[wp.int32], particle_material: wp.array[wp.int32],
                     shape_material: wp.array[wp.int32], mu_world: wp.array2d[wp.float32],
                     pair_mu: wp.array2d[wp.float32], shape_mu: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    if i >= count[0]:
        return
    p = indices[i][0]
    s = shape[i]
    if p < 0 or s < 0:
        return
    mi = particle_material[p]
    if mi < 0:
        return
    mj = shape_material[s]
    if mj >= 0:
        if pair_mu[mi, mj] >= 0.0:
            out[i] = pair_mu[mi, mj]
            return
    w = wp.clamp(particle_world[p], 0, mu_world.shape[0] - 1)
    out[i] = wp.sqrt(mu_world[w, mi] * shape_mu[s])


def _deformables(plan: Any) -> list[Any]:
    from alloy.scene.finish import DEFORMABLE_ROUTES  # noqa: PLC0415

    return [t for t in plan.objects if t.obj is not None and t.decision.route in DEFORMABLE_ROUTES]


def _surface_rows(t: Any) -> bool:
    return any(str(r.field).startswith("material.surface.") for r in getattr(t.placement, "randomize", ()) or ())


def wanted(built: Any) -> bool:
    """Whether the scene needs per-contact friction on its deformables (module docstring)."""
    plan = getattr(built, "_plan", None)
    if plan is None or getattr(built, "pair_table", None) is None:
        return False
    defs = _deformables(plan)
    if not defs or built.model.particle_count == 0:
        return False
    names = {t.obj.material.name for t in defs}
    paired = any(p.a in names or p.b in names for p in plan.scene.pairs)
    return paired or any(_surface_rows(t) for t in defs)


@dataclasses.dataclass
class PairMu:
    """Device tables for ``write_contact_mu`` (module docstring)."""

    built: Any
    mu_world: wp.array          # [W, M] float32
    pair_mu: wp.array           # [M, M] float32, -1 = no scene pair

    @staticmethod
    def build(built: Any) -> PairMu:
        from alloy.scene.randomize import _entries  # noqa: PLC0415

        plan, model = built._plan, built.model
        entries = _entries(plan)
        index = {e.card.name: i for i, e in enumerate(entries)}
        pair = np.full((len(entries), len(entries)), -1.0, np.float32)
        for p in plan.scene.pairs:
            if p.a in index and p.b in index:
                pair[index[p.a], index[p.b]] = pair[index[p.b], index[p.a]] = float(p.mu_dynamic.value)
        out = PairMu(built, wp.zeros((int(built.world_count), len(entries)), dtype=wp.float32, device=model.device),
                     wp.array(pair, dtype=wp.float32, device=model.device))
        out.refresh()
        return out

    def refresh(self, world_mask: Any = None) -> None:
        """Take each deformable's friction per world from the draws the built scene holds now (build, and after a
        reset that redrew them)."""
        from alloy.scene.randomize import _entries, drawn_surface, surface_draws  # noqa: PLC0415

        built, plan = self.built, self.built._plan
        entries = _entries(plan)
        host = np.zeros(self.mu_world.shape, np.float32)
        for i, e in enumerate(entries):
            surface = getattr(e.card, "surface", None)
            host[:, i] = float(surface.mu_dynamic.value) if surface is not None else 0.0
        keys = [e.key for e in entries]
        for t in _deformables(plan):
            i = keys.index(t.material_key)
            for w in range(host.shape[0]):
                draws = surface_draws(built, t.name, w)
                if draws:
                    host[w, i] = float(drawn_surface(entries[i].card, draws).surface.mu_dynamic.value)
        self.mu_world.assign(host)

    def write(self, solver: Any, contacts: Any) -> None:
        """Rewrite the solver's cached friction of every particle-shape record (after its own contact refresh)."""
        out = solver.body_particle_contact_material_mu
        if contacts is None or out.shape[0] == 0 or contacts.soft_contact_max == 0:
            return
        table, model = self.built.pair_table, solver.model
        wp.launch(write_contact_mu, dim=min(int(out.shape[0]), int(contacts.soft_contact_max)), device=solver.device,
                  inputs=[contacts.soft_contact_count, contacts.soft_contact_indices, contacts.soft_contact_shape,
                          model.particle_world, table.particle_material, table.shape_material, self.mu_world,
                          self.pair_mu, model.shape_material_mu],
                  outputs=[out])


def install(sim: Any) -> PairMu | None:
    """Give every Alloy VBD solver of ``sim`` the scene's ``PairMu`` when the scene needs one; None otherwise."""
    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415
    from alloy.solvers.vbd.base import SolverVBDAlloy  # noqa: PLC0415

    solvers = [s for s in entry_solvers(sim.stepper).values() if isinstance(s, SolverVBDAlloy)]
    if not solvers or not wanted(sim.built):
        return None
    pm = PairMu.build(sim.built)
    for s in solvers:
        s.alloy_pair_mu = pm
    sim.reset_hooks.append(pm.refresh)
    return pm


__all__ = ["PairMu", "install", "wanted", "write_contact_mu"]
