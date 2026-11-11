# SPDX-License-Identifier: Apache-2.0
"""Finds the film solver a ``Bonds`` plug-in attaches to, and lays out its bonds (host side, once)."""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.plugins.errors import PluginError

from .adhesives import adhesive
from .layout import (box_distance, film_cells, flat_layout, nearest_grip, rest_directions, shape_distance, to_body,
                     vertex_areas, vertex_spacing, world_ranges)

FREE, SUBSTRATE, GRIP = 0, 1, 2


def film_solver(stepper: Any, particles: int) -> Any:
    """The stepper's film entry solver (a SolverVBDAlloy, e.g. SolverFilm): exactly one, owning every particle."""
    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415 - keeps `import alloy.plugins` light
    from alloy.solvers.vbd.base import SolverVBDAlloy  # noqa: PLC0415

    found = [s for s in entry_solvers(stepper).values()
             if isinstance(s, SolverVBDAlloy) and int(s.model.particle_count) > 0]
    if len(found) != 1:
        raise PluginError(f"bonds: needs exactly one film entry (SolverFilm, route cloth_film), found {len(found)} "
                          f"in {type(stepper).__name__}")
    if float(getattr(found[0], "rigid_contact_k_start_value", -1.0)) >= 0.0:
        raise PluginError("bonds: the film solver ramps its body-particle contact penalty (rigid_avbd_linear_beta "
                          "> 0); formation reads pressure from the fixed contact stiffness: keep the beta at 0 "
                          "(Newton's default; ramping is deprecated)")
    if int(found[0].model.particle_count) != particles:
        raise PluginError(f"bonds: the film entry owns {found[0].model.particle_count} of {particles} particles; it "
                          f"must own them all (one film entry per scene in v1)")
    return found[0]


def per_world(ctx: Any, label: str) -> np.ndarray:
    ids = np.asarray(ctx.shapes(label).numpy(), np.int64)
    if np.any(ids < 0):
        raise PluginError(f"bonds: shape {label!r} is missing in worlds {np.flatnonzero(ids < 0).tolist()}")
    return ids


def _tables(plugin: Any, m: Any) -> tuple[dict, dict]:
    """Per shape: formation (substrate, p_form, dwell, contact ke, margin) and peel (g90, g180, tau)."""
    S, cfg = int(m.shape_count), plugin.config
    sf = {"substrate": np.zeros(S, np.int32), "p_form": np.zeros(S), "dwell": np.zeros(S),
          "ke": 0.5 * (float(m.soft_contact_ke) + m.shape_material_ke.numpy()), "margin": np.zeros(S)}
    margin = getattr(m, "shape_margin", None)
    if margin is not None and margin.shape[0] == S:
        sf["margin"] = margin.numpy().astype(np.float64)
    pl = {"g90": np.zeros(S), "g180": np.zeros(S), "tau": np.ones(S)}
    for label, card in zip(cfg.substrates, cfg.adhesives):
        ad = adhesive(card)
        for s in per_world(plugin.ctx, label):
            sf["substrate"][s], sf["p_form"][s], sf["dwell"][s] = 1, ad.p_form, ad.dwell
            pl["g90"][s], pl["g180"][s], pl["tau"][s] = ad.g90, ad.g180, ad.tau
    return sf, pl


def _grip(plugin: Any, m: Any, bq: np.ndarray, x: np.ndarray, start: np.ndarray, out: dict) -> np.ndarray:
    """Vertices within ``grip_reach`` of each world's grip box are held by it (anchored where they are)."""
    held = np.zeros(len(x), bool)
    if not plugin.config.grip:
        return held
    shapes = per_world(plugin.ctx, plugin.config.grip)
    local = _local(shapes, out["base"])
    for w, s in enumerate(shapes):
        lo, hi = int(start[w]), int(start[w + 1])
        dist, _ = box_distance(m, bq, int(s), x[lo:hi])
        sel = lo + np.flatnonzero(dist <= plugin.config.grip_reach)
        held[sel] = True
        out["bond_state"][sel], out["bond_shape"][sel] = GRIP, local[w]
        if sel.size == 0:
            raise PluginError(f"bonds: no film vertex within {plugin.config.grip_reach} m of grip "
                              f"{plugin.config.grip!r} in world {w}")
    out["anchor"][held] = to_body(m, bq, out["bond_shape"][held] + out["pbase"][held], x[held], point=True)
    return held


def _prebond(plugin: Any, m: Any, bq: np.ndarray, x: np.ndarray, start: np.ndarray, out: dict,
             radius: np.ndarray, margin: np.ndarray) -> None:
    """Free vertices resting on a substrate (a box, cylinder or capsule; within contact radius + margin +
    ``prebond_reach``) start bonded, anchored where they rest."""
    for label in plugin.config.substrates:
        ids = per_world(plugin.ctx, label)
        _local(ids, out["base"])
        for w, s in enumerate(ids):
            lo, hi = int(start[w]), int(start[w + 1])
            dist, nrm = shape_distance(m, bq, int(s), x[lo:hi])
            ok = (dist <= radius[lo:hi] + margin[s] + plugin.config.prebond_reach) & (out["bond_state"][lo:hi] == FREE)
            sel = lo + np.flatnonzero(ok)
            out["bond_state"][sel], out["bond_shape"][sel] = SUBSTRATE, s - out["base"][w]
            shp = np.full(sel.size, s)
            out["anchor"][sel] = to_body(m, bq, shp, x[sel], point=True)
            out["normal"][sel] = to_body(m, bq, shp, nrm[ok], point=False)
            out["bond_dir"][sel] = to_body(m, bq, shp, out["rest_dir"][sel], point=False)


def shape_bases(m: Any, worlds: int) -> np.ndarray:
    """[W] each world's first shape. Per-world shapes are stored world-local (shape - base[w]) in plug-in state, so
    world 0's state copied to another world means the same shapes there (the harness's identical-worlds checks)."""
    sw = m.shape_world.numpy()
    base = np.array([int(np.flatnonzero(sw == w).min()) if np.any(sw == w) else 0 for w in range(worlds)], np.int32)
    return base


def _local(ids: np.ndarray, base: np.ndarray) -> np.ndarray:
    """World w's shape ids [W] to world-local; they must agree across worlds (replicated scenes)."""
    loc = ids - base
    if np.any(loc != loc[0]):
        raise PluginError(f"bonds: shape ids {ids.tolist()} are not the same shape in every world (world bases "
                          f"{base.tolist()}): build the worlds by replicating one template")
    return loc


def layout(plugin: Any, solver: Any) -> dict[str, Any]:
    """Areas, spacing, grips, pre-bonds and nearest-grip distances; writes the plug-in's bond arrays."""
    ctx, view, m = plugin.ctx, solver.model, plugin.ctx.model
    x = view.particle_q.numpy().astype(np.float64)
    tris = view.tri_indices.numpy()
    start = world_ranges(view.particle_world.numpy(), ctx.world_count)
    rest = flat_layout(x, tris, view.tri_poses.numpy(), start)        # material layout (a wrapped label: unfolded)
    area, h = vertex_areas(rest, tris), vertex_spacing(rest, tris)
    bq = m.body_q.numpy().astype(np.float64) if m.body_count else np.zeros((0, 7))
    n = len(x)
    out = {"bond_state": np.zeros(n, np.int32), "bond_shape": np.full(n, -1, np.int32),
           "anchor": np.zeros((n, 3)), "normal": np.zeros((n, 3)), "bond_dir": np.zeros((n, 3)),
           "base": shape_bases(m, ctx.world_count)}
    out["pbase"] = out["base"][view.particle_world.numpy()]
    held = _grip(plugin, m, bq, x, start, out)
    d, gnear = nearest_grip(rest, held, start)
    out["rest_dir"] = rest_directions(x, gnear, d if rest is x else np.linalg.norm(x - x[gnear.clip(0)], axis=1))
    surface, peel = _tables(plugin, m)
    if plugin.config.prebond:
        _prebond(plugin, m, bq, x, start, out, view.particle_radius.numpy(), surface["margin"])
    for name in ("bond_state", "bond_shape", "anchor", "normal", "bond_dir"):
        plugin._flat[name].assign(out[name].astype(plugin._flat[name].numpy().dtype))
    body_of = m.shape_body.numpy()
    mass = m.body_mass.numpy() if m.body_count else np.zeros(0)
    gs = per_world(ctx, plugin.config.grip) if plugin.config.grip else np.full(ctx.world_count, -1)
    grip_mass = np.array([mass[body_of[s]] if s >= 0 and body_of[s] >= 0 else 0.0 for s in gs])
    return {"area": area, "h": h, "d": d, "gnear": gnear, "start": start, "rest_dir": out["rest_dir"],
            "cell": film_cells(area, tris), "k": plugin.config.stiffness * area, "surface": surface,
            "peel": peel, "grip_mass": grip_mass, "base": out["base"]}


__all__ = ["film_solver", "layout", "per_world"]
