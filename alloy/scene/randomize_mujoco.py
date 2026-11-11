# SPDX-License-Identifier: Apache-2.0
"""Per-world surface draws into MuJoCo's own contact values (``alloy.scene.randomize`` surface fields).

MuJoCo's native contacts do not read Alloy's pair table: a contact takes its explicit pair row's friction
(``mujoco:pair_*``, rows per world copy, ``alloy.scene.mujoco_pairs``) or, without one, the larger of its two geoms'
friction (``shape_material_mu``). A per-world surface draw therefore also writes, for each drawn world:

- the object's shapes' ``shape_material_mu`` (the drawn card's dynamic friction): contacts with robot links, whose
  MJCF friction stays authoritative, follow MuJoCo's max rule as at the template;
- every pair row of that world that touches one of the object's shapes: ``pair_friction`` = (μ_d, μ_d, lever,
  rolling, rolling) from the world's pair table (objects with a surface draw always get rows: ``mujoco_pairs``).

Newton hands both to MuJoCo when the solver is built (MuJoCo Warp per world; the CPU route's per-world models).
Restitution and contact-stiffness draws reach the pair table (Newton contacts) only: a MuJoCo pair row keeps the
template's solref, and the build report says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

FRICTION_LEAVES = ("mu_static", "mu_dynamic")


def write_mujoco_surface(built: Any, plan: Any, name: str, leaf: str, worlds: Sequence[int], mu: np.ndarray) -> None:
    """Write object ``name``'s draw of ``leaf`` into MuJoCo's values (module docstring); ``mu[k]``: the drawn card's
    dynamic friction in ``worlds[k]``."""
    if leaf not in FRICTION_LEAVES:
        _note_solref(built, name, leaf)
        return
    from alloy.scene.randomize import _gids  # noqa: PLC0415

    m, h = built.model, built.handles[name]
    own = _gids(h.template.shape, len(h.world_stride.shape), worlds)
    if own.size:
        host = m.shape_material_mu.numpy()
        for k in range(len(worlds)):
            host[own[k]] = mu[k]
        m.shape_material_mu.assign(host)
    _pair_rows(built, plan, worlds, own)


def _pair_rows(built: Any, plan: Any, worlds: Sequence[int], own: np.ndarray) -> None:
    mj = getattr(built.model, "mujoco", None)
    rows = getattr(mj, "pair_friction", None) if mj is not None else None
    if rows is None or not len(rows):
        return
    table = built.pair_table
    wt = table.world_table.numpy()
    pw, g1, g2 = (np.asarray(getattr(mj, a).numpy()) for a in ("pair_world", "pair_geom1", "pair_geom2"))
    fr = rows.numpy()
    for k, w in enumerate(worlds):
        mat = _shape_materials(built, plan, table, w)
        mine = set(own[k].tolist())
        for r in np.flatnonzero(pw == w):
            a, b = int(g1[r]), int(g2[r])
            if (a in mine or b in mine) and a in mat and b in mat:
                mu_d, lever = float(wt[w, mat[a], mat[b]][1]), float(wt[w, mat[a], mat[b]][3])
                fr[r][0] = fr[r][1] = mu_d
                fr[r][2] = lever
    rows.assign(fr)


def _shape_materials(built: Any, plan: Any, table: Any, w: int) -> dict[int, int]:
    """{global shape id in world ``w``: pair-table material index} over the plan's objects."""
    from alloy.scene.randomize import _gids  # noqa: PLC0415

    out = {}
    for t in plan.objects:
        h = built.handles.get(t.name)
        if t.obj is None or h is None or not len(h.template.shape):
            continue
        i = table.material_index(t.material_key)
        for g in _gids(h.template.shape, len(h.world_stride.shape), [w])[0].tolist():
            out[int(g)] = i
    return out


def _note_solref(built: Any, name: str, leaf: str) -> None:
    mj = getattr(built.model, "mujoco", None)
    if mj is None or getattr(mj, "pair_solref", None) is None or not len(mj.pair_solref):
        return
    note = (f"{name}.material.surface.{leaf}", "per-world restitution / contact stiffness reach the pair table "
            "(Newton contacts) only; MuJoCo's native pair rows keep the template's solref")
    if note not in built.report.unsupported:
        built.report.unsupported.append(note)


__all__ = ["FRICTION_LEAVES", "write_mujoco_surface"]
