# SPDX-License-Identifier: Apache-2.0
"""The CPU route's mass-derived constants per world (closes the known gap cpu-route-world-mass-softness, 2026-11-07).

MuJoCo derives the constraints' softness and its solver's scale from the masses at ``qpos0``: ``body_invweight0``,
``dof_invweight0``, ``tendon_invweight0`` (contact, limit and tendon weights), ``dof_M0`` and ``stat.meaninertia``
(the solver's stopping scale up to MuJoCo 3.16; 3.17 scales by the current trace(M)). Newton's C path runs
``mj_setConst`` on the template (world 0) only. Without this
module, every world of the route kept world 0's values, and so did Newton's per-world Warp rows that Alloy's pair
refit reads (``alloy.contact.pair_fit``). A world whose density is drawn per world then landed with world 0's
softness, and in CPU mode A a worker's world 0 is its block's first world, so such worlds differed with the process
count (12 mm after 0.5 s for a dropped box).

``world_constants`` runs ``mj_setConst`` on a copy of the template carrying each world's own per-world fields, for
every world whose mass fields (``MASS_FIELDS``) differ from world 0's. ``publish`` writes the weights into the Warp
model's per-world rows (float32, what the pair refit and contact conversion read). ``apply`` puts the float64 values
into that world's C model. A world with world 0's masses keeps the template's values bit for bit. That is the Warp
route's rule (Newton's ``set_const_0`` per world).

What else ``mj_setConst`` changes on a mass-drawn copy (WP2's review; an arm with a dampratio servo and the Panda with
a heavier link): ``actuator_acc0`` and ``stat.meanmass``, both taken too. MuJoCo 3.17's ``mj_setConst`` no longer
re-derives ``stat`` (its compiler does), so ``stat_from_masses`` sets both stat values as the compiler defines them:
trace(M(qpos0)) / nv and the bodies' mean mass (the world body excluded). It leaves ``actuator_biasprm`` alone: a
position servo's dampratio was turned into a damping gain once, at the template's compile, from world 0's masses. So
a robot link drawn at another mass keeps world 0's servo damping on this route. ``Scene.randomize`` draws no robot
masses.
"""

from __future__ import annotations

import copy
from typing import Any

import mujoco
import numpy as np

from alloy.route.mjc_core_models import _write_field, batched_fields

MASS_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_armature")
CONST_FIELDS = ("body_invweight0", "dof_invweight0", "tendon_invweight0", "dof_M0", "actuator_acc0")
STAT_FIELDS = ("meaninertia", "meanmass")      # stat values derived from the masses (stat_from_masses)
PUBLISHED = ("body_invweight0", "dof_invweight0", "tendon_invweight0")   # Newton's per-world Warp rows


def mass_worlds(diff: dict[str, np.ndarray], nworld: int) -> list[int]:
    """Worlds (1+) whose mass fields differ from world 0's."""
    fields = [diff[f] for f in MASS_FIELDS if f in diff]
    return [w for w in range(1, nworld) if any(np.any(h[w] != h[0]) for h in fields)]


def world_constants(solver: Any, nworld: int) -> dict[int, dict[str, Any]]:
    """{world: its constants} for every world whose masses differ from world 0's (module docstring)."""
    diff = batched_fields(solver, nworld)
    out: dict[int, dict[str, Any]] = {}
    for w in mass_worlds(diff, nworld):
        m = copy.copy(solver.mj_model)
        for name, host in diff.items():
            if name not in CONST_FIELDS:
                _write_field(m, name, host[w], host[w] != host[0])
        mujoco.mj_setConst(m, mujoco.MjData(m))
        stat_from_masses(m)
        out[w] = {**{f: np.array(getattr(m, f), copy=True) for f in CONST_FIELDS},
                  **{f"stat.{f}": float(getattr(m.stat, f)) for f in STAT_FIELDS}}
    return out


def stat_from_masses(m: mujoco.MjModel) -> None:
    """``m.stat.meaninertia`` = trace(M(qpos0)) / nv and ``m.stat.meanmass`` = the mean body mass (world body excluded),
    as MuJoCo's compiler derives them (3.17's ``mj_setConst`` leaves ``stat`` as compiled). The trace matches the
    compiler's to about 1e-10 relative; the mean mass exactly."""
    d = mujoco.MjData(m)                          # qpos = qpos0
    mujoco.mj_fwdPosition(m, d)
    if m.nv:
        m.stat.meaninertia = float(np.asarray(d.M)[np.asarray(m.M_rowadr) + np.asarray(m.M_rownnz) - 1].sum()) / m.nv
    if m.nbody > 1:
        m.stat.meanmass = float(np.asarray(m.body_mass)[1:].mean())


def publish(solver: Any, consts: dict[int, dict[str, Any]], nworld: int) -> None:
    """Every world's weights into the Warp model's per-world rows: its own where it has constants, else the
    template's (a world whose masses went back to world 0's loses its old ones)."""
    tmpl = solver.mj_model
    for f in PUBLISHED:
        arr = getattr(solver.mjw_model, f, None)
        if arr is None or not hasattr(arr, "numpy") or arr.shape[0] != nworld or not np.size(getattr(tmpl, f)):
            continue
        host = arr.numpy()
        for w in range(nworld):
            src = consts[w][f] if w in consts else getattr(tmpl, f)
            host[w] = np.reshape(src, host[w].shape)
        arr.assign(host)


def apply(m: mujoco.MjModel, consts: dict[str, Any] | None) -> None:
    """A world's own constants into its C model (float64, as ``mj_setConst`` gave them)."""
    if not consts:
        return
    for f in CONST_FIELDS:
        getattr(m, f)[...] = consts[f]
    for f in STAT_FIELDS:
        setattr(m.stat, f, consts[f"stat.{f}"])


__all__ = ["CONST_FIELDS", "MASS_FIELDS", "PUBLISHED", "STAT_FIELDS", "apply", "mass_worlds", "publish",
           "stat_from_masses", "world_constants"]
