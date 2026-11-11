# SPDX-License-Identifier: Apache-2.0
"""Per-world MuJoCo C models of the CPU route (SCR-WP8-6).

Newton's ``SolverMuJoCo(use_mujoco_cpu=True, separate_worlds=True)`` compiles one template world (``mj_model``,
float64) and keeps every per-world value in its MuJoCo Warp model (``mjw_model``, CPU arrays, one row per world for the
fields in Newton's ``_MJW_BATCHED_MODEL_FIELDS`` and the expanded ``opt`` fields). Newton's own C path steps the
template only. The route needs one ``MjModel`` per world for multi-world scenes (A5's per-world gravity, A12's
per-world load): a copy of the template, with each batched field (and gravity / wind / magnetic / density /
viscosity) taken from the Warp model element by element only where that world differs from world 0, and gravity
from Newton's per-world ``model.gravity``. Uniform values keep the template's float64 values bit for bit, with what
the C-path plug-ins wrote there (tree weights, pair rows). Limits: Newton's C path writes joint-limit gains
(``jnt_solref``), tendon inverse weights and explicit pair rows into the template and the Warp model's row 0 only;
rows 1+ keep their pre-conversion values (a limited joint's stop at MuJoCo's default (0.02, 1) instead of Newton's
force-space gains, about 85x softer), so those fields follow the template in every world (``TEMPLATE_ONLY``), and
per-world joint-limit gains are reported as unsupported (WP15's review, 2026-11-06). Explicit pair friction is taken
per world from Newton's ``mujoco:pair_friction`` rows, and explicit pair stiffness from its ``mujoco:pair_solref``
rows (``world_pair_rows``): Newton's CPU mode skips its per-world pair update, so per-world friction draws
(``alloy.scene.randomize_mujoco``) and the pair refit's per-world rows (``alloy.contact.pair_fit``, drawn masses)
never reach the Warp model there.

Derived constants: Newton runs ``mj_setConst`` on the template only, so a world whose body masses differ (a per-world
density draw) inherits world 0's ``body_subtreemass``, which MuJoCo C's ``mj_comPos`` divides by: its subtree centres
of mass come out wrong and a spinning released body diverges (WP3's F5: 5 of 8 worlds INVALID at 28-704 m/s). Each
such world's ``body_subtreemass`` is summed again from its own masses (``subtree_mass``, ``mj_setConst``'s rule; world 0
keeps the template's bit for bit). Its other mass-derived constants (``body_invweight0``, ``dof_invweight0``,
``tendon_invweight0``, ``dof_M0``, ``stat.meaninertia``) come from ``mj_setConst`` on a copy
(``alloy.route.mjc_core_consts``), as the Warp route's come from Newton's ``set_const_0`` per world.

Then the CPU-route solver settings (:class:`RouteSettings`), the light-body rows at the route's step, and the noslip
switch: with Alloy's scoped noslip the model's own ``opt.noslip_iterations`` is 0 (the two never run together).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

log = logging.getLogger("alloy.route")

ROUTE_TOLERANCE = 1.0e-9
_OPT_FIELDS = ("wind", "magnetic", "density", "viscosity")
# fields Newton's C path converts for world 0 only (rows 1+ of the Warp model are stale): the template is the truth
TEMPLATE_ONLY = ("jnt_solref", "tendon_invweight0")
PAIR_ROWS = ("pair_friction", "pair_solref")       # read from Newton's per-world rows (world_pair_rows)


@dataclass(frozen=True)
class RouteSettings:
    """The CPU route's MuJoCo C solver settings (SPIKE.md §1, coordinator ruling 2026-11-06 11:19Z): warm start off
    and ``opt.tolerance`` at most 1e-9. With either missing MuJoCo C's Newton solve stops short on A5 (warm start: a
    period-2 cycle that creeps 1.66e-3 m per 10 s; tolerance 1e-8: 4.1 % isotropy spread). The GPU route keeps the
    scene's options."""

    warmstart: bool = False
    tolerance: float = ROUTE_TOLERANCE

    def apply(self, m: mujoco.MjModel) -> None:
        if not self.warmstart:
            m.opt.disableflags = int(m.opt.disableflags) | int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        m.opt.tolerance = min(float(m.opt.tolerance), self.tolerance)


def batched_fields(solver: Any, nworld: int) -> dict[str, np.ndarray]:
    """{field: [nworld, ...] values} of the Warp model's per-world fields that differ between worlds."""
    from alloy._vendor.newton._src.solvers.mujoco.solver_mujoco import _MJW_BATCHED_MODEL_FIELDS

    out: dict[str, np.ndarray] = {}
    if nworld <= 1:
        return out
    mjw = solver.mjw_model
    names = [(f, getattr(mjw, f, None)) for f in _MJW_BATCHED_MODEL_FIELDS]
    names += [(f"opt.{f}", getattr(mjw.opt, f, None)) for f in _OPT_FIELDS]
    for name, arr in names:
        if name in TEMPLATE_ONLY or arr is None or not hasattr(arr, "numpy") or arr.shape[0] != nworld:
            continue
        host = arr.numpy()
        if host.size and not np.all(host == host[:1]):
            out[name] = host
    return out


def world_models(solver: Any, nworld: int, settings: RouteSettings, scoped_noslip: bool) -> list[mujoco.MjModel]:
    """One MjModel per world (module docstring)."""
    tmpl = solver.mj_model
    diff = batched_fields(solver, nworld)
    for name in PAIR_ROWS:
        rows = world_pair_rows(solver, nworld, name)
        if rows is not None:
            diff[name] = rows
    _check_limit_gains(solver, nworld)
    if diff:
        log.info("CPU route: %d worlds with per-world %s", nworld, sorted(diff))
    gravity = world_gravity(solver, nworld)
    models = []
    for w in range(nworld):
        m = copy.copy(tmpl)
        for name, host in diff.items():
            _write_field(m, name, host[w], host[w] != host[0])
        if "body_mass" in diff and np.any(diff["body_mass"][w] != diff["body_mass"][0]):
            m.body_subtreemass[:] = subtree_mass(m)
        if gravity is not None:
            m.opt.gravity[:] = gravity[w]
        settings.apply(m)
        if scoped_noslip:
            m.opt.noslip_iterations = 0
        models.append(m)
    return models


def subtree_mass(m: mujoco.MjModel) -> np.ndarray:
    """``[nbody]`` each body's mass plus its descendants', as ``mj_setConst`` sums it: children into parents from the
    last body down (MuJoCo numbers a parent before its children)."""
    sub = np.array(m.body_mass, dtype=np.float64)
    parent = np.asarray(m.body_parentid)
    for b in range(m.nbody - 1, 0, -1):
        sub[parent[b]] += sub[b]
    return sub


def world_pair_rows(solver: Any, nworld: int, name: str) -> np.ndarray | None:
    """[nworld, npair, ...] explicit pair values ``name`` (``pair_friction``, ``pair_solref``) from Newton's per-world
    ``mujoco:<name>`` rows (row ``world * pairs_per_world + pair``, as Newton's GPU pair update reads them), in MuJoCo's
    pair order (``alloy.contact.mjw_pairs.pair_order``, as the pair refit publishes the template's), or None when there
    are no pairs or every world matches world 0."""
    from alloy.contact.mjw_pairs import pair_order

    npair = int(solver.mj_model.npair)
    attrs = getattr(solver.model, "mujoco", None)
    rows = getattr(attrs, name, None) if attrs is not None else None
    if nworld <= 1 or npair == 0 or rows is None:
        return None
    per_world = len(rows) // int(solver.model.world_count)
    if per_world < npair:
        return None
    host = np.asarray(rows.numpy(), np.float64)[: nworld * per_world].reshape(nworld, per_world, -1)
    order = pair_order(solver, per_world)
    host = host[:, order] if order is not None and len(order) == npair else host[:, :npair]
    return None if np.all(host == host[:1]) else host


def _check_limit_gains(solver: Any, nworld: int) -> None:
    """Warn when Newton's joint-limit gains differ between worlds: the C path converts world 0's only."""
    ke, kd = solver.model.joint_limit_ke, solver.model.joint_limit_kd
    if nworld <= 1 or ke is None or kd is None:
        return
    dof = np.asarray(solver.mjc_jnt_to_newton_dof.numpy()).reshape(nworld, -1)
    if dof.size == 0:
        return
    gains = [np.stack([g.numpy()[np.maximum(dof[w], 0)] for g in (ke, kd)]) for w in range(nworld)]
    if any(not np.array_equal(gains[w], gains[0]) for w in range(1, nworld)):
        log.warning("CPU route: joint-limit gains differ between worlds; Newton's C path converts world 0's only, "
                    "so every world uses world 0's limit stiffness")


def world_gravity(solver: Any, nworld: int) -> np.ndarray | None:
    """[nworld, 3] gravity from Newton's model when worlds differ (Newton's C path writes world 0's into the template
    only, never into the Warp model), else None."""
    g = np.asarray(solver.model.gravity.numpy(), dtype=np.float64).reshape(-1, 3)[:nworld]   # [world_count + 1]
    if nworld <= 1 or g.shape[0] != nworld or np.all(g == g[:1]):
        return None
    return g


def _write_field(m: mujoco.MjModel, name: str, value: np.ndarray, differs: np.ndarray) -> None:
    """Write ``value`` where ``differs`` (this world's Warp row against world 0's)."""
    if not np.any(differs):
        return
    if name.startswith("opt."):
        attr = name[4:]
        cur = getattr(m.opt, attr)
        if np.ndim(cur):
            cur[:] = np.where(np.reshape(differs, np.shape(cur)), np.reshape(value, np.shape(cur)), cur)
        else:
            setattr(m.opt, attr, float(np.asarray(value).reshape(-1)[0]))
        return
    dst = getattr(m, name)
    mask = np.reshape(differs, dst.shape)
    dst[mask] = np.reshape(np.asarray(value), dst.shape)[mask]


@dataclass
class LightRows:
    """The light-body rule's rows (``alloy.contact.light_body``) on the route's models: the solver's C-path rule
    (``_RuledMujoco``) rewrites the template's solref rows before every ``mj_step``; the route writes them into each
    world model when its step changes, from that world's own original rows (taken when the route builds its models,
    so per-world pair rows stay that world's) and its own body masses (a world drawn heavier keeps the file's row)."""

    rows: list[Any]
    rk4: bool
    orig: list[list[np.ndarray]] = field(default_factory=list)     # [world][row table]: the rows before the rule
    dt: float | None = None
    models: tuple[int, ...] = ()                                     # id() of the models ``orig`` was taken from

    @classmethod
    def of(cls, solver: Any) -> LightRows | None:
        rule = getattr(solver, "_mujoco", None)
        rows = getattr(rule, "_rows", None)
        if not rows or getattr(rule, "_model", None) is not solver.mj_model:
            return None
        return cls(list(rows), bool(getattr(rule, "_rk4", False)))

    def apply(self, models: list[mujoco.MjModel], dt: float) -> None:
        ids = tuple(id(m) for m in models)
        if self.dt == dt and self.models == ids:
            return
        from alloy.contact.light_body import light_solref

        if self.models != ids:                   # new (rebuilt) world models: their rows are not ruled yet
            self.orig = [[np.array(getattr(m, r.table)[r.idx], dtype=np.float64) for r in self.rows] for m in models]
            self.models = ids
        for m, before in zip(models, self.orig, strict=True):
            for r, rows in zip(self.rows, before, strict=True):
                getattr(m, r.table)[r.idx] = light_solref(rows, r.masses(m.body_mass), r.dmax, float(dt), self.rk4)
        self.dt = dt


def body_maps(solver: Any, nworld: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per world: (MuJoCo body id per Newton body row of that world, the rows' global Newton body ids)."""
    model = solver.model
    bw = model.body_world.numpy() if model.body_count else np.zeros(0, np.int32)
    m2n = np.asarray(solver.mjc_body_to_newton.numpy()).reshape(nworld, -1)
    maps, globs = [], []
    for w in range(nworld):
        glob = np.flatnonzero(bw == w).astype(np.int64)
        inv = {int(n): b for b, n in enumerate(m2n[w]) if n >= 0}
        maps.append(np.array([inv.get(int(g), -1) for g in glob], dtype=np.int32))
        globs.append(glob)
    return maps, globs


__all__ = ["ROUTE_TOLERANCE", "LightRows", "RouteSettings", "batched_fields", "body_maps", "subtree_mass",
           "world_models"]
