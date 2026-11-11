# SPDX-License-Identifier: Apache-2.0
"""Scoped noslip: MuJoCo's noslip post-solve on the frictional rows that touch a robot (SPEC §3.5 creep rule).

MuJoCo Warp's noslip (pinned 86e2fae0) builds a dense ``M^-1 J'`` over all ``njmax`` rows and runs one thread per
world over every frictional row. On T12 at 4,096 worlds on an L4 that costs 31 ms per step plus ~7 ms per sweep:
19x RT at the recorded 50 iterations against 185x without (docs/notes/wp3-results.md, NOSLIP rows). On T12 only the
rows that touch the robot need it: over the full 20 s replay the resting blocks stay at 0.100 mm with or without
noslip, while the held block (1.9 mm slip in the hand vs 0.37 mm) and the arm's dry joint friction (2 mm hand drift)
do not. :class:`ScopedNoslip` runs the same projected Gauss-Seidel (same rows, order, cone projection and stopping
rule) on those rows only, applies ``M^-1`` per kinematic tree, and allocates its scratch once. One launch per solve:
one warp per world selects the rows, factors only the trees they touch (one lane per tree up to 8 dofs, warp-wide
above) and sweeps (:mod:`alloy.contact.noslip_block`). Up to 64 slots and 64 friction rows the sweeps run on the
Delassus matrix of the scoped rows (:mod:`alloy.contact.noslip_dl`), else in qacc space. T12 at N = 1 on an L4, route
tolerance 1e-10 (SB-6 round 3): 0.10 ms per step at the trace start, 0.22 ms in the grasp (qacc space: 0.22 / 0.42);
T09's 93-dof cable tree is never factored by the pair scope.

It takes MuJoCo Warp's place: ``solver.solve`` calls the module-level ``noslip(m, d)`` when
``opt.noslip_iterations > 0`` (mujoco_warp@86e2fae0:mujoco_warp/_src/solver.py:4037-4050). :func:`install_noslip`
sets the options with ``newton_compat.enable_mujoco_noslip`` and routes that call for this solver's model to the
scoped version; other models keep MuJoCo Warp's.

Scopes: ``"robot"``: dof friction rows of trees with an actuated dof, elliptic contacts between a geom on such a tree
and one that is not (robot-robot contacts, e.g. an arm folding onto itself, keep the main solve's friction: noslip
drops the friction regularizer, and on such a contact it kicked T12's arm to 19-300 m/s, BUL-0010).
``"all"``: every dof friction row and elliptic contact (MuJoCo's noslip minus tendon friction rows). ``pairs`` adds
the contacts between given geom pairs to either scope (precision pairs, the §3.5 creep rule's noslip branch;
:func:`pair_geoms` maps a scene's ``PrecisionPairSpec`` shapes to MuJoCo geoms). ``surfaces`` adds the contacts of
given geoms with any geom outside that set (T09: the cable's links against the bench, not link against link), but not
their trees' dof friction rows; their trees are then factored (warp-wide above 8 dofs).
"""

from __future__ import annotations

import logging
import os
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import warp as wp
from alloy._vendor.mujoco_warp._src import solver as mjw_solver
from alloy._vendor.mujoco_warp._src import types as mjt

from alloy.contact import noslip_kernels as nk
from alloy.contact.noslip_block import LANES, solve_kernel
from alloy.contact.noslip_dl import KC, RMAX, SMAX
from alloy.core.errors import SolverOptionError
from alloy.core.newton_compat import enable_mujoco_noslip

log = logging.getLogger("alloy.contact")

ROBOT, ALL = "robot", "all"
_ACTUATED_TRN = (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)


@dataclass(frozen=True)
class NoslipSpec:
    iterations: int = 50
    tolerance: float = 1e-10
    scope: str = ROBOT
    rows_per_world: int | None = None  # slots per world (one per dof row or contact); None: 64 for "robot",
    # contacts + dofs for "all" or with surfaces. Rows beyond it keep the main solver's forces and flag overflow.
    pairs: tuple[tuple[int, int], ...] = ()  # MuJoCo geom pairs whose contacts are scoped too (pair_geoms)
    surfaces: tuple[int, ...] = ()  # MuJoCo geoms whose contacts with other geoms (not each other) are scoped too;
    # their trees' dof friction rows are not (shape_geoms; T09: the cable's links against the bench)
    restart: bool = False  # True: trees the rows touch restart qacc = qacc_smooth + M^-1 qfrc_constraint first, as
    # MuJoCo's noslip (the same result when the main solve converged). False: start from the main solve's qacc and
    # add only the noslip's own force changes, so a main solve that stopped short keeps its qacc (on T12, restarting
    # every tree reproduced BUL-0010's CPU kick, 19 mm / 19 m/s)
    warm_qcqp: bool = True  # condim-6 cone QCQPs start from the contact's previous-sweep multiplier after sweep 1
    # (same root, 1-2 factorizations instead of 5-20; False = MuJoCo's la = 0 start every sweep)
    force_space: bool = True  # sweep on the Delassus matrix of the scoped rows when they fit (noslip_dl: <= 64 slots
    # and friction rows), else in qacc space


@dataclass(frozen=True, eq=False)
class NoslipScope:
    geom: np.ndarray  # int32 [ngeom] bits: 1 = the geom's body is on a scoped tree, 2 = in a pair, 4 = a surface geom
    dof: np.ndarray  # int32 [nv], 1 = the dof's friction row is scoped
    max_tree_dofs: int
    max_friction_rows: int  # largest condim - 1 (at least 1)
    pairs: np.ndarray  # int32 [npair, 2] scoped geom pairs, (min, max)


def robot_trees(mjm: mujoco.MjModel) -> np.ndarray:
    """Kinematic trees with an actuated dof (joint transmissions, and body/site transmissions' bodies)."""
    trees = set()
    for a in range(mjm.nu):
        trn, target = int(mjm.actuator_trntype[a]), int(mjm.actuator_trnid[a, 0])
        if trn in _ACTUATED_TRN:
            trees.add(int(mjm.dof_treeid[mjm.jnt_dofadr[target]]))
        elif trn == mujoco.mjtTrn.mjTRN_SITE:
            trees.add(int(mjm.body_treeid[mjm.site_bodyid[target]]))
        elif trn == mujoco.mjtTrn.mjTRN_BODY:
            trees.add(int(mjm.body_treeid[target]))
    return np.array(sorted(t for t in trees if t >= 0), dtype=np.int64)


def build_scope(mjm: mujoco.MjModel, scope: str, pairs: Sequence[tuple[int, int]] = (),
                surfaces: Sequence[int] = ()) -> NoslipScope:
    if scope not in (ROBOT, ALL):
        raise SolverOptionError(f"noslip scope must be {ROBOT!r} or {ALL!r}, got {scope!r}")
    pair_arr = np.array(sorted({(min(a, b), max(a, b)) for a, b in pairs}), dtype=np.int32).reshape(-1, 2)
    surf = np.array(sorted(set(surfaces)), dtype=np.int64)
    for name, ids in (("pairs", pair_arr), ("surfaces", surf)):
        if ids.size and (ids.min() < 0 or ids.max() >= mjm.ngeom):
            raise SolverOptionError(f"noslip {name} must name geoms in [0, {mjm.ngeom}), got {ids.tolist()}")
    if scope == ALL:
        on_tree = np.ones(max(mjm.ntree, 1), dtype=bool)
    else:
        on_tree = np.zeros(max(mjm.ntree, 1), dtype=bool)
        on_tree[robot_trees(mjm)] = True
    body_tree = np.asarray(mjm.body_treeid)
    geom = np.where(body_tree[mjm.geom_bodyid] >= 0, on_tree[np.maximum(body_tree[mjm.geom_bodyid], 0)], False)
    geom = geom.astype(np.int32) * nk.ROBOT_BIT
    geom[pair_arr.ravel()] |= nk.PAIR_BIT
    geom[surf] |= nk.SURF_BIT
    dof = on_tree[np.asarray(mjm.dof_treeid)] if mjm.nv else np.zeros(0, bool)
    condim = max([int(mjm.geom_condim.max(initial=1))] + [int(c) for c in mjm.pair_dim])
    return NoslipScope(geom, dof.astype(np.int32), int(mjm.tree_dofnum.max(initial=1)), max(1, condim - 1), pair_arr)


def shape_geoms(solver: Any, shapes: Sequence[int], skip_missing: bool = False) -> tuple[int, ...]:
    """MuJoCo geoms of world-template shape ids (world 0 layout); shapes with no geom raise unless skipped."""
    starts = solver.model.shape_world_start.numpy()
    g2s = solver.mjc_geom_to_newton_shape.numpy()[0]
    s2g = {int(sh): g for g, sh in enumerate(g2s) if sh >= 0}
    out = []
    for s in shapes:
        g = s2g.get(int(starts[0]) + int(s))
        if g is None and not skip_missing:
            raise SolverOptionError(f"shape {s} has no MuJoCo geom")
        if g is not None:
            out.append(g)
    return tuple(out)


def pair_geoms(solver: Any, pairs: Sequence[Any]) -> tuple[tuple[int, int], ...]:
    """MuJoCo geom pairs of a scene's precision pairs (``PrecisionPairSpec``: template shape ids, world 0 layout)."""
    out = []
    for p in pairs:
        try:
            out.append(tuple(shape_geoms(solver, (p.shape_a, p.shape_b))))
        except SolverOptionError as e:
            raise SolverOptionError(f"precision pair ({p.shape_a}, {p.shape_b}) has a shape with no MuJoCo geom") from e
    return tuple(out)


class ScopedNoslip:
    """MuJoCo's noslip on the scoped rows of one Newton ``SolverMuJoCo`` (Warp path, elliptic cone, no sleeping)."""

    def __init__(self, solver: Any, spec: NoslipSpec = NoslipSpec()) -> None:
        if getattr(solver, "use_mujoco_cpu", False):
            raise SolverOptionError("scoped noslip runs on the MuJoCo Warp path only")
        self.stall = 0.0  # the stall stop's fraction, set at install (stall_fraction); 0 = off
        m, d = solver.mjw_model, solver.mjw_data
        _check_model(m, spec)
        self.solver, self.spec = solver, spec
        self.scope = build_scope(solver.mj_model, spec.scope, spec.pairs, spec.surfaces)
        self.maxt, self.maxf = self.scope.max_tree_dofs, self.scope.max_friction_rows
        self.maxsel = _slots(m, d, spec)
        self._alloc(m, d, solver.mjw_model.geom_bodyid.device)

    def _alloc(self, m: Any, d: Any, device: Any) -> None:
        nw, s, f, dd = d.nworld, self.maxsel, self.maxf, 2 * self.maxt
        ntree = max(int(m.ntree), 1)
        self.geom_scope = wp.array(self.scope.geom, dtype=int, device=device)
        self.dof_scope = wp.array(self.scope.dof, dtype=int, device=device)
        self.overflow = wp.zeros(nw, dtype=int, device=device)
        sc = nk.Scratch()
        sc.maxsel, sc.maxt, sc.maxf = s, self.maxt, self.maxf
        sc.restart = int(self.spec.restart)
        sc.count = wp.zeros(nw, dtype=int, device=device)
        sc.rows = wp.zeros((nw, s), dtype=int, device=device)
        sc.meta = wp.zeros((nw, s), dtype=nk.vec8i, device=device)
        sc.J = wp.zeros((nw, s * f * dd), dtype=float, device=device)
        sc.Y = wp.zeros((nw, s * f * dd), dtype=float, device=device)
        sc.A = wp.zeros((nw, s * f * f), dtype=float, device=device)
        sc.L = wp.zeros((nw, ntree * self.maxt * self.maxt), dtype=float, device=device)
        sc.V = wp.zeros((nw, ntree * self.maxt), dtype=float, device=device)
        sc.need = wp.zeros((nw, ntree), dtype=int, device=device)
        sc.lam = wp.zeros((nw, s), dtype=float, device=device)
        fs = self.spec.force_space
        sc.rowmap = wp.zeros((nw, RMAX if fs else 1), dtype=int, device=device)
        sc.C = wp.zeros((nw, RMAX * RMAX if fs else 1), dtype=float, device=device)
        sc.K = wp.zeros((nw, min(s, SMAX) * KC if fs else 1), dtype=float, device=device)
        self.scratch = sc
        self.pairs = wp.array(self.scope.pairs if len(self.scope.pairs) else np.zeros((0, 2), np.int32),
                              dtype=wp.vec2i, device=device)
        tr = nk.Trees()
        tr.dofadr, tr.dofnum, tr.of_dof = m.tree_dofadr, m.tree_dofnum, m.dof_treeid
        self.trees = tr
        lanes = LANES if wp.get_device(device).is_cuda else 1  # CPU tiles run one lane
        self.lanes = lanes
        self.kernel = solve_kernel(max(1, -(-int(m.nv) // lanes)), lanes, self.spec.warm_qcqp, self.spec.force_space)

    @property
    def count(self) -> wp.array:
        """Scoped slots per world in the last solve (dof friction rows + contacts)."""
        return self.scratch.count

    def install(self) -> ScopedNoslip:
        iterations = capped_sweeps(self.spec.iterations)
        self.stall = stall_fraction()
        enable_mujoco_noslip(self.solver, iterations, self.spec.tolerance)
        _route(self.solver.mjw_model, self)
        log.info("scoped noslip (%s, %d geom pairs, %d surface geoms): %d iterations, tolerance %g, %d slots per "
                 "world, trees <= %d dofs", self.spec.scope, len(self.scope.pairs), len(self.spec.surfaces),
                 iterations, self.spec.tolerance, self.maxsel, self.maxt)
        return self

    def uninstall(self) -> None:
        _ROUTES.pop(id(self.solver.mjw_model), None)
        if getattr(self.solver, SOLVER_ATTR, None) is self:
            delattr(self.solver, SOLVER_ATTR)
        enable_mujoco_noslip(self.solver, 0)

    def overflow_worlds(self) -> int:
        """Worlds that had more scoped rows than ``rows_per_world`` (the extra rows kept the main solver's forces)."""
        return int(self.overflow.numpy().sum())

    def run(self, m: Any, d: Any) -> None:
        """Called by MuJoCo Warp's ``solve`` in place of its ``noslip`` (graph-capturable: one fixed launch)."""
        wp.launch_tiled(self.kernel, dim=[d.nworld], block_dim=self.lanes, inputs=[
            d.njmax, int(m.opt.noslip_iterations), int(m.nv), int(m.ntree), d.nefc, m.opt.noslip_tolerance,
            float(self.stall),
            _rows(m, d), self._contacts(m, d), self.trees, _mass(m, d), self.scratch, _dofs(d),
            self.overflow, d.solver_niter])

    def _contacts(self, m: Any, d: Any) -> nk.Contacts:
        c, con = d.contact, nk.Contacts()
        con.dim, con.efc_address, con.geom, con.friction = c.dim, c.efc_address, c.geom, c.friction
        con.geom_bodyid, con.body_treeid = m.geom_bodyid, m.body_treeid
        con.geom_scope, con.dof_scope, con.pairs = self.geom_scope, self.dof_scope, self.pairs
        con.all_contacts = int(self.spec.scope == ALL)
        return con


def _rows(m: Any, d: Any) -> nk.Rows:
    e, r = d.efc, nk.Rows()
    r.sparse = int(bool(m.is_sparse))
    r.type, r.id, r.J_rownnz, r.J_rowadr, r.J_colind, r.J = e.type, e.id, e.J_rownnz, e.J_rowadr, e.J_colind, e.J
    r.D, r.aref, r.frictionloss, r.force = e.D, e.aref, e.frictionloss, e.force
    return r


def _mass(m: Any, d: Any) -> nk.Mass:
    ms = nk.Mass()
    ms.rownnz, ms.rowadr, ms.colind, ms.M = m.M_rownnz, m.M_rowadr, m.M_colind, d.M
    ms.qacc_smooth, ms.qfrc_smooth = d.qacc_smooth, d.qfrc_smooth
    return ms


def _dofs(d: Any) -> nk.Dofs:
    out = nk.Dofs()
    out.qacc, out.qfrc_constraint, out.Ma = d.qacc, d.qfrc_constraint, d.efc.Ma
    return out


SWEEPS_ENV = "ALLOY_NOSLIP_SWEEPS"
STALL_ENV = "ALLOY_NOSLIP_STALL"


def capped_sweeps(iterations: int) -> int:
    """``iterations``, or fewer when ``$ALLOY_NOSLIP_SWEEPS`` asks (opt-in, read at install).

    On a GPU the scoped noslip runs one warp per world, and the launch lasts as long as its slowest world. A world
    whose sweeps stop improving above the tolerance (the float32 floor) runs every sweep, so a few such worlds hold
    the whole batch: PickPlace-v0 at 2048 worlds with random actions, 15.5k env steps/s at 100 sweeps against 23.7k
    at 20, and 48k with nothing touched (one L4, 2026-11-11; contacts still cost the rest). At 20 sweeps PickPlace's
    end states stay within the GPU's run-to-run spread; a hard press differs more (p90 9.9e-3 against 9.0e-4). The
    cap never raises a scene's own value. Example: ``ALLOY_NOSLIP_SWEEPS=20 python train.py``."""
    raw = os.environ.get(SWEEPS_ENV, "").strip()
    if not raw:
        return iterations
    try:
        cap = int(raw)
    except ValueError:
        cap = 0
    if cap < 1:
        raise SolverOptionError(f"{SWEEPS_ENV}={raw!r}: give a whole number of noslip sweeps >= 1, or unset it")
    if cap < iterations:
        log.warning("scoped noslip: %d sweeps (%s), not %d", cap, SWEEPS_ENV, iterations)
    return min(cap, iterations)


def stall_fraction() -> float:
    """The stall stop's fraction from ``$ALLOY_NOSLIP_STALL`` (opt-in, read at install; 0 = off, the default).

    A world stops sweeping when one sweep lowers the cost by no more than this fraction of what all its sweeps so far
    did: a solve that still converges keeps going (its sweeps shrink geometrically, so the fraction is reached only
    near MuJoCo's own tolerance), while a world at the float32 floor, whose sweeps change the cost by rounding noise,
    stops instead of running every sweep and holding the batch (``capped_sweeps``). Unit-free, unlike the tolerance.
    Example: ``ALLOY_NOSLIP_STALL=1e-5``."""
    raw = os.environ.get(STALL_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0.0 < value < 1.0:
        raise SolverOptionError(f"{STALL_ENV}={raw!r}: give a fraction between 0 and 1 (e.g. 1e-5), or unset it")
    log.warning("scoped noslip: stall stop at %g of the cost decrease so far (%s)", value, STALL_ENV)
    return value


def install_noslip(solver: Any, spec: NoslipSpec = NoslipSpec()) -> ScopedNoslip:
    """Scoped noslip on ``solver`` (call after construction, before the first step or graph capture)."""
    return ScopedNoslip(solver, spec).install()


def _slots(m: Any, d: Any, spec: NoslipSpec) -> int:
    if spec.rows_per_world is not None:
        return spec.rows_per_world
    if spec.scope == ROBOT and not spec.surfaces:
        return 64
    return min(int(d.njmax), -(-int(d.naconmax) // int(d.nworld)) + int(m.nv))


def _check_model(m: Any, spec: NoslipSpec) -> None:
    if spec.iterations < 1 or (spec.rows_per_world is not None and spec.rows_per_world < 1):
        raise SolverOptionError(f"scoped noslip needs iterations >= 1 and rows_per_world >= 1, got {spec}")
    if int(m.opt.cone) != int(mjt.ConeType.ELLIPTIC):
        raise SolverOptionError("scoped noslip supports the elliptic cone only (use MuJoCo Warp's for pyramidal)")
    if int(m.opt.enableflags) & int(mjt.EnableBit.SLEEP):
        raise SolverOptionError("scoped noslip does not support sleeping (creep metrics run with sleeping off)")


# ---- routing MuJoCo Warp's noslip call ------------------------------------------------------------------------------
# id(model) -> (weakref to the model, weak proxy to its ScopedNoslip). The registry holds nothing strongly: the solver
# owns its ScopedNoslip (SOLVER_ATTR) and the entry is dropped when the model dies, so a dropped simulation is freed
# (WP8 2026-11-05: a strong entry kept every sim's solver, mjw model and data alive, ~0.42 GiB per 512-world build).
_ROUTES: dict[int, tuple[weakref.ref, Any]] = {}
_MJW_NOSLIP = mjw_solver.noslip
SOLVER_ATTR = "_alloy_scoped_noslip"


def _route(m: Any, inst: ScopedNoslip) -> None:
    key = id(m)
    setattr(inst.solver, SOLVER_ATTR, inst)
    _ROUTES[key] = (weakref.ref(m), weakref.proxy(inst))
    weakref.finalize(m, _ROUTES.pop, key, None)
    if mjw_solver.noslip is not _dispatch:
        mjw_solver.noslip = _dispatch


def _dispatch(m: Any, d: Any) -> None:
    entry = _ROUTES.get(id(m))
    if entry is not None and entry[0]() is m:
        try:
            entry[1].run(m, d)
            return
        except ReferenceError:   # its solver was dropped while the model lived on: MuJoCo Warp's own noslip
            _ROUTES.pop(id(m), None)
    _MJW_NOSLIP(m, d)


__all__ = ["ALL", "ROBOT", "STALL_ENV", "SWEEPS_ENV", "NoslipScope", "NoslipSpec", "ScopedNoslip", "build_scope", "capped_sweeps",
           "install_noslip", "pair_geoms", "robot_trees", "shape_geoms", "stall_fraction"]
