# SPDX-License-Identifier: Apache-2.0
"""``alloy.build(scene, contact_capacity=n)``: MuJoCo's three buffers from one number, and what a run used of them.

Sizes (WP13 plan item 1, SPEED-FIX B1), on every MuJoCo entry:
- contacts: ``nconmax`` = n per world; MuJoCo Warp keeps one pool of n x worlds (``naconmax``) that the worlds share.
- constraint rows: ``njmax`` = r x 2 n + R per world (``POOL_SHARE`` = 2: the pool is shared, so one world may hold
  more than n contacts; rows cover a world at twice its share, the headroom the old 6 rows per contact gave). r is
  the rows one contact makes: the largest condim c of any geom
  or explicit pair, c rows on the elliptic cone, max(1, 2 (c - 1)) on the pyramidal one. R is the scene's own rows,
  each at its maximum (every limit active, every equality on): connect 3, weld 6, joint and tendon equalities 1, one
  per dof or tendon with friction loss, one per limited hinge, slide or ball joint and limited tendon (MuJoCo Warp
  makes one row per limited joint). Contacts added beyond n (the precision append, Newton's start-state raise) get
  max(r, 4) rows each, 4 being what ``alloy.integration.wiring`` reserves per appended contact. A model with flex
  equalities is not counted: it keeps the entry's own rows.
- CCD scratch: ``naccdmax`` stays at ``naconmax`` (MuJoCo Warp's default), so it follows n. MuJoCo Warp takes at
  most one CCD slot per broadphase pair and drops pairs past ``naconmax`` first (BROADPHASE), so this pool cannot
  overflow before the contact pool does.
R needs the compiled MjModel, which exists only inside SolverMuJoCo's constructor; ``rows_from_contacts`` wraps
``mujoco_warp.put_data`` (Newton calls it once per solver, at construction) for the engine build and sets njmax
there. The entry kwarg keeps its old value: Newton compares it with the start state's rows before put_data.
Overflows fail closed in health: contacts and CCD as CONTACT_OVERFLOW, rows (a world that took more than n contacts
from the shared pool) as ROW_OVERFLOW, from MuJoCo Warp's per-world overflow word (no host read while stepping).

``watch_capacity(sim)`` reads what a run used: per-world peak contacts and rows, and the pooled broadphase pairs.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.api")

APPENDED_ROWS = 4          # rows alloy.integration.wiring reserves per appended precision contact (ROWS_PER_CONTACT)
POOL_SHARE = 2             # rows cover a world holding this many times its share n of the shared contact pool
_WRAP = threading.RLock()  # one sized build at a time per process: the put_data wrap is module state
EQ_ROWS = {"mjEQ_CONNECT": 3, "mjEQ_WELD": 6, "mjEQ_JOINT": 1, "mjEQ_TENDON": 1}


def contact_rows(mjm: Any) -> int:
    """Rows one contact makes in ``mjm`` (a mujoco.MjModel): its largest condim on the model's friction cone."""
    import mujoco  # noqa: PLC0415

    dims = [np.asarray(mjm.geom_condim), np.asarray(mjm.pair_dim)]
    c = max([1] + [int(d.max()) for d in dims if d.size])
    if int(mjm.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        return c
    return max(1, 2 * (c - 1))


def model_rows(mjm: Any) -> int | None:
    """The non-contact rows ``mjm`` can make in one world at once (module docstring); None when it has flex
    equalities or flex objects (not counted). Friction loss is read from the compiled template world: a world given
    friction loss later (per-world model values) can pass R, which health reads as ROW_OVERFLOW."""
    import mujoco  # noqa: PLC0415

    eq = np.asarray(mjm.eq_type).reshape(-1)
    by_type = {int(getattr(mujoco.mjtEq, k)): v for k, v in EQ_ROWS.items()}
    if int(getattr(mjm, "nflex", 0)) or any(int(t) not in by_type for t in eq):
        return None
    J = mujoco.mjtJoint
    limited = np.asarray(mjm.jnt_limited).astype(bool) & np.isin(
        np.asarray(mjm.jnt_type), [int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE), int(J.mjJNT_BALL)])
    friction = int((np.asarray(mjm.dof_frictionloss) > 0).sum()) + int((np.asarray(mjm.tendon_frictionloss) > 0).sum())
    return (sum(by_type[int(t)] for t in eq) + friction + int(limited.sum())
            + int(np.asarray(mjm.tendon_limited).astype(bool).sum()))


def njmax_for(mjm: Any, capacity: int, nconmax: int) -> int | None:
    """Rows per world for ``capacity`` contacts per world in ``mjm`` (``nconmax``: what Newton passes, which can be
    larger: appended or start-state contacts); None when ``model_rows`` does not count this model."""
    own = model_rows(mjm)
    if own is None:
        return None
    r = contact_rows(mjm)
    return r * POOL_SHARE * capacity + max(r, APPENDED_ROWS) * max(0, int(nconmax) - capacity) + own


@contextlib.contextmanager
def rows_from_contacts(capacity: int | None) -> Iterator[None]:
    """While the engine builds, MuJoCo Warp Data gets ``njmax_for`` rows (``capacity`` None: nothing changes). It
    swaps the module function for the block, so one build at a time per process (``alloy.build`` is one)."""
    if capacity is None:
        yield
        return
    from alloy._vendor import mujoco_warp  # noqa: PLC0415

    owner = threading.get_ident()
    live = [True]

    def put_data(mjm: Any, mjd: Any, *args: Any, nconmax: int | None = None, njmax: int | None = None,
                 njmax_nnz: int | None = None, **kw: Any) -> Any:
        mine = live[0] and threading.get_ident() == owner   # off after the build; another thread's build: untouched
        rows = njmax_for(mjm, capacity, capacity if nconmax is None else nconmax) if mine else None
        if rows is not None and njmax is not None:
            rows = max(rows, int(mjd.nefc))                 # put_data's own floor: the start state's rows
            njmax_nnz = _sparse_nnz(mjm, nconmax, rows, njmax_nnz)
            log.info("contact capacity %d: njmax %d -> %d (%d rows per contact + %d of the model's own)", capacity,
                     njmax, rows, contact_rows(mjm), model_rows(mjm))
            njmax = rows
        return stock(mjm, mjd, *args, nconmax=nconmax, njmax=njmax, njmax_nnz=njmax_nnz, **kw)

    with _WRAP:                                             # a second thread's sized build waits for this one
        stock = mujoco_warp.put_data
        functools.update_wrapper(put_data, stock)           # keeps mjw_chol / mjw_block's install markers
        mujoco_warp.put_data = put_data
        try:
            yield
        finally:
            live[0] = False                                 # whatever still holds the wrapper: it passes through
            if mujoco_warp.put_data is put_data:            # restore only what is ours (an install or uninstall
                mujoco_warp.put_data = stock                # inside the window keeps its own function)


def _sparse_nnz(mjm: Any, nconmax: int | None, njmax: int, given: int | None) -> int | None:
    """Newton's sparse Jacobian capacity (solver_mujoco.py, njmax_nnz) at ``njmax`` rows, never above ``given``."""
    if given is None:
        return None
    from alloy._vendor.mujoco_warp._src.io import _default_njmax_nnz, is_sparse  # noqa: PLC0415

    if not is_sparse(mjm) or nconmax is None:
        return given
    own = _default_njmax_nnz(mjm, int(nconmax), int(njmax)) + int(mjm.nv) + int(np.asarray(mjm.ten_J_rownnz).sum())
    return min(int(given), int(own))


# ---- what a run used (B2's reading; notes: [internal note]) -----------------------
@wp.kernel
def _count_contacts(nacon: wp.array[wp.int32], worldid: wp.array[wp.int32], count: wp.array[wp.int32]):
    i = wp.tid()
    if i < nacon[0]:
        w = worldid[i]
        if w >= 0 and w < count.shape[0]:
            wp.atomic_add(count, w, 1)


@wp.kernel
def _keep_peaks(count: wp.array[wp.int32], nefc: wp.array[wp.int32], nacon: wp.array[wp.int32],
                ncollision: wp.array[wp.int32], contacts: wp.array[wp.int32], rows: wp.array[wp.int32],
                pooled: wp.array[wp.int32]):
    w = wp.tid()
    contacts[w] = wp.max(contacts[w], count[w])
    count[w] = 0
    rows[w] = wp.max(rows[w], nefc[w])
    if w == 0:
        pooled[0] = wp.max(pooled[0], nacon[0])
        pooled[1] = wp.max(pooled[1], ncollision[0])


class CapacityWatch:
    """Peaks of one MuJoCo solver's buffers since ``watch_capacity`` (``read``). MuJoCo Warp: kept on the device after
    every physics step (graph safe; nothing is read back until ``read``). The CPU's MuJoCo C route: MjData's own
    ``maxuse_con`` / ``maxuse_efc`` (every step; no broadphase buffer there, so ``candidates`` is None)."""

    def __init__(self, solver: Any) -> None:
        self.solver = solver
        self.c_route = hasattr(solver, "finalize_route")             # alloy.route.mjc_core.MjcCoreSolver
        d = solver.mjw_data
        n, dev = int(d.nworld), d.nefc.device
        self._count = wp.zeros(n, dtype=wp.int32, device=dev)
        self._contacts = wp.zeros(n, dtype=wp.int32, device=dev)
        self._rows = wp.zeros(n, dtype=wp.int32, device=dev)
        self._pooled = wp.zeros(2, dtype=wp.int32, device=dev)

    def update(self) -> None:
        """Fold the current step's counts into the peaks (the step wrapper calls it after every physics step)."""
        d = self.solver.mjw_data
        dev = d.nefc.device
        wp.launch(_count_contacts, dim=int(d.naconmax), inputs=[d.nacon, d.contact.worldid, self._count], device=dev)
        wp.launch(_keep_peaks, dim=int(d.nworld), inputs=[self._count, d.nefc, d.nacon, d.ncollision, self._contacts,
                                                          self._rows, self._pooled], device=dev)

    def read(self) -> dict[str, Any]:
        """``{"route", "worlds", "caps", "contacts", "rows", "candidates", "pooled_contacts"}`` (README)."""
        d = self.solver.mjw_data
        caps = {"nconmax": int(d.naconmax) // max(int(d.nworld), 1), "naconmax": int(d.naconmax),
                "njmax": int(d.njmax), "naccdmax": int(getattr(d, "naccdmax", d.naconmax))}
        out: dict[str, Any] = {"route": "mujoco_c" if self.c_route else "mujoco_warp", "worlds": int(d.nworld),
                               "caps": caps}
        if self.c_route:
            datas = getattr(self.solver, "datas", None) or []
            return {**out, "contacts": [int(x.maxuse_con) for x in datas], "rows": [int(x.maxuse_efc) for x in datas],
                    "candidates": None, "pooled_contacts": None}
        pooled = self._pooled.numpy()
        return {**out, "contacts": self._contacts.numpy().tolist(), "rows": self._rows.numpy().tolist(),
                "candidates": int(pooled[1]), "pooled_contacts": int(pooled[0])}


def mujoco_solver(sim: Any) -> Any:
    """``sim``'s MuJoCo entry solver (an ``alloy.Sim`` or an engine simulation); LookupError when it has none."""
    from alloy._vendor import newton  # noqa: PLC0415

    engine = getattr(sim, "engine", sim)
    for s in getattr(engine, "solvers", ()):
        if isinstance(s, newton.solvers.SolverMuJoCo) and getattr(s, "mjw_data", None) is not None:
            return s
    raise LookupError("this simulation has no MuJoCo entry with MuJoCo Warp data")


def watch_capacity(sim: Any) -> CapacityWatch:
    """Start reading ``sim``'s MuJoCo buffer use. Call it after ``alloy.build`` and before the first step (on a GPU
    the step graph is captured then, and the watch must be inside it). Example::

        use = watch_capacity(sim); sim.step(100); print(use.read()["rows"])"""
    solver = mujoco_solver(sim)
    watch = CapacityWatch(solver)
    if watch.c_route:
        return watch
    prev = solver.step

    def step(*args: Any, **kwargs: Any) -> Any:
        out = prev(*args, **kwargs)
        watch.update()
        return out

    solver.step = step
    return watch
