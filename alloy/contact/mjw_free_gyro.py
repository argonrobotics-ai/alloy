# SPDX-License-Identifier: Apache-2.0
"""One-pass constraint guard for MuJoCo Warp's discrete-integrator free-body gyroscopic term (a run-time replacement).

Under the DISCRETE integrator (every scene with a CHAIN cable), ``forward._launch_discrete_free_gyro`` launches
``_discrete_free_gyro`` over (world, free body). Each thread first checks MuJoCo's decoupling guard
(``mjd_freeGyroPossible``): the body is skipped when one of its 6 dofs is touched by an efm stiffness row
(``efm_K_rownnz``), a constraint row (a nonzero ``efc_J`` entry), an active tendon (``efm_ts`` != 0) or actuator
(``efm_as`` != 0), or a flex passive contact (``efm_con_scale`` != 0). It scanned every constraint row of its world,
and every nonzero of each, for each body: O(free bodies x rows x nonzeros) per physics step. On one L4, 64 worlds of
200 resting boxes beside a 1-dof arm and a chain cable, that kernel took 221 of 1,050 ms of kernel time over 20
control periods (about 1.4 ms a launch), the largest single share of the chain's cost.

Here one pass marks the dofs those rows touch (``touched``, per world and dof: one thread per row, tendon, actuator or
contact; every write stores 1, so the order does not matter), and the body reads its 6 marks. The guard is the same
boolean and the gyroscopic solve is MuJoCo Warp's, copied from the pinned source, so ``out`` is bit for bit the stock
kernel's (``tests/contact/test_mjw_free_gyro.py``).

Pin: the source hashes of the two functions it replaces (``PINNED``, at the vendored mujoco_warp) are checked at
``install``; a change keeps the stock launcher, with one warning, as ``alloy.precision.mjw_chol`` does. The hashes are
of the function objects held at import (the stock launcher before any replacement), the same hash as
``alloy.precision.mjw_pin.source_hash`` without importing ``alloy.precision``, whose import installs this module: a
name looked up after that install would hash the replacement. ``ALLOY_MJW_FREE_GYRO=0`` keeps the stock launcher (A/B
runs).
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
from typing import Any

import warp as wp

from alloy._vendor import _warp as _alloy_warp
from alloy._vendor._warp_calls import upstream
from alloy._vendor.mujoco_warp._src import forward
from alloy._vendor.mujoco_warp._src import math
from alloy._vendor.mujoco_warp._src import passive
from alloy._vendor.mujoco_warp._src.types import DisableBit
from alloy._vendor.mujoco_warp._src.types import EnableBit
from alloy._vendor.mujoco_warp._src.types import IntegratorType
from alloy._vendor.mujoco_warp._src.types import mat66
from alloy._vendor.mujoco_warp._src.types import vec6
from alloy._vendor.mujoco_warp._src.types import vec10
from alloy._vendor.mujoco_warp._src.warp_util import cache_kernel

log = logging.getLogger(__name__)

ENV = "ALLOY_MJW_FREE_GYRO"     # "0" keeps MuJoCo Warp's own launcher
PINNED = {"forward._discrete_free_gyro_kernel": "c1ebf49f227d670b",     # mujoco_warp's source, ``mjw_pin.source_hash``
          "forward._launch_discrete_free_gyro": "53cde5644f1fabd5"}
_STATE: dict[str, Any] = {"original": getattr(forward, "_launch_discrete_free_gyro", None),   # stock, at import
                          "pin_ok": None}
_PINNED_OBJECTS = {"forward._discrete_free_gyro_kernel": getattr(forward, "_discrete_free_gyro_kernel", None),
                   "forward._launch_discrete_free_gyro": _STATE["original"]}       # None (moved): "missing"


@_alloy_warp.kernel(module="unique")
def _mark_efc_sparse(nefc_in: wp.array[int], efc_J_rownnz_in: wp.array2d[int], efc_J_rowadr_in: wp.array2d[int],
                     efc_J_colind_in: wp.array3d[int], efc_J_in: wp.array3d[float], touched: wp.array2d[int]):
    worldid, i = wp.tid()
    if i >= nefc_in[worldid]:
        return
    radr = efc_J_rowadr_in[worldid, i]
    for j in range(efc_J_rownnz_in[worldid, i]):
        if efc_J_in[worldid, 0, radr + j] != 0.0:
            touched[worldid, efc_J_colind_in[worldid, 0, radr + j]] = 1


@_alloy_warp.kernel(module="unique")
def _mark_efc_dense(nv: int, nefc_in: wp.array[int], efc_J_in: wp.array3d[float], touched: wp.array2d[int]):
    worldid, i = wp.tid()
    if i >= nefc_in[worldid]:
        return
    for c in range(nv):
        if efc_J_in[worldid, i, c] != 0.0:
            touched[worldid, c] = 1


@_alloy_warp.kernel(module="unique")
def _mark_rows(rownnz: wp.array[int], rowadr: wp.array[int], colind: wp.array[int], active_in: wp.array2d[float],
               touched: wp.array2d[int]):
    """Tendons: a model-level Jacobian pattern, active where ``efm_ts`` is nonzero."""
    worldid, t = wp.tid()
    if active_in[worldid, t] == 0.0:
        return
    radr = rowadr[t]
    for j in range(rownnz[t]):
        touched[worldid, colind[radr + j]] = 1


@_alloy_warp.kernel(module="unique")
def _mark_moments(moment_rownnz_in: wp.array2d[int], moment_rowadr_in: wp.array2d[int],
                  moment_colind_in: wp.array2d[int], efm_as_in: wp.array2d[float], touched: wp.array2d[int]):
    worldid, a = wp.tid()
    if efm_as_in[worldid, a] == 0.0:
        return
    radr = moment_rowadr_in[worldid, a]
    for j in range(moment_rownnz_in[worldid, a]):
        touched[worldid, moment_colind_in[worldid, radr + j]] = 1


@_alloy_warp.kernel(module="unique")
def _mark_contacts(nacon_in: wp.array[int], contact_worldid_in: wp.array[int], efm_con_dof_in: wp.array2d[int],
                   efm_con_scale_in: wp.array[float], efm_con_nnz_in: wp.array[int], touched: wp.array2d[int]):
    cid = wp.tid()
    if cid >= nacon_in[0] or efm_con_scale_in[cid] == 0.0:
        return
    worldid = contact_worldid_in[cid]
    for j in range(efm_con_nnz_in[cid]):
        touched[worldid, efm_con_dof_in[cid, j]] = 1


@cache_kernel
def _free_gyro_kernel(has_fluid: bool, is_inverse: bool):
    """MuJoCo Warp's ``_discrete_free_gyro`` (PINNED), its row scans replaced by the body's 6 ``touched`` marks."""

    @_alloy_warp.kernel(module="unique")
    def _free_gyro(
        opt_timestep: wp.array[float],
        opt_enableflags: int,
        body_dofadr: wp.array[int],
        dof_treeid: wp.array[int],
        M_rownnz: wp.array[int],
        M_rowadr: wp.array[int],
        M_colind: wp.array[int],
        efm_K_rownnz: wp.array[int],
        body_freeadr: wp.array[int],
        qvel_in: wp.array2d[float],
        qacc_in: wp.array2d[float],
        xpos_in: wp.array2d[wp.vec3],
        xmat_in: wp.array2d[wp.mat33],
        subtree_com_in: wp.array2d[wp.vec3],
        crb_in: wp.array2d[vec10],
        M_in: wp.array2d[float],
        tree_awake_in: wp.array2d[int],
        qfrc_smooth_in: wp.array2d[float],
        efm_c_in: wp.array2d[float],
        efm_diag_in: wp.array2d[float],
        efm_fluid_in: wp.array2d[float],
        efm_ca_in: wp.array2d[float],
        touched_in: wp.array2d[int],
        out: wp.array2d[float],
    ):
        worldid, freeid = wp.tid()
        bodyid = body_freeadr[freeid]
        dof_adr = body_dofadr[bodyid]

        treeid = dof_treeid[dof_adr]
        if (opt_enableflags & EnableBit.SLEEP) and (tree_awake_in[worldid, treeid] == 0):
            return

        for r in range(6):
            if efm_K_rownnz[dof_adr + r] != 0:
                return
        for r in range(6):
            if touched_in[worldid, dof_adr + r] != 0:
                return

        timestep = opt_timestep[worldid % opt_timestep.shape[0]]
        crb = crb_in[worldid, bodyid]
        subtree_mass = crb[9]
        Iw = wp.mat33(
            crb[0],
            crb[3],
            crb[4],
            crb[3],
            crb[1],
            crb[5],
            crb[4],
            crb[5],
            crb[2],
        )
        qvel_rot = wp.vec3(
            qvel_in[worldid, dof_adr + 3],
            qvel_in[worldid, dof_adr + 4],
            qvel_in[worldid, dof_adr + 5],
        )

        A = mat66(0.0)
        for r in range(6):
            row = dof_adr + r
            start = M_rowadr[row]
            nnz = M_rownnz[row]
            for k in range(nnz):
                c = M_colind[start + k] - dof_adr
                val = M_in[worldid, start + k]
                A[r, c] = val
                A[c, r] = val
            A[r, r] += efm_diag_in[worldid, row]

        if wp.static(has_fluid):
            for r in range(6):
                row = dof_adr + r
                start = M_rowadr[row]
                nnz = M_rownnz[row]
                for k in range(nnz):
                    c = M_colind[start + k] - dof_adr
                    val = efm_fluid_in[worldid, start + k]
                    A[r, c] += val
                    if c != r:
                        A[c, r] += val

        s = subtree_com_in[worldid, bodyid] - xpos_in[worldid, bodyid]
        lin, rot = math.free_bias_vel_blocks(subtree_mass, xmat_in[worldid, bodyid], Iw, s, qvel_rot)
        h_mass = -timestep * subtree_mass
        for r in range(3):
            for c in range(3):
                A[r, 3 + c] += h_mass * lin[r, c]
                A[3 + r, 3 + c] += timestep * rot[r, c]

        if wp.static(is_inverse):
            for r in range(6):
                row_sum = float(0.0)
                for c in range(6):
                    row_sum += A[r, c] * qacc_in[worldid, dof_adr + c]
                out[worldid, dof_adr + r] = row_sum
        else:
            A_fact, pivot, ok = math.lu_factor_6x6(A)
            if ok:
                b_vec = vec6(0.0)
                for r in range(6):
                    dof = dof_adr + r
                    b_vec[r] = qfrc_smooth_in[worldid, dof] + efm_c_in[worldid, dof] + efm_ca_in[worldid, dof]
                x_vec = math.lu_solve_6x6(A_fact, pivot, b_vec)
                for r in range(6):
                    out[worldid, dof_adr + r] = x_vec[r]

    return _free_gyro


def touched(m: Any, d: Any) -> wp.array:
    """(nworld, nv) int: 1 where a constraint row, an active tendon or actuator, or a flex passive contact touches the
    dof (the rows MuJoCo Warp's guard scans), else 0."""
    dev = d.qacc.device          # named: a caller outside the solver's ScopedDevice gets the data's device anyway
    out = wp.zeros((d.nworld, m.nv), dtype=int, device=dev)
    if m.is_sparse:
        wp.launch(_mark_efc_sparse, dim=(d.nworld, d.efc.J_rownnz.shape[1]),
                  inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J], outputs=[out], device=dev)
    else:
        wp.launch(_mark_efc_dense, dim=(d.nworld, d.efc.J.shape[1]), inputs=[m.nv, d.nefc, d.efc.J], outputs=[out],
                  device=dev)
    if m.ntendon:
        wp.launch(_mark_rows, dim=(d.nworld, m.ntendon),
                  inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.efm_ts], outputs=[out], device=dev)
    if m.nactuator:
        wp.launch(_mark_moments, dim=(d.nworld, m.nactuator),
                  inputs=[d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.efm_as], outputs=[out], device=dev)
    if m.has_flex_passive and m.body_freeadr.size > 0:
        efm_con_dof, _, efm_con_scale, _, efm_con_nnz = passive.build_efm_contact(m, d)
        if efm_con_scale.shape[0]:
            wp.launch(_mark_contacts, dim=efm_con_scale.shape[0],
                      inputs=[d.nacon, d.contact.worldid, efm_con_dof, efm_con_scale, efm_con_nnz], outputs=[out],
                      device=dev)
    return out


def launch(m: Any, d: Any, out: wp.array2d[float], qacc: wp.array2d[float], is_inverse: bool) -> None:
    """``forward._launch_discrete_free_gyro`` with the one-pass guard (same conditions, same ``out``)."""
    if m.opt.integrator != IntegratorType.DISCRETE or not m.flex_interp_assemblable or m.has_non_simple_flex:
        return
    if m.body_freeadr.size == 0:
        return
    off = m.opt.disableflags
    has_fluid = m.has_fluid and not ((off & DisableBit.SPRING) and (off & DisableBit.DAMPER))
    mask = touched(m, d)
    wp.launch(
        _free_gyro_kernel(has_fluid, is_inverse),
        dim=(d.nworld, m.body_freeadr.size),
        device=out.device,
        inputs=[m.opt.timestep, m.opt.enableflags, m.body_dofadr, m.dof_treeid, m.M_rownnz, m.M_rowadr, m.M_colind,
                m.efm_K_rownnz, m.body_freeadr, d.qvel, qacc, d.xpos, d.xmat, d.subtree_com, d.crb, d.M, d.tree_awake,
                d.qfrc_smooth, d.efm_c, d.efm_diag, d.efm_fluid, d.efm_ca, mask],
        outputs=[out],
    )


def _source_hash(fn: Any) -> str:
    """``alloy.precision.mjw_pin.source_hash`` of a function object (the vendored call rewrite undone)."""
    try:
        src = upstream(inspect.getsource(inspect.unwrap(fn)))
    except (OSError, TypeError):
        return "missing"
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def pin_ok() -> bool:
    """Whether mujoco_warp's two replaced functions still match the source this module copies (checked once)."""
    if _STATE["pin_ok"] is None:
        moved = [n for n, h in PINNED.items() if _source_hash(_PINNED_OBJECTS[n]) != h]
        _STATE["pin_ok"] = not moved
        if moved:
            log.warning("mjw_free_gyro: mujoco_warp differs in %s; the one-pass free-body guard is not installed",
                        ", ".join(moved))
    return bool(_STATE["pin_ok"])


def install() -> bool:
    """Replace ``forward._launch_discrete_free_gyro`` (idempotent; ``inverse`` calls it through the module too). False
    (stock launcher kept) when switched off (ENV) or when the pinned sources moved."""
    if os.environ.get(ENV, "1") == "0" or not pin_ok():
        return False
    if forward._launch_discrete_free_gyro is not launch:
        forward._launch_discrete_free_gyro = launch
        log.info("mjw_free_gyro: one-pass constraint guard for MuJoCo Warp's discrete free-body gyroscopic term")
    return True


def uninstall() -> None:
    if _STATE["original"] is not None:
        forward._launch_discrete_free_gyro = _STATE["original"]


def stock() -> Any:
    """MuJoCo Warp's own launcher (for A/B laws), whether or not the replacement is installed."""
    return _STATE["original"]


__all__ = ["ENV", "PINNED", "install", "launch", "pin_ok", "stock", "touched", "uninstall"]
