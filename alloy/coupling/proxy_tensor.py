# SPDX-License-Identifier: Apache-2.0
"""Per-body proxy mass tensors for a VBD destination (known gap coupling-proxy-isotropic-mass, phase 1).

A MuJoCo body's proxy in VBD has one scalar mass, trace(W) / 3 of W = J M^-1 J^T, the translational block at the
body's centre of mass at qpos0. The tensor keeps the whole W: a body on one slider is light along its slide and
stiff across it, as MuJoCo has it.

Per proxy body: Minv_body = s * R0^T W R0 (body frame: the frame Newton's body_q refers to is MuJoCo's body frame,
``xmat``, not the inertial frame), with s = installed inverse mass on the destination / w0, w0 the inverse mass
Newton's scalar install reads for that body (MuJoCo Warp's body_invweight0[0], else the C model's). s carries
``mass_scale`` and per-world masses; for a body whose w0 is trace(W)/3 this is trace(Minv_body)/3 == installed, and a
simple slide root (w0 = 1/m) gets Minv = W. Outside s in [1e-3, 1e3] the body falls back to identity. Eigenvalues are
floored at lambda_max / kappa so M = inverse(Minv) exists. Bodies that are not proxies, or whose W is zero, get
identity * inv_mass. A destination whose proxies are all isotropic (free bodies) keeps the scalar kernels, so the
switch changes nothing there, bit for bit. Off by default: ``ALLOY_PROXY_TENSOR=1`` or
``Scheduler(proxy_tensor=True)``.

W is the one at qpos0 and turns with the body. That is right for a body whose mobile axis is its own (a hinge
tip) and for one that does not turn; it is wrong in direction when the axis is fixed in an ancestor and the body turns
against it (a pole on a sliding cart): there the proxy can be up to kappa times too heavy along the cart's slide
(pinned in tests/coupling/test_proxy_tensor_host.py).

The W shape is taken from the MuJoCo model (world 0's masses); each world's scale comes from its own installed inverse
mass. That is exact while worlds differ by a uniform mass scale; ``world_shapes_agree`` warns when they do not.
Gravity enters VBD's forward step as the proxy's weight through the tensor, which the coupler's rewind cancels as
it does with the scalar mass. The translation-rotation cross block of the 6x6 W is dropped.
Checked on the CPU only (no GPU or CUDA-graph run yet); a destination model with tendons keeps the scalar kernels.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from alloy.core.log import get_logger

log = get_logger("coupling.proxy_tensor")

PROXY_TENSOR_KAPPA = 100.0           # largest allowed ratio of a proxy's mobile to stiffest direction
ISOTROPIC_RTOL = 1.0e-5              # tensors this close to identity * inv_mass keep the scalar kernels
ENV_ENABLE = "ALLOY_PROXY_TENSOR"
ENV_KAPPA = "ALLOY_PROXY_TENSOR_KAPPA"


def proxy_tensor_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0").strip().lower() in ("1", "true", "yes", "on")


def proxy_tensor_kappa(kappa: float | None = None) -> float:
    if kappa is not None:
        return float(kappa)
    return float(os.environ.get(ENV_KAPPA, PROXY_TENSOR_KAPPA))


def mujoco_blocks(mj: Any, mjw_model: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per MuJoCo body: (W (nbody, 3, 3) symmetrised world-frame translational block at the centre of mass at qpos0,
    R0 (nbody, 3, 3) rotation of the body frame, translatable (nbody,) bool, w0 (nbody,) the translational
    invweight0 Newton's install reads: MuJoCo Warp's world 0 when ``mjw_model`` is given, else the C model's). W is
    zero where the body cannot translate at its centre of mass."""
    import mujoco  # noqa: PLC0415

    d = mujoco.MjData(mj)
    d.qpos[:] = mj.qpos0
    mujoco.mj_forward(mj, d)
    jacp, x = np.zeros((3, mj.nv)), np.zeros((3, mj.nv))
    w_all, r_all = np.zeros((mj.nbody, 3, 3)), np.tile(np.eye(3), (mj.nbody, 1, 1))
    ok = np.zeros(mj.nbody, bool)
    for b in range(1, mj.nbody):
        r_all[b] = d.xmat[b].reshape(3, 3)
        if mj.body_invweight0[b, 0] <= 0.0:
            continue
        mujoco.mj_jacBodyCom(mj, d, jacp, None, b)
        mujoco.mj_solveM(mj, d, x, jacp)
        w = jacp @ x.T
        w_all[b] = 0.5 * (w + w.T)
        ok[b] = float(np.trace(w_all[b])) > 0.0
    w0 = np.asarray(mj.body_invweight0, dtype=np.float64).reshape(-1, 2)[:, 0].copy()
    if mjw_model is not None:
        w0 = np.asarray(mjw_model.body_invweight0.numpy(), dtype=np.float64).reshape(-1, mj.nbody, 2)[0, :, 0].copy()
    return w_all, r_all, ok, w0


S_RANGE = (1.0e-3, 1.0e3)


def body_inv_mass_tensor(w_world: np.ndarray, r0: np.ndarray, inv_mass: float, w0: float | None = None) -> np.ndarray:
    """Unfloored body-frame inverse-mass tensor (R0^T W R0) * inv_mass / w0; ``w0`` defaults to trace(W)/3 (the
    non-simple body's own scalar). identity * inv_mass when W or w0 is empty or the scale s leaves ``S_RANGE``."""
    tr = float(np.trace(w_world))
    w0 = tr / 3.0 if w0 is None else float(w0)
    if inv_mass <= 0.0 or tr <= 0.0 or w0 <= 0.0:
        return np.eye(3) * max(inv_mass, 0.0)
    s = inv_mass / w0
    if not np.isfinite(s) or not S_RANGE[0] <= s <= S_RANGE[1]:
        log.warning("proxy tensor: scale %g (installed %g / invweight0 %g) is out of range; identity kept",
                    s, inv_mass, w0)
        return np.eye(3) * inv_mass
    return (r0.T @ w_world @ r0) * s


def floor_inverse(minv: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """(Minv with eigenvalues >= lambda_max / kappa, its inverse M); both zero for a zero tensor."""
    vals, vecs = np.linalg.eigh(0.5 * (minv + minv.T))
    top = float(vals[-1])
    if top <= 0.0:
        return np.zeros((3, 3)), np.zeros((3, 3))
    vals = np.maximum(vals, top / kappa)
    return (vecs * vals) @ vecs.T, (vecs / vals) @ vecs.T


def world_shapes_agree(mjw_model: Any, rtol: float = 1e-6) -> bool:
    """True when every world's body masses are one common multiple of world 0's (W/trace(W) then agrees)."""
    mass = np.asarray(mjw_model.body_mass.numpy(), dtype=np.float64) if mjw_model is not None else None
    if mass is None or mass.ndim < 2 or mass.shape[0] < 2 or mass.shape[-1] == 0:
        return True
    live = mass[0] > 0.0
    ratio = mass[:, live] / mass[0, live]
    return bool(np.allclose(ratio, ratio[:, :1], rtol=rtol))


def _mjc_body_of_local(solver: Any) -> np.ndarray:
    """Newton body (solver-local) -> MuJoCo body, -1 where none; the table is [world, MuJoCo body] -> Newton body."""
    out = np.full(int(solver.model.body_count), -1, np.int64)
    for row in solver.mjc_body_to_newton.numpy():
        ok = row >= 0
        out[row[ok]] = np.flatnonzero(ok)
    return out


def proxy_tensors(inv_mass: np.ndarray, groups: list[tuple[dict[int, int], tuple[np.ndarray, ...]]],
                  kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """(Minv, M) float32 (N, 3, 3) for a destination with per-body installed ``inv_mass`` (N,). Each group is
    (destination body -> MuJoCo body, mujoco_blocks of that source) for the proxies; every other body is
    identity * inv_mass."""
    minv = np.eye(3)[None] * inv_mass[:, None, None].astype(np.float64)
    mass = np.zeros_like(minv)
    live = inv_mass > 0.0
    mass[live] = np.eye(3) / inv_mass[live][:, None, None]
    for rows, (w_all, r_all, ok, w0) in groups:
        for dst, b in rows.items():
            if inv_mass[dst] <= 0.0 or not ok[b]:
                continue
            tensor = body_inv_mass_tensor(w_all[b], r_all[b], float(inv_mass[dst]), w0[b])
            minv[dst], mass[dst] = floor_inverse(tensor, kappa)
    return minv.astype(np.float32), mass.astype(np.float32)


def is_isotropic(minv: np.ndarray, inv_mass: np.ndarray, rtol: float = ISOTROPIC_RTOL) -> bool:
    """True when every tensor is identity * its body's scalar inverse mass to ``rtol`` (free and xyz-slide bodies):
    the tensor kernels would then repeat the scalar ones up to rounding, so the destination keeps the scalar ones."""
    scale = np.maximum(np.abs(inv_mass), 1e-30)[:, None, None]
    return bool(np.all(np.abs(minv - np.eye(3)[None] * inv_mass[:, None, None]) <= rtol * scale))


def _proxy_rows(src: Any, proxy: Any, dst_n: int) -> dict[int, int]:
    """Destination body (entry-local) -> MuJoCo body for one mapping, through the source entry's tables."""
    src_local = src.body_global_to_local.numpy()[proxy.src_ids.numpy().astype(np.int64)]
    mjc = _mjc_body_of_local(src.solver)
    dst_ids = proxy.proxy_ids_local.numpy().astype(np.int64)
    return {int(d): int(mjc[s]) for d, s in zip(dst_ids, src_local, strict=True) if s >= 0 and mjc[s] > 0 and d < dst_n}


def _has_mujoco(solver: Any) -> bool:
    return getattr(solver, "mjc_body_to_newton", None) is not None and getattr(solver, "mj_model", None) is not None


_CUDA_WARNED: set[str] = set()
_SHAPE_WARNED: set[str] = set()


def install_proxy_tensors(sched: Any, kappa: float | None = None) -> None:
    """Hand each VBD destination its tensors; destinations without ``set_proxy_mass_tensors`` are skipped."""
    kappa = proxy_tensor_kappa(kappa)
    per_dst: dict[str, list[tuple[Any, dict[int, int]]]] = {}
    for proxy in sched._proxy_mappings:
        if proxy.src_ids is None or proxy.src_ids.shape[0] == 0:
            continue
        src, dst = sched._entries[proxy.src_name], sched._entries[proxy.dst_name]
        if not _has_mujoco(src.solver) or not hasattr(dst.solver, "set_proxy_mass_tensors"):
            log.debug("proxy tensor: %s -> %s skipped (no MuJoCo source or no tensor interface)",
                      proxy.src_name, proxy.dst_name)
            continue
        if proxy.src_name not in _SHAPE_WARNED and not world_shapes_agree(getattr(src.solver, "mjw_model", None)):
            _SHAPE_WARNED.add(proxy.src_name)
            log.warning("proxy tensor: worlds differ by more than a uniform mass scale; world 0's shape is used")
        rows = _proxy_rows(src, proxy, int(dst.solver.model.body_count))
        per_dst.setdefault(proxy.dst_name, []).append((src, rows))
    for name, items in per_dst.items():
        solver = sched._entries[name].solver
        inv_mass = np.asarray(solver.model.body_inv_mass.numpy(), dtype=np.float64)
        groups = [(rows, mujoco_blocks(src.solver.mj_model, getattr(src.solver, "mjw_model", None)))
                  for src, rows in items]
        minv, mass = proxy_tensors(inv_mass, groups, kappa)
        if is_isotropic(minv, inv_mass):
            solver.set_proxy_mass_tensors(None, None)       # free bodies only: the scalar kernels, bit for bit
            log.debug("proxy tensor: every proxy on %s is isotropic; scalar kernels kept", name)
            continue
        if getattr(getattr(solver, "device", None), "is_cuda", False) and name not in _CUDA_WARNED:
            _CUDA_WARNED.add(name)
            log.warning("proxy tensor on %s: this option is measured on the CPU only, not yet on a GPU", name)
        solver.set_proxy_mass_tensors(minv, mass)
        log.info("proxy tensors installed on %s: %d proxy bodies, kappa %g", name,
                 sum(len(r) for _, r in items), kappa)


class ProxyTensorMixin:
    """Installs the tensors after Newton installs the scalar proxy masses (construction and every model refresh)."""

    _proxy_tensor_on: bool | None = None

    def _proxy_tensor_active(self) -> bool:
        if self._proxy_tensor_on is None:
            self._proxy_tensor_on = proxy_tensor_enabled()
        return bool(self._proxy_tensor_on)

    def _apply_proxy_body_effective_masses(self) -> None:
        super()._apply_proxy_body_effective_masses()                  # type: ignore[misc]
        if self._proxy_tensor_active():
            install_proxy_tensors(self)

    def graph_key(self) -> tuple:
        """A captured exchange keeps the body kernels it was captured with: the switch is part of the key. It leads
        the key, so the other mixins' elements keep their places at its end (read as key[-1] / key[-2:])."""
        key = super().graph_key()                                     # type: ignore[misc]
        return (("proxy_tensor",) if self._proxy_tensor_active() else ()) + key


__all__ = ["PROXY_TENSOR_KAPPA", "ProxyTensorMixin", "body_inv_mass_tensor", "floor_inverse", "install_proxy_tensors",
           "is_isotropic", "mujoco_blocks", "proxy_tensor_enabled", "proxy_tensors", "world_shapes_agree"]
