# SPDX-License-Identifier: Apache-2.0
"""Cost model for world packing (SPEC §3.9, docs/research/parallel.md §4.1, §4.3, §4.7).

One pack of N identical worlds costs ``T(N) = t_floor + N * c_w`` seconds per step: flat while the GPU is not full,
then linear. From it:

    Θ(N)    = N * dt / T(N)                     sim-s per GPU-s (aggregate real-time factor)
    N_knee  = t_floor / c_w ;  Θ(N) = Θ_max * N / (N + N_knee),  Θ_max = dt / c_w
    u(N)    = N * c_w / T(N) = N / (N + N_knee)  share of the device a pack keeps busy when alone

``pack_size`` takes the smallest N reaching ``plateau_fraction`` of Θ_max (``f / (1 − f) * N_knee``, 9·N_knee at
f = 0.9), capped by the request count, device memory and the dense-H share. All functions are pure and host-only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from alloy.core.errors import CapacityError
from alloy.parallel.types import PackPolicy, WorldProfile

log = logging.getLogger("alloy.parallel")

# MuJoCo Warp pads nv for the dense Newton solve: round_up(nv + 1, 16) above 32 DOF (the +1 is the augmented
# Cholesky column), round_up(nv, 4) otherwise (mjw@a9daa05f:mujoco_warp/_src/io.py:1410-1417).
MJW_TILE = 16
MJW_BLOCK_CHOLESKY_DIM = 32  # hfactor exists only above this (mjw@a9daa05f:mujoco_warp/_src/solver.py:40, 86)
FLOAT_BYTES = 4


@dataclass(frozen=True)
class MemoryTerms:
    """Inputs of the per-world memory estimate (parallel §4.7). Byte sizes per item are UNVERIFIED defaults; the
    estimate predicts and cross-checks, it never replaces a probed ``bytes_per_world``. It omits MJWarp's in-step
    scratch, which under a CUDA graph lives in the graph memory pool (measured on L4: ~0.8 MB per world for a 16-box
    pile, ~3 MB for 64 boxes, more than the persistent arrays); probed profiles include it (``counters.live_bytes``)."""

    nv: int
    rows_max: int = 0  # constraint rows per world (njmax)
    contacts_max: int = 0  # contacts per world
    pairs_max: int = 0  # broad-phase candidate pairs per world
    n_bodies: int = 0
    n_particles: int = 0
    dense_h: bool = True  # MJW Newton solver: h (+ hfactor) dense (nworld, nv_pad, nv_pad)
    bytes_per_contact: int = 160  # UNVERIFIED: Newton Contacts + MJW contact struct, float32
    bytes_per_row: int = 96  # UNVERIFIED: MJW efc_* vectors per row (J stored sparse above nv 60)
    bytes_per_body: int = 256  # UNVERIFIED: State + MJW Data per body (q, qd, xpos, xquat, cinert, cvel, ...)
    bytes_per_particle: int = 48  # q, qd, f float32 vec3 + flags


def fit_cost_model(n: Sequence[int], t: Sequence[float]) -> tuple[float, float]:
    """Least-squares fit of T(N) = t_floor + N c_w; returns (t_floor [s], c_w [s/world]).

    Rows are weighted by 1/t so every point counts in relative terms (the N = 1 point pins t_floor, the large-N
    points pin c_w). Both parameters are constrained to be >= 0: if the unconstrained fit makes one negative, it is
    set to 0 and the other refitted. Raises ValueError on fewer than two distinct N or non-positive times.
    """
    n_arr = np.asarray(n, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    _check_fit_input(n_arr, t_arr)
    w = 1.0 / t_arr
    a = np.column_stack([w, n_arr * w])
    b = t_arr * w
    t_floor, c_w = np.linalg.lstsq(a, b, rcond=None)[0]
    if t_floor < 0.0:
        t_floor, c_w = 0.0, float(np.sum(a[:, 1] * b) / np.sum(a[:, 1] ** 2))
    elif c_w < 0.0:
        t_floor, c_w = float(np.sum(a[:, 0] * b) / np.sum(a[:, 0] ** 2)), 0.0
    return float(t_floor), float(c_w)


def _check_fit_input(n: np.ndarray, t: np.ndarray) -> None:
    if n.shape != t.shape or n.ndim != 1:
        raise ValueError(f"n and t must be 1-D and the same length, got {n.shape} and {t.shape}")
    if np.unique(n).size < 2:
        raise ValueError(f"need at least two distinct world counts to fit T(N), got {n.tolist()}")
    if not (np.all(np.isfinite(t)) and np.all(t > 0.0)) or np.any(n < 1):
        raise ValueError(f"times must be finite and > 0 and N >= 1, got n={n.tolist()} t={t.tolist()}")


def step_time(p: WorldProfile, n: int) -> float:
    """Predicted seconds per step of a pack of n worlds."""
    return p.t_floor_s + n * p.c_w_s


def n_knee(p: WorldProfile) -> float:
    """N_knee = t_floor / c_w (inf when c_w is 0: the probe never left the flat part)."""
    if p.c_w_s <= 0.0:
        return math.inf
    return p.t_floor_s / p.c_w_s


def throughput(p: WorldProfile, n: int) -> float:
    """Θ(n) = n dt / T(n), sim-s per GPU-s."""
    return n * p.dt / step_time(p, n)


def occupancy(p: WorldProfile, n: int) -> float:
    """u(n) = n c_w / T(n) in [0, 1): the device share a pack of n worlds uses when it runs alone."""
    t = step_time(p, n)
    return 0.0 if t <= 0.0 else n * p.c_w_s / t


def plateau_n(p: WorldProfile, fraction: float) -> int:
    """Smallest N with Θ(N) >= fraction * Θ_max: ceil(fraction / (1 - fraction) * N_knee)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"plateau_fraction must be in (0, 1), got {fraction}")
    knee = n_knee(p)
    if math.isinf(knee):
        return 2**31 - 1
    return max(1, math.ceil(fraction / (1.0 - fraction) * knee - 1e-9))


def mjw_nv_pad(nv: int) -> int:
    """MuJoCo Warp's padded DOF count for the dense Newton solve (io.py:1410-1417)."""
    if nv > MJW_BLOCK_CHOLESKY_DIM:
        return _round_up(nv + 1, MJW_TILE)
    return _round_up(nv, 4)


def dense_h_bytes(nv: int) -> int:
    """Bytes per world of MJW's dense h and hfactor: 4 nv_pad² each, hfactor only above 32 DOF (solver.py:82-118)."""
    pad = mjw_nv_pad(nv)
    n_mats = 2 if nv > MJW_BLOCK_CHOLESKY_DIM else 1
    return n_mats * FLOAT_BYTES * pad * pad


def estimate_bytes_per_world(m: MemoryTerms) -> int:
    """Per-world memory estimate (parallel §4.7): state + contacts + rows + pairs + dense solver blocks."""
    total = m.n_bodies * m.bytes_per_body + m.n_particles * m.bytes_per_particle
    total += m.contacts_max * m.bytes_per_contact + m.rows_max * m.bytes_per_row + m.pairs_max * 8
    if m.dense_h and m.nv > 0:
        total += dense_h_bytes(m.nv) + 2 * FLOAT_BYTES * mjw_nv_pad(m.nv)  # + grad, Mgrad vectors
    return int(total)


def n_mem(p: WorldProfile, free_mem: int, policy: PackPolicy) -> int:
    """Worlds that fit: floor(((1 − reserve) free_mem − bytes_fixed) / bytes_per_world); may be <= 0."""
    budget = (1.0 - policy.mem_reserve_fraction) * free_mem - p.bytes_fixed
    if p.bytes_per_world <= 0:
        return 2**31 - 1 if budget > 0 else 0
    return int(math.floor(budget / p.bytes_per_world))


def n_dense_h(p: WorldProfile, free_mem: int, policy: PackPolicy) -> int:
    """Worlds whose dense per-world H stays within max_dense_h_fraction of memory (parallel §4.7)."""
    if p.nv <= 0:
        return 2**31 - 1
    return int(math.floor(policy.max_dense_h_fraction * free_mem / dense_h_bytes(p.nv)))


def pack_size(p: WorldProfile, count: int, free_mem: int, policy: PackPolicy) -> int:
    """min(count, ceil(9 N_knee), N_mem, N_denseH); raises CapacityError if not even one world fits."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    fit_mem = min(n_mem(p, free_mem, policy), n_dense_h(p, free_mem, policy))
    if fit_mem < 1:
        raise CapacityError(
            f"one world of scene {p.scene_hash[:12]} needs {p.bytes_fixed + p.bytes_per_world} B "
            f"(dense H {dense_h_bytes(p.nv) if p.nv > 0 else 0} B) but {free_mem} B are free "
            f"(reserve {policy.mem_reserve_fraction:.0%}, dense-H share {policy.max_dense_h_fraction:.0%})"
        )
    n = min(count, plateau_n(p, policy.plateau_fraction), fit_mem)
    log.debug("pack_size scene=%s count=%d knee=%.1f mem_fit=%d -> %d", p.scene_hash[:12], count, n_knee(p),
              fit_mem, n)
    return n


def predicted_bytes(p: WorldProfile, n: int) -> int:
    """bytes_fixed + n bytes_per_world (PAR-11 compares this with the measured high-water mark)."""
    return int(p.bytes_fixed + n * p.bytes_per_world)


def capacities(p: WorldProfile, n: int, policy: PackPolicy) -> dict[str, int]:
    """Buffer capacities of a pack of n worlds (SPEC §4.2: contacts and rows 1.5x p99, pairs 1.25x peak).

    ``rigid_contact_max`` and ``shape_pairs_max`` are totals for the Newton pipeline (newton@ac7f7830:
    newton/_src/sim/collide.py:1488, 1504); ``njmax`` and ``nconmax`` are per world as SolverMuJoCo takes them
    (solvers/mujoco/solver_mujoco.py:4328-4330, 9099; MJW pools nconmax internally, mjw io.py:1923).
    """
    return capacities_from_counts(p.contacts_p99, p.rows_p99, p.pairs_peak, n, policy)


def capacities_from_counts(contacts_p99: float, rows_p99: float, pairs_peak: float, n: int,
                           policy: PackPolicy) -> dict[str, int]:
    """``capacities`` from the three per-world counts directly (a probe row instead of a full WorldProfile)."""
    contacts = max(1, math.ceil(policy.capacity_headroom * contacts_p99))
    rows = max(1, math.ceil(policy.capacity_headroom * rows_p99))
    pairs = max(1, math.ceil(policy.pairs_headroom * pairs_peak))
    return {"rigid_contact_max": contacts * n, "njmax": rows, "nconmax": contacts, "shape_pairs_max": pairs * n}


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple
