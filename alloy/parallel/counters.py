# SPDX-License-Identifier: Apache-2.0
"""Per-world solver counters read between timed chunks, and a graph-safe trace replayer (parallel §4.1, §4.9).

Counters come from the solver's own device arrays, read on the host *outside* the timed window:

- SolverMuJoCo: ``mjw_data.solver_niter`` and ``mjw_data.nefc`` per world, ``mjw_data.overflow`` bitmask per world
  (mjw@a9daa05f:mujoco_warp/_src/types.py:2523, 2527, 2677), split into capacity bits (``overflow``: a buffer
  dropped contacts or rows) and the iteration-limit bits (``iter_limit``: the solve stopped at its iteration cap,
  WP5's UNCONVERGED, a convergence note and not a capacity failure); MuJoCo contacts per world from
  ``mjw_data.contact.worldid[:nacon]`` (types.py:2296, 2672) when SolverMuJoCo runs its own collision;
- Newton contacts per world from ``Contacts.rigid_contact_count`` / ``rigid_contact_shape0`` and ``Model.shape_world``.

``TraceReplayer`` writes one row of a recorded ``Trace`` into the control arrays per control period from a device
step counter, so a replay can sit inside a captured graph (no host writes per period).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

log = logging.getLogger("alloy.parallel")


@dataclass(frozen=True)
class ProbeSample:
    """One host reading of the solver counters (None = the solver does not expose it)."""

    iters: np.ndarray | None = None  # [W] solver iterations of the last step
    rows: np.ndarray | None = None  # [W] active constraint rows
    contacts: np.ndarray | None = None  # [W] active contacts
    pairs: int | None = None  # broad-phase candidate pairs, all worlds
    overflow: int = 0  # worlds with a capacity overflow bit set (every MJW bit except the iteration limits)
    iter_limit: int = 0  # worlds whose solve hit its iteration cap (ITERATIONS / LS_ITERATIONS bits)


Counters = Callable[[], ProbeSample]


def per_world_contacts(model: newton.Model, contacts: newton.Contacts) -> np.ndarray:
    """Active Newton rigid contacts per world, [W]."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_world = model.shape_world.numpy()
    s0 = contacts.rigid_contact_shape0.numpy()[:count]
    worlds = shape_world[s0] if count else np.zeros(0, dtype=np.int64)
    worlds = worlds[worlds >= 0]
    return np.bincount(worlds, minlength=model.world_count)[: model.world_count]


def mujoco_counters(solver: object, model: newton.Model, contacts: newton.Contacts | None = None) -> Counters:
    """Counters for a ``SolverMuJoCo`` (Newton contacts when ``contacts`` is given, else MuJoCo's own)."""
    data = getattr(solver, "mjw_data", None)
    if data is None:
        raise TypeError(f"{type(solver).__name__} has no mjw_data; not a SolverMuJoCo on the MJWarp backend")

    def read() -> ProbeSample:
        if contacts is not None:
            ncon = per_world_contacts(model, contacts)
        else:
            ncon = _mjw_contacts_per_world(data, model.world_count)
        overflow, iter_limit = _overflow_worlds(data)
        return ProbeSample(iters=data.solver_niter.numpy().astype(np.int64), rows=data.nefc.numpy().astype(np.int64),
                           contacts=ncon, overflow=overflow, iter_limit=iter_limit)

    return read


def newton_contact_counters(model: newton.Model, contacts: newton.Contacts) -> Counters:
    """Counters for solvers without iteration readouts (contacts only)."""

    def read() -> ProbeSample:
        return ProbeSample(contacts=per_world_contacts(model, contacts))

    return read


def _overflow_worlds(data: object) -> tuple[int, int]:
    """(worlds with a capacity bit, worlds with an iteration-limit bit) of ``mjw_data.overflow``."""
    if getattr(data, "overflow", None) is None:
        return 0, 0
    from alloy._vendor.mujoco_warp import OverflowType  # noqa: PLC0415

    bits = data.overflow.numpy().astype(np.int64)
    iters = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
    return int(np.count_nonzero(bits & ~iters)), int(np.count_nonzero(bits & iters))


def _mjw_contacts_per_world(data: object, nworld: int) -> np.ndarray:
    nacon = int(data.nacon.numpy()[0])
    if nacon == 0:
        return np.zeros(nworld, dtype=np.int64)
    worldid = data.contact.worldid.numpy()[:nacon]
    return np.bincount(worldid, minlength=nworld)[:nworld]


def kernels_per_call(fn: Callable[[], None], device: str) -> int | None:
    """Kernel launches in one eager call of ``fn`` (Warp CUDA timing, warp/_src/utils.py:2178); None on CPU.

    Runs ``fn`` once (it advances the simulation). PAR-9 proper uses nsys; this counts Warp-launched kernels only.
    """
    dev = wp.get_device(device)
    if not dev.is_cuda:
        return None
    wp.synchronize_device(dev)
    wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
    try:
        fn()
    finally:
        results = wp.timing_end()
    return len(results)


# ---- device memory ----------------------------------------------------------------------------------------------
_GRAPH_MEM_SYMBOL = "wp_cuda_device_get_graph_mem_current"  # warp native/warp.cu:2761 (cudaGraphMemAttrUsedMemCurrent)
_GRAPH_TRIM_SYMBOL = "wp_cuda_device_graph_mem_trim"  # warp native/warp.cu (cudaDeviceGraphMemTrim)
_warned_graph_mem = False


def _native(symbol: str) -> Callable[[int], Any] | None:
    """A function of Warp's native core (no public API in Warp 1.18; private, so looked up defensively, SCR-WP8-4)."""
    try:
        from warp._src.context import runtime  # noqa: PLC0415
    except ImportError:
        return None
    return getattr(getattr(runtime, "core", None), symbol, None)


def graph_pool_bytes(device: str) -> int | None:
    """Bytes held by CUDA graph memory nodes on ``device``; None on CPU or when Warp does not expose the counter.

    MJWarp allocates scratch inside its step (EPA polytopes per contact slot, mujoco_warp/_src/collision_convex.py:1295;
    a qLD copy for implicit integrators, forward.py:531). Under a captured graph that scratch lives in the CUDA graph
    memory pool, which ``wp.get_mempool_used_mem_current`` does not count.
    """
    global _warned_graph_mem
    dev = wp.get_device(device)
    if not dev.is_cuda:
        return None
    fn = _native(_GRAPH_MEM_SYMBOL)
    if fn is None:
        if not _warned_graph_mem:
            log.warning("Warp exposes no %s; probe memory omits the CUDA graph pool", _GRAPH_MEM_SYMBOL)
            _warned_graph_mem = True
        return None
    return int(fn(dev.ordinal))


def trim_graph_pool(device: str) -> None:
    """Hand unused CUDA graph memory back to the driver (no-op on CPU or without the Warp symbol).

    Measured on L4 ([internal note]): the graph pool's used counter keeps a released graph's
    memory until a trim, so without one a probe point only sees its growth over the largest earlier graph.
    """
    dev = wp.get_device(device)
    fn = _native(_GRAPH_TRIM_SYMBOL) if dev.is_cuda else None
    if fn is not None:
        wp.synchronize_device(dev)
        fn(dev.ordinal)


def live_bytes(device: str) -> int | None:
    """Device bytes the process holds live: mempool used + CUDA graph pool used (CUDA only; None on CPU).

    Synchronises the device first. Reserved-but-free mempool blocks are not counted (they are reused by later
    allocations on the device).
    """
    dev = wp.get_device(device)
    if not (dev.is_cuda and wp.is_mempool_enabled(dev)):
        return None
    wp.synchronize_device(dev)
    return int(wp.get_mempool_used_mem_current(dev)) + (graph_pool_bytes(device) or 0)


# ---- trace replay -----------------------------------------------------------------------------------------------
@wp.kernel
def _replay_row(src: wp.array3d[wp.float32], step: wp.array[wp.int32], n_rows: int, index: wp.array[wp.int32],
                stride: int, dst: wp.array[wp.float32]):
    w, i = wp.tid()
    k = wp.min(step[0], n_rows - 1)
    dst[w * stride + index[i]] = src[k, w % src.shape[1], i]


@wp.kernel
def _advance(step: wp.array[wp.int32]):
    step[0] = step[0] + 1


@dataclass(frozen=True)
class ReplayLayout:
    """Where trace column ``i`` of world ``w`` lands in a flat float32 control array: ``w * stride + index[i]``."""

    index: np.ndarray
    stride: int


class TraceReplayer:
    """Copies row ``k`` of each trace input into its control array, k from a device counter (graph-safe).

    ``inputs[name]`` is ``[T, n]`` (same for every world) or ``[T, W_trace, n]`` (world w reads ``w % W_trace``).
    Without a ``layouts`` entry the control array holds ``W * n`` values world-major; with one, the columns scatter
    to ``ReplayLayout.index`` inside each world's block of ``stride`` values (e.g. a few joint_f dofs, or a vec2
    array viewed as float32). Past the last row the last row is held.
    """

    def __init__(self, inputs: dict[str, np.ndarray], targets: dict[str, wp.array], nworld: int, device: str,
                 layouts: dict[str, ReplayLayout] | None = None) -> None:
        self.nworld = nworld
        self.step = wp.zeros(1, dtype=wp.int32, device=device)
        self._plan = [self._bind(name, rows, targets[name], (layouts or {}).get(name), device)
                      for name, rows in inputs.items()]
        self.rows = min((p[0].shape[0] for p in self._plan), default=0)

    def _bind(self, name: str, rows: np.ndarray, dst: wp.array, layout: ReplayLayout | None, device: str) -> tuple:
        rows3 = rows[:, None, :] if rows.ndim == 2 else rows.reshape(rows.shape[0], rows.shape[1], -1)
        n = rows3.shape[2]
        flat = dst.flatten() if dst.ndim > 1 else dst
        if flat.dtype != wp.float32:
            raise ValueError(f"control field {name}: needs a float32 array (view vector arrays as float32)")
        layout = layout or ReplayLayout(np.arange(n), n)
        if len(layout.index) != n:
            raise ValueError(f"control field {name}: layout has {len(layout.index)} columns, the trace {n}")
        need = (self.nworld - 1) * layout.stride + int(np.max(layout.index, initial=-1)) + 1
        if flat.size < need or (layout.stride == n and flat.size != self.nworld * n):
            raise ValueError(f"control field {name}: {flat.size} values, trace gives {self.nworld} x {n} "
                             f"(stride {layout.stride})")
        src = wp.array(np.ascontiguousarray(rows3, dtype=np.float32), dtype=wp.float32, device=device)
        index = wp.array(np.asarray(layout.index, dtype=np.int32), dtype=wp.int32, device=device)
        return src, flat, n, index, layout.stride

    def apply(self) -> None:
        """Write the current row into every control target, then advance the counter."""
        for src, dst, n, index, stride in self._plan:
            wp.launch(_replay_row, dim=(self.nworld, n), inputs=[src, self.step, src.shape[0], index, stride, dst],
                      device=self.step.device)
        wp.launch(_advance, dim=1, inputs=[self.step], device=self.step.device)

    def reset(self) -> None:
        self.step.zero_()
