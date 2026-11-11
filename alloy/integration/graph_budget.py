# SPDX-License-Identifier: Apache-2.0
"""Graph-size budget for captured control periods (SCR-DC5-4 triage, WP5's segfault).

A CUDA graph with too many nodes crashes the driver at launch (SIGSEGV inside ``cudaGraphLaunch``; measured on the
L4, driver 580): WP5's MuJoCo + VBD cloth drop under ``SchedulerADMM`` captures 16 exchanges as one graph of
1,194,978 nodes, because ADMM unrolls its 20 iterations under capture and each runs MuJoCo plus 16 VBD substeps of
10 iterations; 6 exchanges (448,117 nodes) launched on an idle L4 and crashed on a shared one. A fixed exchange count
per graph cannot bound that, so ``IntegratedSimulation`` runs each key's first replayed period as one-exchange graphs,
counts that graph's nodes after its capture (before its first launch), and sizes later chunks with ``chunk_for``:
``chunk * nodes_per_exchange`` within ``GRAPH_NODE_BUDGET``. (A separate probe capture, never launched, beside the
graphs that are launched does not work: it crashed the MPM sand scene's next launch on the L4.) The budget is far below
every crash because it costs nothing measurable: a graph of 2^16 nodes runs for much longer than its launch overhead.
An exchange cannot be split, so a key whose one exchange is over the budget is never launched as a graph: its graph
is dropped and the key runs eagerly (WP5's ordered-VBD exchange, ``SolverVBD(deterministic=RUN_TO_RUN)``, segfaulted
in its first launch; unrolled ADMM graphs that size replay ~9x slower than eager anyway, wp12-results).

The count is of the graph's top-level nodes (``cuGraphGetNodes``); a ``wp.capture_while`` body is one conditional
node there, so loops that stay on the device count once.

A graph that runs out of device memory is stepped down by ``IntegratedSimulation._out_of_memory`` (smaller graphs,
then none), and the period it was for runs eagerly at once. Before it, ``release`` gives the device back what the
process holds unused (SCR-OSS22-9):
- the simulation's captured graphs: each keeps the scratch of the exchanges it recorded (the sand scene's
  one-exchange graphs of its first period held 4 GB that its eager period then lacked); they are captured again when
  next launched;
- the failed graph, which the traceback of the launch that failed still refers to (1.2 GB for the sand scene's
  8-exchange graph);
- simulations dropped earlier (reference cycles: their arrays and graphs stay allocated until a full collection), the
  graph pool's reservation and the memory pool's free blocks. CUDA graphs and CUB's own setup take their memory from
  the driver, not from Warp's pool.
Last, it takes the error the allocation failure leaves for the next CUB native (``native_guard.absorb_stale_error``):
that error, not a stream left capturing nor the memory, is what stopped the eager period ("CUDA error 101: invalid
device ordinal" in a prefix sum; three lines of the 100k pass: sand at 64 worlds, T09 at 64-1,024 worlds).

A scene whose exchange is known to be over the budget skips that capture: ``eager_scene_reason`` names a scene
effect in ``EAGER_EFFECTS`` and ``build_simulation`` then runs the scene without graphs. The accurate cloth profile
(SCR-WP11-5) is one: its VBD iterations are a Python loop, unrolled under capture. A j2 solid on implicit MPM
(``mpm_solid.EFFECT``) is another: it needs the dense grid, and graphs need the rebuildable sparse one.
"""

from __future__ import annotations

import ctypes
import functools
import gc
import traceback
from collections.abc import Callable, Iterable
from typing import Any

import warp as wp

from alloy.integration.mpm_memory import free_bytes
from alloy.integration.native_guard import absorb_stale_error
from alloy.solvers.vbd.cloth_profile import ACCURATE_EFFECT, ACCURATE_ITERATIONS
from alloy.solvers.plastic.mpm_solid import EFFECT as J2_MPM_EFFECT

__all__ = ["EAGER_EFFECTS", "GRAPH_NODE_BUDGET", "OOM_MARKERS", "chunk_for", "countable", "eager_scene_reason",
           "graph_nodes", "out_of_memory", "pow2_floor", "release"]

GRAPH_NODE_BUDGET = 1 << 16   # top-level nodes per captured graph, an upper bound (wp12-results, "graph node budget"):
                               # 1.19 M crashed on the L4; 448 k launched alone but crashed on a shared L4 (WP5)
EAGER_EFFECTS: dict[str, str] = {   # scene effect -> why one exchange can never fit the budget
    ACCURATE_EFFECT: f"the accurate cloth profile's {ACCURATE_ITERATIONS} VBD iterations per step unroll under "
                     "capture, so one exchange is far over the graph node budget (SCR-WP11-5)",
    J2_MPM_EFFECT: "a j2 solid on implicit MPM reads true only on the dense grid, which graphs cannot capture (on the "
                   "rebuildable sparse grid a punch pushed on it from 2 mm above its surface and read 34 % low)",
}


OOM_MARKERS = ("out of memory", "Failed to allocate", "cudaErrorMemoryAllocation", "CUDA_ERROR_OUT_OF_MEMORY")


def out_of_memory(exc: BaseException | None) -> bool:
    """True when ``exc``, or an exception it was raised from, reads as a device allocation failure. On the L4 an MPM
    entry coupled to another one at 64 worlds failed creating its period graph: "Graph creation error: Warp CUDA error
    2: out of memory" (WP13's combination sweep, F10), where the same scene steps eagerly."""
    for _ in range(8):
        if exc is None:
            return False
        if any(m in str(exc) for m in OOM_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def release(device: Any, graph: Any, cache: Any = None, exc: BaseException | None = None) -> int | None:
    """Drop ``graph`` (one that ran out of memory) and hand back to ``device`` what the process holds unused (module
    docstring); on CUDA also every other graph in ``cache`` (the simulation's ``GraphCache``; they are captured again
    when next launched) and what ``exc``'s traceback still holds. Returns the device's free bytes after it; None on
    the CPU, where only ``graph`` is dropped."""
    from alloy.parallel.counters import trim_graph_pool  # noqa: PLC0415 - WP8 (a no-op on the CPU)

    graph.invalidate()
    device = wp.get_device(device)
    if not device.is_cuda:
        return None
    if cache is not None:
        cache.invalidate()
    for _ in range(8):                              # the failed launch's frame holds the wp.Graph and its scratch
        if exc is None:
            break
        traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__
    gc.collect()
    trim_graph_pool(str(device))                    # a failed capture may leave graph-pool reservations
    free = free_bytes(device)
    absorb_stale_error(device)                      # else the eager period's first prefix sum or sort fails (101)
    return free


def eager_scene_reason(scene_effects: Iterable[str]) -> str | None:
    """Why a scene with ``scene_effects`` runs without graphs from the start (``EAGER_EFFECTS``), or None."""
    return next((EAGER_EFFECTS[e] for e in scene_effects if e in EAGER_EFFECTS), None)


def pow2_floor(x: int) -> int:
    """Largest power of two <= ``x`` (1 for ``x`` < 2)."""
    return 1 << (max(int(x), 1).bit_length() - 1)


def chunk_for(nodes_per_exchange: int | None, cap: int, budget: int = GRAPH_NODE_BUDGET) -> int:
    """Exchanges per captured graph: the largest power of two <= ``cap`` whose graph stays within ``budget`` nodes
    (at least 1); ``cap`` when the node count is unknown (None: CPU graph or no CUDA driver library)."""
    cap = pow2_floor(cap)
    if not nodes_per_exchange:
        return cap
    return min(cap, pow2_floor(budget // nodes_per_exchange))


@functools.cache
def _cu_graph_get_nodes() -> Callable | None:
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            lib = ctypes.WinDLL("nvcuda.dll")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return None
    fn = getattr(lib, "cuGraphGetNodes", None)
    if fn is None:
        return None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    fn.restype = ctypes.c_int
    return fn


def graph_nodes(graph: wp.Graph | None) -> int | None:
    """Top-level node count of a captured CUDA graph; None for a CPU graph or without the driver library."""
    handle = getattr(graph, "graph", None)
    fn = _cu_graph_get_nodes()
    if not handle or fn is None:
        return None
    n = ctypes.c_size_t(0)
    if fn(handle, None, ctypes.byref(n)) != 0:
        return None
    return int(n.value)


def countable(device: str) -> bool:
    """True when graphs captured on ``device`` can be counted (a CUDA device and the driver library)."""
    return wp.get_device(device).is_cuda and _cu_graph_get_nodes() is not None
