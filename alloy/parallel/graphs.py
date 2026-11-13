# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture utility for one control period (SPEC §3.9, D18, parallel §4.8).

``PeriodGraph`` owns no stepping logic: ``Simulation`` (WP2/WP12) hands it a ``step_fn`` that runs one control period
and calls ``launch()`` once per period. On CUDA the first launch captures ``step_fn`` into a graph
(``wp.ScopedCapture``, warp@1.18.0:warp/_src/utils.py:2051) and every launch replays it (``wp.capture_launch``,
context.py:13970). Iterative solvers inside ``step_fn`` use ``wp.capture_while`` (context.py:13775), which also runs
eagerly outside a capture, so no Alloy wrapper is needed. One graph per ``(dt_x, n_e)`` key: ``GraphCache``.

CPU graphs (SCR-WP8-5 Ask 1). Warp 1.18 records CPU graphs too: ``wp.capture_begin(device="cpu")`` records an APIC
op stream (context.py:13122) without executing it, and ``wp.capture_launch`` replays it in C++ (T12: 2.0-2.9 ms per
step against 14.6-19 eager, bitwise equal; [internal note] notes/wp8-results/cpu-item11/PROFILE.md §5).
``cpu_graphs_supported`` checks that once per process with a trial capture of one memset. ``use_graph=None`` then
means graphs on CUDA and on such a CPU; a caller passes None only for a step_fn that is graph-safe on CUDA.

Host syncs (PAR-14). A captured graph contains no host code, so a replayed period has zero host syncs by
construction. A host sync inside ``step_fn`` breaks the capture: ``wp.synchronize*`` raises (context.py:12332-12375)
and a blocking device-to-host copy is rejected by CUDA. On CPU neither fails by itself: a sync is a no-op and a host
read during the capture silently sees data the record-only capture never computed. So a CPU capture counts every
``wp.array.numpy()`` of a CPU array and every ``wp.synchronize*`` call made while it records, and fails the same way.
``host_syncs`` counts such failed captures; the capture then raises ``GraphCaptureError`` (nothing falls back to
eager silently). Host reads that bypass ``numpy()`` (raw pointers, ``__array_interface__``) are not seen.

Replay hazard: Python-side ping-pong swaps run once, at capture time. If ``step_fn`` leaves the state references
swapped (an odd number of swaps), every replay would read the stale buffer. Pass ``buffers`` (a callable returning
the arrays whose identity must survive a period) and the capture checks their pointers are unchanged.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Hashable, Iterator, Sequence

import warp as wp

from alloy.core.errors import GraphCaptureError
from alloy.parallel.counters import trim_graph_pool

log = logging.getLogger("alloy.parallel")
__all__ = ["GraphCache", "GraphCaptureError", "PeriodGraph", "cpu_graphs_supported", "graphs_by_default"]

_SYNC_MARKERS = ("graph capture", "capturing", "cudaErrorStreamCapture", "operation not permitted")
_SYNC_FUNCS = ("synchronize", "synchronize_device", "synchronize_stream", "synchronize_event")
_CPU_GRAPHS: dict[str, bool] = {}  # "ok" -> result of the one trial capture per process


def cpu_graphs_supported() -> bool:
    """Whether this Warp records and replays CPU graphs (APIC, Warp >= 1.18). One trial per process: a capture of
    one memset must record without executing it, and the replay must execute it. False (and logged once) on a Warp
    whose ``capture_begin`` refuses the CPU. Not cached when asked during an active CPU capture."""
    if "ok" in _CPU_GRAPHS:
        return _CPU_GRAPHS["ok"]
    cpu = wp.get_device("cpu")
    if cpu.is_capturing:
        return False
    ok, why = _trial_cpu_capture()
    _CPU_GRAPHS["ok"] = ok
    log.log(logging.INFO if ok else logging.WARNING, "Warp %s CPU graphs: %s", wp.__version__,
            "supported" if ok else f"not supported ({why}); CPU periods run eagerly")
    return ok


def _trial_cpu_capture() -> tuple[bool, str]:
    probe = wp.full(1, 7, dtype=wp.int32, device="cpu")
    try:
        wp.capture_begin(device="cpu")
    except Exception as exc:  # noqa: BLE001 - any refusal means no CPU graphs
        return False, f"capture_begin: {exc}"
    try:
        probe.zero_()
    finally:
        graph = wp.capture_end(device="cpu")
    recorded_only = int(probe.numpy()[0]) == 7
    wp.capture_launch(graph)
    if not (recorded_only and int(probe.numpy()[0]) == 0):
        return False, "the trial capture did not record-then-replay a memset"
    return True, ""


def graphs_by_default(device: wp.context.Devicelike) -> bool:
    """``use_graph=None``: graphs on every CUDA device, and on the CPU when ``cpu_graphs_supported()``."""
    dev = wp.get_device(device)
    return dev.is_cuda or (dev.is_cpu and cpu_graphs_supported())


class PeriodGraph:
    """Capture-once, replay-per-period wrapper around ``step_fn`` (one control period)."""

    def __init__(self, step_fn: Callable[[], None], device: str, *, use_graph: bool | None = None,
                 buffers: Callable[[], Sequence[wp.array]] | None = None, pin_mempool: bool = True) -> None:
        self.step_fn = step_fn
        self.device = wp.get_device(device)
        self.use_graph = graphs_by_default(self.device) if use_graph is None else use_graph
        if self.use_graph and self.device.is_cuda and not self.device.is_mempool_supported:
            log.warning("device %s has no mempool; graph capture may fail on allocation", self.device)
        self.buffers = buffers
        self.pin_mempool = pin_mempool
        self.host_syncs = 0  # failed captures caused by a host sync inside step_fn (must stay 0, PAR-14)
        self.captures = 0
        self.launches = 0
        self._graph: wp.Graph | None = None

    @property
    def mode(self) -> str:
        """"graph" once a graph is captured, else "eager"."""
        return "graph" if self._graph is not None else "eager"

    def launch(self) -> None:
        """Run one control period: replay the graph, capturing it first if needed; eager when graphs are off."""
        if not self.use_graph:
            self.step_fn()
            if self.launches == 0:
                self._pin_mempool()
        else:
            if self._graph is None:
                self._capture()
            wp.capture_launch(self._graph)
        self.launches += 1

    def invalidate(self) -> None:
        """Drop the graph; the next launch recaptures (topology change, buffer replacement, capacity growth or a
        solver-option change, parallel §4.8). In-place array edits need no recapture."""
        if self._graph is None:
            return
        log.info("period graph on %s invalidated after %d launches", self.device, self.launches)
        self._graph = None
        trim_graph_pool(str(self.device))  # the old graph's scratch stays reserved in the graph pool until a trim

    def _capture(self) -> None:
        before = self._pointers()
        reads: list[str] = []
        try:
            with _cpu_host_reads(self.device, reads), wp.ScopedCapture(device=self.device) as cap:
                self.step_fn()
        except Exception as exc:
            self._end_dangling_capture()
            if _is_sync_error(exc):
                self.host_syncs += 1
                raise GraphCaptureError(f"host sync inside a captured control period on {self.device}: {exc}") from exc
            raise GraphCaptureError(f"capturing the control period on {self.device} failed: {exc}") from exc
        if reads:
            self.host_syncs += 1
            raise GraphCaptureError(
                f"host read inside a captured control period on {self.device} ({len(reads)}: {', '.join(reads[:4])}): "
                "a CPU capture records without executing, so the read saw stale data")
        after = self._pointers()
        if before != after:
            raise GraphCaptureError(
                "step_fn left its buffers swapped after one period (odd number of ping-pong swaps); replays would read "
                f"stale state. pointers before {before}, after {after}"
            )
        self._graph = cap.graph
        self.captures += 1
        log.info("captured control-period graph #%d on %s", self.captures, self.device)

    def _pointers(self) -> tuple[int, ...]:
        """The buffers' device addresses, 0 for a missing or empty one: a zero-length Warp array has no allocation and
        ``ptr`` None (e.g. joint_q of a scene whose robots have no degrees of freedom)."""
        if self.buffers is None:
            return ()
        return tuple(0 if a is None or a.ptr is None else int(a.ptr) for a in self.buffers())

    def _end_dangling_capture(self) -> None:
        if not self.device.is_capturing:  # CUDA stream capture, or the CPU's APIC recording (context.py:5981)
            return
        try:
            wp.capture_end(device=self.device)
        except Exception as exc:  # noqa: BLE001 - the original capture error is what the caller needs
            log.warning("could not end the broken capture on %s: %s", self.device, exc)

    def _pin_mempool(self) -> None:
        """Eager mode only: release threshold = high-water mark after the first period, so later periods never hand
        their scratch back and reallocate it (parallel §4.7; warp context.py:9359). Not done after a capture: graph
        replays take their scratch from the CUDA graph memory pool, and a pin there would only keep eager warm-up
        scratch alive (measured on L4: +736 MB for a 64-box pile at 256 worlds, enough to OOM a 4,096-world pack)."""
        if not (self.pin_mempool and self.device.is_cuda and wp.is_mempool_enabled(self.device)):
            return
        high = wp.get_mempool_used_mem_high(self.device)
        if high > wp.get_mempool_release_threshold(self.device):
            wp.set_mempool_release_threshold(self.device, high)


def _is_sync_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(m in text for m in _SYNC_MARKERS)


@contextlib.contextmanager
def _cpu_host_reads(device: wp.context.Device, reads: list[str]) -> Iterator[None]:
    """While a CPU capture records: note each ``wp.array.numpy()`` of a CPU array and each ``wp.synchronize*`` call
    in ``reads`` (module doc). Nothing is patched for a CUDA capture, where Warp and CUDA refuse these."""
    if not device.is_cpu:
        yield
        return
    numpy, syncs = wp.array.numpy, {name: getattr(wp, name) for name in _SYNC_FUNCS}

    def read(arr: wp.array):
        if arr.device.is_cpu:
            reads.append("array.numpy()")
        return numpy(arr)

    def noted(name: str) -> Callable:
        def sync(*args, **kwargs):
            reads.append(f"wp.{name}()")
            return syncs[name](*args, **kwargs)
        return sync

    wp.array.numpy = read
    for name in _SYNC_FUNCS:
        setattr(wp, name, noted(name))
    try:
        yield
    finally:
        wp.array.numpy = numpy
        for name, fn in syncs.items():
            setattr(wp, name, fn)


class GraphCache:
    """One ``PeriodGraph`` per key (``(dt_x, n_e)`` tuples, D4/D21) for one pack."""

    def __init__(self, device: str, *, use_graph: bool | None = None,
                 buffers: Callable[[], Sequence[wp.array]] | None = None) -> None:
        self.device = device
        self.use_graph = use_graph
        self.buffers = buffers
        self._graphs: dict[Hashable, PeriodGraph] = {}

    def get(self, key: Hashable, step_fn: Callable[[], None]) -> PeriodGraph:
        """The graph for ``key``, created around ``step_fn`` on first use."""
        graph = self._graphs.get(key)
        if graph is None:
            graph = PeriodGraph(step_fn, self.device, use_graph=self.use_graph, buffers=self.buffers)
            self._graphs[key] = graph
        return graph

    def invalidate(self) -> None:
        """Invalidate every key (e.g. after a capacity growth)."""
        for graph in self._graphs.values():
            graph.invalidate()

    @property
    def host_syncs(self) -> int:
        return sum(g.host_syncs for g in self._graphs.values())

    def __len__(self) -> int:
        return len(self._graphs)
