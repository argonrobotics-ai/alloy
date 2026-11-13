# SPDX-License-Identifier: Apache-2.0
"""VBD self-contact rows sized to the scene (known gap ``docs/known_gaps/vbd-self-contact-buffer-overflow.md``).

Newton keeps each vertex's colliding triangles and each edge's colliding edges in rows of a fixed size (32 and 64
pairs) and drops the pairs past a row's end. The rows are CSR (``*_offsets`` and ``*_buffer_sizes`` per row), so
each row can have its own size with no kernel change: the writers take a row's size from its offsets, the readers
clamp its count with ``*_buffer_sizes``.

``RowsMixin`` (the default since 2026-11-10, coordinator ruling 04:34Z; ``alloy_self_contact_rows="fixed"`` in the
VBD entry's kwargs, or ``ALLOY_SELF_CONTACT_ROWS=fixed`` in the environment, keeps Newton's fixed rows) uses that:

1. Census at the first step, when the reach is known (``reach``): one detection on the start pose, then every row
   whose count is over ``FILL`` of its size gets ``HEADROOM`` times its count. A row never shrinks under Newton's
   32 / 64, so a scene that stays under ``FILL`` keeps its arrays and steps bit for bit as before.
2. After every detection one launch per family folds each row's count into ``need`` (its largest count so far) and
   into two words per family: the most pairs a growable row dropped, and whether a row passed ``FILL``. No host read.
3. ``RowGuard.grow`` (``alloy.integration.row_growth``, once per period, outside any captured graph) reads those
   words. Rows over ``FILL`` grow before they drop a pair; when a row did drop pairs the period re-runs on the larger
   rows, so no accepted period has a dropped pair (one exception on CUDA, item 4: a row that fills from under half
   full to past its size within one period is seen a period late, and that period stands with the flag).
4. The read costs nothing while no row is over ``ARM`` (half full): on CUDA the words go to pinned memory and the
   previous period's copy is read (``PeriodWords``), so the host never waits for the period it just queued and keeps
   running ahead of the device. A blocking read every period cost a calm 64-world drape 5.7 % on an L4 (rc1, XB4-10).
   Once a late read shows a row over ``ARM`` the reads wait for each period again (grow and re-run as above) until
   ``CALM_PERIODS`` periods pass with none over it. A row that goes from under half full to over full within one
   period still drops pairs without a re-run: health keeps the flag, and from then on the rows are tracked and read
   every period, so the next such drop re-runs.
5. Nothing per row is allocated until a row passes ``ARM``: allocating the two per-row ``need`` arrays at build
   alone (never written or read) made every captured period of that drape 5 % slower on the GPU (L4 probes,
   [internal note] notes/gap-13b/vmprobe3; an allocation-layout effect whose mechanism is not known). Until then the fold
   tracks only the words, and a growth sizes rows from the counts of the last detection.

Limits, both logged once with the fix: a row stops at ``ROW_CAP`` pairs, and all rows together at ``bytes_cap``.
Past either, pairs drop as before and health raises SELF_CONTACT_OVERFLOW. Memory: 8 bytes per pair slot, so a
vertex row of 32 is 256 B and an edge row of 64 is 512 B (``row_bytes``).

Not covered, so they keep the fixed rows (logged): a solver whose rows belong to a collision pipeline (rigs that pass
``collision_pipeline``), and Newton's deterministic modes, whose record counts are compiled from the two sizes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.solvers.vbd")

ROWS_KEY = "alloy_self_contact_rows"
ROWS_ENV = "ALLOY_SELF_CONTACT_ROWS"     # the same choice for every VBD entry of a process (the kwarg wins)
ROWS_FIXED, ROWS_SCENE = "fixed", "scene"
ROWS_DEFAULT = ROWS_SCENE          # coordinator ruling 2026-11-10 04:34Z (laws R1-R4; GPU rows on one L4)
BYTES_KEY = "alloy_self_contact_row_bytes"
ROW_CAP = 256                      # pairs per row (Newton's own, unused, *_max_alloc)
HEADROOM = 1.5                     # a grown row holds this times the most pairs it has had
FILL = 0.75                        # a row fuller than this starts a growth, before it drops a pair
GROW_FILL = 0.5                    # a growth takes every row fuller than this
BYTES_FACTOR, BYTES_FLOOR = 4.0, 256 << 20     # default cap: 4x the fixed rows' bytes, at least 256 MiB
SLOT_BYTES = 8                     # two int32 per pair slot
ARM = 0.5                          # on CUDA, a row fuller than this makes the period-end reads wait (item 4)
CALM_PERIODS = 20                  # periods with no row over ARM before the reads go late again
LATE_ON_CPU = False                # laws: run the late reads on the CPU too (they need no device to be checked)
WORDS = 3                          # per family: most pairs dropped, a row over FILL, a row over ARM
FAMILIES: tuple[tuple[str, str], ...] = (("vertex", "vertex_colliding_triangles"), ("edge", "edge_colliding_edges"))


@wp.kernel
def _fold_rows(count: wp.array[wp.int32], size: wp.array[wp.int32], family: int, row_cap: int, fill_num: int,
               arm_num: int, fill_den: int, track: int, need: wp.array[wp.int32], words: wp.array2d[wp.int32]):
    i = wp.tid()
    c = count[i]
    if track != 0:
        if c > need[i]:
            need[i] = c
    s = size[i]
    if s >= row_cap:
        return
    if c > s:
        wp.atomic_max(words, family, 0, c - s)
    if c * fill_den > s * fill_num:
        wp.atomic_max(words, family, 1, 1)
    if c * fill_den > s * arm_num:
        wp.atomic_max(words, family, 2, 1)


def grown_sizes(need: np.ndarray, size: np.ndarray, row_cap: int = ROW_CAP, headroom: float = HEADROOM,
                fill: float = GROW_FILL) -> np.ndarray:
    """Per-row sizes after a growth: a row whose ``need`` is over ``fill`` of its size gets the larger of twice its
    size and ``headroom * need`` (rounded up to 8, at most ``row_cap``); no row shrinks. ``fill`` is under the mark
    that starts a growth (``FILL``), so one growth also takes the rows that would ask next."""
    need, size = np.asarray(need, dtype=np.int64), np.asarray(size, dtype=np.int64)
    want = np.maximum(2 * size, -(-np.ceil(headroom * need).astype(np.int64) // 8) * 8)
    grow = need > fill * size
    return np.where(grow, np.maximum(size, np.minimum(row_cap, want)), size).astype(np.int32)


def row_bytes(sizes: np.ndarray) -> int:
    """Bytes of one family's pair slots."""
    return SLOT_BYTES * int(np.asarray(sizes, dtype=np.int64).sum())


class PeriodWords:
    """Reads the fold's words at a period's end (module docstring, item 4): now (waits for the device), or late
    (this period's words go to pinned memory and the previous period's copy is read)."""

    def __init__(self, words: wp.array, late: bool) -> None:
        self.words, self.can_late, self.late = words, late, late
        self.calm = 0
        self.blocking = 0                    # reads that waited for the period just queued (laws count them)
        cuda = self._cuda = words.device.is_cuda
        self._host = [wp.zeros(words.shape, dtype=wp.int32, device="cpu", pinned=cuda) for _ in range(2 * late)]
        self._events: list[Any] = [None, None]
        self._queued = [False, False]
        self._turn = 0

    def now(self) -> np.ndarray:
        """This period's words, then cleared (a host read: waits for the device)."""
        self.blocking += 1
        out = self.words.numpy().copy()      # on the CPU numpy() is a live view: copy before zeroing
        self.words.zero_()
        return out

    def read(self) -> tuple[np.ndarray, bool]:
        """The words to act on at this period's end, and whether this period dropped pairs (only this period's own
        drops can be re-run; a late read's are a period old)."""
        if not self.late:
            out = self.now()
            self._settle(out)
            return out, bool(out[:, 0].any())
        slot, prev = self._turn % 2, (self._turn + 1) % 2
        self._turn += 1
        self._queue(slot)
        before = self._take(prev)
        if not before[:, 2].any():
            return before, False
        self.late, self.calm = False, 0      # a row passed half full: wait for this period too, and from now on
        self.blocking += 1
        now = self._take(slot)
        return np.maximum(before, now), bool(now[:, 0].any())

    def _settle(self, out: np.ndarray) -> None:
        if not self.can_late:
            return
        self.calm = 0 if out[:, 2].any() else self.calm + 1
        if self.calm >= CALM_PERIODS:
            self.late, self.calm = True, 0

    def _queue(self, slot: int) -> None:
        stream = wp.get_stream(self.words.device) if self._cuda else None   # the period's own (a pack runs on its own)
        wp.copy(self._host[slot], self.words, stream=stream)
        self.words.zero_()
        self._events[slot] = stream.record_event() if stream is not None else None
        self._queued[slot] = True

    def _take(self, slot: int) -> np.ndarray:
        if not self._queued[slot]:
            return np.zeros(self.words.shape, dtype=np.int32)
        if self._events[slot] is not None:
            wp.synchronize_event(self._events[slot])
        self._queued[slot], self._events[slot] = False, None
        return self._host[slot].numpy().copy()


@dataclass(frozen=True)
class Growth:
    """One growth: per family (rows grown, largest row after), the bytes after, and whether pairs had dropped."""
    rows: dict[str, tuple[int, int]]
    bytes: int
    dropped: bool


class RowState:
    """The rows of one solver: per-row need (from the first row past ``ARM``), the words per family, and the growth."""

    def __init__(self, solver: Any, bytes_cap: int | None = None) -> None:
        self.solver, self.device = solver, solver.model.device
        self.need: dict[str, wp.array] | None = None     # module docstring, item 5
        self.rebind = False                               # need arrays came in: captured folds hold the stand-in
        self._no_need = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.words = wp.zeros((len(FAMILIES), WORDS), dtype=wp.int32, device=self.device)
        self.reader = PeriodWords(self.words, late=wp.get_device(self.device).is_cuda or LATE_ON_CPU)
        self._said_late_drop = False
        start = self.bytes()
        self.bytes_cap = int(bytes_cap) if bytes_cap is not None else int(max(BYTES_FACTOR * start, BYTES_FLOOR))
        self.grown: list[Growth] = []
        self.exhausted = False

    def sizes(self, field: str) -> np.ndarray:
        return getattr(self.solver.trimesh_collision_info, f"{field}_buffer_sizes").numpy().copy()

    def bytes(self) -> int:
        return sum(row_bytes(self.sizes(field)) for _, field in FAMILIES)

    def after_detection(self) -> None:
        """Fold the latest detection into ``words`` (and ``need`` once tracked): one launch per family, no host read."""
        info = self.solver.trimesh_collision_info
        den, track = 1000, int(self.need is not None)
        for k, (name, field) in enumerate(FAMILIES):
            count = getattr(info, f"{field}_count")
            need = self.need[name] if self.need is not None else self._no_need
            wp.launch(_fold_rows, dim=count.shape[0],
                      inputs=[count, getattr(info, f"{field}_buffer_sizes"), k, ROW_CAP, int(FILL * den),
                              int(ARM * den), den, track],
                      outputs=[need, self.words], device=self.device)

    def take_rebind(self) -> bool:
        """Whether the folds changed arrays since the last call (the caller drops its captured graphs)."""
        out, self.rebind = self.rebind, False
        return out

    def _track(self) -> None:
        """Start the per-row need from the last detection's counts (the first row past ``ARM``)."""
        if self.need is not None:
            return
        info = self.solver.trimesh_collision_info
        self.need = {name: wp.clone(getattr(info, f"{field}_count")) for name, field in FAMILIES}
        self.rebind = True

    def _need(self, name: str, field: str) -> np.ndarray:
        count = getattr(self.solver.trimesh_collision_info, f"{field}_count").numpy().copy()
        return np.maximum(self.need[name].numpy(), count) if self.need is not None else count

    def grow(self, now: bool = False) -> Growth | None:
        """Grow the rows that are over ``FILL`` (host reads: call it outside a captured graph), from this period's
        words when ``now`` or the reads wait, else from the previous period's (``PeriodWords``). None when no row
        asked, or when the bytes cap leaves no room."""
        if self.exhausted:
            return None
        if now:
            words = self.reader.now()
            dropped = bool(words[:, 0].any())
        else:
            words, dropped = self.reader.read()
        if words.any():
            self._track()
        if not words[:, :2].any():
            return None
        if words[:, 0].any() and not dropped:
            self._say_late_drop()
        new = {field: grown_sizes(self._need(name, field), self.sizes(field)) for name, field in FAMILIES}
        total = sum(row_bytes(s) for s in new.values())
        if total > self.bytes_cap:
            self._exhaust(total)
            return None
        rows = {name: self._resize(field, new[field]) for name, field in FAMILIES}
        if not any(n for n, _ in rows.values()):
            return None
        out = Growth(rows, total, dropped)
        self.grown.append(out)
        log.info("VBD self-contact rows grown (%s): %s; %.1f MiB of %.1f MiB", "pairs had dropped" if out.dropped
                 else "before a drop", ", ".join(f"{n} {k} rows, largest {m}" for k, (n, m) in rows.items()),
                 total / 2**20, self.bytes_cap / 2**20)
        return out

    def _say_late_drop(self) -> None:
        if not self._said_late_drop:
            self._said_late_drop = True
            log.warning("VBD self-contact rows: a row went from under half full to over full within one period, so "
                        "its pairs dropped before the late read saw it; health keeps SELF_CONTACT_OVERFLOW for that "
                        "period, and from now on the rows are tracked per row and read every period (said once)")

    def _exhaust(self, wanted: int) -> None:
        self.exhausted = True
        log.warning("VBD self-contact rows need %.1f MiB, over the cap of %.1f MiB: they stay as they are and pairs "
                    "past a row's end drop (health: SELF_CONTACT_OVERFLOW). Fix: fewer worlds or a coarser cloth, "
                    "or raise %s in the VBD entry's kwargs", wanted / 2**20, self.bytes_cap / 2**20, BYTES_KEY)

    def _resize(self, field: str, sizes: np.ndarray) -> tuple[int, int]:
        """Give one family its new per-row sizes, in place on the solver's struct (the detector shares it)."""
        info = self.solver.trimesh_collision_info
        old = self.sizes(field)
        changed = int((sizes != old).sum())
        if changed == 0:
            return 0, int(old.max(initial=0))
        offsets = np.zeros(sizes.shape[0] + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        if 2 * offsets[-1] > np.iinfo(np.int32).max:
            raise MemoryError(f"VBD self-contact {field} rows would hold {offsets[-1]} pairs: past int32 offsets")
        setattr(info, field, wp.zeros(int(2 * offsets[-1]), dtype=wp.int32, device=self.device))
        setattr(info, f"{field}_buffer_sizes", wp.array(sizes, dtype=wp.int32, device=self.device))
        setattr(info, f"{field}_offsets", wp.array(offsets.astype(np.int32), dtype=wp.int32, device=self.device))
        getattr(info, f"{field}_count").zero_()      # the new rows are empty until the next detection fills them
        return changed, int(sizes.max(initial=0))


class RowsMixin:
    """Sizes the self-contact rows to the scene (module docstring). Mixed in front of a SolverVBD class, behind
    `reach.ReachMixin`, so the census sees the reach of the first step.

    Two things the caller owes: the first step runs eagerly (the census reads on the host, so it cannot sit inside
    a captured graph), and every step detects self contact before its solve (Alloy's routes do: grown rows are
    empty until the next detection, so a step without one would solve with no self contact)."""

    def __init__(self, model, *args, alloy_self_contact_row_bytes: int | None = None, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.alloy_rows: RowState | None = None
        self._rows_census_done = False
        if not self.particle_enable_self_contact or getattr(self, "collision_pipeline", None) is not None:
            log.info("VBD self-contact rows: fixed (no self contact, or the rows belong to a collision pipeline)")
            return
        if getattr(self, "_particle_contact_use_gather", False):
            # Newton's deterministic modes compile their record counts from the fixed 32 / 64: a grown row would
            # write past the record arrays (WP11's review).
            log.warning("VBD self-contact rows: fixed, since this solver runs a deterministic mode (its record "
                        "arrays are sized for the fixed rows)")
            return
        self.alloy_rows = RowState(self, alloy_self_contact_row_bytes)

    def step(self, state_in, state_out, control, contacts, dt):
        if self.alloy_rows is not None and not self._rows_census_done:
            self._rows_census(state_in)
        return super().step(state_in, state_out, control, contacts, dt)

    def _collision_detection_penetration_free(self, current_state, *, reset_reference: bool = True):
        super()._collision_detection_penetration_free(current_state, reset_reference=reset_reference)
        if self.alloy_rows is not None:
            self.alloy_rows.after_detection()

    def _rows_census(self, state) -> None:
        """Count the start pose's pairs at the step's reach and grow the rows that need it (first step: eager)."""
        self._rows_census_done = True
        det, radius = self.trimesh_collision_detector, self._self_contact_query_radius
        kw = {"min_query_radius": self.particle_rest_shape_contact_exclusion_radius,
              "min_distance_filtering_ref_pos": self.particle_q_rest}
        det.refit(state.particle_q)
        det.vertex_triangle_collision_detection(radius, **kw)
        det.edge_edge_collision_detection(radius, **kw)
        self.alloy_rows.after_detection()
        self.alloy_rows.grow(now=True)


def with_rows(cls: type) -> type:
    """``cls`` with `RowsMixin` in front."""
    return type(f"{cls.__name__}Rows", (RowsMixin, cls), {})


def pop_rows(kw: dict[str, Any]) -> bool:
    """Pop the rows mode from a VBD entry's kwargs; True for rows sized to the scene."""
    mode = kw.pop(ROWS_KEY, None) or os.environ.get(ROWS_ENV) or ROWS_DEFAULT
    if mode not in (ROWS_FIXED, ROWS_SCENE):
        raise ValueError(f"{ROWS_KEY} (or ${ROWS_ENV}) = {mode!r}: use {ROWS_FIXED!r} or {ROWS_SCENE!r}")
    scene = mode == ROWS_SCENE and bool(kw.get("particle_enable_self_contact"))
    if not scene:
        kw.pop(BYTES_KEY, None)
    return scene


__all__ = ["ARM", "BYTES_KEY", "CALM_PERIODS", "FAMILIES", "FILL", "GROW_FILL", "Growth", "HEADROOM", "PeriodWords",
           "ROWS_DEFAULT", "ROWS_ENV", "ROWS_KEY", "ROWS_SCENE", "ROW_CAP", "RowState", "RowsMixin", "grown_sizes",
           "pop_rows", "row_bytes", "with_rows"]
