# SPDX-License-Identifier: Apache-2.0
"""FrameExporter (SCR-WP13-EXPORT): picked worlds' frames for a live viewer, without slowing or changing the run.

- ``capture()`` runs in the training thread, between steps. It is the only method that issues CUDA work.
  - It refuses (returns False, issues nothing) while the device is capturing a graph.
  - It launches the gathers on the simulation's current stream, ordered after the step's graph launch, into a
    staging slot.
  - It then makes one async copy into the slot's pinned host buffers, on a copy stream that waits on an event.
  - It also checks, by event query, which earlier copies have landed.
- ``poll()`` may run in any thread and never blocks. It makes no CUDA call: it reads host flags and pinned memory only,
  so a publisher thread cannot disturb a graph capture. It returns the newest landed frames, or None.
  - A frame becomes pollable at the first ``capture()`` (or ``flush()``) after its copy lands.
- Two slots. A capture takes a free slot, or the oldest landed one (that frame is dropped, unpolled). With both busy
  it drops the new frame. On the CPU everything is synchronous.
- The gathers only read the simulation, so stepping with export on equals export off, bit for bit.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import warp as wp

from alloy.export import frames as F
from alloy.export import kernels as K
from alloy.export.plan import ExportPlan, build_plan

MAX_WORLDS = 8                                          # picked worlds per exporter (LIVE brief: <= 8 per VM)
_SPECS = (("body_q", wp.transform), ("mesh", wp.vec3), ("part", wp.vec3), ("flags", wp.int32), ("rod", wp.vec3),
          ("active", wp.int8), ("counts", wp.int32), ("tet_live", wp.int8), ("mesh_flags", wp.int32),
          ("cut_live", wp.int8))


def version(host: dict[str, np.ndarray], p: int) -> int:
    """Picked world ``p``'s topology version: broken bonds + retired bodies * 2**20 + dead cuttable tets * 2**40.
    A virtual-node split kills exactly one cuttable whole tet, folded side or not (WP10), so the version rises by
    2**40 per split and never runs back within an episode (the fragments are not counted: they are only drawn).
    ``plan._bounds`` keeps every count below its field (bonds and bodies < 2**20, cuttable tets < 2**23): the
    version fits an int64 and two topologies never read alike."""
    dead = int((host["cut_live"][p] == 0).sum()) if host["cut_live"].size else 0
    return int(host["counts"][p, 0]) + (int(host["counts"][p, 1]) << 20) + (dead << 40)


def _picks(sim: Any, worlds: list[int] | int) -> list[int]:
    """The picked worlds as a list, checked (alloy errors with a fix, AGENTS.md section 9)."""
    from alloy.api.errors import FeatureUnavailableError, SceneError  # noqa: PLC0415

    if getattr(sim, "engine", None) is None:
        raise FeatureUnavailableError("frame export needs a one-process Sim (this one steps its worlds in worker "
                                      "processes)", "build the run you export without processes=, or export inside "
                                      "each worker", field="sim")
    picked = [int(worlds)] if isinstance(worlds, (int, np.integer)) else [int(w) for w in worlds]
    if not picked or len(picked) > MAX_WORLDS or len(set(picked)) != len(picked):
        raise SceneError(f"pick 1 to {MAX_WORLDS} distinct worlds (got {picked})",
                         f"pass up to {MAX_WORLDS} different world indices, e.g. worlds=[0, 5]", field="worlds")
    if min(picked) < 0 or max(picked) >= sim.worlds:
        raise SceneError(f"worlds must be in [0, {sim.worlds}) (got {picked})",
                         f"pick indices from 0 to {sim.worlds - 1}", field="worlds")
    return picked


def _capturing(device: Any) -> bool:
    """The device is recording a CUDA graph (no CUDA work may be issued then)."""
    return bool(wp.get_device(device).is_capturing)


class _Slot:
    """Device staging and pinned host twins for one frame of every picked world."""

    def __init__(self, shapes: dict[str, tuple[int, ...]], device: Any) -> None:
        cuda = wp.get_device(device).is_cuda
        self.dev = {k: wp.zeros(shapes[k], dtype=dt, device=device) for k, dt in _SPECS}
        self.host = {k: wp.zeros(shapes[k], dtype=dt, device="cpu", pinned=cuda) for k, dt in _SPECS} if cuda else None
        self.gathered = wp.Event(device) if cuda else None
        self.copied = wp.Event(device) if cuda else None
        self.state = "free"                     # free -> busy (staging) -> issued (copy queued) -> landed -> reading
        self.meta: tuple[int, float] = (0, 0.0)
        self.seq = 0

    def harvest(self) -> None:
        """Training thread only (an event query is a CUDA call): an issued slot whose copy has finished is landed."""
        if self.state == "issued" and (self.copied is None or self.copied.is_complete):
            self.state = "landed"


class FrameExporter:
    """Frames of ``worlds`` (at most ``MAX_WORLDS``) of an ``alloy.api`` Sim; particle sets capped at ``max_points``
    per world. Example::

        exp = FrameExporter(sim, worlds=[0, 5])
        sim.step(1); exp.capture()   # training thread, between steps
        frames = exp.poll()          # any thread: None until a copy has landed; then one Frame per picked world
    """

    def __init__(self, sim: Any, worlds: list[int] | int, max_points: int = 20000) -> None:
        worlds = _picks(sim, worlds)
        self.sim, self.plan = sim, build_plan(sim, worlds, max_points)
        self.device = sim.engine.model.device
        self._rows = self._device_rows(self.plan)
        P = len(worlds)
        shapes = {"body_q": self.plan.body_rows.shape, "mesh": self.plan.mesh_rows.shape,
                  "part": self.plan.part_rows.shape, "flags": self.plan.part_rows.shape,
                  "rod": self.plan.rod_rows.shape, "active": self.plan.body_rows.shape, "counts": (P, 2),
                  "tet_live": self.plan.tet_rows.shape, "mesh_flags": self.plan.mesh_rows.shape,
                  "cut_live": self.plan.cut_rows.shape}
        self._slots = [_Slot(shapes, self.device) for _ in range(2)]
        cuda = wp.get_device(self.device).is_cuda
        self._copy_stream = wp.Stream(self.device) if cuda else None
        self._radius = self._radii()
        self._sent: dict[int, int] = {}                 # picked index -> topology version last delivered
        self._lock, self._seq = threading.Lock(), 0
        self.dropped = 0                                # frames given up: no free slot, or never polled
        self.refused = 0                                # captures refused because a graph capture was running

    def _device_rows(self, plan: ExportPlan) -> dict[str, Any]:
        arr = lambda a: wp.array(np.ascontiguousarray(a, np.int32), dtype=wp.int32, device=self.device)  # noqa: E731
        rows = {"body": arr(plan.body_rows), "mesh": arr(plan.mesh_rows), "part": arr(plan.part_rows),
                "rod": arr(plan.rod_rows), "bonds": arr(plan.bond_range), "tet": arr(plan.tet_rows),
                "cut": arr(plan.cut_rows),
                "rod_local": wp.array(plan.rod_local, dtype=wp.vec3, device=self.device)}
        rows["flags"] = [arr(s.flag_rows) for s in plan.sets]
        return rows

    def _radii(self) -> dict[str, Any]:
        """Particle sets' radii at their subset ([k] each), and each mesh's particle radius (``mesh:<name>``: half
        its thickness, WP11)."""
        r = self.sim.engine.model.particle_radius
        radius = r.numpy() if r is not None and r.shape[0] else None
        out: dict[str, Any] = {s.name: (radius[self.plan.part_rows[0, self.plan.part[s.name]]].astype(np.float32)
                                        if radius is not None else np.zeros(len(s.local), np.float32))
                               for s in self.plan.sets}
        for name, sl in self.plan.mesh.items():
            out[f"mesh:{name}"] = float(radius[self.plan.mesh_rows[0, sl.start]]) if radius is not None else 0.0
        return out

    # --- capture (training thread) -------------------------------------------------------------------------------

    def capture(self) -> bool:
        """Stage the current state of the picked worlds (non-blocking; training thread, between steps). False when the
        frame was dropped, or refused because the device is capturing a graph (then nothing is issued)."""
        if _capturing(self.device):
            self.refused += 1
            return False
        with self._lock:
            for sl in self._slots:
                sl.harvest()
            slot = self._free_slot()
            if slot is None:
                self.dropped += 1
                return False
            slot.state = "busy"
            self._seq += 1
            slot.seq = self._seq
            slot.meta = (int(round(self.sim.t / self.sim.dt)), float(self.sim.t))
        self._gather(slot)
        if slot.host is not None:
            cur = wp.get_stream(self.device)
            cur.record_event(slot.gathered)
            self._copy_stream.wait_event(slot.gathered)
            for k, dev in slot.dev.items():
                if dev.size:
                    wp.copy(slot.host[k], dev, stream=self._copy_stream)
            self._copy_stream.record_event(slot.copied)
        with self._lock:
            slot.state = "issued"
            if slot.host is None:
                slot.state = "landed"                   # the CPU path is synchronous
        return True

    def flush(self) -> None:
        """Wait for every issued copy and make it pollable (training thread; blocks)."""
        if _capturing(self.device):
            return
        wp.synchronize_device(self.device)
        with self._lock:
            for sl in self._slots:
                sl.harvest()

    def _free_slot(self) -> _Slot | None:
        free = [s for s in self._slots if s.state == "free"]
        if free:
            return free[0]
        ready = sorted((s for s in self._slots if s.state == "landed"), key=lambda s: s.seq)
        if ready:
            self.dropped += 1                           # an unpolled frame gives way to the newer one
            return ready[0]
        return None

    def _gather(self, slot: _Slot) -> None:
        st, d, r, dev = self.sim.engine.state, slot.dev, self._rows, self.device
        launch = lambda k, a, ins, out: (wp.launch(k, dim=a.shape, inputs=ins, outputs=[out], device=dev)  # noqa: E731
                                         if a.size else None)
        launch(K.gather_transforms, d["body_q"], [st.body_q, r["body"]], d["body_q"])
        launch(K.gather_points, d["mesh"], [st.particle_q, r["mesh"]], d["mesh"])
        launch(K.gather_points, d["part"], [st.particle_q, r["part"]], d["part"])
        for s, rows in zip(self.plan.sets, r["flags"], strict=True):
            col = self.plan.part[s.name].start
            if rows.size:
                wp.launch(K.gather_flags, dim=rows.shape, inputs=[s.flags, rows, col], outputs=[d["flags"]], device=dev)
        launch(K.rod_points, d["rod"], [st.body_q, r["rod"], r["rod_local"]], d["rod"])
        m = self.sim.engine.model
        launch(K.tet_live, d["tet_live"], [m.tet_materials, r["tet"]], d["tet_live"])
        launch(K.tet_live, d["cut_live"], [m.tet_materials, r["cut"]], d["cut_live"])
        if d["mesh_flags"].size:                        # virtual-node copies gain ACTIVE when a split wakes them
            wp.launch(K.gather_flags, dim=r["mesh"].shape, inputs=[m.particle_flags, r["mesh"], 0],
                      outputs=[d["mesh_flags"]], device=dev)
        self._topology(d)

    def _topology(self, d: dict[str, Any]) -> None:
        topo = getattr(self.sim.engine, "topology", None)
        r, dev = self._rows, self.device
        bonds = getattr(topo, "bonds", None)            # a virtual-node cutter has none: its cuts are dead tets
        if bonds is None or not bonds.count or getattr(topo, "body_active", None) is None:
            if d["active"].size:
                d["active"].fill_(1)
            d["counts"].zero_()
            return
        active = topo.body_active
        if d["active"].size:
            wp.launch(K.gather_bytes, dim=d["active"].shape, inputs=[active, r["body"]], outputs=[d["active"]],
                      device=dev)
        wp.launch(K.topology_counts, dim=len(self.plan.worlds), device=dev,
                  inputs=[r["bonds"], topo.tables.world_order, topo.bonds.intact, r["body"], active,
                          int(self.plan.body_rows.shape[1])], outputs=[d["counts"]])

    # --- poll (any thread) ---------------------------------------------------------------------------------------

    def poll(self) -> list[F.Frame] | None:
        """The newest landed frames (one per picked world), or None (no CUDA call; any thread)."""
        with self._lock:
            done = [s for s in self._slots if s.state == "landed"]
            if not done:
                return None
            slot = max(done, key=lambda s: s.seq)
            for s in done:
                if s is not slot:
                    s.state, self.dropped = "free", self.dropped + 1
            slot.state = "reading"                      # a capture leaves it alone while we copy it out
        src = slot.host if slot.host is not None else slot.dev  # pinned host memory (or CPU arrays): no CUDA call
        host = {k: src[k].numpy().copy() for k, _ in _SPECS}
        meta = slot.meta
        versions = [version(host, p) for p in range(len(self.plan.worlds))]
        with self._lock:
            slot.state = "free"
            fresh = {p for p, v in enumerate(versions) if self._sent.get(p) != v}
            for p in fresh:
                self._sent[p] = versions[p]
        return F.assemble(self.plan, host, meta, fresh, self._radius, versions)

    def resend_topology(self) -> None:
        """Every picked world's next delivered frame carries its topology again, e.g. after a viewer reconnects
        (cheaper than a new exporter: no plan, no allocation). No CUDA call; any thread."""
        with self._lock:
            self._sent.clear()


__all__ = ["MAX_WORLDS", "FrameExporter"]
