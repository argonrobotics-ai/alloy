# SPDX-License-Identifier: Apache-2.0
"""SolverPuncture: SolverVBDMomentum plus needle puncture and suture anchors (SC11; design docs/notes/sc11-puncture-
design.md).

``step`` = [events: TIP/puncture/growth/handover, crossing tests] -> [particle -> anchor lists] -> VBD step with the
anchors' blocks gathered into each colour before its solve (SolverVBDAlloy's ``after_accumulate`` colour callback,
SCR-SC11-2; a thread's colour is solved whole there by chain.py) -> [per-anchor forces, needle loads].
A model with no puncturable tet (``puncture:fp`` > 0) or no needle steps as SolverVBDMomentum (default off).
Needles carried by a body need ``integrate_with_external_rigid_solver=True`` (one-way: the body pose comes from the
rigid solver; the loads are reported in ``loads`` and are not fed back here).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.solvers.vbd import SolverVBDMomentum

from . import chain, csr, events, monitor, terms
from .config import NeedleSpec, PunctureConfig
from .ledger import PunctureLedger
from .needle import NeedleArrays
from .obstacles import ObstacleConfig
from .obstacles import build as build_obstacles
from .topology import build_topology, tissue_values, upload

log = logging.getLogger("alloy.puncture")


class SolverPuncture(SolverVBDMomentum):
    """SolverVBDMomentum with puncture anchors. Extra keywords ``puncture`` (PunctureConfig), ``needles``
    (NeedleSpec list, sorted by world) and ``obstacles`` (rigid shapes the threads touch, obstacles.py); everything
    else goes to SolverVBD."""

    def __init__(self, model: newton.Model, *, puncture: PunctureConfig | None = None,
                 needles: Sequence[NeedleSpec] = (), obstacles: ObstacleConfig | None = None, **vbd_kwargs) -> None:
        super().__init__(model, **vbd_kwargs)
        self.puncture = puncture if puncture is not None else PunctureConfig()
        self.obstacle_config = obstacles if obstacles is not None else ObstacleConfig()
        self.topology = build_topology(model)
        self.needle_specs = list(needles)
        self._on = self.topology.any and bool(self.needle_specs)
        if not self._on:
            log.debug("SolverPuncture: no puncturable tissue or no needle; stepping as SolverVBDMomentum")
            return
        if any(s.body >= 0 for s in needles) and not self.integrate_with_external_rigid_solver:
            raise AssemblyError("SolverPuncture: needles on bodies need integrate_with_external_rigid_solver=True")
        worlds = [s.world for s in needles]
        if worlds != sorted(worlds):
            raise AssemblyError("SolverPuncture: needles must be sorted by world")
        self._alloc(model)
        self._dt = 0.0
        # Anchor forces exchange momentum with the needle, but every component they touch is blocked from the coarse
        # step (_mark_blocked), and thread particles belong to no component: external=False keeps the coarse step.
        self.add_colour_callback(self._after_accumulate, external=False, when="after_accumulate")
        log.info("SolverPuncture: %d needles, %d tissue tets, %d surface faces, %d anchor slots per world",
                 len(needles), self.topology.tets.size, self.topology.face_tri.shape[0], self.puncture.anchors_per_world)

    # --- construction -------------------------------------------------------------------------------------------

    def _alloc(self, model: newton.Model) -> None:
        dev, top, cfg = model.device, self.topology, self.puncture
        W, A, n = top.world_count, cfg.anchors_per_world, len(self.needle_specs)
        i32 = lambda a: wp.array(np.asarray(a, np.int32), dtype=wp.int32, device=dev)   # noqa: E731
        self.needles = NeedleArrays.build(self.needle_specs, dev)
        self.surface = upload(top, dev)
        self.tet_world = i32(np.repeat(np.arange(W), np.diff(top.tet_start)))
        self.face_world = i32(np.repeat(np.arange(W), np.diff(top.face_start)) if top.face_tri.size else [0])
        ts = terms.Tissue()
        ts.tet_indices, ts.tet_poses = model.tet_indices, model.tet_poses
        for name in ("fp", "fcut", "flen", "eta"):
            setattr(ts, name, wp.array(tissue_values(model, name), dtype=float, device=dev))
        self.tissue = ts
        an = terms.Anchors()
        an.state, an.needle, an.tet = (wp.zeros(W * A, dtype=wp.int32, device=dev) for _ in range(3))
        an.seg, an.cut = wp.full(W * A, -1, dtype=wp.int32, device=dev), wp.zeros(W * A, dtype=wp.int32, device=dev)
        an.w = wp.zeros(W * A, dtype=wp.vec4, device=dev)
        an.ell, an.alpha = wp.zeros(W * A, dtype=float, device=dev), wp.zeros(W * A, dtype=float, device=dev)
        an.tdir = wp.zeros(W * A, dtype=wp.vec3, device=dev)
        an.trk, an.hole = wp.zeros(W * A, dtype=wp.int32, device=dev), wp.zeros(W * A, dtype=wp.int32, device=dev)
        self.anchors = an
        pr = terms.Params()
        pr.k_n, pr.k_tip, pr.h_a, pr.eps_v, pr.mu = cfg.k_n, cfg.k_tip, cfg.h_a, cfg.eps_v, cfg.mu
        pr.lookahead = cfg.lookahead
        self.params = pr
        tr = events.Tracks()
        tr.state, tr.want, tr.loc_tet, tr.best_f, tr.punctures, tr.flags, tr.count, tr.need_newest = (
            wp.zeros(n, dtype=wp.int32, device=dev) for _ in range(8))
        tr.tip, tr.newest, tr.hole, tr.guard_j = (wp.full(n, -1, dtype=wp.int32, device=dev) for _ in range(4))
        tr.guard_v, tr.guard_on = wp.zeros((n, 4), dtype=wp.int32, device=dev), wp.zeros(n, dtype=wp.int32, device=dev)
        tr.tent_s = wp.zeros(n, dtype=float, device=dev)
        tr.fill = wp.full(n, -1, dtype=wp.int32, device=dev)
        tr.best_t, tr.fax, tr.want_s, tr.seg_t0 = (wp.zeros(n, dtype=float, device=dev) for _ in range(4))
        tr.seg_o, tr.tip_tet = wp.zeros(n, dtype=wp.vec3, device=dev), wp.zeros(n, dtype=wp.int32, device=dev)
        tr.needle_start = i32(np.searchsorted(np.asarray([s.world for s in self.needle_specs]), np.arange(W + 1)))
        tr.capacity = A
        self.tracks = tr
        self._swage_ties(an, tr, W, A, dev)
        P = int(model.particle_count)
        self._counts = wp.zeros(P + 1, dtype=wp.int32, device=dev)
        self._row = wp.zeros(P + 1, dtype=wp.int32, device=dev)
        self._cursor = wp.zeros(P + 1, dtype=wp.int32, device=dev)
        self._ent = wp.zeros(W * A * csr.SLOTS, dtype=wp.int32, device=dev)
        self._build_chain(model, dev)
        self._f_anchor = wp.zeros(W * A, dtype=wp.vec3, device=dev)
        self._off = wp.zeros(W * A, dtype=float, device=dev)
        ld = csr.Loads()
        ld.force, ld.torque, ld.thread = (wp.zeros(n, dtype=wp.vec3, device=dev) for _ in range(3))
        ld.axial, ld.length = wp.zeros(n, dtype=float, device=dev), wp.zeros(n, dtype=float, device=dev)
        ld.anchors = wp.zeros(n, dtype=wp.int32, device=dev)
        ld.offset = wp.zeros(W, dtype=float, device=dev)
        self.loads = ld
        self._no_bodies = wp.zeros(1, dtype=wp.transform, device=dev)
        self._inside = wp.zeros(n * monitor.SAMPLES, dtype=wp.int32, device=dev)
        self.untracked_count = wp.zeros(n, dtype=wp.int32, device=dev)
        self.untracked_depth = wp.zeros(n, dtype=float, device=dev)
        self.world_untracked_depth = wp.zeros(W, dtype=float, device=dev)    # health: WP5 plug-in check
        self.ledger = PunctureLedger(self) if self.puncture.ledger else None
        self.hole_filter = None

    def _swage_ties(self, an, tr, W: int, A: int, dev) -> None:
        """Two permanent SWAGE anchors per needle with a thread, in the first slots of its world: the thread's first
        particle at the swage and its second one thread spacing on along the arc (a coaxial crimp, not a pin joint)."""
        state, needle, seg, alpha = an.state.numpy(), an.needle.numpy(), an.seg.numpy(), an.alpha.numpy()
        q = self.model.particle_q.numpy()
        used = np.zeros(W, np.int32)
        for i, sp in enumerate(self.needle_specs):
            if sp.thread_count > 1:
                gap = float(np.linalg.norm(q[sp.thread_start + 1] - q[sp.thread_start]))
                for k, s in ((0, 0.0), (1, gap)):
                    j = sp.world * A + used[sp.world]
                    state[j], needle[j], seg[j], alpha[j] = int(terms.SWAGE), i, k, s
                    used[sp.world] += 1
        an.state.assign(state)
        an.needle.assign(needle)
        an.seg.assign(seg)
        an.alpha.assign(alpha)
        tr.used = wp.array(used, dtype=wp.int32, device=dev)

    def health_depth_checks(self):
        """WP5 plug-in check (SCR-SC11-1 item 4): needle inside tissue farther than a track's reach from every anchor
        of it, per world [m]; deeper than 0.5 mm is PEN_FAIL. Empty when the plug-in is off."""
        if not self._on:
            return ()
        from alloy.precision.types import DepthCheck

        return (DepthCheck("puncture_untracked_depth", self.world_untracked_depth, 0.5e-3),)

    def attach_hole_filter(self, region=None, r_hole: float = 1.0e-3) -> None:
        """Thread-vs-pad self-contact ignored near open holes (holes.py); needs particle_enable_self_contact."""
        from .holes import HoleContactFilter
        ranges = [(sp.thread_start, sp.thread_count) for sp in self.needle_specs if sp.thread_count > 1]
        self.hole_filter = HoleContactFilter(self, ranges, region, r_hole)

    # --- step ---------------------------------------------------------------------------------------------------

    def _poses(self, state_in, state_out):
        if self.model.body_count == 0 or state_in.body_q is None:
            return self._no_bodies, self._no_bodies
        return state_in.body_q, state_out.body_q

    def step(self, state_in, state_out, control, contacts, dt: float, *, observables=None) -> None:
        if not self._on:
            super().step(state_in, state_out, control, contacts, dt, observables=observables)
            return
        q0, q1 = self._poses(state_in, state_out)
        lg = self.ledger
        if lg is not None:
            lg.before_events(state_in.particle_q, q0)
        self._events(state_in.particle_q, q0, q1)
        if lg is not None:
            lg.after_events(state_in.particle_q, q0)
        if self.hole_filter is not None:
            self.hole_filter.update(self, state_in.particle_q)
        self._build_lists()
        self._q0, self._q1, self._dt = q0, q1, dt
        super().step(state_in, state_out, control, contacts, dt, observables=observables)
        self._measure(state_out.particle_q, q1, dt)
        if lg is not None:
            lg.after_step(state_out, q1, q0, dt)

    def _events(self, x0, q0, q1) -> None:
        dev, W = self.model.device, self.topology.world_count
        an, ts, nd, pr, tr, sf = self.anchors, self.tissue, self.needles.dev, self.params, self.tracks, self.surface
        wp.launch(events.update, dim=W, inputs=[sf, an, ts, nd, pr, tr, x0, q0, q1], device=dev)
        wp.launch(events.locate, dim=self.topology.tets.size, inputs=[sf, ts, nd, tr, x0, self.tet_world, q0],
                  device=dev)
        F = int(self.topology.face_tri.shape[0])
        if F:
            for kern in (events.cross_param, events.cross_pick):
                wp.launch(kern, dim=F, inputs=[sf, nd, pr, tr, x0, self.face_world, q0, q1], device=dev)
        wp.launch(events.create, dim=W, inputs=[sf, an, ts, nd, pr, tr, x0, q0, q1], device=dev)

    def _build_lists(self) -> None:
        dev, W = self.model.device, self.topology.world_count
        an, ts, nd, tr = self.anchors, self.tissue, self.needles.dev, self.tracks
        self._counts.zero_()
        wp.launch(csr.count, dim=W, inputs=[an, ts, nd, tr, self._counts], device=dev)
        wp.utils.array_scan(self._counts, self._row, inclusive=False)
        wp.copy(self._cursor, self._row)
        wp.launch(csr.fill, dim=W, inputs=[an, ts, nd, tr, self._cursor, self._ent], device=dev)

    def _build_chain(self, model, dev) -> None:
        """Chain arrays for the threads (chain.py) and the colours that hold them."""
        threads = [(i, sp) for i, sp in enumerate(self.needle_specs) if sp.thread_count > 1]
        q = model.particle_q.numpy()
        if any(sp.thread_count > int(chain.MAXN) for _, sp in threads):
            raise AssemblyError(f"SolverPuncture: a thread has more than {int(chain.MAXN)} particles")
        rest = [float(np.linalg.norm(np.diff(q[sp.thread_start:sp.thread_start + sp.thread_count], axis=0),
                                     axis=1).mean()) for _, sp in threads]
        arr = lambda v, dt: wp.array(np.asarray(v if v else [0], dtype=dt), device=dev)   # noqa: E731
        ch = chain.Chain()
        ch.start = arr([sp.thread_start for _, sp in threads], np.int32)
        ch.count = arr([sp.thread_count for _, sp in threads], np.int32)
        ch.rest = arr(rest, np.float32)
        ch.ks = arr([sp.thread_ea / r for (_, sp), r in zip(threads, rest)], np.float32)
        ch.kb = arr([sp.thread_ei / r**3 for (_, sp), r in zip(threads, rest)], np.float32)
        ch.needle = arr([i for i, _ in threads], np.int32)
        ch.row, ch.ent = self._row, self._ent
        self.chain = ch
        self.chain_world = arr([sp.world for _, sp in threads], np.int32)
        self.obstacles = build_obstacles(self.obstacle_config, [sp.world for _, sp in threads], dev)
        self._n_threads = len(threads)
        members = {sp.thread_start + k for _, sp in threads for k in range(sp.thread_count)}
        groups = [np.asarray(g.numpy()) for g in model.particle_color_groups]
        self._thread_colours = {c for c, g in enumerate(groups) if g.size and int(g[0]) in members}
        for c in self._thread_colours:
            if not set(groups[c].tolist()) <= members:
                raise AssemblyError("SolverPuncture: a thread shares a colour with other particles; build the model "
                                    "with thread.isolate_thread_colours")
        self._chain_A = wp.zeros((max(self._n_threads, 1), int(chain.MAXN), 5), dtype=wp.mat33, device=dev)
        self._chain_g = wp.zeros((max(self._n_threads, 1), int(chain.MAXN)), dtype=wp.vec3, device=dev)

    def _after_accumulate(self, solver, color: int, state_in, contacts) -> None:
        """Colour callback (after Newton's accumulation, before the colour's solve): a thread's colour gets the whole
        chain step (chain.py); any other colour adds its anchors' force and Hessian blocks."""
        dev, dt = self.model.device, self._dt
        if color in self._thread_colours:
            wp.launch(chain.chain_step, dim=self._n_threads,
                      inputs=[dt, self.chain, self.anchors, self.tissue, self.needles.dev, self.params,
                              state_in.particle_q, self.particle_q_prev, self.inertia, self.model.particle_mass,
                              self._q1, self._q0, self.particle_forces, self.particle_hessians, self.obstacles,
                              self.model.particle_radius, self._chain_A, self._chain_g], device=dev)
            return
        ids = self.model.particle_color_groups[color]
        wp.launch(terms.gather_colour, dim=ids.size,
                  inputs=[dt, ids, self.anchors, self.tissue, self.needles.dev, self.params, state_in.particle_q,
                          self.particle_q_prev, self._q1, self._q0, self._row, self._ent],
                  outputs=[self.particle_forces, self.particle_hessians], device=dev)

    def _mark_blocked(self, contacts) -> None:
        super()._mark_blocked(contacts)
        if self._on:
            wp.launch(csr.mark_components, dim=self.anchors.state.shape[0],
                      inputs=[self.anchors, self.tissue, self.needles.dev, self.coarse.comp_of, self.coarse.blocked],
                      device=self.model.device)

    def _measure(self, x, q1, dt: float) -> None:
        dev = self.model.device
        an, ts, nd, pr, tr = self.anchors, self.tissue, self.needles.dev, self.params, self.tracks
        wp.launch(terms.anchor_forces, dim=an.state.shape[0],
                  inputs=[dt, an, ts, nd, pr, x, self.particle_q_prev, q1, self._q0, self._f_anchor, self._off],
                  device=dev)
        wp.launch(csr.loads, dim=self.topology.world_count,
                  inputs=[an, ts, nd, tr, x, q1, self._f_anchor, self._off, self.loads], device=dev)
        wp.launch(monitor.samples_inside, dim=self.topology.tets.size,
                  inputs=[self.surface, self.tet_world, tr.needle_start, ts, nd, x, q1, self._inside], device=dev)
        wp.launch(monitor.untracked, dim=self.topology.world_count,
                  inputs=[an, ts, nd, tr.needle_start, tr.capacity, tr.used, pr.h_a, tr.tent_s, x, q1, self._inside,
                          self.untracked_count, self.untracked_depth, self.world_untracked_depth], device=dev)
