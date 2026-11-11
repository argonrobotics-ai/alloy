# SPDX-License-Identifier: Apache-2.0
"""TopologyController: bond breaking, tear, cut and crumble on a fixed topology (SPEC §3.8, D19, TOP-1..TOP-5).

Each step in §2.3 calls the controller twice.
- ``pre_step(state, contacts, dt)`` runs before the solve. It places the blade edges and applies TET_CUT gating
  and cut damage, which changes stiffnesses only.
- ``apply(state_next)`` runs after the solve. It tests every bond for a break against the new state. It then
  commits at most ``max_events_per_step`` events per world in index order (the excess is counted as overflow and
  retried), records the ledger before, toggles native elements, converts and parks crumbled chunks, and records
  the ledger after.
- ``request_break`` / ``request_crumble`` queue external events; the next ``apply`` commits them under the same
  cap and ledger (SCR-DC5-1).
- ``attach_self_contact(vbd)`` makes a self-contact SolverVBD ignore a seam twin's partner faces only while the twins
  are joined (SCR-SC5-2, ``seam_contact``); ``pre_step`` refreshes its filter lists from the bond state.
- ``coupled=<stepper>``: a SolverMuJoCo entry that runs on a coupled entry view gets weld breaks through the view
  (SCR-DC5-21, ``solver_hooks``).
- ``cut_front_blade`` / ``blade_half_thickness`` give the health monitor its cut-front predicate (SCR-SC5-3,
  ``cut_front``): which twin a Blade is cutting this step, and that blade's half-thickness.

Everything is allocated at construction. Steps only launch kernels (one thread per bond, source, particle or
world; no float atomics), so the sequence can be captured in a graph and is bitwise repeatable.
VBD adjacency and colouring are never touched (the fixed-arrays test checks this).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology import breaking, crumble_runtime as crt, cut, tear
from alloy.topology.cut_front import CutFront
from alloy.topology.ledger import LedgerState, WorldIndex
from alloy.topology.seam_contact import SeamContactFilter
from alloy.topology.solver_hooks import SolverHooks
from alloy.topology.tables import build_bond_tables
from alloy.topology.types import Blade, BondKind, BondSet, TopologyBudget, TopologyLedger

log = logging.getLogger("alloy.topology")


class TopologyController:
    @dataclass(frozen=True)
    class Config:
        blade_reach: float | None = None                          # [m] extra blade contact reach; None = cut default
        cut_alignment: float | None = cut.SEAM_ALIGNMENT   # a blade cuts only seams with |n . its face| >= this
        # (cut.py; BladeContact's min_alignment); None = any seam its edge reaches (the rule before 2026-11-09)
        park_offset: tuple[float, float, float] = (100.0, 0.0, 0.0)   # [m] per parking slot, same height

    def __init__(self, model: newton.Model, bonds: BondSet, budget: TopologyBudget, blades: Sequence[Blade] = (),
                 entries: Mapping[str, newton.solvers.SolverBase] | None = None, *, config: Config | None = None,
                 coupled: object | None = None):
        """``entries``: the live solver per entry name. ``coupled``: the coupled stepper those entries run under
        (Scheduler, OneWayFollower, stock proxy), needed when a SolverMuJoCo entry runs on its entry view and the
        model has equality bonds: without it weld breaks never reach MuJoCo, and a warning says so (SCR-DC5-21)."""
        self.model, self.bonds, self.budget = model, bonds, budget
        self.config = config or TopologyController.Config()
        self.world_count = max(int(model.world_count), 1)
        dev = model.device
        self.tables = build_bond_tables(model, bonds, self.world_count)
        self.blades = cut.build_blade_tables(model, blades, self.world_count)
        kind = bonds.kind.numpy().astype(np.int64)
        self.crumble = crt.build_crumble_tables(model, kind, bonds.a.numpy(), bonds.b.numpy(), self.world_count)
        self.index = WorldIndex.from_model(model)
        self._ledger = LedgerState(model, self.index)
        n = max(bonds.count, 1)
        self.phase = wp.array(np.where(bonds.intact.numpy() == 0, 2, 0).astype(np.int8) if bonds.count
                              else np.zeros(1, np.int8), dtype=wp.int8, device=dev)
        self._want, self._commit = (wp.zeros(n, dtype=wp.int8, device=dev) for _ in range(2))
        self.strain, self.force = (wp.zeros(n, dtype=wp.float32, device=dev) for _ in range(2))
        # a bond's force and break force supplied by its entry solver (-1 = none): the film's seams, held by its strain
        # projection, carry no spring force (alloy/solvers/film/seams.py, SolverFilm.topology_link)
        self.ext_force, self.ext_threshold = (wp.full(n, -1.0, dtype=wp.float32, device=dev) for _ in range(2))
        self._linked = False
        self.contact = wp.full(n, -1, dtype=wp.int32, device=dev)
        self.cut_done = wp.zeros(n, dtype=wp.int8, device=dev)        # 1 = broken by a blade cut
        self._request = wp.zeros(n, dtype=wp.int8, device=dev)        # 1 = external break pending (SCR-DC5-1)
        self._all_worlds = wp.full(self.world_count, 1, dtype=wp.int8, device=dev)
        self._per_world = np.diff(self.tables.world_start.numpy())    # bonds per world (host validation)
        self._world = self.tables.world
        managed = np.concatenate([self.tables.eq.numpy()[:bonds.count],
                                  self.crumble.park_eq.numpy() if self.crumble is not None else np.zeros(0, np.int32)])
        self.hooks = SolverHooks(model, entries or {}, self.tables, bonds, coupled=coupled, managed_eq=managed)
        self._has_cut = bool(bonds.count and (kind == int(BondKind.TET_CUT)).any() and self.blades.count)
        self._reach = cut.resolve_reach(self.config.blade_reach)
        self._alignment = cut.resolve_alignment(self.config.cut_alignment)
        self._normal, self._side_tet = self._seam_frames(n)
        self._cut_track = cut.build_cut_damage_track(bonds.count, dev)
        self._cut_blades = cut.cut_blades_struct(self.blades)
        self._last_ledger = TopologyLedger.empty(self.world_count)
        self._front = CutFront.build(model, bonds, blades)
        self._seam: SeamContactFilter | None = None
        has_tet_cut = bool(bonds.count and (kind == int(BondKind.TET_CUT)).any())
        for s in (entries or {}).values():
            if has_tet_cut and _self_contact_vbd(s, model):
                self.attach_self_contact(s)
            self._link_external(s, n)
        log.info("TopologyController: %d bonds (%s), %d blades, %d crumble sources, %d worlds, cap %d/step",
                 bonds.count, _kind_summary(kind), self.blades.count, self.crumble.count if self.crumble else 0,
                 self.world_count, budget.max_events_per_step)

    # -- TopologyLike ------------------------------------------------------------------------------------------
    def pre_step(self, state: newton.State, contacts: newton.Contacts | None, dt: float) -> None:
        """Blade placement, TET_CUT gating and cut damage (stiffness writes only). A no-op without TET_CUT bonds
        and blades. Behaviour: docs/notes/cleanroom-cut-spec.md. It first copies the topology-managed equality rows
        into a coupled MuJoCo entry's view (SCR-DC5-21) and, with an attached self-contact VBD, refreshes the
        seam-twin contact filter from the bond state (SCR-SC5-2)."""
        self.hooks.sync_views()
        if self._seam is not None:
            self._seam.update(self.bonds.intact, state.particle_q)
        if not self._has_cut:
            return
        m, b, t, h, bl = self.model, self.bonds, self.tables, self.hooks, self.blades
        wp.launch(cut.blade_segments, dim=bl.count, device=m.device,
                  inputs=[bl.body, bl.edge_a, bl.edge_b, bl.axis, state.body_q, bl.p0, bl.p1, bl.face])
        tr = self._cut_track
        wp.launch(cut.cut_pre_step, dim=b.count, device=m.device, inputs=[
            b.kind, b.a, b.b, t.world, b.intact, t.spring, t.elem_start, t.elem, h.tet_indices, h.tet_poses,
            state.particle_q, m.particle_radius, t.cut_strain, t.cut_force, self._cut_blades, self._reach,
            self.phase, h.spring_stiffness, b.damage, self.contact, tr.blade, tr.edge0, tr.edge1, self._normal,
            self._side_tet, self._alignment])
        self._front.refresh(self.phase, tr.blade, bl.body)

    def _seam_frames(self, n: int) -> tuple[wp.array, wp.array]:
        """Per bond, for the cut alignment gate: the rest seam normal and a tet holding twin a (-1: rest normal),
        the frame ``BladeContact`` reads its seam normals in. Placeholders when the gate is off or nothing cuts.
        Bonds without a normal are cut ungated, and a model without the ``bond_normal`` column turns the gate off;
        both are logged once here."""
        m, dev = self.model, self.model.device
        none = wp.zeros(n, dtype=wp.vec3, device=dev), wp.full(n, -1, dtype=wp.int32, device=dev)
        if not (self._has_cut and self._alignment > 0.0):
            return none
        from alloy.topology.blade_contact import side_a_tets

        normal = getattr(getattr(m, "topology", None), "bond_normal", None)
        if normal is None or normal.shape[0] != self.bonds.count:
            log.warning("TopologyController: no seam normal per bond on this model (topology:bond_normal); the cut "
                        "alignment gate %.3g is off and blades cut any seam they reach", self._alignment)
            self._alignment = 0.0
            return none
        is_cut = self.bonds.kind.numpy() == int(BondKind.TET_CUT)
        bare = int((is_cut & (np.linalg.norm(normal.numpy(), axis=1) == 0.0)).sum())
        if bare:
            log.warning("TopologyController: %d of %d TET_CUT bonds have no seam normal; the cut alignment gate "
                        "%.3g does not apply to them", bare, int(is_cut.sum()), self._alignment)
        tets = m.tet_indices.numpy() if m.tet_count else np.zeros((0, 4), np.int64)
        tet = side_a_tets(tets, np.where(is_cut, self.bonds.a.numpy().astype(np.int64), -1))
        return normal, wp.array(tet, dtype=wp.int32, device=dev)

    def apply(self, state: newton.State) -> None:
        """Test breaks on the post-step state, commit under the cap, toggle natives, crumble, ledger."""
        if self.bonds.count:
            self._evaluate(state)
        self._commit_events()
        self._ledger.before(state)
        self._toggle()
        if self.crumble is not None:
            self._convert_and_park(state)
        self.hooks.notify()
        self._ledger.after(state)

    def ledger(self) -> TopologyLedger:
        """Host copy of the period since the last call, then a fresh period (a consuming read)."""
        self._last_ledger = self._ledger.snapshot()
        self._ledger.reset_period()
        return self._last_ledger

    # -- extra API ---------------------------------------------------------------------------------------------
    @property
    def step_events(self) -> wp.array:
        """[W] int32 events applied by the last ``apply`` (SCR-WP10-2: health re-baselines these worlds)."""
        return self._ledger.step_events

    @property
    def body_active(self) -> wp.array:
        """[body_count] int8; 0 for bodies retired by crumble (SCR-WP10-2)."""
        return self._ledger.body_active

    @property
    def cut_front_blade(self) -> wp.array:
        """[particle_count] int32: body id of the Blade cutting (phase 1) a TET_CUT bond this particle is a twin
        of, else -1; refreshed by ``pre_step`` (SCR-SC5-3, ``cut_front``). A soft contact (i, s) is a cut-front
        contact when ``model.shape_body[s] == cut_front_blade[i]``."""
        return self._front.blade

    @property
    def blade_half_thickness(self) -> wp.array:
        """[body_count] float32: a Blade body's blade half-thickness [m], 0 elsewhere (SCR-SC5-3). The cut-front
        depth limit is ``blade_half_thickness[cut_front_blade[i]] + particle_radius[i]``."""
        return self._front.half

    @property
    def seam_contact(self) -> SeamContactFilter | None:
        """The attached seam-twin contact filter (``attach_self_contact``), or None."""
        return self._seam

    @property
    def last_ledger(self) -> TopologyLedger:
        return self._last_ledger

    def request_crumble(self, bodies: Sequence[int]) -> None:
        """Manual trigger (host): the listed chunk bodies crumble at the next ``apply`` (subject to the cap)."""
        if self.crumble is None:
            raise AssemblyError("request_crumble: no crumble sources in this model")
        req = self.crumble.request.numpy()
        src = {int(b): k for k, b in enumerate(self.crumble.body.numpy())}
        unknown = [int(b) for b in bodies if int(b) not in src]
        if unknown:
            raise AssemblyError(f"request_crumble: bodies {unknown} are not crumble sources")
        for b in bodies:
            req[src[int(b)]] = 1
        self.crumble.request.assign(req)

    def request_break(self, bond_ids: Sequence[int] | np.ndarray | wp.array,
                      world_mask: Sequence[bool] | np.ndarray | wp.array | None = None) -> None:
        """External, ledgered bond break (SCR-DC5-1). The listed bonds break at the next ``apply`` in every world
        whose ``world_mask`` entry is nonzero (None = every world). They go through the same commit as a criterion
        break: capped per world in index order (overflow stays pending), counted in ``step_events`` and the
        ledger, native elements toggled. A request on a bond that is already broken is dropped.

        ``bond_ids`` are world-local: the rank of the bond among its world's bonds in index order. For a replicated
        template this is the row index ``add_*_bond`` returned while the template was built.
        Device inputs (int32 ids and an int8 ``[W]`` mask on the model's device) launch one kernel with no host
        sync, so the call can sit inside a captured graph. Host inputs are validated (``AssemblyError``) and
        uploaded.
        """
        if not self.bonds.count:
            raise AssemblyError("request_break: no bonds in this model")
        dev = self.model.device
        if _on_device(bond_ids, wp.int32, dev) and (world_mask is None or _on_device(world_mask, wp.int8, dev)):
            ids, mask = bond_ids, self._all_worlds if world_mask is None else world_mask
            if mask.shape[0] != self.world_count:
                raise AssemblyError(f"request_break: world_mask has {mask.shape[0]} entries, the model has "
                                    f"{self.world_count} worlds")
        else:
            ids, mask = self._host_request(bond_ids, world_mask)
        if ids.shape[0] == 0:
            return
        t = self.tables
        wp.launch(breaking.request_bonds, dim=(self.world_count, ids.shape[0]), device=dev,
                  inputs=[ids, mask, t.world_start, t.world_order, self._request])

    def attach_self_contact(self, solver: newton.solvers.SolverVBD) -> None:
        """SCR-SC5-2: ``solver`` (SolverVBD with self-contact, built on this model, no collision pipeline,
        ``particle_rest_shape_contact_exclusion_radius=0``) ignores a seam vertex's partner faces and edges only
        while the twins are joined by intact CLOTH_EDGE / TET_CUT bonds. After a cut, and once its twins have moved
        apart by the contact distance, the faces collide like any other surface (see ``seam_contact``). Allocates
        now; every ``pre_step`` then updates in place (capturable). ``AssemblyError`` if the solver does not
        qualify. A qualifying solver in ``entries`` of a model with TET_CUT bonds is attached at construction."""
        if self._seam is not None:
            raise AssemblyError("attach_self_contact: a self-contact solver is already attached")
        self._seam = SeamContactFilter(self.model, self.bonds, solver)

    def preset_broken(self, bond_ids: Sequence[int] | np.ndarray) -> None:
        """Host setup: mark bonds broken and toggle their native elements now. This is not an event (no ledger,
        no cap); it is for scripted pre-cut scenes and tests."""
        ids = np.asarray(bond_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0 or not self.bonds.count:
            return
        mask = np.zeros(self.bonds.count, np.int8)
        mask[ids] = 1
        for arr, val in ((self.bonds.intact, 0), (self.phase, 2), (self.bonds.damage, 1.0)):
            host = arr.numpy()
            host[ids] = val
            arr.assign(host)
        self._commit.assign(mask)
        self._toggle()
        self._commit.zero_()
        self.hooks.notify()

    def capacity_report(self) -> Mapping[str, int]:
        led = self._ledger.overflow.numpy()
        return {"bonds": self.bonds.count, "crumble_sources": self.crumble.count if self.crumble else 0,
                "spare_vertices_budget": self.budget.spare_vertices, "spare_particles_budget":
                self.budget.spare_particles, "spare_tets_budget": self.budget.spare_tets,
                "max_events_per_step": self.budget.max_events_per_step, "overflow_this_period": int(led.sum())}

    def state_arrays(self) -> Mapping[str, wp.array]:
        """Every array the next step reads that the controller changes (for WorldSnapshot, digest, save/load)."""
        out = {"topology.intact": self.bonds.intact, "topology.damage": self.bonds.damage,
               "topology.phase": self.phase, "topology.cut_done": self.cut_done,
               "topology.break_request": self._request}
        if self._has_cut:
            out.update(self._cut_track.arrays())
        if self._seam is not None:
            out["topology.contact_released"] = self._seam.released
        if self.crumble is not None:
            out.update({"topology.crumble_done": self.crumble.done, "topology.crumble_request": self.crumble.request})
        out.update(self._ledger.arrays())
        out.update(self.hooks.mutated_model_arrays())
        return out

    def state_worlds(self) -> Mapping[str, wp.array]:
        """Item -> world maps for ``state_arrays`` (WorldSnapshot ``extra_world``)."""
        out = {k: self._world for k in ("topology.intact", "topology.damage", "topology.phase", "topology.cut_done",
                                        "topology.break_request")}
        if self._has_cut:
            out.update({k: self._world for k in self._cut_track.arrays()})
        if self._seam is not None:
            out["topology.contact_released"] = self._world
        if self.crumble is not None:
            sw = np.repeat(np.arange(self.world_count), np.diff(self.crumble.world_start.numpy()))
            src_world = np.empty(self.crumble.count, np.int32)
            src_world[self.crumble.world_order.numpy()] = sw
            arr = wp.array(src_world, dtype=wp.int32, device=self.model.device)
            out["topology.crumble_done"] = out["topology.crumble_request"] = arr
        out.update(self.hooks.mutated_model_worlds())
        return out

    # -- internals ---------------------------------------------------------------------------------------------
    def _link_external(self, solver, n: int) -> None:
        """An entry solver with ``topology_link(intact) -> (force, threshold) | None`` supplies those per bond."""
        link = getattr(solver, "topology_link", None)
        out = link(self.bonds.intact) if callable(link) and self.bonds.count else None
        if out is None:
            return
        force, threshold = out
        if force.shape[0] != n or threshold.shape[0] != n:
            raise AssemblyError(f"topology_link of {type(solver).__name__}: {force.shape[0]} bond forces for {n} bonds")
        if self._linked:
            raise AssemblyError(f"topology_link: a second solver ({type(solver).__name__}) supplies bond forces; one "
                                "film entry per scene")
        self.ext_force, self.ext_threshold, self._linked = force, threshold, True
        log.info("TopologyController: bond forces from %s", type(solver).__name__)

    def _evaluate(self, state: newton.State) -> None:
        m, b, t, h = self.model, self.bonds, self.tables, self.hooks
        if t.has_particle:
            wp.launch(tear.eval_particle_bonds, dim=b.count, device=m.device, inputs=[
                b.kind, b.a, b.b, b.intact, b.break_force, b.break_strain, t.spring, t.rest, self.phase,
                t.elem_start, t.elem, h.tri_indices, h.tri_poses, h.tet_indices, h.tet_poses, state.particle_q,
                m.spring_stiffness, self.ext_force, self.ext_threshold, self._want, self.strain, self.force])
        if t.has_eq:
            wp.launch(breaking.eval_body_bonds, dim=b.count, device=m.device, inputs=[
                b.kind, b.a, b.b, self._world, b.intact, b.break_force, b.break_strain, t.rest, t.rel0,
                h.mjc_eq, state.body_q, h.efc(), self._want, self.strain, self.force])

    def _commit_events(self) -> None:
        m, b, t, cap = self.model, self.bonds, self.tables, int(self.budget.max_events_per_step)
        if b.count:
            wp.launch(breaking.commit_bonds, dim=self.world_count, device=m.device, inputs=[
                t.world_start, t.world_order, self._want, self._request, b.intact, b.damage, cap, self._commit,
                self.phase, self._ledger.step_events, self._ledger.overflow, self.cut_done])
        else:
            self._ledger.step_events.zero_()
        c = self.crumble
        if c is not None:
            wp.launch(crt.crumble_triggers, dim=c.count, device=m.device, inputs=[
                c.trigger, c.done, c.request, c.bond_start, c.bond, b.intact if b.count else self._want, c.want])
            wp.launch(crt.commit_crumbles, dim=self.world_count, device=m.device, inputs=[
                c.world_start, c.world_order, c.want, c.done, c.request, cap, c.commit, self._ledger.step_events,
                self._ledger.overflow])

    def _host_request(self, bond_ids, world_mask) -> tuple[wp.array, wp.array]:
        """Validate host ids and mask for ``request_break`` and upload them."""
        ids = np.asarray(_host(bond_ids), dtype=np.int64).reshape(-1)
        mask = (np.ones(self.world_count, np.int8) if world_mask is None
                else (np.asarray(_host(world_mask)).reshape(-1) != 0).astype(np.int8))
        if mask.shape[0] != self.world_count:
            raise AssemblyError(f"request_break: world_mask has {mask.shape[0]} entries, the model has "
                                f"{self.world_count} worlds")
        counts = self._per_world[mask != 0]
        limit = int(counts.min()) if counts.size else 0
        bad = ids[(ids < 0) | (ids >= limit)] if counts.size else ids[:0]
        if bad.size:
            raise AssemblyError(f"request_break: world-local bond ids {bad.tolist()} outside [0, {limit}) "
                                f"for the masked worlds")
        dev = self.model.device
        return wp.array(ids.astype(np.int32), dtype=wp.int32, device=dev), wp.array(mask, dtype=wp.int8, device=dev)

    def _toggle(self) -> None:
        m, b, t, h = self.model, self.bonds, self.tables, self.hooks
        if not b.count:
            return
        wp.launch(breaking.toggle_bond_natives, dim=b.count, device=m.device, inputs=[
            self._commit, t.spring, t.eq, h.spring_stiffness, h.spring_damping, h.eq_enabled])
        if t.edge_ids.shape[0]:
            wp.launch(breaking.toggle_cross_seam_edges, dim=t.edge_ids.shape[0], device=m.device, inputs=[
                t.edge_ids, t.edge_bonds, self._commit, m.edge_bending_properties])

    def _convert_and_park(self, state: newton.State) -> None:
        m, c, h = self.model, self.crumble, self.hooks
        if c.part.shape[0]:
            wp.launch(crt.convert_particles, dim=c.part.shape[0], device=m.device, inputs=[
                c.part, c.part_src, c.commit, c.body, h.crumble_local, state.body_q, state.body_qd, m.body_com,
                state.particle_q, state.particle_qd, h.flag_targets(), h.mpm_targets(state)])
        wp.launch(crt.park_bodies, dim=c.count, device=m.device, inputs=[
            c.commit, c.body, c.joint, c.q_start, c.qd_start, c.slot, c.park_eq, wp.vec3(*self.config.park_offset),
            m.joint_X_p, m.joint_X_c, state.body_q, state.body_qd, state.joint_q, state.joint_qd, h.eq_enabled,
            h.eq_relpose, self._ledger.body_active])


def _self_contact_vbd(solver, model: newton.Model) -> bool:
    """Auto-attach rule for ``entries``: a SolverVBD with its own self-contact detector on this model."""
    det = getattr(solver, "trimesh_collision_detector", None)
    return (isinstance(solver, newton.solvers.SolverVBD) and getattr(solver, "particle_enable_self_contact", False)
            and getattr(solver, "collision_pipeline", None) is None and det is not None and det.model is model)


def _host(x):
    return x.numpy() if isinstance(x, wp.array) else x


def _on_device(x, dtype, device) -> bool:
    return (isinstance(x, wp.array) and x.ndim == 1 and x.dtype == dtype
            and wp.get_device(x.device) == wp.get_device(device))


def _kind_summary(kind: np.ndarray) -> str:
    counts = np.bincount(kind, minlength=len(BondKind)) if len(kind) else np.zeros(len(BondKind), int)
    return ", ".join(f"{k.name}={counts[k]}" for k in BondKind if counts[k]) or "none"
