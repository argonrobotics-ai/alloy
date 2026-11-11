# SPDX-License-Identifier: Apache-2.0
"""Wire every real part around a ``BuiltScene`` and build the ``Simulation`` (SPEC §2.3, §7 WP12).

``wire(built, profile)`` returns the parts; ``build_simulation(built, profile, plugins=[...])`` returns an
``IntegratedSimulation`` with them injected. Parts and their owners:

- stepper (WP2 single entry / WP4 Scheduler, SchedulerADMM): ``alloy.integration.stepper``;
- contact layer (WP3 ``AlloyContactLayer``): built when the scene has precision pairs, when the profile asks, or
  under ``native="newton"`` (one deterministic Newton buffer for every entry, D10/D13). Under ``native="mujoco"``
  the precision rows are appended into the SolverMuJoCo entry through ``bind_mujoco(...).install()`` (D10); its
  ``nconmax``/``njmax`` grow by the layer's per-world precision capacity so the appended rows fit. A coupled scene
  with precision pairs under ``native="mujoco"`` is accepted only when the collision filters prove that no
  non-MuJoCo entry can contact a precision-pair shape (SCR-DC5-3, ``alloy.integration.precision_reach``); the
  precision stream is then appended into MuJoCo and the other entries read WP2's Newton fallback buffer. Otherwise
  it is refused, naming the pair (use ``native="newton"``);
- gap monitor (WP3 ``GapMonitor``, ``layer.monitor()``) when the layer has precision pairs;
- health (WP5 ``HealthMonitor``) with the entry solvers (health adapters, capacity sources, determinism) and the
  layer/append overflow counters as capacity flags;
- topology (WP10 ``TopologyController``) when the scene has bonds, with the entry solvers as hooks and a ``Blade``
  per world on each object whose card has a ``blade:`` block (SCR-SC5-1); WP10's ``VNodeTopology`` instead when its
  soft objects carry virtual-node copies (``topology: {cut_model: copies}``).

Under ``native="newton"`` precision pairs with L1 (``ccd != "off"``) need ``use_speculative_contacts`` on
SolverMuJoCo (euler/implicitfast only; §3.5 integrator table), which ``wire`` adds to the entry's kwargs."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor.newton.solvers import SolverBase

from alloy.core.errors import AssemblyError, PrecisionSpecError
from alloy.core.protocols import ContactLayerLike, GapMonitorLike, HealthLike, StepperLike, TopologyLike
from alloy.coupling.types import SchedulerPolicy
from alloy.integration import mpm_memory, mpm_pairs, mpm_proxy
from alloy.integration.built_copy import replace_built
from alloy.integration.graph_budget import eager_scene_reason
from alloy.integration.mpm_grid import GridGuard, entry_cells, graph_cell_capacity
from alloy.integration.precision_pin import RungPinWatch, pin_exchange
from alloy.integration.precision_reach import (CoupledPrecisionContacts, check_isolated, describe_reach,
                                               rearm_each_step)
from alloy.integration.profiles import IntegrationProfile, profile
from alloy.integration.simulation import IntegratedSimulation
from alloy.integration.stepper import (CONTACT_FREE_DESTINATIONS, dynamic_entries, entry_solvers, make_stepper,
                                      mujoco_solver)
from alloy.precision import cloth_stretch
from alloy.precision.determinism import refresh_coupled_constants
from alloy.precision.types import PrecisionConfig
from alloy.route.mjc_core_config import CPU_ROUTE_KEY
from alloy.scene.steppers import NewtonContacts, fallback_needs_rigid, needs_newton_contacts
from alloy.scene.types import BuiltScene, EntrySpec

log = logging.getLogger("alloy.integration")

__all__ = ["Parts", "build_simulation", "wire"]

ROWS_PER_CONTACT = 4          # njmax growth per appended contact (elliptic/pyramidal condim-3 rows, peg harness)
L1_INTEGRATORS = ("euler", "implicitfast")


@dataclass
class Parts:
    """Everything ``wire`` built for one scene (``built`` may carry adjusted entry kwargs)."""

    built: BuiltScene
    profile: IntegrationProfile
    precision: PrecisionConfig
    native: str
    stepper: StepperLike
    solvers: dict[str, SolverBase]
    contact_layer: ContactLayerLike | None = None
    binding: Any = None
    gap_monitor: GapMonitorLike | None = None
    health: HealthLike | None = None
    topology: TopologyLike | None = None
    contact_adjusters: list[Any] = field(default_factory=list)    # ContactAdjuster beside the topology (SCR-SC5-1)
    notes: list[str] = field(default_factory=list)
    boundary_reach: list[Any] = field(default_factory=list)       # accepted boundary-only reaches (SCR-JOB2-1)
    rung_pin: Any = None                                          # RungPinWatch of a pinned MuJoCo entry (SCR-JOB2-1)
    mpm_grid: GridGuard | None = None                             # rebuildable MPM grids, checked per period


def wire(built: BuiltScene, prof: str | IntegrationProfile = "standard", *,
         stepper: StepperLike | None = None) -> Parts:
    """Build the stepper, contact layer, gap monitor, health and topology for ``built`` (module docstring).
    ``stepper`` replaces the one ``make_stepper`` would build (fakes, custom couplers)."""
    from alloy.coupling.unsupported import check_built  # noqa: PLC0415

    check_built(built)              # route pairs no solver steps together: refused before any part (WP13 F1/F4)
    p = profile(prof)
    use_graph = p.wants_graph(wp.get_device(built.model.device).is_cuda)[0] and not eager_scene_reason(
        built.report.scene_effects)                 # an eager scene (a j2 solid on MPM) is planned without graphs
    built = mpm_memory.plan(built, use_graph)       # too big as graphs: eager; for neither: refused (SCR-WP12-3)
    eager_why = mpm_memory.eager_reason(built)
    use_graph = use_graph and eager_why is None
    _register_adapters()
    cfg = p.precision_config(str(built.model.device))
    native = cfg.native_contacts
    notes: list[str] = [f"graphs off: {eager_why}"] if eager_why else []
    boundary = []
    if _coupled_precision(built, native, p):              # before any solver is built (SCR-DC5-3, SCR-JOB2-1)
        boundary = check_isolated(built, dynamic_entries(built), gap_fail_closed=_gap_fail_closed(p))
        notes.extend(f"boundary-only reach (proxy wrench, GapMonitor fails closed): {line}"
                     for line in describe_reach(built, boundary))
    built = _prepare_entries(built, native, p, notes)
    from alloy.scene.integrator_route import precision_entries  # noqa: PLC0415  (lazy: its template import inits Warp)
    built = precision_entries(built, p.preset, notes)      # F6 R1: the precision profile keeps rk4
    built = _route_entries(built, p, notes)
    if use_graph:
        built = _graph_entries(built, notes)
    if stepper is None:
        stepper = make_stepper(built, native, p.scheduler_policy, stock=p.stock_coupler)
    solvers = entry_solvers(stepper)
    notes += mpm_pairs.apply_pair_friction(built, solvers) + mpm_proxy.install_for_pairs(built, solvers)  # MPM pairs
    for (src, dst), ids in sorted(getattr(stepper, "kinematic_proxies", dict)().items()):   # SCR-SC8-2
        notes.append(f"kinematic proxies {src} -> {dst}: {len(ids)} world-welded source bodies (mass 0 in {dst})")
    parts = Parts(built, p, cfg, native, stepper, solvers, notes=notes, boundary_reach=boundary)
    parts.mpm_grid = GridGuard.for_solvers(solvers, {e.name: len(e.particles) for e in built.entries
                                                     if e.capability == "mpm"})
    parts.rung_pin = RungPinWatch.for_built(built, pin_exchange(p.scheduler_policy or SchedulerPolicy(), built))
    parts.contact_layer, parts.binding = _contact_layer(built, native, p, solvers, notes)
    parts.gap_monitor = _gap_monitor(parts.contact_layer, p)
    parts.topology = _topology(built, solvers, stepper, notes) if p.topology else None
    parts.contact_adjusters = _contact_adjusters(built, parts.topology, notes)
    if parts.contact_layer is None:          # after topology and adjusters, as Simulation built it (SCR-WP4-6/7)
        parts.contact_layer = _fallback_layer(built, native, p, notes)
    parts.health = _health(parts) if p.health else None   # enables the ordered MuJoCo route when the profile asks
    if parts.health is not None and refresh_coupled_constants(stepper):
        notes.append("coupler re-installed its proxy masses from the routed SolverMuJoCo's ordered constants")
    _health_topology(parts.health, parts.topology, notes)
    _health_soft_contacts(parts.health, built)
    cloth_stretch.install(parts.health, notes)          # SCR-WP5-17: CLOTH_STRETCH on every cloth triangle
    log.info("wired %s: native=%s, stepper %s, layer %s, gap %s, health %s, topology %s", p.name, native,
             type(stepper).__name__, type(parts.contact_layer).__name__, parts.gap_monitor is not None,
             parts.health is not None, parts.topology is not None)
    return parts


def build_simulation(built: BuiltScene, prof: str | IntegrationProfile = "standard", *, seed: int = 0,
                     stepper: StepperLike | None = None, plugins: Sequence[Any] = (),
                     **overrides: Any) -> IntegratedSimulation:
    """``IntegratedSimulation`` with every part ``wire`` built; ``overrides`` are profile fields. ``plugins``
    (``alloy.plugins`` objects, or functions of the built scene that return one) run in the wired stepper, installed
    before the simulation snapshots its state arrays, so reset and rollbacks restore their state too; their host is
    ``sim.plugins``. Example: ``build_simulation(built, "standard", plugins=[lambda b: Drag(body="cube")])``."""
    p = profile(prof, **overrides)
    parts = wire(built, p, stepper=stepper)
    host = install_plugins(parts, plugins) if plugins else None
    use_graph, cpu_graph = _scene_graphs(parts.built, p)
    sim = IntegratedSimulation(parts.built, precision=parts.precision, contact_layer=parts.contact_layer,
                               gap_monitor=parts.gap_monitor, health=parts.health, topology=parts.topology,
                               stepper=parts.stepper, use_graph=use_graph, cpu_graph=cpu_graph, seed=seed,
                               contact_adjusters=parts.contact_adjusters)
    sim.parts = parts
    _spin_per_rigid_step(sim, parts)
    if p.graph == "on":                 # graphs asked for: every key keeps them, no timing choice (lever #3 is "auto")
        sim.graph_choice = None
    if host is not None:
        sim.plugins = host
    return sim


def install_plugins(parts: Parts, plugins: Sequence[Any]) -> Any:
    """A ``PluginHost`` running ``plugins`` (plug-ins, or functions of ``parts.built`` returning one) in
    ``parts.stepper``; raises ``PluginError`` naming an entry that is neither."""
    from alloy.plugins.base import Plugin  # noqa: PLC0415 - plug-ins load only when asked for
    from alloy.plugins.errors import PluginError  # noqa: PLC0415
    from alloy.plugins.host import PluginHost  # noqa: PLC0415

    made = []
    for x in plugins:
        item = x if isinstance(x, Plugin) or not callable(x) else x(parts.built)
        if not isinstance(item, Plugin):
            raise PluginError(f"plugins entry {x!r} gave {type(item).__name__}: pass a plug-in or a function of the "
                              "built scene that returns one")
        made.append(item)
    host = PluginHost(parts.built.model, made, control_dt=parts.built.control_dt)
    host.install(parts.stepper)
    return host


def _scene_graphs(built: BuiltScene, p: IntegrationProfile) -> tuple[bool, bool]:
    """``p.wants_graph`` for the scene's device, off when a scene effect puts one exchange over the graph node
    budget (``graph_budget.eager_scene_reason``): capturing it would only cost time and memory before the drop; or
    when ``wire`` found that the MPM step fits only without graphs (``mpm_memory.plan``)."""
    use_graph, cpu_graph = p.wants_graph(wp.get_device(built.model.device).is_cuda)
    reason = (eager_scene_reason(built.report.scene_effects) or mpm_memory.eager_reason(built)) if use_graph else None
    if reason is None:
        return use_graph, cpu_graph
    level = logging.WARNING if p.graph == "on" else logging.INFO     # "on" was asked for explicitly
    log.log(level, "graphs off for this scene (profile graph=%r): %s", p.graph, reason)
    return False, False


def _register_adapters() -> None:
    """WP4's state adapters for Newton solvers (MJWarp warm starts, VBD history, ...), so snapshots, latching,
    rollback and digests include them (§2.2 "solver internals"); idempotent."""
    from alloy.coupling.adapters import register_builtin_adapters  # noqa: PLC0415 - WP4

    register_builtin_adapters()


def _spin_per_rigid_step(sim: IntegratedSimulation, parts: Parts) -> None:
    """Health's SPIN judges rotation per rigid step: an exchange of ``dt_x`` runs a body entry's current substeps
    (a coupled MuJoCo entry can take k per exchange; WP3's review of SCR-WP5-14). A graph's key holds the substeps,
    so a captured period keeps the value it was captured with."""
    start = getattr(parts.health, "start", None)
    names = sorted({e.name for e in parts.built.entries if len(e.bodies)})
    if start is not None and names:
        start.rigid_substeps = lambda: max(int(getattr(sim, "substeps", {}).get(n, 1)) for n in names)


# --- entries ------------------------------------------------------------------------------------------------


def _prepare_entries(built: BuiltScene, native: str, p: IntegrationProfile, notes: list[str]) -> BuiltScene:
    """Adjust the MuJoCo entry for precision pairs (append capacity, speculative rows under newton)."""
    if not built.precision_pairs:
        return built
    entries = [_mujoco_kwargs(e, built, native, p, notes) if e.capability == "mujoco" else e for e in built.entries]
    return replace_built(built, entries=entries)


def _route_entries(built: BuiltScene, p: IntegrationProfile, notes: list[str]) -> BuiltScene:
    """A profile that asks for Alloy's ordered MuJoCo route builds its MuJoCo entries with a sparse Jacobian, the
    route's proven path. MuJoCo picks dense below nv 60, and the route refuses dense (coordinator ruling 2026-11-06
    06:23Z). An entry that sets its own ``jacobian`` keeps it, and the route then names it as the reason it is off."""
    if not p.mjw_route:
        return built
    out = []
    for e in built.entries:
        if e.capability == "mujoco" and "jacobian" not in e.solver_kwargs:
            e = dataclasses.replace(e, solver_kwargs={**dict(e.solver_kwargs), "jacobian": "sparse"})
            notes.append(f"mujoco entry {e.name}: jacobian=sparse (profile asks for the ordered route)")
        out.append(e)
    return replace_built(built, entries=out)


def _graph_entries(built: BuiltScene, notes: list[str]) -> BuiltScene:
    """Graph capture needs solvers that never read the device on the host inside a step. SolverImplicitMPM's default
    sparse grid re-reads the particle bounds on the host every step (newton@ac7f7830:newton/_src/solvers/
    implicit_mpm/solver_implicit_mpm.py:2786-2798); a positive ``max_active_cell_count`` makes it a persistent,
    in-place rebuilt grid (:1632-1638), reserved from the cells in use at the start (``mpm_grid``; ``GridGuard``
    grows it per period). That grid is sparse: an entry without its own ``grid_type`` gets the sparse grid under
    capture even when ALLOY_MPM_GRID asks for dense, whose bounds are read on the host (known gap
    mpm-sparse-floor-offset). Eager scenes keep Newton's per-step grid: on the reserved grid, 64 dough worlds ran
    1.48x slower in 3.4x the memory (SCR-WP12-3 a2, L4)."""
    out = []
    for e in built.entries:
        if e.capability == "mpm" and e.particles and int(e.solver_kwargs.get("max_active_cell_count", -1)) <= 0:
            cells = graph_cell_capacity(len(e.particles), entry_cells(built.model, e))
            kw = {"grid_type": "sparse", **dict(e.solver_kwargs), "max_active_cell_count": cells}
            e = dataclasses.replace(e, solver_kwargs=kw)
            notes.append(f"mpm entry {e.name}: max_active_cell_count={cells} of {len(e.particles)} particles, "
                         f"grid_type={kw['grid_type']} (graph capture needs a persistent grid)")
        out.append(e)
    return replace_built(built, entries=out)


def _mujoco_kwargs(e: EntrySpec, built: BuiltScene, native: str, p: IntegrationProfile,
                   notes: list[str]) -> EntrySpec:
    kw = dict(e.solver_kwargs)
    if kw.setdefault(CPU_ROUTE_KEY, "warp") == "warp":    # SCR-WP8-8: the C route cannot host appended precision rows
        notes.append(f"mujoco entry {e.name}: {CPU_ROUTE_KEY}=warp (precision pairs need MuJoCo Warp on the CPU too)")
    if native == "mujoco":
        extra = p.precision_contacts_per_world
        if kw.get("nconmax") is not None:
            kw["nconmax"] = int(kw["nconmax"]) + extra
        if kw.get("njmax") is not None:
            kw["njmax"] = int(kw["njmax"]) + ROWS_PER_CONTACT * extra
        notes.append(f"mujoco nconmax/njmax grown by {extra}/{ROWS_PER_CONTACT * extra} for appended precision rows")
    else:
        integ = str(kw.get("integrator", "implicitfast")).lower()
        if any(s.ccd != "off" for s in built.precision_pairs) and integ in L1_INTEGRATORS:
            kw["use_speculative_contacts"] = True
            notes.append("mujoco use_speculative_contacts=True (L1 rows under native='newton')")
    return dataclasses.replace(e, solver_kwargs=kw)


# --- contact layer and gap monitor --------------------------------------------------------------------------


def _gap_fail_closed(p: IntegrationProfile) -> bool:
    """The pairs' GapMonitor is built and its PAIR_PEN_FAIL reaches a health monitor, which latches the world
    INVALID: what a boundary-only reach needs (SCR-JOB2-1)."""
    return p.gap_monitor != "never" and p.contact_layer != "never" and bool(p.health)


def _coupled_precision(built: BuiltScene, native: str, p: IntegrationProfile) -> bool:
    """Precision pairs under ``native="mujoco"`` beside other dynamic entries (SCR-DC5-3)."""
    if native != "mujoco" or not built.precision_pairs or p.contact_layer == "never":
        return False
    dyn = dynamic_entries(built)
    return not (len(dyn) == 1 and dyn[0].capability == "mujoco")


def _contact_layer(built: BuiltScene, native: str, p: IntegrationProfile, solvers: dict[str, SolverBase],
                   notes: list[str]) -> tuple[ContactLayerLike | None, Any]:
    pairs = tuple(built.precision_pairs)
    if p.contact_layer == "never":
        return None, None
    if native == "mujoco" and not pairs and p.contact_layer != "always":
        notes.append("no Alloy contact layer: MuJoCo collides natively (coupled scenes get the fallback buffer)")
        return None, None
    from alloy.contact.pipeline import AlloyContactLayer  # noqa: PLC0415 - WP3

    coupled = _coupled_precision(built, native, p)      # wire() ran check_isolated before building solvers
    layer = AlloyContactLayer(built.model, pairs, native=native,
                              precision_contacts_per_world=p.precision_contacts_per_world)
    binding = None
    if native == "mujoco" and pairs:
        mj = mujoco_solver(solvers)
        if mj is None:
            raise PrecisionSpecError("precision pairs under native_contacts='mujoco' need a SolverMuJoCo entry")
        binding = layer.bind_mujoco(mj)
        binding.install()
    if coupled:
        notes.append("precision pairs isolated from the non-MuJoCo entries: precision stream appended into MuJoCo, "
                     "the other entries read the Newton fallback buffer (SCR-DC5-3)")
        bulk = _newton_fallback(built, native, p, notes)        # bounded like the plain fallback (SCR-WP4-6/7)
        coupled_layer = CoupledPrecisionContacts(layer, built.model, built.pair_table, bulk=bulk)
        rearm_each_step(mj, coupled_layer, binding)
        return coupled_layer, binding
    return layer, binding


def _fallback_layer(built: BuiltScene, native: str, p: IntegrationProfile, notes: list[str]) -> NewtonContacts | None:
    """WP2's Newton fallback buffer, built here (not by Simulation) so it takes the profile's ``contact_budget`` and
    its overflow counter reaches the health monitor (SCR-WP4-6). Soft contacts only when MuJoCo runs its own contacts
    and owns every rigid shape (SCR-WP4-7, checked at build time). None when Simulation would build none."""
    if not needs_newton_contacts(dynamic_entries(built), native):
        return None
    return _newton_fallback(built, native, p, notes)


def _newton_fallback(built: BuiltScene, native: str, p: IntegrationProfile, notes: list[str]) -> NewtonContacts:
    """The fallback buffer with the profile's ``contact_budget``, soft-only when MuJoCo owns every rigid shape; also
    the non-MuJoCo entries' buffer beside a precision stream (``CoupledPrecisionContacts``). Contact-free (SPH, MPM)
    particles get no soft candidates (SCR-JOB2-4)."""
    dyn = dynamic_entries(built)
    rigid = fallback_needs_rigid(dyn, native, built.model.shape_body.numpy())
    free = contact_free_particles(dyn)
    particles = None if free.size == 0 else np.setdiff1d(np.arange(int(built.model.particle_count or 0)), free)
    layer = NewtonContacts(built.model, budget=p.contact_budget, rigid=rigid, particles=particles)
    kind = "rigid + soft" if rigid else "soft-only, rigid contacts from MuJoCo"
    skip = f"; no soft candidates for {free.size} contact-free (SPH / MPM) particles" if free.size else ""
    notes.append(f"fallback contacts: {kind} ({p.contact_budget or 'Newton auto sizes'}){skip}")
    return layer


def contact_free_particles(entries: list[EntrySpec]) -> np.ndarray:
    """Particle ids of contact-free entries (``CONTACT_FREE_DESTINATIONS``, SPH and MPM): the fallback buffer gives
    them no soft candidates (SCR-JOB2-4)."""
    ids = [np.asarray(e.particles, dtype=np.int64) for e in entries
           if e.capability in CONTACT_FREE_DESTINATIONS and len(e.particles)]
    return np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)


def _gap_monitor(layer: ContactLayerLike | None, p: IntegrationProfile) -> GapMonitorLike | None:
    if layer is None or p.gap_monitor == "never" or not getattr(layer, "pairs", ()):
        return None
    return layer.monitor(sample_spacing=p.gap_sample_spacing_m)


# --- health and topology ------------------------------------------------------------------------------------


def _health(parts: Parts) -> HealthLike:
    from alloy.precision.health import HealthMonitor  # noqa: PLC0415 - WP5

    bulk = getattr(parts.contact_layer, "bulk", None)        # the fallback beside a precision stream
    flags = [a for a in (getattr(parts.contact_layer, "overflow", None), getattr(parts.binding, "overflow", None),
                         getattr(bulk, "overflow", None)) if a is not None]
    flags += list(getattr(parts.stepper, "capacity_flags", lambda: [])())   # SCR-DC5-19: WP4's proxy pipeline
    watch = getattr(parts, "rung_pin", None)
    pin = watch.health_source() if watch is not None else None
    flags += [pin] if pin is not None else []                                # SCR-JOB2-1: precision rung pinned
    built = parts.built
    return HealthMonitor(built.model, parts.precision.checks, precision_pairs=tuple(built.precision_pairs),
                         capacity_flags=flags, solvers=list(parts.solvers.values()), config=parts.precision)


NO_SOFT_CONTACT_ENTRIES = ("sph", "mpm")    # solvers that handle shapes themselves, never reading Newton soft contacts


def _health_soft_contacts(health: HealthLike | None, built: BuiltScene) -> None:
    """Health does not judge the soft contacts of SPH / MPM particles: no solver reads those rows, and a wrong-signed
    one (a Warp parity sign at a degenerate point, WP2 11-07) froze liquid at rest in a mesh cup with PEN_FAIL."""
    skip = [i for e in built.entries if e.capability in NO_SOFT_CONTACT_ENTRIES for i in e.particles]
    if health is not None and skip and hasattr(health, "skip_soft_contacts"):
        health.skip_soft_contacts(skip)


def _health_topology(health: HealthLike | None, topology: TopologyLike | None, notes: list[str]) -> None:
    """Topology arrays the health monitor reads, all kept live on device by the controller; set here, before any
    graph capture. Crumbled bodies leave KE/PE (SCR-WP10-2). With blades, soft contacts of a Blade body with a twin
    whose bond it is cutting (phase 1) leave the PEN rule for the cut-front check (SCR-SC5-3; WP10's predicate). Soft
    tets of a cut scene are checked against the FEM model's strain range (STRAIN_RANGE, WP10's SoftStrain measure)."""
    if health is None or topology is None:
        return
    if hasattr(health, "set_body_active"):
        health.set_body_active(topology.body_active)
    blades = getattr(getattr(topology, "blades", None), "count", 0)
    if blades and hasattr(health, "set_cut_front") and hasattr(topology, "cut_front_blade"):
        health.set_cut_front(topology.cut_front_blade, topology.blade_half_thickness)
        notes.append(f"health: cut-front PEN rule for {blades} blade(s) (SCR-SC5-3)")
    model = getattr(health, "model", None)
    if model is not None and int(model.tet_count) and hasattr(health, "set_soft_strain"):   # ruling 2026-11-06 14:17Z
        from alloy.topology.soft_strain import SoftStrain  # noqa: PLC0415 - WP10's measure

        health.set_soft_strain(SoftStrain(model))
        notes.append("health: soft tet strain past the FEM model's range flags STRAIN_RANGE (DEGRADED)")


def _topology(built: BuiltScene, solvers: dict[str, SolverBase], stepper: StepperLike,
              notes: list[str]) -> TopologyLike | None:
    """WP10's controller over the scene's bonds (None without any). ``coupled=stepper`` lets its hooks mirror weld
    rows into a MuJoCo entry's own ModelView under a coupled stepper (SCR-DC5-21); ignored on a single entry."""
    from alloy.topology.bonds import bonds_from_model, has_bonds  # noqa: PLC0415 - WP10
    from alloy.topology.controller import TopologyController  # noqa: PLC0415
    from alloy.topology.types import TopologyBudget  # noqa: PLC0415

    bonds = built.bonds if built.bonds is not None else (bonds_from_model(built.model)
                                                         if has_bonds(built.model) else None)
    copies = _copies_topology(built, bonds, notes)
    if copies is not None:
        return copies
    if bonds is None or bonds.count == 0:
        return None
    spares = dict(built.spares or {})
    budget = TopologyBudget(spare_particles=int(spares.get("particles", 0)),
                            spare_vertices=int(spares.get("vertices", 0)), spare_tets=int(spares.get("tets", 0)))
    blades = _blades(built)
    notes.append(f"topology controller over {bonds.count} bonds, {len(blades)} blade(s)")
    return TopologyController(built.model, bonds, budget, tuple(blades), entries=solvers, coupled=stepper)


def _copies_topology(built: BuiltScene, bonds: Any, notes: list[str]) -> TopologyLike | None:
    """WP10's ``VNodeTopology`` when the scene's soft objects carry virtual-node copies (cards with ``topology:
    {cut_model: copies}``); None without copies. One cut model per scene: copies beside seam bonds are refused.
    Health's cut-front PEN rule stays off (it reads a controller's ``blades``): under the lead rule the cut faces ride
    the blade faces, so the plain PEN rule holds at the cut."""
    from alloy.topology.copies_model import copies_from_model, has_copies  # noqa: PLC0415 - WP10
    from alloy.topology.vnode_topology import VNodeTopology  # noqa: PLC0415

    if not has_copies(built.model):
        return None
    if bonds is not None and bonds.count:
        raise AssemblyError("a scene cuts with one model: objects with topology.cut_model 'copies' and pre-split seam "
                            "bonds cannot share it (set every cut object's cut_model alike)")
    blades = _blades(built)
    topo = VNodeTopology(built.model, copies_from_model(built.model), blades)
    notes.append(f"virtual-node cutter over {topo.cutter.copies.count} cuttable tets, {len(blades)} blade(s)")
    if not blades:
        log.warning("objects with topology.cut_model 'copies' but no object with a blade: block; nothing will cut them")
    return topo


def _blades(built: BuiltScene) -> list[Any]:
    """One ``Blade`` per world on the body (and first shape) of every object whose card has a ``blade:`` block
    (SCR-SC5-1); the assembler carries the card's ``BladeSpec`` on the object's handle (WP2)."""
    from alloy.topology.types import Blade  # noqa: PLC0415 - WP10

    out = []
    for h in built.handles.values():
        spec = h.blade
        if spec is None:
            continue
        if not len(h.template.body) or not len(h.template.shape):
            raise AssemblyError(f"object {h.name!r} has a blade: block but no body or shape to carry it")
        nb, ns = len(h.world_stride.body), len(h.world_stride.shape)
        out.extend(Blade(body=h.template.body.start + w * nb, shape=h.template.shape.start + w * ns,
                         edge_a=tuple(spec.edge_a), edge_b=tuple(spec.edge_b), edge_radius=spec.edge_radius,
                         softness=spec.softness, cut_spring_ke=spec.cut_spring_ke)
                   for w in range(built.world_count))
    return out


def _contact_adjusters(built: BuiltScene, topology: TopologyLike | None, notes: list[str]) -> list[Any]:
    """SC5's side-aware blade contact (WP10's ``BladeContact``, [internal note] ruling item 1) beside the
    controller, over the controller's bonds, when the scene has bonds and blades (SCR-SC5-1; WP0's option (a)).
    None for WP10's copies topology (``VNodeTopology`` meets blades on its own faces in its ``adjust_contacts``)."""
    blades = _blades(built) if topology is not None and hasattr(topology, "bonds") else []
    if not blades:                                     # WP10's copies topology meets blades on its own faces
        return []
    try:
        from alloy.topology.blade_contact import BladeContact  # noqa: PLC0415 - WP10
    except ImportError:
        notes.append("blades without the side-aware blade contact (alloy.topology.blade_contact not on main, WP10)")
        log.warning("%d blade(s) but no alloy.topology.blade_contact: cut seams keep Newton's stock contact",
                    len(blades))
        return []
    notes.append(f"side-aware blade contact over {len(blades)} blade(s) (SCR-SC5-1)")
    return [BladeContact(built.model, topology.bonds, blades)]
