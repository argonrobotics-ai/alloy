# SPDX-License-Identifier: Apache-2.0
"""BuiltScene -> StepperLike, with WP4's Scheduler / SchedulerADMM for coupled scenes (SPEC §2.3, D5, D18, D21).

- one dynamic entry: WP2's ``SingleEntryStepper`` (no coupler, D5);
- ``CouplerChoice.kind == "proxy"``: ``alloy.coupling.Scheduler`` (residual-gated lagged/staggered proxy, adaptive
  clocks, energy watchdog). Every entry is constructed at its ``max_substeps`` rung so Newton allocates the substep
  scratch state (D21) and starts at ``EntrySpec.substeps``. WP2's ``PairPolicy`` carries no Newton proxy mapping, so
  one is built from the source entry (its bodies, else its particles), with a per-step destination collision
  pipeline (``collide_interval=1``: WP4 needs it under graph capture);
- ``"admm"``: ``alloy.coupling.SchedulerADMM`` (3+ entries or strong pairs).

``stock=True`` keeps WP2's stock proxy for two-entry scenes (comparison runs). ``entry_solvers`` returns the live
solver per entry name for any of these steppers (the Scheduler keeps them in the pinned private
``SolverCoupled._entries``, ``alloy.core.newton_compat.PRIVATE_HOOKS``).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.contact.particle_gains import attach as attach_record_gains
from alloy.contact.particle_gains import force_gains, particle_view
from alloy.core.errors import CouplingError
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.core.protocols import StepperLike
from alloy.core.registry import solver as capability
from alloy.coupling.full_surface import contact_view, moving_shapes
from alloy.coupling.proxy_contacts import CONTACT_FREE_DESTINATIONS, ProxyContactPipeline
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
from alloy.integration import mpm_clock, mpm_proxy
from alloy.integration.kinematic_carrier import (CARRIER, carried_into_mpm, carrier_entry, carrier_pairs, carrier_spec,
                                                 without_carrier)
from alloy.integration.precision_pin import pin_exchange
from alloy.scene.steppers import SingleEntryStepper, StockProxyStepper, solver_kwargs
from alloy.scene.types import BuiltScene, EntrySpec

log = logging.getLogger("alloy.integration")

__all__ = ["coupler_pairs", "dynamic_entries", "entry_solvers", "make_stepper", "mujoco_solver", "proxy_bodies"]


def dynamic_entries(built: BuiltScene) -> list[EntrySpec]:
    """Entries that own something dynamic (the assembler gives the static-shape holder an empty name)."""
    return [e for e in built.entries if e.name]


def make_stepper(built: BuiltScene, native: str, policy: SchedulerPolicy | None = None, *,
                 stock: bool = False) -> StepperLike:
    """The stepper ``Simulation`` runs for ``built`` (see module docstring)."""
    dyn = dynamic_entries(built)
    if not dyn:
        raise CouplingError("scene has no dynamic entry to step")
    if len(dyn) == 1:
        return SingleEntryStepper(built, dyn[0], native)
    if stock:
        return StockProxyStepper(built, native)
    dyn = adopt_kinematic(dyn, built.model)
    kind = built.coupler.kind
    if kind not in ("proxy", "admm"):
        raise CouplingError(f"{len(dyn)} dynamic entries but coupler kind {kind!r} (D5 expects proxy or admm)")
    dyn = _with_carrier(dyn, built.model, built.coupler)
    by_name = {e.name: e for e in dyn}
    pairs = coupler_pairs(built, dyn)
    pol = mpm_clock.pair_policy(policy if policy is not None else SchedulerPolicy(), built)   # S5, opt-in
    pol = pin_exchange(pol, built)                             # SCR-JOB2-1: a precision pin's cap wins
    pol = mpm_proxy.wall_policy(pol, built)                    # SPH-MPM particle walls: L2 force gate
    if kind == "admm":
        from alloy.coupling.scheduler import SchedulerADMM  # noqa: PLC0415

        entries = _with_force_gains([_alloy_entry(e, native) for e in dyn], built, pol)
        stepper = SchedulerADMM(built.model, entries, list(pairs), pol)
    else:
        from alloy.coupling.compose import build_coupled_stepper  # noqa: PLC0415

        # the Scheduler; with one-way pairs (SCR-DC5-7) wrapped in a OneWayFollower, one pass for a lone one-way pair
        proxied = [_with_proxy(p, by_name, pol, built.model) for p in pairs]
        floors = full_surface_iterations(proxied, by_name)
        entries = [carrier_entry(e) if e.capability == CARRIER else _alloy_entry(e, native, floors.get(e.name, 0))
                   for e in dyn]
        entries = _with_force_gains(_with_contact_view(entries, proxied), built, pol)
        stepper = build_coupled_stepper(built.model, entries, proxied, pol)
        carried_into_mpm(dyn, entry_solvers(stepper))
    log.info("stepper: %s over entries %s, pairs %s", type(stepper).__name__, [e.name for e in dyn],
             [(p.source, p.destination, p.mode) for p in pairs])
    return stepper


def adopt_kinematic(dyn: Sequence[EntrySpec], model: Any) -> list[EntrySpec]:
    """SCR-OSS3-5 / SCR-OSS22-6 (WP9's fix): KINEMATIC bodies no entry owns (a driven tray, a still jug) join the rigid
    entry with their joints, so its proxies carry them into the particle entries. Unowned, a coupled scene's SPH never
    saw them and the water fell through, and MuJoCo lost them too. ``sim.drive`` writes their body_q/qd in the
    full state, and the rigid entry and its proxies follow. A coupled scene with no rigid entry leaves them to
    ``_with_carrier``."""
    out = list(dyn)
    n = int(model.body_count or 0)
    if len(out) < 2 or n == 0:
        return out
    owned = {int(b) for e in out for b in e.bodies}
    flags = model.body_flags.numpy()
    kinematic = [b for b in range(n) if int(flags[b]) & int(newton.BodyFlags.KINEMATIC) and b not in owned]
    if not kinematic:
        return out
    hub = next((i for i, e in enumerate(out) if e.capability == "mujoco"), None)
    if hub is None:
        return out
    child = model.joint_child.numpy()
    joints = [j for j in range(int(model.joint_count or 0)) if int(child[j]) in set(kinematic)]
    e = out[hub]
    out[hub] = dataclasses.replace(e, bodies=tuple(sorted(set(e.bodies) | set(kinematic))),
                                   joints=tuple(sorted(set(e.joints) | set(joints))))
    log.info("entry %r owns the unowned kinematic bodies %s (joints %s) for the coupler (SCR-OSS3-5)", e.name,
             kinematic, joints)
    return out


def _with_carrier(dyn: Sequence[EntrySpec], model: Any, coupler: Any) -> list[EntrySpec]:
    """``dyn`` plus the kinematic carrier entry when no rigid entry owns the scene's kinematic bodies
    (``alloy.integration.kinematic_carrier``). They stay unowned (warned) under ADMM, which has no one-way stage, and
    in a scene whose own pairs hold a one-way pair (a follower leaf; the carrier's pairs would make its destination
    a chain: WP4's review. The assembler makes one-way pairs from a rigid hub only, so none reaches here today)."""
    spec = carrier_spec(dyn, model)
    if spec is None:
        return list(dyn)
    one_way = [(p.source, p.destination) for p in coupler.pairs if p.one_way]
    if coupler.kind != "proxy" or one_way:
        why = f"the {coupler.kind} coupler" if coupler.kind != "proxy" else f"a scene with one-way pairs {one_way}"
        log.warning("kinematic bodies %s: no rigid entry to own them and %s cannot carry them; the particle entries "
                    "%s do not see them (SCR-OSS3-5)", list(spec.bodies), why, [e.name for e in dyn])
        return list(dyn)
    log.info("entry %r owns the kinematic bodies %s (joints %s): no rigid entry carries them", spec.name,
             list(spec.bodies), list(spec.joints))
    return [*dyn, spec]


def _alloy_entry(spec: EntrySpec, native: str, min_iterations: int = 0) -> AlloyEntry:
    """A Newton ``SolverCoupled.Entry`` built at the construction rung (D21), started at the initial rung; its
    solver's ``iterations`` raised to ``min_iterations`` (``FULL_SURFACE_MIN_ITERATIONS``)."""
    cap, kw = capability(spec.capability), solver_kwargs(spec, native)
    if min_iterations and int(kw.get("iterations", 0)) < min_iterations:
        log.info("entry %r (%s): %d solver iterations, raised from %s for full-surface proxy contacts (G4)",
                 spec.name, spec.capability, min_iterations, kw.get("iterations"))
        kw["iterations"] = int(min_iterations)
    rung = max(int(spec.max_substeps), int(spec.substeps), 1)
    if len(spec.shapes):     # SCR-JOB2-5: Newton's Entry.shapes means owned shapes (and drops the default world walls)
        log.warning("entry %r: EntrySpec.shapes (%d) not passed to the coupler: Newton reads an entry's shapes as "
                    "owned; static walls reach particle entries without them (SCR-JOB2-5)", spec.name, len(spec.shapes))
    entry = SolverCoupledProxy.Entry(name=spec.name, solver=lambda view, cap=cap, kw=kw: cap.factory(view, kw),
                                bodies=list(spec.bodies), particles=list(spec.particles), joints=list(spec.joints),
                                substeps=rung)
    return AlloyEntry(entry, initial_rung=max(int(spec.substeps), 1), adaptive=int(spec.max_substeps) > 1,
                      capability=spec.capability)


def _with_contact_view(entries: list[AlloyEntry], pairs: Sequence[PairPolicy]) -> list[AlloyEntry]:
    """SCR-SC4-4: a destination whose proxy pipeline has full-surface contacts sees those shapes as reduced hulls in
    its own view (``alloy.coupling.full_surface.contact_view``)."""
    shapes: dict[str, set[int]] = {}
    for p in pairs:
        pipe = getattr(p.proxy, "collision_pipeline", None) if p.proxy is not None else None
        if isinstance(pipe, ProxyContactPipeline) and pipe.full_surface and pipe.full_surface_shapes is not None:
            shapes.setdefault(p.destination, set()).update(int(x) for x in pipe.full_surface_shapes)
    out = []
    for ae in entries:
        if ae.entry.name in shapes and ae.entry.configure_view is None:
            view = contact_view(sorted(shapes[ae.entry.name]))
            ae = dataclasses.replace(ae, entry=dataclasses.replace(ae.entry, configure_view=view))
        out.append(ae)
    return out


def _with_force_gains(entries: list[AlloyEntry], built: BuiltScene, pol: SchedulerPolicy) -> list[AlloyEntry]:
    """Destinations with particles read the card's K on MuJoCo-body shapes, not MuJoCo's acceleration-space native
    gains, and their records against those bodies are sized per body to the exchange clock's bound
    (``alloy.contact.particle_gains``; C3 box_on_pad: VBD's reaction was 1,300x the pad's impulse)."""
    gains = force_gains(built)
    if gains is None:
        return entries
    bound = float(pol.clock.safety) * float(pol.clock.omega_dt_max)
    out = []
    for ae in entries:
        if ae.capability != "mujoco" and ae.entry.particles:
            view = particle_view(gains, then=ae.entry.configure_view)
            ae = dataclasses.replace(ae, entry=dataclasses.replace(ae.entry, configure_view=view,
                                                                   solver=_attached(ae.entry.solver, gains, bound)))
        out.append(ae)
    return out


def _attached(factory: Any, gains: Any, bound: float) -> Any:
    """An entry's solver factory that sizes the coupled records per body (``particle_gains.attach``)."""

    def make(view: Any) -> Any:
        solver = factory(view)
        attach_record_gains(solver, gains, bound)
        return solver

    return make


def coupler_pairs(built: BuiltScene, dyn: Sequence[EntrySpec] | None = None) -> tuple[PairPolicy, ...]:
    """The scene's coupler pairs, or the default source -> destination pair of a two-entry scene; with a kinematic
    carrier among ``dyn``, plus its one-way pair into every other entry."""
    dyn = dynamic_entries(built) if dyn is None else dyn
    return (tuple(built.coupler.pairs) or (_default_pair(without_carrier(dyn)),)) + carrier_pairs(dyn)


def proxy_bodies(built: BuiltScene, source_capability: str = "mujoco") -> dict[str, set[int]]:
    """Global body ids each destination entry receives as proxies from a ``source_capability`` source under the proxy
    coupler: the pair's own Newton mapping if it names one, else the source entry's bodies (``_with_proxy``)."""
    dyn = adopt_kinematic(dynamic_entries(built), built.model)
    if built.coupler.kind != "proxy" or len(dyn) < 2:
        return {}
    by_name = {e.name: e for e in dyn}
    out: dict[str, set[int]] = {}
    pairs = tuple(built.coupler.pairs) or ((_default_pair(dyn),) if len(dyn) == 2 else ())   # no proxy to report
    for pair in pairs:
        src = by_name.get(pair.source)
        if src is None or src.capability != source_capability or pair.destination not in by_name:
            continue
        bodies = pair.proxy.bodies if pair.proxy is not None else src.bodies
        out.setdefault(pair.destination, set()).update(int(b) for b in bodies)
    return out


def _default_pair(dyn: Sequence[EntrySpec]) -> PairPolicy:
    """Two entries with no pair from the assembler: the first entry (rigid by assembler order) is the source."""
    if len(dyn) != 2:
        raise CouplingError(f"no coupler pairs for {len(dyn)} entries (the assembler must list them, D5)")
    return PairPolicy(source=dyn[0].name, destination=dyn[1].name, mode="lagged")


def _proxy_particles(src: EntrySpec, dst: EntrySpec, model: Any) -> list[int]:
    """A particle source's proxied particles: all of them, except into SPH, which takes only surface vertices (the
    vertices of ``model.tri_indices``): its soft faces are triangles with three proxied vertices, so interior tet
    vertices would cost sync and residual entries for nothing (SPH-VBD P1, WP9)."""
    particles = list(src.particles)
    tris = getattr(model, "tri_indices", None)
    if dst.capability != "sph" or tris is None or not int(getattr(model, "tri_count", 0) or 0):
        return particles
    surface = np.intersect1d(np.asarray(particles, dtype=np.int64), np.unique(tris.numpy().reshape(-1)))
    if not surface.size:
        return particles
    log.info("proxy %s -> %s: %d surface vertices of %d particles", src.name, dst.name, surface.size, len(particles))
    return surface.tolist()


def _with_proxy(pair: PairPolicy, by_name: Mapping[str, EntrySpec], pol: SchedulerPolicy, model: Any) -> PairPolicy:
    """Fill ``PairPolicy.proxy`` (Newton's mapping of the source entry into the destination) when absent."""
    if pair.proxy is not None:
        return pair
    src = by_name.get(pair.source)
    if src is None or pair.destination not in by_name:
        raise CouplingError(f"pair {pair.source}->{pair.destination} names an entry the scene does not have "
                            f"({sorted(by_name)})")
    bodies = list(src.bodies)
    particles = [] if bodies else _proxy_particles(src, by_name[pair.destination], model)
    relax = "aitken" if pair.mode == "staggered" else "fixed"
    pipeline = _proxy_pipeline(model, by_name[pair.destination], pol)
    proxy = SolverCoupledProxy.Proxy(source=pair.source, destination=pair.destination, bodies=bodies,
                                     particles=particles, mode=pair.mode, proxy_relaxation_mode=relax,
                                     collision_pipeline=pipeline, collide_interval=None if pipeline is None else 1)
    return dataclasses.replace(pair, proxy=proxy)




def _proxy_pipeline(model: Any, dst: EntrySpec, pol: SchedulerPolicy) -> ProxyContactPipeline | None:
    """The proxy's contacts (SCR-DC5-19): none for a destination that ignores contacts; else soft pairs that can
    collide, rigid pairs only on the destination's own bodies (none for a particle-only destination), per-world caps
    from the policy. A destination with particles gives soft candidates to those it solves only (WP10 lever #4, WP3
    2026-11-07): the view holds every entry's particles (a robot scene's SPH water beside a VBD towel: 2,940 of 3,616
    per world, each tested against all 129 Panda meshes); the destination solves its own and those proxied into it
    (Newton's coupler disables the rest: zero mass, not ACTIVE)."""
    if dst.capability in CONTACT_FREE_DESTINATIONS:
        return None
    shape_body = model.shape_body.numpy() if int(model.shape_count or 0) else np.zeros(0, np.int32)
    owned = np.flatnonzero(np.isin(shape_body, np.asarray(dst.bodies, dtype=shape_body.dtype)))
    full = _full_surface(dst, pol)
    sticky = {"contact_matching": "sticky"} if _steps_a_rod(model, dst) else {}
    return ProxyContactPipeline(pol.proxy_soft_contacts_per_world, pol.proxy_rigid_contacts_per_world,
                                rigid_shapes=owned.tolist(), full_surface=full,
                                full_surface_shapes=moving_shapes(model) if full else None,
                                particles=list(getattr(dst, "particles", ())) or None, proxied=True, **sticky)


def _steps_a_rod(model: Any, dst: EntrySpec) -> bool:
    """True when the destination steps an elastic rod (a ROD joint's child among its bodies). Its proxy contacts then
    match "sticky": a matched contact keeps its body-frame points across steps, so friction holds a gripped rod
    instead of dragging it on after the gripper stops (SCR-WP9-ROD F3: docs/notes/wp9-results-rod.md, F3).
    Every other destination keeps unmatched contacts, bit for bit."""
    if not int(model.joint_count or 0) or not dst.bodies:
        return False
    rod = model.joint_type.numpy() == int(newton.JointType.ROD)
    return bool(np.isin(model.joint_child.numpy()[rod], np.asarray(dst.bodies)).any())


# SCR-SC4-4b known gap (coordinator ruling 2026-11-04 20:08Z), keyed by the destination entry's capability: these
# keep per-vertex proxy contact when the policy leaves it automatic, until the follow-up passes. C16's film gather
# (SolverFilm) latches PEN_FAIL with edge/face records: an UNCONVERGED exchange, then a finger kicked to 2.5-5 m/s.
# SchedulerPolicy(proxy_full_surface=True) still turns them on (the follow-up's runs).
FULL_SURFACE_KNOWN_GAPS: Mapping[str, str] = {
    "film": "SCR-SC4-4b (WP4): C16's film gather latches PEN_FAIL with full-surface contacts (re-checked at 40 VBD "
            "iterations 2026-11-08, one world per run: 1 of 4 latched at 4.5 s, in 2 a strap slid out of the pinch "
            "during the spread, the 4th reached 6.9 s unlatched in the 5.5 h cap; [internal note] notes/sc4-c16-1108); "
            "the film pinch law passes full-surface at 40, drops the film at 10 (G4, "
            "tests/coupling/test_film_pinch_law.py)",
}


# G4 (alloy-sc4, 2026-11-06, scr/SC4.md "G4 root cause"): full-surface proxy contacts give a clamped film particle
# several records (vertex-face per nearby triangle, face-vertex, edge-edge) where per-vertex gives one. At 1/960 s,
# 10 VBD iterations do not converge the clamp's normal/friction balance and the film creeps out of the pinch; 40 hold
# it (film pinch law, full-surface: coupled 45.7 of 50 mm vs plain 43.9, gap >= 1.11 mm; CPU wall x2.8). Keyed by the
# destination's capability: an entry receiving full-surface proxy contacts runs at least this many iterations.
FULL_SURFACE_MIN_ITERATIONS: Mapping[str, int] = {"film": 40}


def full_surface_iterations(pairs: Sequence[PairPolicy], by_name: Mapping[str, EntrySpec]) -> dict[str, int]:
    """Entry name -> iteration floor, for destinations whose proxy pipeline has full-surface contacts."""
    out: dict[str, int] = {}
    for p in pairs:
        pipe = getattr(p.proxy, "collision_pipeline", None) if p.proxy is not None else None
        dst = by_name.get(p.destination)
        if dst is None or not (isinstance(pipe, ProxyContactPipeline) and pipe.full_surface):
            continue
        floor = FULL_SURFACE_MIN_ITERATIONS.get(dst.capability, 0)
        if floor:
            out[p.destination] = max(out.get(p.destination, 0), floor)
    return out


def _full_surface(dst: EntrySpec, pol: SchedulerPolicy) -> bool:
    """SCR-SC4-4: edge/face soft contacts for a destination whose solver consumes them (SolverVBD or a subclass:
    vbd, film). Per-vertex contact lets a rigid feature narrower than the cloth's vertex gap pass between vertices,
    which threads the cloth onto a gripper pad (C7). ``proxy_full_surface`` overrides; True elsewhere raises
    (Newton's other solvers raise on such records). Automatic leaves ``FULL_SURFACE_KNOWN_GAPS`` per-vertex."""
    cls = capability(dst.capability).solver_cls
    consumes = isinstance(cls, type) and issubclass(cls, newton.solvers.SolverVBD)
    if pol.proxy_full_surface is None:
        gap = FULL_SURFACE_KNOWN_GAPS.get(dst.capability) if consumes else None
        if gap:
            log.warning("proxy contacts for %s (%s): per-vertex, known gap %s", dst.name, dst.capability, gap)
        return consumes and not gap
    if pol.proxy_full_surface and not consumes:
        raise CouplingError(f"proxy_full_surface: destination {dst.name} ({dst.capability}) does not consume "
                            "edge/face soft contacts (only SolverVBD and its subclasses do)")
    return bool(pol.proxy_full_surface)


def entry_solvers(stepper: Any) -> dict[str, SolverBase]:
    """Live solver per entry name: single-entry, stock proxy, Scheduler/SchedulerADMM or a fake with ``solver``."""
    if isinstance(stepper, SingleEntryStepper):
        return {stepper.name: stepper.solver}
    inner = getattr(stepper, "solver", None) if isinstance(stepper, StockProxyStepper) else stepper
    entries = getattr(inner, "_entries", None)
    if isinstance(entries, Mapping):
        return {n: e.solver for n, e in entries.items() if getattr(e, "solver", None) is not None}
    single = getattr(stepper, "solver", None)
    return {"solver": single} if isinstance(single, SolverBase) else {}


def mujoco_solver(solvers: Mapping[str, SolverBase]) -> newton.solvers.SolverMuJoCo | None:
    """The SolverMuJoCo entry, if any (the precision append binds to it under ``native="mujoco"``, D10)."""
    found = [s for s in solvers.values() if isinstance(s, newton.solvers.SolverMuJoCo)]
    if len(found) > 1:
        raise CouplingError(f"{len(found)} SolverMuJoCo entries; the precision append binds to exactly one")
    return found[0] if found else None
