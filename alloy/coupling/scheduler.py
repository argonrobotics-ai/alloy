# SPDX-License-Identifier: Apache-2.0
"""Scheduler (proxy coupling) and SchedulerADMM on Newton's coupled framework (SPEC §3.6, D5, D18, D21).

``Scheduler`` subclasses ``SolverCoupledProxy`` and overrides two private methods pinned in
``alloy.core.newton_compat.PRIVATE_HOOKS``: ``_step_coupled`` (residual-gated passes with per-world freeze) and
``_blend_proxy_feedback`` (lagged ω = 1; staggered per-world Aitken). Newton's ``_step_proxy`` runs unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.core.defaults import GRAVITY
from alloy.core.errors import CouplingError
from alloy.core.protocols import DetCapability
from alloy.core.newton_compat import SolverCoupledADMM, SolverCoupledProxy
from alloy.coupling import register_builtins
from alloy.coupling.admm_loop import AdmmIterationConfig, AdmmIterationLoop
from alloy.coupling.axis_mass import ProxyMassMixin
from alloy.coupling.clock_census import ClockCensusMixin
from alloy.coupling.clock_memory import ClockMemoryMixin
from alloy.coupling.contact_gate import gate_for
from alloy.coupling.endpoints import EndpointSet, build_endpoint_set, clock_masses
from alloy.coupling.graph_budget_rungs import GraphBudgetRungsMixin
from alloy.coupling.kinematic import KinematicProxyMixin
from alloy.coupling.match_reset import MatchResetMixin
from alloy.coupling.mixin import NON_PRECISION_TOL_SCALE, SchedulerMixin
from alloy.coupling.proxy_contacts import (CONTACT_FREE_DESTINATIONS, ProxyContactPipeline, contact_free,
                                          entry_contact_sizing)
from alloy.coupling.proxy_loop import IterationConfig, ProxyIterationLoop
from alloy.coupling.proxy_tensor import ProxyTensorMixin
from alloy.coupling.record_momentum import install_record_momentum
from alloy.coupling.static_walls import static_wall_shapes
from alloy.coupling.stiffness_cap import ExchangeStiffnessCap, bind_axis_masses, install_stiffness_caps
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
from alloy.coupling.vbd_iterations import VbdStepIterationsMixin

log = logging.getLogger("alloy.coupling")
register_builtins()

STAGGERED_MODE = 1   # newton _ProxyMode.STAGGERED (solver_coupled_proxy.py:95-100)


def strong_pair(m_a: float, m_s: float, policy: PairPolicy) -> bool:
    """D5: strong if m_a/m_s > added_mass_ratio_max, a saturating actuator clamps through it, or it must hold
    static friction below CREEP_GENERAL. ``policy.added_mass`` (fluid pairs: C_a·ρ_f·V_sub) overrides ``m_a``."""
    added = policy.added_mass if policy.added_mass is not None else m_a
    if policy.saturating_actuator or policy.static_hold:
        return True
    if m_s <= 0.0:
        return added > 0.0
    return added / m_s > policy.added_mass_ratio_max


def _native(solver: Any) -> bool:
    return getattr(solver, "determinism", None) == DetCapability.NATIVE


def _validate_pair(pair: PairPolicy) -> None:
    if pair.one_way and pair.mode != "lagged":
        raise CouplingError(f"one-way pair {pair.source}->{pair.destination}: mode must be 'lagged' (got "
                            f"{pair.mode!r}); a one-way destination is stepped once per exchange, never iterated")
    if pair.mode in ("admm", "monolithic"):
        raise CouplingError(f"pair {pair.source}->{pair.destination}: mode {pair.mode!r} needs SchedulerADMM or a "
                            "monolithic route, not the proxy Scheduler")
    if pair.proxy is None:
        raise CouplingError(f"pair {pair.source}->{pair.destination}: a lagged/staggered pair needs a "
                            "SolverCoupledProxy.Proxy mapping")
    if pair.mode == "staggered" and str(pair.proxy.proxy_relaxation_mode).lower() == "fixed":
        raise CouplingError(
            f"pair {pair.source}->{pair.destination}: staggered coupling with fixed relaxation is rejected; "
            "fixed ω = 1 diverged on box-on-cloth (coupling-time §3.2). Use proxy_relaxation_mode='aitken'.")


def _iteration_config(pairs: Sequence[PairPolicy], policy: SchedulerPolicy) -> IterationConfig:
    tol, ait = policy.tol, policy.aitken
    staggered = any(p.mode == "staggered" for p in pairs)
    precision = any(p.precision for p in pairs)
    scale = 1.0 if precision else NON_PRECISION_TOL_SCALE
    if staggered:
        k_min, k_max = tol.k_min, max(tol.k_max, policy.lagged_iterations)
        if any(p.mode == "lagged" for p in pairs):        # a chain (P3): its lagged pairs keep their 2 passes
            k_min = max(k_min, policy.lagged_iterations)
    else:
        k_min = k_max = policy.lagged_iterations
    return IterationConfig(k_min=k_min, k_max=k_max, tol_vel=tol.vel * scale, tol_pos=tol.pos * scale,
                           tol_force=tol.force_rel, force_floor=tol.force_floor, gravity=GRAVITY,
                           omega0=ait.omega0, omega0_strong=ait.omega0_strong, omega_min=ait.omega_min,
                           omega_max=ait.omega_max, carry_omega=ait.carry_across_steps,
                           slew_c=policy.exchange_slew_c, force_norm=tol.force_norm)


def _contact_factories_of(pairs: Sequence[PairPolicy]) -> list[ProxyContactPipeline]:
    """The pairs' proxy pipelines built by ``ProxyContactPipeline`` (SCR-DC5-19)."""
    return [p.proxy.collision_pipeline for p in pairs
            if isinstance(getattr(p.proxy, "collision_pipeline", None), ProxyContactPipeline)]


def _one_pass_config(policy: SchedulerPolicy) -> IterationConfig:
    """Iteration settings for a Scheduler whose only pair is one-way: one pass (no residual loop runs)."""
    cfg = _iteration_config([], policy)
    return dataclasses.replace(cfg, k_min=1, k_max=1)


class Scheduler(ClockMemoryMixin, KinematicProxyMixin, ProxyTensorMixin, ProxyMassMixin, ClockCensusMixin,
                VbdStepIterationsMixin, MatchResetMixin, GraphBudgetRungsMixin, SchedulerMixin, SolverCoupledProxy):
    """Proxy coupling with residual-gated iterations, adaptive clocks and an energy watchdog: two entries, or more over
    an acyclic proxy graph where Newton supports it (``alloy.coupling.compose``)."""

    def __init__(self, model: Any, entries: Sequence[AlloyEntry], pairs: Sequence[PairPolicy],
                 policy: SchedulerPolicy = SchedulerPolicy(), *, capture_while: bool | None = None,
                 proxy_tensor: bool | None = None) -> None:
        """``proxy_tensor``: install per-body proxy mass tensors in VBD destinations (None: env ALLOY_PROXY_TENSOR).
        Side simulations of the same scene (``rerun_worlds`` packs, ``processes=`` workers) read only the environment
        variable: set ALLOY_PROXY_TENSOR, not this argument, when they are in use.
        ``capture_while``: run the residual loop as a CUDA conditional node under graph capture. None (auto) =
        only when every entry solver is NATIVE (Alloy kernels, allocation-free); Newton's solvers allocate inside a
        step, which conditional bodies forbid, so their loop is unrolled under capture: to the passes that converge
        worlds, with short worlds re-run by the full loop (``capture_passes``, lever #1)."""
        register_builtins()                             # into the registry in use (isolated_registry, WP12)
        if not pairs:
            raise CouplingError("Scheduler needs at least one PairPolicy (one dynamic entry needs no coupler, D5)")
        for pair in pairs:
            _validate_pair(pair)
        proxies = [dataclasses.replace(p.proxy, source=p.source, destination=p.destination, mode=p.mode,
                                       proxy_relaxation=1.0, proxy_relaxation_mode="fixed") for p in pairs]
        coupled = [p for p in pairs if not p.one_way]
        cfg = _iteration_config(coupled, policy) if coupled else _one_pass_config(policy)
        coupling = SolverCoupledProxy.Config(proxies=proxies, iterations=cfg.k_max)
        self._proxy_tensor_on = proxy_tensor            # read when Newton installs the proxy masses
        self._init_kinematic(model)                     # SCR-SC8-2: before Newton installs proxy inertia
        with entry_contact_sizing(_contact_factories_of(pairs)):    # SCR-DC5-19: VBD contact state hint
            super().__init__(model=model, entries=[ae.entry for ae in entries], coupling=coupling)
        self._init_kinematic_guard()
        self._one_way_routes = frozenset((p.source, p.destination) for p in pairs if p.one_way)
        self._one_way_passes = 1                       # SCR-DC5-8; tests set 2 to prove the second pass neutral
        self._caps = install_stiffness_caps(self, policy.exchange_stiffness_c,   # SCR-SC4-4d
                                            policy.exchange_damping_c)
        self._record_momentum = install_record_momentum(self)  # gaps-now #10a: feedback = the particles' impulse
        self._sets = self._build_sets(coupled, policy)
        bind_axis_masses(self._caps, self._sets)
        if capture_while is None:
            capture_while = all(_native(self._entries[ae.entry.name].solver) for ae in entries)
        self._loop = ProxyIterationLoop(self, self._sets, cfg, capture_while=capture_while)
        if not self._sets:
            self._loop.iters_used.fill_(1)         # one-way only: every exchange is one pass
        self._init_scheduler(entries, pairs, policy)
        self._pair_mass = self._pair_masses()
        self._gate = gate_for(self)     # lever #2: None unless ClockPolicy.contact_gate

    def _build_sets(self, pairs: Sequence[PairPolicy], policy: SchedulerPolicy) -> list[EndpointSet]:
        by_route = {(p.source, p.destination): p for p in pairs}
        sets = []
        for kind, mappings in (("body", self._proxy_mappings), ("particle", self._proxy_particle_mappings)):
            for m in mappings:
                if (m.src_name, m.dst_name) not in by_route:   # one-way: no residuals, Aitken or bond
                    continue
                es = build_endpoint_set(m, kind, self._entries[m.src_name], self._entries[m.dst_name], self.model,
                                        policy.aitken.omega0)
                pair = by_route[(m.src_name, m.dst_name)]
                es.staggered = pair.mode == "staggered"
                sets.append(es)
        return sets

    def _pair_masses(self) -> dict[tuple[str, str], float]:
        """Per pair, its lightest proxy's mass along the source body's most mobile axis (``axis_mass``: the installed
        isotropic proxy mass is heavier along it), for the clock bound and the strong-pair class; proxies that cannot
        touch the destination are left out (``endpoints.clock_masses``)."""
        out: dict[tuple[str, str], float] = {}
        for es in self._sets:
            m = clock_masses(es)
            if m.size:
                key = (es.src_name, es.dst_name)
                out[key] = min(out.get(key, np.inf), float(np.min(m[m > 0])) if np.any(m > 0) else np.inf)
        for pair in self._alloy_pairs:
            if pair.one_way:                       # no feedback, so no added-mass instability (D5 does not apply)
                continue
            key = (pair.source, pair.destination)
            m_s = out.get(key, 0.0)
            if strong_pair(0.0, m_s, pair):
                self._strong.add(key)
                for es in self._sets:
                    if (es.src_name, es.dst_name) == key:
                        es.strong = True
                log.warning("pair %s->%s is strong (D5); route it monolithic or to ADMM", *key)
        return out

    # ------------------------------------------------------------------ pinned private overrides
    def _step_coupled(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        del state_out
        for cap in self._caps:                     # SCR-SC4-4d: the cap is c m_eff / dt_x^2 at this exchange
            cap.dt_x = dt
        if not self._sets:                         # one-way pair only (SCR-DC5-8): one pass, nothing to iterate
            self._one_way_step(state_in, control, contacts, dt)
            return
        self._loop.run(state_in, control, contacts, dt)
        if self._gate is not None:                  # lever #2: flag unarmed proxies that touched
            self._gate.after_exchange()
        for es in self._sets:
            self._watchdog.accumulate_bond(es, self._entries[es.src_name], self._entries[es.dst_name], dt)

    def _one_way_step(self, state_in: Any, control: Any, contacts: Any, dt: float) -> None:
        """The source steps, its proxies sync, the destination steps at its rung; a second pass would repeat the
        same inputs (the feedback is held at zero) and is skipped."""
        for k in range(self._one_way_passes):
            if k > 0:
                self._distribute_state(state_in, dt=dt, iteration_restart=True)
            self._step_proxy(state_in, control, contacts, dt, iteration_restart=k > 0)

    def _contact_factories(self) -> list[Any]:
        return _contact_factories_of(self._alloy_pairs)

    def _contact_routes(self) -> list[tuple[Any, str]]:
        return [(p.proxy.collision_pipeline, p.destination) for p in self._alloy_pairs
                if isinstance(getattr(p.proxy, "collision_pipeline", None), ProxyContactPipeline)]

    def _blend_proxy_feedback(self, proxy: Any) -> None:
        if (proxy.src_name, proxy.dst_name) in self._one_way_routes:
            proxy.coupling_forces.zero_()          # one-way: the source never receives the destination's load
            return
        if not self._loop.on_harvest(proxy):
            super()._blend_proxy_feedback(proxy)

    # ------------------------------------------------------------------ mixin hooks
    def _escalate_mode(self) -> bool:
        """Second escalation rung: lagged -> staggered with Aitken ω0_strong (coupling-time §4.4.5)."""
        switched = False
        for es in self._sets:
            if not es.staggered:
                es.staggered = True
                es.strong = True
                es.mapping.mode = STAGGERED_MODE
                switched = True
        if switched:
            cfg = self._loop.cfg
            tol = self._alloy_policy.tol
            self._loop.cfg = dataclasses.replace(cfg, k_min=tol.k_min, k_max=max(tol.k_max, cfg.k_max))
            log.warning("coupling escalation: lagged pairs switched to staggered + Aitken (omega0 %.2f)",
                        cfg.omega0_strong)
        return switched

    def _pair_stiffness_mass(self) -> list[tuple[float, float]]:
        """(k_c, m_eff) per pair for the exchange-clock bound. A pair-stated ``interface_stiffness`` (D17) always binds.
        Otherwise a contact-based destination takes the contact stiffness (``_default_stiffness``), and a contact-free
        one (SPH: no penalty spring at the wall, its force comes from the pressure solve) gets no bound: the
        added-mass instability does not shrink with dt; D5's coupling mode and the watchdog handle it (SCR-JOB2-3)."""
        out = []
        for pair in self._alloy_pairs:
            k_c = self._pair_k_c(pair)
            if k_c is not None:
                out.append((k_c, float(self._pair_mass.get((pair.source, pair.destination), np.inf))))
        return out

    def _pair_k_c(self, pair: PairPolicy) -> float | None:
        """The pair's contact stiffness for the clock bound: its stated ``interface_stiffness``, else the
        destination's contact stiffness; None for a contact-free destination (SPH)."""
        k_c = getattr(pair, "interface_stiffness", None)
        if k_c:
            return float(k_c)
        ae = self._alloy_entries.get(pair.destination)
        if contact_free(self._entries[pair.destination].solver, getattr(ae, "capability", None)):
            return None
        return float(self._default_stiffness())

    def _interface_bound(self) -> float:
        """With the contact gate on (lever #2), the bound over the armed proxies only (contact_gate.py); the contact-band
        bound (R2-E3) holds whether or not a pair is armed."""
        if self._gate is None:
            return super()._interface_bound()
        return min(self._gate.bound(), self._band_bound)

    def _pair_damping_mass(self) -> list[tuple[float, float]]:
        """(c_d, m_eff) per pair whose destination resolves contacts: the destination's contact damping
        (``_default_damping``), which the lagged exchange holds as a force for a whole interval. Contact-free
        destinations (SPH) have none."""
        return [(c_d, float(self._pair_mass.get((pair.source, pair.destination), np.inf)))
                for pair in self._alloy_pairs if (c_d := self._pair_c_d(pair)) is not None]

    def _pair_c_d(self, pair: PairPolicy) -> float | None:
        """The pair's contact damping for the clock bound; None when undamped or contact-free (SPH)."""
        c_d = self._default_damping()
        ae = self._alloy_entries.get(pair.destination)
        if not c_d > 0.0 or contact_free(self._entries[pair.destination].solver, getattr(ae, "capability", None)):
            return None
        return c_d

    def _default_damping(self) -> float:
        """Destination contact damping [N s/m]: VBD's body-particle mix, the mean of soft_contact_kd and the shape's kd
        (newton rigid_vbd_kernels: avg_kd), with the largest shape kd (Newton's defaults 10 and 100: 55)."""
        kd = float(getattr(self.model, "soft_contact_kd", 0.0) or 0.0)
        if self.model.shape_count:
            kd = 0.5 * (kd + float(np.max(self.model.shape_material_kd.numpy())))
        return kd

    def _default_stiffness(self) -> float:
        """Destination contact stiffness when the pair does not state one: max(soft_contact_ke, shape ke) [N/m]."""
        ke = float(getattr(self.model, "soft_contact_ke", 0.0) or 0.0)
        if self.model.shape_count:
            ke = max(ke, float(np.max(self.model.shape_material_ke.numpy())))
        return ke

    def _loop_arrays(self) -> Mapping[str, Any]:
        lp = self._loop
        return {"res_vel": lp.res_vel, "res_pos": lp.res_pos, "res_force": lp.res_force, "iters_used": lp.iters_used}

    def _coupler_state_arrays(self) -> Mapping[str, wp.array]:
        return self._loop.state_arrays()

    def _coupler_world_maps(self) -> Mapping[str, wp.array]:
        return self._loop.world_maps()

    @property
    def endpoint_sets(self) -> tuple[EndpointSet, ...]:
        """Per-mapping endpoint arrays (f_used, f_raw, residual inputs) for tests and metrics."""
        return tuple(self._sets)

    @property
    def feedback_slew(self) -> Any:
        """SCR-SC4-4d lagged feedback slew limit (``alloy.coupling.slew.FeedbackSlewLimit``), or None."""
        return self._loop.slew

    @property
    def stiffness_caps(self) -> tuple[ExchangeStiffnessCap, ...]:
        """SCR-SC4-4d exchange stiffness caps, one per VBD destination of a coupled body-proxy pair."""
        return tuple(self._caps)

    @property
    def iteration_loop(self) -> ProxyIterationLoop:
        return self._loop


def admm_contact_pairs(entries: Sequence[AlloyEntry], pairs: Sequence[PairPolicy]) -> list[Any]:
    """Newton ``ContactPair``s for SchedulerADMM: one per two-way pair. Newton's ``Config.contact_pairs`` defaults to
    (), which turns every ADMM-managed cross-entry contact off (WP12, engine item 2: an ADMM-routed body fell through
    its partner). A pair with a contact-free end (``CONTACT_FREE_DESTINATIONS``: its solver reads neither contacts nor
    the particle forces that carry ADMM's rows) gets none, as its proxy gets no pipeline on the Scheduler path."""
    capability = {ae.entry.name: ae.capability for ae in entries}
    out, seen = [], set()
    for pair in pairs:
        key = frozenset((pair.source, pair.destination))
        if pair.one_way or key in seen:
            continue
        seen.add(key)
        free = [n for n in (pair.source, pair.destination) if capability.get(n) in CONTACT_FREE_DESTINATIONS]
        if free:      # either end: D5 orders two non-rigid entries by name, so MPM can be the source
            log.warning("SchedulerADMM: %s<->%s has a contact-free end (%s): no ADMM contact rows, the pair carries "
                        "nothing", pair.source, pair.destination, ", ".join(f"{n}: {capability[n]}" for n in free))
            continue
        out.append(SolverCoupledADMM.ContactPair(source=pair.source, destination=pair.destination))
    return out


def admm_entry_views(model: Any, entries: Sequence[AlloyEntry]) -> list[AlloyEntry]:
    """Entries that own no body and no shape get a ``configure_view`` that stops body shapes from colliding in their
    view. Newton hides other entries' shapes from every entry that has something visible, but
    ``SolverCoupled._apply_entry_shape_visibility`` returns early for an entry with nothing visible: a particle-only
    entry in a scene without static shapes. That entry kept a partner's body shapes as static colliders beside
    ADMM's rows: VBD pushed a cloth out of a falling MuJoCo box one way and the box fell through (box-on-cloth without
    a ground; with this hook it rests at 1.036 m, the lagged proxy at 1.035). A no-op when Newton already hid them.
    Static walls on world-welded bodies (``static_wall_shapes``, SCR-JOB2-5) stay visible to particles."""
    if not int(model.shape_count or 0):
        return list(entries)
    walls = np.asarray(static_wall_shapes(model), dtype=np.int64)
    on_body = np.setdiff1d(np.flatnonzero(model.shape_body.numpy() >= 0), walls)
    if not on_body.size:
        return list(entries)
    out = []
    for ae in entries:
        e = ae.entry
        if len(e.bodies) or len(getattr(e, "shapes", ()) or ()):
            out.append(ae)
            continue
        hook = _hide_body_shapes(int(model.shape_count), on_body, e.name, e.configure_view)
        out.append(dataclasses.replace(ae, entry=dataclasses.replace(e, configure_view=hook)))
    return out


def _hide_body_shapes(count: int, shapes: np.ndarray, name: str, then: Any) -> Any:
    """``configure_view``: clear the collision flags of ``shapes`` (model ids), then run the entry's own hook. Only a
    full-layout view can hold the leak: a compacted view keeps just the entry's visible shapes (world shapes and
    static walls for a particle-only entry), and its own ``shape_body`` cannot tell a partner's shape from a world one
    (both read -1), so a compacted view is left as Newton built it."""
    mask = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES | newton.ShapeFlags.HYDROELASTIC)

    def configure(view: Any) -> None:
        if int(view.shape_count or 0) != count:
            log.debug("SchedulerADMM: entry %r view is compacted (%s of %d shapes): no partner body shapes to hide",
                      name, view.shape_count, count)
            if then is not None:
                then(view)
            return
        flags = view.shape_flags.numpy().copy()
        if np.any(flags[shapes] & mask):
            flags[shapes] &= ~mask
            view.shape_flags = wp.array(flags, dtype=view.shape_flags.dtype, device=view.shape_flags.device)
            log.info("SchedulerADMM: entry %r owns no body or shape; %d body shapes stop colliding in its view (ADMM "
                     "rows carry them)", name, shapes.size)
        if then is not None:
            then(view)

    return configure


class SchedulerADMM(ClockMemoryMixin, ProxyMassMixin, MatchResetMixin, GraphBudgetRungsMixin, SchedulerMixin,
                    SolverCoupledADMM):
    """Three or more entries, or strong pairs without a monolithic route (D5): Newton's Jacobi ADMM with a Boyd-style
    residual stop (2 <= k <= 20), per-world freeze, the shared clock/watchdog/report.

    Without ``admm_config``, cross-entry contacts are ADMM rows for every two-way pair (``admm_contact_pairs``); a
    caller's ``admm_config`` is used as given (Newton's default there has no contact rows). A particle-only entry's
    view never collides with body shapes (``admm_entry_views``)."""

    def __init__(self, model: Any, entries: Sequence[AlloyEntry], pairs: Sequence[PairPolicy],
                 policy: SchedulerPolicy = SchedulerPolicy(), admm_config: Any | None = None, *,
                 capture_while: bool | None = None) -> None:
        register_builtins()
        for pair in pairs:
            if pair.one_way:
                raise CouplingError(f"SchedulerADMM has no one-way stage ({pair.source}->{pair.destination}); "
                                    "compose it with alloy.coupling.compose.build_coupled_stepper")
        k_min, k_max = policy.admm_k
        cfg = admm_config if admm_config is not None else SolverCoupledADMM.Config(
            contact_pairs=admm_contact_pairs(entries, pairs))
        cfg = dataclasses.replace(cfg, iterations=int(k_max))
        entries = admm_entry_views(model, entries)
        super().__init__(model=model, entries=[ae.entry for ae in entries], coupling=cfg)
        if capture_while is None:
            capture_while = all(_native(self._entries[ae.entry.name].solver) for ae in entries)
        self._admm = AdmmIterationLoop(self, AdmmIterationConfig(int(k_min), int(k_max), policy.admm_eps_abs,
                                                                 policy.admm_eps_rel), capture_while=capture_while)
        self._init_scheduler(entries, pairs, policy)
        self._rho_dt = float(policy.admm_rho_dt) if admm_config is None else None    # a caller's rho stays as given
        for pair in pairs:
            if pair.saturating_actuator or pair.static_hold:
                self._strong.add((pair.source, pair.destination))

    def _step_coupled(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        del state_out
        if self._rho_dt is not None:
            rho = self._rho_dt * 4.0 ** getattr(self, "_rho_raises", 0) / float(dt)
            if rho != float(self._coupling.rho):
                self._coupling = dataclasses.replace(self._coupling, rho=rho)
        self._admm.run(state_in, control, contacts, dt, self._watchdog.interface_work)

    def _escalate_mode(self) -> bool:
        """Second escalation rung: ρ ×4 (coupling-time §4.4.5). With a caller's ``admm_config`` (a fixed ρ): at most
        three times. With ρ per dt_x (``admm_rho_dt``): while ρ dt_x stays at most 1 (an iteration removes no more than
        a contact's approach), so the default 0.25 raises once, to 1."""
        raises = getattr(self, "_rho_raises", 0)
        if raises >= 3 or (self._rho_dt is not None and self._rho_dt * 4.0 ** (raises + 1) > 1.0):
            return False
        self._rho_raises = raises + 1
        self._coupling = dataclasses.replace(self._coupling, rho=float(self._coupling.rho) * 4.0)
        log.warning("coupling escalation: ADMM rho -> %.4g", float(self._coupling.rho))
        return True

    def _loop_arrays(self) -> Mapping[str, Any]:
        lp = self._admm
        return {"res_vel": lp.primal, "res_pos": lp.res_pos, "res_force": lp.dual, "iters_used": lp.iters_used}

    def _coupler_state_arrays(self) -> Mapping[str, wp.array]:
        return self._admm.state_arrays()

    @property
    def iteration_loop(self) -> AdmmIterationLoop:
        return self._admm
