# SPDX-License-Identifier: Apache-2.0
"""Alloy's CPU route: the articulated core in MuJoCo C (float64) with Alloy's plug-ins (SCR-WP8-6, ruled GO on
route (i) 2026-11-06 11:19Z).

``MjcCoreSolver`` is Newton's ``SolverMuJoCo(use_mujoco_cpu=True, separate_worlds=True)`` (Alloy's converted model,
every model-level plug-in the ``mujoco`` factory installs) stepped by the route instead of Newton's C path:
- **one MjModel and one MjData per world** (:mod:`alloy.route.mjc_core_models`): per-world gravity, masses and the
  rest come from Newton's per-world arrays;
- **numpy views** in place of Newton's per-step Warp push/pull (:mod:`alloy.route.mjc_core_views`): MuJoCo keeps its
  float64 state between steps, Newton's state is pushed only when someone changed it;
- **per-step plug-ins**: Alloy's scoped noslip (the scene's noslip request, scope "robot", or the static hold, scope
  "all") and the box-box line contacts (BUL-0014), in C through WP2's extension (``alloy.route._mjc``) when it loads,
  else in numpy (:mod:`alloy.route.mjc_core_tick`, 1.2 to 1.9x slower; one warning per reason, shown even with no
  log handler, names why and the fix: a missing C compiler is the usual one, F18);
- **contact export** into Newton ``Contacts`` (``update_contacts``) and **health** mirrored into the Warp data that
  Alloy's HealthMonitor reads (contacts, solver iterations, and each world's ``overflow`` word with MuJoCo Warp's bits
  for dropped contacts or constraint rows, so a capacity overflow fails closed). A world MuJoCo C flags (non-finite
  or huge qpos / qvel / qacc) is frozen (skipped) with NaN in its Newton state until reset, so it reads as a blow-up,
  as on MuJoCo Warp: the route turns MuJoCo C's auto-reset off.

**CPU-route solver settings differ from the GPU route** (:class:`alloy.route.mjc_core_models.RouteSettings`): warm
start off and ``opt.tolerance`` at most 1e-9. MuJoCo C's Newton solve otherwise stops short where MuJoCo Warp does not
(A5.onset creeps 1.66e-3 m per 10 s with warm start, A5.isotropy spreads 4.1 % at tolerance 1e-8; SPIKE.md §1). The
GPU route keeps the scene's options. CPU and GPU trajectories differ, as they did before.

**Not on this route** (they stay GPU / MuJoCo Warp): many-world graphs, WP5's ordered route (MuJoCo C is deterministic
as it is: one world, one thread, float64), coordinate carry (float64 qpos needs none: taken off at construction),
the pivot clamp, SDF / precision-pair collision and the Newton collision pipeline couplers. The factory refuses those
entries for this route and builds them as before (:mod:`alloy.route.mjc_core_factory`). rk4 entries step with plain
``mj_step`` (no mid-step plug-ins, as Alloy's rk4 has no noslip).

**No graph capture.** The route steps MuJoCo C on the host, which a Warp CPU graph (a record-only op stream) cannot
hold: ``GRAPH_CAPTURABLE`` is False, and a step made while the CPU records a graph raises ``GraphCaptureError`` naming
the warp CPU route (``alloy:cpu_route="warp"`` / ``ALLOY_CPU_ROUTE=warp``), which captures. Code that captures
control periods can ask first (``graph_capturable``).

**Speed.** ``step`` is one physics step (Newton's API, the slow path: DirectSim's ``pre_substep`` hooks and per-step
observers need it). ``step_many`` runs ``nstep`` steps with the control held, one C call per world: plain T12 ticks 32
steps per call (``DirectSim.step_period`` takes it when the period has no hooks or per-step observers, SCR-WP8-8).

**Newton API on the route.** ``update_contacts`` and the ``CONTACT_F`` observable (the plug-in host's sensors: tactile,
sound, thermal) come from the route's contact export; other observables raise. Newton's template ``mj_data`` mirrors
world 0 after every call (code that reads it on the MuJoCo C path, as ``alloy.tasks.dc_rack_pen`` does); every world's
contacts are in the Warp data's contact arrays (``mjw_data``).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import mujoco
import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.core.errors import GraphCaptureError, SolverOptionError
from alloy.route.mjc_core_cabi import EXTENSION, CWorld, load_extension
from alloy.route.mjc_core_config import RouteConfig
from alloy.route.mjc_core_consts import apply as apply_constants, publish, world_constants
from alloy.route.mjc_core_models import LightRows, body_maps, world_models
from alloy.route.mjc_core_noslip import ScopedNoslipNumpy
from alloy.route.mjc_core_line import LineRule
from alloy.route.mjc_core_tick import NumpyPlugins, WorldOut, arena_full, plain_tick, tick
from alloy.route.mjc_core_views import NewtonViews, overflow_bits
from alloy.route.mjc_core_window import STATE_BITS, WindowStart

log = logging.getLogger("alloy.route")

_WARNED: set[str] = set()


def fallback_fix(why: str, platform_name: str | None = None) -> str:
    """What to do about a C step that did not build or load (``why``: ``alloy.route._mjc.reason``)."""
    if "ALLOY_NO_EXT" in why:
        return "unset ALLOY_NO_EXT for the fast C step"
    if "compiler" in why or "compiling" in why:
        how = "xcode-select --install" if (platform_name or sys.platform) == "darwin" else "apt install gcc"
        return f"install a C compiler for the fast C step ({how}, or set CC)"
    return "reinstall alloy-engine to rebuild the C step for this MuJoCo"


def _warn_fallback(why: str) -> None:
    """One user-facing warning (printed to stderr when no log handler shows it): Alloy's package logger carries a
    NullHandler, so a plain log warning vanished and the fallback was silent (F18)."""
    from alloy.api._say import warn  # noqa: PLC0415 - the API's user-warning helper; imported when needed

    why = why.rstrip(". ")
    short = why if len(why) <= 240 else why[:240] + "..."
    warn("CPU route: the fast C step (%s) is unavailable: %s. Robot scenes run on the numpy step instead, 1.2 to "
         "1.9x slower (one arm to 16 arms; a pile of rigid bodies with no robot is not affected); %s.",
         EXTENSION, short, fallback_fix(why))
_PLAIN = (int(mujoco.mjtIntegrator.mjINT_RK4), int(mujoco.mjtIntegrator.mjINT_DISCRETE))


class MjcCoreSolver(newton.solvers.SolverMuJoCo):
    """The CPU route's solver (module docstring). Build it through the ``mujoco`` factory (``alloy:cpu_route`` /
    ``ALLOY_CPU_ROUTE``); ``route`` holds the route's plug-in choices."""

    GRAPH_CAPTURABLE = False
    NO_GRAPH_REASON = ("the MuJoCo C CPU route steps on the host, which a CPU graph cannot record; the warp CPU route "
                       "(alloy:cpu_route='warp' or ALLOY_CPU_ROUTE=warp) captures")

    def __init__(self, model: newton.Model, *, route: RouteConfig | None = None, **kw: Any) -> None:
        from alloy.scene.coordinate_carry import uninstall

        if str(model.device) != "cpu":
            raise SolverOptionError(f"MjcCoreSolver is the CPU route; the model is on {model.device}")
        kw.update(use_mujoco_cpu=True, separate_worlds=True)
        super().__init__(model, **kw)
        uninstall(self)              # float64 qpos needs no carry (the default-on install wraps every SolverMuJoCo)
        self.route = route or RouteConfig()
        self.nworld = int(model.world_count)
        self.worlds: list[mujoco.MjModel] = []
        self.datas: list[mujoco.MjData] = []
        self.outs: list[Any] = []
        self.backend, self.backend_reason = "", ""
        self._ready, self._dt = False, None
        self._pooled: dict[str, np.ndarray] = {}
        self._last_out: newton.State | None = None
        self._frozen = np.zeros(self.nworld, dtype=bool)
        self._consts: dict[int, dict[str, Any]] = {}
        self._stale = False
        self._refresh_constants()                    # before the pair refit reads the per-world weights

    def _refresh_constants(self) -> None:
        """Each mass-drawn world's own mass-derived constants (``mjc_core_consts``), published to the Warp rows."""
        had = bool(self._consts)
        self._consts = world_constants(self, self.nworld)
        if self._consts or had:
            publish(self, self._consts, self.nworld)
            log.info("CPU route: own mass-derived constants for %d of %d worlds", len(self._consts), self.nworld)

    def _make_worlds(self) -> list[mujoco.MjModel]:
        """Per-world C models: the template, each world's own fields and mass-derived constants, then the tree-weld
        rule's chainless weights for its bodies (``alloy.scene.tree_invweight``, float64 per world, as the template
        holds world 0's)."""
        worlds = world_models(self, self.nworld, self.route.settings, self.route.noslip is not None)
        tree, own = getattr(self, "_alloy_tree_sets", None), getattr(self, "_alloy_tree_invw", None)
        for w, m in enumerate(worlds):
            m.opt.disableflags = int(m.opt.disableflags) | int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
            apply_constants(m, self._consts.get(w))
            if tree is not None and own is not None and w < own[1].shape[0]:
                m.body_invweight0[tree.target] = own[1][w]
        return worlds

    def _rebuild_if_stale(self) -> None:
        """After a model change, the per-world models again before the next step or reset, once every rule wrapped
        around ``notify_model_changed`` (tree weld, pair refit) has run."""
        if not self._stale:
            return
        self._stale = False
        self.worlds = self._make_worlds()
        if self._light is not None:
            self._light.dt = None
        dt, self._dt = self._dt, None
        if dt is not None:
            self._set_dt(dt)

    # ---- build ------------------------------------------------------------------------------------------------------
    def finalize_route(self) -> MjcCoreSolver:
        """Per-world models, data, plug-ins and views; after the factory's model-level plug-ins, before the first
        step (``step`` calls it when needed)."""
        if self._ready:
            return self
        noslip = self.route.noslip
        self.worlds = self._make_worlds()
        self.datas = [mujoco.MjData(m) for m in self.worlds]
        self._light = LightRows.of(self)
        self._maps, globs = body_maps(self, self.nworld)
        self.views = NewtonViews(self, self.nworld, globs)
        self.window = WindowStart(self.views)
        self._plain = int(self.mj_model.opt.integrator) in _PLAIN
        self._scope = self.route.scope(self.mj_model) if noslip is not None else None
        self._make_outputs()
        self._ready = True
        log.info("CPU route (MuJoCo C core, %s plug-ins): %d worlds, noslip %s, line contacts %s, %s", self.backend,
                 self.nworld, "off" if noslip is None else f"{noslip.scope} {noslip.iterations}",
                 "on" if self.route.line else "off", self.route.settings)
        return self

    def _make_outputs(self) -> None:
        ncon = self.route.ncon_max
        lib, why = (None, "rk4 / discrete integrator: plain mj_step") if self._plain else load_extension()
        if self.route.backend == "numpy" and lib is not None:
            lib, why = None, "numpy plug-ins requested"
        if lib is None and not self._plain and why not in _WARNED and self.route.backend != "numpy":
            _WARNED.add(why)
            _warn_fallback(why)
        self.backend, self.backend_reason = ("c", "") if lib is not None else ("numpy", why)
        if self._plain:
            self.backend = "mj_step"
        cap = self.route.traj_cap
        if lib is not None:
            self.outs = [CWorld(lib, noslip=self.route.noslip, scope=self._scope, line=self.route.line,
                                body_map=bm, ncon_max=ncon, traj_cap=cap) for bm in self._maps]
            return
        self.outs = [WorldOut(ncon_max=ncon, nbody=int(bm.size), traj_cap=cap) for bm in self._maps]
        self._numpy = [self._plugins(m, bm) for m, bm in zip(self.worlds, self._maps)]

    def _plugins(self, m: mujoco.MjModel, body_map: np.ndarray) -> NumpyPlugins:
        spec = self.route.noslip
        noslip = None
        if spec is not None and not self._plain:
            noslip = ScopedNoslipNumpy(spec, self._scope, self.route.max_slots())
        line = LineRule(m) if self.route.line and not self._plain else None
        return NumpyPlugins(noslip, line, body_map, export_contacts=self.route.ncon_max > 0)

    def set_noslip(self, spec: Any) -> None:
        """Alloy's scoped noslip on the route after construction (``alloy.contact.noslip.NoslipSpec``; None = off): the
        route's counterpart of ``install_noslip``, which refuses the MuJoCo C path. The models' own noslip goes to 0
        while it is on. Rebuilds the plug-ins (counters restart)."""
        self.route = self.route.replace(noslip=spec)
        if not self._ready:
            return
        self._scope = self.route.scope(self.mj_model) if spec is not None else None
        for m in self.worlds:
            if spec is not None:
                m.opt.noslip_iterations = 0
        self._make_outputs()

    def set_trajectory(self, traj_cap: int) -> None:
        """Record every physics step's body_q / body_qd (``outs[w].traj_body_q`` / ``traj_body_qd``, rows 0 ..
        nstep - 1 of the last call); 0 turns it off. Rebuilds the output buffers (counters restart)."""
        self.route = self.route.replace(traj_cap=int(traj_cap))
        if self._ready:
            self._make_outputs()

    # ---- step -------------------------------------------------------------------------------------------------------
    def step(self, state_in: newton.State, state_out: newton.State, control: newton.Control | None,
             contacts: newton.Contacts | None, dt: float, observables: Any = None) -> None:
        """One physics step (Newton's API; ``contacts`` is not read: MuJoCo C runs its own collision). ``observables``:
        ``CONTACT_F`` only (the step's contacts and their world-frame forces on shape 0)."""
        kinds = set() if observables is None else {k for k in self.SUPPORTED_OBSERVABLES if observables.is_requested(k)}
        if kinds - {self.ObservableKind.CONTACT_F}:
            raise SolverOptionError(f"CPU route: observables {sorted(kinds)} are MuJoCo Warp features; the route "
                                    "exports CONTACT_F only (use the GPU route or alloy:cpu_route='warp')")
        self.step_many(state_in, state_out, control, dt, 1)
        if kinds:
            self.views.flag_overflow(self.views.to_contacts(self._pooled, observables.contacts, state_out,
                                                            force=observables.contact_f))

    @property
    def supported_observables(self) -> frozenset[str]:
        return frozenset({self.ObservableKind.CONTACT_F})

    def get_max_contact_count(self) -> int:
        """The pooled contact export capacity (all worlds)."""
        return int(self.mjw_data.naconmax)

    def step_many(self, state_in: newton.State, state_out: newton.State, control: newton.Control | None, dt: float,
                  nstep: int) -> None:
        """``nstep`` physics steps of ``dt`` with ``control`` and ``state_in``'s forces held, one C call per world;
        ``state_out`` gets the last step's state, the contact export the last step's contacts."""
        if wp.get_device("cpu").is_capturing:
            raise GraphCaptureError(f"CPU route step inside a graph capture: {self.NO_GRAPH_REASON}")
        self.finalize_route()
        self._rebuild_if_stale()
        self._set_dt(float(dt))
        changed = self.views.changed_worlds(state_in)
        self.views.push_state(self.datas, state_in, changed)
        self._frozen[changed] = False
        self.views.push_control(self.datas, state_in, control)
        bits = np.zeros(self.nworld, np.int32)
        for w in range(self.nworld):
            if not self._frozen[w]:
                bits[w] = self._tick(w, int(nstep))
        self.views.pull(self.datas, self.outs, state_out, state_in)
        self.views.poison(state_out, np.flatnonzero(self._frozen))
        self.views.set_overflow(bits)
        self._after_tick(state_out)

    def _tick(self, w: int, nstep: int) -> int:
        """One world's ``nstep`` steps; returns its overflow word for the call (what it dropped)."""
        m, d, out = self.worlds[w], self.datas[w], self.outs[w]
        before, full0 = out.counters(), arena_full(d)
        if self.backend == "c":
            out.tick(m, d, nstep)
        elif self._plain:
            plain_tick(m, d, self._numpy[w], nstep, out)
        else:
            tick(m, d, self._numpy[w], nstep, out)
        after, full1 = out.counters(), arena_full(d)
        if after["nonfinite"] != before["nonfinite"]:
            self._frozen[w] = True
            log.warning("CPU route: world %d went non-finite; frozen until reset", w)
        return overflow_bits(after["contact_overflow"] - before["contact_overflow"], full1[0] - full0[0],
                             full1[1] - full0[1])

    def _set_dt(self, dt: float) -> None:
        if self._dt == dt:
            return
        for m in self.worlds:
            m.opt.timestep = dt
        if self._light is not None:
            self._light.apply(self.worlds, dt)
        self.mj_model.opt.timestep = dt
        self._dt = dt

    def _after_tick(self, state_out: newton.State) -> None:
        self._last_out = state_out
        ncons = [o.ncon for o in self.outs]
        self._pooled = self.views.gather(self.outs, ncons) if self.route.ncon_max > 0 else {}
        if self.route.ncon_max > 0:
            self.views.flag_overflow(self.views.mirror_health(self._pooled, self.datas))
        mujoco.mj_copyData(self.mj_data, self.mj_model, self.datas[0])   # template mirrors world 0 (~20 us on T12)

    # ---- Newton API -------------------------------------------------------------------------------------------------
    def update_contacts(self, contacts: newton.Contacts, state: newton.State | None = None) -> None:
        """The last call's contacts (after its last step) as Newton ``Contacts``; body-local points at ``state``
        (default: the last output state)."""
        dropped = self.views.to_contacts(self._pooled, contacts, state if state is not None else self._last_out)
        self.views.flag_overflow(dropped)                   # contacts beyond rigid_contact_max: fail closed

    def reset(self, state: newton.State, world_mask: Any = None, flags: Any = None) -> None:
        """Newton's reset of ``state`` (model defaults in the selected worlds), then those worlds' MjData restart from
        it: data cleared, state pushed, counters zeroed, unfrozen."""
        super().reset(state, world_mask=world_mask, flags=flags)
        if not self._ready:
            return
        self._rebuild_if_stale()
        self.window.clear()
        mask = None if world_mask is None else np.asarray(
            world_mask.numpy() if hasattr(world_mask, "numpy") else world_mask, dtype=bool)[: self.nworld]
        worlds = np.arange(self.nworld) if mask is None else np.flatnonzero(mask)
        for w in worlds.tolist():
            mujoco.mj_resetData(self.worlds[w], self.datas[w])
            self.outs[w].zero()
        self.views.push_state(self.datas, state, worlds)
        self._frozen[worlds] = False
        if self.views.last_q is None:
            return
        if mask is None:
            self.views.remember(state)
        else:
            self._remember_worlds(state, worlds)

    def _remember_worlds(self, state: newton.State, worlds: np.ndarray) -> None:
        q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
        sel_q, sel_qd = np.isin(self.views.coord_world, worlds), np.isin(self.views.dof_world, worlds)
        self.views.last_q[sel_q] = q[sel_q]
        self.views.last_qd[sel_qd] = qd[sel_qd]

    def coupling_notify_input_state_update(self, state: newton.State, flags: Any, *, iteration_restart: bool = False,
                                           dt: float = 0.0) -> None:
        """The coupler's input updates: a window start's state distribution saves each world's float64 MuJoCo state,
        a restart pass restores it where the handed-in state is the window start (``mjc_core_window``)."""
        super().coupling_notify_input_state_update(state, flags, iteration_restart=iteration_restart, dt=dt)
        if not self._ready:
            return
        if iteration_restart:
            n = self.window.restore(self.worlds, self.datas, state).size
            log.debug("CPU route: restart pass, %d of %d worlds continue from the window's float64 state", n,
                      self.nworld)
        elif int(flags) & STATE_BITS:
            self.window.save(self.worlds, self.datas, state)

    def notify_model_changed(self, flags: Any) -> None:
        """Newton's update of its Warp and template models, each world's mass-derived constants, then the route's
        per-world models again before the next step (data kept)."""
        super().notify_model_changed(flags)
        if hasattr(self, "_consts"):                 # not during Newton's own __init__
            self._refresh_constants()
        if getattr(self, "_ready", False):
            self._stale = True                       # rebuilt before the next step (_rebuild_if_stale)

    def route_report(self) -> dict[str, Any]:
        """Backend, settings, plug-ins and the summed counters (gate rows, logs)."""
        tot: dict[str, float] = {}
        for o in self.outs:
            for k, v in o.counters().items():
                tot[k] = max(tot.get(k, 0.0), v) if k in ("pen_max", "qvel_max") else tot.get(k, 0) + v
        return {"route": "mjc_core", "backend": self.backend, "backend_reason": self.backend_reason,
                "worlds": self.nworld, "settings": vars(self.route.settings),
                "noslip": None if self.route.noslip is None else vars(self.route.noslip), "line": self.route.line,
                "frozen": int(self._frozen.sum()), "counters": tot}


def graph_capturable(solver: Any) -> tuple[bool, str]:
    """(can a control period holding ``solver`` be captured as a graph, why not): False with the reason for the
    MuJoCo C CPU route (``MjcCoreSolver.GRAPH_CAPTURABLE``); True for every other solver."""
    if getattr(solver, "GRAPH_CAPTURABLE", True):
        return True, ""
    return False, str(getattr(solver, "NO_GRAPH_REASON", f"{type(solver).__name__} cannot be captured"))


__all__ = ["MjcCoreSolver", "graph_capturable"]
