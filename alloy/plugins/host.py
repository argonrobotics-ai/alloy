# SPDX-License-Identifier: Apache-2.0
"""Runs force and sensor plug-ins inside a simulation's step.

``PluginHost`` wraps the stepper's ``step`` (on the instance, so ``isinstance`` checks on the stepper still hold):

    before:  zero the force buffers -> each ForcePlugin.apply -> save state_in.body_f/particle_f -> add the buffers
    stepper.step(state_in, state_out, ...)
    after:   restore body_f/particle_f in state_in and state_out -> ForcePlugin.update / SensorPlugin.measure ->
             time += dt

Plug-in forces therefore never accumulate in ``State`` and other writers of ``body_f`` keep their values. The
stepper's ``state_arrays`` also lists every plug-in state array (``plugin/<label>/<name>``), so the simulation's
snapshots, ``reset``, rollbacks and digests cover plug-in state. Install the host before the simulation is built:
``simulation(built, plugins)`` does that for ``alloy.sim.Simulation``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import Forces, ForcePlugin, Plugin, SensorPlugin
from alloy.plugins.context import PluginContext
from alloy.plugins.errors import PluginError
from alloy.precision.health_work import ForceWork

log = logging.getLogger("alloy.plugins")

PREFIX = "plugin"
HOST_WORK = ("synchronize", "synchronize_device", "synchronize_stream", "zeros", "zeros_like", "empty", "empty_like",
             "full", "full_like", "ones", "ones_like", "clone")


@contextlib.contextmanager
def no_host_work(label: str, hook: str) -> Iterator[None]:
    """While a hook runs: ``wp.array.numpy()``, ``wp.synchronize*`` and allocations raise ``PluginError`` (a step may
    be a captured graph: hooks launch kernels only). The harness turns this on (``PluginHost.strict``)."""
    def blocked(name: str):
        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise PluginError(f"{label}.{hook}() called {name} during the step; steps may replay as a captured graph, "
                              f"so hooks only launch kernels: allocate in setup(), read results in read()")
        return fail

    saved = {name: getattr(wp, name) for name in HOST_WORK if hasattr(wp, name)}
    numpy = wp.array.numpy
    try:
        for name in saved:
            setattr(wp, name, blocked(f"wp.{name}()"))
        wp.array.numpy = blocked("wp.array.numpy()")
        yield
    finally:
        for name, fn in saved.items():
            setattr(wp, name, fn)
        wp.array.numpy = numpy


@wp.kernel
def _add_wrench(dst: wp.array[wp.spatial_vector], src: wp.array[wp.spatial_vector]):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


@wp.kernel
def _add_vec3(dst: wp.array[wp.vec3], src: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


@wp.kernel
def _advance_time(t: wp.array[wp.float64], dt: wp.float64):
    w = wp.tid()
    t[w] = t[w] + dt


class PluginHost:
    """Owns the plug-ins of one simulation: sets them up on ``model`` and runs their hooks around each step.

    Example::

        host = PluginHost(built.model, [Magnet(bodies=("a", "b"), moments=(0.1, 0.1))], control_dt=built.control_dt)
        sim = Simulation(built, stepper=host.install(default_stepper(built, "mujoco")))
        sim.step_period(); host.read()["magnet"]
    """

    def __init__(self, model: newton.Model, plugins: Sequence[Plugin], *, control_dt: float) -> None:
        self.model = model
        self.plugins: dict[str, Plugin] = {}
        for p in plugins:
            if not isinstance(p, (ForcePlugin, SensorPlugin)):
                raise PluginError(f"{p!r} is a {p.kind or type(p).__name__} plug-in; only force and sensor plug-ins "
                                  f"run in the step (build objects into the scene, register solvers)")
            if p.label in self.plugins:
                raise PluginError(f"two plug-ins labelled {p.label!r}: pass label='...' to tell them apart")
            self.plugins[p.label] = p
        self.ctx = PluginContext(model, float(control_dt))
        self.forces = Forces(wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device),
                             wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device))
        self._saved = Forces(wp.zeros_like(self.forces.body_f), wp.zeros_like(self.forces.particle_f))
        self._forcing = [p for p in self.plugins.values() if isinstance(p, ForcePlugin) and p.applies_forces]
        self.work = ForceWork(model) if self._forcing else None    # F4: health books the forces' work as W_ext
        self._checked_dt: set[float] = set()
        self.applied_q: wp.array | None = None       # harness: body_q the last apply() read (record_applied)
        self.strict = False                           # harness: hooks may not sync, read back or allocate
        self._exported: Any = None                    # contacts the solver exports for plug-ins (needs_contacts)
        self._inertia: list[Any] = []                 # pushes of body mass properties a plug-in set (changes_inertia)
        for p in self.plugins.values():
            self._setup(p)

    def _setup(self, p: Plugin) -> None:
        if p.ctx is not None:
            raise PluginError(f"plug-in {p.label!r} is already set up on a model: create a new instance per simulation")
        p.ctx = self.ctx
        p.setup(self.ctx)
        log.info("plug-in %s (%s) set up: state %s, inputs %s", p.label, type(p).__name__, sorted(p.state_arrays()),
                 sorted(p.inputs()))

    def __getitem__(self, label: str) -> Plugin:
        try:
            return self.plugins[label]
        except KeyError:
            raise PluginError(f"no plug-in labelled {label!r}; labels: {sorted(self.plugins)}") from None

    # --- the step ----------------------------------------------------------------------------------------------

    def before(self, state: newton.State, contacts: Any, dt: float) -> None:
        """Before the solver step: plug-in forces into ``state`` (saved first)."""
        if self._exported is None:
            self.ctx.contacts = contacts
        if dt not in self._checked_dt:
            for p in self.plugins.values():
                p.check_dt(dt)
            self._checked_dt.add(dt)
        if not self._forcing:
            return
        self.forces.body_f.zero_()
        self.forces.particle_f.zero_()
        for p in self._forcing:
            with self._guard(p, "apply"):
                p.apply(state, self.forces, dt)
        for push in self._inertia:                        # body masses a plug-in set in apply, into the solver
            push()
        if self.applied_q is not None and state.body_q is not None:
            wp.copy(self.applied_q, state.body_q)
        if self.work is not None:
            self.work.begin(state)
        self._push(state, "body_f", self._saved.body_f, self.forces.body_f, _add_wrench)
        self._push(state, "particle_f", self._saved.particle_f, self.forces.particle_f, _add_vec3)

    def after(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """After the solver step: restore the forces in ``state_in`` and ``state_out``, then plug-in updates and
        sensors. Newton's ``SolverCoupled.step`` ends with ``_copy_state(state_in, state_out)``
        (newton/_src/solvers/coupled/solver_coupled.py:3024 at pin 4f3b41ef), which copies ``body_f`` /
        ``particle_f`` with the plug-in forces in them; restoring only ``state_in`` let a constant plug-in force grow by
        one copy per exchange in every coupled scene. For uncoupled steppers ``state_out`` now also carries the forces
        the user set: a force written once on ``sim.state`` acts every exchange (before: every other exchange, as the
        two states ping-ponged)."""
        if self.work is not None:
            self.work.end(self.forces.body_f, self.forces.particle_f, state_out, dt)
        self.restore_forces(state_in)
        if state_out is not state_in:
            self.restore_forces(state_out)
        for p in self.plugins.values():
            if isinstance(p, ForcePlugin):
                with self._guard(p, "update"):
                    p.update(state_in, state_out, dt)
            else:
                with self._guard(p, "measure"):
                    p.measure(state_in, state_out, dt)
        wp.launch(_advance_time, dim=self.ctx.world_count, inputs=[self.ctx.time, wp.float64(dt)],
                  device=self.model.device)

    def _guard(self, p: Plugin, hook: str) -> contextlib.AbstractContextManager:
        return no_host_work(p.label, hook) if self.strict else contextlib.nullcontext()

    def restore_forces(self, state: newton.State) -> None:
        """Put back the ``body_f`` / ``particle_f`` that ``before`` saved (plug-in forces never stay in State)."""
        if self._forcing:
            self._pop(state, "body_f", self._saved.body_f)
            self._pop(state, "particle_f", self._saved.particle_f)

    @staticmethod
    def _push(state: newton.State, attr: str, saved: wp.array, add: wp.array, kernel: Any) -> None:
        target = getattr(state, attr, None)
        if target is None or add.shape[0] == 0:
            return
        wp.copy(saved, target)
        wp.launch(kernel, dim=add.shape[0], inputs=[target, add], device=target.device)

    @staticmethod
    def _pop(state: newton.State, attr: str, saved: wp.array) -> None:
        target = getattr(state, attr, None)
        if target is not None and saved.shape[0] > 0:
            wp.copy(target, saved)

    def record_applied(self) -> None:
        """Keep a copy of the ``body_q`` each ``apply`` read (the harness's action-reaction check uses it)."""
        if self.applied_q is None and self.model.body_count > 0:
            self.applied_q = wp.zeros(self.model.body_count, dtype=wp.transform, device=self.model.device)

    # --- state ---------------------------------------------------------------------------------------------------

    def state_arrays(self) -> dict[str, wp.array]:
        """Every plug-in state array as ``plugin/<label>/<name>``, plus the per-world clock ``plugin/time``."""
        out = {f"{PREFIX}/time": self.ctx.time}
        for label, p in self.plugins.items():
            out.update({f"{PREFIX}/{label}/{k}": v for k, v in p.state_arrays().items()})
        return out

    def read(self) -> dict[str, dict[str, np.ndarray]]:
        """``{label: plugin.read()}``: host copies, ``[W, ...]`` per array (syncs the device)."""
        return {label: p.read() for label, p in self.plugins.items()}

    def install(self, stepper: Any) -> Any:
        """Run this host's hooks in ``stepper.step`` and list plug-in state in ``stepper.state_arrays``. Call before
        the simulation is built (it snapshots ``state_arrays`` at construction). Returns ``stepper``."""
        if getattr(stepper, "_alloy_plugin_host", None) is not None:
            raise PluginError("this stepper already runs a PluginHost: put every plug-in in one host")
        inner_step, inner_arrays = stepper.step, stepper.state_arrays
        host = self

        def step(state_in, state_out, control, contacts, dt, *args, **kwargs):
            host.before(state_in, contacts, dt)
            inner_step(state_in, state_out, control, contacts, dt, *args, **kwargs)
            host.after(state_in, state_out, dt)

        def state_arrays() -> Mapping[str, wp.array]:
            return {**dict(inner_arrays()), **host.state_arrays()}

        stepper.step, stepper.state_arrays = step, state_arrays
        stepper._alloy_plugin_host = self
        self._export_contacts(stepper)
        self._inertia = self._inertia_pushes(stepper)
        for p in self.plugins.values():                   # plug-ins that extend a solver (e.g. bonds in a film)
            attach = getattr(p, "attach", None)
            if callable(attach):
                attach(stepper)
        log.info("plug-in host installed on %s: %s", type(stepper).__name__, sorted(self.plugins))
        return stepper

    def _inertia_pushes(self, stepper: Any) -> list[Any]:
        """With a plug-in that sets body masses in ``apply`` (``changes_inertia``), each solver's push of them after
        ``apply`` (``inertia_pushes``)."""
        needing = sorted(p.label for p in self.plugins.values() if getattr(p, "changes_inertia", False))
        if not needing:
            return []
        solvers = list(stepper.solvers()) if hasattr(stepper, "solvers") else [getattr(stepper, "solver", stepper)]
        pushes = inertia_pushes(solvers, self.ctx.world_count, needing)
        log.info("plug-ins %s set body masses: pushed into %d solver(s) after apply", needing, len(pushes))
        return pushes

    def _export_contacts(self, stepper: Any) -> None:
        """Plug-ins with ``needs_contacts`` read the contacts and forces of a solver that runs its own collision
        (MuJoCo Warp): its step gets Newton contact storage and a CONTACT_F observables container, and the context
        points at both (``ctx.contacts``, ``ctx.contact_force``). About +1-3 % per step at 64-2,048 worlds (L4)."""
        needing = sorted(p.label for p in self.plugins.values() if getattr(p, "needs_contacts", False))
        if not needing:
            return
        solver = getattr(stepper, "solver", None)
        kind = newton.solvers.SolverBase.ObservableKind.CONTACT_F
        if solver is None or not hasattr(solver, "get_max_contact_count") or kind not in solver.supported_observables:
            raise PluginError(f"{needing} need contact forces, but {type(solver).__name__} does not export its own "
                              f"contacts with forces: use the MuJoCo route on MuJoCo Warp (not use_mujoco_cpu)")
        if self.ctx.world_count > 1 and _steps_world_zero_only(solver):
            raise PluginError(f"{needing} need contact forces in every world, but Newton's own MuJoCo C step "
                              f"(use_mujoco_cpu) simulates world 0 only of {self.ctx.world_count}, so the others "
                              "would read nothing: use Alloy's CPU route (alloy:cpu_route='mjc', the CPU default), "
                              "MuJoCo Warp, or one world")
        contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=self.model.device)
        observables = solver.observables({kind}, contacts=contacts)
        inner = solver.step

        def step(state_in, state_out, control, step_contacts, dt, **kwargs):
            if step_contacts is None and "observables" not in kwargs:
                return inner(state_in, state_out, control, contacts, dt, observables=observables, **kwargs)
            return inner(state_in, state_out, control, step_contacts, dt, **kwargs)

        solver.step = step
        self._exported = contacts
        self.ctx.contacts, self.ctx.contact_force = contacts, observables.contact_f


MASS_FIELDS = ("body_mass", "body_ipos", "body_inertia", "body_iquat")


def _steps_world_zero_only(solver: Any) -> bool:
    """Newton 1.8's own MuJoCo C step (``use_mujoco_cpu``, separate worlds) simulates the template world only (its
    docstring; worlds 1+ stay still, with no error). Alloy's CPU route (``MjcCoreSolver``, the same flag) steps one
    MjData per world. Before the bump that path exported no CONTACT_F, so contact plug-ins were refused there."""
    from alloy.route.mjc_core import MjcCoreSolver  # noqa: PLC0415 - keeps `import alloy.plugins` light

    return bool(getattr(solver, "use_mujoco_cpu", False)) and not isinstance(solver, MjcCoreSolver)


def inertia_pushes(solvers: Sequence[Any], world_count: int, needing: Sequence[str]) -> list[Any]:
    """How each solver takes the body masses, centres and inertias a plug-in set in ``model`` during ``apply``
    (``changes_inertia``), as callables run after ``apply``; fails closed:
    - MuJoCo Warp: Newton's per-world update of its mass, centre and inertia (``_update_model_inertial_properties``, a
      PRIVATE_HOOKS entry: device kernels, graph-safe). set_const is skipped on purpose (as GA writes its masses
      without mj_setConst): invweight0 and subtreemass keep their build values, so contact softness follows the build
      masses;
    - Alloy's MuJoCo C route (one MjModel per world, ``alloy.route.mjc_core``): that update, then each world's model
      with every body on its own inertial frame (``body_sameframe`` 0: MuJoCo C ignores ipos and iquat where the
      compiler found them at the body frame) and its midphase boxes moved into it (``refit_bvh``); Newton's own MuJoCo
      C path (one model): one world only. The route's
      light-body rows (``LightRows``) keep the build masses, as invweight0 does;
    - Featherstone: its public ``notify_model_changed(BODY_INERTIAL_PROPERTIES)`` (it caches the masses);
    - XPBD and semi-implicit read the model's arrays every step: nothing to push;
    - any other solver, and coupled steppers (their entries may hold copies of the body arrays): refused."""
    import alloy._vendor.newton.solvers as ns  # noqa: PLC0415

    from alloy.route.mjc_core import MjcCoreSolver  # noqa: PLC0415 - keeps `import alloy.plugins` light

    pushes: list[Any] = []
    for s in solvers:
        if getattr(s, "_entries", None) is not None:
            raise PluginError(f"{list(needing)} set body masses at run time, which a coupled stepper's entries do not "
                              f"take ({type(s).__name__}): carry the water as weight (Pool(carry='weight'))")
        if isinstance(s, ns.SolverMuJoCo):
            hook = getattr(s, "_update_model_inertial_properties", None)
            if hook is None:
                raise PluginError(f"{list(needing)}: this Newton's SolverMuJoCo has no _update_model_inertial_properties "
                                  f"(a PRIVATE_HOOKS entry): the masses would not reach it")
            if not getattr(s, "use_mujoco_cpu", False):
                pushes.append(hook)
            elif isinstance(s, MjcCoreSolver):
                pushes.append(lambda s=s: _push_mjc_worlds(s))
            elif world_count == 1:
                pushes.append(lambda s=s: _push_mjc(s))
            else:
                raise PluginError(f"{list(needing)} set body masses per world, but Newton's MuJoCo C path steps one "
                                  f"model for every world: use one world, Alloy's MuJoCo C route or MuJoCo Warp")
        elif isinstance(s, ns.SolverFeatherstone):
            pushes.append(lambda s=s: s.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))
        elif not isinstance(s, (ns.SolverXPBD, ns.SolverSemiImplicit)):
            raise PluginError(f"{list(needing)} set body masses at run time, which {type(s).__name__} is not known to "
                              f"take (MuJoCo, Featherstone, XPBD, semi-implicit): carry the water as weight")
    return pushes


def _push_mjc_worlds(solver: Any) -> None:
    """Alloy's MuJoCo C route: Newton's per-world update, then each world's MjModel from its row and its midphase
    boxes."""
    solver._update_model_inertial_properties()
    rows = [getattr(solver.mjw_model, f).numpy() for f in MASS_FIELDS]
    for w, m in enumerate(solver.worlds):
        for f, row in zip(MASS_FIELDS, rows):
            getattr(m, f)[:] = row[w]
        _own_frames(solver, m)


def _push_mjc(solver: Any) -> None:
    """Newton's MuJoCo C path (one model, one world): its model takes world 0's row and its midphase boxes."""
    solver._update_model_inertial_properties()
    for f in MASS_FIELDS:
        getattr(solver.mj_model, f)[:] = getattr(solver.mjw_model, f).numpy()[0]
    _own_frames(solver, solver.mj_model)


def _own_frames(solver: Any, m: Any) -> None:
    """MuJoCo C reads body_ipos and body_iquat only where the compiled ``body_sameframe`` lets it (1: neither, 3: not
    the rotation; WP0's probe): every body takes its own inertial frame (0), then its midphase boxes follow it."""
    m.body_sameframe[1:] = 0
    refit_bvh(solver, m)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows of (w, x, y, z) products a·b."""
    w = a[:, :1] * b[:, :1] - np.sum(a[:, 1:] * b[:, 1:], axis=1, keepdims=True)
    return np.hstack((w, a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] + np.cross(a[:, 1:], b[:, 1:])))


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows of v rotated by unit (w, x, y, z) q."""
    u, w = q[:, 1:], q[:, :1]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def refit_bvh(solver: Any, m: Any) -> None:
    """MuJoCo C keeps each body's midphase boxes in its inertial frame (``mj_collideTree``): re-express the compiled
    boxes (``_canonical_bvh_aabb``, ``_canonical_body_ipos``, ``_canonical_body_iquat``: PRIVATE_HOOKS) in the current
    one, as Newton's ``_refit_body_bvh_to_inertial_frame`` (WP0: a 5 cm shift lost every contact). Always from the
    compiled boxes: rotated boxes grow into their enclosing box."""
    box = getattr(solver, "_canonical_bvh_aabb", None)
    bodies = np.flatnonzero(m.body_bvhnum)
    if box is None or not bodies.size:
        return
    inv = m.body_iquat[bodies] * np.array([1.0, -1.0, -1.0, -1.0])
    dq = _quat_mul(inv, solver._canonical_body_iquat[bodies])
    rot = np.stack([_quat_rot(dq, np.tile(e, (bodies.size, 1))) for e in np.eye(3)], axis=2)     # columns R·e_k
    off = _quat_rot(inv, solver._canonical_body_ipos[bodies] - m.body_ipos[bodies])
    num, adr = m.body_bvhnum[bodies], m.body_bvhadr[bodies]
    nodes = np.concatenate([np.arange(a, a + n) for a, n in zip(adr, num)])
    which = np.repeat(np.arange(bodies.size), num)
    m.bvh_aabb[nodes, :3] = np.einsum("nij,nj->ni", rot[which], box[nodes, :3]) + off[which]
    m.bvh_aabb[nodes, 3:] = np.einsum("nij,nj->ni", np.abs(rot[which]), box[nodes, 3:])


def simulation(built: Any, plugins: Sequence[Plugin], *, cls: type | None = None, **sim_kwargs: Any) -> Any:
    """A simulation of ``built`` running ``plugins`` (``cls``: ``alloy.sim.Simulation`` by default, or e.g.
    ``alloy.integration.simulation.IntegratedSimulation`` with ``use_graph=True``); the host is ``sim.plugins``.

    Example::

        sim = simulation(built, [Magnet(bodies=("a", "b"), moments=(0.1, 0.1))])
        for _ in range(50):
            sim.step_period()
        print(sim.plugins.read()["magnet"])
    """
    from alloy.scene.steppers import default_stepper  # noqa: PLC0415 - keeps `import alloy.plugins` light
    from alloy.sim import Simulation  # noqa: PLC0415

    if sim_kwargs.pop("stepper", None) is not None:
        raise PluginError("simulation(built, plugins) builds the stepper; to use your own, call "
                          "PluginHost(...).install(stepper) and pass it to Simulation yourself")
    host = PluginHost(built.model, plugins, control_dt=built.control_dt)
    dyn = [e for e in built.entries if e.name]
    native = Simulation._native(sim_kwargs.get("precision"), dyn, sim_kwargs.get("contact_layer"))
    sim = (cls or Simulation)(built, stepper=host.install(default_stepper(built, native)), **sim_kwargs)
    sim.plugins = host
    return sim


__all__ = ["MASS_FIELDS", "PluginHost", "inertia_pushes", "refit_bvh", "simulation"]
