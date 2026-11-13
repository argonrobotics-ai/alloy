# SPDX-License-Identifier: Apache-2.0
"""``AlloyRigidSolver``: Alloy's rigid route stepping the model Isaac Lab built.

It is Alloy's bundled ``SolverMuJoCo`` (Newton 1.8 + MuJoCo Warp 3.17, with the run-time guards Alloy installs on
them) built on :func:`~alloy.isaaclab.bridge.bridge_model` of Isaac Lab's model, so Isaac Lab's own ``State``,
``Control`` and ``Contacts`` objects pass straight through: the arrays are shared, nothing is copied per step.

By default Alloy takes the task's own step (``AlloySolverCfg.rigid_step="task"``). ``rigid_step="alloy"`` opts into
Alloy's robot step rule: one physics step of the task runs as ``n`` equal Alloy steps of at most 1/600 s (a 5 ms
step runs as 4 x 1.25 ms, as ``alloy.Scene`` robots do).
"""

from __future__ import annotations

import inspect
import logging
import math
import sys
import weakref
from dataclasses import dataclass
from typing import Any

import alloy.precision  # noqa: F401  (Alloy's MuJoCo Warp guards, as alloy.build has them: mjw_chol, mjw_lazy_factor)
from alloy._vendor.newton.solvers import SolverMuJoCo

from .bridge import bridge_model, joint_type_ok, resync_arrays

log = logging.getLogger("alloy.isaaclab")

ALLOY_ROBOT_STEP = 1.0 / 600.0  # alloy.core.timebase: robots step at the largest control_dt / 2**j at or below this
_CFG_ONLY = {"class_type", "solver_type", "ls_parallel", "rigid_step", "health", "health_report_every",
            "health_speed_limit", "noslip_iterations", "noslip_tolerance"}


def solver_kwargs(solver_cfg: Any) -> dict[str, Any]:
    """The cfg fields that ``SolverMuJoCo.__init__`` takes (the rest is Isaac Lab or Alloy metadata)."""
    valid = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"}
    kwargs = {k: getattr(solver_cfg, k) for k in valid if hasattr(solver_cfg, k) and k not in _CFG_ONLY}
    if kwargs.get("update_data_interval") == 1:   # Newton 1.8 deprecates the option; 1 (every step) is its behaviour
        kwargs.pop("update_data_interval")
    return kwargs


def substeps_for(rigid_step: float | str | None, dt: float) -> int:
    """Alloy steps per task step: ``"alloy"`` -> the power of two that brings ``dt`` to 1/600 s or below; ``"task"`` /
    ``None`` -> 1; a number (seconds) -> enough equal steps to stay at or below it."""
    if rigid_step in (None, "task"):
        return 1
    limit = ALLOY_ROBOT_STEP if rigid_step == "alloy" else float(rigid_step)
    if not limit > 0.0:
        raise ValueError(f"AlloySolverCfg.rigid_step must be 'alloy', 'task' or a positive number of seconds, "
                         f"got {rigid_step!r}")
    if rigid_step == "alloy":
        n = 1
        while dt / n > limit * (1.0 + 1e-9):
            n *= 2
        return n
    return max(1, math.ceil(dt / limit - 1e-9))


def step_note(rigid_step: float | str | None) -> str:
    """How Alloy steps one Isaac Lab physics step, for the log line."""
    if rigid_step in (None, "task"):
        return "step: the task's own (rigid_step='task')"
    if rigid_step == "alloy":
        return "step: at most 1/600 s inside each physics step (rigid_step='alloy')"
    return f"step: at most {float(rigid_step):g} s inside each physics step (rigid_step={rigid_step!r})"


def physics_line(solver: AlloyRigidSolver) -> str:
    """One line in Isaac Lab's [INFO] style, so a run's log says which physics stepped it."""
    import alloy
    from alloy._vendor import mujoco_warp, newton

    rep = solver.bridge_report
    return (f"[INFO] [alloy] physics: Alloy {alloy.__version__} rigid route (Newton {newton.__version__}, MuJoCo Warp "
            f"{mujoco_warp.__version__}) on {solver.model.world_count} worlds; {step_note(solver.rigid_step)}; "
            f"noslip {solver.noslip}; model bridged: {rep.shared} arrays shared, {len(rep.defaulted)} defaults, "
            f"{len(rep.dropped)} not carried ({', '.join(rep.dropped) or 'none'})")


_LATEST: weakref.ref | None = None   # Isaac Lab runs one simulation per process


def active_solver() -> AlloyRigidSolver | None:
    """The Alloy solver of the running Isaac Lab simulation (latest built), or ``None``; for its ``health``."""
    return None if _LATEST is None else _LATEST()


class AlloyRigidSolver(SolverMuJoCo):
    """Alloy's ``SolverMuJoCo`` on Isaac Lab's model; ``step`` runs ``substeps_for(cfg.rigid_step, dt)`` steps."""

    def __init__(self, host_model: Any, solver_cfg: Any, deterministic: Any = None):
        joint_type_ok(host_model)
        self.host_model = host_model
        bridged, self.bridge_report = bridge_model(host_model)
        self.rigid_step = getattr(solver_cfg, "rigid_step", "task")
        self._substeps: dict[float, int] = {}
        kwargs = solver_kwargs(solver_cfg)
        if deterministic is not None:   # Isaac Lab's request (NewtonCfg.deterministic / deterministic_mode), as is
            kwargs["deterministic"] = deterministic
        super().__init__(bridged, **kwargs)
        self.noslip = "off"
        if int(getattr(solver_cfg, "noslip_iterations", 0) or 0) > 0:   # before any step or CUDA graph capture
            from alloy.scene.noslip_route import install_entry_noslip

            n = int(solver_cfg.noslip_iterations)
            self.noslip = install_entry_noslip(self, (n, float(getattr(solver_cfg, "noslip_tolerance", 1e-6))))
            if self.noslip == "off":   # never step without the noslip the task asked for
                raise NotImplementedError(
                    f"AlloySolverCfg.noslip_iterations={n} cannot run with integrator="
                    f"{getattr(solver_cfg, 'integrator', '?')!r}: MuJoCo Warp has no noslip for the discrete "
                    "integrator. Set noslip_iterations=0 or another integrator (the task presets use 'implicitfast')")
        self.health = None
        if getattr(solver_cfg, "health", "report") != "off":
            from .health import IsaacLabHealth

            self.health = IsaacLabHealth(self, getattr(solver_cfg, "health_speed_limit", None))
        global _LATEST
        _LATEST = weakref.ref(self)

    def substeps(self, dt: float) -> int:
        n = self._substeps.get(dt)
        if n is None:
            n = self._substeps[dt] = substeps_for(self.rigid_step, dt)
            log.info("Alloy steps a %.4g s Isaac Lab physics step as %d x %.4g s", dt, n, dt / n)
        return n

    @dataclass(eq=False)
    class Observables(SolverMuJoCo.Observables):
        """SolverMuJoCo's observables, indexed like Isaac Lab's own model: its contact sensor checks that the
        observables belong to the model it was built on, and the bridged model shares that model's arrays."""

        @property
        def model(self) -> Any:
            solver = self._solver
            return None if solver is None else getattr(solver, "host_model", solver.model)

    def step(self, state_in, state_out, control, contacts, dt: float, *, observables: Any = None) -> None:
        n = self.substeps(dt)
        h = dt / n
        for i in range(n):
            extra = {"observables": observables} if observables is not None and i == n - 1 else {}
            super().step(state_in if i == 0 else state_out, state_out, control, contacts, h, **extra)
        if self.health is not None:
            self.health.update(state_in, state_out, contacts, dt)

    # Isaac Lab's contact buffer is its own Newton's Contacts, the same layout as the vendored class; the checks
    # below accept it where the vendored base would reject it (observables) or skip it (body-pair reduction)
    def _validate_observable_contacts(self, contacts: Any) -> None:
        if not _is_host_contacts(contacts):
            return super()._validate_observable_contacts(contacts)
        if contacts.device != self.model.device:
            raise ValueError("Solver observables and Contacts must be on the solver device.")

    def _require_unreduced_contacts(self, contacts: Any, *, supports_body_pair_reduced_contacts: bool | None = None,
                                    configuration: str | None = None) -> None:
        supported = (type(self).supports_body_pair_reduced_contacts if supports_body_pair_reduced_contacts is None
                     else supports_body_pair_reduced_contacts)
        if _is_host_contacts(contacts) and not supported and getattr(contacts, "rigid_contact_body_pair_reduced", False):
            raise ValueError(f"{type(self).__name__} is not validated for body-pair-reduced contacts; build Isaac "
                             "Lab's collision pipeline without body-pair contact reduction")
        super()._require_unreduced_contacts(contacts, supports_body_pair_reduced_contacts=supports_body_pair_reduced_contacts,
                                            configuration=configuration)

    def notify_model_changed(self, flags: int) -> None:
        resync_arrays(self.host_model, self.model)
        super().notify_model_changed(flags)


def _is_host_contacts(contacts: Any) -> bool:
    """True for Isaac Lab's own Newton ``Contacts`` (the ``newton`` package, not Alloy's vendored copy)."""
    host = sys.modules.get("newton")
    host_cls = getattr(host, "Contacts", None) if host is not None else None
    return host_cls is not None and isinstance(contacts, host_cls)
