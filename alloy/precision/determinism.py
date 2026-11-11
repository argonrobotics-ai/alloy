# SPDX-License-Identifier: Apache-2.0
"""Effective determinism and precision kwargs (SPEC D9, §3.7; precision research §4.1, §4.4).

``effective_determinism`` reports what actually runs, not what was asked for. Per solver it reads
``SolverCapability.determinism`` (the single source of truth, D9) and then probes the live instance:

- Steel: the public ``deterministic`` option (newton@ac7f7830:newton/_src/solvers/steel/solver_steel.py:1510, 2093).
- SolverMuJoCo: Alloy's ordered route (``mjw_route``: contact/row sorts, Fixed64 and ordered sums) when
  ``enable_solver_determinism`` enabled it on a covered path; else ``_deterministic`` (Warp lowering,
  solvers/mujoco/solver_mujoco.py:4504), scoped around its step.
- XPBD / VBD / SemiImplicit: the Warp ``deterministic`` module option the instance recorded in
  ``SolverBase._module_options`` (solvers/solver.py:942-957; xpbd/solver_xpbd.py:196-202, vbd/solver_vbd.py:935-997);
  Newton re-applies an instance's options before it launches, so this is what runs for that instance.

Rules: on CPU every solver is RUN_TO_RUN (Warp CPU launches run serially). NATIVE needs its own ordered mode on.
WARP_LOWERING needs Warp's deterministic lowering on; SolverMuJoCo additionally needs <= 64 worlds and Warp <= 1.18
(research §3.3: OOM at 1024 worlds, no compile on Warp >= 1.19). MuJoCo Warp's own ``DeterminismType`` does not make
it repeatable at the pin (P3-1, ``mjw_det``): bits set by hand are reported as the reason, never as an upgrade.
Alloy's route (P3-1 round 5) is the upgrade: RUN_TO_RUN on any world count, graph-capturable, for covered paths
only; an uncovered path stays NONE with ``mjw_route.route_reason``. NONE stays NONE. GPU_TO_GPU is never effective in v1 (Steel refuses it, no cross-architecture evidence). The pack's mode is the
minimum over its solvers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import warp as wp

from alloy.core import registry
from alloy.core.protocols import DetCapability
from alloy.precision.mjw_det import downgrade_reason
from alloy.precision.types import Determinism, PrecisionConfig

log = logging.getLogger("alloy.precision")

MUJOCO_DET_MAX_WORLDS = 64
# Used only when a solver class has no registered SolverCapability (logged as a reason); research §4.1 table.
FALLBACK_CAPABILITY: dict[str, DetCapability] = {
    "SolverSteel": DetCapability.NATIVE,
    "SolverXPBD": DetCapability.WARP_LOWERING,
    "SolverVBD": DetCapability.WARP_LOWERING,
    "SolverSemiImplicit": DetCapability.WARP_LOWERING,
    "SolverMuJoCo": DetCapability.WARP_LOWERING,
    "CollisionPipeline": DetCapability.NATIVE,  # sorted contacts when deterministic=True (sim/collide.py:2260, 2971)
}


@dataclass(frozen=True)
class SolverDet:
    name: str
    capability: DetCapability
    mode: Determinism
    reason: str | None


def capability_of(solver: Any) -> tuple[DetCapability, str | None]:
    """(capability, note) from the registry, else the fallback table, else NONE."""
    finder = getattr(registry, "capability_for", None)
    cap = finder(solver) if finder is not None else None
    if cap is not None:
        return DetCapability(cap.determinism), None
    plugin = getattr(type(solver), "determinism", None)
    if isinstance(plugin, (int, DetCapability)) and not isinstance(plugin, bool):
        return DetCapability(plugin), None  # Alloy plug-in ClassVar (§3.1 AlloySolverPlugin)
    name = type(solver).__name__
    if name in FALLBACK_CAPABILITY:
        return FALLBACK_CAPABILITY[name], f"{name}: no registered SolverCapability, fallback table used"
    return DetCapability.NONE, f"{name}: no registered SolverCapability, assumed NONE"


def _as_det(mode: Any) -> Determinism:
    try:
        return Determinism(int(mode))
    except (TypeError, ValueError):
        return Determinism.NONE


def probe_mode(solver: Any) -> Determinism:
    """The deterministic mode the live solver instance runs (NONE when it exposes none)."""
    name = type(solver).__name__
    if name == "SolverSteel":
        return _as_det(getattr(solver, "deterministic", 0))
    if name == "SolverMuJoCo":
        return _as_det(getattr(solver, "_deterministic", 0))
    options = getattr(solver, "_module_options", None)  # SolverBase per-instance Warp module options
    if options:
        return min(_as_det(o.get("deterministic", 0)) for o in options.values())
    if hasattr(solver, "deterministic"):  # CollisionPipeline(deterministic=True), Alloy layers
        value = solver.deterministic
        return Determinism.RUN_TO_RUN if value is True else _as_det(value)
    return Determinism.NONE


def has_mode(solver: Any) -> bool:
    """True if the object carries a determinism switch the probe can read."""
    return (type(solver).__name__ in ("SolverSteel", "SolverMuJoCo") or bool(getattr(solver, "_module_options", None))
            or hasattr(solver, "deterministic"))


def _warp_version() -> tuple[int, ...]:
    return tuple(int(p) for p in wp.config.version.split(".")[:2] if p.isdigit())


def _mjw_route_det(solver: Any, cap: DetCapability) -> SolverDet | None:
    """SolverMuJoCo on Alloy's ordered route (RUN_TO_RUN), refused by it (NONE + reason), or None (never tried)."""
    from alloy.precision.mjw_route import route_enabled, route_reason

    if route_enabled(solver):
        return SolverDet("SolverMuJoCo", cap, Determinism.RUN_TO_RUN, None)
    why = route_reason(solver)
    if why is not None:
        return SolverDet("SolverMuJoCo", cap, Determinism.NONE, f"SolverMuJoCo: Alloy's ordered route not used: {why}")
    return None


def solver_det(solver: Any, world_count: int, device: wp.context.Device) -> SolverDet:
    """The mode one solver runs and why it is below RUN_TO_RUN (None if not)."""
    name = type(solver).__name__
    cap, note = capability_of(solver)
    if not device.is_cuda:
        return SolverDet(name, cap, Determinism.RUN_TO_RUN, note)
    if cap == DetCapability.NONE:
        return SolverDet(name, cap, Determinism.NONE, f"{name}: capability NONE")
    if name == "SolverMuJoCo" and probe_mode(solver) < Determinism.RUN_TO_RUN:
        routed = _mjw_route_det(solver, cap)
        if routed is not None:
            return routed
    mode = probe_mode(solver) if cap != DetCapability.NATIVE or has_mode(solver) else Determinism.RUN_TO_RUN
    if mode < Determinism.RUN_TO_RUN:
        mjw = downgrade_reason(solver) if name == "SolverMuJoCo" else None
        if mjw is not None:
            return SolverDet(name, cap, Determinism.NONE, mjw)
        kind = "its ordered mode" if cap == DetCapability.NATIVE else "Warp deterministic lowering"
        return SolverDet(name, cap, Determinism.NONE, f"{name}: {kind} is off")
    if name == "SolverMuJoCo" and world_count > MUJOCO_DET_MAX_WORLDS:
        return SolverDet(name, cap, Determinism.NONE,
                         f"SolverMuJoCo: deterministic lowering above {MUJOCO_DET_MAX_WORLDS} worlds is not supported")
    if name == "SolverMuJoCo" and _warp_version() >= (1, 19):
        return SolverDet(name, cap, Determinism.NONE, "SolverMuJoCo: deterministic lowering breaks on Warp >= 1.19")
    return SolverDet(name, cap, Determinism.RUN_TO_RUN, note)


def enable_solver_determinism(cfg: PrecisionConfig, solvers: Sequence[Any]) -> list[str]:
    """Turn on the determinism a solver needs a live instance for, before its first step or graph capture.

    Today: Alloy's ordered SolverMuJoCo route (``mjw_route.enable_route``) on CUDA when ``cfg`` asks for at least
    RUN_TO_RUN, the caller requests the route explicitly (``cfg.mjw_route`` or ``cfg.mjw_route_cg``; coordinator
    ruling 2026-11-03 23:15Z: a preset's RUN_TO_RUN alone does not) and the solver does not already run Warp's
    lowering. Returns why a SolverMuJoCo stays NONE."""
    if Determinism(cfg.determinism) < Determinism.RUN_TO_RUN:
        return []
    if not (getattr(cfg, "mjw_route", False) or getattr(cfg, "mjw_route_cg", False)):
        return []
    flat = list(unwrap_solvers(solvers))
    if not _device_of(flat).is_cuda:
        return []
    from alloy.precision.mjw_route import enable_route, route_enabled

    reasons = []
    for s in flat:
        if type(s).__name__ != "SolverMuJoCo" or route_enabled(s) or probe_mode(s) >= Determinism.RUN_TO_RUN:
            continue
        why = enable_route(s, allow_cg=bool(getattr(cfg, "mjw_route_cg", False)))
        if why is not None:
            reasons.append(f"SolverMuJoCo: Alloy's ordered route not used: {why}")
    for s in solvers:
        refresh_coupled_constants(s)
    return reasons


def refresh_coupled_constants(coupler: Any) -> bool:
    """Re-install a coupler's copies of its routed SolverMuJoCo's build-time constants. Returns whether it did.

    Enabling the route recomputes the solver's constants under the ordered sums (``mjw_route._recompute_constants``:
    ``set_const`` gives body / dof ``invweight0`` and ``body_subtreemass``). A coupler built before that has
    already installed proxy effective masses from the racing build-time values: Newton's
    ``SolverCoupledProxy._apply_proxy_body_effective_masses`` reads ``coupling_eval_effective_mass``, i.e.
    ``body_invweight0``, at setup. On GPU those differed by an ulp between in-process builds in some worlds
    (job_bottle_route, WP4 2026-11-06: SPH view body_mass in worlds 5 and 10), so every build after the first
    diverged from period 1. Newton's public ``notify_model_changed(BODY_INERTIAL_PROPERTIES)`` on the coupler
    forwards to its entries (the routed recompute repeats, ordered) and re-installs the proxy masses from them.
    A solver that is not a coupler (no ``_entries``), or holds no routed SolverMuJoCo, is left alone."""
    if getattr(coupler, "_entries", None) is None or not hasattr(coupler, "notify_model_changed"):
        return False
    from alloy.precision.mjw_route import route_enabled

    if not any(type(s).__name__ == "SolverMuJoCo" and route_enabled(s) for s in unwrap_solvers([coupler])):
        return False
    from alloy._vendor import newton

    coupler.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    log.info("determinism: %s re-installed proxy constants from the ordered SolverMuJoCo route", type(coupler).__name__)
    return True


def _device_of(solvers: Sequence[Any]) -> wp.context.Device:
    for s in solvers:
        model = getattr(s, "model", None)
        if model is not None and getattr(model, "device", None) is not None:
            return wp.get_device(model.device)
    return wp.get_device()


def notice_floor(config: PrecisionConfig, device: wp.context.Devicelike) -> Determinism:
    """The mode below which a monitor sets the DET_DOWNGRADED notice: the requested mode, and on a GPU at least
    RUN_TO_RUN whatever was requested, so profile "fast" (which requests NONE) lists the notice there like the
    others (GPU contact does not repeat at any profile; coordinator ruling 2026-11-10 02:23Z).
    Example: ``det_mode < notice_floor(config, model.device)``."""
    if wp.get_device(device).is_cuda:
        return max(config.determinism, Determinism.RUN_TO_RUN)
    return config.determinism


def effective_determinism(cfg: PrecisionConfig, solvers: Sequence[Any], world_count: int
                          ) -> tuple[Determinism, list[str]]:
    """(mode that runs, reasons it is below the request). ``solvers`` may also hold contact pipelines or Alloy
    layers exposing ``deterministic``; coupled steppers are unwrapped into their entries' solvers."""
    requested = Determinism(cfg.determinism)
    reasons: list[str] = []
    if requested == Determinism.NONE:
        return Determinism.NONE, ["NONE requested"]
    if requested == Determinism.GPU_TO_GPU:
        reasons.append("GPU_TO_GPU is not supported in v1 (Steel refuses it; no cross-architecture evidence): "
                       "RUN_TO_RUN at most")
        requested = Determinism.RUN_TO_RUN
    flat = list(unwrap_solvers(solvers))
    device = _device_of(flat)
    mode = requested
    for s in flat:
        det = solver_det(s, world_count, device)
        mode = min(mode, det.mode)
        if det.reason:
            reasons.append(det.reason)
    if not flat:
        reasons.append("no solvers given")
    log.info("effective determinism %s (requested %s) for %d worlds: %s", mode.name,
             Determinism(cfg.determinism).name, world_count, "; ".join(reasons) or "no downgrade")
    return mode, reasons


def unwrap_solvers(solvers: Sequence[Any]):
    for s in solvers:
        entries = getattr(s, "_entries", None)  # SolverCoupled entries (PRIVATE_HOOKS, D21)
        if entries is None:
            yield s
            continue
        values = entries.values() if isinstance(entries, dict) else entries
        for e in values:
            inner = getattr(e, "solver", None)
            if inner is not None:
                yield from unwrap_solvers([inner])


def _ordered(solver: Any) -> bool:
    """NATIVE, or a SolverMuJoCo on Alloy's ordered route (P3-1 round 5: in the D9 GPU population)."""
    if capability_of(solver)[0] == DetCapability.NATIVE:
        return True
    if type(solver).__name__ != "SolverMuJoCo":
        return False
    from alloy.precision.mjw_route import route_enabled

    return route_enabled(solver)


def native_share(solvers: Sequence[Any]) -> float:
    """1.0 if every solver path is ordered: NATIVE, or SolverMuJoCo on Alloy's route (D9 population), else 0.0
    (one pack shares one path)."""
    flat = list(unwrap_solvers(solvers))
    if not flat:
        return 0.0
    return 1.0 if all(_ordered(s) for s in flat) else 0.0
