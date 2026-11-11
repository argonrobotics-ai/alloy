# SPDX-License-Identifier: Apache-2.0
"""Health adapters and capacity sources for stock Newton solvers and contacts (SPEC §3.7, §2.2).

- SolverMuJoCo: ``iters = mjw_data.solver_niter`` ((nworld,) int32, mujoco_warp@a9daa05f:mujoco_warp/_src/types.py:2523);
  MJWarp keeps no per-world residual, so ``residual`` stays NaN and UNCONVERGED comes from the per-world
  ``Data.overflow`` ITERATIONS bit (solver.py:3828). The same bitmask (types.py:162-179, 2518) gives CONTACT_OVERFLOW
  (broad/narrow phase, CCD, height field, EPA horizon) and ROW_OVERFLOW (nefc, njmax_nnz, nvmax).
- SolverSteel: ``constraint_overflow`` ((world_count + 1,) bool, newton@ac7f7830:newton/_src/solvers/steel/
  solver_steel.py:2582) -> ROW_OVERFLOW.
- newton.Contacts: ``rigid_contact_count > rigid_contact_max``, ``soft_contact_count > soft_contact_max`` and the
  reducer's ``_reduction_overflow`` (sim/contacts.py:256-265, sim/collide.py:333-339) are global counts, so they flag
  every world (the dropped contacts cannot be attributed; fail closed). A stepper that discards the rigid stream
  declares it (``HealthMonitor(rigid_contacts_unused=True)``, SCR-JOB1-3); the rigid count, Newton contacts over
  naconmax and the reducer are then ignored, and soft-contact overflow still fails closed.

Adapter convention (``HealthSample``): ``iters`` per world; ``residual`` normalised by the solver's own tolerance,
so ``> 1`` means unconverged; NaN when the solver reports none.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import warp as wp

from alloy._vendor import newton
from alloy.core.registry import HealthSample, register_health_adapter
from alloy.precision.types import HealthFlags

log = logging.getLogger("alloy.precision")


@dataclass(frozen=True)
class CapacitySource:
    """One overflow signal. ``kind``: ``nonzero`` (int32, [W] or [1] global), ``bool`` ([>= W] per world),
    ``bitmask`` (MJWarp per-world overflow word) or ``over`` (``array[index] > cap``, global)."""

    array: wp.array
    kind: str = "nonzero"
    flag: int = int(HealthFlags.CONTACT_OVERFLOW)
    per_world: bool = True
    index: int = 0
    cap: int = 0
    bits: tuple[int, int, int] = (0, 0, 0)  # bitmask: (contact bits, row bits, iteration bits)
    label: str = ""
    rigid: bool = False  # a rigid contact stream source (count or reducer): off when the stream is declared unused
    reset: Callable[[wp.array | None], None] | None = None  # clears a sticky source's masked worlds at a world reset
    invalid: bool = False  # nonzero: a nonzero value also makes the world INVALID (self-contact pass-through)


def _mjw_bits() -> tuple[int, int, int]:
    from alloy._vendor.mujoco_warp import OverflowType as O

    contact = O.BROADPHASE | O.NARROWPHASE | O.CCD | O.HFIELD | O.EPA_HORIZON
    rows = O.NEFC | O.NJMAX_NNZ | O.NVMAX
    # Newton's own pipeline/nconmax bits (16, 17) go to Observables.overflow only (solvers/mujoco/kernels.py:3550-3558),
    # so Newton-contact overflow is checked separately against naconmax (_mujoco_sources). CONTACT_MATCH and TACTILE
    # are sensor-buffer overflows (sensor.py), not dropped contacts: not mapped.
    return int(contact), int(rows), int(O.ITERATIONS | O.LS_ITERATIONS)


def _mjw_data(solver: newton.solvers.SolverBase):
    return getattr(solver, "mjw_data", None)


def mujoco_health(solver: newton.solvers.SolverBase, sample: HealthSample) -> None:
    """MuJoCo iterations of the last solve; residual NaN (MJWarp has none per world)."""
    data = _mjw_data(solver)
    if data is None or getattr(data, "solver_niter", None) is None:
        return
    wp.copy(sample.iters, data.solver_niter, count=min(sample.iters.shape[0], data.solver_niter.shape[0]))


def solver_sources(solver: newton.solvers.SolverBase, contacts: newton.Contacts | None = None,
                   cfg: object | None = None) -> list[CapacitySource]:
    """Overflow sources a stock solver exposes (empty for solvers with none). ``cfg``: the monitor's CheckPolicy."""
    name = type(solver).__name__
    if isinstance(solver, newton.solvers.SolverMuJoCo):   # subclasses too: rk4's SyncedMuJoCo, WP8's MjcCoreSolver
        return _mujoco_sources(solver, contacts)
    if isinstance(solver, newton.solvers.SolverVBD):     # cloth self contact over its row buffers (gap #13)
        return _vbd_sources(solver, cfg)
    if name == "SolverSteel" and getattr(solver, "constraint_overflow", None) is not None:
        return [CapacitySource(solver.constraint_overflow, "bool", int(HealthFlags.ROW_OVERFLOW),
                               label="steel.constraint_overflow")]
    return []


def _vbd_sources(solver: newton.solvers.SolverVBD, cfg: object | None = None) -> list[CapacitySource]:
    from alloy.precision import self_contact

    sc = self_contact.watch(solver)
    if sc is None:
        return []
    out = [CapacitySource(sc.dropped, "nonzero", self_contact.flag(), per_world=True, label="vbd self-contact rows",
                          reset=sc.reset)]
    mode = self_contact.cross_mode(cfg)                  # the pass-through meter: off (default) / report / invalid
    if mode != "off":
        sc.enable_cross(invalid=mode == "invalid")
    if sc.invalid:                                       # dropped pairs whose surfaces then crossed: INVALID
        out.append(CapacitySource(sc.crossed, "nonzero", self_contact.flag(), per_world=True, invalid=True,
                                  label="vbd self-contact crossings"))
    return out


def _mujoco_sources(solver, contacts: newton.Contacts | None) -> list[CapacitySource]:
    from alloy.precision import health_tunnel  # GAP-E2: wraps this solver's step once, idempotent

    tunnel = health_tunnel.sources(solver) if contacts is None else []
    data = _mjw_data(solver)
    if data is None or getattr(data, "overflow", None) is None:
        return tunnel
    out = tunnel + [CapacitySource(data.overflow, "bitmask", bits=_mjw_bits(), label="mjw_data.overflow")]
    from alloy.precision import mjw_chol  # BUL-0016: Newton Cholesky pivots raised to mjMINVAL (counted, DEGRADED)
    from alloy.precision.health_reset import zero_worlds  # noqa: PLC0415

    if mjw_chol.install():
        clamps, _ = mjw_chol.prepare(solver.mjw_model, data)
        out.append(CapacitySource(clamps, "nonzero", int(HealthFlags.FALLBACK_USED), label="newton cholesky clamps",
                                  reset=zero_worlds(clamps)))          # a world reset starts uncounted (SCR-WP5-19)
    naconmax = int(getattr(data, "naconmax", 0) or 0)
    # private flag (solvers/mujoco/solver_mujoco.py, PRIVATE_HOOKS): True when MJWarp runs its own collision
    native = getattr(solver, "_use_mujoco_contacts", getattr(solver, "use_mujoco_contacts", True))
    if contacts is not None and naconmax > 0 and not native:
        # Newton contacts beyond MJWarp's shared buffer are dropped in the conversion (kernels.py:494)
        out.append(CapacitySource(contacts.rigid_contact_count, "over", per_world=False, cap=naconmax,
                                  label="newton contacts > naconmax", rigid=True))
    return out


def contact_sources(contacts: newton.Contacts, rigid: bool = True) -> list[CapacitySource]:
    """Global overflow signals of a Newton contact buffer. ``rigid=False`` (SCR-JOB1-3: a stepper that discards the
    rigid stream, so its dropped candidates reach no solver) leaves out the rigid count and the narrow phase's
    contact reducer, which only rigid shape pairs pass through. Soft-contact overflow always stays (fail closed)."""
    out = []
    if rigid:
        out.append(CapacitySource(contacts.rigid_contact_count, "over", per_world=False,
                                  cap=int(contacts.rigid_contact_max), label="rigid_contact_count > rigid_contact_max",
                                  rigid=True))
    soft = getattr(contacts, "soft_contact_count", None)
    if soft is not None and int(getattr(contacts, "soft_contact_max", 0) or 0) > 0:
        out.append(CapacitySource(soft, "over", per_world=False, cap=int(contacts.soft_contact_max),
                                  label="soft_contact_count > soft_contact_max"))
    reduction = getattr(contacts, "_reduction_overflow", None)
    if rigid and reduction is not None:
        out.append(CapacitySource(reduction, "nonzero", per_world=False, label="contact reduction overflow",
                                  rigid=True))
    return out


def user_sources(flags: Sequence[wp.array | tuple[wp.array, int]], world_count: int) -> list[CapacitySource]:
    """``capacity_flags`` of ``HealthMonitor``: int32 arrays ([W] per world or [1] global), optionally paired with
    the HealthFlags bit to raise (default CONTACT_OVERFLOW); bool arrays are per world."""
    out = []
    for item in flags:
        arr, bit = (item if isinstance(item, tuple) else (item, int(HealthFlags.CONTACT_OVERFLOW)))
        if arr.dtype == wp.bool:
            out.append(CapacitySource(arr, "bool", int(bit), label="capacity_flags"))
        else:
            out.append(CapacitySource(arr, "nonzero", int(bit), per_world=arr.shape[0] >= world_count,
                                      label="capacity_flags"))
    return out


@dataclass
class _Registered:
    done: bool = False
    classes: list[str] = field(default_factory=list)


_STATE = _Registered()


def register_builtin_adapters() -> None:
    """Register the stock-solver health adapters once (idempotent)."""
    if _STATE.done:
        return
    register_health_adapter(newton.solvers.SolverMuJoCo, mujoco_health)
    _STATE.classes.append("SolverMuJoCo")
    _STATE.done = True
    log.debug("health adapters registered: %s", _STATE.classes)


@dataclass(frozen=True)
class MjwContactSource:
    """A SolverMuJoCo's own contacts, read on device every check (SCR-SC0-1): on the native MuJoCo contact route
    no Newton ``Contacts`` reaches the monitor, so these are the only rigid contacts there are."""

    data: object                 # mujoco_warp Data: nacon, contact.dist / worldid / geom
    geom_shape: wp.array         # [nworld, ngeom] Newton shape per MuJoCo geom; empty when not the monitor's shapes


def mjw_contact_sources(solvers: Sequence[object], model: newton.Model) -> list[MjwContactSource]:
    """Every SolverMuJoCo (coupled entries unwrapped) whose MuJoCo Warp Data can hold contacts for all of
    ``model``'s worlds. With ``use_mujoco_contacts=False`` its contacts are Newton's, converted, so reading them
    too only repeats a max."""
    from alloy.precision.determinism import unwrap_solvers

    out: list[MjwContactSource] = []
    for s in unwrap_solvers(solvers):
        d = getattr(s, "mjw_data", None)
        if d is None or not hasattr(d, "contact") or int(getattr(d, "naconmax", 0)) == 0:
            continue
        if int(d.nworld) != int(model.world_count):
            log.warning("health: %s has %d MuJoCo worlds for %d model worlds; its contacts are not read",
                        type(s).__name__, int(d.nworld), int(model.world_count))
            continue
        shapes = getattr(s, "mjc_geom_to_newton_shape", None)
        if getattr(s, "model", None) is not model or shapes is None:  # a coupled entry's view: other shape ids
            shapes = wp.zeros((0, 0), dtype=wp.int32, device=model.device)
        out.append(MjwContactSource(d, shapes))
    return out
