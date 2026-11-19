# SPDX-License-Identifier: Apache-2.0
"""Where the proxy scheduler calls the coupling of cloth that takes up water (``add_cloth(absorb=...)``, opt-in;
``alloy.coupling.liquid_cloth_absorb``).

That coupling changes cloth and liquid masses only through these calls, written from its state, so the masses never
change inside an exchange:
- ``finish_exchange(state_out)`` at the exchange end: after Newton's coupler has written the entry solvers' results
  into ``state_out`` (its ``_reconcile_state``) and before the energy watchdog reads it. It merges the absorbed water's
  momentum into the cloth, then writes every mass.
- ``restore()`` after the state was put back: a period rollback or replay (``Simulation`` restores the snapshot, then
  calls the stepper), ``sim.reset(worlds)`` (``reset_worlds``), and the stepper's reset before the watchdog re-bases
  its energy datum. It writes the masses again without a merge.
Placement in ``Scheduler``'s bases: ``AbsorbRestoreMixin`` first (``reset_worlds``' chain ends in ``MatchResetMixin``),
``AbsorbExchangeMixin`` between ``SchedulerMixin`` and Newton's coupler, so its ``step`` and ``reset`` run inside
``SchedulerMixin``'s, before the watchdog. ``SchedulerADMM`` copies the parent's masses into its views every exchange,
so it refuses absorbing cloth (``refuse_with_admm``).
"""

from __future__ import annotations

from typing import Any

from alloy.core.errors import CouplingError

MODULE = "alloy.coupling.liquid_cloth_absorb"


def cloth_absorb(solver: Any) -> Any:
    """The absorb coupling for ``solver``, or None: no SPH entry absorbs, or this build has no absorb module."""
    try:
        from alloy.coupling.liquid_cloth_absorb import exchange_coupling  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != MODULE:
            raise
        return None
    return exchange_coupling(solver)


def refuse_with_admm(solver: Any) -> None:
    """A ``SchedulerADMM`` with an absorbing SPH entry is refused at build (the absorb module may refuse first)."""
    if cloth_absorb(solver) is not None:
        raise CouplingError("cloth that takes up water (absorb=) needs the proxy coupler, but this scene was built "
                            "with ADMM coupling, which re-copies the parent masses into its views every exchange. Fix: "
                            "drop absorb= from the cloth, or build it with proxy coupling (two dynamic entries "
                            "without a strong ADMM pair)")


class AbsorbRestoreMixin:
    """The masses written again from the restored state, before the rest of the scheduler re-plans or resets."""

    _cloth_absorb: Any = None

    def _absorb_restore(self) -> None:
        if self._cloth_absorb is not None:
            self._cloth_absorb.restore()

    def on_rollback(self, world_mask: Any = None) -> None:
        self._absorb_restore()
        super().on_rollback(world_mask)

    def replay_period(self) -> None:
        self._absorb_restore()
        super().replay_period()

    def reset_worlds(self, world_mask: Any = None) -> None:
        super().reset_worlds(world_mask)
        self._absorb_restore()


class AbsorbExchangeMixin:
    """Between ``SchedulerMixin`` and Newton's coupler: after the coupler's exchange or reset, before the watchdog."""

    _cloth_absorb: Any = None

    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float, *,
             observables: Any = None) -> None:
        super().step(state_in, state_out, control, contacts, dt, observables=observables)
        if self._cloth_absorb is not None:
            self._cloth_absorb.finish_exchange(state_out)

    def reset(self, state: Any, world_mask: Any = None, flags: Any = None) -> None:
        super().reset(state, world_mask=world_mask, flags=flags)
        if self._cloth_absorb is not None:
            self._cloth_absorb.restore()
