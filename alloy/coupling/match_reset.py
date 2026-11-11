# SPDX-License-Identifier: Apache-2.0
"""Clear the proxy pipelines' cross-step contact matching whenever worlds go back in time (WP4 item (a), the follow-up
to SCR-WP9-ROD F3; coordinator 2026-11-08 18:00Z).

A rod entry's proxy pipeline matches its contacts "sticky" (``alloy.integration.stepper._proxy_pipeline``): Newton's
``ContactMatcher`` keeps the last collide's contacts and replays a matched contact's body-frame anchors, the friction
anchor that holds a gripped rod. Its arrays have no per-world axis, so no snapshot holds them (a masked restore would
raise). When worlds go back, their matcher would still hold contacts from after that point, and the next collide
could replay those anchors (they match within the 0.5 mm witness). So the matcher is cleared for the worlds that go back:
- ``sim.reset(world_mask)``: ``IntegratedSimulation.reset`` calls ``reset_contact_matching``. The coupler's own
  ``reset`` does too;
- a rollback (``on_rollback``) clears the rolled-back worlds, all of them when Simulation rolls back every world.
  ``replay_period`` clears every world;
- a per-world rollback also re-steps the worlds it keeps, then puts back their end state
  (``IntegratedSimulation._restore_kept``). Their matcher then holds the re-run's contacts, a period ahead, so they are
  cleared when the next period starts (the first ``choose_clock`` after the re-run).
A cleared world's next collide matches nothing: one step of fresh contacts, as before F3. Then it anchors again. A
pipeline without matching (every non-rod destination) is not touched.

``sim.reset`` also needs the per-world history the snapshot does not hold to start as a fresh build's
(``reset_worlds``; WP4 item (b), SCR-WP9-ROD F3's follow-up). Without it, a coupled rod grip's episode after a reset
missed the fresh one's by 3.9e-5 m from its first period, and by 0.11 m once its first contact came:
- a VBD entry's rigid history (``joint_rho``, the pose rebaseline mask) went on from the last episode. Newton's
  ``SolverVBD.reset(state, mask, flags=0)`` is its episode reset: the history cold-starts, the restored state stays;
- the force-jump re-run's cooldown (``OnsetRefiner.last``) and the ladder's repeat window (``PeriodLadder.last``) ran
  across the reset. The last episode's first-contact re-run left the next episode's first contact unarmed: no re-run,
  another path. They are cleared for the reset worlds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy._vendor.newton.solvers import SolverVBD

NEVER = -(2**62)                                   # OnsetRefiner / PeriodLadder: no event yet


def world_bools(world_mask: Any) -> np.ndarray | None:
    """A world mask (Warp or numpy, bool or int, per world, optionally with Newton's trailing global entry) as bool;
    None stays None (every world)."""
    if world_mask is None:
        return None
    host = world_mask.numpy() if hasattr(world_mask, "numpy") else world_mask
    return np.asarray(host).reshape(-1) != 0


class MatchResetMixin:
    """For couplers whose proxies may build matching pipelines (Scheduler, SchedulerADMM): clears them as the module
    docstring says. Goes before SchedulerMixin in the bases."""

    _match_kept: np.ndarray | None = None          # worlds a per-world rollback kept: cleared at the next period
    _match_rerun_next: bool = False                # the next choose_clock is the rolled-back period's re-run

    def reset_contact_matching(self, world_mask: Any = None) -> int:
        """Clear the matching of every proxy pipeline that has it, for ``world_mask`` (None: every world). Returns the
        pipelines cleared."""
        worlds = world_bools(world_mask)
        if worlds is not None and not worlds.any():
            return 0
        return sum(f.reset_matching(worlds) for f in self._contact_factories())

    def reset_worlds(self, world_mask: Any = None) -> None:
        """``sim.reset(world_mask)`` (``IntegratedSimulation.reset``, after the snapshot put the state back): the
        matching, every VBD entry's rigid history (``SolverVBD.reset`` with flags 0) and the re-run cooldowns of those
        worlds (None: every world) start as a fresh build's (module docstring)."""
        worlds = world_bools(world_mask)
        if worlds is not None and not worlds.any():
            return
        self.reset_contact_matching(worlds)
        for entry in self._entries.values():
            if isinstance(entry.solver, SolverVBD):
                entry.solver.reset(entry.state_0, world_mask=_newton_mask(entry.solver, worlds), flags=0)
        for memory in (getattr(self, "_onset", None), getattr(self, "_ladder", None)):
            last = getattr(memory, "last", None)
            if last is not None and last.size:
                last[slice(None) if worlds is None else worlds[: last.size]] = NEVER

    def reset(self, state: Any, world_mask: Any = None, flags: Any = None) -> None:
        super().reset(state, world_mask=world_mask, flags=flags)
        self.reset_contact_matching(world_mask)
        self._match_kept, self._match_rerun_next = None, False

    def on_rollback(self, world_mask: Any = None) -> None:
        back = world_bools(world_mask)
        self.reset_contact_matching(back)
        if back is not None and not back.all():
            kept = ~back
            self._match_kept = kept if self._match_kept is None else self._match_kept | kept
        self._match_rerun_next = True
        super().on_rollback(world_mask)

    def replay_period(self) -> None:
        self.reset_contact_matching(None)
        super().replay_period()

    def choose_clock(self, control_dt: float, last: Any) -> tuple[float, dict[str, int]]:
        if self._match_rerun_next:                 # the re-run itself: the kept worlds are not back in place yet
            self._match_rerun_next = False
        elif self._match_kept is not None:         # the period after it: Simulation has restored the kept worlds
            kept, self._match_kept = self._match_kept, None
            self.reset_contact_matching(kept)
        return super().choose_clock(control_dt, last)


def _newton_mask(solver: Any, worlds: np.ndarray | None) -> Any:
    """Newton's reset mask for ``solver`` (world_count + 1 bools, the global entry unset); None for every world."""
    if worlds is None:
        return None
    w = max(int(solver.model.world_count or 0), 1)
    return wp.array(np.append(worlds[:w], False), dtype=wp.bool, device=solver.device)
