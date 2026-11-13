# SPDX-License-Identifier: Apache-2.0
"""Per-world exchange clock, the residency rule (known gap coupling-shared-exchange-clock; host logic only).

A coupled scene steps every world at one exchange step, ``control_dt / 2^j``, with j the finest any world needs. With
two packs built from the same scene, a coarse one of all W worlds and a fine one of ``slots`` worlds, the few worlds
that need a finer step can live ("reside") in the fine pack across periods. Their slots in the coarse pack are parked:
still stepped, but out of every clock bound. The coarse pack then runs at the rung of the worlds left in it.

``WorldClock`` decides who resides, once per control period, from each world's own rung ``j_world`` (the clock
census). It holds no simulation state and moves nothing: the caller applies the returned ``Moves``.

The rule. Sort the eligible worlds by rung, finest first (ties: lower world index). With the k finest as residents:

    cost(k) = W * 2^Jc + (slots * 2^Jf * fine_cost_num / fine_cost_den if k > 0 else 0)    per control period

where Jc is the finest rung among the other worlds and Jf the finest among the k. cost(0) is today's shared clock. The
fine weight is a build-time integer ratio (1/1 counts exchanges); every comparison cross-multiplies, so the decision
holds no float. The smallest k of least cost is chosen, and only when ``shared * min_gain_den >= cost * min_gain_num``.

Hysteresis. A chosen world enters at once, only while that gate holds. Residents are not dropped when the gate dips:
they are all released after ``leave_after`` updates in a row with the gate failing. A resident outside the best set
also leaves after ``leave_after`` updates in a row outside it, so a world whose rung flickers is not copied back and
forth. It leaves at once when it stops being eligible. A resident kept this way still counts in the fine pack's rung.
A chosen world that finds no free slot waits outside (and bounds the coarse pack) until one frees; with ``evict=True``
(the default) it instead displaces a strictly coarser resident that is outside the chosen set.

Re-run asks. ``update(..., rerun=mask)`` turns a world's re-run ask into an entry at the rung ``j_world`` gives, gate or
not. It gets a slot, or is returned in ``Moves.refused`` (the caller then re-runs the whole pack). Such an entrant has
no minimum stay: it leaves at the first update where its need is no finer than the coarse level and the best set does
not hold it. ``enter_now`` is the same entry in the middle of a period, into free slots only (all the asking worlds or
none: nobody is displaced after the coarse pack has stepped).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Moves", "WorldClock", "WorldClockConfig", "scaled_cost", "split_cost"]

_NONE = -1


@dataclass(frozen=True)
class WorldClockConfig:
    """``slots``: worlds the fine pack holds. ``leave_after``: updates outside the chosen set (or with the gain gate
    failing) before a resident leaves. ``min_gain_num / min_gain_den``: the least shared / split cost ratio worth
    taking residents for. ``fine_cost_num / fine_cost_den``: the fine pack's launch-cost weight per exchange.
    ``evict``: a strictly finer newcomer may displace an unwanted resident before its ``leave_after`` is up."""

    slots: int
    leave_after: int = 10
    min_gain_num: int = 5
    min_gain_den: int = 4
    fine_cost_num: int = 1
    fine_cost_den: int = 1
    evict: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.slots, bool) or int(self.slots) < 1:
            raise ValueError(f"WorldClockConfig.slots must be an integer >= 1, got {self.slots!r}")
        if int(self.leave_after) < 1:
            raise ValueError(f"WorldClockConfig.leave_after must be >= 1, got {self.leave_after!r}")
        for name in ("min_gain_num", "min_gain_den", "fine_cost_num", "fine_cost_den"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"WorldClockConfig.{name} must be an integer >= 1, got {value!r}")
        if self.min_gain_num < self.min_gain_den:
            raise ValueError(f"WorldClockConfig min_gain must be >= 1, got {self.min_gain_num}/{self.min_gain_den}")


@dataclass(frozen=True)
class Moves:
    """One update's transfers, in the order to apply them: every ``leave`` (slot, world), then every ``enter``
    (world, slot). ``refused``: re-run asks that found no slot (the caller re-runs the whole pack)."""

    enter: tuple[tuple[int, int], ...] = ()
    leave: tuple[tuple[int, int], ...] = ()
    refused: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.enter or self.leave)


def split_cost(j_world: np.ndarray, resident: np.ndarray, slots: int) -> tuple[int, int, int]:
    """(exchanges per period, Jc, Jf) for the given residents: W * 2^Jc + slots * 2^Jf, or the shared clock's
    W * 2^J when nobody resides (then Jf = Jc = J)."""
    j = np.asarray(j_world, dtype=np.int64)
    resident = np.asarray(resident, dtype=bool)
    if not resident.any():
        top = int(j.max())
        return j.size * 2**top, top, top
    rest = j[~resident]
    j_coarse = int(rest.max()) if rest.size else 0
    j_fine = int(j[resident].max())
    return j.size * 2**j_coarse + int(slots) * 2**j_fine, j_coarse, j_fine


def scaled_cost(j_world: np.ndarray, resident: np.ndarray, slots: int, fine_num: int = 1, fine_den: int = 1) -> int:
    """The weighted cost times ``fine_den``, in exact Python integers: W * 2^Jc * den + slots * 2^Jf * num (the
    shared clock's W * 2^J * den when nobody resides)."""
    j = np.asarray(j_world, dtype=np.int64)
    resident = np.asarray(resident, dtype=bool)
    if not resident.any():
        return j.size * 2 ** int(j.max()) * int(fine_den)
    _, j_coarse, j_fine = split_cost(j, resident, slots)
    return j.size * 2**j_coarse * int(fine_den) + int(slots) * 2**j_fine * int(fine_num)


class WorldClock:
    """Host state: which worlds reside in the fine pack, in which slot, and for how long each has been unwanted."""

    def __init__(self, worlds: int, config: WorldClockConfig) -> None:
        if int(worlds) < 2 or config.slots >= int(worlds):
            raise ValueError(f"WorldClock needs worlds >= 2 and slots < worlds, got worlds={worlds}, "
                             f"slots={config.slots}")
        self.worlds, self.config = int(worlds), config
        self.reset()

    def reset(self) -> None:
        """Nobody resides; counters back to zero (``sim.reset()``)."""
        self.slot_of = np.full(self.worlds, _NONE, dtype=np.int64)
        self.world_in = np.full(self.config.slots, _NONE, dtype=np.int64)
        self._out = np.zeros(self.worlds, dtype=np.int64)        # per resident: updates in a row outside the set
        self._probation = np.zeros(self.worlds, dtype=bool)      # re-run entrants with no minimum stay
        self._j = np.zeros(self.worlds, dtype=np.int64)
        self._gate_fail = 0                                      # updates in a row with the gain gate failing
        self.updates = self.enters = self.leaves = 0

    @property
    def resident(self) -> np.ndarray:
        return self.slot_of >= 0

    # ---- the decision ----
    def _scaled(self, j: np.ndarray, resident: np.ndarray) -> int:
        c = self.config
        return scaled_cost(j, resident, c.slots, c.fine_cost_num, c.fine_cost_den)

    def _best(self, j: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, bool]:
        """(the cheapest set of the finest eligible worlds, whether it clears the gain gate); gate or not, the set is
        returned so residents can be judged against it."""
        order = [int(w) for w in np.lexsort((np.arange(self.worlds), -j)) if eligible[w]]
        best = np.zeros(self.worlds, dtype=bool)
        shared = self._scaled(j, best)
        best_cost, best_k = shared, 0
        trial = best.copy()
        for k, w in enumerate(order[: self.config.slots], start=1):
            trial[w] = True
            cost = self._scaled(j, trial)
            if cost < best_cost:
                best_cost, best_k = cost, k
        best[order[:best_k]] = True
        gate = bool(best_k) and shared * self.config.min_gain_den >= best_cost * self.config.min_gain_num
        return best, gate

    def choose(self, j_world: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        """The cost-optimal resident set for these rungs (bool [W]), with no memory of who resides now; empty when
        it does not clear the gain gate."""
        best, gate = self._best(np.asarray(j_world, dtype=np.int64), np.asarray(eligible, dtype=bool))
        return best if gate else np.zeros(self.worlds, dtype=bool)

    def update(self, j_world: np.ndarray, eligible: np.ndarray, rerun: np.ndarray | None = None) -> Moves:
        """Take this period's rungs (int [W]), eligibility (bool [W]) and optional re-run asks (bool [W]); return the
        transfers to make.

        ``j_world`` must be each world's own need WITHOUT the attempt's extra halvings: a one-period re-run is not
        need. A world flagged in ``rerun`` wants entry this period at its ``j_world`` even if the gain gate fails,
        limited by free slots (plus eviction when on); those left without a slot come back in ``Moves.refused``."""
        j = np.asarray(j_world, dtype=np.int64)
        eligible = np.asarray(eligible, dtype=bool)
        ask = np.zeros(self.worlds, dtype=bool) if rerun is None else np.asarray(rerun, dtype=bool)
        if not (j.shape == eligible.shape == ask.shape == (self.worlds,)):
            raise ValueError(f"WorldClock.update needs ({self.worlds},) arrays, got {j.shape}, {eligible.shape} "
                             f"and {ask.shape}")
        best, gate = self._best(j, eligible)
        chosen = best if gate else np.zeros(self.worlds, dtype=bool)
        leave = [self._release(int(w)) for w in np.flatnonzero(self._overdue(j, eligible, best, gate, chosen, ask))]
        if not self.resident.any():
            self._gate_fail = 0
        enter, refused = self._admit(j, chosen, ask, eligible, leave)
        self._j = j.copy()
        self.updates += 1
        self.enters += len(enter)
        self.leaves += len(leave)
        return Moves(enter=tuple(enter), leave=tuple(leave), refused=tuple(refused))

    def _overdue(self, j: np.ndarray, eligible: np.ndarray, best: np.ndarray, gate: bool, chosen: np.ndarray,
                 ask: np.ndarray) -> np.ndarray:
        """Residents that leave now; also advances the out and gate-failure counters."""
        resident = self.resident
        self._gate_fail = 0 if gate or not resident.any() else self._gate_fail + 1
        self._out[resident & best] = 0
        self._out[resident & ~best] += 1
        self._probation[resident & best] = False
        overdue = resident & ((self._out >= self.config.leave_after) | ~eligible)
        if self._gate_fail >= self.config.leave_after:
            overdue |= resident
        rest = ~resident                                          # a re-run entrant has no minimum stay: once its
        j_coarse = int(j[rest].max()) if rest.any() else 0        # need is no finer than the coarse pack's, it leaves
        overdue |= resident & self._probation & ~best & ~ask & (j <= j_coarse)
        return overdue

    def _admit(self, j: np.ndarray, chosen: np.ndarray, ask: np.ndarray, eligible: np.ndarray,
               leave: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[int]]:
        """Give slots to the chosen and asking worlds, finest first; (entries, refused re-run asks)."""
        enter: list[tuple[int, int]] = []
        refused = [int(w) for w in np.flatnonzero(ask & ~eligible & ~self.resident)]
        wanted = (chosen | (ask & eligible)) & ~self.resident
        for w in sorted(np.flatnonzero(wanted).tolist(), key=lambda w: (-int(j[w]), w)):
            slot = self._free_slot()
            if slot == _NONE and self.config.evict:
                victim = self._victim(j, chosen | ask, int(j[w]))
                if victim != _NONE:
                    leave.append(self._release(victim))
                    slot = self._free_slot()
            if slot == _NONE:
                refused.extend([int(w)] if ask[w] else [])
                continue
            self.slot_of[w], self.world_in[slot], self._out[w] = slot, w, 0
            self._probation[w] = bool(ask[w])
            enter.append((int(w), int(slot)))
        return enter, sorted(refused)

    def _free_slot(self) -> int:
        free = np.flatnonzero(self.world_in == _NONE)
        return int(free[0]) if free.size else _NONE

    def _release(self, world: int) -> tuple[int, int]:
        slot = int(self.slot_of[world])
        self.slot_of[world], self.world_in[slot], self._out[world] = _NONE, _NONE, 0
        self._probation[world] = False
        return slot, world

    def enter_now(self, worlds: Any) -> Moves:
        """Mid-period re-run asks (world ids): each takes a free slot, or all are ``refused`` (no eviction: a resident
        cannot leave between the two packs' steps of one period)."""
        worlds = [int(w) for w in np.asarray(worlds, dtype=np.int64).reshape(-1)]
        free = np.flatnonzero(self.world_in == _NONE)
        if len(worlds) > free.size or any(self.slot_of[w] >= 0 for w in worlds):
            return Moves(refused=tuple(worlds))
        enter = []
        for w, slot in zip(worlds, free.tolist(), strict=False):
            self.slot_of[w], self.world_in[slot], self._out[w], self._probation[w] = slot, w, 0, True
            enter.append((w, int(slot)))
        self.enters += len(enter)
        return Moves(enter=tuple(enter))

    def release(self, world: int) -> tuple[int, int]:
        """A caller takes resident ``world`` out now (a write to it, a reset): (slot, world). It may enter again at
        the next update."""
        if self.slot_of[world] < 0:
            raise ValueError(f"WorldClock.release: world {world} is not resident")
        self.leaves += 1
        return self._release(int(world))

    def _victim(self, j: np.ndarray, keep: np.ndarray, j_entrant: int) -> int:
        """The resident a strictly finer entrant displaces: outside ``keep`` (the chosen and asking worlds), coarsest
        first, then the longest unwanted, then the lower world index. -1 when there is none."""
        pool = [int(w) for w in np.flatnonzero(self.resident & ~keep) if int(j[w]) < j_entrant]
        if not pool:
            return _NONE
        return min(pool, key=lambda w: (int(j[w]), -int(self._out[w]), w))

    # ---- reads ----
    def rungs(self) -> tuple[int, int]:
        """(Jc, Jf) for the residents after the last update; equal when nobody resides."""
        _, j_coarse, j_fine = split_cost(self._j, self.resident, self.config.slots)
        return j_coarse, j_fine

    def cost(self, j_world: np.ndarray) -> dict[str, Any]:
        """Exchanges per control period for these rungs: the shared clock's, and with the current residents, as plain
        counts (``shared``, ``split``) and weighted by the fine pack's launch cost (``*_weighted`` is an exact
        ``(numerator, denominator)`` pair; ``gain`` is the weighted ratio, for display only)."""
        j = np.asarray(j_world, dtype=np.int64)
        shared = self.worlds * 2 ** int(j.max())
        split, j_coarse, j_fine = split_cost(j, self.resident, self.config.slots)
        den = self.config.fine_cost_den
        shared_w = self._scaled(j, np.zeros(self.worlds, dtype=bool))
        split_w = self._scaled(j, self.resident)
        return {"shared": shared, "split": split, "shared_weighted": (shared_w, den), "split_weighted": (split_w, den),
                "gain": shared_w / split_w, "j_coarse": j_coarse, "j_fine": j_fine}

    def report(self) -> dict[str, Any]:
        j_coarse, j_fine = self.rungs()
        return {"residents": int(self.resident.sum()), "resident_worlds": np.flatnonzero(self.resident).tolist(),
                "j_coarse": j_coarse, "j_fine": j_fine, "enters": self.enters, "leaves": self.leaves,
                "updates": self.updates}

    # ---- snapshots ----
    def state(self) -> dict[str, np.ndarray]:
        """Everything a later update depends on, as copies."""
        return {"slot_of": self.slot_of.copy(), "world_in": self.world_in.copy(), "out": self._out.copy(),
                "probation": self._probation.copy(), "j": self._j.copy(),
                "counts": np.array([self.updates, self.enters, self.leaves, self._gate_fail], dtype=np.int64)}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        if state["slot_of"].shape != (self.worlds,) or state["world_in"].shape != (self.config.slots,):
            raise ValueError("WorldClock.load_state: the state is for another world or slot count")
        self.slot_of, self.world_in = state["slot_of"].copy(), state["world_in"].copy()
        self._out, self._j = state["out"].copy(), state["j"].copy()
        self._probation = state["probation"].copy()
        self.updates, self.enters, self.leaves, self._gate_fail = (int(v) for v in state["counts"])
