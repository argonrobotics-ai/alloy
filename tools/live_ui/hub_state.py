# SPDX-License-Identifier: Apache-2.0
"""The hub's live model of a run: the workers, every world's latest numbers, the reward line, the picks.

Global world ids: each worker gets a block of ids in the order it first said hello (``offset``); a reconnecting
worker keeps its block. Picks per worker: worlds open in a browser (kept while viewed, plus a few seconds) and
worlds flagged automatically (best and worst episode return, any health alarm), at most 8 per worker. Auto picks
are sticky for ``STICKY_S`` so streams do not churn.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_PICKED = 8
STICKY_S = 20.0
VIEW_TTL_S = 8.0
FIELDS = ("reward", "ret", "success", "done", "health", "ep_ret", "ep_success", "paused")


@dataclass
class Worker:
    name: str
    offset: int
    worlds: int
    hello: dict[str, Any]
    writer: Any = None                      # asyncio StreamWriter while connected
    numbers: dict[str, np.ndarray] = field(default_factory=dict)
    header: dict[str, Any] = field(default_factory=dict)
    picked: list[int] = field(default_factory=list)     # local indices
    auto: dict[str, tuple[int, float]] = field(default_factory=dict)   # reason -> (local world, since)
    last_seen: float = 0.0

    @property
    def connected(self) -> bool:
        return self.writer is not None


class RunState:
    def __init__(self, run: str, load_picks: int = 0) -> None:
        self.run = run
        self.load_picks = load_picks                     # > 0: always stream this many worlds per worker (load test)
        self.workers: dict[str, Worker] = {}
        self.viewed: dict[int, float] = {}               # global world -> last time a browser asked for it
        self.line: list[tuple[float, float]] = []        # (t, mean episode return) of the whole run
        self.success_line: list[tuple[float, float]] = []
        self.marks: list[dict[str, Any]] = []            # commands on the timeline
        self._ep_acc: dict[str, tuple[float, float, float]] = {}

    # --- workers ---------------------------------------------------------------------------------------------------

    def hello(self, hello: dict[str, Any], writer: Any) -> Worker:
        name = str(hello["worker"])
        w = self.workers.get(name)
        if w is None:
            offset = sum(x.worlds for x in self.workers.values())
            w = self.workers[name] = Worker(name, offset, int(hello["worlds"]), hello)
        if w.worlds != int(hello["worlds"]):              # same name, new size: re-lay the blocks in hello order
            w.worlds, w.numbers = int(hello["worlds"]), {}
            offset = 0
            for x in self.workers.values():
                x.offset, offset = offset, offset + x.worlds
        w.hello, w.writer, w.last_seen = hello, writer, time.time()
        w.picked = []
        return w

    def owner(self, gw: int) -> tuple[Worker, int] | None:
        for w in self.workers.values():
            if w.offset <= gw < w.offset + w.worlds:
                return w, gw - w.offset
        return None

    @property
    def world_count(self) -> int:
        return sum(w.worlds for w in self.workers.values())

    # --- numbers ---------------------------------------------------------------------------------------------------

    def numbers(self, w: Worker, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        w.numbers, w.header, w.last_seen = arrays, header, time.time()
        n, s, k = header.get("episodes", 0.0), header.get("ep_return_sum", 0.0), header.get("ep_success_sum", 0.0)
        a = self._ep_acc.get(w.name, (0.0, 0.0, 0.0))
        self._ep_acc[w.name] = (a[0] + n, a[1] + s, a[2] + k)

    def tick_line(self, now: float, every: float = 5.0) -> None:
        """Append one point to the reward line from the episodes that ended since the last point."""
        if self.line and now - self.line[-1][0] < every:
            return
        n = sum(a[0] for a in self._ep_acc.values())
        if n <= 0:
            return
        r = sum(a[1] for a in self._ep_acc.values()) / n
        s = sum(a[2] for a in self._ep_acc.values()) / n
        self.line.append((now, round(r, 4)))
        self.success_line.append((now, round(s, 4)))
        self._ep_acc.clear()

    def steps_per_s(self) -> float:
        return float(sum(w.header.get("steps_per_s", 0.0) * w.worlds for w in self.workers.values() if w.connected))

    def overview(self) -> dict[str, Any]:
        """What the overview page draws: header numbers, the reward line, one status per world."""
        status, ret = [], []
        for w in sorted(self.workers.values(), key=lambda x: x.offset):
            st = _status(w.numbers, w.worlds, w.connected)
            status += st
            ret += [round(float(x), 3) for x in w.numbers.get("ret", np.zeros(w.worlds))]
        return {"type": "overview", "run": self.run, "t": time.time(), "worlds": self.world_count,
                "steps_per_s": round(self.steps_per_s(), 1), "line": self.line[-400:],
                "success": self.success_line[-400:], "status": status, "ret": ret,
                "workers": [{"name": w.name, "offset": w.offset, "worlds": w.worlds, "connected": w.connected,
                             "step": w.header.get("step", 0), "paused": w.header.get("paused_run", False),
                             "settings": w.hello.get("settings", []), "robots": w.hello.get("robots", [])}
                            for w in sorted(self.workers.values(), key=lambda x: x.offset)],
                "marks": self.marks[-200:]}

    # --- picks -----------------------------------------------------------------------------------------------------

    def view(self, gw: int) -> None:
        self.viewed[gw] = time.time()

    def picks(self, w: Worker, now: float) -> list[int]:
        """The worlds this worker should stream in full, viewed first, then the sticky auto flags."""
        if self.load_picks:
            return list(range(min(self.load_picks, MAX_PICKED, w.worlds)))
        viewed = [gw - w.offset for gw, t in self.viewed.items()
                  if now - t < VIEW_TTL_S and w.offset <= gw < w.offset + w.worlds]
        nums = w.numbers
        if nums:
            ep = nums["ep_ret"]
            alarms = np.nonzero(nums["health"] > 0)[0]
            self._flag(w, "best", int(np.argmax(ep)), now)
            self._flag(w, "worst", int(np.argmin(ep)), now)
            if len(alarms):
                self._flag(w, "alarm", int(alarms[0]), now)
        out = list(dict.fromkeys(viewed + [v[0] for v in w.auto.values()]))
        return out[:MAX_PICKED]

    @staticmethod
    def _flag(w: Worker, reason: str, world: int, now: float) -> None:
        cur = w.auto.get(reason)
        if cur is None or now - cur[1] > STICKY_S:
            w.auto[reason] = (world, now)

    def flags(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for w in self.workers.values():
            for reason, (world, _t) in w.auto.items():
                out.setdefault(w.offset + world, []).append(reason)
        return out


def _status(nums: dict[str, np.ndarray], n: int, connected: bool) -> list[int]:
    """0 running, 1 success (now, or the world's last episode succeeded), 2 health alarm, 3 paused, 4 no data."""
    if not nums or not connected:
        return [4] * n
    st = np.zeros(n, np.int64)
    st[(nums["success"] > 0) | (nums["ep_success"] > 0)] = 1
    st[nums["paused"] > 0] = 3
    st[nums["health"] > 0] = 2
    return st.tolist()
