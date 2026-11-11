# SPDX-License-Identifier: Apache-2.0
"""When the live demo stops itself: a hard limit after start, and an idle limit once someone has watched.

The hub writes the viewer state (``ViewerClock.to_json``) every few seconds; a supervisor calls :func:`stop_reason`
and tears the demo down (VMs deleted) when it returns a reason.

- Hard stop: ``hard_s`` after the demo started, whatever happens.
- Idle stop: once the first viewer has connected, ``idle_s`` with no viewer connected. Time before the first viewer
  does not count.
- A hub that stopped writing its state counts as no viewer since its last write.

Example::

    stop_reason(now=t, started=t0, state={"viewers": 0, "first_viewer": t1, "last_viewer": t2, "t": t},
                hard_s=4 * 3600, idle_s=3600)        # -> None, "hard limit 4.0 h" or "idle 60 min"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ViewerClock:
    """The hub's count of connected viewers and when the first and last were seen."""
    viewers: int = 0
    first_viewer: float | None = None
    last_viewer: float | None = None

    def connect(self, now: float) -> None:
        self.viewers += 1
        self.first_viewer = self.first_viewer or now
        self.last_viewer = now

    def disconnect(self, now: float) -> None:
        self.viewers = max(0, self.viewers - 1)
        self.last_viewer = now

    def to_json(self, now: float) -> dict[str, Any]:
        if self.viewers:
            self.last_viewer = now
        return {"t": now, "viewers": self.viewers, "first_viewer": self.first_viewer, "last_viewer": self.last_viewer}


def stop_reason(now: float, started: float, state: dict[str, Any] | None, hard_s: float,
                idle_s: float) -> str | None:
    if now - started >= hard_s:
        return f"hard limit {hard_s / 3600:.1f} h"
    if not state or state.get("first_viewer") is None:
        return None
    viewers, last = int(state.get("viewers", 0)), state.get("last_viewer") or state["first_viewer"]
    if viewers > 0:
        last = max(last, float(state.get("t", last)))      # viewers at the hub's last write
        viewers_now = now - float(state.get("t", now)) < 120  # a silent hub has no viewers
        if viewers_now:
            return None
    if now - last >= idle_s:
        return f"idle {idle_s / 60:.0f} min"
    return None
