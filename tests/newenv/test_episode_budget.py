# SPDX-License-Identifier: Apache-2.0
"""Every request's action phases fit in one episode of its ``episode_s``.

The driver resets only where a phase says so, and an agent's env resets itself when ``episode_steps`` is reached;
a phase that outlasts the episode is cut by that reset and its checks read across it (the pilot's E15 failures).
"""
from __future__ import annotations

import importlib.util

import pytest

from bench.newenv.make_task import SCRIPTS
from bench.newenv.requests import load_requests

MARGIN_S = 0.5      # the last row of a phase must not land on the episode's own end


def _phases(checker: str) -> list[dict]:
    spec = importlib.util.spec_from_file_location(f"script_{checker}", SCRIPTS / f"{checker}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(getattr(mod, "PHASES", []))


def longest_since_reset(rest_s: float, phases: list[dict]) -> float:
    """Seconds since the last reset at the end of each phase, worst case (the driver resets before 'rest').

    >>> longest_since_reset(2.0, [{"seconds": 1.0}, {"seconds": 3.0, "reset": True}, {"seconds": 2.5}])
    5.5
    """
    t = worst = rest_s
    for ph in phases:
        t = float(ph["seconds"]) if ph.get("reset") else t + float(ph["seconds"])
        worst = max(worst, t)
    return worst


@pytest.mark.parametrize("req", load_requests(), ids=lambda r: r.id)
def test_phases_fit_in_one_episode(req):
    episode = float(req.entry["episode_s"])
    worst = longest_since_reset(float(req.entry.get("rest_s", 2.0)), _phases(req.checker))
    assert worst <= episode - MARGIN_S, f"{req.id}: {worst:g} s since the last reset, episode_s {episode:g}"
