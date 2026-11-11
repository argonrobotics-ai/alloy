# SPDX-License-Identifier: Apache-2.0
"""Target discovery: an unknown name imports every plug-in package (contrib included) before failing (SCR-WP9-5)."""

from __future__ import annotations

import pytest

from alloy.bully import targets
from alloy.core.errors import RegistryError
from alloy.testing.bully import stable_fake_target


def test_unknown_target_loads_plugins_including_contrib(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_load_plugins(*, include_contrib: bool = False) -> list[str]:
        calls.append(include_contrib)
        targets.register_target("plugin_scene_under_test", stable_fake_target)
        return ["alloy.solvers.fake"]

    monkeypatch.setattr(targets, "load_plugins", fake_load_plugins)
    monkeypatch.delitem(targets._TARGETS, "plugin_scene_under_test", raising=False)
    try:
        assert targets.get_target("plugin_scene_under_test").name
        assert calls == [True]
    finally:
        targets._TARGETS.pop("plugin_scene_under_test", None)


def test_unknown_target_still_raises_after_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(targets, "load_plugins", lambda *, include_contrib=False: [])
    with pytest.raises(RegistryError, match="no bully target"):
        targets.get_target("no_such_scene")


def test_cli_stdout_is_one_json_record(monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]) -> None:
    """Library prints during a command (e.g. MuJoCo Warp overflow warnings) go to stderr, not into the record."""
    import json

    from alloy.bully import __main__ as cli

    def noisy_names() -> list[str]:
        print("nefc overflow - please increase njmax")
        return ["a"]

    monkeypatch.setattr(cli, "target_names", noisy_names)
    assert cli.main(["targets"]) == 0
    out, err = capfd.readouterr()
    assert json.loads(out) == {"targets": ["a"]}
    assert "nefc overflow" in err


def test_cli_native_prints_reach_stderr(monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]) -> None:
    """Writes straight to file descriptor 1 (as kernel printf does) also stay out of the record."""
    import json
    import math
    import os

    from alloy.bully import __main__ as cli

    def native_noise() -> list[str]:
        os.write(1, b"kernel printf noise\n")
        return ["b"]

    monkeypatch.setattr(cli, "target_names", native_noise)
    assert cli.main(["targets"]) == 0
    out, err = capfd.readouterr()
    assert json.loads(out) == {"targets": ["b"]}
    assert "kernel printf noise" in err


def test_sb7_lists_each_s1_episode_and_its_trace_replays(monkeypatch: pytest.MonkeyPatch,
                                                         capfd: pytest.CaptureFixture[str]) -> None:
    """SB-7 names every S1 episode (seed, world): that world's random-legal trace, run alone, trips S1 again."""
    import json
    import math

    from alloy.bully import __main__ as cli
    from alloy.bully.replay import EpisodeRunner
    from alloy.bully.stress import random_legal_traces
    from alloy.bully.types import BullyLimits
    from alloy.testing.bully import SeededSpec, seeded_fake_target

    monkeypatch.setitem(targets._TARGETS, "fake_touchy", lambda: seeded_fake_target(SeededSpec(blowup_speed=0.3),
                                                                                    name="fake_touchy"))
    args = ["sb7", "--target", "fake_touchy", "--episodes", "24", "--worlds", "16", "--horizon", "0.5", "--seed", "3"]
    assert cli.main(args) == 0
    rec = json.loads(capfd.readouterr().out)
    trips, listed = rec["rate"]["trips"], rec["s1_worlds"]
    assert trips > 0 and len(listed) == trips == sum(rec["s1_causes"].values())
    assert sum(rec["s2_causes"].values()) == rec["s2_episodes"]
    assert 0 <= rec["s2_max_pen_episodes"] <= rec["s2_episodes"]
    assert all(e["seed"] in (3, 4) and (e["seed"] == 3 or e["world"] < 8) for e in listed)
    # the amended row's S1 episodes are a subset of the old D12 row's; the rest are listed as D12-only
    assert rec["rate_d12"]["trips"] == trips + len(rec["d12_only_worlds"])
    from alloy.bench.sb7 import POPULATION                 # audit AB-laws-7: the rate carries the population count

    assert f"changed {len(POPULATION)} times" in rec["population"], rec["population"]
    assert all(e["s1_d12"] for e in listed)
    target, pick = targets.get_target("fake_touchy"), listed[-1]
    trace = random_legal_traces(target, 16, 0.5, seed=pick["seed"])[pick["world"]]
    assert EpisodeRunner(target, 1, "cpu", BullyLimits()).run(trace[None]).severity[0] == 1


def test_sb7_d12_count_keeps_free_fall_speed_trips() -> None:
    """``rate_d12`` re-classifies with the pre-SCR-WP7-7 speed row: a free fall past v_max (amended value 4, D12 value
    11) is S1 only there; a contact kick (11 under both) and other S1 rows are S1 under both."""
    import numpy as np

    from alloy.bully.__main__ import _d12_severity
    from alloy.bully.severity import classify
    from alloy.bully.types import BullyLimits, EpisodeResult
    from alloy.precision.types import HealthReport

    mon = {"impossible_speed": np.array([4.0, 11.0, 3.0]), "impossible_speed_d12": np.array([11.0, 11.0, 3.0]),
           "pen_s1_ratio": np.array([0.0, 0.0, 2.0])}
    res = EpisodeResult(np.ones(3, np.int32), np.zeros(3, np.int64), mon, HealthReport.ok(3), np.zeros(3))
    assert _d12_severity(res, BullyLimits()).tolist() == [1, 1, 1]
    assert classify(res.health, mon, BullyLimits()).tolist() == [0, 1, 1]
