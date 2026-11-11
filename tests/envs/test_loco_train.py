# SPDX-License-Identifier: Apache-2.0
"""The Go2 training module's parts that need no simulation, PyTorch or rsl_rl: the runner settings and the
reporting tally (fed hand-made stand-ins for tensors)."""

from __future__ import annotations

import numpy as np

from alloy.envs.loco.train import STEPS_PER_ENV, _Tally, ppo_config


class _T(np.ndarray):
    """A NumPy array with the one tensor method the tally calls (`.float()`)."""

    def float(self) -> _T:
        return self.astype(np.float64).view(_T)


def _t(values) -> _T:
    return np.asarray(values).view(_T)


def test_ppo_config_has_the_documented_settings() -> None:
    """24 steps per env, the requested iterations and seed, a 512-256-128 actor and critic, scalar-std Gaussian."""
    cfg = ppo_config(1000, 7)
    assert cfg["num_steps_per_env"] == STEPS_PER_ENV == 24 and cfg["max_iterations"] == 1000 and cfg["seed"] == 7
    assert cfg["actor"]["hidden_dims"] == [512, 256, 128] and cfg["critic"]["hidden_dims"] == [512, 256, 128]
    assert cfg["actor"]["distribution_cfg"]["std_type"] == "scalar"
    assert cfg["algorithm"]["num_learning_epochs"] == 5 and cfg["algorithm"]["entropy_coef"] == 0.01


def test_tally_averages_terms_and_counts_ended_episodes() -> None:
    """Two steps: term means over steps; one episode ended in a fall, one by the time limit; `take` resets."""
    tally = _Tally()
    tally.add(_t([False, False]), {"terms": {"a": _t([1.0, 3.0])}})
    done = np.array([True, True])
    info = {"terms": {"a": _t([5.0, 7.0])}, "_episode": done,
            "episode": {"r": _t([2.0, 4.0]), "l": _t([10, 30])}}
    tally.add(np.array([True, False]), info)
    row = tally.take()
    assert row["terms"] == {"a": 4.0} and row["episodes"] == 2 and row["fell_share"] == 0.5
    assert row["episode_return"] == 3.0 and row["episode_steps"] == 20.0
    empty = tally.take()
    assert empty["episodes"] == 0 and empty["terms"] == {}
