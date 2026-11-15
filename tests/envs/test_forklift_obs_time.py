# SPDX-License-Identifier: Apache-2.0
"""A forklift port's observation carries the episode's time (audit T1-017).

``TruckControls.proprio`` observed the truck with ``observe(0.0)``, so the first value of every observation, the
episode's time in seconds, read 0 at every step.
"""
from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import make


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_the_first_observation_value_is_the_episode_time():
    env = make("ForkliftStartAndPark-v0", device="cpu")
    obs, info = env.reset(seed=0)
    times = [float(np.asarray(obs)[0, 0])]
    for _ in range(3):
        obs, _r, _d, _t, info = env.step(info["expert_action"])
        times.append(float(np.asarray(obs)[0, 0]))
    env.close()
    assert np.allclose(times, [0.0, 0.02, 0.04, 0.06], atol=1e-6)        # one 20 ms control period per step
