# SPDX-License-Identifier: Apache-2.0
"""A carriage port's other action changes only that tool in that world, and the expert's action stays the script's
(audit T1-015).

``CarriageCommand.write`` replaced every world's and every tool's program with a two-key ramp as soon as one value of
one tool differed from the expert's, and ``read`` (``info["expert_action"]``) then read those ramps back: a nudge of
world 0's first tool cut world 1's 80-key program to 2 keys, so world 1, stepped with its expert action, no longer
followed the script.
"""
from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import make

NAME, STEPS = "ConcretePatchFlush-v0", 3


def _carriage(env):  # noqa: ANN001, ANN202
    return next(p for p in env._slots[0].mover.parts if type(p).__name__ == "CarriageCommand")


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_a_nudged_tool_leaves_the_other_tools_and_worlds_on_the_script():
    env, ref = make(NAME, num_envs=2, device="cpu"), make(NAME, num_envs=2, device="cpu")
    obs, info = env.reset(seed=0)
    obs_ref, info_ref = ref.reset(seed=0)
    keys = _carriage(env).c.n.numpy().copy()                             # (W * T,) program keys per tool
    for _ in range(STEPS):
        assert np.array_equal(info["expert_action"], info_ref["expert_action"])     # the script's, nudged or not
        a = info["expert_action"].copy()
        a[0, :3] += np.float32(0.01)                                     # world 0, first tool: 1 cm off its path
        obs, _r, _t, _tr, info = env.step(a)
        obs_ref, _r, _t, _tr, info_ref = ref.step(info_ref["expert_action"])
    n = _carriage(env).c.n.numpy()
    assert n[0] == 2 and np.array_equal(n[1:], keys[1:])                # only world 0's first tool took the ramp
    assert np.array_equal(info["expert_action"], info_ref["expert_action"])
    assert np.array_equal(np.asarray(obs)[1], np.asarray(obs_ref)[1])    # world 1 follows the script, as the port
    assert not np.array_equal(np.asarray(obs)[0], np.asarray(obs_ref)[0])
    env.close()
    ref.close()
