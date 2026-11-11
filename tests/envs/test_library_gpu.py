# SPDX-License-Identifier: Apache-2.0
"""GPU smoke of every library task: build 8 worlds, reset, step with random actions, finite observations, no world
frozen as INVALID. Needs CUDA (and the tasks' library objects fetched); slow tasks (Pour, ClothFold) take a few
minutes on the first, compiling run."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import make

TASKS = ("PickPlace-v0", "PegInsert-v0", "DrawerOpen-v0", "Pour-v0", "ClothFold-v0")


def _needs(name: str) -> str | None:
    from alloy.envs.library_tasks import CABINET, PICK_OBJECTS

    return {"PickPlace-v0": PICK_OBJECTS[0], "DrawerOpen-v0": CABINET}.get(name)


@pytest.mark.gpu
@pytest.mark.parametrize("name", TASKS)
def test_library_task_gpu_smoke(name):
    torch = pytest.importorskip("torch")
    from alloy.api import catalog

    card = _needs(name)
    if card is not None and not catalog.can_place(card):
        pytest.skip(f"{card} is not available here (alloy fetch {card})")
    env = make(name, num_envs=8, device="cuda:0")
    try:
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(3):
            a = torch.as_tensor(rng.uniform(-1, 1, (8, env.writer.size)), dtype=torch.float32, device="cuda")
            obs, r, term, trunc, info = env.step(a)
        assert obs.is_cuda and torch.isfinite(obs).all() and torch.isfinite(r).all()
        assert not info["invalid"].any() and set(info["terms"]) == set(dict(env.task.reward_terms()))
    finally:
        env.close()
