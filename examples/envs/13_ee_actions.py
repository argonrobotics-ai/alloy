# SPDX-License-Identifier: Apache-2.0
"""End-effector actions: a hand-written controller moves a Panda's hand to each world's target in task space.

Run: python examples/envs/13_ee_actions.py [num_envs] [device]     (default 8 worlds, GPU if there is one; CPU: 4 cpu)
The action is the hand's motion (dx, dy, dz, 2 cm per step at action 1); the env turns it into joint targets by
damped least squares on the arm's Jacobian. A proportional controller `action = clip(k * (target - hand) / delta)`
drives every world, with no learning. Prints one JSON line: the share of worlds whose hand ends within 3 cm of the
target and the mean final distance. The first build compiles kernels (a few minutes on a fresh machine).
"""
import json
import sys

import numpy as np

from alloy.envs.diffik import panda_reach_ee

STEPS, GAIN, DELTA = 60, 0.5, 0.02       # control steps; the controller's gain (<1: the servos lag); m per step


def to_numpy(x):
    """Any array the env returns (NumPy or torch) as NumPy."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main(num_envs: int = 8, device: str | None = None) -> dict:
    env = panda_reach_ee(num_envs=num_envs, device=device, max_steps=STEPS + 10)
    try:
        env.reset()
        for _ in range(STEPS):
            state = env.state
            error = to_numpy(state["target.pos"]) - to_numpy(state["panda/hand.pos"])
            env.step(env.xp.asarray(np.clip(GAIN * error / DELTA, -1.0, 1.0)))
        state = env.state
        dist = np.linalg.norm(to_numpy(state["target.pos"]) - to_numpy(state["panda/hand.pos"]), axis=1)
        return {"worlds": num_envs, "within_3cm": float((dist < 0.03).mean()), "mean_distance_m": float(dist.mean())}
    finally:
        env.close()


if __name__ == "__main__":
    print(json.dumps(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8, sys.argv[2] if len(sys.argv) > 2 else None)))
