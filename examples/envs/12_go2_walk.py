# SPDX-License-Identifier: Apache-2.0
"""A tour of the Go2 locomotion environment: observations, a held stand, a push, and a hand-written trot.

Run: python examples/envs/12_go2_walk.py [num_envs]     (default 4; CPU is fine; the first build compiles kernels)
Prints: the observation groups, 1 s of standing under a forward command (height, contacts, reward terms), the speed
a 40 N push gives the base, then 4 s of an open-loop trot at command (0.5, 0, 0): distance, speed, falls, health.
The trot is a hand-written sine gait that shows the action interface. It is not a trained policy.
Rough numbers on an AMD EPYC 7B13 CPU, 4 worlds (about 21 s in all): after 1 s of stand the base
is 0.264 m up (the servos sag from the 0.288 m of the reset pose) with 4 feet down; the 40 N push gives about 0.14
m/s, about half of F t / m = 0.26 m/s (the legs and the ground resist); the trot walks about 1.93 m in 4 s (0.48
m/s), 0 falls, health ok. Training: python -m alloy.envs.loco.train, docs/guide/locomotion.md.
"""
import sys

import numpy as np

from alloy.envs.loco import Go2VelocityConfig, go2_velocity

DT = 0.02                       # control period, s
GAIT_HZ, THIGH, CALF = 2.0, -2.0, 2.5     # trot: thigh swing and calf lift, in action units (x 0.25 rad)


def np_(x):
    """Any array the env returns (NumPy or torch) as NumPy."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def trot(t, worlds):
    """Actions (worlds, 12) at time t: diagonal pairs FL+RR and FR+RL swing half a cycle apart."""
    a = np.zeros((worlds, 12), np.float32)
    for leg, phase in enumerate((0.0, np.pi, np.pi, 0.0)):          # leg order FL, FR, RL, RR
        p = 2 * np.pi * GAIT_HZ * t + phase
        a[:, 3 * leg + 1] = THIGH * np.sin(p)                       # thigh joint: negative swings the foot forward
        a[:, 3 * leg + 2] = -CALF * max(0.0, np.cos(p))             # calf joint: bend the knee to lift the foot
    return a


def feet_centre(env):
    return np_(env.state["feet.pos"]).mean(1)[:, :2]


def main():
    worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    env = go2_velocity(num_envs=worlds, config=Go2VelocityConfig(pushes=None))   # no random pushes in this tour
    env.reset(seed=0)
    for group in env.obs_groups:
        names = env.obs_names(group)
        print(f"observation group {group}: {len(names)} numbers, first {names[:3]}")

    env.commands.set((0.5, 0.0, 0.0))                       # the same command in every world
    zeros = np.zeros((worlds, 12), np.float32)
    for _ in range(50):                                     # 1 s holding the stand
        _, _, terminated, _, info = env.step(zeros)
    print(f"stand: height {np_(env.state['base.height']).mean():.3f} m, "
          f"feet down {np_(env.state['feet.contact']).sum(1).mean():.1f} of 4, "
          f"terminated {int(np_(terminated).sum())}")
    print("reward terms (world 0):", {k: round(float(np_(v)[0]), 3) for k, v in info["terms"].items()})

    force = 40.0                                            # N along x, world frame, for 5 steps = 0.1 s
    v0 = np_(env.state["base.lin_vel_w"])[:, 0].copy()
    env.pusher.hold(np.tile([force, 0.0, 0.0], (worlds, 1)).astype(np.float32))
    for _ in range(5):
        env.step(zeros)
    env.pusher.clear()
    dv = float((np_(env.state["base.lin_vel_w"])[:, 0] - v0).mean())
    print(f"push: {force:.0f} N for 0.1 s changed the base x speed by {dv:.2f} m/s "
          f"(F t / m = {force * 0.1 / 15.2:.2f} m/s for the robot's 15.2 kg)")

    env.reset()                                             # back to the start state, same fixed command
    start = feet_centre(env)
    falls = 0
    for i in range(200):                                    # 4 s of trot
        _, _, terminated, _, _ = env.step(trot(i * DT, worlds))
        falls += int(np_(terminated).sum())
    walked = (feet_centre(env) - start)[:, 0]
    print(f"trot: walked {walked.mean():.2f} m in 4 s (mean speed {walked.mean() / 4:.2f} m/s, command 0.5), "
          f"falls {falls}, sim.check ok {env.sim.check()['ok']}")
    env.close()
    print("Train a policy: python -m alloy.envs.loco.train --num-envs 4096 --iterations 400 "
          "(guide: docs/guide/locomotion.md)")


if __name__ == "__main__":
    main()
