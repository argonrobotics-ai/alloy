# SPDX-License-Identifier: Apache-2.0
"""The Go2 on rough ground: a terrain field with a level curriculum, and a hand-written trot on it.

Run: python examples/envs/go2_walk/rough_terrain.py     (CPU is fine; the first build compiles kernels)
Prints: the field (size, highest point of each level and kind), nine robots standing on their tiles, 6 s of an
open-loop trot at command (0.5, 0, 0): per level how far the robots got and how many fell, then a reset that applies
the curriculum: the levels before and after.
The trot is a sine gait copied from 12_go2_walk.py. It is not a trained policy and does not look at the ground, so
higher levels are expected to trip it.
Measured on an AMD EPYC 7B13 CPU, no GPU (about 40 s in all, warm kernels): the tiles cover 15 m x 15 m, with 4 m
of flat border around them; the highest points are 0 m at level 0, 0.030 / 0.075 / 0.039 m at level 1 and 0.060 /
0.150 / 0.080 m at level 2 (rough / waves / blocks). Every robot stands 0.264 m up with a feet force sum of 149.2 N.
The trot walks 2.6 to 2.7 m at level 0 (3 m asked), 1.1 to 2.6 m at level 1 and 2.0 to 2.5 m at level 2; no robot
fell or left the field in this run. The reset moved the levels from [0 0 0 1 1 1 2 2 2] to [1 1 1 2 1 0 2 2 2]:
robots that went over half a tile (2.5 m) went up one level (one from level 2 drew level 2 again), the one that
walked under half of what was asked (1.1 m of 3) went down, the others stayed. The distances vary by a few tenths
of a metre between machines and versions. sim.check ok True.
"""
import numpy as np

from alloy.envs.loco import Go2RoughConfig, TerrainSpec, go2_rough

DT = 0.02                       # control period, s
GAIT_HZ, THIGH, CALF = 2.0, -2.0, 2.5     # trot: thigh swing and calf lift, in action units (x 0.25 rad)
WORLDS, SECONDS = 9, 6.0                  # 3 levels x 3 kinds: one robot per tile


def host(x):
    """Any array the env returns (NumPy or torch) as NumPy."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def trot(t, worlds):
    """Actions (worlds, 12) at time t: diagonal pairs FL+RR and FR+RL swing half a cycle apart."""
    a = np.zeros((worlds, 12), np.float32)
    for leg, phase in enumerate((0.0, np.pi, np.pi, 0.0)):          # leg order FL, FR, RL, RR
        p = 2 * np.pi * GAIT_HZ * t + phase
        a[:, 3 * leg + 1] = THIGH * np.sin(p)
        a[:, 3 * leg + 2] = -CALF * max(0.0, np.cos(p))
    return a


def tile_rows(env):
    """(level, kind) of every world."""
    return host(env.levels).tolist(), host(env.kinds).tolist()


def print_field(env):
    t = env.terrain
    print(f"field: {t.levels * t.spec.tile:.0f} m x {t.kinds * t.spec.tile:.0f} m of tiles ({t.levels} levels x "
          f"{t.kinds} kinds) plus {t.pad * t.cell:.0f} m of flat border, {t.heights.shape} nodes")
    n, p = int(round(t.spec.tile / t.cell)), t.pad          # nodes per tile edge; the border's nodes come first
    for level in range(t.levels):
        tops = [float(t.heights[p + k * n:p + (k + 1) * n + 1, p + level * n:p + (level + 1) * n + 1].max())
                for k in range(t.kinds)]
        print(f"  level {level} highest point (m): " + ", ".join(f"{k} {h:.3f}" for k, h in zip(t.spec.kinds, tops)))


def print_standing(env):
    pos, force = host(env.state["go2/base.pos"]), host(env.state["feet.force"]).sum(1)
    levels, kinds = tile_rows(env)
    for w in range(env.num_envs):
        print(f"  world {w}: level {levels[w]} {env.terrain.spec.kinds[kinds[w]]:6s} base at "
              f"({pos[w, 0]:6.2f}, {pos[w, 1]:6.2f}, {pos[w, 2]:.3f}), feet force sum {force[w]:.1f} N")


def main():
    terrain = TerrainSpec(levels=3, tile=5.0)
    env = go2_rough(num_envs=WORLDS, config=Go2RoughConfig(pushes=None, start_levels=3, terrain=terrain),
                    autoreset="disabled")
    env.reset(seed=0)
    print_field(env)
    for _ in range(50):                                     # 1 s holding the stand
        env.step(env.xp.zeros((WORLDS, 12)))
    print("standing after 1 s:")
    print_standing(env)

    env.commands.set((0.5, 0.0, 0.0))
    start = host(env.state["go2/base.pos"])[:, :2].copy()
    levels_before = tile_rows(env)[0]
    ended, left = np.zeros(WORLDS, bool), np.zeros(WORLDS, bool)
    for i in range(int(SECONDS / DT)):
        _, _, terminated, _, _ = env.step(trot(i * DT, WORLDS))
        ended |= host(terminated).astype(bool)
        left |= ~host(env.terrain.inside(env.state["go2/base.pos"][:, 0:2])).astype(bool)   # off the field's edge
    fell = ended & ~left                                    # an episode that ended on the field: a fall
    walked = np.linalg.norm(host(env.state["go2/base.pos"])[:, :2] - start, axis=1)
    print(f"trot, command 0.5 m/s for {SECONDS:.0f} s (asks for {0.5 * SECONDS:.1f} m):")
    for level in range(terrain.levels):
        idx = [w for w in range(WORLDS) if levels_before[w] == level]
        print(f"  level {level}: walked {' '.join(f'{walked[w]:.2f}' for w in idx)} m, "
              f"fell {int(fell[idx].sum())}, left the field {int(left[idx].sum())} (of {len(idx)})")

    env.reset()                                             # the curriculum moves each world by its episode
    print("levels before the reset:", levels_before)
    print("levels after the reset: ", host(env.levels).tolist())
    print("sim.check ok:", env.sim.check()["ok"])
    env.close()


if __name__ == "__main__":
    main()
