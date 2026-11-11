# SPDX-License-Identifier: Apache-2.0
"""Replays recorded demonstrations and measures how often a scripted policy, clean and noisy, lifts the cube.

Needs h5py: pip install h5py
Run:  python 03_replay_evaluate.py [--data FILE] [--out alloy_demos] [--worlds 4] [--episodes 16] [--noise 0.3]
Without --data it first records 4 scripted episodes into <out>/replay_demo.hdf5; --data takes a file written by
01_record_demos.py instead. Prints (1) how many recorded episodes the replay reproduced and the
largest state difference, (2) a table of success rate with a 95 % interval for the clean and the noisy policy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py  # noqa: F401  (pip install h5py)
import numpy as np

import alloy.envs
from alloy.api.errors import AlloyAPIError
from alloy.il import EvalResult, HDF5Writer, Recorder, evaluate, load_hdf5, replay

SQUEEZE_STEPS = 25            # close both fingers for this many steps, then lift while holding


def scripted_policy(env: Any) -> Any:
    """A policy(obs) that squeezes, then lifts (the one that recorded the demonstrations)."""
    def policy(_obs: Any) -> Any:
        steps = env.xp.to_numpy(env.state["steps"])
        actions = np.ones((env.num_envs, 3), np.float32)
        actions[steps < SQUEEZE_STEPS, 0] = 0.0
        return env.xp.asarray(actions)
    return policy


def noisy_policy(env: Any, noise: float, seed: int = 0) -> Any:
    """The scripted policy plus uniform noise in [-noise, noise] on every action, clipped to [-1, 1]."""
    rng, base = np.random.default_rng(seed), scripted_policy(env)

    def policy(obs: Any) -> Any:
        actions = env.xp.to_numpy(base(obs)) + rng.uniform(-noise, noise, (env.num_envs, 3))
        return env.xp.asarray(np.clip(actions, -1.0, 1.0).astype(np.float32))
    return policy


def record_demos(env: Any, path: Path, episodes: int = 4) -> None:
    """Record `episodes` successful scripted episodes into `path`."""
    policy = scripted_policy(env)
    with Recorder(env, [HDF5Writer(path, overwrite=True)], task="lift the cube and hold it") as rec:
        rec.reset(seed=0)
        while rec.saved < episodes:
            rec.step(policy(None))


def print_replay(env: Any, episodes: list[Any]) -> None:
    s = replay(env, episodes).summary()
    print(f"replay: {s['matches']} of {s['episodes']} episodes reproduced the recorded outcome "
          f"({s['success']} succeeded, {s['recorded_success']} had succeeded when recorded)")
    print(f"        largest state difference {s['max_state_error']:.3g} (observation {s['max_obs_error']:.3g})")
    print("        exactly 0 is expected only on the CPU, with the recording's world count and seed")


def table_row(label: str, result: EvalResult) -> str:
    lo, hi = result.interval
    return f"{label:<22}{result.success_rate:>8.0%}   [{lo:.0%}, {hi:.0%}]   {result.episodes} episodes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=None, help="HDF5 file from 01_record_demos.py (default: record 4 episodes)")
    p.add_argument("--out", default="alloy_demos", help="where the recorded file goes (default ./alloy_demos)")
    p.add_argument("--worlds", type=int, default=4, help="parallel worlds (default 4)")
    p.add_argument("--device", default=None, help="cpu or cuda:0 (default: automatic)")
    p.add_argument("--episodes", type=int, default=16, help="episodes per evaluation (default 16)")
    p.add_argument("--noise", type=float, default=0.3, help="uniform action noise of the noisy policy")
    return p.parse_args()


def main(args: argparse.Namespace) -> int:
    if args.data and not Path(args.data).exists():
        print(f"{args.data} not found. Record it first:  python 01_record_demos.py")
        return 1
    if args.data:
        path = Path(args.data)
        spec, episodes = load_hdf5(path)
        env = alloy.envs.make(spec.env, num_envs=args.worlds, device=args.device)
    else:
        path = Path(args.out) / "replay_demo.hdf5"          # its own file: 01_record_demos.py writes lift.hdf5
        path.parent.mkdir(parents=True, exist_ok=True)
        env = alloy.envs.make("GripperLift-v0", num_envs=args.worlds, device=args.device)
        record_demos(env, path)
        print(f"no --data given: recorded 4 scripted episodes into {path.resolve()}")
        spec, episodes = load_hdf5(path)
    print(f"loaded {len(episodes)} episodes of {spec.env}, replaying on {env.device} with {args.worlds} worlds")
    print_replay(env, episodes)
    clean = evaluate(env, scripted_policy(env), episodes=args.episodes, seed=1)
    noisy = evaluate(env, noisy_policy(env, args.noise), episodes=args.episodes, seed=1)
    print(f"{'policy':<22}{'success':>8}   95% interval")
    print(table_row("scripted", clean))
    print(table_row(f"scripted + noise {args.noise:g}", noisy))
    env.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(parse_args()))
    except AlloyAPIError as e:
        print(f"error: {e.message}\nfix: {e.fix}")
        sys.exit(1)
