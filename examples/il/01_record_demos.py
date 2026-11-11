# SPDX-License-Identifier: Apache-2.0
"""Records scripted demonstrations of a gripper lifting a cube and saves them as a robomimic-style HDF5 file.

Needs h5py: pip install h5py
Run:  python 01_record_demos.py [--worlds 4] [--episodes 8] [--camera] [--lerobot] [--out alloy_demos]
Prints: episodes saved and dropped, steps and wall time, the file written, and what the file holds when read back,
then the command of the next example. With --camera the cube is also seen by a 64x64 camera ("front"); the first
camera run compiles kernels (minutes on a CPU), later runs start in seconds. --lerobot also writes a LeRobotDataset
v3 folder (needs the lerobot package).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import h5py  # noqa: F401  (pip install h5py)
import numpy as np

import alloy.envs
from alloy.api.errors import AlloyAPIError
from alloy.il import HDF5Writer, LeRobotWriter, Recorder, load_hdf5

SQUEEZE_STEPS = 25            # close both fingers on the cube for this many steps, then lift while holding


def scripted_actions(env: Any) -> np.ndarray:
    """Actions (lift, close_left, close_right) for every world: squeeze first, then lift."""
    steps = env.xp.to_numpy(env.state["steps"])
    actions = np.ones((env.num_envs, 3), np.float32)
    actions[steps < SQUEEZE_STEPS, 0] = 0.0
    return env.xp.asarray(actions)


def record(env: Any, writers: list[Any], episodes: int, keep: str) -> tuple[Recorder, int]:
    """Run the scripted policy until `episodes` episodes are kept; returns the closed recorder and the steps taken."""
    steps = 0
    with Recorder(env, writers, task="lift the cube and hold it", keep=keep) as rec:
        rec.reset(seed=0)
        while rec.saved < episodes:
            rec.step(scripted_actions(env))
            steps += 1
    return rec, steps


def show_file(path: Path) -> None:
    """Re-open the file and print what a training script would find in it."""
    spec, episodes = load_hdf5(path)
    first = episodes[0]
    print(f"read back {path}: {len(episodes)} episodes, env {spec.env}")
    print("  first episode obs:", {k: v.shape for k, v in first.obs.items()})
    print(f"  first episode actions: {first.actions.shape}, success: {first.success}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--worlds", type=int, default=4, help="parallel worlds (default 4)")
    p.add_argument("--episodes", type=int, default=8, help="episodes to keep (default 8)")
    p.add_argument("--device", default=None, help="cpu or cuda:0 (default: automatic)")
    p.add_argument("--camera", action="store_true", help="also record a 64x64 camera image (slow first compile)")
    p.add_argument("--out", default="alloy_demos", help="output directory (default ./alloy_demos)")
    p.add_argument("--lerobot", action="store_true", help="also write a LeRobot dataset to <out>/lift_lerobot")
    p.add_argument("--keep", default="success", choices=["success", "all"], help="which episodes to keep")
    return p.parse_args()


def main(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "lift.hdf5"
    writers: list[Any] = [HDF5Writer(path, overwrite=True)]
    if args.lerobot:
        writers.append(LeRobotWriter(out / "lift_lerobot", overwrite=True))
    name = "GripperLiftCamera-v0" if args.camera else "GripperLift-v0"
    env = alloy.envs.make(name, num_envs=args.worlds, device=args.device)
    t0 = time.perf_counter()
    rec, steps = record(env, writers, args.episodes, args.keep)
    wall = time.perf_counter() - t0
    print(f"recorded {rec.saved} episodes ({rec.dropped} dropped) from {name} on {env.device}, "
          f"{args.worlds} worlds, {steps} steps, {wall:.1f} s")
    print(f"wrote {path.resolve()}")
    if args.lerobot:
        print(f"wrote {(out / 'lift_lerobot').resolve()} (LeRobotDataset v3)")
    show_file(path)
    env.close()
    print(f"next: python 03_replay_evaluate.py --data {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(parse_args()))
    except AlloyAPIError as e:
        print(f"error: {e.message}\nfix: {e.fix}")
        sys.exit(1)
