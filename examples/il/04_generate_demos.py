# SPDX-License-Identifier: Apache-2.0
"""Generates scripted demonstrations at scale: every world runs the task's shipped demonstrator until enough
successful episodes are saved.

Run:  python 04_generate_demos.py [--env GripperLift-v0] [--episodes 64] [--worlds 16] [--noise 0.1]
                                  [--out alloy_demos] [--lerobot] [--device cuda:0]
Writes <out>/generated.hdf5 (robomimic layout; needs ``pip install h5py``) and, with --lerobot, a LeRobotDataset in
<out>/generated_lerobot (needs ``pip install 'lerobot[dataset]'``). Prints progress and a summary: episodes saved,
episodes tried, the script's success rate with its 95 % interval, episodes and world-steps per second.
The default (64 episodes, 16 worlds) takes about a minute on a CPU after the first build; use a GPU and more worlds
for thousands: ``--episodes 10000 --worlds 1024``. Shipped demonstrators: GripperLift-v0, GripperLiftCamera-v0,
PickPlace-v0 (its first build fetches the object's files and needs the ``import`` extra).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py  # noqa: F401  (HDF5Writer needs it; importing it here gives a plain ImportError when it is missing)

import alloy.envs
from alloy.il import HDF5Writer, LeRobotWriter, demonstrator, generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env", default="GripperLift-v0", help="an env with a shipped demonstrator")
    p.add_argument("--episodes", type=int, default=64, help="successful episodes to save (default 64)")
    p.add_argument("--worlds", type=int, default=16, help="parallel worlds (default 16)")
    p.add_argument("--noise", type=float, default=0.1, help="uniform action noise, 0 to 1 (default 0.1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="alloy_demos", help="output folder (default ./alloy_demos)")
    p.add_argument("--lerobot", action="store_true", help="also write a LeRobotDataset v3.0")
    p.add_argument("--device", default=None, help="cpu or cuda:0 (default: the GPU if there is one)")
    return p.parse_args()


def main(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = alloy.envs.make(args.env, num_envs=args.worlds, device=args.device)
    writers = [HDF5Writer(out / "generated.hdf5", overwrite=True)]
    if args.lerobot:
        writers.append(LeRobotWriter(out / "generated_lerobot", repo_id="local/generated", overwrite=True))
    policy = demonstrator(env, noise=args.noise, seed=args.seed)
    print(f"{args.env}: {args.worlds} worlds on {env.device}, generating {args.episodes} episodes")
    result = generate(env, policy, writers, args.episodes, seed=args.seed, task=f"scripted {args.env}",
                      progress=lambda r: print(f"  {r.saved} / {r.wanted} saved, {r.attempted} tried, "
                                               f"{r.wall_s:.0f} s", flush=True))
    lo, hi = result.interval
    print(f"saved {result.saved} of {result.wanted} episodes to {(out / 'generated.hdf5').resolve()}")
    print(f"script success {result.success_rate:.0%} [{lo:.0%}, {hi:.0%}] over {result.attempted} episodes; "
          f"{result.episodes_per_s:.1f} episodes/s, {result.world_steps_per_s:.0f} world-steps/s")
    (out / "generated.json").write_text(json.dumps(result.summary(), indent=1))
    env.close()
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
