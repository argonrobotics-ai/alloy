# SPDX-License-Identifier: Apache-2.0
"""PPO on Alloy with the live layer: the test run of JOBLIVE (docs/notes/live-test.md).

The trainer is examples/envs/04_train_ppo.py's compact PPO, with a small CNN on the policy camera next to the state
vector. The scene is GripperLift (examples/envs) with a cable on the ground in view: the soft object.

    python tools/live_ui/train.py --worlds 64 --hub <hub-host>:47100 --worker vm1 --run cable-lift
    python tools/live_ui/train.py --worlds 64 --no-live                 # the same run without the live layer
    python tools/live_ui/train.py --ab-every 10 ...                       # live on / off in alternating windows
    python tools/live_ui/train.py --replay live_logs/run/commands_vm1.jsonl ...   # re-apply a command log

Prints one JSON line per iteration (samples/s, env steps/s, episode return, success, whether live was on).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy.live import LiveConfig, LiveWorker, Settings  # noqa: E402
from ppo import PPO, ActorCritic  # noqa: E402
from task import build_env  # noqa: E402


def add_settings(settings: Settings, env, ppo: PPO, level: list[float]) -> None:
    """The whitelist: reward term weights, curriculum level, action noise, entropy, learning rate."""
    terms = env.rewarder.terms
    for i, (name, (w0, _check)) in enumerate(terms):
        if name == "effort":
            continue

        def get(i=i):
            return float(env.rewarder.terms[i][1][0])

        def put(v, i=i):
            n, (_w, c) = env.rewarder.terms[i]
            env.rewarder.terms[i] = (n, (float(v), c))
        hi = max(abs(float(w0)) * 4, 1.0)
        settings.add(f"reward.{name}", get, put, lo=0.0, hi=hi, label=f"Reward: {name}")
    settings.add("curriculum", lambda: level[0], lambda v: level.__setitem__(0, float(v)), lo=0.02, hi=0.15,
                 label="Lift goal (m)")
    settings.add("action_noise", lambda: ppo.noise_scale, lambda v: setattr(ppo, "noise_scale", float(v)),
                 lo=0.1, hi=5.0, label="Action noise")
    settings.add("entropy", lambda: ppo.entropy_coef, lambda v: setattr(ppo, "entropy_coef", float(v)),
                 lo=0.0, hi=0.05, label="Entropy")
    settings.add("lr", lambda: ppo.lr, ppo.set_lr, lo=1e-5, hi=3e-3, label="Learning rate")


def state_hash(env, net) -> str:
    """A hash of the physics state and the policy weights: equal hashes = an exact replay."""
    h = hashlib.sha256()
    st = env.sim.engine.state
    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
        arr = getattr(st, name, None)
        if arr is not None:
            h.update(arr.numpy().tobytes())
    for p in net.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main() -> None:
    a = parse()
    if a.camera <= 0:
        raise SystemExit("--camera must be > 0: the policy reads the camera image")
    torch.manual_seed(a.seed)
    level = [0.08]
    env = build_env(a.worlds, a.camera, level, device=a.device, seed=a.seed)
    net = ActorCritic(env.single_observation_space["state"].shape[0], env.single_action_space.shape[0],
                      a.camera).to(env.xp.tdevice)
    ppo = PPO(net, env, horizon=a.horizon, lr=a.lr)
    settings = Settings()
    add_settings(settings, env, ppo, level)
    live = None
    if a.hub or a.replay:
        cfg = LiveConfig(hub=a.hub or "127.0.0.1:1", worker=a.worker, run=a.run, replay=a.replay,
                         log_dir=a.log_dir, video_fps=a.fps)
        live = LiveWorker(env, cfg, settings)
    obs, _ = env.reset(seed=a.seed)
    out = open(a.out, "a") if a.out else None  # noqa: SIM115
    t_start = time.perf_counter()
    for it in range(1, a.iters + 1):
        live_on = live is not None and not (a.ab_every and ((it - 1) // a.ab_every) % 2 == 1)
        if live is not None:
            live.enabled = live_on
        t0 = time.perf_counter()
        batch, obs, stats = ppo.rollout(obs, live if live_on else None)
        t_roll = time.perf_counter() - t0
        ppo.update(batch, live.active if live_on else None)
        t_it = time.perf_counter() - t0
        row = {"iter": it, "live": live_on, "env_steps_per_s": round(a.horizon / t_roll, 2),
               "samples_per_s": round(a.horizon * a.worlds / t_it, 1), "iter_s": round(t_it, 3),
               "episodes": stats["episodes"], "return": stats["return"], "success": stats["success"],
               "noise": ppo.noise_scale, "minutes": round((time.perf_counter() - t_start) / 60, 2)}
        print(json.dumps(row), flush=True)
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        if a.minutes and time.perf_counter() - t_start > a.minutes * 60:
            break
    print(json.dumps({"done": True, "iters": it, "hash": state_hash(env, net),
                      "live_steps": live.step_count if live else 0}), flush=True)
    if live is not None:
        live.close()
    env.close()


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--minutes", type=float, default=0.0, help="stop after this many minutes (0: run all iters)")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--camera", type=int, default=128, help="policy camera size (px)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--hub", default="")
    p.add_argument("--no-live", dest="hub", action="store_const", const="")
    p.add_argument("--worker", default="vm1")
    p.add_argument("--run", default="cable-lift")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--replay", default=None)
    p.add_argument("--log-dir", default="live_logs")
    p.add_argument("--ab-every", type=int, default=0, help="alternate live on/off every N iterations")
    p.add_argument("--out", default="")
    return p.parse_args()


if __name__ == "__main__":
    main()
