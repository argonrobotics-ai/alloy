# SPDX-License-Identifier: Apache-2.0
"""Train a policy on GripperLift-v0 with a compact PPO, entirely on the GPU (needs PyTorch).

Alloy gives the environment; this file is a minimal trainer to show it learns. For rsl_rl, Stable-Baselines3 or
skrl use alloy.envs.wrappers (examples/envs/11_train_rl.py, docs/guide/rl.md); for another RL library, write a short
adapter to its env class: the env has the shape of Gymnasium's vector API and returns torch tensors on the
simulation device. Success = the cube is up 8 cm and held when
the episode ends; every episode starts with the cube moved and turned a little. On one L4 GPU with 512 worlds most
episodes succeed after under a minute of training (about 97 % by iteration 100, 99 % after 200: 4.5 min with the
first-run compile); PandaReach-v0 with 1,024 worlds reaches ~97 % in about 4 minutes; DrawerOpen-v0 with 256 worlds
opens 15 cm in ~27 % of episodes after 80 iterations (2.6 min).
What this trainer does not learn: PickPlace-v0 stayed at 0 % success with 1,024 worlds on one L4 in two runs of 21
and 30 min (the longer: 180 iterations, 5.9 M samples; its "held" term rose, "lift" never did), and the env slows
about 4x once the fingers touch the scanned object. Picking an object up and carrying it to a bin is a long, sparse
task: it needs reward shaping, a curriculum or demonstrations (its scripted policy, examples/envs/05_pick_place.py,
puts the object in the bin in 60 to 62 of 64 worlds on one L4), and far more samples than this file's defaults.
Use PickPlace-v0 with your own trainer.
Run: python examples/envs/04_train_ppo.py [num_envs] [iterations] [env]    (CPU smoke test: ... 4 2)
     env: GripperLift-v0 (default), PandaReach-v0 (a 7-joint arm reaching random targets) or DrawerOpen-v0.
Prints one JSON line per 10 iterations (episodes that ended in them, their success rate) and a summary.
Needs PyTorch: pip install 'alloy-engine[envs]'   (from a clone: bash install.sh --extras envs).
"""
import json
import sys
import time

try:
    import torch
    from torch import nn
except ImportError:
    from alloy.envs.errors import install_fix

    sys.exit(f"examples/envs/04_train_ppo.py needs PyTorch, which is not installed: {install_fix('torch')}")

from alloy.envs import make

NUM_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ENV = sys.argv[3] if len(sys.argv) > 3 else "GripperLift-v0"
HORIZON, EPOCHS, MINIBATCHES, GAMMA, LAM, CLIP, LR = 32, 4, 4, 0.99, 0.95, 0.2, 3e-4


class Normalizer(nn.Module):
    """Running mean / std of observations."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x: torch.Tensor) -> None:
        m, v, k = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d, total = m - self.mean, self.count + k
        self.mean += d * k / total
        self.var = (self.var * self.count + v * k + d * d * self.count * k / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


class ActorCritic(nn.Module):
    def __init__(self, n_obs: int, n_act: int) -> None:
        super().__init__()
        self.norm = Normalizer(n_obs)
        self.pi = nn.Sequential(nn.Linear(n_obs, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, n_act))
        self.v = nn.Sequential(nn.Linear(n_obs, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((n_act,), -0.5))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.pi(self.norm(obs)), self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(self.norm(obs)).squeeze(-1)


def rollout(env, net, obs, stats):
    """HORIZON steps of every world; truncated worlds bootstrap from their final observation."""
    buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
    for _ in range(HORIZON):
        with torch.no_grad():
            d = net.dist(obs)
            act = d.sample()
            buf["logp"].append(d.log_prob(act).sum(-1))
            buf["val"].append(net.value(obs))
        buf["obs"].append(obs)
        buf["act"].append(act)
        obs, rew, term, trunc, info = env.step(act)
        if "_final_obs" in info:
            with torch.no_grad():
                rew = rew + GAMMA * net.value(info["final_obs"]) * (trunc & info["_final_obs"]).float()
        for name, value in info["terms"].items():               # each reward term's mean, for the log
            stats.setdefault("terms", {})[name] = stats.get("terms", {}).get(name, 0.0) + float(value.mean())
        stats["steps"] = stats.get("steps", 0) + 1
        if "_episode" in info:
            m = info["_episode"]
            stats["episodes"] += int(m.sum())
            stats["successes"] += int((info["episode"]["success"] & m).sum())
        buf["rew"].append(rew)
        buf["done"].append((term | trunc).float())
        net.norm.update(obs)
    with torch.no_grad():
        last = net.value(obs)
    return {k: torch.stack(v) for k, v in buf.items()}, obs, last


def advantages(b, last):
    adv, gae = torch.zeros_like(b["rew"]), torch.zeros_like(last)
    for t in reversed(range(HORIZON)):
        nxt = last if t == HORIZON - 1 else b["val"][t + 1]
        delta = b["rew"][t] + GAMMA * nxt * (1 - b["done"][t]) - b["val"][t]
        gae = delta + GAMMA * LAM * (1 - b["done"][t]) * gae
        adv[t] = gae
    return adv, adv + b["val"]


def update(net, opt, b, adv, ret):
    flat = {k: v.reshape(HORIZON * NUM_ENVS, *v.shape[2:]) for k, v in b.items()}
    adv, ret = adv.reshape(-1), ret.reshape(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = adv.shape[0]
    for _ in range(EPOCHS):
        for idx in torch.randperm(n, device=adv.device).chunk(MINIBATCHES):
            d = net.dist(flat["obs"][idx])
            ratio = (d.log_prob(flat["act"][idx]).sum(-1) - flat["logp"][idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[idx]).mean()
            vf = 0.5 * (net.value(flat["obs"][idx]) - ret[idx]).pow(2).mean()
            loss = pg + vf - 0.001 * d.entropy().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()


def main() -> None:
    torch.manual_seed(0)
    env = make(ENV, num_envs=NUM_ENVS, backend="torch")
    net = ActorCritic(env.single_observation_space.shape[0], env.single_action_space.shape[0]).to(env.xp.tdevice)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    obs, _ = env.reset(seed=0)
    t0, window, last = time.perf_counter(), {"episodes": 0, "successes": 0}, None
    for it in range(1, ITERS + 1):
        for group in opt.param_groups:                       # linear learning-rate decay keeps the end stable
            group["lr"] = LR * (1.0 - (it - 1) / ITERS)
        b, obs, value = rollout(env, net, obs, window)        # counts the episodes that end into `window`
        adv, ret = advantages(b, value)
        update(net, opt, b, adv, ret)
        if it % 10 == 0 or it == ITERS:
            rate = window["successes"] / window["episodes"] if window["episodes"] else None
            last = rate if rate is not None else last
            print(json.dumps({"iter": it, "minutes": round((time.perf_counter() - t0) / 60, 2),
                              "samples": it * HORIZON * NUM_ENVS, "episodes": window["episodes"],
                              "success_rate": None if rate is None else round(rate, 3),
                              "mean_reward": round(float(b["rew"].mean()), 4),
                              "terms": {k: round(v / window["steps"], 4) for k, v in window.get("terms", {}).items()}}),
                  flush=True)
            window = {"episodes": 0, "successes": 0}
    print(json.dumps({"done": True, "env": ENV, "device": env.device, "num_envs": NUM_ENVS, "iterations": ITERS,
                      "minutes": round((time.perf_counter() - t0) / 60, 2),
                      "success_rate_last_window": None if last is None else round(last, 3)}))
    env.close()


if __name__ == "__main__":
    main()
