# SPDX-License-Identifier: Apache-2.0
"""The compact PPO of examples/envs/04_train_ppo.py, for dict observations (state vector + policy camera image).

Steering hooks: ``noise_scale`` multiplies the policy's action std when sampling, ``entropy_coef`` and ``set_lr``
change the update; the live worker is called at every step boundary and after every step of a rollout.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

GAMMA, LAM, CLIP, EPOCHS, MINIBATCHES = 0.99, 0.95, 0.2, 4, 4


class Normalizer(nn.Module):
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
    """State MLP + a small CNN on the camera image, shared by the policy and value heads."""

    def __init__(self, n_obs: int, n_act: int, image: int) -> None:
        super().__init__()
        self.norm = Normalizer(n_obs)
        self.image = image
        self.cnn = nn.Sequential(nn.Conv2d(3, 16, 8, 4), nn.ReLU(), nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, 2), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_img = self.cnn(torch.zeros(1, 3, image, image)).shape[1] if image else 0
        self.img_fc = nn.Sequential(nn.Linear(n_img, 64), nn.Tanh()) if image else None
        n_in = n_obs + (64 if image else 0)
        self.pi = nn.Sequential(nn.Linear(n_in, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, n_act))
        self.v = nn.Sequential(nn.Linear(n_in, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((n_act,), -0.5))

    def features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.norm(obs["state"])
        if self.img_fc is None:
            return x
        img = obs["front"].permute(0, 3, 1, 2).float() / 255.0
        return torch.cat([x, self.img_fc(self.cnn(img))], dim=-1)

    def dist(self, obs: dict[str, torch.Tensor], noise: float = 1.0) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.pi(self.features(obs)), self.log_std.exp() * noise)

    def value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.v(self.features(obs)).squeeze(-1)


class PPO:
    def __init__(self, net: ActorCritic, env: Any, horizon: int = 32, lr: float = 3e-4) -> None:
        self.net, self.env, self.horizon = net, env, horizon
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.noise_scale, self.entropy_coef = 1.0, 0.001

    @property
    def lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def set_lr(self, v: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = float(v)

    def rollout(self, obs: dict[str, torch.Tensor], live: Any = None) -> tuple[dict, dict, dict]:
        env, net = self.env, self.net
        buf = {k: [] for k in ("state", "front", "act", "logp", "val", "rew", "done")}
        ep_n, ep_r, ep_s = 0, 0.0, 0
        for _ in range(self.horizon):
            if live is not None:
                obs = live.boundary(obs)
            with torch.no_grad():
                d = net.dist(obs, self.noise_scale)
                act = d.sample()
                buf["logp"].append(d.log_prob(act).sum(-1))
                buf["val"].append(net.value(obs))
            buf["state"].append(obs["state"])
            if "front" in obs:
                buf["front"].append(obs["front"])
            buf["act"].append(act)
            nxt, rew, term, trunc, info = live.env_step(act) if live is not None else env.step(act)
            if "_final_obs" in info:
                with torch.no_grad():
                    rew = rew + GAMMA * net.value(info["final_obs"]) * (trunc & info["_final_obs"]).float()
            if "_episode" in info:
                m = info["_episode"]
                ep_n += int(m.sum())
                ep_r += float((info["episode"]["r"] * m).sum())
                ep_s += int((info["episode"]["success"] & m).sum())
            if live is not None:
                live.after_step(nxt, rew, term | trunc, info)
            buf["rew"].append(rew)
            buf["done"].append((term | trunc).float())
            net.norm.update(obs["state"])
            obs = nxt
        with torch.no_grad():
            last = net.value(obs)
        batch = {k: torch.stack(v) for k, v in buf.items() if v}
        batch["last"] = last
        stats = {"episodes": ep_n, "return": round(ep_r / ep_n, 4) if ep_n else None,
                 "success": round(ep_s / ep_n, 4) if ep_n else None}
        return batch, obs, stats

    def update(self, b: dict[str, torch.Tensor], active: Any = None) -> None:
        H, W = b["rew"].shape
        adv, gae = torch.zeros_like(b["rew"]), torch.zeros_like(b["last"])
        for t in reversed(range(H)):
            nxt = b["last"] if t == H - 1 else b["val"][t + 1]
            delta = b["rew"][t] + GAMMA * nxt * (1 - b["done"][t]) - b["val"][t]
            gae = delta + GAMMA * LAM * (1 - b["done"][t]) * gae
            adv[t] = gae
        ret = adv + b["val"]
        weight = torch.ones(H, W, device=adv.device) if active is None else active.expand(H, W).float()
        flat = {k: v.reshape(H * W, *v.shape[2:]) for k, v in b.items() if k != "last"}
        adv, ret, weight = adv.reshape(-1), ret.reshape(-1), weight.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(EPOCHS):
            for idx in torch.randperm(H * W, device=adv.device).chunk(MINIBATCHES):
                obs = {"state": flat["state"][idx]}
                if "front" in flat:
                    obs["front"] = flat["front"][idx]
                d = self.net.dist(obs)
                ratio = (d.log_prob(flat["act"][idx]).sum(-1) - flat["logp"][idx]).exp()
                w = weight[idx]
                pg = -(torch.min(ratio * adv[idx], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[idx]) * w).mean()
                vf = 0.5 * ((self.net.value(obs) - ret[idx]).pow(2) * w).mean()
                loss = pg + vf - self.entropy_coef * d.entropy().sum(-1).mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
