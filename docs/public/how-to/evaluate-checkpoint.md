# How do I evaluate a trained policy on held-out seeds?

Load the checkpoint into an rsl_rl `OnPolicyRunner` in the Alloy physics engine (`pip install alloy-engine`, with its
`envs` extra, plus `pip install rsl-rl-lib`), build the environment with a seed that training never used, and run the
policy without noise for one full episode in every world. Then count successes and INVALID episodes.

```python
import torch
import alloy.envs
from alloy.envs.wrappers import RslRlVecEnv
from alloy.envs.wrappers.train_rsl_rl import default_config
from rsl_rl.runners import OnPolicyRunner

env = alloy.envs.make("PandaReach-v0", num_envs=64, seed=1000, backend="torch")      # training used seed 0
venv = RslRlVecEnv(env)
cfg = default_config(150, 1000, "PandaReach-v0")                                      # the trainer's own settings
cfg["obs_groups"] = venv.default_obs_groups()
runner = OnPolicyRunner(venv, cfg, log_dir=None, device=str(venv.device))
runner.load("runs/rsl_rl/PandaReach-v0/model_150.pt")
policy = runner.get_inference_policy(device=venv.device)

obs = venv.get_observations()
with torch.no_grad():
    for _ in range(venv.max_episode_length):                 # every world runs one full episode
        obs, rewards, dones, extras = venv.step(policy(obs))
episodes, successes, mean_return = venv.take_stats()
print(f"{successes} / {episodes} episodes succeeded, {venv.tally.take_invalid()} invalid")
venv.close()
```

```text
18 / 64 episodes succeeded, 0 invalid
```

This output came from a checkpoint trained with
`python -m alloy.envs.wrappers.train PandaReach-v0 --lib rsl_rl --num-envs 64 --iterations 150 --device cpu`, which
took 6.6 minutes on a 4-core CPU and printed a success rate of 0.352 on its last iteration. On seed 1000 the same
policy succeeded in 18 of 64 episodes (28 %): a small CPU run, enough to check the code. rsl_rl also prints a
summary of both networks before that line. For a policy worth evaluating, train with
`--num-envs 1024 --iterations 150` on a GPU: it reached 100 % success in training in our one run of it. Its
checkpoint loads the same way.

The policy is the mean action of the network: the same checkpoint and seed gave the same count in two CPU runs.
`alloy.il.evaluate` gives a 95 % interval as well, but it needs a function from observations to actions, which you
would write around the rsl_rl model.

## Keep in mind

- The number is for the seeds you pick: 64 episodes give a wide interval, so evaluate more worlds before you compare.
- A world the engine froze counts as an episode and never as a success; read the invalid count first.
- A GPU run [does not repeat bit for bit](../../known_gaps/determinism-gpu-rigid.md) once bodies touch.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Training, libraries and tasks: [Train with RL](../guides/rl.md).
- Record demonstrations instead: [Imitation learning](../guides/imitation.md).
- What the invalid count means: [World health](../guides/health.md).
