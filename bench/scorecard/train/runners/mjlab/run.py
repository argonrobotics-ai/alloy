# SPDX-License-Identifier: Apache-2.0
"""Row 4 (time to train), mjlab runner: PPO via rsl_rl, protocol record out.

  python run.py --task go2_flat --seed 0 --num-envs 4096 --max-iterations 1000 --max-wall-s 4500 --out run.jsonl
  python run.py ... --variant native     # the task's own agent cfg, unmodified (matched = the protocol's settings)
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()
os.environ.setdefault("MUJOCO_GL", "disable")  # headless: no EGL context needed for training
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from schema import RunHeader, RunWriter  # copied next to the runner on the VM
except ImportError:  # in the worktree
    from bench.scorecard.train.schema import RunHeader, RunWriter

import torch  # noqa: E402

from metrics import MetricsRecorder, install_probe  # noqa: E402
from tasks import GO2_PORT, TASKS, matched_agent_cfg  # noqa: E402


class MaxWallReached(Exception):
    pass


def build_runner(args, spec, writer_probe_holder):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    env_cfg = spec.env_cfg()
    agent_cfg = spec.agent_cfg()
    if args.variant == "matched":
        agent_cfg = matched_agent_cfg(agent_cfg, spec.protocol_ppo)
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg.seed = env_cfg.seed = args.seed
    agent_cfg.logger, agent_cfg.upload_model = "tensorboard", False
    agent_cfg.save_interval = 10**9  # no checkpoints / ONNX export in the timed loop
    agent_cfg.run_name = f"{args.task}_{args.variant}_s{args.seed}"
    probe = spec.probe()
    install_probe(env_cfg, probe)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    recorder = MetricsRecorder(probe)
    env = _RecordingWrapper(env, recorder, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(spec.agent_task_id or spec.task_id) or __import__("mjlab.rl", fromlist=["x"]).MjlabOnPolicyRunner
    log_dir = Path(args.log_dir) / agent_cfg.experiment_name / time.strftime("%Y%m%d_%H%M%S")
    runner = runner_cls(env, dataclasses.asdict(agent_cfg), str(log_dir), "cuda:0")
    return runner, recorder, env_cfg, agent_cfg


def _RecordingWrapper(env, recorder, clip_actions):
    from mjlab.rl import RslRlVecEnvWrapper

    class Recording(RslRlVecEnvWrapper):
        def step(self, actions):
            obs, rew, dones, extras = super().step(actions)
            recorder.record(self.unwrapped, rew)
            return obs, rew, dones, extras

    return Recording(env, clip_actions=clip_actions)


def versions() -> dict[str, str]:
    from importlib.metadata import version

    return {p: version(p) for p in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "torch")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iterations", type=int, required=True)
    ap.add_argument("--max-wall-s", type=float, default=4500.0)
    ap.add_argument("--variant", choices=("matched", "native"), default="matched")
    ap.add_argument("--robot", choices=("shipped", "go2"), default="shipped",
                    help="go2_flat only: shipped = mjlab's Go1 task; go2 = our Go2 port (go2.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default=str(Path.home() / "mjlab_logs"))
    args = ap.parse_args(argv)

    spec = GO2_PORT if (args.task == 'go2_flat' and args.robot == 'go2') else TASKS[args.task]
    if spec.unavailable:
        print(f"{args.task}: not available in mjlab: {spec.unavailable}")
        return 3
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    runner, recorder, env_cfg, agent_cfg = build_runner(args, spec, None)
    header = RunHeader(
        framework="mjlab", task=args.task, task_id=spec.task_id, seed=args.seed, num_envs=args.num_envs,
        variant=args.variant, gpu=torch.cuda.get_device_name(0), versions=versions(),
        settings={**spec.header_settings(env_cfg, agent_cfg), "max_wall_s": args.max_wall_s},
    )
    writer = RunWriter(args.out, header, process_start=T_PROCESS)
    return train(runner, recorder, writer, args, agent_cfg)


def train(runner, recorder, writer, args, agent_cfg) -> int:
    steps_per_iter = agent_cfg.num_steps_per_env * args.num_envs
    state = {"n": 0}
    orig_log = runner.logger.log

    def log_and_record(*a, **kw):
        torch.cuda.synchronize()
        state["n"] += 1
        m = recorder.flush()
        pt = writer.point(state["n"], state["n"] * steps_per_iter, m)
        print(f"it {pt.iteration} wall {pt.wall_s:.0f}s " + " ".join(f"{k}={v:.3g}" for k, v in m.items()), flush=True)
        if pt.wall_s > args.max_wall_s:
            raise MaxWallReached
        return orig_log(*a, **kw)

    runner.logger.log = log_and_record
    runner.save = lambda *a, **k: None  # rsl_rl saves at it 0; keep checkpoint / ONNX export out of the timed loop
    writer.start_training()
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except MaxWallReached:
        print("max wall reached; stopping", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
