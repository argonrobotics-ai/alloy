# SPDX-License-Identifier: Apache-2.0
"""Gemini audit T1-002: the rsl_rl trainers' checkpoints record the iterations done, and one run writes one log.

rsl_rl 5.x's ``OnPolicyRunner.learn`` leaves ``current_learning_iteration`` at the last 0-based index and opens a new
log writer at every call, so ``train_rsl_rl.train`` and ``alloy.envs.loco.train``, which called it once per report
chunk, saved ``model_80.pt`` with ``iter`` 72 (one lost per chunk) and one TensorBoard file per chunk. Both now run
one ``learn`` call through ``learn_reporting``. The first laws use a stand-in runner that counts as rsl_rl 5.5.2 does
(no rsl_rl needed); the last trains GripperLift-v0 for a few iterations on the CPU where rsl_rl is installed."""
from __future__ import annotations

import pytest

from alloy.envs.wrappers.train_rsl_rl import learn_reporting


class _Logger:
    def __init__(self) -> None:
        self.writers, self.logged = 0, []

    def log(self, *, it: int, **_: object) -> None:
        self.logged.append(it)


class _Runner:
    """rsl_rl 5.5.2's ``OnPolicyRunner.learn`` as far as counting goes: a new writer per call, the count left at the
    last 0-based index, and a final save under that count."""

    def __init__(self, logger: bool = True) -> None:
        self.current_learning_iteration, self.saved, self.random = 0, [], []
        self.logger = _Logger() if logger else None
        self.its: list[int] = []

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self.random.append(init_at_random_ep_len)
        if self.logger is not None:
            self.logger.writers += 1
        start = self.current_learning_iteration
        for it in range(start, start + num_learning_iterations):
            self.current_learning_iteration = it
            self.its.append(it)
            if self.logger is not None:
                self.logger.log(it=it)
        self.saved.append(self.current_learning_iteration)


def test_one_learn_call_reports_each_chunk_and_leaves_the_count_done():
    runner, reports = _Runner(), []
    learn_reporting(runner, 6, 2, reports.append, random_start=True)
    assert reports == [2, 4, 6] and runner.current_learning_iteration == 6
    assert runner.logger.writers == 1 and runner.saved == [6]          # was: 3 writers, the last save under 3
    assert runner.random == [True] and "log" not in vars(runner.logger)   # the logger's own method again
    learn_reporting(runner, 4, 3, reports.append)                       # a second run continues the count
    assert reports[3:] == [3, 4] and runner.current_learning_iteration == 10
    assert runner.its == list(range(10))                                # no iteration index repeats


def test_a_runner_without_a_logger_keeps_the_count_by_chunk():
    runner, reports = _Runner(logger=False), []
    learn_reporting(runner, 4, 1, reports.append)
    assert reports == [1, 2, 3, 4] and runner.current_learning_iteration == 4    # was: stuck at 0
    assert runner.its == [0, 1, 2, 3]


@pytest.mark.compile_heavy
def test_train_saves_the_iterations_done_and_one_log(tmp_path):
    pytest.importorskip("rsl_rl")
    torch = pytest.importorskip("torch")
    from alloy.envs.wrappers.train_rsl_rl import train

    summary = train("GripperLift-v0", num_envs=4, iterations=6, seed=0, device="cpu", log_dir=str(tmp_path),
                    report_every=2, env_kwargs={"max_steps": 10})
    ckpt = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)
    assert summary["checkpoint"].endswith("model_6.pt") and ckpt["iter"] == 6       # was: iter 3
    assert len(list(tmp_path.glob("events.out.tfevents.*"))) <= 1                   # was: one file per chunk
