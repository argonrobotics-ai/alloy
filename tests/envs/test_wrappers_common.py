# SPDX-License-Identifier: Apache-2.0
"""The RL-library wrappers' shared parts, and the rule that every wrapper module imports on a base install (no
PyTorch, no RL library): only building a wrapper asks for the library, with the pip command."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.envs import EnvError
from alloy.envs.wrappers import _common, wrap
from alloy.envs.wrappers.train import main, parser

MODULES = ("_common", "for_rsl_rl", "for_sb3", "for_skrl", "train", "train_rsl_rl", "train_sb3", "train_skrl")
OPTIONAL = ("torch", "rsl_rl", "tensordict", "tensorboard", "stable_baselines3", "skrl", "gymnasium")

BASE_INSTALL = f"""
import sys
for name in {OPTIONAL!r}:
    sys.modules[name] = None                      # as if not installed
import alloy.envs.wrappers as w
for name in {MODULES!r}:
    __import__("alloy.envs.wrappers." + name)
from alloy.envs import EnvError
for lib, cls in (("rsl_rl", w.RslRlVecEnv), ("sb3", w.Sb3VecEnv), ("skrl", w.SkrlEnv)):
    for build in (lambda: cls(None), lambda: w.wrap(None, lib)):
        try:
            build()
        except EnvError as exc:
            assert "pip install" in exc.fix, exc.fix
        else:
            raise SystemExit(f"{{lib}}: no EnvError without the library")
from alloy.envs.wrappers.train import main
assert main(["GripperLift-v0", "--lib", "rsl_rl", "--num-envs", "2", "--iterations", "1"]) == 2
print("ok")
"""


def test_every_wrapper_module_imports_without_the_libraries():
    out = subprocess.run([sys.executable, "-c", BASE_INSTALL], capture_output=True, text=True, timeout=170)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")
    assert "fix: pip install torch rsl-rl-lib" in out.stderr


def test_unknown_library_is_refused():
    with pytest.raises(EnvError, match="no wrapper named 'rllib'") as err:
        wrap(None, "rllib")
    assert "rsl_rl" in err.value.fix and err.value.field == "lib"


def test_state_groups_reads_the_three_observation_forms():
    flat, image = np.zeros((2, 5), np.float32), np.zeros((2, 8, 8, 3), np.uint8)
    plain = SimpleNamespace(obs_groups=(), cameras={})
    assert list(_common.state_groups(plain, flat)) == ["policy"] and _common.group_names(plain) == ("policy",)
    camera = SimpleNamespace(obs_groups=(), cameras={"front": object()})
    got = _common.state_groups(camera, {"state": flat, "front": image})
    assert list(got) == ["policy"] and got["policy"] is flat and _common.image_keys(camera) == ["front"]
    grouped = SimpleNamespace(obs_groups=("policy", "critic"), cameras={"front": object()})
    got = _common.state_groups(grouped, {"policy": flat, "critic": flat[:, :3], "front": image})
    assert list(got) == ["policy", "critic"] and _common.group_names(grouped) == ("policy", "critic")


def test_episode_tally_counts_only_ended_worlds():
    tally = _common.EpisodeTally()
    assert tally.add({}) == (0, 0) and tally.success_rate() is None
    info = {"_episode": np.array([True, False, True]), "episode": {"success": np.array([True, True, False])}}
    assert tally.add(info) == (2, 1) and tally.success_rate() == 0.5
    assert tally.take() == (2, 1) and tally.episodes == 0


def test_command_line_defaults():
    args = parser().parse_args(["PandaReach-v0", "--lib", "skrl", "--num-envs", "64"])
    assert (args.env, args.lib, args.num_envs, args.iterations, args.report_every) == ("PandaReach-v0", "skrl", 64,
                                                                                       200, 10)
    with pytest.raises(SystemExit):
        parser().parse_args(["PandaReach-v0", "--lib", "rllib"])
    assert callable(main)


def test_environment_from_a_module_function():
    from alloy.envs.wrappers.train import env_from_factory

    with pytest.raises(EnvError, match="cannot load the environment function 'no_such_module_xyz:make'"):
        env_from_factory("no_such_module_xyz:make", 4, "cpu", 0)
    with pytest.raises(EnvError, match="cannot load the environment function 'types:nothing_here'"):
        env_from_factory("types:nothing_here", 4, "cpu", 0)
    with pytest.raises(EnvError, match="returned SimpleNamespace, not an environment"):
        env_from_factory("types:SimpleNamespace", 4, "cpu", 0)      # called with num_envs=, device=, seed=


def test_an_env_that_is_not_a_vector_env_is_refused():
    port_like = SimpleNamespace(num_envs=2)                       # like a task port: no array backend
    for check in (_common.require_vector_env, _common.require_torch_env):
        with pytest.raises(EnvError, match="takes an alloy.envs.VectorEnv, not SimpleNamespace") as err:
            check(port_like, "rsl_rl")
        assert "task ports" in err.value.fix and err.value.field == "env"
