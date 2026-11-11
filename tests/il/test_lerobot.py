# SPDX-License-Identifier: Apache-2.0
"""LeRobotWriter: a v3.0 dataset through lerobot's own API, read back with ``LeRobotDataset`` (synthetic)."""

from __future__ import annotations

import builtins
import logging
import time

import numpy as np
import pytest

from alloy.api.errors import FeatureUnavailableError
from alloy.envs.errors import EnvError
from alloy.il.episode import DatasetSpec, Episode
from alloy.il.lerobot import LeRobotWriter

LENGTHS = (5, 7, 4)
REPO = "local/alloy_test"
TASK = "lift the cube"


def make_spec(fps: float = 20.0, extra: bool = False) -> DatasetSpec:
    shapes, dtypes = {"state": (6,), "front": (64, 64, 3)}, {"state": "float32", "front": "uint8"}
    if extra:
        shapes["depth"], dtypes["depth"] = (64, 64), "float32"
    return DatasetSpec(env="Synthetic-v0", task=TASK, fps=fps, obs_shapes=shapes, obs_dtypes=dtypes,
                       obs_names=[f"s{i}" for i in range(6)], action_names=["ax", "ay", "az"])


def smooth_image(seed: int, t: int) -> np.ndarray:
    """A smooth gradient that moves with t, so lossy video keeps it close."""
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    img = np.stack([xx * 3 + 8 * t, yy * 3 + 5 * seed, (xx + yy) * 1.5 + 4 * t], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_episode(seed: int, length: int, success: bool, extra: bool = False) -> Episode:
    rng = np.random.default_rng(seed)
    obs = {"state": rng.normal(size=(length, 6)).astype(np.float32),
           "front": np.stack([smooth_image(seed, t) for t in range(length)])}
    if extra:
        obs["depth"] = np.zeros((length, 64, 64), np.float32)
    done = np.zeros(length, bool)
    done[-1] = True
    return Episode(obs=obs, next_obs=obs, actions=rng.normal(size=(length, 3)).astype(np.float32),
                   rewards=rng.normal(size=length).astype(np.float32), terminated=done.copy() & success,
                   truncated=done.copy() & (not success), success=success)


def write_dataset(root, video: bool, extra: bool = False, overwrite: bool = False) -> list[Episode]:
    pytest.importorskip("lerobot")
    eps = [make_episode(i, n, success=i != 1, extra=extra) for i, n in enumerate(LENGTHS)]
    with LeRobotWriter(root, repo_id=REPO, video=video, overwrite=overwrite) as writer:
        writer.open(make_spec(extra=extra))
        for ep in eps:
            writer.write(ep)
    return eps


def read_back(root):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    return LeRobotDataset(REPO, root=root)


def test_missing_package_error(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("lerobot"):
            raise ImportError("No module named 'lerobot'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("importlib.import_module", lambda name, *a: fake_import(name))
    with pytest.raises(FeatureUnavailableError) as err:
        LeRobotWriter(tmp_path / "ds").open(make_spec())
    assert "pip install 'lerobot[dataset]'" in err.value.fix
    assert err.value.field == "lerobot"


def test_non_integer_fps(tmp_path):
    pytest.importorskip("lerobot")
    with pytest.raises(EnvError) as err:
        LeRobotWriter(tmp_path / "ds").open(make_spec(fps=1 / 0.03))
    assert "integer fps" in err.value.message and "dt" in err.value.fix


def test_write_before_open(tmp_path):
    with pytest.raises(EnvError):
        LeRobotWriter(tmp_path / "ds").write(make_episode(0, 3, True))


def test_overwrite_rules(tmp_path):
    pytest.importorskip("lerobot")
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "keep.txt").write_text("x")
    with pytest.raises(EnvError):
        LeRobotWriter(plain).open(make_spec())
    with pytest.raises(EnvError, match="not a LeRobot dataset"):
        LeRobotWriter(plain, overwrite=True).open(make_spec())
    assert (plain / "keep.txt").exists()
    root = tmp_path / "ds"
    write_dataset(root, video=False)
    with pytest.raises(EnvError):
        LeRobotWriter(root).open(make_spec())
    write_dataset(root, video=False, overwrite=True)       # overwrite of a real dataset: replaced
    assert read_back(root).num_episodes == 3


def test_image_mode_exact_and_layout(tmp_path):
    root = tmp_path / "ds"
    t0 = time.perf_counter()
    eps = write_dataset(root, video=False)
    ds = read_back(root)
    print(f"image mode write+read: {time.perf_counter() - t0:.1f} s")
    check_layout(ds, eps)
    frame = ds[0]["observation.images.front"]                   # torch float (C, H, W) in [0, 1]
    assert tuple(frame.shape) == (3, 64, 64)
    err = np.abs(frame.numpy().transpose(1, 2, 0) * 255 - eps[0].obs["front"][0]).max()
    print(f"image mode max abs error (0-255): {err:.3f}")
    assert err <= 1.0


def test_video_mode_close(tmp_path):
    root = tmp_path / "ds"
    t0 = time.perf_counter()
    eps = write_dataset(root, video=True)
    ds = read_back(root)
    print(f"video mode write+read: {time.perf_counter() - t0:.1f} s")
    check_layout(ds, eps)
    worst = 0.0
    for i in (0, 4, 5, 12, 15):
        ep, local = (0, i) if i < 5 else ((1, i - 5) if i < 12 else (2, i - 12))
        got = ds[i]["observation.images.front"].numpy().transpose(1, 2, 0) * 255
        worst = max(worst, float(np.abs(got - eps[ep].obs["front"][local]).max()))
    print(f"video mode max abs error (0-255): {worst:.1f}")
    assert worst < 40


def check_layout(ds, eps) -> None:
    assert ds.num_episodes == 3 and ds.num_frames == sum(LENGTHS) and ds.fps == 20
    assert ds.meta.info.codebase_version == "v3.0"
    assert ds.meta.info.robot_type == "alloy"
    assert [e.index for e in eps] == [0, 1, 2]
    starts = np.cumsum((0,) + LENGTHS)
    for e, ep in enumerate(eps):
        for t in range(ep.length):
            item = ds[int(starts[e]) + t]
            assert int(item["episode_index"]) == e and int(item["frame_index"]) == t
            assert item["task"] == TASK
            assert np.array_equal(item["observation.state"].numpy(), ep.obs["state"][t])
            assert np.array_equal(item["action"].numpy(), ep.actions[t])
            assert float(item["next.reward"]) == float(ep.rewards[t])
            last = t == ep.length - 1
            assert bool(item["next.done"]) == last
            assert bool(item["next.success"]) == (last and ep.success)


def test_skipped_key_warns_once(tmp_path, caplog):
    pytest.importorskip("lerobot")
    caplog.set_level(logging.WARNING, logger="alloy.il")
    root = tmp_path / "ds"
    write_dataset(root, video=False, extra=True)
    warnings = [r for r in caplog.records if "depth" in r.getMessage()]
    assert len(warnings) == 1
    assert "observation.depth" not in read_back(root).features
