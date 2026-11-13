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


# --- layouts beyond one state vector and one camera -----------------------------------------------------------------


def grouped_spec(width: int = 4, cameras: tuple[str, ...] = ()) -> DatasetSpec:
    shapes = {"policy": (width,), "critic": (6,), **{c: (16, 16, 3) for c in cameras}}
    dtypes = {"policy": "float32", "critic": "float32", **{c: "uint8" for c in cameras}}
    names = {"policy": [f"gripper.q[{i}]" for i in range(width)], "critic": [f"c{i}" for i in range(6)]}
    return DatasetSpec(env="Synthetic-v0", task=TASK, fps=20.0, obs_shapes=shapes, obs_dtypes=dtypes,
                       obs_names=names["policy"], obs_key_names=names, action_names=["ax", "ay", "az"])


def episode_for(spec: DatasetSpec, length: int = 4, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed)
    obs = {k: (rng.integers(0, 255, (length, *s), dtype=np.uint8) if spec.obs_dtypes[k] == "uint8"
               else rng.normal(size=(length, *s)).astype(np.float32)) for k, s in spec.obs_shapes.items()}
    done = np.zeros(length, bool)
    done[-1] = True
    return Episode(obs=obs, next_obs=obs, actions=rng.normal(size=(length, 3)).astype(np.float32),
                   rewards=np.zeros(length, np.float32), terminated=done, truncated=~done & False, success=True)


def write_spec(root, spec: DatasetSpec, n: int = 2, **kwargs) -> list[Episode]:
    pytest.importorskip("lerobot")
    eps = [episode_for(spec, seed=i) for i in range(n)]
    with LeRobotWriter(root, repo_id=REPO, video=False, **kwargs) as writer:
        writer.open(spec)
        for ep in eps:
            writer.write(ep)
    return eps


def test_observation_groups_map_to_state_and_named_vectors(tmp_path):
    spec = grouped_spec()
    eps = write_spec(tmp_path / "ds", spec)
    ds = read_back(tmp_path / "ds")
    feats = ds.meta.features
    assert feats["observation.state"]["names"] == spec.obs_key_names["policy"]
    assert feats["observation.critic"]["names"] == spec.obs_key_names["critic"] and "observation.policy" not in feats
    assert np.allclose(np.asarray(ds[0]["observation.state"]), eps[0].obs["policy"][0])
    assert np.allclose(np.asarray(ds[0]["observation.critic"]), eps[0].obs["critic"][0])


def test_a_camera_only_task_has_no_state_feature_and_a_slash_in_a_key_is_stored_as_underscore(tmp_path, caplog):
    spec = DatasetSpec(env="Synthetic-v0", task=TASK, fps=20.0, action_names=["ax", "ay", "az"], obs_names=[],
                       obs_shapes={"state": (0,), "gripper/wrist": (16, 16, 3)},
                       obs_dtypes={"state": "float32", "gripper/wrist": "uint8"})
    with caplog.at_level(logging.WARNING, logger="alloy.il"):
        eps = write_spec(tmp_path / "ds", spec)
    assert "skips" not in caplog.text
    ds = read_back(tmp_path / "ds")
    assert "observation.state" not in ds.meta.features and "observation.images.gripper_wrist" in ds.meta.features
    assert ds.num_frames == sum(ep.length for ep in eps)
    clash = DatasetSpec(env="Synthetic-v0", task=TASK, fps=20.0, action_names=["ax"], obs_names=[],
                        obs_shapes={"a/b": (16, 16, 3), "a_b": (16, 16, 3)},
                        obs_dtypes={"a/b": "uint8", "a_b": "uint8"})
    with pytest.raises(EnvError, match="two observation keys") as err:
        LeRobotWriter(tmp_path / "clash").open(clash)
    assert "rename" in err.value.fix


def test_env_state_moves_named_columns_to_environment_state(tmp_path):
    spec = make_spec()
    spec.obs_shapes, spec.obs_dtypes = {"state": (6,)}, {"state": "float32"}
    spec.obs_names = ["gripper.q[0]", "gripper.q[1]", "cube.pos[0]", "cube.pos[1]", "cube.pos[2]", "action[0]"]
    eps = write_spec(tmp_path / "ds", spec, env_state=["cube"])
    ds = read_back(tmp_path / "ds")
    feats = ds.meta.features
    assert feats["observation.state"]["names"] == ["gripper.q[0]", "gripper.q[1]", "action[0]"]
    assert feats["observation.environment_state"]["names"] == ["cube.pos[0]", "cube.pos[1]", "cube.pos[2]"]
    assert np.allclose(np.asarray(ds[1]["observation.environment_state"]), eps[0].obs["state"][1, 2:5])
    assert np.allclose(np.asarray(ds[1]["observation.state"]), eps[0].obs["state"][1, [0, 1, 5]])
    for bad, text in ((["lid"], "matches no observation column"), (["gripper", "cube", "action"], "every")):
        with pytest.raises(EnvError, match=text):
            LeRobotWriter(tmp_path / "bad", env_state=bad).open(spec)


def test_lerobot_policy_configs_accept_a_state_only_dataset_with_env_state(tmp_path):
    pytest.importorskip("lerobot")
    types = pytest.importorskip("lerobot.configs.types")
    act = pytest.importorskip("lerobot.policies.act.configuration_act")
    spec = make_spec()
    spec.obs_shapes, spec.obs_dtypes = {"state": (6,)}, {"state": "float32"}
    spec.obs_names = ["gripper.q[0]", "gripper.q[1]", "cube.pos[0]", "cube.pos[1]", "cube.pos[2]", "action[0]"]

    def validate(root) -> None:
        feats = read_back(root).meta.features
        kinds = {"observation.state": types.FeatureType.STATE, "observation.environment_state": types.FeatureType.ENV}
        inputs = {k: types.PolicyFeature(type=kinds[k], shape=tuple(feats[k]["shape"])) for k in kinds if k in feats}
        action = types.PolicyFeature(type=types.FeatureType.ACTION, shape=tuple(feats["action"]["shape"]))
        outputs = {"action": action}
        act.ACTConfig(input_features=inputs, output_features=outputs).validate_features()

    write_spec(tmp_path / "plain", spec)
    with pytest.raises(ValueError, match="environment state"):
        validate(tmp_path / "plain")
    write_spec(tmp_path / "split", spec, env_state=["cube"])
    validate(tmp_path / "split")


# --- interruption, checkpoints and resume ---------------------------------------------------------------------------


def test_an_interrupted_recording_is_named_as_unclosed_and_hdf5_converts_at_the_end(tmp_path):
    pytest.importorskip("lerobot")
    pytest.importorskip("h5py")
    from alloy.il import HDF5Writer, copy_hdf5  # noqa: PLC0415
    from alloy.il.lerobot import unclosed_files  # noqa: PLC0415

    spec, root = grouped_spec(), tmp_path / "ds"
    writer = LeRobotWriter(root, repo_id=REPO, video=False)
    writer.open(spec)
    writer.write(episode_for(spec))
    assert unclosed_files(root)                              # as a kill would leave it: episodes in, no footer
    with pytest.raises(EnvError, match="was not closed") as err:
        LeRobotWriter(root, repo_id=REPO, video=False, resume=True).open(spec)
    assert "copy_hdf5" in err.value.fix and "overwrite=True" in err.value.fix
    writer.close()
    assert unclosed_files(root) == [] and read_back(root).num_episodes == 1
    eps = [episode_for(spec, seed=i) for i in range(3)]
    with HDF5Writer(tmp_path / "long.hdf5") as h5:
        h5.open(spec)
        for ep in eps:
            h5.write(ep)
    assert copy_hdf5(tmp_path / "long.hdf5", LeRobotWriter(tmp_path / "late", repo_id=REPO, video=False)) == 3
    ds = read_back(tmp_path / "late")
    assert ds.num_episodes == 3 and np.allclose(np.asarray(ds[4]["observation.critic"]), eps[1].obs["critic"][0])


def test_resume_appends_and_refuses_another_layout_or_an_unclosed_dataset(tmp_path):
    pytest.importorskip("lerobot")
    spec, root = grouped_spec(), tmp_path / "ds"
    write_spec(root, spec, n=2)
    with LeRobotWriter(root, repo_id=REPO, video=False, resume=True) as writer:
        writer.open(spec)
        ep = episode_for(spec, seed=9)
        writer.write(ep)
        assert ep.index == 2 and writer.first_index == 2 and writer.episodes == 1
    ds = read_back(root)
    assert ds.num_episodes == 3 and np.allclose(np.asarray(ds[8]["observation.state"]), ep.obs["policy"][0])
    with pytest.raises(EnvError, match="does not match"):
        LeRobotWriter(root, repo_id=REPO, video=False, resume=True).open(grouped_spec(width=5))
    with pytest.raises(EnvError, match="resume and overwrite"):
        LeRobotWriter(root, resume=True, overwrite=True)
    broken = tmp_path / "broken"
    (broken / "data").mkdir(parents=True)
    with pytest.raises(EnvError, match="no closed LeRobot dataset") as err:
        LeRobotWriter(broken, repo_id=REPO, resume=True).open(spec)
    assert "overwrite=True" in err.value.fix
    write_spec(tmp_path / "fresh", spec, n=1, resume=True)                # nothing there yet: a new dataset
    assert read_back(tmp_path / "fresh").num_episodes == 1


def test_overwrite_replaces_what_an_interrupted_recording_left_and_an_empty_dataset_is_removed(tmp_path, caplog):
    pytest.importorskip("lerobot")
    spec, partial = grouped_spec(), tmp_path / "partial"
    (partial / "data" / "chunk-000").mkdir(parents=True)
    (partial / "meta").mkdir()
    write_spec(partial, spec, n=1, overwrite=True)
    assert read_back(partial).num_episodes == 1
    empty = tmp_path / "empty"
    with caplog.at_level(logging.WARNING, logger="alloy.il"), LeRobotWriter(empty, repo_id=REPO) as writer:
        writer.open(spec)
    assert not empty.exists() and "no episode was written" in caplog.text


def test_missing_package_on_python_311_names_the_python_it_needs(monkeypatch, tmp_path):
    import alloy.il.lerobot as module  # noqa: PLC0415

    def no_lerobot(name, *args):
        raise ImportError("No module named 'lerobot'")

    monkeypatch.setattr("importlib.import_module", no_lerobot)
    monkeypatch.setattr(module.sys, "version_info", (3, 11, 9, "final", 0))
    with pytest.raises(FeatureUnavailableError) as err:
        LeRobotWriter(tmp_path / "ds").open(make_spec())
    assert "Python 3.12" in err.value.message and "3.11" in err.value.message
    assert "HDF5Writer" in err.value.fix and "lerobot[dataset]" not in err.value.fix
