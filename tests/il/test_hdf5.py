# SPDX-License-Identifier: Apache-2.0
"""HDF5 episode writer and loader: round trip, layout, crash safety, and robomimic reading the file."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from alloy.api.errors import FeatureUnavailableError
from alloy.envs.errors import EnvError
from alloy.il.episode import DatasetSpec, Episode
from alloy.il.hdf5 import HDF5Writer, load_hdf5

HAS_H5PY = importlib.util.find_spec("h5py") is not None
needs_h5py = pytest.mark.skipif(not HAS_H5PY, reason="h5py is not installed")


def make_spec() -> DatasetSpec:
    return DatasetSpec(env="Toy-v0", task="lift", fps=100.0, obs_shapes={"state": (4,), "cam": (6, 8, 3)},
                       obs_dtypes={"state": "float32", "cam": "uint8"}, obs_names=list("abcd"),
                       action_names=["x", "y"], state_names=["s0", "s1", "s2"], env_kwargs={"num_envs": 3})


def make_episode(steps: int, seed: int = 0, success: bool = True, states: bool = True) -> Episode:
    rng = np.random.default_rng(seed)

    def obs() -> dict[str, np.ndarray]:
        return {"state": rng.normal(size=(steps, 4)).astype(np.float32),
                "cam": rng.integers(0, 255, size=(steps, 6, 8, 3), dtype=np.uint8)}

    terminated = np.zeros(steps, bool)
    terminated[-1] = success
    truncated = np.zeros(steps, bool)
    truncated[-1] = not success
    return Episode(obs=obs(), next_obs=obs(), actions=rng.normal(size=(steps, 2)).astype(np.float32),
                   rewards=rng.normal(size=steps).astype(np.float32), terminated=terminated, truncated=truncated,
                   success=success, states=rng.normal(size=(steps, 3)).astype(np.float32) if states else None,
                   start={"q": rng.normal(size=(5,)).astype(np.float32)}, world=seed)


def write_file(path: Path, lengths: list[int]) -> list[Episode]:
    eps = [make_episode(n, seed=i, success=i % 2 == 0) for i, n in enumerate(lengths)]
    with HDF5Writer(path) as writer:
        writer.open(make_spec())
        for ep in eps:
            writer.write(ep)
    return eps


def test_missing_h5py_names_the_install_command(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "h5py", None)       # makes `import h5py` raise ImportError
    with pytest.raises(FeatureUnavailableError) as err:
        HDF5Writer(tmp_path / "a.hdf5").open(make_spec())
    assert "pip install h5py" in err.value.fix and err.value.field == "h5py"
    with pytest.raises(FeatureUnavailableError):
        load_hdf5(tmp_path / "a.hdf5")


@needs_h5py
def test_round_trip_is_exact(tmp_path):
    eps = write_file(tmp_path / "d.hdf5", [5, 3])
    spec, back = load_hdf5(tmp_path / "d.hdf5")
    assert spec == make_spec()
    assert len(back) == 2
    for a, b in zip(eps, back):
        for group in ("obs", "next_obs"):
            for key, arr in getattr(a, group).items():
                got = getattr(b, group)[key]
                assert got.dtype == arr.dtype and np.array_equal(got, arr)
        for name in ("actions", "rewards", "terminated", "truncated", "states"):
            assert np.array_equal(getattr(a, name), getattr(b, name))
        assert np.array_equal(a.start["q"], b.start["q"])
        assert (a.success, a.world, b.index) == (b.success, b.world, a.index)


@needs_h5py
def test_episode_without_states_round_trips(tmp_path):
    with HDF5Writer(tmp_path / "d.hdf5") as w:
        w.open(make_spec())
        w.write(make_episode(4, states=False))
    assert load_hdf5(tmp_path / "d.hdf5")[1][0].states is None


@needs_h5py
def test_demos_load_in_numeric_order_and_can_be_picked(tmp_path):
    write_file(tmp_path / "d.hdf5", [2] * 12)
    _, eps = load_hdf5(tmp_path / "d.hdf5")
    assert [e.index for e in eps] == list(range(12))
    assert [e.world for e in eps] == list(range(12))
    _, some = load_hdf5(tmp_path / "d.hdf5", episodes=[10, 2])
    assert [e.index for e in some] == [2, 10]


@needs_h5py
def test_layout_attrs_dones_and_masks(tmp_path):
    import h5py
    write_file(tmp_path / "d.hdf5", [5, 3, 4])
    with h5py.File(tmp_path / "d.hdf5", "r") as f:
        assert f["data"].attrs["total"] == 12
        env_args = json.loads(f["data"].attrs["env_args"])
        assert env_args == {"env_name": "Toy-v0", "type": 2, "env_kwargs": {"num_envs": 3}}
        assert json.loads(f["data"].attrs["alloy_spec"])["task"] == "lift"
        demo = f["data/demo_1"]
        assert demo.attrs["num_samples"] == 3 and not demo.attrs["success"]
        assert demo["dones"][()].tolist() == [0, 0, 1]
        assert demo["actions"].dtype == np.float32 and demo["rewards"].dtype == np.float32
        assert demo["next_obs/cam"].dtype == np.uint8 and demo["obs/cam"].compression == "gzip"
        assert demo["obs/state"].compression is None
        assert [m.decode() for m in f["mask/all"][()]] == ["demo_0", "demo_1", "demo_2"]
        assert [m.decode() for m in f["mask/success"][()]] == ["demo_0", "demo_2"]


@needs_h5py
def test_counters_and_episode_index(tmp_path):
    ep = make_episode(4)
    w = HDF5Writer(tmp_path / "d.hdf5")
    w.open(make_spec())
    w.write(make_episode(3))
    w.write(ep)
    w.close()
    assert (w.episodes, w.samples, ep.index) == (2, 7, 1)


@needs_h5py
def test_existing_file_needs_overwrite(tmp_path):
    write_file(tmp_path / "d.hdf5", [3])
    with pytest.raises(EnvError) as err:
        HDF5Writer(tmp_path / "d.hdf5").open(make_spec())
    assert "overwrite=True" in err.value.fix
    with HDF5Writer(tmp_path / "d.hdf5", overwrite=True) as w:
        w.open(make_spec())
    assert load_hdf5(tmp_path / "d.hdf5")[1] == []


@needs_h5py
def test_written_episodes_are_readable_without_close(tmp_path):
    w = HDF5Writer(tmp_path / "d.hdf5")
    w.open(make_spec())
    ep = make_episode(6)
    w.write(ep)
    shutil.copy(tmp_path / "d.hdf5", tmp_path / "copy.hdf5")      # what a crashed run leaves on disk
    w.close()
    _, eps = load_hdf5(tmp_path / "copy.hdf5")
    assert len(eps) == 1 and np.array_equal(eps[0].obs["state"], ep.obs["state"])


@needs_h5py
def test_misuse_raises_env_errors(tmp_path):
    w = HDF5Writer(tmp_path / "d.hdf5")
    with pytest.raises(EnvError):
        w.write(make_episode(3))                                  # before open
    w.open(make_spec())
    bad = make_episode(3)
    bad.obs["state"] = bad.obs["state"][:, :3]
    with pytest.raises(EnvError):
        w.write(bad)
    bad = make_episode(3)
    del bad.next_obs["cam"]
    with pytest.raises(EnvError):
        w.write(bad)
    bad = make_episode(3)
    bad.actions = bad.actions[:, :1]
    with pytest.raises(EnvError):
        w.write(bad)
    w.close()
    assert w.episodes == 0


@needs_h5py
def test_foreign_file_is_refused(tmp_path):
    import h5py
    with h5py.File(tmp_path / "other.hdf5", "w") as f:
        f.create_group("data")
    with pytest.raises(EnvError) as err:
        load_hdf5(tmp_path / "other.hdf5")
    assert "not written by Alloy" in err.value.message


@needs_h5py
def test_robomimic_reads_the_file(tmp_path):
    FileUtils = pytest.importorskip("robomimic.utils.file_utils")      # skips if its own deps are missing
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.utils.dataset import SequenceDataset

    eps = write_file(tmp_path / "d.hdf5", [5, 3, 4])
    path = str(tmp_path / "d.hdf5")
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["state"], "rgb": ["cam"]}})
    ds = SequenceDataset(path, obs_keys=("state",), dataset_keys=("actions", "rewards", "dones"), seq_length=1,
                         hdf5_cache_mode="all", hdf5_use_swmr=False)
    assert len(ds) == 12
    item = ds[5]                                                   # second demo, step 0
    assert np.array_equal(item["obs"]["state"][0], eps[1].obs["state"][0])
    assert np.array_equal(item["actions"][0], eps[1].actions[0])
    assert item["rewards"][0] == eps[1].rewards[0]
    ds_ok = SequenceDataset(path, obs_keys=("state",), dataset_keys=("actions",), filter_by_attribute="success",
                            hdf5_cache_mode=None, hdf5_use_swmr=False)
    assert len(ds_ok) == 9
    assert FileUtils.get_env_metadata_from_dataset(path)["env_name"] == "Toy-v0"
    meta = FileUtils.get_shape_metadata_from_dataset(path)
    assert meta["ac_dim"] == 2 and meta["all_shapes"]["state"] == [4]
