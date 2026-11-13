# SPDX-License-Identifier: Apache-2.0
"""alloy.sysid.readers: load_recording dispatch and errors, HDF5 and LeRobot readers, missing optional packages.

The HDF5 and LeRobot tests need h5py / pyarrow and skip where those are not installed."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import alloy
from alloy.sysid.errors import SysidError
from alloy.sysid.readers import load_hdf5, load_lerobot, load_recording, save_hdf5
from alloy.sysid.recording import Recording, make_episode, save_csv


def make_recording(extras: bool = True) -> Recording:
    rng = np.random.default_rng(3)
    eps = []
    for n in (14, 22):
        t = 0.01 * np.arange(n)
        pos = rng.normal(size=(n, 2))
        eps.append(make_episode(t, rng.normal(size=(n, 2)), pos, pos * 2 if extras else None,
                                pos * 3 if extras else None))
    return Recording(["a", "b"], eps)


def raises_field(field: str, fn, *args, **kwargs) -> SysidError:
    with pytest.raises(SysidError) as info:
        fn(*args, **kwargs)
    assert info.value.field == field
    return info.value


def assert_same(rec: Recording, back: Recording) -> None:
    assert back.joints == rec.joints
    assert len(back.episodes) == len(rec.episodes)
    for old, new in zip(rec.episodes, back.episodes):
        for name in ("t", "command", "position", "velocity", "torque"):
            a, b = getattr(old, name), getattr(new, name)
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_allclose(b, a, atol=1e-12, rtol=0.0)


def test_load_recording_dispatches_csv_and_txt(tmp_path):
    rec = make_recording()
    path = save_csv(rec, tmp_path / "take.csv")
    assert_same(rec, load_recording(path))
    assert_same(rec, load_recording(str(path.rename(tmp_path / "TAKE.TXT"))))


def test_load_recording_unknown_suffix(tmp_path):
    path = tmp_path / "take.xyz"
    path.write_text("t\n")
    err = raises_field("data", load_recording, path)
    assert ".xyz" in err.message


def test_load_recording_missing_file(tmp_path):
    err = raises_field("data", load_recording, tmp_path / "nope.csv")
    assert "does not exist" in err.message


def test_load_recording_folder_without_info_json(tmp_path):
    err = raises_field("data", load_recording, tmp_path)
    assert "meta/info.json" in err.message


def test_hdf5_reader_without_h5py_names_the_pip_command(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    for call in (lambda: load_hdf5(tmp_path / "x.hdf5"), lambda: save_hdf5(make_recording(), tmp_path / "x.hdf5"),
                 lambda: load_recording(_touch(tmp_path / "y.h5"))):
        with pytest.raises(alloy.FeatureUnavailableError) as info:
            call()
        assert "pip install h5py" in info.value.fix


def _touch(path):
    path.write_bytes(b"")
    return path


def test_lerobot_reader_without_pyarrow_names_the_pip_command(tmp_path, monkeypatch):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(alloy.FeatureUnavailableError) as info:
        load_recording(tmp_path)
    assert "pip install pyarrow" in info.value.fix


def test_hdf5_round_trip(tmp_path):
    pytest.importorskip("h5py")
    rec = make_recording()
    assert_same(rec, load_recording(save_hdf5(rec, tmp_path / "d" / "take.hdf5")))


def test_hdf5_round_trip_without_extras(tmp_path):
    pytest.importorskip("h5py")
    rec = make_recording(extras=False)
    assert_same(rec, load_hdf5(save_hdf5(rec, tmp_path / "take.h5")))


def write_top_level(path, joints=True, command=True) -> None:
    h5py = pytest.importorskip("h5py")
    t = 0.01 * np.arange(12)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("t", data=t + 3.0)
        fh.create_dataset("position", data=np.ones((12, 2)))
        if command:
            fh.create_dataset("command", data=np.zeros((12, 2)))
        if joints:
            fh.attrs["joints"] = ["a", "b"]


def test_hdf5_single_episode_at_top_level(tmp_path):
    write_top_level(tmp_path / "one.hdf5")
    rec = load_hdf5(tmp_path / "one.hdf5")
    assert rec.joints == ["a", "b"] and len(rec.episodes) == 1
    assert rec.episodes[0].t[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(rec.episodes[0].position, 1.0, atol=1e-12)


def test_hdf5_missing_joints_attribute(tmp_path):
    write_top_level(tmp_path / "one.hdf5", joints=False)
    err = raises_field("data", load_hdf5, tmp_path / "one.hdf5")
    assert "joints" in err.message


def test_hdf5_missing_command(tmp_path):
    write_top_level(tmp_path / "one.hdf5", command=False)
    err = raises_field("data", load_hdf5, tmp_path / "one.hdf5")
    assert "command" in err.message


def test_hdf5_without_position_anywhere(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "empty.hdf5", "w") as fh:
        fh.create_dataset("t", data=np.arange(12.0))
    err = raises_field("data", load_hdf5, tmp_path / "empty.hdf5")
    assert "position" in err.message


def write_lerobot(root, state_width: int = 2, names=("a.pos", "b.pos"), state_names=None) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(5)
    n = 12
    lengths = (n, n + 3)
    total = sum(lengths)
    action = rng.normal(size=(total, 2))
    state = rng.normal(size=(total, state_width))
    table = pa.table({
        "action": pa.array(action.tolist(), type=pa.list_(pa.float64())),
        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float64())),
        "timestamp": pa.array(np.concatenate([0.02 * np.arange(k) for k in lengths]), type=pa.float64()),
        "episode_index": pa.array(np.repeat([0, 1], lengths), type=pa.int64()),
    })
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features = {"action": {"names": list(names)}}
    if state_names is not None:
        features["observation.state"] = {"names": list(state_names)}
    (root / "meta" / "info.json").write_text(json.dumps({"features": features}))
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    write_lerobot.expected = (action, state, lengths)


def test_lerobot_reads_episodes_names_and_values(tmp_path):
    write_lerobot(tmp_path / "ds")
    action, state, lengths = write_lerobot.expected
    rec = load_recording(tmp_path / "ds")
    assert rec.joints == ["a", "b"]
    assert [len(e.t) for e in rec.episodes] == list(lengths)
    np.testing.assert_allclose(rec.episodes[0].command, action[: lengths[0]], atol=1e-12)
    np.testing.assert_allclose(rec.episodes[1].position, state[lengths[0]:], atol=1e-12)
    assert rec.episodes[1].t[0] == pytest.approx(0.0, abs=1e-12)
    assert rec.dt == pytest.approx(0.02, abs=1e-12)
    assert rec.notes


def test_lerobot_columns_pick_state_columns(tmp_path):
    write_lerobot(tmp_path / "ds", state_width=4)
    _, state, lengths = write_lerobot.expected
    rec = load_lerobot(tmp_path / "ds", columns=[1, 3])
    np.testing.assert_allclose(rec.episodes[0].position, state[: lengths[0]][:, [1, 3]], atol=1e-12)


def test_lerobot_names_from_state_feature_and_explicit_joints(tmp_path):
    write_lerobot(tmp_path / "ds", names=(), state_names=("robot/x.pos", "robot/y"))
    assert load_lerobot(tmp_path / "ds").joints == ["x", "y"]
    assert load_lerobot(tmp_path / "ds", joints=["p", "q"]).joints == ["p", "q"]


def test_lerobot_state_action_width_mismatch(tmp_path):
    from dataclasses import fields

    from alloy.sysid.fitter import FitConfig

    write_lerobot(tmp_path / "ds", state_width=3)
    err = raises_field("data", load_lerobot, tmp_path / "ds")
    assert "joints=" in err.fix and "columns=" in err.fix
    assert "--column-names" in err.fix and "--state-columns" in err.fix
    options = {f.name for f in fields(FitConfig)}           # the fix names fit options that exist
    assert {"column_names", "state_columns"} <= options and "column_names=" in err.fix and "state_columns=" in err.fix


IL_STATE = ("arm/shoulder.q", "arm/shoulder.qd", "arm/elbow.q", "arm/elbow.qd")


def test_lerobot_il_recording_picks_the_position_columns_by_name(tmp_path):
    """An alloy.il recording: actions named by joint, a state of <joint>.q and <joint>.qd columns."""
    write_lerobot(tmp_path / "ds", state_width=4, names=("arm/shoulder", "arm/elbow"), state_names=IL_STATE)
    _, state, lengths = write_lerobot.expected
    rec = load_recording(tmp_path / "ds")
    assert rec.joints == ["shoulder", "elbow"]
    np.testing.assert_allclose(rec.episodes[0].position, state[: lengths[0]][:, [0, 2]], atol=1e-12)


def test_lerobot_actions_named_by_actuator_read_by_joint(tmp_path):
    write_lerobot(tmp_path / "ds", state_width=4, names=("arm/motor_e", "arm/motor_s"), state_names=IL_STATE)
    _, state, lengths = write_lerobot.expected
    rec = load_recording(tmp_path / "ds", aliases={"motor_s": "shoulder", "motor_e": "elbow"})
    assert rec.joints == ["elbow", "shoulder"]
    np.testing.assert_allclose(rec.episodes[0].position, state[: lengths[0]][:, [2, 0]], atol=1e-12)
    err = raises_field("data", load_lerobot, tmp_path / "ds")          # without the robot: no column of that name
    assert "state_columns=" in err.fix


def test_lerobot_unlabelled_or_ambiguous_state_is_not_guessed(tmp_path):
    write_lerobot(tmp_path / "a", state_width=4, names=("x", "y"), state_names=("x.q", "x.q", "y.q", "y.qd"))
    raises_field("data", load_lerobot, tmp_path / "a")
    write_lerobot(tmp_path / "b", state_width=4, names=("x", "y"))
    raises_field("data", load_lerobot, tmp_path / "b")


def test_lerobot_folder_without_info_json_does_not_need_pyarrow(tmp_path):
    raises_field("data", load_lerobot, tmp_path)
