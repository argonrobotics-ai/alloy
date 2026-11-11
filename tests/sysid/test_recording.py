# SPDX-License-Identifier: Apache-2.0
"""alloy.sysid.recording: make_episode checks, Recording.select / dt / duration, and the CSV round trip and errors."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.sysid.errors import SysidError
from alloy.sysid.recording import Recording, load_csv, make_episode, save_csv


def arrays(n: int = 20, joints: int = 2, dt: float = 0.01, t0: float = 0.0):
    t = t0 + dt * np.arange(n)
    rng = np.random.default_rng(1)
    return t, rng.normal(size=(n, joints)), rng.normal(size=(n, joints))


def make_recording(with_extra: bool = True) -> Recording:
    eps = []
    for k, n in enumerate((15, 25)):
        t, cmd, pos = arrays(n, 2, 0.02 * (k + 1))
        vel, tau = (cmd * 0.5, pos * 3.0) if with_extra else (None, None)
        eps.append(make_episode(t, cmd, pos, vel, tau))
    return Recording(["a", "b"], eps)


def raises_field(field: str, fn, *args):
    with pytest.raises(SysidError) as info:
        fn(*args)
    assert info.value.field == field
    assert info.value.message and info.value.fix
    return info.value


def test_make_episode_shifts_time_to_zero():
    t, cmd, pos = arrays(t0=5.0)
    ep = make_episode(t, cmd, pos)
    assert ep.t[0] == pytest.approx(0.0, abs=1e-12)
    assert ep.t[-1] == pytest.approx(0.19, abs=1e-12)
    assert ep.duration == pytest.approx(0.19, abs=1e-12)
    assert ep.dt == pytest.approx(0.01, abs=1e-12)


def test_make_episode_one_dimensional_arrays_become_one_joint():
    t = np.arange(12) * 0.01
    ep = make_episode(t, np.zeros(12), np.ones(12), np.zeros(12), np.zeros(12))
    assert ep.command.shape == ep.position.shape == ep.velocity.shape == ep.torque.shape == (12, 1)


def test_make_episode_keeps_optional_columns_none():
    t, cmd, pos = arrays()
    ep = make_episode(t, cmd, pos)
    assert ep.velocity is None and ep.torque is None


def test_make_episode_shape_mismatch_names_the_array():
    t, cmd, pos = arrays()
    raises_field("command", make_episode, t, cmd[:-1], pos)
    raises_field("velocity", make_episode, t, cmd, pos, np.zeros((20, 3)))


def test_make_episode_column_count_mismatch():
    t, cmd, pos = arrays(joints=2)
    raises_field("command", make_episode, t, cmd[:, :1], pos)


def test_make_episode_wrong_length_position_is_a_sysid_error():
    t, cmd, pos = arrays()
    raises_field("position", make_episode, t, cmd, pos[:-1])


def test_make_episode_wrong_length_one_dimensional_command_is_a_sysid_error():
    t = np.arange(12) * 0.01
    raises_field("command", make_episode, t, np.zeros(5), np.zeros(12))


def test_make_episode_rejects_nan_and_inf():
    t, cmd, pos = arrays()
    bad = pos.copy()
    bad[3, 1] = np.nan
    raises_field("position", make_episode, t, cmd, bad)
    bad = cmd.copy()
    bad[0, 0] = np.inf
    raises_field("command", make_episode, t, bad, pos)


def test_make_episode_too_few_samples():
    t, cmd, pos = arrays(n=9)
    raises_field("t", make_episode, t, cmd, pos)


def test_make_episode_time_must_rise_strictly():
    t, cmd, pos = arrays()
    t[7] = t[6]
    raises_field("t", make_episode, t, cmd, pos)
    t[7] = t[6] - 0.001
    raises_field("t", make_episode, t, cmd, pos)
    t, cmd, pos = arrays()
    t[4] = np.nan
    raises_field("t", make_episode, t, cmd, pos)


def test_recording_dt_and_duration_over_episodes():
    rec = make_recording()
    assert rec.duration == pytest.approx(14 * 0.02 + 24 * 0.04, abs=1e-12)
    # 14 diffs of 0.02 and 24 of 0.04: the median is 0.04
    assert rec.dt == pytest.approx(0.04, abs=1e-12)


def test_select_reorders_and_keeps_extras():
    rec = make_recording()
    sel = rec.select(["b", "a"])
    assert sel.joints == ["b", "a"]
    for old, new in zip(rec.episodes, sel.episodes):
        np.testing.assert_array_equal(new.command[:, 0], old.command[:, 1])
        np.testing.assert_array_equal(new.position[:, 1], old.position[:, 0])
        np.testing.assert_array_equal(new.velocity[:, 0], old.velocity[:, 1])
        np.testing.assert_array_equal(new.torque[:, 1], old.torque[:, 0])
    assert rec.joints == ["a", "b"]


def test_select_unknown_joint():
    err = raises_field("joints", make_recording().select, ["a", "elbow"])
    assert "elbow" in err.message
    assert "['a', 'b']" in err.fix


def test_csv_round_trip_is_exact_with_velocity_torque_and_two_episodes(tmp_path):
    rec = make_recording()
    path = save_csv(rec, tmp_path / "sub" / "take.csv")
    back = load_csv(path)
    assert back.joints == rec.joints
    assert len(back.episodes) == 2
    for old, new in zip(rec.episodes, back.episodes):
        for name in ("t", "command", "position", "velocity", "torque"):
            np.testing.assert_allclose(getattr(new, name), getattr(old, name), atol=1e-12, rtol=0.0)
    assert back.source == str(path)


def test_csv_round_trip_without_extras_has_no_velocity_columns(tmp_path):
    rec = make_recording(with_extra=False)
    path = save_csv(rec, tmp_path / "take.csv")
    header = path.read_text().splitlines()[0].split(",")
    assert header[0] == "t" and header[-1] == "episode"
    assert not any(h.startswith(("vel.", "tau.")) for h in header)
    back = load_csv(path)
    assert back.episodes[0].velocity is None and back.episodes[0].torque is None


def write(tmp_path, text: str):
    path = tmp_path / "x.csv"
    path.write_text(text)
    return path


def good_rows(n: int = 12) -> str:
    return "".join(f"{0.01 * i},0.1,0.2\n" for i in range(n))


def test_load_csv_no_data_rows(tmp_path):
    raises_field("data", load_csv, write(tmp_path, "t,cmd.a,pos.a\n"))
    raises_field("data", load_csv, write(tmp_path, ""))


def test_load_csv_missing_t_column(tmp_path):
    raises_field("data", load_csv, write(tmp_path, "time,cmd.a,pos.a\n" + good_rows()))


def test_load_csv_without_pos_columns(tmp_path):
    raises_field("data", load_csv, write(tmp_path, "t,cmd.a,x.a\n" + good_rows()))


def test_load_csv_pos_without_cmd(tmp_path):
    err = raises_field("data", load_csv, write(tmp_path, "t,cmd.b,pos.a\n" + good_rows()))
    assert "'a'" in err.message


def test_load_csv_non_number_cell(tmp_path):
    raises_field("data", load_csv, write(tmp_path, "t,cmd.a,pos.a\n" + good_rows() + "0.5,abc,0.1\n"))


def test_load_csv_ragged_rows(tmp_path):
    raises_field("data", load_csv, write(tmp_path, "t,cmd.a,pos.a\n" + good_rows() + "0.5,0.1\n"))


def test_load_csv_header_whitespace_and_blank_lines_are_tolerated(tmp_path):
    path = write(tmp_path, "t , cmd.a , pos.a\n" + good_rows() + "\n")
    rec = load_csv(path)
    assert rec.joints == ["a"]
    assert rec.episodes[0].position.shape == (12, 1)


def test_load_csv_split_by_episode_column_orders_and_resets_time(tmp_path):
    rows = "".join(f"{100 + 0.01 * i},0.0,0.0,1\n" for i in range(10))
    rows += "".join(f"{0.01 * i},0.0,0.0,0\n" for i in range(11))
    rec = load_csv(write(tmp_path, "t,cmd.a,pos.a,episode\n" + rows))
    assert [len(e.t) for e in rec.episodes] == [11, 10]
    assert all(e.t[0] == 0.0 for e in rec.episodes)
