# SPDX-License-Identifier: Apache-2.0
"""Teleop devices: keyboard bindings and hold rule, terminal reads, SpaceMouse reports, and the loop. No simulation."""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from alloy.api.errors import FeatureUnavailableError
from alloy.envs.errors import EnvError
from alloy.il import spacemouse as sm
from alloy.il import teleop
from alloy.il.teleop import KeyboardTeleop, ScriptKeys, TeleopCommand, TerminalKeys, teleop_loop


NAMES = ["gripper/lift", "gripper/close_left", "gripper/close_right"]


def keyboard(script: str, **kw) -> KeyboardTeleop:
    kw.setdefault("hold", 0.05)
    return KeyboardTeleop(NAMES, keys=ScriptKeys.parse(script), **kw)


def column(dev: KeyboardTeleop, steps: int, col: int = 0) -> list[float]:
    return [float(dev.read().action[col]) for _ in range(steps)]


def test_default_table_for_three_columns():
    table = teleop.default_bindings(3)
    assert table == {"w": (0, 1.0), "s": (0, -1.0), "a": (1, 1.0), "d": (1, -1.0), "q": (2, 1.0), "e": (2, -1.0)}
    text = keyboard("").help()
    assert "gripper/close_left" in text and "enter" in text and "esc" in text


def test_hold_window_is_n_steps_from_the_press():
    # hold=0.05, dt=0.01 -> n=5: pressed at step 0 drives steps 0..4, zero at step 5
    assert column(keyboard("w .*9"), 7) == [1, 1, 1, 1, 1, 0, 0]


def test_key_repeat_keeps_the_column_alive_and_zero_hold_is_one_step():
    assert column(keyboard("w*3 .*6"), 9) == [1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert column(keyboard("w .*3", hold=0.0), 3) == [1, 0, 0]


def test_latest_press_wins_on_one_column():
    assert column(keyboard("w .*1 s .*8"), 8) == [1, 1, -1, -1, -1, -1, -1, 0]


def test_link_copies_value_to_other_columns():
    dev = keyboard("a", link={"gripper/close_left": ["gripper/close_right"]})
    assert dev.read().action.tolist() == [0.0, 1.0, 1.0]


def test_space_zeroes_and_flags():
    dev = keyboard("w space .*1 w enter backspace esc")
    assert column(dev, 2) == [1, 0]
    assert dev.read().action[0] == 0.0
    assert dev.read().action[0] == 1.0
    assert (c := dev.read()).save and not c.discard and not c.quit
    assert dev.read().discard
    assert dev.read().quit
    assert dev.read().action.dtype == np.float32


def test_script_keys_parse_and_exhausted():
    keys = ScriptKeys.parse("w*2 .*1 enter")
    assert not keys.exhausted
    assert [keys.poll() for _ in range(4)] == [["w"], ["w"], [], ["enter"]]
    assert keys.exhausted
    assert ScriptKeys([(1, "x")]).length == 2


def test_decode_keys():
    assert teleop.decode_keys("w\x1b[A\x1b[D\r\x7f \x1b") == ["w", "up", "left", "enter", "backspace", "space", "esc"]


@pytest.mark.parametrize("kwargs", [
    {"bindings": {"w": ("nope", 1.0)}}, {"bindings": {"w": (7, 1.0)}}, {"bindings": {"w": (0, 1.5)}},
    {"bindings": {"enter": (0, 1.0)}}, {"link": {"gripper/lift": ["zzz"]}}, {"hold": -1.0}])
def test_keyboard_errors_have_a_fix(kwargs):
    with pytest.raises(EnvError) as err:
        KeyboardTeleop(NAMES, keys=ScriptKeys([]), **kwargs)
    assert err.value.fix


def test_more_than_nine_columns_need_bindings():
    with pytest.raises(EnvError) as err:
        KeyboardTeleop([f"c{i}" for i in range(10)], keys=ScriptKeys([]))
    assert "bindings" in err.value.fix
    KeyboardTeleop([f"c{i}" for i in range(10)], keys=ScriptKeys([]), bindings={"w": (9, 1.0)})


def test_terminal_keys_needs_a_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    with pytest.raises(EnvError) as err:
        TerminalKeys()
    assert "TTY" in err.value.message and err.value.fix
    with pytest.raises(EnvError):
        KeyboardTeleop(NAMES)


@pytest.mark.skipif(os.name == "nt", reason="pty is POSIX only")
def test_terminal_keys_reads_a_pty_and_restores_it(monkeypatch):
    import pty
    import termios

    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    with os.fdopen(slave, "r", closefd=False) as stdin:
        monkeypatch.setattr(sys, "stdin", stdin)
        with TerminalKeys() as keys:
            assert termios.tcgetattr(slave) != before          # cbreak is on
            assert keys.poll() == []
            os.write(master, b"w\x1b[A")
            got = []
            for _ in range(50):
                got += keys.poll()
                if len(got) >= 2:
                    break
        assert got == ["w", "up"]
        assert termios.tcgetattr(slave) == before
    os.close(master)
    os.close(slave)


# --- SpaceMouse --------------------------------------------------------------------------------------------------


def report(report_id: int, *values: int) -> bytes:
    return bytes([report_id]) + b"".join(v.to_bytes(2, "little", signed=True) for v in values)


def test_parse_report_ids():
    state = sm.SpaceMouseState()
    sm.parse_report(report(1, 10, -20, 30), state)
    assert state.raw == [10, -20, 30, 0, 0, 0]
    sm.parse_report(report(2, -1, 2, -3), state)
    assert state.raw == [10, -20, 30, -1, 2, -3]
    sm.parse_report(report(1, 1, 2, 3, 4, 5, 6), state)
    assert state.raw == [1, 2, 3, 4, 5, 6]
    sm.parse_report(report(3, 0)[:2] + b"", state)
    assert state.buttons == 0
    sm.parse_report(bytes([3, 3, 0]), state)
    assert state.buttons == 3
    sm.parse_report(b"", state)
    sm.parse_report(bytes([9, 1, 2]), state)
    sm.parse_report(bytes([1, 1]), state)           # too short: ignored
    assert state.raw == [1, 2, 3, 4, 5, 6]


def test_axis_values_deadband_and_clipping():
    state = sm.SpaceMouseState(raw=[350, -700, 10, 17, 18, -175])
    vals = sm.axis_values(state, 0.05)
    assert vals["x"] == 1.0 and vals["y"] == -1.0
    assert vals["z"] == 0.0 and vals["roll"] == 0.0           # 10/350 and 17/350 are inside 0.05
    assert vals["pitch"] == pytest.approx(18 / 350) and vals["yaw"] == -0.5


class FakeHid:
    def __init__(self, reports):
        self.reports, self.closed = list(reports), False

    def read(self, n):
        return list(self.reports.pop(0)) if self.reports else []

    def close(self):
        self.closed = True


def mouse(reports, **kw):
    return sm.SpaceMouseTeleop(NAMES, axes={"x": ("gripper/lift", 1.0), "yaw": (1, -0.5)},
                               buttons={"left": (2, 1.0)}, device=FakeHid(reports), **kw)


def test_fake_device_maps_to_action_and_both_buttons_save_once():
    dev = mouse([report(1, 175, 0, 0), report(2, 0, 0, 350), bytes([3, 1, 0])])
    assert dev.read().action.tolist() == [0.5, -0.5, 1.0]
    dev = mouse([bytes([3, 3, 0])])
    cmd = dev.read()
    assert cmd.save and cmd.action.tolist() == [0.0, 0.0, 0.0]
    assert not dev.read().save                                  # held: no second save
    dev.device.reports.append(bytes([3, 0, 0]))
    dev.read()
    dev.device.reports.append(bytes([3, 3, 0]))
    assert dev.read().save
    dev.close()
    assert dev.device.closed and "both btns" in dev.help()


def test_spacemouse_errors_have_a_fix(monkeypatch):
    with pytest.raises(EnvError) as err:
        sm.SpaceMouseTeleop(NAMES, axes={"w": (0, 1.0)}, device=FakeHid([]))
    assert err.value.fix
    monkeypatch.setitem(sys.modules, "hid", None)
    with pytest.raises(FeatureUnavailableError) as missing:
        sm.SpaceMouseTeleop(NAMES, axes={"x": (0, 1.0)})
    assert "pip install hidapi" in missing.value.fix


def test_no_matching_hid_device_lists_what_was_seen(monkeypatch):
    fake = types.SimpleNamespace(enumerate=lambda: [{"vendor_id": 0x1234, "product_id": 5, "product_string": "kbd"}])
    monkeypatch.setitem(sys.modules, "hid", fake)
    with pytest.raises(EnvError) as err:
        sm.open_hid_device(0x256F, None)
    assert "1234:0005" in err.value.message and err.value.fix


# --- the loop ----------------------------------------------------------------------------------------------------


class FakeSession:
    def __init__(self, num_envs=2, dt=0.01):
        self.num_envs, self.env = num_envs, types.SimpleNamespace(sim=types.SimpleNamespace(dt=dt))
        self.steps, self.finishes = [], []

    def step(self, actions):
        self.steps.append(actions)
        return ("obs", 0.0, False, False, {})

    def finish(self, save):
        self.finishes.append(save)


def test_loop_steps_tiles_and_counts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(teleop.time, "sleep", sleeps.append)
    dev = keyboard("w*2 enter w backspace w*2")
    session, seen = FakeSession(), []
    stats = teleop_loop(session, dev, on_step=lambda i, out: seen.append((i, out[0])))
    assert stats == {"steps": 5, "saved": 1, "discarded": 1}
    assert session.finishes == [True, False]
    assert all(a.shape == (2, 3) and a.dtype == np.float32 for a in session.steps)
    assert seen[0] == (0, "obs") and len(sleeps) == 7 and min(sleeps) >= 0.0


def test_loop_quit_drops_episode_and_realtime_off_never_sleeps(monkeypatch):
    monkeypatch.setattr(teleop.time, "sleep", lambda s: pytest.fail("slept"))
    closed = []
    dev = keyboard("w*3 esc w*3")
    dev.keys.close = lambda: closed.append(True)
    session = FakeSession()
    stats = teleop_loop(session, dev, realtime=False)
    assert stats["steps"] == 3 and session.finishes == [False] and closed == [True]


def test_loop_max_steps_and_keyboard_interrupt():
    class Boom:
        def read(self):
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    session = FakeSession()
    dev = sm.SpaceMouseTeleop(NAMES, axes={"x": (0, 1.0)}, device=FakeHid([]))
    assert teleop_loop(session, dev, realtime=False, max_steps=4)["steps"] == 4
    boom = Boom()
    teleop_loop(session, boom, realtime=False)
    assert session.finishes == [False] and boom.closed
    assert isinstance(TeleopCommand(np.zeros(1, np.float32)).save, bool)


def test_linked_columns_lose_their_default_keys_and_show_in_help():
    names = ["gripper/lift", "gripper/close_left", "gripper/close_right"]
    device = KeyboardTeleop(names, keys=ScriptKeys([]), link={"gripper/close_left": ["gripper/close_right"]})
    assert set(device.bindings) == {"w", "s", "a", "d"}          # q / e would drive a column the link overwrites
    assert "gripper/close_left + gripper/close_right" in device.help()


def test_loop_counts_what_the_recorder_wrote():
    class Session:
        num_envs, saved = 1, 0
        env = type("E", (), {"sim": type("S", (), {"dt": 0.01})()})()

        def step(self, actions):
            self.stepped = True
            return None

        def finish(self, save=True):
            self.saved += int(save and getattr(self, "stepped", False))
            self.stepped = False

    session = Session()
    keys = ScriptKeys.parse("enter w enter")                     # the first enter has no episode to save
    stats = teleop_loop(session, KeyboardTeleop(["a"], keys=keys), realtime=False)
    assert stats == {"steps": 1, "saved": 1, "discarded": 0}


def test_spacemouse_links_columns_and_takes_discard_and_quit_from_keys():
    held = bytes([3, 1, 0])                                      # left button down
    dev = sm.SpaceMouseTeleop(NAMES, axes={"z": ("gripper/lift", 1.0)}, buttons={"left": (1, 1.0), "right": (1, -1.0)},
                              link={"gripper/close_left": ["gripper/close_right"]}, device=FakeHid([held]),
                              keys=ScriptKeys.parse(". backspace space enter esc"))
    assert dev.read().action.tolist() == [0.0, 1.0, 1.0]         # one button, both fingers
    cmd = dev.read()
    assert cmd.discard and not cmd.save and not cmd.quit and cmd.action.tolist() == [0.0, 1.0, 1.0]
    assert dev.read().action.tolist() == [0.0, 0.0, 0.0]         # space zeroes this step
    assert dev.read().save and dev.read().quit
    assert "linked to gripper/close_left" in dev.help() and "backspace" in dev.help()
    plain = sm.SpaceMouseTeleop(NAMES, axes={"z": (0, 1.0)}, device=FakeHid([bytes([3, b]) for b in range(4)]))
    cmds = [plain.read() for _ in range(4)]
    assert not any(c.discard or c.quit for c in cmds) and "there is no discard" in plain.help()
    with pytest.raises(EnvError):
        sm.SpaceMouseTeleop(NAMES, axes={"z": (0, 1.0)}, link={"nope": [1]}, device=FakeHid([]))


def test_spacemouse_keys_end_and_discard_in_the_loop():
    dev = sm.SpaceMouseTeleop(NAMES, axes={"z": (0, 1.0)}, device=FakeHid([]),
                              keys=ScriptKeys.parse(".*3 backspace .*2 esc .*5"))
    session = FakeSession()
    stats = teleop_loop(session, dev, realtime=False)
    assert stats["steps"] == 5 and session.finishes == [False, False] and stats["discarded"] == 1
