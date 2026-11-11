# SPDX-License-Identifier: Apache-2.0
"""Keyboard teleoperation: turn key presses into one action row per control step, and a loop that records them.

A device fills the action columns of an Alloy env (``env.action_names()``); ``teleop_loop`` feeds them to a
session (an ``alloy.il.Recorder``) at the env's control rate. The keyboard reads the terminal, so it works over
ssh with no display. Terminals send no key-up events, so a pressed key drives its column for ``hold`` seconds
after its last press event (OS key repeat keeps it alive while held). Time is counted in control steps, not wall
clock, so a run is repeatable and testable. Example::

    from alloy.il.teleop import KeyboardTeleop, teleop_loop
    device = KeyboardTeleop(env.action_names(), hold=0.15, dt=env.sim.dt)
    print(device.help())
    recorder.reset(seed=0)                 # the loop steps the recorder: reset it first
    teleop_loop(recorder, device)          # enter = save episode, backspace = discard, esc = quit
"""

from __future__ import annotations

import atexit
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError


SAVE_KEY, DISCARD_KEY, QUIT_KEY, ZERO_KEY = "enter", "backspace", "esc", "space"
RESERVED_KEYS = (SAVE_KEY, DISCARD_KEY, QUIT_KEY, ZERO_KEY)
DEFAULT_PAIRS = (("w", "s"), ("a", "d"), ("q", "e"), ("r", "f"), ("t", "g"), ("y", "h"), ("u", "j"), ("i", "k"),
                 ("o", "l"))
_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}
_WINDOWS_ARROWS = {"H": "up", "P": "down", "M": "right", "K": "left"}
_CHAR_NAMES = {"\r": SAVE_KEY, "\n": SAVE_KEY, "\x7f": DISCARD_KEY, "\x08": DISCARD_KEY, " ": ZERO_KEY}


@dataclass
class TeleopCommand:
    """What a device wants for one control step: an action row plus episode flags.

    Example: ``TeleopCommand(action=np.zeros(3, np.float32), save=True)`` ends the episode and keeps it."""

    action: np.ndarray          # (A,) float32 in [-1, 1]
    save: bool = False          # end the episode now and keep it
    discard: bool = False       # end the episode now and drop it
    quit: bool = False          # stop the session


def resolve_column(action_names: Sequence[str], ref: int | str, field: str = "bindings") -> int:
    """The action column index for a column name or index; raises ``EnvError`` listing the valid ones.

    Example: ``resolve_column(["g/lift", "g/close"], "g/close")`` -> 1."""
    count = len(action_names)
    if isinstance(ref, str):
        if ref not in action_names:
            raise EnvError(f"unknown action column {ref!r}", f"use one of {list(action_names)} or an index",
                           field=field)
        return list(action_names).index(ref)
    if isinstance(ref, bool) or not isinstance(ref, (int, np.integer)) or not 0 <= ref < count:
        raise EnvError(f"action column {ref!r} is out of range for {count} columns",
                       f"use an index in 0..{count - 1} or a name from env.action_names()", field=field)
    return int(ref)


def check_value(value: float, what: str, field: str = "bindings") -> float:
    """``value`` as a float in [-1, 1], else ``EnvError``. Example: ``check_value(0.5, "w")`` -> 0.5."""
    if not -1.0 <= float(value) <= 1.0:
        raise EnvError(f"{what} has value {value}, outside [-1, 1]", "use a value in [-1, 1]", field=field)
    return float(value)



def decode_keys(text: str) -> list[str]:
    """Key names from raw terminal text: arrow escape sequences, enter, backspace, space, esc and plain characters.

    Example: ``decode_keys("w\\x1b[A")`` -> ``["w", "up"]``."""
    keys, i = [], 0
    while i < len(text):
        if text.startswith("\x1b[", i) and i + 2 < len(text):
            keys.append(_ARROWS.get(text[i + 2], "esc"))
            i += 3
            continue
        char = text[i]
        keys.append("esc" if char == "\x1b" else _CHAR_NAMES.get(char, char))
        i += 1
    return keys


class ScriptKeys:
    """Keys played from a list, one poll per control step: for tests and unattended runs.

    ``poll()`` call k (counting from 0) returns the keys scheduled for step k. Example::

        keys = ScriptKeys.parse("w*20 e*10 enter")     # w on steps 0-19, e on 20-29, enter on step 30
        device = KeyboardTeleop(names, keys=keys)"""

    def __init__(self, events: Sequence[tuple[int, str]], length: int | None = None) -> None:
        self._events: dict[int, list[str]] = {}
        for step, key in events:
            self._events.setdefault(int(step), []).append(key)
        last = max(self._events, default=-1) + 1
        self.length = max(last, 0 if length is None else int(length))
        self.calls = 0

    @classmethod
    def parse(cls, script: str) -> ScriptKeys:
        """``"w*20 e*10 enter"``: ``key*count`` presses on that many consecutive steps, ``.*n`` is n idle steps.

        Example: ``ScriptKeys.parse("w*3 .*2 enter")`` presses w on steps 0-2, idles 3-4, enter on 5."""
        events: list[tuple[int, str]] = []
        step = 0
        for token in script.split():
            key, _, count = token.partition("*")
            repeat = int(count) if count else 1
            if key != ".":
                events.extend((step + k, key) for k in range(repeat))
            step += repeat
        return cls(events, length=step)

    @property
    def exhausted(self) -> bool:
        """True once every scheduled step has been polled. Example: ``ScriptKeys([]).exhausted`` is True."""
        return self.calls >= self.length

    def poll(self) -> list[str]:
        """The keys for the current call number. Example: ``ScriptKeys([(0, "w")]).poll()`` -> ``["w"]``."""
        keys = list(self._events.get(self.calls, []))
        self.calls += 1
        return keys

    def close(self) -> None:
        """Nothing to release. Example: ``keys.close()``."""

class TerminalKeys:
    """Keys from the real terminal (stdin in cbreak mode); the terminal is restored by ``close()``, the context
    manager, or at exit. Raises ``EnvError`` when stdin is not a TTY.

    Example::

        with TerminalKeys() as keys:
            print(keys.poll())          # ["w", "up"] if those were pressed since the last call"""

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise EnvError("keyboard teleop needs a terminal (stdin is not a TTY)",
                           "run it in a terminal, or pass keys=ScriptKeys([...]) to play keys from a list",
                           field="keys")
        self._saved: Any = None
        self._fd = -1
        if os.name != "nt":
            import termios  # noqa: PLC0415
            import tty  # noqa: PLC0415

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            atexit.register(self.close)

    def poll(self) -> list[str]:
        """Keys pressed since the last call, without blocking. Example: ``keys.poll()`` -> ``["w"]``."""
        return self._poll_windows() if os.name == "nt" else self._poll_posix()

    def _poll_posix(self) -> list[str]:
        import select  # noqa: PLC0415
        if self._saved is None:
            return []
        data = b""
        while select.select([self._fd], [], [], 0)[0]:
            chunk = os.read(self._fd, 1024)
            if not chunk:
                break
            data += chunk
        return decode_keys(data.decode("utf-8", errors="ignore"))

    @staticmethod
    def _poll_windows() -> list[str]:  # pragma: no cover - needs Windows
        import msvcrt  # noqa: PLC0415
        keys: list[str] = []
        while msvcrt.kbhit():
            char = msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                keys.append(_WINDOWS_ARROWS.get(msvcrt.getwch(), "esc"))
            else:
                keys.extend(decode_keys(char))
        return keys

    def close(self) -> None:
        """Restore the terminal's settings (safe to call twice). Example: ``keys.close()``."""
        if self._saved is None:
            return
        import termios  # noqa: PLC0415
        saved, self._saved = self._saved, None
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, saved)
        except termios.error:
            pass

    def __enter__(self) -> TerminalKeys:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()



def default_bindings(count: int) -> dict[str, tuple[int, float]]:
    """The default key table for ``count`` columns: column i gets ``DEFAULT_PAIRS[i]`` (first key +1, second -1).

    Example: ``default_bindings(3)`` -> ``{"w": (0, 1.0), "s": (0, -1.0), "a": (1, 1.0), ...}``."""
    if count > len(DEFAULT_PAIRS):
        raise EnvError(f"{count} action columns are more than the {len(DEFAULT_PAIRS)} default key pairs",
                       "pass bindings={'key': (column, value), ...} for the columns you drive", field="bindings")
    table: dict[str, tuple[int, float]] = {}
    for column, (plus, minus) in enumerate(DEFAULT_PAIRS[:count]):
        table[plus], table[minus] = (column, 1.0), (column, -1.0)
    return table


class KeyboardTeleop:
    """A keyboard that fills the action row of an env: a pressed key drives its column for ``hold`` seconds.

    Hold rule: a key polled at control step k drives its column on steps k .. k+n-1 with ``n = max(1,
    round(hold / dt))`` and the column is 0 again at step k+n unless pressed again; the latest press wins when
    several keys share a column. Fixed keys: enter = save, backspace = discard, esc = quit, space = zero all.

    Example::

        device = KeyboardTeleop(["gripper/lift", "gripper/close_left", "gripper/close_right"],
                                link={"gripper/close_left": ["gripper/close_right"]}, dt=0.01)
        cmd = device.read()          # TeleopCommand: cmd.action is (3,) float32"""

    def __init__(self, action_names: Sequence[str], *,
                 bindings: Mapping[str, tuple[int | str, float]] | None = None, keys: Any = None,
                 hold: float = 0.15, dt: float = 0.01, link: Mapping[int | str, Sequence[int | str]] | None = None
                 ) -> None:
        if dt <= 0 or hold < 0:
            raise EnvError(f"hold={hold} and dt={dt} must be hold >= 0 and dt > 0", "pass seconds", field="hold")
        self.action_names = list(action_names)
        raw = default_bindings(len(self.action_names)) if bindings is None else bindings
        self.bindings = self._resolve(raw)
        self.links = self._resolve_links(link or {})
        if bindings is None:                 # a linked column follows its source: its default keys would do nothing
            followers = {dst for dsts in self.links.values() for dst in dsts}
            self.bindings = {k: b for k, b in self.bindings.items() if b[0] not in followers}
        self.hold_steps = max(1, round(hold / dt))
        self.keys = TerminalKeys() if keys is None else keys
        self._step = 0
        self._press = 0                      # presses so far; _order[key] is the latest press's number
        self._until: dict[str, int] = {}     # key -> first step it no longer drives its column
        self._order: dict[str, int] = {}

    def _resolve(self, raw: Mapping[str, tuple[int | str, float]]) -> dict[str, tuple[int, float]]:
        table: dict[str, tuple[int, float]] = {}
        for key, (ref, value) in raw.items():
            if key in RESERVED_KEYS:
                raise EnvError(f"key {key!r} is reserved ({', '.join(RESERVED_KEYS)})",
                               "bind a different key", field="bindings")
            column = resolve_column(self.action_names, ref)
            table[key] = (column, check_value(value, f"key {key!r}"))
        return table

    def _resolve_links(self, link: Mapping[int | str, Sequence[int | str]]) -> dict[int, list[int]]:
        return {resolve_column(self.action_names, src, "link"):
                [resolve_column(self.action_names, dst, "link") for dst in dsts] for src, dsts in link.items()}

    def _action(self) -> np.ndarray:
        action = np.zeros(len(self.action_names), np.float32)
        newest = dict.fromkeys(range(action.size), 0)
        for key, until in self._until.items():
            column, value = self.bindings[key]
            if until > self._step and self._order[key] > newest[column]:
                newest[column], action[column] = self._order[key], value
        for src, dsts in self.links.items():
            action[dsts] = action[src]
        return action

    def read(self) -> TeleopCommand:
        """Poll the keys and return this control step's command (call once per step).

        Example: ``cmd = device.read(); env.step(np.tile(cmd.action, (num_envs, 1)))``."""
        flags: set[str] = set()
        for key in self.keys.poll():
            if key == ZERO_KEY:
                self._until.clear()
            elif key in RESERVED_KEYS:
                flags.add(key)
            elif key in self.bindings:
                self._press += 1
                self._until[key], self._order[key] = self._step + self.hold_steps, self._press
        cmd = TeleopCommand(self._action(), save=SAVE_KEY in flags, discard=DISCARD_KEY in flags,
                            quit=QUIT_KEY in flags)
        self._step += 1
        return cmd

    def help(self) -> str:
        """A table of keys, the column each drives and the direction. Example: ``print(device.help())``."""
        rows = [f"{'key':<10}{'action column':<28}direction"]
        for key, (column, value) in self.bindings.items():
            names = " + ".join(self.action_names[c] for c in (column, *self.links.get(column, ())))
            rows.append(f"{key:<10}{names:<28} {value:+g}")
        rows += [f"{SAVE_KEY:<10}save the episode and start the next", f"{DISCARD_KEY:<10}discard the episode",
                 f"{ZERO_KEY:<10}zero every column now", f"{QUIT_KEY:<10}quit"]
        return "\n".join(rows)

    def close(self) -> None:
        """Release the key source (restores the terminal). Example: ``device.close()``."""
        close = getattr(self.keys, "close", None)
        if close is not None:
            close()



def teleop_loop(session: Any, device: Any, *, realtime: bool = True, max_steps: int | None = None,
                on_step: Callable[[int, Any], None] | None = None) -> dict[str, int]:
    """Drive ``session`` (an ``alloy.il.Recorder``) from ``device`` until quit, ``max_steps`` periods, or the
    device's ``ScriptKeys`` run out. Save/discard end the running episode; quit (or Ctrl-C) drops it. The device is
    always closed. Returns ``{"steps": n, "saved": k, "discarded": m}``.

    Call ``session.reset()`` first (the loop raises "step before reset" otherwise).

    Example: ``recorder.reset(); teleop_loop(recorder, KeyboardTeleop(env.action_names(), dt=env.sim.dt))``."""
    dt, deadline = float(session.env.sim.dt), time.perf_counter()
    stats = {"steps": 0, "saved": 0, "discarded": 0}
    try:
        for index in (count() if max_steps is None else range(max_steps)):
            if getattr(getattr(device, "keys", None), "exhausted", False):
                break
            cmd = device.read()
            if cmd.quit:
                session.finish(save=False)
                break
            _apply(session, cmd, stats, index, on_step)
            if realtime:
                deadline += dt
                time.sleep(max(0.0, deadline - time.perf_counter()))
    except KeyboardInterrupt:
        session.finish(save=False)
    finally:
        device.close()
    return stats


def _apply(session: Any, cmd: TeleopCommand, stats: dict[str, int], index: int,
           on_step: Callable[[int, Any], None] | None) -> None:
    """Run one command on the session and update the counters."""
    if cmd.save or cmd.discard:
        before = getattr(session, "saved", None)
        session.finish(save=cmd.save)
        wrote = 1 if before is None else int(session.saved - before)       # a Recorder counts what it wrote
        stats["saved" if cmd.save else "discarded"] += wrote if cmd.save else 1
        return
    out = session.step(np.tile(cmd.action, (session.num_envs, 1)))
    stats["steps"] += 1
    if on_step is not None:
        on_step(index, out)
