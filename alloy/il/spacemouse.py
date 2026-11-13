# SPDX-License-Identifier: Apache-2.0
"""SpaceMouse teleoperation: a 3Dconnexion 6-axis mouse that fills the action row of an env.

The report parser is tested on synthetic reports; this device class has not been run on SpaceMouse hardware by
Alloy's tests. The real device needs the optional ``hid`` package (``pip install hidapi``). Example::

    from alloy.il.spacemouse import SpaceMouseTeleop
    device = SpaceMouseTeleop(env.action_names(), axes={"x": ("gripper/lift", 1.0)},
                              buttons={"left": ("gripper/close_left", 1.0)})
    recorder.reset(seed=0)                 # the loop steps the recorder: reset it first
    teleop_loop(recorder, device)          # both buttons together = save the episode
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.api.errors import FeatureUnavailableError
from alloy.envs.errors import EnvError
from alloy.il.teleop import (DISCARD_KEY, QUIT_KEY, SAVE_KEY, ZERO_KEY, TeleopCommand, check_value,
                             resolve_column)


AXIS_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
BUTTON_BITS = {"left": 1, "right": 2}
FULL_SCALE = 350.0          # raw int16 counts that read as 1.0
MAX_REPORTS_PER_READ = 64


@dataclass
class SpaceMouseState:
    """The latest raw axes (int16 counts, x y z roll pitch yaw) and button bitmask (bit 0 left, bit 1 right).

    Example: ``state = SpaceMouseState(); parse_report(bytes([1, 100, 0, 0, 0, 0, 0]), state); state.raw[0]``."""

    raw: list[int] = field(default_factory=lambda: [0] * 6)
    buttons: int = 0


def _int16s(payload: bytes, count: int) -> list[int]:
    """``count`` little-endian signed 16-bit integers from the start of ``payload``."""
    return [int.from_bytes(payload[2 * i:2 * i + 2], "little", signed=True) for i in range(count)]


def parse_report(data: bytes | Sequence[int], state: SpaceMouseState) -> None:
    """Update ``state`` from one HID report (little-endian int16). Report id 1: x, y, z (6 bytes) or, on newer
    models, x, y, z, roll, pitch, yaw (12 bytes); id 2: roll, pitch, yaw; id 3: button bitmask. Short or
    unknown reports are ignored.

    Example: ``parse_report(bytes([1, 0x5E, 0x01, 0, 0, 0, 0]), state)`` sets ``state.raw[0]`` to 350."""
    data = bytes(data)
    if not data:
        return
    report, payload = data[0], data[1:]
    if report == 1 and len(payload) >= 12:
        state.raw[:] = _int16s(payload, 6)
    elif report == 1 and len(payload) >= 6:
        state.raw[:3] = _int16s(payload, 3)
    elif report == 2 and len(payload) >= 6:
        state.raw[3:] = _int16s(payload, 3)
    elif report == 3 and len(payload) >= 1:
        state.buttons = payload[0]


def axis_values(state: SpaceMouseState, deadband: float) -> dict[str, float]:
    """Each axis as raw / 350 clipped to [-1, 1], zero inside the deadband.

    Example: ``axis_values(state, 0.05)["x"]`` -> ``1.0`` for a raw 350 or more."""
    values = np.clip(np.asarray(state.raw, np.float64) / FULL_SCALE, -1.0, 1.0)
    values[np.abs(values) < deadband] = 0.0
    return dict(zip(AXIS_NAMES, (float(v) for v in values)))


def open_hid_device(vendor_id: int, product_id: int | None) -> Any:
    """Open the first matching device through the ``hid`` package (``FeatureUnavailableError`` without it,
    ``EnvError`` listing what was found when nothing matches).

    Example: ``dev = open_hid_device(0x256F, None); dev.read(64)``."""
    try:
        import hid  # noqa: PLC0415
    except ImportError as exc:
        raise FeatureUnavailableError(
            "the SpaceMouse device needs the hid package (hidapi)",
            "pip install hidapi   (Linux also needs read access to the device: a udev rule for vendor 256f, "
            "or run as root)", field="hid") from exc
    found = list(hid.enumerate())
    matches = [d for d in found if d.get("vendor_id") == vendor_id
               and (product_id is None or d.get("product_id") == product_id)]
    if not matches:
        seen = [f"{d.get('vendor_id', 0):04x}:{d.get('product_id', 0):04x} {d.get('product_string', '')}".strip()
                for d in found] or ["no HID devices"]
        raise EnvError(f"no SpaceMouse found (vendor {vendor_id:#06x}); HID devices seen: {seen}",
                       "plug it in, check read access (udev rule for vendor 256f), or pass vendor_id=/product_id= "
                       "(older Logitech-branded models use 0x046D)", field="device")
    device = hid.device()
    device.open_path(matches[0]["path"])
    device.set_nonblocking(1)
    return device


class SpaceMouseTeleop:
    """A 3Dconnexion SpaceMouse as a teleop device: axes drive action columns, buttons hold a value, both buttons
    together save the episode (once per press).

    ``axes`` maps "x", "y", "z", "roll", "pitch", "yaw" to ``(column, scale)``; ``buttons`` maps "left"/"right" to
    ``(column, value)``, held while pressed. ``link={"gripper/close_left": ["gripper/close_right"]}`` copies a
    column's value to others (one button closes both fingers), as ``KeyboardTeleop``'s does. The device has two
    buttons, so discard and quit come from the keyboard: ``keys=TerminalKeys()`` (or ``ScriptKeys``) adds Enter
    (save), Backspace (discard), Esc (quit) and Space (zero the action for this step); without ``keys`` only
    save exists and the loop ends with Ctrl-C or ``max_steps``. ``device=`` injects any object with ``read(n)``
    and ``close()``.

    Example::

        device = SpaceMouseTeleop(names, axes={"z": ("gripper/lift", 1.0)}, buttons={"left": (1, 1.0)})
        cmd = device.read()"""

    def __init__(self, action_names: Sequence[str], *, axes: Mapping[str, tuple[int | str, float]],
                 buttons: Mapping[str, tuple[int | str, float]] | None = None, deadband: float = 0.05,
                 device: Any = None, vendor_id: int = 0x256F, product_id: int | None = None,
                 link: Mapping[int | str, Sequence[int | str]] | None = None, keys: Any = None) -> None:
        self.action_names = list(action_names)
        self.axes = self._resolve(axes, AXIS_NAMES, "axes")
        self.buttons = self._resolve(buttons or {}, tuple(BUTTON_BITS), "buttons")
        self.links = {resolve_column(self.action_names, src, "link"):
                      [resolve_column(self.action_names, dst, "link") for dst in dsts]
                      for src, dsts in (link or {}).items()}
        self.keys = keys
        self.deadband = deadband
        self.state = SpaceMouseState()
        self._both_down = False
        self.device = open_hid_device(vendor_id, product_id) if device is None else device

    def _resolve(self, raw: Mapping[str, tuple[int | str, float]], allowed: tuple[str, ...], field_name: str
                 ) -> dict[str, tuple[int, float]]:
        table: dict[str, tuple[int, float]] = {}
        for name, (ref, value) in raw.items():
            if name not in allowed:
                raise EnvError(f"unknown {field_name} key {name!r}", f"use one of {list(allowed)}",
                               field=field_name)
            if field_name == "buttons":
                check_value(value, f"button {name!r}", field_name)
            table[name] = (resolve_column(self.action_names, ref, field_name), float(value))
        return table

    def _drain(self) -> None:
        for _ in range(MAX_REPORTS_PER_READ):
            data = self.device.read(64)
            if not data:
                return
            parse_report(data, self.state)

    def read(self) -> TeleopCommand:
        """Drain pending reports and return this control step's command (call once per step).

        Example: ``cmd = device.read(); cmd.action.shape`` -> ``(A,)``."""
        self._drain()
        both = self.state.buttons & 3 == 3
        save, self._both_down = both and not self._both_down, both
        action = np.zeros(len(self.action_names), np.float64)
        values = axis_values(self.state, self.deadband)
        for name, (column, scale) in self.axes.items():
            action[column] += values[name] * scale
        for name, (column, value) in self.buttons.items():
            if not both and self.state.buttons & BUTTON_BITS[name]:
                action[column] += value
        pressed = set(self.keys.poll()) if self.keys is not None else set()
        if ZERO_KEY in pressed:
            action[:] = 0.0
        for source, targets in self.links.items():
            action[targets] = action[source]
        return TeleopCommand(np.clip(action, -1.0, 1.0).astype(np.float32), save=save or SAVE_KEY in pressed,
                             discard=DISCARD_KEY in pressed, quit=QUIT_KEY in pressed)

    def help(self) -> str:
        """A table of the mapped axes and buttons. Example: ``print(device.help())``."""
        rows = [f"{'input':<10}{'action column':<28}scale"]
        rows += [f"{n:<10}{self.action_names[c]:<28}{s:g}" for n, (c, s) in self.axes.items()]
        rows += [f"{'btn ' + n:<10}{self.action_names[c]:<28}{v:g}" for n, (c, v) in self.buttons.items()]
        rows += [f"{'  also':<10}{self.action_names[d]:<28}(linked to {self.action_names[s]})"
                 for s, dsts in self.links.items() for d in dsts]
        rows.append(f"{'both btns':<10}save the episode")
        if self.keys is not None:
            rows += [f"{SAVE_KEY:<10}save the episode", f"{DISCARD_KEY:<10}discard the episode",
                     f"{ZERO_KEY:<10}zero every column for this step", f"{QUIT_KEY:<10}quit"]
        else:
            rows.append("no keyboard: Ctrl-C quits and drops the running episode; there is no discard")
        return "\n".join(rows)

    def close(self) -> None:
        """Close the HID device. Example: ``device.close()``."""
        self.device.close()
        close = getattr(self.keys, "close", None)
        if close is not None:
            close()
