# SPDX-License-Identifier: Apache-2.0
"""The gift-bag tasks' stand-in robot: a gantry with a wrist and a two-finger gripper, and its scripted moves.

Joints (all position servos): ``gantry_x``, ``gantry_y``, ``gantry_z`` (slides, offsets from the palm's start),
``wrist_yaw`` (hinge about z under the palm), ``close_l`` / ``close_r`` (fingers squeezing across the wrist's x).
The fingers are 300 mm long, so the palm stays above a bag's rim while the fingertips set a carton on its bottom.
Links weigh at least the carton (fingers 0.5 kg): the lightest body a lagged pair carries onto the film sets the
exchange clock. The carton is held by friction only (pads mu 1.0, 15 mm of squeeze at kp 800: 12 N per finger).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FINGER_LEN = 0.30                       # palm to fingertip [m]
FINGER_GAP = 0.065                      # inner face of an open finger from the wrist axis [m]
FINGER_T = 0.004                        # finger half thickness [m]
HOLD = 0.25                             # palm above the carton's centre while held [m]
APPROACH = 0.15                         # palm starts this far above the holding height [m]


HOOK_TINE = 0.04                        # the hook's tine: from the shaft along the wrist's +x [m]
HOOK_T = 0.003                          # tine half thickness [m]


def _tool(tool: str, tine_width: float = 0.024) -> tuple[str, str, str]:
    """(wrist range, tool bodies under the wrist, tool actuators) for ``fingers`` or ``hook``."""
    t, fz, fh = FINGER_T, -0.5 * FINGER_LEN, 0.5 * FINGER_LEN
    pad = 'friction="1.0 0.005 0.0001"'
    if tool == "hook":                  # a 300 mm shaft and a 40 mm tine at its foot, along the wrist's +x
        geoms = (f'<geom name="hook_shaft" type="box" size="0.004 0.012 {fh}" pos="0 0 {fz}" mass="0.4"/>\n'
                 f'<geom name="hook_tine" type="box" size="{0.5 * HOOK_TINE} {0.5 * tine_width:g} {HOOK_T}" '
                 f'pos="{0.5 * HOOK_TINE} 0 {-FINGER_LEN + HOOK_T}" mass="0.1"/>')
        return "-3.3 3.3", geoms, ""
    if tool != "fingers":
        raise ValueError(f"tool {tool!r}: fingers or hook")
    bodies = (f'<body name="finger_l" pos="{-(FINGER_GAP + t)} 0 {fz}">'
              f'<joint name="close_l" type="slide" axis="1 0 0" range="0 0.06" damping="6"/>'
              f'<geom type="box" size="{t} 0.02 {fh}" mass="0.5" {pad}/></body>\n'
              f'<body name="finger_r" pos="{FINGER_GAP + t} 0 {fz}">'
              f'<joint name="close_r" type="slide" axis="-1 0 0" range="0 0.06" damping="6"/>'
              f'<geom type="box" size="{t} 0.02 {fh}" mass="0.5" {pad}/></body>')
    return "-1.2 1.2", bodies, "FINGERS"


def tool_boxes(tool: str, tine_width: float = 0.024) -> list[tuple[str, tuple, tuple]]:
    """The tool's collision boxes as (body under the gripper's name, centre in that body's frame, half sizes), the
    numbers of ``_tool``: the two fingers (their own bodies), or the hook's shaft and tine (on the wrist)."""
    if tool == "hook":
        fz = -0.5 * FINGER_LEN
        return [("wrist", (0.0, 0.0, fz), (0.004, 0.012, 0.5 * FINGER_LEN)),
                ("wrist", (0.5 * HOOK_TINE, 0.0, -FINGER_LEN + HOOK_T), (0.5 * HOOK_TINE, 0.5 * tine_width, HOOK_T))]
    if tool != "fingers":
        raise ValueError(f"tool {tool!r}: fingers or hook")
    half = (FINGER_T, 0.02, 0.5 * FINGER_LEN)
    return [("finger_l", (0.0, 0.0, 0.0), half), ("finger_r", (0.0, 0.0, 0.0), half)]


def gantry_mjcf(start: tuple[float, float, float], finger_kp: float = 800.0, tool: str = "fingers",
                tine_width: float = 0.024, wrist_armature: float = 0.0) -> str:
    """The gantry with its palm at ``start``; joint coordinates are offsets from there. ``tool``: the two-finger
    gripper, or a hook (a shaft with a ``tine_width`` wide tine at its foot, for pushing panels such as carton flaps,
    or holding a tape's tab). ``wrist_armature`` [kg·m²]: rotor inertia on the wrist, for a tool that a lagged force
    pulls off its axis (the wrist alone is ~3e-4)."""
    x, y, z = start
    wrange, bodies, acts = _tool(tool, tine_width)
    arm = f' armature="{wrist_armature:g}"' if wrist_armature else ""
    if acts:
        acts = (f'<position joint="close_l" kp="{finger_kp}" kv="20"/>'
                f'<position joint="close_r" kp="{finger_kp}" kv="20"/>')
    lim_xy = lim_z = ""
    if tool == "hook":                  # a pushing robot is force-limited (40 N; z also carries its own 34 N)
        lim_xy, lim_z = ' forcelimited="true" forcerange="-40 40"', ' forcelimited="true" forcerange="-75 75"'
    return f"""<mujoco model="gift_gantry"><compiler angle="radian"/><worldbody>
  <body name="carriage_x" pos="{x} {y} {z}">
    <joint name="gantry_x" type="slide" axis="1 0 0" range="-0.9 0.9" damping="80"/>
    <geom type="box" size="0.01 0.01 0.01" mass="2.0" contype="0" conaffinity="0"/>
    <body name="carriage_y">
      <joint name="gantry_y" type="slide" axis="0 1 0" range="-0.4 0.4" damping="80"/>
      <geom type="box" size="0.01 0.01 0.01" mass="2.0" contype="0" conaffinity="0"/>
      <body name="palm">
        <joint name="gantry_z" type="slide" axis="0 0 1" range="-0.6 0.8" damping="80"/>
        <geom type="box" size="0.085 0.085 0.01" mass="2.0" contype="0" conaffinity="0"/>
        <body name="wrist">
          <joint name="wrist_yaw" type="hinge" axis="0 0 1" range="{wrange}" damping="0.5"{arm}/>
          <geom type="cylinder" size="0.02 0.005" pos="0 0 -0.015" mass="1.0" contype="0" conaffinity="0"/>
          {bodies}
        </body>
      </body>
    </body>
  </body></worldbody>
  <actuator>
    <position joint="gantry_x" kp="3000" kv="300"{lim_xy}/><position joint="gantry_y" kp="3000" kv="300"{lim_xy}/>
    <position joint="gantry_z" kp="20000" kv="600"{lim_z}/><position joint="wrist_yaw" kp="60" kv="2"/>
    {acts}
  </actuator></mujoco>"""


@dataclass(frozen=True)
class Plan:
    """The pick-and-place moves, as offsets from the palm's start (over the carton's nominal start)."""
    to_x: float                         # x offset of the drop point (the bag's centre) [m]
    lift: float                         # carton bottom above its start while carried [m]
    drop: float                         # carton bottom above the counter at the release [m]
    squeeze: float                      # finger travel when closed [m]
    window: float                       # hands-off window [s]
    to_y: Any = 0.0                     # y of the drop point, or one per world (a deliberate miss, for controls) [m]


def schedule(p: Plan, starts: np.ndarray) -> list[tuple[str, float, dict]]:
    """(phase, seconds, targets) with ramps (start, end); ``starts`` [worlds, 3]: each carton's (dx, dy, yaw) from
    its nominal start. The wrist grasps at the carton's yaw and turns it square while carrying."""
    dx, dy, yaw = (np.asarray(starts[:, k], float) for k in range(3))
    zero = np.zeros_like(dx)
    hold = -APPROACH
    up, down, q = hold + p.lift, hold + p.drop, p.squeeze
    return [
        ("approach", 0.8, {"gantry_x": (zero, dx), "gantry_y": (zero, dy), "wrist_yaw": (zero, yaw)}),
        ("descend", 0.6, {"gantry_z": (0.0, hold)}),
        ("close", 0.6, {"close_l": (0.0, q), "close_r": (0.0, q)}),
        ("lift", 1.0, {"gantry_z": (hold, up)}),
        ("carry", 1.2, {"gantry_x": (dx, zero + p.to_x), "gantry_y": (dy, zero + p.to_y), "wrist_yaw": (yaw, zero)}),
        ("lower", 0.9, {"gantry_z": (up, hold + 0.06)}),
        ("place", 0.6, {"gantry_z": (hold + 0.06, down)}),
        ("settle", 0.3, {}),
        ("open", 0.4, {"close_l": (q, 0.0), "close_r": (q, 0.0)}),
        ("lift_out", 1.0, {"gantry_z": (down, hold + 0.40)}),
        ("aside", 0.6, {"gantry_x": (zero + p.to_x, zero)}),
        ("hands_off", p.window, {}),
    ]


def drive(sim, phases: list[tuple[str, float, dict]], on_phase=None, record=(), every: int = 0,
          watch=(), watch_prefix: str = "hands_off") -> list[dict]:
    """Runs ``phases`` on a built sim; ``on_phase(name, trajectory)`` after each. Rows: phase end time and the
    carton's height per world (frames too when ``every`` > 0). In the phases whose name starts with ``watch_prefix``
    the bodies ``watch`` are also recorded at every control period, as the row's ``"watch"`` frames."""
    rows = []
    for name, sec, targets in phases:
        if targets:
            sim.set_targets({k: v[0] for k, v in targets.items()})
        watching = bool(watch) and name.startswith(watch_prefix)
        tr = sim.run(sec, targets=targets or None, record=["candle", *record, *(watch if watching else ())],
                     every=1 if watching else max(every, 1))
        row = {"phase": name, "t_end": round(float(sim.t), 2),
               "candle_z": [round(float(z), 4) for z in tr["candle.pos"][-1, :, 2]]}
        if every:
            row["frames"] = {k: tr[k] for k in tr.keys()}
        if watching:
            row["watch"] = {k: tr[k] for k in tr.keys()}
        rows.append(row)
        if on_phase is not None:
            on_phase(name, tr)
    return rows


def fingers_clear(sim, points: dict[str, np.ndarray], margin: float = 0.005) -> list[bool]:
    """Per world: no point of ``points`` ({name: [worlds, n, 3]}) inside a finger's box grown by ``margin``."""
    st = sim.state(["gripper/finger_l", "gripper/finger_r"])
    half = np.array([FINGER_T, 0.02, 0.5 * FINGER_LEN]) + margin
    out = []
    for w in range(sim.worlds):
        pts = np.concatenate([p[w] for p in points.values()])
        clear = True
        for f in ("gripper/finger_l", "gripper/finger_r"):
            R = _rot(st[f + ".quat_wxyz"][w])
            local = (pts - st[f + ".pos"][w]) @ R
            clear &= not bool(np.any(np.all(np.abs(local) <= half, axis=1)))
        out.append(bool(clear))
    return out


def _rot(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ga_verdict(criteria: dict[str, bool | None], reasons: dict[str, str], port_checks: dict[str, bool],
               notes: tuple[str, ...] = ()) -> dict:
    """One world's honest verdict against the reference benchmark's (GA's) criteria.

    ``criteria``: one entry per GA criterion, by GA's name: True or False when judged from state, None when not
    judged (``reasons`` must then say why). ``port_checks``: checks the port makes that are not GA criteria.
    ``success`` is every judged GA criterion and every port check true; None criteria take no part in it."""
    missing = [k for k, v in criteria.items() if v is None and not reasons.get(k)]
    if missing:
        raise ValueError(f"GA criteria {missing} are not judged and have no reason")
    judged = [bool(v) for v in criteria.values() if v is not None]
    n, total = len(judged), len(criteria)
    return {"ga_criteria": {k: None if v is None else bool(v) for k, v in criteria.items()},
            "ga_not_judged": {k: reasons[k] for k, v in criteria.items() if v is None},
            "ga_judged": f"{n} of {total}", "ga_judged_n": n, "ga_total_n": total,
            "port_checks": {k: bool(v) for k, v in port_checks.items()}, "ga_notes": list(notes),
            "success": bool(judged and all(judged) and all(port_checks.values()))}
