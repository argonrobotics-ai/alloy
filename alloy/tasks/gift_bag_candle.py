# SPDX-License-Identifier: Apache-2.0
"""Bag a boxed candle: put it in the already-open kraft gift bag, upright on the bottom board, and leave the bag
standing open on the mat (the reference task set's gift-bags bag-candle task; checks in gift_bag_checks.py).

Scene (public API, counter top at z = 0):

- a 400 x 300 x 3 mm rubber mat (the reference counter's);
- ``gift_bag_kraft``: the open kraft gift bag (Uline S-7098, 203 x 114 x 260 mm, 100 g/m² kraft; film route with fold
  lines as crease hinges; gift_bag_mesh.py), standing on the mat;
- ``gift_bag_board``: a 200 x 100 x 0.51 mm chipboard bottom board lying on the bag's bottom panel;
- the boxed candle: a 90 x 90 x 110 mm carton, 420 g (the reference's ESTIMATE, a rigid box) on the counter beside the
  mat; each world moves it by up to ±20 mm and turns it by up to ±30 deg (the reference's seeds), drawn from ``seed``;
- the gantry gripper (gift_gantry.py), standing in for the robot; the carton is held by friction only.

Scripted demo (gift_gantry.schedule): over the carton, turn the wrist to it, descend, close, lift it over the rim,
carry it over the mouth while turning it square (turned 30 deg a 90 mm carton is 123 mm across, the bag 114 mm deep),
lower it to a few mm above the board, open, lift out and move aside, then the hands-off window (3 s, the reference's)
in which the checks are read.

Verdict against the reference benchmark ("GA", Gym-Anything gift-bags bag-candle, 11 criteria: its bag-purchase checks,
the tear check and the hands-off window). Each world's ``ga_criteria`` has one entry per GA criterion, true or false
when judged from state with GA's threshold, None when not. Judged (10 of 11): hands_off_window, mouth_open, standing,
bottom_squared, on_mat, base_at_rest (bottom drift over the last 2 s; the window is run as 1 s + 2 s to take that mark),
board_in_bag, candle_in_bag, candle_upright, candle_on_board. Not judged (``ga_not_judged``): bag_not_torn (the film
route records no crack length).

``hands_off_window`` (gift_hands_off.py): GA fails it if any robot link carries a reaction force above 1e-3 N at any of
the window's 300 steps. The public API reads no contact forces, so the window is recorded at every control period (0.01
s; the bag, board, carton and the two fingers) and each finger box is tested against the bag's and board's vertices,
the carton's box, the mat and the counter at a 5 mm distance (the engine's robot contact gap). That is a distance, not
a force: a finger within 5 mm without load counts as contact, and a graze of the film between two of its ~12 mm cells
can be missed. The palm, wrist and carriages have no collision shapes. This replaces the former port check
``fingers_clear_at_end`` (an end-state test of the same distance), so there is no port check. The film route needs a
GPU: the window test is covered by synthetic frames on the CPU (tests/tasks/test_gift_bag.py), not run on the demo here.
``success`` is every judged criterion true; it is not GA's success, which needs all 11.

Example::

    from alloy.tasks.gift_bag_candle import CandleSpec, run
    out = run(CandleSpec(), device="cuda:0", worlds=4)
    print(out["success_rate"], out["ga_judged"], out["worlds"][0])

Command line::

    python -m alloy.tasks.gift_bag_candle --device cuda:0 --worlds 4 --out bag_candle.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from .gift_bag_checks import evaluate, layout
from .gift_bag_mesh import BagMeshSpec, bag_mesh
from .gift_gantry import APPROACH, HOLD, Plan, drive, gantry_mjcf, schedule, tool_boxes
from .gift_hands_off import (MARGIN, WorldBox, contact_track, join_frames, tool_world_boxes,
                             window_verdict)

KRAFT_H = 135e-6                        # the bag's paper (gift_bag_kraft card)
BOARD_H = 0.00051                       # gift_bag_board's caliper
MAT = (0.4, 0.3, 0.003)                 # full sizes [m]
REST_ON_MAT = 1.0e-4                    # GA's rest height of 135 um kraft on a rigid surface, (1 - phi/2) t [m]
REST_WINDOW = 2.0                       # GA judges base_at_rest over the last 2 s of the window [s]


@dataclass(frozen=True)
class CandleSpec:
    candle: tuple = (0.09, 0.09, 0.11)  # full edges [m]: the reference's 90 x 90 x 110 mm SBS carton
    candle_mass: float = 0.42           # kg (the reference's ESTIMATE)
    candle_friction: float = 0.4        # carton board on the fingers' pads
    candle_x: float = 0.30              # nominal start on the counter beside the mat, on the centre line [m]: turned
                                        # 30 deg and moved 20 mm toward it, a carton clears the mat by 8.5 mm
    jitter_xy: float = 0.02             # each world's start moved by up to this in x and y [m]
    jitter_yaw_deg: float = 30.0        # and turned by up to this
    finger_kp: float = 800.0            # squeeze: kp x 15 mm = 12 N per finger
    squeeze: float = 0.035              # finger travel when closed [m] (20 mm to the carton, 15 mm past its face)
    lift: float = 0.32                  # carton bottom above its start while carried [m] (rim at 0.26)
    drop: float = 0.008                 # carton bottom above the counter at the release [m] (board top ~4 mm)
    window: float = 3.0                 # hands-off window [s]
    dt: float = 0.01                    # control period [s]


def starts(s: CandleSpec, worlds: int, seed: int) -> np.ndarray:
    """[worlds, 3] each carton's (dx, dy, yaw) from the nominal start, uniform in the jitter ranges."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-s.jitter_xy, s.jitter_xy, (worlds, 2))
    yaw = np.radians(rng.uniform(-s.jitter_yaw_deg, s.jitter_yaw_deg, worlds))
    return np.column_stack([xy, yaw])


def _heights(s: CandleSpec) -> dict[str, float]:
    z_bag = MAT[2] + 0.5 * KRAFT_H + 1.0e-4
    z_board = z_bag + 0.5 * KRAFT_H + 0.5 * BOARD_H + 3.0e-4
    return {"bag": z_bag, "board": z_board, "candle": 0.5 * s.candle[2]}


def palm_start(s: CandleSpec) -> tuple[float, float, float]:
    return (s.candle_x, 0.0, 0.5 * s.candle[2] + HOLD + APPROACH)


def build_scene(s: CandleSpec, st: np.ndarray, with_bag: bool = True):
    """The task's scene for the starts ``st``; ``with_bag=False`` leaves out the bag and board (the script alone)."""
    import alloy

    h = _heights(s)
    scene = alloy.Scene(worlds=len(st), dt=s.dt)
    scene.add_ground()
    scene.add_box("mat", size=MAT, pos=(0.0, 0.0, 0.5 * MAT[2]), static=True, friction=0.9)
    if with_bag:
        scene.add_object("bag", "gift_bag_kraft", pos=(0.0, 0.0, h["bag"]))
        scene.add_object("board", "gift_bag_board", pos=(0.0, 0.0, h["board"]))
    pos = np.column_stack([s.candle_x + st[:, 0], st[:, 1], np.full(len(st), h["candle"])])
    scene.add_box("candle", size=s.candle, pos=pos, mass=s.candle_mass, friction=s.candle_friction)
    scene.add_robot("gripper", mjcf=gantry_mjcf(palm_start(s), s.finger_kp))
    return scene


def build(s: CandleSpec, worlds: int = 1, seed: int = 0, device: str | None = None, with_bag: bool = True):
    """(sim, starts): the built scene with each world's carton turned to its start yaw."""
    import alloy

    st = starts(s, worlds, seed)
    sim = alloy.build(build_scene(s, st, with_bag), device=device, seed=seed)
    z = _heights(s)["candle"]
    for w in range(worlds):
        half = 0.5 * st[w, 2]
        sim.set_pose("candle", pos=(s.candle_x + st[w, 0], st[w, 1], z),
                     quat_wxyz=(math.cos(half), 0.0, 0.0, math.sin(half)), world=w)
    return sim, st


def plan(s: CandleSpec) -> Plan:
    return Plan(to_x=-s.candle_x, lift=s.lift, drop=s.drop, squeeze=s.squeeze, window=s.window)


def _layout(s: CandleSpec):
    m = bag_mesh(BagMeshSpec())
    return layout(BagMeshSpec(), (0.0, 0.0), (0.5 * MAT[0], 0.5 * MAT[1]), BOARD_H, s.candle, m.bottom, m.rim,
                  m.triangles, MAT[2] + REST_ON_MAT)


def split_window(phases: list, last: float = REST_WINDOW) -> tuple[list, str]:
    """``phases`` with the closing hands-off window cut in two so that the state ``last`` s before the end can be
    marked; (phases, name of the phase after which to mark). A window of ``last`` s or less is not cut."""
    name, sec, targets = phases[-1]
    if sec <= last:
        return phases, phases[-2][0]
    return [*phases[:-1], (name, sec - last, targets), (f"{name}_last", last, targets)], name


WATCH = ["bag", "board", "gripper/finger_l", "gripper/finger_r"]    # recorded every period in the hands-off window


def hands_off_window(s: CandleSpec, frames: dict) -> list[dict]:
    """Per world, GA's hands_off_window from the window's frames (``frames``: a trajectory's arrays, [frames, worlds,
    ...], at every control period, first frame the window's start): no finger box within 5 mm of the bag's or board's
    vertices, the carton, the mat or the counter. See gift_hands_off.py for what differs from GA's force test."""
    fingers = tool_world_boxes(frames, "gripper", tool_boxes("fingers"))
    carton = WorldBox.of_body(frames["candle.pos"], frames["candle.quat_wxyz"], (0.0, 0.0, 0.0),
                              tuple(0.5 * e for e in s.candle))
    mat = WorldBox(np.array([0.0, 0.0, 0.5 * MAT[2]]), np.eye(3), tuple(0.5 * e for e in MAT))
    hit = contact_track(fingers, [carton, mat], [frames["bag.points"], frames["board.points"]], 0.0, MARGIN)
    return window_verdict(hit, int(round(s.window / s.dt)), s.dt)


def _world_verdict(L, s: CandleSpec, st: np.ndarray, w: int, state: dict, marks: dict, hands: dict) -> dict:
    """One world's verdict: GA's criteria from state, plus the port's own start data."""
    v = evaluate(L, state["bag.points"][w], state["board.points"][w], state["candle.pos"][w],
                 state["candle.quat_wxyz"][w], bottom_mark=marks["bottom0"][w],
                 hands_off=hands["hands_off_window"])
    v.update({k: x for k, x in hands.items() if k != "hands_off_window"})
    v.update(start_dx_mm=round(st[w, 0] * 1e3, 1), start_dy_mm=round(st[w, 1] * 1e3, 1),
             start_yaw_deg=round(math.degrees(st[w, 2]), 1), held_over_rim=bool(marks["held"][w]))
    return v


def run(s: CandleSpec = CandleSpec(), device: str | None = None, worlds: int = 1, seed: int = 0,
        record_every: int = 0) -> dict:
    t0 = time.perf_counter()
    sim, st = build(s, worlds, seed, device)
    build_s = time.perf_counter() - t0
    L = _layout(s)
    marks: dict = {}

    phases, mark_after = split_window(schedule(plan(s), st))

    def on_phase(name: str, tr) -> None:
        if name == "carry":
            marks["held"] = (sim.state(["candle"])["candle.pos"][:, 2] - 0.5 * s.candle[2] > s.lift - 0.02).tolist()
        if name == mark_after:
            marks["bottom0"] = sim.state(["bag"])["bag.points"][:, L.bottom].mean(1)

    t1 = time.perf_counter()
    rows = drive(sim, phases, on_phase, record=["bag", "board"] if record_every else (), every=record_every,
                 watch=WATCH)
    wall = time.perf_counter() - t1
    state = sim.state(["bag", "board", "candle"])
    hands = hands_off_window(s, join_frames([r.pop("watch") for r in rows if "watch" in r]))
    per = [_world_verdict(L, s, st, w, state, marks, hands[w]) for w in range(worlds)]
    check = sim.check()
    out = {"spec": asdict(s), "device": str(getattr(sim.engine.model, "device", device)), "seed": seed,
           "worlds": per, "ga_judged": per[0]["ga_judged"], "success_rate": float(np.mean([v["success"] for v in per])),
           "success": all(v["success"] for v in per), "check_ok": bool(check["ok"]),
           "problems": [p["what"] for p in check["problems"][:5]], "build_s": round(build_s, 1),
           "wall_s": round(wall, 1), "sim_s": round(float(sim.t), 2)}
    if record_every:
        out["frames"] = [r.pop("frames") for r in rows]
    out["phases"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = run(CandleSpec(), device=a.device, worlds=a.worlds, seed=a.seed)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("success", "success_rate", "ga_judged", "check_ok", "wall_s", "sim_s")}))
    print(json.dumps(out["worlds"][0]))


if __name__ == "__main__":
    main()
