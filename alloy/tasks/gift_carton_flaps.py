# SPDX-License-Identifier: Apache-2.0
"""Open the flaps: open all four top flaps of an un-taped supply carton and leave them open, contents inside, carton
upright (the reference task set's gift-bags open-flaps task).

Scene (public API, table top at z = 0): the carton (gift_carton.py: RSC 356 x 254 x 203 mm inside, C-flute, flaps on
elastic-plastic crease hinges), the same in every world; its contents as the reference's: eight
tea cartons 150 x 80 x 70 mm, 120 g, standing in two rows, and a glass jar 85 x 110 mm, 450 g (ESTIMATES); and the
gantry with a hook (gift_gantry.py) standing in for the robot.

Scripted demo (gift_flap_path.py): the hook opens the sprung major, then the other major, then the two minors (a minor
cannot open first: it lies under the majors), each pushed to 200 deg and let go; then 3 s hands off. Checks over that
window (the reference's): each flap at least 90 deg open through the whole window; the carton moved at most 30 mm and
tilted at most 10 deg; every content item inside the carton and moved at most 10 mm; and the hands-off window itself
(``window_contacts``, gift_hands_off.py): at no control period of the 3 s is a hook box within 5 mm of the carton's
walls and bottom, a flap board, a content item or the counter. The window is recorded at every period (0.01 s), flap
angles included, as the reference reads it.

Example::

    from alloy.tasks.gift_carton_flaps import FlapsSpec, run
    out = run(FlapsSpec(), device="cpu", worlds=2)
    print(out["success_rate"], out["ga_judged"], out["worlds"][0])
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .gift_carton import FLAPS, CartonSpec, _hinges, _rot as quat_rot, carton_mjcf, flap_angles
from .gift_flap_path import Push, flap_frame, tip_offset, tool_yaw, waypoints
from .gift_gantry import _tool, ga_verdict, gantry_mjcf, tool_boxes
from .gift_hands_off import NOTE as HANDS_OFF_NOTE
from .gift_hands_off import MARGIN, WorldBox, contact_track, tool_world_boxes, window_verdict

WRIST_LIMIT = float(_tool("hook")[0].split()[1])      # the hook gantry's wrist_yaw joint range is +-this [rad]

TEA = ((0.08, 0.07, 0.15), 0.12)        # full edges (x, y, z) [m], kg
JAR = ((0.0425, 0.11), 0.45)            # radius, height [m], kg
ORDER = ("major_p", "major_m", "minor_p", "minor_m")
GA_FLAP = {"major_p": "flap_major_ypos", "major_m": "flap_major_yneg", "minor_p": "flap_minor_xpos",
           "minor_m": "flap_minor_xneg"}                          # the port's flap names -> GA's
NOT_JUDGED: dict[str, str] = {}         # every criterion of open-flaps is judged
NOTES = (HANDS_OFF_NOTE, "the jar is taken as the box around it for the hands-off test")
SWEEP_STEPS = 20


@dataclass(frozen=True)
class FlapsSpec:
    carton: CartonSpec = field(default_factory=CartonSpec)
    push: Push = field(default_factory=Push)
    jitter_xy: float = 0.02             # carton start per world: x and y within this [m]
    jitter_yaw_deg: float = 5.0         # and yaw within this [deg]; the contents move with it
    open_deg: float = 90.0              # pass: every flap at least this open through the window
    carton_move: float = 0.03           # m
    carton_tilt_deg: float = 10.0
    contents_move: float = 0.01         # m
    window: float = 3.0                 # hands-off window [s]
    dt: float = 0.01


def contents(s: CartonSpec) -> list[tuple[str, str, tuple, float, np.ndarray]]:
    """(name, kind, size, mass, centre in the carton frame): two rows of four tea cartons along -y, the jar at +y."""
    (tx, ty, tz), tm = TEA
    gap, out = 0.006, []                # apart and off the walls, so items do not all sit in each other's margins
    for r in range(2):
        y = -0.5 * s.width + gap + (r + 0.5) * ty + r * gap
        for k in range(4):
            x = (k - 1.5) * (tx + gap)
            out.append((f"tea_{r}{k}", "box", (tx, ty, tz), tm, np.array([x, y, s.caliper + 0.5 * tz + 2e-4])))
    (jr, jh), jm = JAR
    jar = np.array([0.0, 0.5 * s.width - jr - gap, s.caliper + 0.5 * jh + 2e-4])
    out.append(("jar", "cylinder", (jr, jh), jm, jar))
    return out


def starts(s: FlapsSpec, worlds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-s.jitter_xy, s.jitter_xy, (worlds, 2))
    return np.column_stack([xy, np.radians(rng.uniform(-s.jitter_yaw_deg, s.jitter_yaw_deg, worlds))])


def palm_start(s: FlapsSpec) -> tuple[float, float, float]:
    return (0.0, 0.0, s.carton.height + 0.45)


def _yaw_R(yaw: float) -> np.ndarray:
    c, n = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -n, 0.0], [n, c, 0.0], [0.0, 0.0, 1.0]])


def build(s: FlapsSpec, worlds: int = 1, seed: int = 0, device: str | None = None, with_contents: bool = True):
    """(sim, starts): the scene, each world's carton at its start (x, y, yaw) with its contents
    (``with_contents=False``: an empty carton). Positions go in at build, the yaws by ``set_pose`` before the first
    step (``sim.reset`` returns to the unturned build)."""
    import alloy

    st = starts(s, worlds, seed)
    scene = alloy.Scene(worlds=worlds, dt=s.dt)
    scene.add_ground()
    scene.add_robot("carton", mjcf=carton_mjcf(s.carton), pos=np.column_stack([st[:, :2], np.zeros(worlds)]))
    items = contents(s.carton) if with_contents else []
    centres = {}
    for name, kind, size, mass, c in items:
        centres[name] = np.stack([np.array([w[0], w[1], 0.0]) + _yaw_R(w[2]) @ c for w in st])
        if kind == "box":
            scene.add_box(name, size=size, pos=centres[name], mass=mass, friction=0.5)
        else:
            scene.add_cylinder(name, size[0], size[1], pos=centres[name], mass=mass, friction=0.5)
    scene.add_robot("hook", mjcf=gantry_mjcf(palm_start(s), tool="hook"))
    sim = alloy.build(scene, device=device, seed=seed)
    if np.any(st[:, 2]):
        q = np.column_stack([np.cos(0.5 * st[:, 2]), np.zeros((worlds, 2)), np.sin(0.5 * st[:, 2])])
        sim.set_pose("carton", pos=np.column_stack([st[:, :2], np.zeros(worlds)]), quat_wxyz=q)
        for name, pos in centres.items():
            sim.set_pose(name, pos=pos, quat_wxyz=q)
    return sim, st


FLAP_BODIES = ["carton/carton"] + [f"carton/{f}" for f in FLAPS]


def read(sim, s: FlapsSpec) -> tuple[list, np.ndarray]:
    """Per world: ((carton pos, R), flap angles in deg)."""
    x = sim.state(FLAP_BODIES)
    poses, ang = [], []
    for w in range(sim.worlds):
        qc = x["carton/carton.quat_wxyz"][w]
        poses.append((np.asarray(x["carton/carton.pos"][w], float), quat_rot(qc)))
        a = flap_angles(s.carton, qc, {f: x[f"carton/{f}.quat_wxyz"][w] for f in FLAPS})
        ang.append([a[f] for f in FLAPS])
    return poses, np.asarray(ang)


def _segments(s: FlapsSpec, flap: str, poses, ang) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    """Per waypoint: (phase, seconds, [worlds, 3] palm targets, [worlds] yaws); worlds share the timing."""
    rows = []
    for w, pose in enumerate(poses):
        phi0 = math.radians(ang[w, FLAPS.index(flap)])
        pts = waypoints(s.carton, flap, pose, phi0, s.push)
        yaw = tool_yaw(flap_frame(s.carton, flap, pose)[2])
        rows.append([(ph, tip - tip_offset(yaw), sec, yaw) for ph, tip, sec in pts])
    out = []
    for k in range(len(rows[0])):
        out.append((rows[0][k][0], rows[0][k][2], np.stack([r[k][1] for r in rows]), np.array([r[k][3] for r in rows])))
    return out


def unwrap_yaw(wanted, cur, limit: float = WRIST_LIMIT) -> np.ndarray:
    """The yaw ``wanted + 2 pi k`` inside +-``limit`` that is nearest the current target ``cur`` (the shortest turn
    when it stays in range, the other way round when it would leave it)."""
    wanted, cur = np.broadcast_arrays(np.asarray(wanted, float), np.asarray(cur, float))
    k = np.arange(-3, 4)[:, None]
    cand = wanted.reshape(1, -1) + 2 * np.pi * k
    dist = np.where(np.abs(cand) <= limit, np.abs(cand - cur.reshape(1, -1)), np.inf)
    best = cand[np.argmin(dist, axis=0), np.arange(cand.shape[1])]
    if not np.all(np.isfinite(dist.min(axis=0))):
        raise ValueError(f"no yaw equivalent to {wanted} lies inside +-{limit} rad")
    return best.reshape(wanted.shape)


def drive(sim, s: FlapsSpec, on_phase=None, p0=None, cur: dict | None = None) -> list[dict]:
    """Opens the flaps in ORDER; the joint targets are offsets of the palm from its start ``p0`` (the gantry's MJCF
    start, default ``palm_start``), starting from the targets ``cur`` (default: all 0, the hook at its start)."""
    p0 = np.asarray(palm_start(s) if p0 is None else p0, float)
    zero = np.zeros(sim.worlds)
    cur = dict(cur) if cur is not None else {k: zero for k in ("gantry_x", "gantry_y", "gantry_z", "wrist_yaw")}
    log = []
    for flap in ORDER:
        poses, ang = read(sim, s)
        for phase, sec, palm, yaw in _segments(s, flap, poses, ang):
            yaw = unwrap_yaw(yaw, cur["wrist_yaw"])
            new = {"gantry_x": palm[:, 0] - p0[0], "gantry_y": palm[:, 1] - p0[1], "gantry_z": palm[:, 2] - p0[2],
                   "wrist_yaw": yaw}
            sim.run(sec, targets={k: (cur[k], new[k]) for k in cur})
            cur = new
        log.append({"flap": flap, "t": round(float(sim.t), 2), "angles": np.round(read(sim, s)[1], 1).tolist()})
        if on_phase is not None:
            on_phase(flap)
    home = np.asarray(palm_start(s), float) - p0                # back over the carton, 0.1 m above the hook's start
    away = dict(cur, gantry_x=np.full(sim.worlds, home[0]), gantry_y=np.full(sim.worlds, home[1]),
                gantry_z=np.full(sim.worlds, home[2] + 0.1))
    sim.run(0.8, targets={k: (cur[k], away[k]) for k in cur})
    return log


ITEMS = [c[0] for c in contents(CartonSpec())]


def _angles_track(s: FlapsSpec, tr) -> np.ndarray:
    """[frames, worlds, 4] flap angles from a recorded trajectory."""
    F, W = tr["carton/carton.quat_wxyz"].shape[:2]
    out = np.zeros((F, W, len(FLAPS)))
    for f in range(F):
        for w in range(W):
            a = flap_angles(s.carton, tr["carton/carton.quat_wxyz"][f, w],
                            {k: tr[f"carton/{k}.quat_wxyz"][f, w] for k in FLAPS})
            out[f, w] = [a[k] for k in FLAPS]
    return out


def _carton_boxes(s: CartonSpec, tr) -> list[WorldBox]:
    """The carton's walls and bottom, its flap boards and the contents as boxes, per recorded frame."""
    t, L, W, H = s.caliper, s.length, s.width, s.height
    walls = [((0, 0, 0.5 * t), (0.5 * L + t, 0.5 * W + t, 0.5 * t)),
             ((0, 0.5 * W + 0.5 * t, 0.5 * H + 0.5 * t), (0.5 * L + t, 0.5 * t, 0.5 * H - 0.5 * t)),
             ((0, -0.5 * W - 0.5 * t, 0.5 * H + 0.5 * t), (0.5 * L + t, 0.5 * t, 0.5 * H - 0.5 * t)),
             ((0.5 * L + 0.5 * t, 0, 0.5 * H + 0.5 * t), (0.5 * t, 0.5 * W, 0.5 * H - 0.5 * t)),
             ((-0.5 * L - 0.5 * t, 0, 0.5 * H + 0.5 * t), (0.5 * t, 0.5 * W, 0.5 * H - 0.5 * t))]
    pose = lambda n: (tr[f"{n}.pos"], tr[f"{n}.quat_wxyz"])
    out = [WorldBox.of_body(*pose("carton/carton"), c, h) for c, h in walls]
    for f, h in _hinges(s).items():
        d = np.asarray(h["in"], float)
        half = np.abs(d) * 0.5 * s.flap + (1 - np.abs(d)) * 0.5 * h["w"]
        half[2] = 0.5 * s.caliper
        out.append(WorldBox.of_body(*pose(f"carton/{f}"), d * 0.5 * s.flap, half))
    for name, kind, size, _, _ in contents(s):
        half = tuple(0.5 * x for x in size) if kind == "box" else (size[0], size[0], 0.5 * size[1])
        out.append(WorldBox.of_body(*pose(name), (0.0, 0.0, 0.0), half))
    return out


def window_contacts(s: FlapsSpec, tr, tine_width: float = 0.024) -> list[dict]:
    """Per world, GA's hands_off_window from the window's frames ``tr`` (every control period, first frame the
    window's start): no hook box within 5 mm of the carton, its flaps, the contents or the counter."""
    hook = tool_world_boxes(tr, "hook", tool_boxes("hook", tine_width))
    hit = contact_track(hook, _carton_boxes(s.carton, tr), [], 0.0, MARGIN)
    return window_verdict(hit, int(round(s.window / s.dt)), s.dt)


def evaluate(s: FlapsSpec, start: dict, tr, tine_width: float = 0.024) -> list[dict]:
    """Per world, the reference's checks over the hands-off window ``tr`` (recorded at every control period);
    ``tine_width``: the hook's tine."""
    ang = _angles_track(s, tr)
    hands = window_contacts(s, tr, tine_width)
    out = []
    for w in range(ang.shape[1]):
        low = ang[:, w].min(0)
        c0, c1 = start["carton/carton.pos"][w], tr["carton/carton.pos"][-1, w]
        R1 = quat_rot(tr["carton/carton.quat_wxyz"][-1, w])
        tilt = math.degrees(math.acos(float(np.clip(R1[2, 2], -1.0, 1.0))))
        moved, inside = [], []
        for it in ITEMS:
            p0, p1 = start[f"{it}.pos"][w], tr[f"{it}.pos"][-1, w]
            moved.append(float(np.linalg.norm(p1 - p0)))
            q = R1.T @ (p1 - c1)
            inside.append(bool(abs(q[0]) < 0.5 * s.carton.length and abs(q[1]) < 0.5 * s.carton.width
                               and 0.0 < q[2] < s.carton.height))
        v = {f"{f}_open": bool(low[k] >= s.open_deg) for k, f in enumerate(FLAPS)}
        v.update({f"{f}_min_deg": round(float(low[k]), 1) for k, f in enumerate(FLAPS)})
        v.update(carton_in_place=bool(np.linalg.norm(c1 - c0) <= s.carton_move),
                 carton_upright=bool(tilt <= s.carton_tilt_deg),
                 contents_in_place=bool(all(inside) and max(moved) <= s.contents_move),
                 carton_move_mm=round(float(np.linalg.norm(c1 - c0)) * 1e3, 2), carton_tilt_deg=round(tilt, 2),
                 contents_max_move_mm=round(max(moved) * 1e3, 2),
                 **hands[w])
        v["success"] = all(v[k] for k in v if isinstance(v[k], bool))
        out.append(v)
    return out


def ga_report(v: dict, first: dict[str, bool] | None = None, notes: tuple[str, ...] = ()) -> dict:
    """``v`` (one world's ``evaluate`` row) with the verdict against GA's open-flaps criteria: ``ga_*``
    and ``success`` from them (no port check: the hook's window is a GA criterion). ``first``: judged criteria of a
    task that adds to these (open-carton's ``seam_open``); ``notes``: further ``ga_notes``."""
    criteria = dict(first or {})
    criteria.update({GA_FLAP[f] + "_open": v[f"{f}_open"] for f in FLAPS})
    criteria.update({k: v[k] for k in ("carton_in_place", "carton_upright", "contents_in_place")})
    criteria["hands_off_window"] = v["hands_off_window"]
    return {**v, **ga_verdict(criteria, NOT_JUDGED, {}, NOTES + notes)}


def run(s: FlapsSpec = FlapsSpec(), device: str | None = None, worlds: int = 1, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    sim, _ = build(s, worlds, seed, device)
    build_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    sim.run(0.3)                                             # settle: the flaps sag onto their supports
    names = FLAP_BODIES + ITEMS
    start = sim.state(names)
    log = drive(sim, s)
    tr = sim.run(s.window, record=names + ["hook/wrist"], every=1)
    wall = time.perf_counter() - t1
    per = [ga_report(v) for v in evaluate(s, start, tr)]
    check = sim.check()
    return {"spec": asdict(s), "device": str(getattr(sim.engine.model, "device", device)), "worlds": per,
            "ga_judged": per[0]["ga_judged"], "success_rate": float(np.mean([v["success"] for v in per])),
            "success": all(v["success"] for v in per), "check_ok": bool(check["ok"]),
            "problems": [p["what"] for p in check["problems"][:5]], "build_s": round(build_s, 1),
            "wall_s": round(wall, 1), "sim_s": round(float(sim.t), 2), "flaps": log}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--push-deg", type=float, default=Push().target_deg)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = run(FlapsSpec(push=Push(target_deg=a.push_deg)), device=a.device, worlds=a.worlds)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("success", "success_rate", "ga_judged", "check_ok", "wall_s", "sim_s")}))
    print(json.dumps(out["worlds"][0]))


if __name__ == "__main__":
    main()
