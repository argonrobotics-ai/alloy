# SPDX-License-Identifier: Apache-2.0
"""hold-task quality watch and scorer, shared by every engine (Isaac Lab side and Alloy side). numpy only.

A quality run is the speed loop (hold targets with noise, one 5 ms env step) for ``QUALITY_STEPS`` steps at
>= 1,024 worlds, with these read every step from the engine's own state:

- ``speed``  (W,) the fastest body's linear speed this step [m/s]
- ``zmin``   (W,) the lowest body origin this step, env frame [m]
- ``finite`` (W,) every body position and velocity is finite

and, every ``every`` steps, the pose of the task's tracked bodies (``TASKS[task]["track"]``), env frame.
``Watch.save`` writes one npz; ``score`` turns it into the S1 glitch count and the task's contact-held count.

S1 (per world, any step): ``nonfinite``; ``speed`` over 20 m/s (Alloy's SPEED_FAIL); ``tunnel``: a body origin
more than 1 cm below the task's floor (stand tasks: only in worlds where the robot still stands at the end; a fallen
robot is counted by ``held``, and its collider-free links reach under the floor); ``pen``: the cube more than 1 cm
into its support (cube tasks at rest). Contact depth itself is not read: PhysX does not expose it through Isaac Lab.
Held: a resting object moves under 5 mm after 0.5 s; a standing robot keeps 70 % of its base height at 0.5 s; an
in-hand cube ends within 10 cm of its start in the palm frame origin.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DT = 0.005              # one env step [s]
QUALITY_STEPS = int(__import__("os").environ.get("PXS_QUALITY_STEPS", 800))     # 4 s
SETTLE_S = 0.5          # held and rest measures compare the end of the run with this time
SPEED_FAIL = 20.0       # m/s
PEN_FAIL = 0.01         # m
STAND_KEEP = 0.7        # a standing robot keeps this share of its settled base height
INHAND_KEEP = 0.10      # m: an in-hand cube stays this close to its start position in the palm frame
NOISE = {"revolute": 0.02, "prismatic": 0.002}   # hold-target noise, uniform +-, rad or m, redrawn every env step

# kind: grasp = cube placed between the closed fingers; rest = object on its support; stand = robot on the ground;
# free = no contact. track = (label, scene entity, body name or None for the root). floor_z: lowest valid body
# origin height, env frame. cube = (edge m, support top z or None).
TASKS = {
    "Isaac-Lift-Franka": {"kind": "grasp", "track": (("hand", "robot", "panda_hand"), ("object", "object", None)),
                          "floor_z": 0.0, "cube": (0.05, 0.255)},
    "Isaac-Reach-Franka": {"kind": "free", "track": (("ee", "robot", "panda_hand"),), "floor_z": -1.05},
    "Isaac-Reach-UR10": {"kind": "free", "track": (("ee", "robot", "ee_link"),), "floor_z": -1.05},
    "Isaac-Velocity-Flat-AnymalD": {"kind": "stand", "track": (("base", "robot", None),), "floor_z": 0.0},
    "Isaac-Velocity-Flat-UnitreeGo2": {"kind": "stand", "track": (("base", "robot", None),), "floor_z": 0.0},
    "Isaac-Velocity-Flat-G1": {"kind": "stand", "track": (("base", "robot", None),), "floor_z": 0.0},
    "Isaac-Reorient-Cube-Allegro": {"kind": "inhand", "track": (("palm", "robot", None), ("object", "object", None)),
                                    "floor_z": -10.0},
}
# The rest of Skild's 20 single tasks. Floors here are loose (-1.1 m under the env origin for table tasks, -1 m for
# rough terrain, whose height varies), so "tunnel" only catches a body leaving the world; nothing checks them against
# the scene. Kuka-Allegro tasks: the object rests on the table at the start ("rest": it stays where it settled).
_STAND = {"kind": "stand", "track": (("base", "robot", None),)}
_REST = {"kind": "rest", "track": (("object", "object", None),), "floor_z": -1.1}
TASKS.update({
    "Isaac-Reach-Franka-OSC": {"kind": "free", "track": (("ee", "robot", "panda_hand"),), "floor_z": -1.1},
    "Isaac-Lift-KukaAllegro": _REST, "Isaac-Reorient-KukaAllegro": _REST,
    "Isaac-Lift-KukaAllegro-Camera": _REST, "Isaac-Reorient-KukaAllegro-Camera": _REST,
    "Isaac-Ant": {**_STAND, "floor_z": 0.0}, "Isaac-Humanoid": {**_STAND, "floor_z": 0.0},
    "Isaac-Humanoid-Direct": {**_STAND, "floor_z": 0.0},
    "Isaac-Cartpole": {"kind": "free", "track": (("base", "robot", None),), "floor_z": -10.0},
    "Isaac-Velocity-Flat-H1": {**_STAND, "floor_z": 0.0},
    "Isaac-Velocity-Rough-G1": {**_STAND, "floor_z": -1.0}, "Isaac-Velocity-Rough-AnymalD": {**_STAND, "floor_z": -1.0},
    "Isaac-Velocity-Rough-Cassie": {**_STAND, "floor_z": -1.0},
})


class Watch:
    """Running per-world reductions and sparse tracked poses for one quality run."""

    def __init__(self, worlds: int, tracked: int, every: int = 10):
        self.every = every
        self.max_speed = np.zeros(worlds, np.float32)
        self.min_z = np.full(worlds, np.inf, np.float32)
        self.nonfinite = np.zeros(worlds, bool)
        self.first_bad = np.full(worlds, -1, np.int32)
        self.tracked = tracked
        self.t, self.pos, self.quat = [], [], []

    def step(self, k: int, speed, zmin, finite, pos=None, quat_xyzw=None) -> None:
        """Feed step ``k`` (0 = the start state). ``pos`` (W, K, 3) and ``quat_xyzw`` (W, K, 4) when k % every == 0."""
        speed, zmin, finite = np.asarray(speed), np.asarray(zmin), np.asarray(finite, bool)
        bad = ~finite | ~np.isfinite(speed) | ~np.isfinite(zmin)
        self.nonfinite |= bad
        self.max_speed = np.maximum(self.max_speed, np.where(bad, 0.0, speed))
        self.min_z = np.minimum(self.min_z, np.where(bad, np.inf, zmin))
        newly = (bad | (speed > SPEED_FAIL)) & (self.first_bad < 0)
        self.first_bad[newly] = k
        if k % self.every == 0 and pos is not None:
            self.t.append(k * DT)
            self.pos.append(np.asarray(pos, np.float32))
            self.quat.append(np.asarray(quat_xyzw, np.float32))

    def save(self, path: str | Path, meta: dict) -> None:
        np.savez_compressed(path, max_speed=self.max_speed, min_z=self.min_z, nonfinite=self.nonfinite,
                            first_bad=self.first_bad, t=np.asarray(self.t, np.float32), pos=np.asarray(self.pos),
                            quat_xyzw=np.asarray(self.quat), meta=json.dumps(meta, default=str))


def up95(k: int, n: int) -> float:
    """One-sided 95 % Clopper-Pearson upper bound on a rate with k hits in n."""
    if n == 0:
        return float("nan")
    if k >= n:
        return 1.0

    def cdf(p: float) -> float:
        logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
        m = max(logs)
        return math.exp(m) * sum(math.exp(x - m) for x in logs)

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cdf(mid) > 0.05 else (lo, mid)
    return hi


def _rot_cols(q):
    """Rotation matrices (…, 3, 3) from xyzw quaternions."""
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _held(kind: str, t, pos, quat, labels) -> tuple[np.ndarray, np.ndarray]:
    """(held (W,) bool, drift (W,) m) between SETTLE_S and the end of the run, by task kind."""
    i0, i1 = int(np.argmin(np.abs(t - SETTLE_S))), len(t) - 1
    ix = {name: j for j, name in enumerate(labels)}
    with np.errstate(invalid="ignore"):
        if kind == "grasp":        # the cube in the hand frame
            rot = _rot_cols(quat[:, :, ix["hand"]])
            rel = np.einsum("fwji,fwj->fwi", rot, pos[:, :, ix["object"]] - pos[:, :, ix["hand"]])
            drift = np.linalg.norm(rel[i1] - rel[i0], axis=-1)
            return (drift < 0.01) & (np.linalg.norm(rel[i1], axis=-1) < 0.2), drift
        if kind == "inhand":       # the cube stays on the hand: within 10 cm of where it started, palm frame origin
            rel = pos[:, :, ix["object"]] - pos[:, :, ix["palm"]]
            drift = np.linalg.norm(rel[i1] - rel[0], axis=-1)
            return drift < INHAND_KEEP, drift
        if kind == "rest":         # the object stays where it settled
            p = pos[:, :, ix["object"]]
            drift = np.linalg.norm(p[i1] - p[i0], axis=-1)
            return drift < 0.005, drift
        if kind == "stand":        # the base keeps its height; drift = horizontal travel
            p = pos[:, :, ix["base"]]
            drift = np.linalg.norm(p[i1, :, :2] - p[i0, :, :2], axis=-1)
            return p[i1, :, 2] > STAND_KEEP * p[i0, :, 2], drift        # against the settled height (it spawns above)
    worlds = pos.shape[1]
    return np.ones(worlds, bool), np.zeros(worlds)


def _cube_pen(task: dict, pos, quat, labels) -> np.ndarray:
    """(W,) deepest cube penetration into its support over the saved frames [m]; zeros when not a rest task."""
    worlds = pos.shape[1]
    if task.get("cube") is None or task["kind"] != "rest":
        return np.zeros(worlds)
    edge, top = task["cube"]
    j = list(labels).index("object")
    rot = _rot_cols(quat[:, :, j])
    half = 0.5 * edge * np.abs(rot[..., 2, :]).sum(-1)
    return np.nan_to_num(top - (pos[:, :, j, 2] - half), nan=0.0).max(0)


def score(npz_path: str | Path, task_id: str, kind: str | None = None, task: dict | None = None) -> dict:
    """S1 and contact-held summary of one quality npz. ``kind`` overrides the task's kind (lift at rest: 'rest');
    ``task`` replaces the ``TASKS`` entry (the Alloy rows' scenes: other body names and floor heights)."""
    d = np.load(npz_path, allow_pickle=False)
    task = dict(task or TASKS[task_id]); task["kind"] = kind or task["kind"]
    labels = [x[0] for x in task["track"]]
    worlds = len(d["max_speed"])
    causes = {"nonfinite": d["nonfinite"], "speed": d["max_speed"] > SPEED_FAIL,
              "tunnel": d["min_z"] < task["floor_z"] - PEN_FAIL,
              "pen": _cube_pen(task, d["pos"], d["quat_xyzw"], labels) > PEN_FAIL}
    held, drift = _held(task["kind"], d["t"], d["pos"], d["quat_xyzw"], labels)
    held &= ~d["nonfinite"]
    if task["kind"] == "stand":     # a robot that fell lies on links without colliders (arms, hands): their origins
        causes["tunnel"] &= held    # pass under the floor without any tunnelling. Fallen worlds are counted by held.
    s1 = np.zeros(worlds, bool)
    for flag in causes.values():
        s1 |= flag
    drift = drift[np.isfinite(drift)]
    k = int(s1.sum())
    return {"task": task_id, "kind": task["kind"], "episodes": worlds, "s1": k, "s1_rate": k / worlds,
            "s1_up95": up95(k, worlds), "causes": {c: int(f.sum()) for c, f in causes.items() if f.any()},
            "held": int(held.sum()), "held_rate": float(held.mean()),
            "drift_mm": {q: float(np.percentile(drift, p) * 1e3) if len(drift) else None
                         for q, p in (("p50", 50), ("p95", 95), ("max", 100))},
            "max_speed_mps": float(d["max_speed"].max()), "min_z_m": float(d["min_z"].min()),
            "seconds": float(d["t"][-1]) if len(d["t"]) else 0.0, "meta": json.loads(str(d["meta"]))}


if __name__ == "__main__":
    import sys
    print(json.dumps(score(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None), indent=1))
