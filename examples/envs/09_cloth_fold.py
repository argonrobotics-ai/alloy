# SPDX-License-Identifier: Apache-2.0
"""ClothFold-v0 (experimental): fold a towel along its diagonal with the Cartesian gripper (scripted policy).

The pinch holds the 2 mm towel through the fold in every world; about 40 % of the corners end within 3 cm of their
target, so expect partial folds (JSON).

Lower the fingers onto the towel 1.2 cm in from its +x +y corner and close them. Lift as soon as they stop closing
(on the cloth they gather, or shut), turn the held point half a circle about the fold line (the other diagonal) onto
its mirror image, which lays the corner on the -x -y corner, and let go. Pulling it straight across drags the whole
towel instead. The held towel drags the lower layer a few cm during the turn, and the held corner sits a little off
the grasp point, so over the turn's second half the target moves until the held corner itself lies on the -x -y
corner as that corner is now: aimed at where the corner was, the fold missed it in 47 of 64 worlds; aimed with the
grasp point on its drift, the corners stopped 2.5 cm short along the diagonal in every world (median; now 0.3 cm).
ClothFold's gripper squeezes the pinched ridge with ~32 N
(``gantry_gripper(squeeze=GRIP_SQUEEZE)``): at its default 2.6 N on the 3.5 mm ridge the cloth slid out of the
fingertips during the turn in 43 of 64 worlds.
While the fingers close, the z servo presses the fingertips on the towel with a stated force, ``PRESS_N`` (35 N
beyond the gantry's 4.5 N weight): its target follows the palm press / kp below it, so the force stays at that
value wherever the palm is. Two steps before the lift the press is taken off, and at all other times the z target
never goes below the fingertips' contact height (no wind-up). A touch alone slides over flat cloth without
gathering it (0 of 64 folded); with the press the fingers gather a ridge and stop on it.
64 worlds on an L4 (GPU runs differ run to run): folded 25, 27 and 27 in
three runs (and 12, 14 of 32 in two 32-world groups; the towel never slipped out); before the corner aim, 15-25 of 64
in eight runs; before the squeeze and the drift target, 11-16 in three. The misses: the stiff towel is dragged along
by its lifted half (placed off, 15-18 of 64) or springs off the target as the fingers open (14-19). A world the
engine marks INVALID (a towel vertex in the pinch thrown past 20 m/s, 3-5 per run) is reset by the env and counts as
not folded.
Slow, use a GPU: on an L4 a step takes about 8 s for 4 worlds and 13 s for 64 (more during the fold), since the
30 g fingers on the cloth set 512 force exchanges per 20 ms step (``alloy.estimate``); a fresh kernel cache adds about
10 min at the first step. A whole fold (142 steps) takes about 35 min for 4 worlds, an hour for 64.
Run: python examples/envs/09_cloth_fold.py [num_envs] [press_N] [grasp]   (the whole fold; the numbers above: 64 35)
grasp "attach" holds the pinched towel to the fingers until they open (``make("ClothFold-v0", grasp="attach")``): a
shortcut as cloth benchmarks grasp, not a friction grasp (``alloy.envs.cloth_attach``); the default is "friction".
With no arguments it runs one world through the pinch and the lift only (QUICK_FOLD_STEPS of the turn) and prints
whether the corner rose 5 cm (it did by turn step 8 in all 64 worlds of the runs above): about 64 steps, roughly
10 min on an L4 with a warm kernel cache (one world steps about as fast as four: 7.6 s a step before the pinch).
"""
import json
import sys

import numpy as np

from alloy.envs import make
from alloy.envs.library_tasks import grasp_point
from alloy.envs.robots import FINGER_CLOSED
from alloy.envs.soft_tasks import corner

QUICK = len(sys.argv) == 1                # no arguments: one world, the pinch and the lift only (docs run)
NUM_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PRESS_N = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0  # N, while closing only: a touch gathers nothing
GRASP = sys.argv[3] if len(sys.argv) > 3 else "friction"    # "attach": the opt-in shortcut (not friction)
KP_Z = 3000.0                       # N/m, the gantry's z servo
FINGERTIP_DROP = 0.07               # m, palm centre to fingertips
OVER, DOWN, PINCH, EASE, FOLD, RELEASE, DONE = range(7)
STEPS = {OVER: 25, DOWN: 10, PINCH: 25, EASE: 2, FOLD: 70, RELEASE: 10}
# PINCH: at most (it ends once the fingers stop); EASE: the press off, the fingers still closing
QUICK_FOLD_STEPS = 12              # turn steps the quick run takes: the corner is past 5 cm up by then
STALL_M = 0.001                     # "stopped closing": under 1 mm in one step, after moving over 2 mm in one

env = make("ClothFold-v0", num_envs=NUM_ENVS, grasp=GRASP)
xp = env.xp
env.reset(seed=0)
s = env.state
a0, b0 = xp.to_numpy(corner(s, 1.0)), xp.to_numpy(corner(s, -1.0))
centre = 0.5 * (a0 + b0)
toward_a = (a0 - centre) / np.maximum(np.linalg.norm(a0 - centre, axis=1, keepdims=True), 1e-9)
top = np.median(xp.to_numpy(s["towel.points"])[..., 2], axis=1) + 0.001       # the towel's top (2 mm thick)
base_z = xp.to_numpy(s["gripper/palm.pos"])[:, 2] - xp.to_numpy(s["gripper/z.q"])
contact_t = top + FINGERTIP_DROP - base_z                    # z target with the fingertips on the towel


def fold_point(frac: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Where the held point goes: a half circle about the fold line (the towel's other diagonal) onto its mirror."""
    rel = (held - centre) * [1.0, 1.0, 0.0]
    r = (rel * toward_a).sum(1, keepdims=True)                     # its distance from the fold line
    th = np.pi * np.minimum(frac, 1.0)[:, None]
    return centre + rel - r * toward_a + r * (np.cos(th) * toward_a + np.sin(th) * np.array([0.0, 0.0, 1.0]))


def action(s, phase: np.ndarray, since: np.ndarray, tz: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Every world's action for its phase. The z target (``tz``) stays at or above ``contact_t``, except in a pinch,
    where it sits ``PRESS_N / KP_Z`` under the palm (a force set-point)."""
    g, ca, cb = xp.to_numpy(grasp_point(s)), xp.to_numpy(corner(s, 1.0)), xp.to_numpy(corner(s, -1.0))
    c = ca - [0.012, 0.012, 0.0]                                    # pinch 1.2 cm in from the corner
    palm = xp.to_numpy(s["gripper/palm.pos"])
    a = np.zeros((NUM_ENVS, 4), np.float32)
    a[:, 3] = np.where((phase == PINCH) | (phase == EASE) | (phase == FOLD), 1.0, -1.0)
    near = (phase == OVER) | (phase == DOWN)
    a[near, :2] = np.clip((c[near, :2] - g[near, :2]) / 0.01, -1, 1)
    a[near, 2] = np.clip((np.where(phase == OVER, 0.08, -1.0) - palm[:, 2])[near] / 0.01, -1, 1)
    frac = (since + 1) / (STEPS[FOLD] - 10)
    want = fold_point(frac, held) + [0.0, 0.0, 0.03]                       # 3 cm above the pinch
    # 2nd half: move the target from the mirror point to where the held corner lies on the -x -y corner as it is now
    aim = (cb - ca + g - fold_point(np.ones(NUM_ENVS), held)) * [1.0, 1.0, 0.0]
    want += np.clip((frac - 0.5) / 0.5, 0.0, 1.0)[:, None] * aim
    fold = phase == FOLD
    a[fold, :3] = np.clip((want[fold] - g[fold]) / 0.01, -1, 1)
    a[:, 2] = np.clip(np.maximum(a[:, 2], (contact_t - tz) / 0.01), -1, 1)      # anti-windup at the contact
    press = phase == PINCH                                          # kp * (palm - target) = PRESS_N
    a[press, 2] = np.clip((palm[:, 2] - base_z - PRESS_N / KP_Z - tz)[press] / 0.01, -1, 1)
    return a


def advance(phase: np.ndarray, since: np.ndarray, q: np.ndarray, dq: np.ndarray, fastest: np.ndarray) -> np.ndarray:
    """Which worlds end their phase after this step (a pinch ends once the fingers have closed and stopped)."""
    end = np.array([since[w] + 1 >= STEPS.get(int(p), 1 << 30) for w, p in enumerate(phase)])
    stopped = ((fastest > 2 * STALL_M) & (dq < STALL_M)) | (q > FINGER_CLOSED - 0.0005)
    return np.where(phase == PINCH, end | stopped, end)


phase, since = np.full(NUM_ENVS, OVER), np.zeros(NUM_ENVS, int)
tz, q = xp.to_numpy(s["gripper/z.q"]), xp.to_numpy(s["gripper/close.q"])
fastest = np.zeros(NUM_ENVS)                          # the fingers' fastest close step in this pinch
pinch_steps, gap_at_lift, peak = np.zeros(NUM_ENVS, int), np.full(NUM_ENVS, np.nan), np.zeros(NUM_ENVS)
held = a0.copy()                                      # the towel point between the fingers, taken at the lift
for _ in range(sum(STEPS.values())):                  # 142 steps, under the task's 150
    obs, reward, terminated, truncated, info = env.step(xp.asarray(action(s, phase, since, tz, held)))
    s = env.state
    tz, q_new = xp.to_numpy(info["targets"])[:, 2], xp.to_numpy(s["gripper/close.q"])
    fastest = np.where(phase == PINCH, np.maximum(fastest, q_new - q), 0.0)
    end = advance(phase, since, q_new, q_new - q, fastest)
    lift = end & (phase == PINCH)
    pinch_steps[lift], gap_at_lift[lift] = since[lift] + 1, 2 * (FINGER_CLOSED - q_new[lift])
    held[lift, :2] = xp.to_numpy(grasp_point(s))[lift, :2]
    peak = np.maximum(peak, np.where(phase == FOLD, xp.to_numpy(corner(s, 1.0))[:, 2], 0.0))
    phase, since, q = np.where(end, phase + 1, phase), np.where(end, 0, since + 1), q_new
    if QUICK and ((phase > FOLD) | ((phase == FOLD) & (since >= QUICK_FOLD_STEPS))).all():
        break
gap, lifted = xp.to_numpy(info["terms"]["corner_gap"]), np.isfinite(gap_at_lift)
print(json.dumps({"worlds": NUM_ENVS, "press_N": PRESS_N, "grasp": GRASP, "quick": QUICK,
                  "folded": None if QUICK else float(xp.to_numpy(info["success"]).mean()),
                  "mean_corner_gap_m": None if QUICK else round(float(gap.mean()), 3),   # quick: mid-turn, no result
                  "lifted_5cm": float((peak > 0.05).mean()),
                  "mean_pinch_steps": round(float(pinch_steps.mean()), 1),
                  "mean_finger_gap_at_lift_mm": round(float(gap_at_lift[lifted].mean()) * 1000, 1) if lifted.any()
                  else None}))
env.close()
