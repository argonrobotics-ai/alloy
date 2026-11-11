# SPDX-License-Identifier: Apache-2.0
"""Ready-made tasks with liquids and cloth.

- ``Pour-v0``: 100 ml of water pours from a spout 25 cm up (``Scene.add_pour``, 100 ml/s); the agent moves a bowl on an
  x / y carriage under the stream and keeps it there. Success: at least 75 % of the water in the bowl.
- ``ClothFold-v0`` (experimental): fold a 24 cm towel along its diagonal with the Cartesian gripper (cloth
  self-contact on, so the folded layers rest on each other). Slow: about 8 s a step on an L4 for 4 worlds, 13 s for
  64, more during the fold (the 30 g fingers on the cloth set 512 force exchanges a step). The scripted demo
  (examples/envs/09_cloth_fold.py) presses the fingertips on the towel with a stated 35 N while the fingers close (a
  touch alone gathers nothing) and takes the press off before the lift; the gripper squeezes the pinch with ~32 N
  (``GRIP_SQUEEZE``) and over the turn's second half it aims the held corner at the far corner where that corner is
  now. On 64 worlds of an L4 the pinch holds in all 64 and the demo folds 25-27 in three runs; worlds the env resets
  after an INVALID (a towel vertex in the pinch thrown past 20 m/s) count as not folded. ``grasp="attach"`` is an
  opt-in shortcut, not a friction grasp (``alloy.envs.cloth_attach``; docs/guide/envs.md).

A liquid scene resets one world at a time like any other (SPH's snapshot keeps each world's particles apart), so a
world that pours well ends its episode on its own. Tilting a liquid-filled cup held by a robot does not work yet (the
liquid locks the hinge), which is why the agent moves the bowl rather than the cup. Example::

    from alloy.envs import make
    env = make("Pour-v0", num_envs=16)            # GPU strongly advised: about 1.4 steps/s for 8 worlds on an L4
"""

from __future__ import annotations

from typing import Any

from alloy.envs import checks as C
from alloy.envs.actions import ActionSpec
from alloy.envs.library_tasks import front_camera, maybe_vary
from alloy.envs.randomize import jitter_joint
from alloy.envs.state import BatchState
from alloy.envs.task import Task

TRAY = "tray"
SPOUT = (0.0, 0.0, 0.25)
BOWL_INNER = (0.088, 0.088, 0.05)          # m, inside of the bowl (x, y, depth)


def bowl_carriage(reach: float = 0.2, wall: float = 0.003) -> str:
    """MJCF of an open square bowl (``BOWL_INNER``) on x / y slides with position servos ("<robot>/x", "/y");
    the bowl body is "<robot>/bowl", its floor's underside at the robot's origin. The x carriage carries its mass as
    an inertial, with no geom: a camera-only geom would hide the bowl's collision boxes from every camera
    (the importer shows a model's collision shapes only when it has no visual ones). Example: ``bowl_carriage()``."""
    hx, hy, d = BOWL_INNER[0] / 2 + wall, BOWL_INNER[1] / 2 + wall, BOWL_INNER[2] / 2
    walls = "".join(
        f'<geom type="box" size="{sx:.4f} {sy:.4f} {d:.4f}" pos="{px:.4f} {py:.4f} {d + wall:.4f}" mass="0.03"/>'
        for sx, sy, px, py in ((hx, wall, 0, -hy + wall), (hx, wall, 0, hy - wall),
                               (wall, hy - 2 * wall, -hx + wall, 0), (wall, hy - 2 * wall, hx - wall, 0)))
    return f"""<mujoco><worldbody>
 <body name="carriage_x"><joint name="x" type="slide" axis="1 0 0" range="{-reach} {reach}" damping="50"/>
  <inertial pos="0 0 0" mass="0.2" diaginertia="2e-06 2e-06 2e-06"/>
  <body name="bowl"><joint name="y" type="slide" axis="0 1 0" range="{-reach} {reach}" damping="50"/>
   <geom type="box" size="{hx:.4f} {hy:.4f} {wall:.4f}" pos="0 0 {wall:.4f}" mass="0.08"/>
   {walls}
  </body></body></worldbody>
 <actuator><position joint="x" kp="2000" kv="150"/><position joint="y" kp="2000" kv="150"/></actuator></mujoco>"""


def pour_scene(worlds: int = 1, seed: int = 0, volume: float = 100e-6, vary: bool = True) -> Any:
    """``volume`` m^3 of water pouring from ``SPOUT`` at 100 ml/s, and the bowl carriage on the floor.

    Example: ``alloy.build(pour_scene(8), worlds=8)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415

    scene = Scene(worlds=worlds, dt=0.02)
    scene.add_ground()
    scene.add_pour("water", volume, pos=SPOUT, rate=1e-4, speed=0.5)
    scene.add_robot(TRAY, mjcf=bowl_carriage(), pos=(0.0, 0.0, 0.002))
    return maybe_vary(scene, vary, seed)


def in_bowl(s: BatchState) -> Any:
    """(W,) fraction of the water inside the bowl."""
    d = s["water.points"] - s[f"{TRAY}/bowl.pos"][:, None, :]
    hx, hy = BOWL_INNER[0] / 2, BOWL_INNER[1] / 2
    ins = (abs(d[..., 0]) < hx) & (abs(d[..., 1]) < hy) & (d[..., 2] > 0.0) & (d[..., 2] < BOWL_INNER[2] + 0.01)
    return s.xp.float(ins).mean(-1)


def pour_task(camera: bool = False, max_steps: int = 100, start_offset: float = 0.08, goal: float = 0.75) -> Task:
    """Reward: the water in the bowl (fraction, every step) minus the bowl's distance from under the spout;
    success: at least ``goal`` of the water in the bowl. Each reset moves the bowl up to ``start_offset`` m.

    Example: ``VectorEnv(pour_scene(8), pour_task(), worlds=8)``."""
    caught = C.custom("in_bowl", in_bowl)
    under = C.distance(f"{TRAY}/bowl", (SPOUT[0], SPOUT[1], 0.0))
    cams = {"front": front_camera((0.35, -0.35, 0.35), (0.0, 0.0, 0.08))} if camera else {}
    return Task(
        name="Pour-v0", obs=[TRAY, f"{TRAY}/bowl.pos", "action"],
        actions=ActionSpec([f"{TRAY}/x", f"{TRAY}/y"], mode="delta", delta=0.01),
        reward={"in_bowl": (1.0, caught), "under": (-1.0, under)},
        success=caught >= goal, end_on_success=False, max_steps=max_steps, cameras=cams,
        randomize=[jitter_joint(f"{TRAY}/x", by=start_offset), jitter_joint(f"{TRAY}/y", by=start_offset)])


# --- ClothFold ---------------------------------------------------------------------------------------------------

GRIP = "gripper"
TOWEL = (0.24, 0.24)                       # m, square towel
# m of close travel past the fingers' meeting point (gantry_gripper's ``squeeze``): the stopped fingers press the
# 2-layer towel ridge with ~32 N instead of 2.6 N, so the pinch holds through the fold (gap #2, SC4, 2026-11-08)
GRIP_SQUEEZE = 0.02


def cloth_fold_scene(worlds: int = 1, seed: int = 0, resolution: int = 13, vary: bool = True) -> Any:
    """A 24 cm square towel (2 mm, 30 g, cloth self-contact on, so folded layers rest on each other) flat on the
    table, and the Cartesian gripper above its +x +y corner. Example: ``alloy.build(cloth_fold_scene(4), worlds=4)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415

    from alloy.envs.robots import gantry_gripper  # noqa: PLC0415

    scene = Scene(worlds=worlds, dt=0.02)
    scene.add_ground()
    scene.add_cloth("towel", size=TOWEL, pos=(0.0, 0.0, 0.003), resolution=resolution, mass=0.03, thickness=0.002,
                    friction=0.8, self_contact=True)
    grip = gantry_gripper(squeeze=GRIP_SQUEEZE)
    scene.add_robot(GRIP, mjcf=grip, pos=(TOWEL[0] / 2 - 0.01, TOWEL[1] / 2 - 0.01, 0.2))
    return maybe_vary(scene, vary, seed)


_CORNERS: dict[tuple[int, float], int] = {}


def corner(s: BatchState, sign: float) -> Any:
    """(W, 3) the towel's particle nearest its (+x, +y) corner (``sign`` 1) or (-x, -y) corner (-1) at the start
    (its index is found once per towel size)."""
    pts = s["towel.points"]
    key = (int(pts.shape[1]), float(sign))
    if key not in _CORNERS:
        start = s.start["towel.points"] if s.start is not None else pts
        _CORNERS[key] = int(s.xp.to_numpy((start[0, :, 0] + start[0, :, 1]) * sign).argmax())
    return pts[:, _CORNERS[key], :]


def cloth_fold_task(camera: bool = False, max_steps: int = 150, grasp: str = "friction") -> Task:
    """Fold the towel along its diagonal: put its +x +y corner on its -x -y corner. Reward: the gripper reaching
    the corner, then the two corners coming together; success: corners within 3 cm and the folded corner down
    (under 2 cm). ``grasp``: "friction" (default: the fingers hold the towel by friction alone) or "attach" (an
    opt-in shortcut, not a friction grasp: towel vertices between the closed fingers are held to a finger until
    it opens, as cloth benchmarks grasp; ``alloy.envs.cloth_attach``).
    Example: ``VectorEnv(cloth_fold_scene(4), cloth_fold_task(), worlds=4)``;
    ``make("ClothFold-v0", grasp="attach")``."""
    if grasp not in ("friction", "attach"):
        from alloy.envs.errors import EnvError  # noqa: PLC0415

        raise EnvError(f"ClothFold-v0: grasp must be 'friction' or 'attach', got {grasp!r}",
                       "pass grasp='friction' (the default) or grasp='attach'", field="grasp")
    from alloy.envs.library_tasks import grasp_point  # noqa: PLC0415
    from alloy.envs.robots import gantry_actions  # noqa: PLC0415

    a, b = ("corner_a", lambda s: corner(s, 1.0)), ("corner_b", lambda s: corner(s, -1.0))
    gap = C.custom("corner_gap", lambda s: s.xp.norm(corner(s, 1.0) - corner(s, -1.0)))
    reach = C.custom("reach", lambda s: s.xp.norm(grasp_point(s) - corner(s, 1.0)))
    folded = C.custom("folded", lambda s: (gap(s) < 0.03) & (corner(s, 1.0)[:, 2] < 0.02))
    cams = {"front": front_camera((0.3, -0.3, 0.35), (0.0, 0.0, 0.0))} if camera else {}
    return Task(
        name="ClothFold-v0", obs=[GRIP, a, b, "towel", "action"], actions=gantry_actions(GRIP, yaw=False),
        reward={"reach": (-0.5, reach), "corner_gap": (-2.0, gap), "folded": (5.0, folded)},
        success=folded, end_on_success=False, max_steps=max_steps, cameras=cams,
        randomize=[jitter_joint(f"{GRIP}/x", by=0.02), jitter_joint(f"{GRIP}/y", by=0.02)],
        on_build=_attach_on_grasp if grasp == "attach" else None)


def _attach_on_grasp(sim: Any) -> None:
    from alloy.envs.cloth_attach import install  # noqa: PLC0415

    install(sim, GRIP)


__all__ = ["BOWL_INNER", "GRIP_SQUEEZE", "SPOUT", "TOWEL", "TRAY", "bowl_carriage", "cloth_fold_scene",
           "cloth_fold_task", "corner", "in_bowl", "pour_scene", "pour_task"]
