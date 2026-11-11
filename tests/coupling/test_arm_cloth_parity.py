# SPDX-License-Identifier: Apache-2.0
"""CPU-vs-GPU parity for an arm working cloth (R2-E3, coordinator 2026-11-07).

The trial agent's B3 fold ([internal note], R2-E3) froze 56/64 worlds on an L4. On the CPU,
about 1 world in 14 froze. The cause was the same on both devices: proxy contacts re-collided once per exchange,
outrun by a whipping sheet (fixed in alloy/coupling/contact_band.py). The GPU's worlds move in near lockstep, so they
hit it together. Before that, the same start on both devices agreed to micrometres up to the lift (270 periods:
bodies within 0.002 mm, cloth within 0.005 mm). This law keeps the coupled arm-cloth contact the same on both devices.

The scene is B3's arm and sheet, with the arm working the sheet in a way real cotton supports. B3's own pinch held
only because its 4 mm sheet bent about 300x its fabric (R2-E1, main 0bda7b5cd). With the fabric's bending, the 3 cm
overhang hangs down the table's side, and the plates meet the table. Here the arm brings its gripper, plates nearly
shut, down onto a 4 mm cotton sheet lying on a table. The lower 1 kg plate presses the sheet and drags it 6 cm across
the table (pad mu 2.0 against the table's 0.5), then lifts off. Open-loop joint targets, 2 worlds. The same scene,
seed and script run on the CPU (Warp route, the suite's default) and on CUDA:
- both devices stay healthy;
- the arm bodies and the cloth agree to 0.05 mm and 0.2 mm through the press and the drag;
- the drag carried the sheet on both.
L4, 11-07 (main d8eadb63c): bodies within 0.012 mm and cloth within 0.010 mm over the 130 periods; the sheet dragged
53.9 mm in both worlds on both devices ([internal note] notes/wp4-r2e3/parity.log).
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy

H0, L1, L2, L3 = 0.30, 0.36, 0.36, 0.11      # m: shoulder height, links, wrist -> plate centre (the trial's arm)
TOP = 0.15                                     # m: table top
SHUT = 0.041                                   # m: each plate's closing travel: 4 mm between the pads' faces
UNDER = 0.045 - SHUT + 0.002                   # m: tool point to the lower pad's bottom face, plates at SHUT
PRESS = 0.002                                  # m: the lower pad's target below the sheet's top
DRAG = 0.06                                    # m
BODY_TOL, CLOTH_TOL = 5.0e-5, 2.0e-4           # m: device parity through the press and the drag
COMPARED = 130                                 # periods compared: approach 60 + press 30 + drag 40

NO_CONTACT = 'contype="0" conaffinity="0"'
ARM = f"""<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
 <body name="turret" pos="0 0 {H0}">
  <joint name="yaw" type="hinge" axis="0 0 1" range="-3 3" damping="5"/>
  <geom type="capsule" fromto="0 0 0 0 0 0.02" size="0.05" mass="1.0" {NO_CONTACT}/>
  <body name="upper" pos="0 0 0.02">
   <joint name="shoulder" type="hinge" axis="0 1 0" range="-2.2 1.2" damping="8"/>
   <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.03" mass="1.5" {NO_CONTACT}/>
   <body name="fore" pos="{L1} 0 0">
    <joint name="elbow" type="hinge" axis="0 1 0" range="-2.6 2.6" damping="5"/>
    <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.025" mass="1.0" {NO_CONTACT}/>
    <body name="hand" pos="{L2} 0 0">
     <joint name="wrist" type="hinge" axis="0 1 0" range="-2.6 2.6" damping="4"/>
     <geom type="box" size="0.02 0.03 0.05" mass="0.5" {NO_CONTACT}/>
     <body name="finger_top">
      <joint name="close_t" type="slide" axis="0 0 -1" range="0 0.045" damping="30"/>
      <geom name="pad_top" type="box" size="0.02 0.07 0.002" pos="{L3} 0 0.045" mass="1" friction="2 0.005 0.0001"/>
      <geom type="box" size="0.01 0.01 0.0225" pos="0 0 0.0225" mass="0.01" {NO_CONTACT}/>
     </body>
     <body name="finger_bot">
      <joint name="close_b" type="slide" axis="0 0 1" range="0 0.045" damping="30"/>
      <geom name="pad_bot" type="box" size="0.02 0.07 0.002" pos="{L3} 0 -0.045" mass="1" friction="2 0.005 0.0001"/>
      <geom type="box" size="0.01 0.01 0.0225" pos="0 0 -0.0225" mass="0.01" {NO_CONTACT}/>
     </body>
    </body>
   </body>
  </body>
 </body>
</worldbody><actuator>
 <position name="a_yaw" joint="yaw" kp="800" kv="60"/>
 <position name="a_shoulder" joint="shoulder" kp="1500" kv="100"/>
 <position name="a_elbow" joint="elbow" kp="1000" kv="70"/>
 <position name="a_wrist" joint="wrist" kp="600" kv="40"/>
 <position name="a_close_t" joint="close_t" kp="3000" kv="60"/>
 <position name="a_close_b" joint="close_b" kp="3000" kv="60"/>
</actuator></mujoco>"""
"""B3's arm (the trial's arm.arm_side_mjcf, half_w 0.07): 1 kg plates 4 x 14 cm, 4 mm thick, mu 2.0, closing kp 3000."""


def _ik(p: np.ndarray) -> dict[str, np.ndarray]:
    """B3's planar IK (arm.ik_tilt, pitch 0): joint targets [W] that put the plates' centre at p [W, 3]."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    rw, zw = np.hypot(x, y) - L3, z - (H0 + 0.02)
    d = np.minimum(np.hypot(rw, zw), L1 + L2 - 1e-3)
    phi1 = np.arctan2(zw, rw) + np.arccos(np.clip((L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d), -1, 1))
    phi2 = np.arctan2(zw - L1 * np.sin(phi1), rw - L1 * np.cos(phi1))
    sh, el = -phi1, phi1 - phi2
    return {"yaw": np.arctan2(y, x), "shoulder": sh, "elbow": el, "wrist": -sh - el}


def _script(start: np.ndarray, press: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """(tool point [W, 3], each plate's closing travel) per period: from the arm's zero pose to 5 cm over the press
    point while the plates shut (0.3 s), down to 8 mm over it (0.3 s), press (0.2 s) and hold (0.1 s), drag (0.4 s),
    lift off (0.2 s)."""
    steps: list[tuple[np.ndarray, float]] = []

    def seg(a, b, n, f0, f1):
        for k in range(1, n + 1):
            s = k / n
            steps.append((a + (b - a) * s, f0 + (f1 - f0) * s))
        return b

    up = np.array([0.0, 0.0, 1.0])
    p = seg(start, press + 0.05 * up, 30, 0.0, SHUT)
    p = seg(p, press + 0.008 * up, 30, SHUT, SHUT)
    p = seg(p, press, 20, SHUT, SHUT)
    p = seg(p, press, 10, SHUT, SHUT)
    p = seg(p, press + np.array([DRAG, 0.0, 0.0]), 40, SHUT, SHUT)
    seg(p, p + 0.05 * up, 20, SHUT, SHUT)
    return steps


def _run(device: str) -> dict[str, np.ndarray | bool]:
    W = 2
    pos = np.array([[0.50, -0.004, TOP + 0.004], [0.505, 0.003, TOP + 0.004]])
    scene = alloy.Scene(worlds=W, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.45, 0.4, TOP), pos=(0.595, 0, TOP / 2), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.20, 0.16), pos=pos, resolution=(11, 9), mass=0.03, thickness=0.004,
                    friction=1.0, self_contact=True)
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0))
    sim = alloy.build(scene, seed=0, device=device)
    sim.step(1)
    sim.reset()
    sim.settle(0.3)
    p0 = sim.state(["cloth"])["cloth.points"].copy()
    top = p0[:, :, 2].mean(1) + 0.002                                # the sheet's top (particle radius = h / 2)
    press = np.stack([p0[:, :, 0].min(1) + 0.04, p0[:, :, 1].mean(1), top - PRESS + UNDER], 1)
    start = np.tile([L1 + L2 + L3, 0.0, H0 + 0.02], (W, 1))         # the tool point at the arm's zero pose
    bodies = ["arm/hand", "arm/finger_top", "arm/finger_bot"]
    out = {"bodies": [], "cloth": []}
    for tool, f in _script(start, press):
        targets = _ik(tool)
        targets.update(close_t=f, close_b=f)
        sim.set_targets(targets)
        sim.step(1)
        s = sim.state(bodies + ["cloth"])
        out["bodies"].append(np.stack([s[b + ".pos"] for b in bodies], 1))
        out["cloth"].append(s["cloth.points"].copy())
    res: dict[str, np.ndarray | bool] = {k: np.stack(v) for k, v in out.items()}
    res["start"] = p0
    res["ok"] = bool(sim.check()["ok"])
    return res


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_arm_dragging_cloth_runs_the_same_on_cpu_and_gpu():
    """B3's arm presses and drags a 4 mm cotton sheet, 2 worlds, on the CPU (Warp route) and CUDA: both healthy, the
    arm and cloth within 0.05 / 0.2 mm through the press and the drag (COMPARED periods), the sheet dragged on both."""
    import warp as wp
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device")
    cpu, gpu = _run("cpu"), _run("cuda:0")
    assert cpu["ok"] and gpu["ok"]
    n = COMPARED
    body = float(np.abs(cpu["bodies"][:n] - gpu["bodies"][:n]).max())
    cloth = float(np.linalg.norm(cpu["cloth"][:n] - gpu["cloth"][:n], axis=-1).max())
    moved = [(r["cloth"][-1][:, :, 0].mean(1) - r["start"][:, :, 0].mean(1)).tolist() for r in (cpu, gpu)]
    print(f"arm+cloth parity: body {body * 1e3:.4f} mm, cloth {cloth * 1e3:.4f} mm over {n} periods; "
          f"sheet dragged cpu {moved[0]} gpu {moved[1]} m")
    assert body < BODY_TOL and cloth < CLOTH_TOL, (body, cloth)
    assert min(min(m) for m in moved) > 0.03, moved                    # the plate carried the sheet on both
