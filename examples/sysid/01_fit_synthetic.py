# SPDX-License-Identifier: Apache-2.0
"""System identification with no robot: fit a 2-joint arm to a synthetic recording and check the answer.

alloy.sysid.demo simulates the arm with hidden joint parameters (a different servo stiffness, more joint damping, dry
friction, a 12 ms action delay) and sensor noise, and returns that recording with the hidden values. alloy.sysid.fit
finds them again from the commands and positions alone. The script prints the report, a table of file / hidden /
fitted values, and the tracking error of the recording's positions with and without add_robot(sysid=).

Run: python examples/sysid/01_fit_synthetic.py            (defaults suit a GPU: a few minutes; a slow 4-core CPU
about 10 min). Quick CPU check: --worlds 16 --generations 4 --device cpu (about 1 min, rough numbers).
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np

import alloy
import alloy.sysid

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.6">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2" frictionloss="0.1" armature="0.01"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  <body name="link2" pos="0.3 0 0"><joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
  <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.5"/></body>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/><general joint="elbow" gainprm="80" biastype="affine"
  biasprm="0 -80 -4" forcerange="-20 20"/></actuator></mujoco>"""


def replay_rms_deg(path, rec, sysid, device) -> np.ndarray:
    """RMS error (degrees, per joint) between the recording's positions and a replay of its commands."""
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", path, sysid=sysid)
    sim = alloy.build(scene, device=device)
    ep = rec.episodes[0]
    sim_q = []
    for row in ep.command[:-1]:                           # command k is held until sample k + 1
        sim.set_targets({j: float(row[k]) for k, j in enumerate(rec.joints)})
        sim.step(1)
        sim_q.append([float(sim.state(world=0)[f"arm/{j}.q"]) for j in rec.joints])
    return np.degrees(np.sqrt(np.mean((np.array(sim_q) - ep.position[1:]) ** 2, axis=0)))


def print_table(result, hidden, joints) -> None:
    print("\njoint     parameter        file    hidden    fitted")
    for k, joint in enumerate(joints):
        rows = {"kp": "kp", "kd + damping": None, "friction": "friction", "delay (ms)": "delay"}
        for label, key in rows.items():
            get = (lambda d: d["kd"][k] + d["damping"][k]) if key is None else (lambda d, f=key: d[f][k])
            scale = 1e3 if key == "delay" else 1.0
            file_value = 0.0 if key == "delay" else get(result.nominal)
            hid, fit = scale * get(hidden), scale * get(result.values)
            print(f"{joint:9s} {label:13s} {file_value:9.4g} {hid:9.4g} {fit:9.4g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worlds", type=int, default=32, help="candidate parameter sets per generation (>= 16)")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--device", default=None, help="cpu or cuda:0 (default: a GPU when there is one)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arm.xml"                      # --robot and add_robot take a file
        path.write_text(ARM)
        rec, hidden = alloy.sysid.demo(path, seconds=6.0, device=args.device or "cpu")
        print(f"synthetic recording: {rec.joints}, {rec.duration:.1f} s; hidden values {dict(hidden)}")
        result = alloy.sysid.fit(path, rec, worlds=args.worlds, generations=args.generations, device=args.device)
        result.save(Path(tmp) / "arm.sysid.json")
        print(alloy.sysid.report(result, robot_name="arm"))
        print_table(result, hidden, rec.joints)
        plain = replay_rms_deg(path, rec, None, args.device)
        fitted = replay_rms_deg(path, rec, result, args.device)
    for k, joint in enumerate(rec.joints):
        print(f"{joint}: tracking error vs the recording {plain[k]:.4f} deg without sysid, {fitted[k]:.4f} deg with")


if __name__ == "__main__":
    main()
