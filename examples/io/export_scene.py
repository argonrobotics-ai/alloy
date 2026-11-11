# SPDX-License-Identifier: Apache-2.0
"""Export an Alloy scene to MJCF, URDF and USD, then load the MJCF in MuJoCo and step it.

Run:  python examples/io/export_scene.py [scene_name] [out_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco

from alloy.io import export_scene


def main(scene: str = "demo_peg_d12", out: str = "out/io_export") -> dict:
    res = export_scene(scene, out)                 # MJCF + URDF (one file per object) + USD, checked on load
    print(json.dumps(res.to_json()["files"], indent=2))
    for row in res.unmapped:                       # what the files cannot carry, and what was done instead
        print(f"not exported: {row.item}: {row.reason}")
    m = mujoco.MjModel.from_xml_path(res.files["mjcf"])
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, nstep=int(0.2 / m.opt.timestep))    # 0.2 s in MuJoCo with Alloy's contact values
    print(f"MuJoCo stepped {d.time:.3f} s; body heights: {d.xpos[1:, 2].round(4).tolist()}")
    return res.to_json()


if __name__ == "__main__":
    main(*sys.argv[1:3])
    sys.exit(0 if Path(sys.argv[2] if len(sys.argv) > 2 else "out/io_export").exists() else 1)
