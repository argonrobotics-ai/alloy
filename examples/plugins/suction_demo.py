# SPDX-License-Identifier: Apache-2.0
"""Suction cup demo: a gantry cup picks a box, carries it 0.2 m and lets it go by venting (engine item 10).

    python examples/plugins/suction_demo.py            # 4 worlds on the CPU, prints the cup pressure and seal

The scene, script and checks are ``alloy/tasks/suction_pick.py``; the plug-in is ``alloy.plugins.library.suction``.
"""

from __future__ import annotations

import numpy as np

from alloy.plugins.harness.scene import build_simulation
from alloy.tasks import suction_pick as T


def main() -> None:
    plug = T.plugin()
    sim = build_simulation(T.scene(4), [plug], "cpu", 4)
    dofs = T._dofs(sim.model)
    box = [i for i, lab in enumerate(sim.model.body_label) if lab.split("/")[-1] == "box"]
    tgt = sim.control.joint_target_q.numpy()
    for k in range(T.PERIODS):
        t = k * T.CONTROL_DT
        q = T.targets(t + T.CONTROL_DT)
        for a in range(3):
            tgt[dofs[:, a]] = q[a]
        sim.control.joint_target_q.assign(tgt.astype(np.float32))
        T.set_inputs(plug, t)
        sim.step_period()
        if (k + 1) % 20 == 0:
            r = plug.read()
            z = sim.state.body_q.numpy()[box[0], 2]
            print(f"t {t + T.CONTROL_DT:4.2f} s  cup {r['pressure'][0, 0] / 1e3:6.1f} kPa  seal {r['seal'][0]:.2f}  "
                  f"force {r['force'][0]:6.1f} N  box z {z * 1e3:6.1f} mm")
    end = sim.state.body_q.numpy()[box]
    print("box ends at x =", np.round(end[:, 0], 4), "(target", T.PLACE[0], ")")


if __name__ == "__main__":
    main()
