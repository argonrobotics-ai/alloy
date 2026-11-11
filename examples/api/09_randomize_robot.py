# SPDX-License-Identifier: Apache-2.0
"""Per-world randomization of a robot: the shipped Go2 stands in 8 worlds, each with its own mass, floor friction and
servo stiffness, and a payload on its back that is drawn again at every reset.

Each randomized field is one scene.randomize(target, field, lo, hi) row: one uniform draw per row and world, seeded by
alloy.build(seed=). sim.randomized() shows what every world holds now; the standing height differs per world because
mass and servo stiffness do. Building the Go2 prints many NewtonWarning lines about geom margins: they are expected.
Run: python examples/api/09_randomize_robot.py
"""
import alloy
from alloy.cards.library import default_root

W = 8
scene = alloy.Scene(worlds=W, dt=0.01)
scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")     # the file brings its own floor
scene.randomize("go2", "mass_scale", 0.8, 1.2)                      # every link's mass and inertia x one draw per world
scene.randomize("go2", "friction", 0.4, 1.2)                        # sliding friction of the robot's geoms and floor
scene.randomize("go2", "kp_scale", 0.8, 1.2)                        # servo stiffness of every actuated joint
scene.randomize("go2/base", "mass_add", 0.0, 3.0, when="reset")     # payload in kg, drawn again at every reset
sim = alloy.build(scene, seed=0)


def table(heights=None):
    drawn = sim.randomized()
    print("world  mass_scale  friction  kp_scale  payload_kg" + ("  base_z_m" if heights is not None else ""))
    for w in range(W):
        row = f"{w:5d}  {drawn['go2.mass_scale'][w]:10.3f}  {drawn['go2.friction'][w]:8.3f}  "
        row += f"{drawn['go2.kp_scale'][w]:8.3f}  {drawn['go2/base.mass_add'][w]:9.3f}"
        print(row + ("" if heights is None else f"  {heights[w]:8.4f}"))


sim.run(1.0)
table(sim.state(["go2/base"])["go2/base.pos"][:, 2])
print("healthy:", sim.check()["ok"])
sim.reset()                                  # build rows keep their values; the when="reset" row is drawn again
print("after sim.reset(): payload_kg", sim.randomized()["go2/base.mass_add"].round(3))
