# SPDX-License-Identifier: Apache-2.0
"""What failure looks like: an API error with its fix, and a world that goes wrong, caught by health and check().

Shows: alloy.SceneError / UnknownNameError (.message, .fix, .to_dict()), sim.health(), sim.check(), and why a
failed world must be read from health (it is frozen at its last good state while the others go on).
Run: python examples/basics/05_errors_and_health.py
"""
import json

import numpy as np

import alloy

errors = []
try:
    alloy.Scene().add_sphere("ball", radius=-0.1)
except alloy.SceneError as e:
    errors.append(e.to_dict())

scene = alloy.Scene(worlds=2, dt=0.01)
scene.add_ground()
# world 0 starts the two boxes apart; world 1 starts them deeply overlapped (a modelling mistake)
scene.add_box("a", size=(0.1, 0.1, 0.1), pos=np.array([[0, 0, 0.05], [0, 0, 0.05]]), mass=1.0)
scene.add_box("b", size=(0.1, 0.1, 0.1), pos=np.array([[0.3, 0, 0.05], [0.02, 0, 0.06]]), mass=1.0)
sim = alloy.build(scene)
sim.run(0.5, record=[])
try:
    sim.state(["c"])
except alloy.UnknownNameError as e:
    errors.append(e.to_dict())
health = sim.health()
report = sim.check()
print(json.dumps({"errors": errors, "ok": report["ok"],
                  "severity_per_world": [w["severity"] for w in health["worlds"]],
                  "problems": report["problems"]}, default=str))
