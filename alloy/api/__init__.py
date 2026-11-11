# SPDX-License-Identifier: Apache-2.0
"""Alloy's public Python API: build a scene, simulate it, read its state, check it, render it, estimate its cost.

Everything here is also available at the top level (``alloy.Scene``, ``alloy.build``, ...). Names outside
``alloy.api`` are engine internals and may change between versions; this API follows semantic versioning from 0.1.0.

Example (a file in, a trajectory out)::

    import alloy
    sim = alloy.load("robot.xml")                 # MJCF/URDF/USD, a scene card, or a library scene name
    traj = sim.run(2.0)                           # every body, joint and particle object; traj.save("t.npz")

Example (compose a scene in Python)::

    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5), mass=1.0)
    sim = alloy.build(scene)                      # GPU if there is one, else CPU
    traj = sim.run(1.0, record=["cube"])          # traj["cube.pos"] has shape (frames, worlds, 3)
    print(sim.state(world=0)["cube.pos"], sim.check()["ok"])

Containers and sockets: ``scene.add_open_cylinder`` / ``add_open_box`` / ``add_block_with_hole`` (static boxes,
exact contact). Everything is named: objects by the name you gave them, robot parts as "<robot>/<body or joint>"; ``sim.names()``
lists them. Errors say what to change (``err.fix``). Runnable examples: ``examples/api/``.
"""

from __future__ import annotations

from alloy.api._lazy import PUBLIC as __all__  # noqa: F401  (standard library only: `import alloy` runs this)
from alloy.api._lazy import module_hooks

__getattr__, __dir__ = module_hooks(globals(), "alloy.api")
