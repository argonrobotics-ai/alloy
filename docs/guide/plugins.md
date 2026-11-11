# Plug-ins: your own physics

A plug-in adds a force (wind, magnets, buoyancy), a sensor (range finder, tactile), an object (a hydraulic
cylinder) or a whole solver. The kit tests it automatically (laws you state, no crashes, determinism, worlds that do
not leak into each other, speed). In a simulation a plug-in is ordinary Python running with your permissions: load
only plug-ins you trust. `python -m alloy.plugins check my_plugin --sandbox` runs the check, and only the check,
confined on Linux (no network, limited files).

```bash
python -m alloy.plugins list                        # built-in and installed plug-ins, JSON
python -m alloy.plugins new my_wind --kind force    # a working template to edit
python -m alloy.plugins check my_wind               # the harness: JSON report, a fix for every failure
```

- Write one by subclassing `ForcePlugin`, `SensorPlugin`, `ObjectPlugin` or `SolverPlugin` from `alloy.plugins`.
  The full guide, with a complete force plug-in, the hooks of each kind, rules the harness enforces and pitfalls:
  `alloy/plugins/GUIDE.md`.
- Use one in a simulation: `alloy.build(scene, plugins=[...])` takes plug-ins (or functions of the built scene that
  return one). They then run inside `sim.run`, `step` and `reset`, and `sim.plugins.read()` gives their read-outs:

```python
import numpy as np
import alloy
import alloy.plugins

scene = alloy.Scene(dt=0.01)
scene.add_ground(friction=0.1)
for name, x in (("a", 0.0), ("b", 0.05)):                     # two 2 cm cubes, 3 cm apart
    scene.add_box(name, size=(0.02, 0.02, 0.02), pos=(x, 0, 0.0101), density=7500, friction=0.1)
magnets = alloy.plugins.create("magnet", bodies=("a", "b"), moments=(1.0, 1.0), axes=((1, 0, 0), (1, 0, 0)))
sim = alloy.build(scene, plugins=[magnets])
traj = sim.run(0.5, record=["a", "b"])
gap = np.linalg.norm(traj["b.pos"][:, 0] - traj["a.pos"][:, 0], axis=1) - 0.02
print(f"gap {gap[0] * 1e3:.0f} mm -> {gap[-1] * 1e3:.1f} mm")   # the cubes snap together
```

- For a scene assembled with the engine layers: `alloy.plugins.simulation(built, [plugin])`, stepped with
  `step_period()`; see `examples/plugins/`.
- A force plug-in's work (wind on a ball) counts as energy from outside in the health checks, so it raises no
  ENERGY_JUMP ([health.md](health.md)).
- Heat on particles (warming, melting, boiling, cooking) is a built-in plug-in: [heat.md](heat.md); its limits are in
  `docs/known_gaps/heat-*.md`.
- Share one as a Python package with an `alloy.plugins` entry point; `pip install` it and every command finds it by
  name.
