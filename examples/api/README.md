# alloy.api examples

Each file runs end to end on its own: `python examples/api/<file> [out_dir]`. It uses the GPU when there is one, else
the CPU, which is slower. The first run of a new scene compiles kernels (about a minute on a GPU; cached after that).

| File | Shows |
|---|---|
| `01_drop_box.py` | `alloy.Scene`, `add_ground`, `add_box(size=, mass=, static=)`, `sim.run(record=)`, saving a trajectory |
| `02_gripper_lift.py` | a robot from an MJCF string, `run(targets={"joint": (start, end)})` ramps, reading robot parts by name |
| `03_many_worlds.py` | one start per world (`pos` as a (worlds, 3) array), `randomize`, `estimate`, world-steps per second |
| `04_cloth_on_ball.py` | `add_cloth`, `sim.routes()`, cloth vertices and `sim.faces()` |
| `05_pour_liquid.py` | `add_open_cylinder` (a cup from static boxes), `add_pour` (a stream: how to fill a container), `water.active` |
| `06_tilt_pour.py` | pouring from a held container: a kinematic `add_open_box` jug turned about its lip with `sim.drive(angular_velocity=, pivot=)`; where to put the cup (a few cm past the lip) |
| `07_heat_water.py` | heat: `add_liquid(temperature=)` (K), `set_heater` on a pan's floor, `set_heat(time_scale=)`, `sim.heat()` and the recorded `water.temperature` |
| `08_robot_per_world.py` | a free-base robot (MJCF `<freejoint/>`: a carton with a flap) started per world (`pos` (worlds, 3), `randomize`) and lifted with `sim.set_pose`, its links following |

Names: objects keep the names you give them, and robot parts are `"<robot>/<body or joint>"`. `sim.names()` lists
every name. State keys are `"<name>.pos"`, `"<name>.quat_wxyz"` (or `quat_xyzw`), `"<name>.vel"`, `"<joint>.q"` and
`"<cloth or liquid>.points"`, each with a leading world axis (`traj.world(0)` drops it).
