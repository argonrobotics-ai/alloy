# Alloy guide

`AGENTS.md` at the repository root is the quick start; read it first. These pages go one level deeper, one topic
each. Every code block is complete and runs as shown (GPU if present, else CPU).

| Page | What it covers |
|---|---|
| [known_limits.md](known_limits.md) | **read first:** every known engine limit, its size, the workaround and the planned fix |
| [install.md](install.md) | pip install, optional extras, GPU or CPU and what each costs, the kernel compile cache, quiet logs |
| [scenes.md](scenes.md) | shapes, meshes, materials and friction, library objects, loading and saving scenes |
| [robots.md](robots.md) | MJCF / URDF / USD robots, the shipped robots, joint names, targets, grasping |
| [deformables.md](deformables.md) | cloth and liquid: sizes, resolution, containers, volumes, which solver runs what |
| [rods.md](rods.md) | cables, wires and cords: bend and twist, clamps, wrenches, gripping, accuracy, cost, limits |
| [worlds.md](worlds.md) | many worlds, per-world starts, randomization, seeds, timing, cost estimates, profiles |
| [randomize_robots.md](randomize_robots.md) | per-world robot values (opt-in): link mass, centre of mass, friction, servo gains, armature and gravity, drawn at build or at every reset; a random start at every reset and timed pushes (CPU-checked only); uniform draws only, with what is not randomized yet |
| [health.md](health.md) | health flags and their fixes, `sim.check()`, precision contact, the error classes |
| [cards.md](cards.md) | the YAML cards behind every scene and the engine layers under the API |
| [camera.md](camera.md) | images of a scene: a quick look, a camera over every world, training images, depth, segmentation, instance ids, classes and optical flow (rigid motion only) |
| [frame-export.md](frame-export.md) | hand frames of picked worlds of a running simulation to a viewer you write (`alloy.export`), without changing the run |
| [viewer.md](viewer.md) | watch a running simulation in a browser tab, a desktop window, Rerun or a USD file: pause, step, pick worlds, rigid contact points and forces (`alloy.viewer`); opt-in joint sliders and mouse dragging (CPU-tested; dragging robot links not tested) |
| [check.md](check.md) | `alloy.check`: is a scene physically sound before you use it, and what to change |
| [import_export.md](import_export.md) | MJCF / URDF / USD in and out |
| [plugins.md](plugins.md) | your own forces, sensors, objects and solvers, with a test kit (plug-ins run with your permissions) |
| [heat.md](heat.md) | heat on particles: warming, melting, boiling, cooking; heaters, air, read-outs and checks |
| [cutting.md](cutting.md) | cutting soft objects with a kinematic knife: cuttable cards (`cut_model: copies` or `seams`), the knife's `blade:` edge, limits |
| [cli.md](cli.md) | the `alloy` command line (JSON in and out) and the MCP tool server |
| [envs.md](envs.md) | Gym / PyTorch vector environments over parallel worlds |
| [rl.md](rl.md) | adapters for rsl_rl, Stable-Baselines3 and skrl, a one-command PPO train, observation groups for a privileged critic; which library reached 90 % success on which task and which did not (one to six runs per row); opt-in first versions of hand-motion actions for arms, noise / delay / history layers and PD / DC-motor actuator models, each with what it does not cover |
| [locomotion.md](locomotion.md) | legged robots: the Go2 velocity environment on flat ground or on rough terrain tiles with a level curriculum (commands, pushes, policy and critic observations, foot contact forces) and a PPO train command; no height scan, stairs or gaps, no actuator model |
| [imitation.md](imitation.md) | imitation-learning data: record episodes to robomimic HDF5 or LeRobotDataset v3, keyboard teleoperation, scripted demo generation (3 of 8 ready-made envs), replay, policy evaluation (no training code) |
| [sysid.md](sysid.md) | system identification (optional): fit a simulated robot's servo stiffness, damping, dry friction and action delay to a recording of the real one; one command, a report, then `add_robot(sysid=)`; what is not fitted |
| [compose.md](compose.md) | the scene builder: a sentence in, a checked scene and variations out |
| [variants.md](variants.md) | `alloy.vary` / `alloy vary`: copies of a scene with objects moved and textures swapped |
| [fetch.md](fetch.md) | `alloy fetch`: download library objects from the Hugging Face Hub into the local cache |
| [create.md](create.md) | your own objects from a mesh, a photo or a text, as checked cards; `alloy audit` |

Reference: the docstrings in `alloy/api/*.py` (each public function has an example); `help(alloy.Scene)`.
Runnable programs: `examples/` (index in `examples/README.md`).
