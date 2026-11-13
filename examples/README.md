# Examples

Runnable scripts, one feature each. Every script runs as is from the repository root:

```bash
python examples/api/01_drop_box.py          # GPU if present, else CPU; the first run compiles kernels
```

- Each prints its key numbers (most as one JSON line); options and output folders are in its docstring.
- Areas: `api` and `basics` use the public API (`import alloy`); `cli` the `alloy` command line and MCP server;
  `envs` Gym environments; `il` imitation-learning data (record, teleoperate, replay, evaluate; they need `pip install h5py`); `sysid` system identification (fit a robot's joint parameters to a recording); `compose` the scene builder; `create` your own objects; `soft` soft bodies, films, food and fluids; `tactile` touch sensors; `export` frames for a viewer you write; `viewer` Alloy's own viewer (pause, step, contacts; both scripts run with `--backend null --seconds 0.2` when no viewer package is installed); `topology` cutting soft bodies; `camera`, `check`, `io` and `plugins` those packages.
- They are generic demonstrations, not solutions to particular tasks: copy one, then change sizes, materials and
  robots to match yours. `AGENTS.md` explains every call they make; `docs/guide/` goes deeper.
- All of them run in the test suite (`python -m pytest tests/oss_docs -m slow`).

## api

| Example | What it shows |
|---|---|
| [`01_drop_box.py`](api/01_drop_box.py) | Drop a box onto a table and save its trajectory (positions, wxyz quaternions) to an .npz file. |
| [`02_gripper_lift.py`](api/02_gripper_lift.py) | A two-finger gripper (an MJCF string with position actuators) grasps a cube, lifts it 12 cm and holds it. |
| [`03_many_worlds.py`](api/03_many_worlds.py) | Many parallel worlds: the same box drop in N worlds, each with its own start (pos as an (N, 3) array), plus a randomized mass; reports world-steps per second while stepping (compile excluded by a warm-up step). |
| [`04_cloth_on_ball.py`](api/04_cloth_on_ball.py) | A cotton cloth falls onto a static ball and drapes over it; saves vertex frames and triangles. |
| [`05_pour_liquid.py`](api/05_pour_liquid.py) | Pour 200 ml of water from a nozzle into a round cup (a static fixture of box walls) and let it settle. |
| [`06_tilt_pour.py`](api/06_tilt_pour.py) | Pour from a jug by tilting it: a kinematic open box of water turns about its lip over a cup. |
| [`07_heat_water.py`](api/07_heat_water.py) | Heat water in a pan: the pan's floor is a heater held at 80 °C; read the water's temperature as it warms. |
| [`08_robot_per_world.py`](api/08_robot_per_world.py) | A free-base robot placed per world: a carton with a hinged flap (an MJCF with <freejoint/> on its root body) starts at its own spot in every world, with a randomized x; sim.set_pose then lifts every carton (all links follow) and drops it again. Robot shapes collide like every scene object: contacts start 5 mm apart (contact_gap=). |
| [`09_randomize_robot.py`](api/09_randomize_robot.py) | Per-world randomization of a robot: the shipped Go2 stands in 8 worlds, each with its own mass, floor friction and servo stiffness, and a payload on its back that is drawn again at every reset. |
| [`10_randomize_robot_pushes.py`](api/10_randomize_robot_pushes.py) | Timed pushes and random starts: the shipped Go2 stands in 8 worlds, each pushed on its base at its own random times with its own random force (`scene.add_push`), and each starting at its own place and heading after a reset. Checked on the CPU only. |

## basics

| Example | What it shows |
|---|---|
| [`01_materials_friction.py`](basics/01_materials_friction.py) | Blocks of different materials slide down the same static ramp; friction decides which ones stop. |
| [`02_robot_joint_targets.py`](basics/02_robot_joint_targets.py) | A robot arm from a file (a shipped MJCF model) moves joints to targets, smoothly, and reports where its hand is. |
| [`03_per_world_targets.py`](basics/03_per_world_targets.py) | One scene, many worlds, a different joint target in each: a sweep run as a single batched simulation. |
| [`04_randomize_seed.py`](basics/04_randomize_seed.py) | Per-world object properties: a free object's density, friction and start position, drawn per world at build from a seed; same seed, same worlds. |
| [`05_errors_and_health.py`](basics/05_errors_and_health.py) | What failure looks like: an API error with its fix, and a world that goes wrong, caught by health and check(). |
| [`06_save_and_load.py`](basics/06_save_and_load.py) | A scene as data: to JSON and back, to YAML cards on disk and back, and an edit of a loaded scene. |
| [`07_estimate_cost.py`](basics/07_estimate_cost.py) | Plan a big run before building it: estimated time, memory and first-run compile for several world counts. |
| [`08_precision_fit.py`](basics/08_precision_fit.py) | A sub-millimetre fit from the library (a 12 mm peg over a 12.2 mm bore) on the precision profile. |
| [`09_engine_layers.py`](basics/09_engine_layers.py) | The engine layers under alloy.build: hand-written YAML cards, CardLibrary, SceneAssembler, build_simulation. |
| [`10_cloth_and_rigid.py`](basics/10_cloth_and_rigid.py) | Cloth and rigid bodies in one scene: a sheet lands on a table, then a ball drops onto the sheet. |
| [`11_record_trajectory.py`](basics/11_record_trajectory.py) | Recording: chosen names every k periods, the Trajectory dict, its meta, saving to .npz and reading it back. |
| [`12_custom_cards.py`](basics/12_custom_cards.py) | Your own card directory: a new material and object written as YAML cards (with sources), used from Python. |
| [`13_scene_variants.py`](basics/13_scene_variants.py) | Scene variants: four copies of one table scene with the free objects moved and each look given another texture of its family (``alloy.vary``), then one variant built and run like any scene. |
| [`14_fetch_objects.py`](basics/14_fetch_objects.py) | Fetch objects from Alloy's object library on the Hugging Face Hub, then drop them on a table. |

## camera

| Example | What it shows |
|---|---|
| [`deform_gate_scenes.py`](camera/deform_gate_scenes.py) | The camera's deformable test scenes (cloth, soft bodies), simulated and written as scene packs for reference renders. |
| [`depth_segmentation.py`](camera/depth_segmentation.py) | Depth, segmentation, instance, class and optical-flow images: 4 worlds from above, saved as one PNG (RGB, depth, segmentation, instances, flow); prints the ids seen and the flow of a block moved 2 cm. |
| [`liquid_gate_scene.py`](camera/liquid_gate_scene.py) | The camera's liquid test scene, simulated and written as a scene pack for reference renders. |
| [`quickstart.py`](camera/quickstart.py) | Camera quickstart: an Alloy scene in 4 parallel worlds, one snapshot, then a camera batch every physics step. |
| [`scene_variants_sheet.py`](camera/scene_variants_sheet.py) | Scene variants in the camera: n variants of one scene (``alloy.scene.variants.vary``: objects moved, textures swapped within their family, colours tinted), each built and rendered with plain ``sim.render()``, which draws every item's own look, tiled into one contact sheet. |
| [`textured_mesh.py`](camera/textured_mesh.py) | Textured meshes without a pack: a checker-textured cube on a ground, rendered to a PNG with ``mesh_scene``. |

## check

| Example | What it shows |
|---|---|
| [`check_and_fix.py`](check/check_and_fix.py) | Check a scene, apply the suggested fixes, check again. |

## cli

| Example | What it shows |
|---|---|
| [`01_explore.sh`](cli/01_explore.sh) | Look around the library and check a scene without simulating it (seconds, no GPU needed). |
| [`02_recipe_drop.sh`](cli/02_recipe_drop.sh) | Describe a scene as JSON (a recipe), check it, then drop a box onto a table and save the trajectory. |
| [`03_session.sh`](cli/03_session.sh) | Many calls in one process: build once, then step, ramp a joint, read state and check, all against the same |
| [`04_mcp_client.py`](cli/04_mcp_client.py) | Talk to Alloy's MCP tool server the way an AI agent's client does: start it on stdio, list its tools, build a scene, step it, read the state and close it. Standard library only (no MCP SDK needed). |
| [`05_vary.sh`](cli/05_vary.sh) | Scene variants: four copies of a table scene with the free objects moved (+-2 cm, +-15 degrees) and each look given |
| [`06_conveyor.sh`](cli/06_conveyor.sh) | Act on a running simulation without Python: a kinematic belt carries a tote (drive), stops, and the tote is put back above it (pose) and falls. Also lists the library's robots, whose names a recipe takes as a robot's path. |

## compose

| Example | What it shows |
|---|---|
| [`robot_scene.py`](compose/robot_scene.py) | A robot at the table and objects within its reach: compose a scene around your own robot file. |
| [`search_library.py`](compose/search_library.py) | What can the scene builder use? Search the card library (Alloy's own cards plus every imported library) by noun. |
| [`sentence_to_scene.py`](compose/sentence_to_scene.py) | A sentence in, a physics-checked scene and two variations out, saved as scene cards, then simulated. |
| [`soft_scene.py`](compose/soft_scene.py) | Soft scene from a sentence: water filled into a mug and a towel draped over a jar, checked, saved and simulated; `--cable` adds a USB cable plugged into the jar (the first cable on a machine waits about 11 min for the engine's cable self-test). |
| [`soft_sentences.py`](compose/soft_sentences.py) | The soft sentence suite: each sentence -> a checked scene plus N variations, one JSON line per sentence. |
| [`soft_speed.py`](compose/soft_speed.py) | Build time and step rate of saved composed scenes, at one world and at larger world counts. |
| [`spec_json.py`](compose/spec_json.py) | Exact control: write the scene request as JSON (a SceneSpec) instead of a sentence. |
| [`tilt_pour.py`](compose/tilt_pour.py) | Compose a tilting pour from a sentence and run it: water in a jar held over a bin, tipped about the jar's lip. |

## create

| Example | What it shows |
|---|---|
| [`create_compose_audit.py`](create/create_compose_audit.py) | Your own object, in a scene, audited: ``alloy create`` -> ``alloy.compose`` -> ``alloy audit``. |
| [`generator_setup.sh`](create/generator_setup.sh) | Set up the alloy.create 3D generator (photo/text -> mesh) on a Linux machine with an NVIDIA GPU (16 GB or more), |
| [`mesh_to_card.py`](create/mesh_to_card.py) | Turn a mesh file into a checked object card. |
| [`photo_to_card.py`](create/photo_to_card.py) | Turn one photo of an object into a checked object card. |
| [`text_to_card.py`](create/text_to_card.py) | Turn a text description into a checked object card. |

## envs

| Example | What it shows |
|---|---|
| [`01_random_actions.py`](envs/01_random_actions.py) | A ready-made Gym environment with random actions: every world steps at once, finished worlds restart by themselves. |
| [`02_custom_task.py`](envs/02_custom_task.py) | Your own environment: a scene, a robot from an MJCF string, and a task built from checkers. |
| [`03_camera_obs.py`](envs/03_camera_obs.py) | Camera observations: every world renders a 64x64 image each step, next to the state vector. |
| [`04_train_ppo.py`](envs/04_train_ppo.py) | Train a policy on GripperLift-v0 with a compact PPO, entirely on the GPU (needs PyTorch). |
| [`05_pick_place.py`](envs/05_pick_place.py) | PickPlace-v0: pick a YCB object off the table and drop it in a bin, with a scripted policy (no learning). |
| [`06_peg_insert.py`](envs/06_peg_insert.py) | PegInsert-v0: the gripper holds a 10 mm square peg above a block; insert it into a 10.5 mm hole (scripted policy). |
| [`07_drawer_open.py`](envs/07_drawer_open.py) | DrawerOpen-v0: grasp the drawer handle of a Lightwheel cabinet (from the RoboCasa pack) and pull it open (scripted policy). |
| [`08_pour.py`](envs/08_pour.py) | Pour-v0: water pours from a spout (Scene.add_pour, 100 ml at 100 ml/s); move a bowl under the stream to catch it. |
| [`09_cloth_fold.py`](envs/09_cloth_fold.py) | ClothFold-v0 (experimental): fold a towel along its diagonal with the Cartesian gripper (scripted policy). |
| [`10_task_ports.py`](envs/10_task_ports.py) | An unofficial port of a Gym-Anything task as a Gym environment: run one episode on the port's own script (the expert's actions), or on yours, and read the port's verdict at the end (Alloy's own checks: docs/guide/envs.md). |
| [`11_train_rl.py`](envs/11_train_rl.py) | Train GripperLift with an RL library (rsl_rl, Stable-Baselines3 or skrl; each a separate `pip install`) and a critic that sees more than the policy. Not every library learns every task: `docs/guide/rl.md`. |
| [`12_go2_walk.py`](envs/12_go2_walk.py) | A tour of the Go2 locomotion environment: the observation groups, a held stand, a push, and a hand-written open-loop trot (not a trained policy), on flat ground; training is in `docs/guide/locomotion.md`. |
| [`13_ee_actions.py`](envs/13_ee_actions.py) | Hand-motion (end-effector) actions, opt-in: a hand-written controller moves a Panda's hand to each world's target; the env turns the hand's motion into joint targets by differential inverse kinematics. No learning. |
| [`14_sim2real_layers.py`](envs/14_sim2real_layers.py) | Noise, delay and history layers on GripperLift, opt-in: observation noise, observation delay, observation history and action delay change what the policy sees, never the simulation. Nothing here has run on a real robot. |
| [`go2_walk/rough_terrain.py`](envs/go2_walk/rough_terrain.py) | The Go2 on rough ground: a terrain field with a level curriculum and a hand-written trot on it (not a trained policy; it does not look at the ground, so higher levels trip it). |

## export

| Example | What it shows |
|---|---|
| [`rerun_live.py`](export/rerun_live.py) | Live frames of picked worlds into Rerun while a run steps (needs `pip install rerun-sdk`). A tiny logger: the export itself is plain numpy (``alloy.export``), so any viewer works the same way. |

## il

| Example | What it shows |
|---|---|
| [`01_record_demos.py`](il/01_record_demos.py) | Records scripted demonstrations of a gripper lifting a cube and saves them as a robomimic-style HDF5 file. |
| [`02_teleop_keyboard.py`](il/02_teleop_keyboard.py) | Drives a gripper by keyboard (or a key script) and records the episodes you keep as an HDF5 file. |
| [`03_replay_evaluate.py`](il/03_replay_evaluate.py) | Replays recorded demonstrations and measures how often a scripted policy, clean and noisy, lifts the cube. |
| [`04_generate_demos.py`](il/04_generate_demos.py) | Generates scripted demonstrations in many worlds: each world runs the task's shipped demonstrator until enough successful episodes are saved (HDF5, or a LeRobotDataset with `--lerobot`). Scripts ship for 3 of the 8 ready-made envs; no trajectory augmentation. |

## sysid

| Example | What it shows |
|---|---|
| [`01_fit_synthetic.py`](sysid/01_fit_synthetic.py) | Fits a 2-joint arm to a synthetic recording made with hidden joint parameters: prints the report, a file / hidden / fitted table, and the tracking error with and without `add_robot(sysid=)`. |

## io

| Example | What it shows |
|---|---|
| [`export_scene.py`](io/export_scene.py) | Export an Alloy scene to MJCF, URDF and USD, then load the MJCF in MuJoCo and step it. |
| [`import_file.py`](io/import_file.py) | Bring your own asset: import an MJCF, URDF or USD file into Alloy cards, then assemble and step it in Alloy. |

## plugins

| Example | What it shows |
|---|---|
| [`hydraulic_demo.py`](plugins/hydraulic_demo.py) | A hydraulic cylinder lifts a 0.5 kg plate: pump on for 0.6 s, then off (the fluid holds the load). |
| [`ink_demo.py`](plugins/ink_demo.py) | A pen block slides across a sheet of paper and leaves an ink track; saves the ink map as a PGM image. |
| [`magnet_demo.py`](plugins/magnet_demo.py) | Two NdFeB-like magnet cubes on a slippery table attract and snap together. |
| [`suction_demo.py`](plugins/suction_demo.py) | Suction cup demo: a gantry cup picks a box, carries it 0.2 m and lets it go by venting (engine item 10). |

## soft

| Example | What it shows |
|---|---|
| [`soft_film_drop.py`](soft/soft_film_drop.py) | Drop a library film or cloth on the floor in one line: plastic film sheets (film_hdpe_sheet, film_ldpe_sheet, film_lldpe_sheet), an HDPE grocery bag (bag_hdpe_tshirt), paper sheets (paper_kraft_sheet, paper_wrap_sheet, paper_board_sheet), cotton terry towels (towel_cotton, towel_cotton_hand) and cloth sheets (cloth_cotton_sheet, cloth_polyester_sheet). |
| [`soft_fluid_cup.py`](soft/soft_fluid_cup.py) | A cup of water from the card library in one line: 200 ml as a cylinder inside the library cup tumbler_pp. |
| [`soft_fluid_fills.py`](soft/soft_fluid_fills.py) | A box of liquid or grains from the card library in one line, then let it settle and report where it is. |
| [`soft_food_cucumber.py`](soft/soft_food_cucumber.py) | A cucumber (soft solid, library flesh values) lies on the ground; prints its route, lowest point and health. |
| [`soft_food_dough.py`](soft/soft_food_dough.py) | A ball of wheat dough (elastoplastic paste) dropped onto the ground; it slumps a little and stays put. |
| [`soft_food_fruit.py`](soft/soft_food_fruit.py) | A tomato and a peach (soft solids, library whole-fruit values) rest side by side on the ground. |
| [`soft_food_gel_foam.py`](soft/soft_food_gel_foam.py) | A 10 % ballistic gelatin block and a flexible polyurethane foam pad, each a soft solid with library values. |
| [`soft_food_sponge.py`](soft/soft_food_sponge.py) | A wet cellulose kitchen sponge (114 x 64 x 25 mm), a soft solid: dropped on the table, it settles and rests. |
| [`soft_food_tissue_pad.py`](soft/soft_food_tissue_pad.py) | Suture practice pads: a soft silicone fat layer and a stiffer silicone skin layer, side by side. |

## tactile

| Example | What it shows |
|---|---|
| [`engine_gel.py`](tactile/engine_gel.py) | The physics side's gel sensor (engine penetration + marker fields) rendered as GelSight images. |
| [`gripper_touch.py`](tactile/gripper_touch.py) | A two-finger gripper squeezes a thin rod; a GelSight sensor on the left finger shows the touch, in every world. |

## topology

| Example | What it shows |
|---|---|
| [`cut_anywhere.py`](topology/cut_anywhere.py) | Cut a soft block along any blade path: its card opts in with `topology: {cut_model: copies}`, and a kinematic knife with a `blade:` edge slices it, with no pre-split seams. |

## viewer

| Example | What it shows |
|---|---|
| [`01_view_scene.py`](viewer/01_view_scene.py) | Watch a simulation in a viewer: a block and a ball drop onto a table, then rest. |
| [`02_pause_step_contacts.py`](viewer/02_pause_step_contacts.py) | Pause, step by program, pick worlds and read contacts: four worlds, a block dropped on a table in each. |
| [`03_joint_sliders.py`](viewer/03_joint_sliders.py) | Joint sliders (opt-in, `joints=True`): move the Franka Panda's joints from the viewer's panel; a slider sets that joint's target in every world. |
| [`04_drag.py`](viewer/04_drag.py) | Mouse dragging (opt-in, `drag=True`): grab a box in the viewer and pull it with a spring; let go and it falls. Tested on free boxes; dragging robot links is not tested. |
