# What ships

What you get with `pip install alloy-engine`, and what more you can download: ready-made environments, robots
and objects. Every name and count on this page was read from the package on 2026-11-11.

```python
import alloy
import alloy.envs
from alloy.api.catalog import objects
from alloy.cards.library import default_root

for name in sorted(alloy.envs.REGISTRY):
    print(name)
robots = sorted(p.name for p in (default_root() / "robots").glob("emb_e?_*")
                if not p.name.endswith(("_route", "_stiff")))
print(len(robots), "robots,", len(alloy.materials()), "materials,", len(objects()) >= 118)
```

```text
ClothFold-v0
DrawerOpen-v0
GripperLift-v0
GripperLiftCamera-v0
PandaReach-v0
PegInsert-v0
PickCube-v0
PickPlace-v0
Pour-v0
9 robots, 90 materials, True
```

## Ready-made environments

`alloy.envs.make("<name>", num_envs=64)` builds one ([Train with RL](guides/rl.md)). All nine give state
observations; one adds a camera image.

| Name | Task | Observations |
|---|---|---|
| `GripperLift-v0` | a two-finger gripper lifts a cube 8 cm and holds it | state |
| `GripperLiftCamera-v0` | the same task | state and a camera image |
| `PandaReach-v0` | a Franka Panda arm moves its hand to a random target | state |
| `PickCube-v0` | a Franka Panda arm picks a 4 cm cube and carries it to a point in the air | state |
| `PickPlace-v0` | a gantry gripper puts a scanned object in a bin | state |
| `PegInsert-v0` | a gripper inserts a 10 mm peg 2 cm into a 10.5 mm hole | state |
| `DrawerOpen-v0` | a gripper pulls a cabinet drawer 15 cm open | state |
| `Pour-v0` | move a bowl under a stream of water and catch it | state |
| `ClothFold-v0` | experimental: fold a towel along its diagonal; slow, needs a GPU | state |

Two more for legged robots: `from alloy.envs.loco import go2_velocity, go2_rough` (a Unitree Go2 following velocity
commands on flat ground, and on rough terrain).

## Robots

Nine robot files are in the package. Load one with
`scene.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")` ([Robots](guides/robots.md)).

| Folder | Robot | Kind | Licence of the source files |
|---|---|---|---|
| `emb_e1_fr3` | Franka FR3 with the Franka Hand | arm | Apache-2.0 |
| `emb_e2_panda` | Franka Panda with the Franka Hand | arm | Apache-2.0 |
| `emb_e3_ur10e_2f85` | Universal Robots UR10e with a Robotiq 2F-85 gripper | arm | BSD-3-Clause |
| `emb_e4_fr3_bimanual` | two Franka FR3 arms | two arms | Apache-2.0 |
| `emb_e5_fr3_duo` | Franka Mobile FR3 Duo: a mobile base, a spine, two FR3 arms | mobile, two arms | Apache-2.0 |
| `emb_e6_g1` | Unitree G1 with Dex3 hands | humanoid | BSD-3-Clause |
| `emb_e7_h1_2` | Unitree H1-2 with Inspire hands | humanoid | BSD-3-Clause |
| `emb_e8_go2` | Unitree Go2 | quadruped | BSD-3-Clause |
| `emb_e9_leap` | LEAP Hand (right) | hand | MIT |

Your own robot loads from an MJCF or URDF file the same way.

## Objects

**In the package: 118 object cards and 90 materials.** `objects()` lists the cards on your disk;
`scene.add_object("mug", "<card name>", pos=(0, 0, 0.4))` places one.

**On the Hugging Face Hub: 5,423 more object cards**, each with its meshes and textures. Download by name:

```bash
pip install "alloy-engine[import]"
alloy fetch --list                 # what there is, by library and category
alloy fetch ycb_025_mug            # one object, into ~/.cache/alloy/imports
```

| Library | Objects | What | Licence |
|---|---|---|---|
| YCB | 78 | scanned household objects | CC BY 4.0 |
| Google Scanned Objects | 1,033 | scanned household objects | CC BY 4.0 |
| Objaverse | 1,989 | everyday objects in 240 categories | CC BY 4.0 (1,965), CC0 1.0 (24) |
| Kitchen objects | 1,900 | kitchen objects and fixtures by Lightwheel, as packaged by RoboCasa | CC BY 4.0 |
| Gym-Anything parts | 393 | one card per rigid part of its articulated objects | MIT |
| Alloy's own | 30 | household, kitchen, lab and snap-fit parts | Apache-2.0 (25), CC0 1.0 (5) |

No account is needed. The kitchen library is objects only: no whole kitchen scenes ship.

## Keep in mind

- Most object masses are estimates, not measurements: 5,180 of the 5,423 Hub cards are marked as estimates. Weigh
  the object or read its card before a result depends on its mass.
- Friction comes from the material a card names, never from a measurement of that object.
- Each card records its source's licence and credit line; check both before commercial use.
- `ClothFold-v0` is experimental and slow. Training an arm to grasp gives results that vary between runs, and
  `PickPlace-v0` does not learn to place ([Train with RL](guides/rl.md)).

## Next

- [Train with RL](guides/rl.md): step and train one of the environments.
- [Robots](guides/robots.md): load a robot and drive its joints.
- [Keep in mind](keep-in-mind.md): what does not work yet.
