# What is the Alloy physics engine?

The Alloy physics engine is a simulator for robot learning. One scene can hold robots, cloth, liquids, cables and
other materials, stepped together in many parallel worlds on one GPU. It is made by Argon Robotics and installs with
`pip install alloy-engine`. This page answers the questions people ask first.

## What is Alloy?

A physics engine for robot learning: robots, cloth, liquids, cables and more in one scene, with many worlds on one
GPU. Rigid bodies and robots step in MuJoCo Warp, cloth and soft bodies in Newton's VBD solver, and liquids in
Alloy's own solver.

## Who makes it?

Argon Robotics.

## How do I install it?

In a Python virtual environment, run `pip install alloy-engine`. The import name is `alloy`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install alloy-engine
```

A first scene, a block dropped on a table:

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.6), mass=0.2)
sim = alloy.build(scene)
sim.run(1.0)
print(sim.state(world=0)["block.pos"].round(3), sim.check()["ok"])
```

```text
[0.    0.    0.425] True
```

The block lands on the table and rests with its centre at 0.425 m. `True` means every world passed its health checks.

## Which Python versions and platforms?

Python 3.11 or 3.12, on Linux. It also runs on an Apple-silicon Mac, on the CPU: an independent cold test ran the
quick start, a cloth scene, a robot and the camera there. Windows is not tested.

## Do I need a GPU? How slow is the CPU?

No. Alloy uses an NVIDIA GPU when there is one (driver 580 or newer), and the CPU otherwise, with the same code.
On an AMD EPYC 7B13 CPU, liquid runs about 40,000 particle-steps/s, and with a C compiler installed a robot arm runs
about real time on one core. Cloth, liquid and soft bodies are much slower on a CPU and use one core. On an
Apple-silicon Mac with no NVIDIA GPU, the quick start ran in 44 s including the first compile, and a 9 x 9 vertex
towel ran about 20x slower than real time (an independent cold test). See
[a Mac without a GPU](how-to/mac-without-gpu.md).

## What is it built on?

Alloy's physics core is built on Newton and MuJoCo Warp (both Apache-2.0), on Warp. Rigid bodies and robots step in
MuJoCo Warp, cloth and soft bodies in Newton's VBD solver, sand and pastes in Newton's MPM solver, and liquids in
Alloy's own DFSPH solver.

## What is the licence?

Apache-2.0.

## How does a coding agent use it?

`python -m alloy.docs` prints where the installed docs, examples and `AGENTS.md` are. The page list is at
https://alloy.argonrobotics.ai/llms.txt. `alloy mcp` serves the same operations as MCP tools, for example
`claude mcp add alloy -- alloy mcp`. See [agents](agents.md) and
[letting a coding agent build a simulation](how-to/coding-agents.md).

## What does not work yet?

A fast small body can pass through a thin static part: a 2 cm steel ball passes through a 1 cm plate from 9.9 m/s.
No call attaches an object to a gripper: friction is the only grasp. A free strip of cloth bends about 35 % too soft
along a mesh axis. The full list is in [Keep in mind](keep-in-mind.md).

## Next

- [Quickstart](quickstart.md)
- [What ships](what-ships.md)
- [Errors and fixes](errors.md)
