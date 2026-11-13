# Alloy physics engine

Open source under Apache-2.0: https://github.com/argonrobotics-ai/alloy

![A cloth sheet draped over a stand, simulated in Alloy](https://alloy.argonrobotics.ai/docs/img/cloth.png)

[![arXiv](https://img.shields.io/badge/arXiv-2611.16312-b31b1b.svg)](https://arxiv.org/abs/2611.16312)

**Install:** `pip install alloy-engine`, inside a Python 3.11 or 3.12 virtual environment (Linux x86_64, or macOS
on Apple silicon), then the quick start below. On an older Python (a stock Ubuntu 22.04 has 3.10) pip answers "from
versions: none": `uv venv -p 3.12 --seed && source .venv/bin/activate` makes a 3.12 environment with pip in it.

Alloy is a physics engine for robot learning, built on [Newton](https://github.com/newton-physics/newton),
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and [Warp](https://github.com/NVIDIA/warp): rigid
bodies and robots step in MuJoCo Warp (MuJoCo C on the CPU), cloth and soft bodies in Newton's VBD, sand and pastes
in Newton's MPM, liquids in Alloy's own DFSPH. One scene can hold rigid robots (sub-millimetre contact for declared
precision fits such as pegs and threads), cloth and soft bodies, liquids, granular material, cables and objects that
cut or tear, stepped together in parallel worlds on one NVIDIA GPU: thousands for free rigid bodies, 512 to 4,096
for a robot arm (on one L4, a Franka Panda with a table and a cube ran best at 2,048), tens to hundreds with cloth or
liquid. Without a GPU everything runs on the CPU, slowly.

Keep in mind: the engine's known limits, each with its size and workaround, are on one page,
https://alloy.argonrobotics.ai/docs/keep-in-mind/ (installed as `public/keep-in-mind.md` in the docs folder, with a
page per limit in `known_gaps/`).

Objects are described once as **cards** (YAML: shape plus material values with their sources); a router picks the
solver for
each (MuJoCo Warp for rigid bodies, VBD for cloth and soft bodies, DFSPH for liquids, ...), and one `Simulation`
loop steps them together.

What it is for:

- A robot arm pouring liquid: build the cup into the robot's hand in its MJCF and the robot pours from it into a
  second container (`docs/guide/robots.md`, Grasping). For liquid pouring reinforcement learning, the ready-made
  `Pour-v0` env streams water from a spout; the agent moves a bowl under the stream and is rewarded for catching it.
- Cloth and towel folding for imitation learning: `ClothFold-v0` (experimental) is a task that folds a towel along its
  diagonal with a gripper; a scripted fold ships as `examples/envs/09_cloth_fold.py` (expect partial folds), and
  `alloy.il` records the env's episodes as demonstrations (HDF5, robomimic layout). The recording holds observations
  and actions, not the towel's vertices, and a recording with cloth cannot be replayed.
- An MJCF robot with sand and water: one scene holds a robot from an MJCF file, sand (MPM) and water (DFSPH), stepped
  together. The sand is a solid wall to the water, which does not soak in (`docs/known_gaps/sph-mpm-mixing.md`). It is
  slow on a CPU.
- Cable and wire manipulation: `scene.add_rod` makes a cable or wire that bends, twists and stretches as its material
  says, and a robot can grip and twist it; a copper wire keeps a bend past yield (`docs/guide/rods.md`).
- An MCP server physics simulator for robotics: `alloy mcp` gives an AI agent the engine as MCP tools (build a
  scene, step it, read its state and health, render it); see MCP server below.

> Status: early public release (0.x). APIs may change between minor versions.

## Install

Linux x86_64 (CPU or NVIDIA GPU) or macOS on Apple silicon (CPU). Linux aarch64 installs the same way but is not
tested yet. Newton and MuJoCo Warp come inside Alloy; Warp, MuJoCo and the rest come from PyPI.

Inside a Python 3.11 or 3.12 virtual environment (`python3 -m venv .venv && source .venv/bin/activate`; a stock
Ubuntu 24.04 refuses pip outside one (PEP 668, "externally-managed-environment"), and its `python3 -m venv` needs
`sudo apt install python3-venv` first):

```bash
pip install alloy-engine                       # the import name is alloy
python -c "import alloy; print(alloy.__version__)"
pip install "alloy-engine[import,mesh]"        # optional extras (below)
```

Extras, each added the same way: `import` (the object library below, mesh import, `collision="decompose"`), `mesh`
(soft bodies from meshes), `envs` (PyTorch, `huggingface_hub` and Gymnasium, for the Gym envs and torch camera
images; about 3 GB), `il` (HDF5 demonstration files), `lerobot` (LeRobot datasets; Python 3.12), `usd` (reading USD
files), `viewer` (the live viewer) and `barcode`. The docs, the runnable examples and AGENTS.md install with the
package: `python -m alloy.docs` prints where.

**NVIDIA GPU:** the pinned Warp (`warp-lang` 1.18.0) needs an R580-series or newer NVIDIA driver and a Turing
(`sm_75`: T4, RTX 20 series) or newer GPU (Warp's install notes). Stock cloud images often ship without a driver:
install one first (`nvidia-smi` then lists the GPU). Without a GPU it can use, Alloy runs on the CPU
(`alloy.build(scene)` picks `cuda:0` when there is one). On a fresh Ubuntu machine with an NVIDIA GPU:

```bash
sudo ubuntu-drivers install && sudo reboot     # the NVIDIA driver (or start from a CUDA image instead)
nvidia-smi                                     # after the reboot: lists the GPU
pip install "alloy-engine[envs]"               # in your virtual environment: Alloy plus PyTorch for the GPU
```

A C compiler lets CPU robot scenes use MuJoCo's C step (gcc on Linux; on a Mac, Apple's command line tools:
`xcode-select --install`); without one they run on a slower fallback, and Alloy says so.

The first run of a new kind of scene compiles kernels once: about 45 s on a CPU (41 s on an 8-vCPU Linux machine,
44 s on an Apple-silicon Mac) and 3 to 4 minutes on an NVIDIA GPU (190 to 240 s on L4s). The result is cached, so later
runs start in seconds.

AI agents: start with AGENTS.md, the whole quick start in one file: `python -m alloy.docs agents` prints it. Every
page is also on the docs site, https://alloy.argonrobotics.ai/docs/, and its index for language models is
https://alloy.argonrobotics.ai/llms.txt.

## Quick start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/argonrobotics-ai/alloy/blob/main/notebooks/quickstart.ipynb)
In a browser: the quick start notebook installs Alloy on Colab's free CPU runtime, drops a box, drapes a cloth and
shows one rendered frame, in a few minutes.

Drop a box onto a table and save its trajectory (`size` is full edge lengths in metres; under a minute on a CPU
once kernels are cached):

```python
import alloy

scene = alloy.Scene(dt=0.005)                       # control period: 5 ms per step
scene.add_ground()
scene.add_box("table", size=(0.8, 0.8, 0.70), pos=(0, 0, 0.35), static=True)   # top at z = 0.70 m
scene.add_box("box", size=(0.08, 0.08, 0.08), pos=(0, 0, 0.90), mass=0.5)     # kg
sim = alloy.build(scene)                            # GPU if there is one, else CPU
traj = sim.run(1.5, record=["box"])                 # traj["box.pos"]: (frames, worlds, 3)
traj.world(0).save("drop.npz")                      # t, box.pos, box.quat_wxyz, ...
z = float(traj.world(0)["box.pos"][-1, 2])
print(f"box rests at z = {z:.3f} m, health ok: {sim.check()['ok']}")   # 0.740: table top + half the box
assert abs(z - 0.74) < 0.005
```

More: the examples' `api/` scripts (a gripper lift, many parallel worlds, cloth, liquid) and AGENTS.md.

## Object library

The starter library (materials, contact pairs, 118 objects, demo scenes, robot models) installs with
Alloy. About 5,400 more object cards, with collision hulls and estimated mass and friction, are a public Hugging
Face dataset, `argonrobotics-ai/alloy-library`, kept outside the package: 1,111 YCB and Google Scanned Objects
scans; 1,989 Objaverse objects; 1,900 kitchen objects and fixtures made by Lightwheel (NVIDIA's
PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF dataset, CC BY 4.0, as packaged by RoboCasa, with our
collision hulls, estimated masses and materials; no RoboCasa kitchens or tasks); and 393 cards from Gym-Anything,
one per rigid part of its articulated objects. 263 of the kitchen cards carry a flag for a failed mass, size or
physics check. The cards record where each mass comes from: measured or from a catalogue for 64 of the YCB and
Google Scanned objects, an estimate from the object's category or mesh volume for more than 9 in 10 (the note says
ESTIMATE); friction is that of the library material the card names, not measured per object. Each one has its
card, meshes, convex hulls, textures and a thumbnail. Alloy downloads only the objects you ask for, into
`~/.cache/alloy/imports/<library>/`; no account or login is needed. Fetching needs the `import` extra (it brings
`huggingface_hub`): `pip install "alloy-engine[import]"`. With the extra, a build also fetches a Hub object you
name that is not on disk yet; without it, `add_object` raises `SceneError` for any object not on disk, its fix
naming the extra and `alloy fetch <name>`. To fetch ahead of time:

```bash
alloy fetch --list --library ycb_gso                 # what there is; nothing is downloaded
alloy fetch ycb_025_mug ycb_024_bowl                 # objects by name; files you already have are not downloaded again
alloy fetch --library robocasa --category blenders   # one category; --library alone fetches a whole library (GBs)
```

Then use them by card name: `scene.add_object("mug", "ycb_025_mug", pos=(0, 0, 0.1))`, or
`alloy compose "a mug and a bowl on a table"`. `ALLOY_CACHE` moves the download folder. Each object's card records
the licence its source declares (CC BY 4.0, MIT, CC0 1.0 or Apache-2.0) and, in the comment at its top, the credit
to give (the dataset's `ATTRIBUTION.md` collects them); check both before you share an object or an image of it, or
use it commercially.

## MCP server

`alloy mcp` serves the engine as Model Context Protocol (MCP) tools over stdio, with nothing beyond
`pip install alloy-engine`: `list`, `scene`, `build`, `step`, `run`, `state`, `targets`, `reset`, `drive`, `pose`,
`wrench`, `settle`, `heater`, `controls`, `close`, `check`, `render`, `estimate`, `import`, `export`, `vary`,
`compose`, `create`, `audit` and `fetch`, the operations of the `alloy` command line with the same JSON. To add it
to Claude Code:

```bash
claude mcp add alloy -- alloy mcp
```

`fetch`, `create` and `compose`'s objects from the Hub library need the `import` extra
(`pip install "alloy-engine[import]"`). The server code is
[`alloy/mcp/`](https://github.com/argonrobotics-ai/alloy/tree/main/alloy/mcp).

<!-- mcp-name: io.github.argonrobotics-ai/alloy -->

## What is in the box

What `pip install alloy-engine` puts in site-packages (`python -m alloy.docs` prints the docs, examples and
AGENTS.md paths):

| Path | What |
|---|---|
| `alloy/` | the engine package: cards, scene assembly, solvers, contact, coupling, precision, parallel worlds |
| `alloy/_data/library/` | the starter card library: materials (with sources), contact pairs, objects, scenes, robots |
| `alloy/_data/docs/` | the docs: `public/` (from `index.md`), `known_gaps/` (a page per limit), `guide/` (reference) |
| `alloy/_data/examples/` | runnable scripts (index: `README.md` there) |
| `alloy/_data/AGENTS.md` | the quick start for AI agents |

## Built on

Alloy's physics core is built on [Newton](https://github.com/newton-physics/newton) and
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) (both Apache-2.0), on
[Warp](https://github.com/NVIDIA/warp). Rigid bodies and robots step in MuJoCo Warp through Newton's `SolverMuJoCo`
(in MuJoCo C on the CPU), cloth and soft bodies in Newton's `SolverVBD`, sand and pastes in Newton's implicit MPM
solver, and liquids in Alloy's own DFSPH solver. Newton and MuJoCo Warp ship inside Alloy as `alloy/_vendor/newton`
(with our fixes) and `alloy/_vendor/mujoco_warp` (files unchanged), with their own names, licences and copyright
headers; `alloy/_vendor/VENDORED.md` (in the package) lists the upstream commit each starts from and every
change we made. By default Alloy replaces some of MuJoCo Warp's solver and collision steps at run time (listed in
the same file).

Installing Alloy does not install or replace a `newton` or `mujoco-warp` package of your own, but it pins
`mujoco==3.17.0` and `warp-lang==1.18.0` (from PyPI) and replaces other versions of those two in the
environment. A `newton` or `mujoco-warp` of your own can share the environment only if it accepts exactly those
versions; otherwise keep it in its own virtual environment.

## Citation

The paper on Alloy is on arXiv: https://arxiv.org/abs/2611.16312. To cite Alloy, cite from the arXiv page.

## Licence

Apache-2.0 (see `LICENSE`). Third-party code, robot models and data keep their own licences; see `NOTICE`.
Issues and questions: https://github.com/argonrobotics-ai/alloy/issues. Contributing:
https://github.com/argonrobotics-ai/alloy/blob/main/CONTRIBUTING.md. Security reports: privately, through
https://github.com/argonrobotics-ai/alloy/security (see SECURITY.md), not in public issues.
