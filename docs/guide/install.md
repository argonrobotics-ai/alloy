# Install and devices

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11 or 3.12
pip install alloy-engine                               # the import name is alloy
python -c "import alloy; print(alloy.__version__)"
```

On a fresh Ubuntu machine with an NVIDIA GPU (stock cloud images ship without a driver):

```bash
sudo apt install ubuntu-drivers-common         # stock cloud images lack the ubuntu-drivers tool
sudo ubuntu-drivers install && sudo reboot    # the NVIDIA driver; or start from a CUDA image instead
nvidia-smi                                    # lists the GPU; its "Driver Version" must be 580 or newer
pip install "alloy-engine[envs]"              # in your virtual environment; the envs extra brings PyTorch
```

If `ubuntu-drivers` picked a driver older than 580, `ubuntu-drivers list` shows the newer ones to install.

- Python 3.11 or 3.12 on Linux. The install pulls pinned versions of MuJoCo and Warp from PyPI; Newton and MuJoCo
  Warp come inside Alloy (`alloy/_vendor/`, `AGENTS.md` section 12), so a `newton` or `mujoco-warp` you installed
  yourself is left alone. The engine is tested on exactly those versions.
- **A stock Ubuntu 24.04** refuses pip outside a virtual environment (PEP 668: "externally-managed-environment"),
  and `python3 -m venv` fails there until `sudo apt install python3-venv`. A machine with neither Python 3.11 nor
  3.12 needs one of them first ([uv](https://docs.astral.sh/uv/) can fetch one: `uv venv --python 3.12`).
- **The card library** (materials, objects, scenes, robots) ships inside the wheel. If an install has none, an
  unknown-material error says where it looked; `ALLOY_LIBRARY=<folder>` points at a copy of the library.
- **GPU or CPU.** With an NVIDIA GPU and driver, `alloy.build(scene)` runs on `cuda:0`; without one it runs on the
  CPU, with the same API. Choose explicitly with `alloy.build(scene, device="cpu")` or `device="cuda:1"`. Asking for a
  GPU that is not there raises `alloy.BuildError` with the fix. The pinned Warp (`warp-lang` 1.18.0) needs an
  R580-series or newer NVIDIA driver and a Turing (`sm_75`) or newer GPU (Warp's install notes). Stock cloud GPU
  images ship without a driver: the fresh-Ubuntu block above installs one.
- **What the CPU runs.** Rigid bodies and robots step in MuJoCo C in float64 (warm start off, solver tolerance at most
  1e-9): with a C compiler installed, a robot arm runs about real time on one core. That step is a C extension that is
  not in the wheel: it is compiled on first use with the system compiler (`cc`, `gcc`, `clang` or `$CC`) and cached
  under `~/.cache/alloy/mjc/`. Without a compiler (a stock Ubuntu 24.04 has none: `sudo apt install gcc`; macOS:
  `xcode-select --install`) Alloy prints one warning and robot scenes run 1.2 to 1.9x slower on the numpy path, and
  that path was not part of the release validation runs, so install a compiler before you rely on a robot result. The
  measured rows: on a stock Ubuntu 24.04 cloud VM (8 vCPUs) a Panda arm ran 745 world-steps/s with the C step and 646
  without (1.15x) in one world, 2,941 and 1,577 (1.87x) in 16 worlds, and 704 and 563 (1.25x) pushing a box. A pile of
  30 boxes does not use the C step and ran the same (386 and 388). The worst pair on record is 3.9x, an arm and blocks
  with noslip and line contacts on a loaded host. With `gcc` installed the first build compiles the step in under a
  second. A scene with cloth, liquid, soft bodies, precision fits or kinematic objects runs on Warp's CPU backend
  instead, much slower; the `alloy.route` log says "CPU route refused (...)" with the reason. CPU and GPU runs give
  the same physics model, not the same bits: do not compare them step for step. Contact-rich outcomes such as a grasp
  can differ between the CPU (MuJoCo C) and GPU (MuJoCo Warp) routes. One measured case: with the same scene, joint
  plan and seeds, a Panda picking a pepper mill placed it in 8 of 8 worlds on a GPU and 0 of 8 on the CPU, where the
  mill tilted in the closed fingers and dropped; health read OK on both, and the cause is not found yet. Check a grasp
  on the device you train on. `ALLOY_CPU_ROUTE=warp` runs rigid bodies on Warp's CPU backend too (the GPU's code path,
  several times slower).
- **Warp's CPU backend uses one core**, whatever the machine has: it runs each kernel as a serial loop, and no
  setting adds threads. Measured on the CPU (an AMD EPYC 7B13 unless a line names another machine; the Xeon VM
  named below ran 1.5-1.7x slower):
  - liquid: about 40,000 particle-steps per second on an AMD EPYC 7B13 and 25,000-27,000 on a 2.8 GHz Xeon cloud VM
    (GCP n2-standard-16), linear in the particle count (one world at rest: 512 particles 10-15 s per simulated
    second, 13,800 5-7.5 min; 2 worlds of 507 particles took 10 min for a 6 s pour). A free body in or beside the
    liquid costs 10-21x more (a floating body 21x on the Xeon; a pour beside a free ball 10.5x and 12.3x on the two
    machines; a 12,000-particle floating cup: 77 min per simulated second, 4 min on one L4 GPU);
  - cloth (a 20 x 20 sheet over a ball, 2 worlds): 52 s per 0.3 s simulated (33 s with `self_contact=False`). At
    VBD's own 10 iterations (`Scene(cloth_accuracy="fast")`) it takes 13.6 s;
  - cloth beside a robot arm: about 10-45 s per 10 ms control step, 3.4-3.9x a trial at 10 iterations, as "fast" runs
    (2.5-12 s, and a 2.2 s fold took 30-65 min per world). The lightest robot body near the cloth sets how many force
    exchanges each step takes (0.05 kg pads 128, 1 kg pads 32), and wall time follows that count: `deformables.md`,
    Cost. On one L4 GPU, 64 worlds of a 12 s fold took 47 min at 10 iterations ("fast").

- **Many worlds on a CPU: split them over processes.** `alloy.build(scene, device="cpu", processes="auto")` starts
  one worker process per core (at most one per world; `processes=4` for four), each stepping its own block of
  worlds. The sim it returns takes the same calls; `sim.close()` or a `with` block stops the workers:
  `with alloy.build(scene, device="cpu", processes="auto") as sim: traj = sim.run(1.0, record=["ball"])`.
  - Speed-up, 64 worlds on 16 cores, world-steps/s with 1 and 16 processes: liquid in a box 70 to 1,013 (14x), a
    20 x 20 cloth 47.5 to 613 (13x), a paste (mayonnaise) 5.9 to 120 (20x), the shipped Panda arm scene 4,499 to
    29,456 (6.5x: each process pays a fixed cost per step, so rigid scenes gain most with many worlds per process).
    Blocks are fixed, so when worlds cost unequal amounts (a body floating in a liquid) the slowest block sets the
    pace. One L4 GPU ran 64 worlds of a 6 s pour in 3.4 min.
  - Same results: seeds and randomized values go by world number, so a world steps bit for bit as it does in one
    process, while every process keeps the same clock. An adaptive step (a liquid's CFL step, finer coupling steps)
    can choose another clock in one process than in another: the sim warns, and `sim.worker_physics_steps` gives each
    process's count.
  - `sim.plugins` and the engine objects stay in the workers: asking for them raises `FeatureUnavailableError`.
    Each `step` or `state` call is a round trip to every worker, so one long `run` is faster than many short ones.
- **Optional extras.** `import` (mesh tools: `collision="decompose"`, `alloy.create`, the Hub object library),
  `mesh` (soft bodies from meshes; `pytetwild`) and `usd` (USD files: `alloy.load("x.usd")`, `add_robot`,
  `alloy.io.import_file`). `pip install alloy-engine` installs none of them; add them with pip:
  `pip install "alloy-engine[import,mesh]"`.
  A missing extra makes the build raise `FeatureUnavailableError` with the command.
- **First run compiles kernels.** The first build and step of a new kind of scene compiles Warp kernels: 3 to 4
  minutes on a GPU for a rigid or robot scene (190 to 240 s measured on fresh L4 machines; the quick start took
  190 s), about 5 min for a rod beside a robot and 10 min for cloth beside a free body, which load more solvers;
  longer on a CPU. The `ClothFold-v0` environment is the slowest start measured: `make` took 129 s, `reset` 39 s, and
  its first `step` then compiled for 9 to 10 min (545 and 583 s on two fresh L4s). They are cached in `~/.cache/warp`;
  later runs of the same kind of scene start in seconds. Nothing shortens the first compile, so plan for it: run one
  short build and step as a warm-up job before a long run, do not time or give up on the first run, and time a second
  run.
- **Logs.** The API silences Warp's per-module load lines. Set `ALLOY_VERBOSE=1` to see them. Alloy logs through
  Python `logging` under the `alloy` logger: `logging.basicConfig(level=logging.INFO)` shows build times.
  Warnings the API wants you to see (a slow soft-body route, a container that will not hold liquid) print to stderr
  as `alloy: ...` when you have not configured logging.

Check the install end to end (one box, half a second of simulated time):

```python
import alloy

scene = alloy.Scene()
scene.add_ground()
scene.add_box("box", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.3), mass=1.0)
sim = alloy.build(scene)
sim.run(0.5)
print(sim.device, sim.state(world=0)["box.pos"], sim.check()["ok"])   # box at rest near z = 0.05
```
