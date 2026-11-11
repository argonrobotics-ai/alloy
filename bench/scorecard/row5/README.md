# Row 5: camera throughput

How many images per second an engine produces with physics running: every loop is one control step of the engine's
own setting for the task, then one render of a camera in every world, the images left on the GPU. Two scenes
(`cartpole`, `franka_cube`), three outputs (`rgb`, `rgb+depth`, `rgb+depth+seg`), three sizes (64, 128, 256 px), three
world counts (256, 1,024, 4,096): 54 cells per engine and renderer setting. The protocol is in `PROTOCOL.md`
(cells, timing, memory, row format). Each cell runs in its own process, alone on the GPU, and appends a JSON row.
The rows measured on 2026-11-10 are in `rows/` and the table built from them in `TABLE.md`
(`python3 bench/scorecard/row5/make_table.py --raw bench/scorecard/row5/rows --out bench/scorecard/row5/TABLE.md`).

Engines in the public table: Alloy (path tracer, at 1 spp and at 16 spp), MuJoCo Warp's renderer (ray caster, the copy
vendored in Alloy), Madrona-MJX (ray tracer and rasterizer).

## Reproduce (one command per engine)

Alloy and MuJoCo Warp, from a clone with a GPU and the venv of `install.sh` (PyTorch is installed with a GPU):

    python bench/scorecard/row5/run_cells.py --engine alloy --raw raw          # rows: raw/alloy_g4.jsonl
    python bench/scorecard/row5/run_cells.py --engine mujoco_warp --raw raw    # rows: raw/mujoco_warp_g4.jsonl

Both resume (a cell with a row is skipped), write PNG grids to `raw/png/` and per-cell logs to `raw/logs/`. A cell
that dies or passes 600 s gets a row with status `error` or `timeout`. `--optix` adds Alloy's OptiX cells after
`python -m alloy.camera.optix build`; `--commit <sha>` (or `ALLOY_COMMIT`) is recorded in each row; `--smoke` runs one tiny
cell per kind. One cell by hand: `python bench/scorecard/row5/alloy_cam.py --scene franka_cube --size 128 --worlds 1024
--setting fast --outputs rgb --out rows.jsonl`.

Madrona-MJX builds its own environment (own Python, own pins). On a fresh Ubuntu 22.04
GPU machine:

    bash bench/scorecard/row5/madrona/setup.sh && bash bench/scorecard/row5/madrona/run_all.sh

Madrona writes `~/work/out/madrona_g4.jsonl`; copy it to `raw/` beside the other rows. Then:

    python3 bench/scorecard/row5/make_table.py --raw raw --out TABLE.md [--extra "Label=rows.jsonl"]

A series appears only when its rows file exists; a cell without a row shows "not measured".

## Pinned versions and the machine

- GPU: one RTX PRO 6000 Blackwell (GCP spot `g4-standard-48`), NVIDIA driver 580, Ubuntu 22.04 (a cloud deep-learning
  image with CUDA 12.9 works). A spot VM can be preempted: the run commands resume.
- Alloy and MuJoCo Warp: the clone's own pins (`warp-lang` 1.18.0, `mujoco` 3.16.0); MuJoCo Warp is `alloy._vendor.mujoco_warp`
  (upstream 86e2fae0, no patches). The row records `alloy_commit`.
- Madrona-MJX: madrona_mjx 1505699168fd (madrona submodule b46e6ab7), mujoco_playground f2159f3, jax 0.5.3, mujoco and
  mujoco-mjx 3.4.0, brax 0.12.5, Python 3.11 (`madrona/setup.sh`).

## Cloud recipe and wall time

Create a spot `g4-standard-48` (zone with stock; US is fine), run one engine per VM or one after another, copy `raw/`
home, delete the VM. Approximate wall time on that GPU: Alloy about 4 h (including the 16 spp cells), MuJoCo Warp about
25 min, Madrona about 20 min plus about 40 min to build. Never time a first run on a machine:
kernel compilation and graph capture happen there (the cell scripts warm up first). Never run two engines at once on
one GPU.

## Row format and caveats

Row fields are listed in `PROTOCOL.md`; rows carry extra fields too (spread of repeats, loop counts, build time,
`sim_check_ok` for Alloy, `sanity` for the others).

- Each engine uses its own control step: Alloy and MuJoCo Warp 10 ms (8 and 2 to 5 physics substeps), Madrona 40 ms
  (cartpole) and 50 ms (Franka). `control_dt_s` and `physics_substeps` are in every row.
- Alloy starts an MJCF robot at q=0 and ignores the keyframe; the harness ramps the servos to the hold pose before timing.
  MuJoCo Warp starts from the keyframe.
- MuJoCo Warp draws the Panda's collision hulls (geom group 3 is enabled); Alloy draws the Panda's visual meshes.
  The cartpole has a floor plane in MuJoCo Warp's MJCF and Alloy's `add_ground()`.
- Alloy `physics_only_steps_per_s` counts world-steps per second.
- Timing must stop after the render really finished: `torch.cuda.synchronize()` alone is not enough for every
  renderer: where it is not, the harness reads a value that depends on every image back to the host before stopping.
- Madrona-MJX has no segmentation output: its `rgb+depth+seg` rows are `unsupported`. Its depth comes from the same
  render call as rgb. Its env's own grayscale / brightness post-processing is removed.
- Madrona-MJX `franka_cube` did not run on this GPU with our build (the table says "our build failed"; not a measurement of Madrona). After the Franka render context initialises, every JAX kernel not
  yet loaded fails with `XlaRuntimeError: INTERNAL: error before calling cuModuleGetFunction (1): cudaErrorInvalidValue`
  when worlds >= 2 (1 world works), with both the raytracer and the rasterizer. Ruled out: geom groups, the Madrona device
  heap size, JAX preallocation, XLA graph level 0, command buffers off, eager CUDA module loading, GPU memory. Those rows
  are kept as `error` with the message, never as measured.
- Madrona needs the build option `-DLOAD_VULKAN=ON` (with it off the rasterizer segfaults) and two fixes for Blackwell and
  CUDA 12.9: `cub::ThreadReduce` in `sort_archetype.cpp` (done by `setup.sh`), and `MADRONA_MWGPU_KERNEL_CACHE` must be
  unset (it names a cache file; a directory segfaults). Some cloud images ship the NVIDIA driver without its EGL/Vulkan
  libraries: `maniskill/nvidia_egl.sh` installs them (`madrona/setup.sh` calls it).
- Alloy's `t12` rows (128 px) come from `alloy.camera.bench --claim t12`, 10 timed iterations, quality full: an extra
  row, never part of the comparison.
- Memory is the process's GPU memory from `nvidia-smi`, sampled at 2 to 5 Hz through the timed part.
  JAX preallocation is off for Madrona (`XLA_PYTHON_CLIENT_PREALLOCATE=false`), otherwise it would take 75 % of the GPU.
