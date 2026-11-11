# Row 5: camera throughput, protocol

One GPU class for every engine: RTX PRO 6000 Blackwell (GCP g4-standard-48, spot). Each engine in its own process
per cell, alone on the GPU.

## Cells
- Scenes (same task in every engine, the engine's own shipped version of it; say where the files differ):
  `cartpole` (cart + pole, one camera) and `franka_cube` (Franka Panda, table, one cube, one fixed camera ~1 m away
  looking at the cube). Alloy also reports `t12` (its gated-quality scene) as an extra row, never as the comparison.
- Outputs: `rgb`, `rgb+depth`, `rgb+depth+seg`. An engine that cannot produce one writes a row with
  `"status": "unsupported"` and the reason.
- Size: 64x64, 128x128, 256x256. Worlds: 256, 1024, 4096. One camera per world.
- Physics ON: every loop = one control step of the engine's own recommended setting for that task (say dt and
  substeps) with small random or hold-position actions, then one render of every world, images left on the GPU as
  a tensor (no copy to host), device sync before the clock stops. No observations other than the images, no reset.
- Timing: warm up (compile, graph capture, >= 20 loops), then >= 3 repeats of >= 50 loops; report the median and
  min / max of repeats. `images_per_s = worlds * loops / wall`. Also `render_only_images_per_s` when the engine
  lets render be timed alone, and `physics_only_steps_per_s`. A slow cell may use fewer loops; the row says so
  (`reduced_loops`, `loops_per_repeat`, `warmup_loops`).
- Clock stop: the timer stops only after the render really finished. `torch.cuda.synchronize()` alone is not enough
  for every renderer; where it is not, read a value that depends on every image back to the host before stopping.
- GPU memory: peak `memory.used` of the process from `nvidia-smi --query-compute-apps` (or whole-GPU used minus the
  idle baseline) sampled at >= 2 Hz through the timed part, in MiB. Out of memory = a row with `"status": "oom"`.
- Renderer settings: each engine at its own recommended fast setting for training images (say it: rasterizer /
  ray caster / path tracer, samples, denoiser, shadows, textures). Alloy reports two settings: `fast` 1 spp (like
  for like with ray casters) and `default` 16 spp (the camera's default), labelled.
- Sanity: save one PNG grid (4 worlds) per scene / engine / output so a reader sees the image is real, non-black
  and the worlds differ after physics. A cell whose image is blank or frozen is `"status": "invalid"`.

## Row format (JSON lines, one per cell)
{"engine", "engine_version", "scene", "outputs", "width", "height", "worlds", "status",
 "images_per_s", "images_per_s_min", "images_per_s_max", "render_only_images_per_s", "physics_only_steps_per_s",
 "gpu_mem_mib", "renderer", "render_settings", "control_dt_s", "physics_substeps", "gpu", "driver",
 "commit_or_version_pins", "command", "utc"}

Statuses: `ok`, `invalid` (blank / frozen / identical worlds, or the engine's own health check failed), `oom`,
`timeout`, `error` / `failed`, `unsupported`.

## Files
`raw/<engine>_g4.jsonl` + `raw/logs/` + `raw/png/`; harness scripts in this folder (`setup.sh` pins every version;
one run command per engine, see README.md).

## Rules
Run from a `git archive` snapshot of a fixed commit, never a live tree. Delete the cloud VM when done or on
failure, after the rows are copied home. Retry a VM only on infrastructure failures, never a cell that failed.
