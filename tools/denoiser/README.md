# Denoiser teacher data and training

Scripts that build the teacher set for `alloy.camera.denoiser` from Blender Cycles renders of alloy scenes, and
train or fine-tune the network on it. The inference API itself is `alloy/camera/denoiser` (torch only).

| Step | Script | Runs in |
|---|---|---|
| 1. Look variants (lights, HDRIs, glass / liquid / metal swaps, camera orbits) | `looks.py` | Python 3 |
| 2. Job list (bundle x look x camera x frames, hold-outs) | `jobs.py PLAN.json --out WORK` | Python 3 |
| 3. Render, pack, upload, in parallel and resumable | `run_jobs.py WORK/jobs.jsonl --work R --out P` | Python 3 + Blender 4.5 |
| 3a. One job's renders (noisy inputs at several spp and seeds + a high-spp target) | `blender/render_bundle.py` | Blender 4.5 |
| 3b. EXR folders -> float16 arrays | `pack.py JOB_DIR OUT_DIR` | Python 3 + `OpenEXR` |

Conventions (the same as Cycles' denoising passes, which the in-engine camera also follows):
- input `rgb`: noisy linear radiance (`Noisy Image`); `albedo`: `Denoising Albedo`; `normal`: `Denoising Normal`
  in camera space (x right, y up, z forward; a face towards the camera has z = -1);
- target: the same frame at 4,096 spp (2,048 above 256x256) + OIDN, another seed;
- the loss is computed in the scene's own display space through a LUT baked from Blender (`lut.npy`).

Example, one small job end to end:

```
python tools/denoiser/jobs.py plan.json --out work        # plan.json: see the jobs.py docstring
python tools/denoiser/run_jobs.py work/jobs.jsonl --work work/render --out work/packed --parallel 2 \
    --blender /path/to/blender
```
