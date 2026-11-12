# Render quality gates

Does the fast camera render what Blender Cycles renders, at the speed it claims? These gates answer pass or fail
for every scene, image size and GPU, measured against Cycles' own noise.

## The matrix

| | |
|---|---|
| Scenes | C2 water + glass, C3 opaque latch, C4 clear lid, a 1,024-world grid, T12 diffuse benchmark (`matrix.json`) |
| Sizes | 128x128, 640x480, 1280x720 |
| GPUs | NVIDIA L4, NVIDIA RTX PRO 6000 |

A cell (scene x size x GPU) **passes** when all of these pass on every frame of the scene:

| Check | Engine dir | Rule |
|---|---|---|
| parity | `alloy` (raw tracer) | best converged image (highest spp; lower-LPIPS denoise state) vs the Cycles 4,096-spp reference: FLIP mean and LPIPS each <= 2 x the gap of Cycles seed 1 at 128 spp + denoise |
| equal-spp | `alloy` | per (spp, denoise) row: ratio = engine gap / worst control-seed gap (FLIP and LPIPS); a row passes at <= 1.1. Verdict: row pass rate >= the worst control seed's rate (each seed judged against the others), and no row above 1.25 |
| G1 end to end | `alloy_e2e` | the image at the setting claimed for that scene and size is within the parity limit. Claims: `scenes.<scene>.claims.<res>` in `matrix.json`, else the size default `claims.<res>`; each is the fastest setting that passes every gate (C3: 5 spp + learned at 128 px, 12 spp + learned at 640x480, 16 spp + learned C3 weights at 1280x720; T12: 40 spp + learned T12 weights at 128 px, 64 spp + OIDN at 640x480 and 1280x720; C2: 32 spp + learned at every size, C2 weights at 640x480 and 1280x720; C4: 8 spp + learned C4 weights at 128 px, 32 spp + learned C4 weights at 640x480 and 1280x720) |
| speed | rows | at the claimed setting only: 1,000 images/s at 128x128 over 1,024 worlds on one RTX PRO 6000 alone (C3, T12); other sizes and GPUs are reported. No speed claim without a passing quality gate |

It **fails** when any check fails and is **pending** when images, references or speed rows are missing.
Every score is on raw 8-bit sRGB made from linear EXR with no dither; histogram-matched numbers are reported only.

## Layout

```
ROOT/<scene>/<frame>/<engine>/<W>x<H>_<spp>spp_<on|off>.png
```

- `cycles`: reference (seed 0); `cycles/<res>_4096spp_on.png` is the reference image.
- `cycles_s1` ... `cycles_s4`: control seeds at 1, 4, 16, 64, 128 spp, denoise on and off (T12, a single frame, has
  `cycles_s1` ... `cycles_s8`: with 10 rows per size, 4 seeds missed Cycles' own tail; seeds 0 and 7 of Cycles exceed
  seeds 1-4 by 17 % in FLIP on its 1280x720 4-spp denoised row).
- `alloy`, `alloy_e2e`, or any other folder: engines under test.

## Commands (JSON on stdout)

```bash
python -m bench.render_gates spec                                   # scenes, frames, sizes, claims, speed bars
python -m bench.render_gates fetch --dest gate_root --scenes c3 --sizes 128x128   # needs ALLOY_RENDER_GATES_URI
python -m bench.render_gates score gate_root --gpu l4 --out report_l4.json
python -m bench.render_gates matrix --report l4=report_l4.json --report rtx_pro_6000=report_rtx.json \
    --rows speed.jsonl --out matrix.json --md MATRIX.md
python -m bench.render_gates nightly --work gate_work --gpu l4 \
    --render "python -m alloy.camera.gate_render --scene {scene} --res {res} --frames {frames} --pack {pack} --out {root}"
```

Speed rows (JSONL) carry `gpu`, `scene`, `res`, `worlds`, `shared_gpu`, `setting` ("claim" for the claimed
setting) and `images_per_s`.

From Python:

```python
from pathlib import Path
from bench.render_gates import matrix, score
from bench.render_gates.metrics import Scorer

report = score.score_root(Path("gate_root"), Scorer(Path(".cache/gates")), gpu="l4")
m = matrix.build_matrix(matrix.load_spec(), {"l4": report}, rows=[])
print(matrix.to_markdown(m, matrix.load_spec()))
```

## One converter for every image

Every scored PNG, Cycles' and the engine's, is made by `bench.render_gates.view.to_png(linear, view)` from linear
RGB, with the reference folder's `view.json` (view transform, look, exposure, gamma, white balance; dither 0):

```python
import json
from bench.render_gates.view import to_png
png = to_png(linear_hwc_float32, json.load(open("gate_root/c3/420/cycles/view.json")))   # (H, W, 3) uint8
```

AgX and its looks are read from an installed Blender's OpenColorIO config at run time (`/opt/blender*` or
`$BLENDER_OCIO`) and baked once into a cached LUT; nothing from Blender ships here. White balance follows the
published methods (Robertson's isotemperature lines, Bradford adaptation). `python -m bench.render_gates topng ROOT`
rebuilds a reference tree's PNGs from its EXRs.

## Scene packs

The scene each engine renders: one pack per scene with every gate frame (meshes, materials, textures, lights, camera,
view). `fetch --kind pack --dest PACKS` unpacks them to `PACKS/<scene>/`; the nightly passes that folder to the render
command as `{pack}`.

## References

Cycles 4.5, OptiX, motion blur off, adaptive sampling off, GPU OIDN (accurate, albedo + normal), seed 0 at 4,096
spp; control seeds 1-4 at 1/4/16/64/128 spp, denoise on and off. Each PNG is rebuilt from the linear EXR through the
scene's view (AgX look and white balance; T12: Standard) with dither 0. Tars per scene, size and kind (PNG, EXR) are
hashed in `refs_manifest.json`; `fetch` verifies the sha256 before unpacking.

## Dependencies

Scoring needs `flip-evaluator` (BSD-3-Clause), `lpips` (BSD-2-Clause) with `torch` and `torchvision`, and
preferably `Pillow`. Views other than Standard need `opencolorio` (BSD-3-Clause) and an installed Blender 4.5 (its
config is read, not copied); EXR input needs `OpenEXR` (BSD-3-Clause). The gate logic, layout, matrix and tests need
only numpy.

## Tests

`pytest tests/render_gates` (CPU, seconds; synthetic images and a stand-in metric).
