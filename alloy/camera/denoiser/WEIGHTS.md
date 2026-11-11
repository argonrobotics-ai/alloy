# Learned denoiser weights

The weights of Alloy's learned denoiser (`alloy.camera.denoiser`): a small kernel-predicting U-Net that turns a
path-traced frame of a few samples per pixel, with its albedo and normal images, into a clean image.

## Where they are

- Listed in `weights.json` in this folder: file name, sha256, size, training size, spp range and notes per entry.
- Published in the Alloy library on the Hugging Face Hub: dataset `argonrobotics-ai/alloy-library`, folder
  `denoiser/` (public, no account needed).
- Downloaded on first use over https, checked against the sha256 in `weights.json`, and cached in
  `~/.cache/alloy/denoiser` (`$ALLOY_CACHE/denoiser` when `ALLOY_CACHE` is set).
- `python -m alloy.camera.denoiser.weights [NAME ... | all]` fetches them ahead of time and prints the files.
- `ALLOY_DENOISER_URL=<folder or URL>` reads them from your own copy (https, http, `file://`, a directory, or
  `gs://`); `ALLOY_DENOISER_URL=""` or `HF_HUB_OFFLINE=1` turns downloads off. A cached file needs no network.

## Licence

Apache-2.0, the same as Alloy. The weights are our own work: every file was trained from random initial values, or
continued from an earlier network of ours, on images we rendered. They hold no third-party weights.

## Training data

Teacher images are Blender Cycles renders of Alloy scenes (inputs at 1-64 samples per pixel, targets at 2,048-4,096
samples per pixel cleaned with Intel Open Image Denoise). What the renders show:

| What | Source | Licence |
|---|---|---|
| Scene objects (connectors, rack parts, bottles, jars, mugs, trays, benches) | Alloy's own generated geometry | ours |
| YAM arm and gripper (bench scenes, the T12 scene) | I2RT Robotics, github.com/i2rt-robotics/i2rt | MIT |
| UR5e arm and Robotiq 2F-85 gripper (one scene) | MuJoCo Menagerie (ROS-Industrial models) | BSD-3-Clause |
| Environment maps (HDRIs) and surface textures | Poly Haven, ambientCG | CC0 |

No scanned-object dataset, no asset whose licence bars commercial use and no maker's CAD is in the training
images. The one texture that is not from Poly Haven or ambientCG (a grey floor tile in the rack scene) is our own
generated image.

## Tools used in training

- Blender Cycles (GPL; its license does not cover rendered images) and Intel Open Image Denoise (Apache-2.0)
  made the teacher images.
- The loss includes LPIPS (Zhang et al. 2018, BSD-2-Clause; AlexNet backbone from torchvision). It is used as a
  training loss only: no LPIPS or AlexNet weights are in these files.
- PyTorch (BSD-3-Clause) trained the networks.

## How good they are

The check ("passes"): on a frame, FLIP and LPIPS against a 4,096 spp reference are within 2x of what Cycles at
128 spp with Open Image Denoise (OIDN) gets. Every frame below was held out of training; inputs are Cycles renders.

New scenes, no tuning (four scenes of library objects on a table, three cameras each; views passing out of 12):

| Denoiser | 128 px: 16 / 32 / 64 / 128 spp | 640x480: 16 / 32 / 64 / 128 spp |
|---|---|---|
| `default` (`v3`) | 1 / 9 / 12 / 12 | 1 / 5 / 6 / 9 |
| OIDN | 2 / 9 / 11 / 12 | 0 / 2 / 12 / 12 |

So on a new scene the default matches OIDN at 128 px, and at 640x480 it is ahead at 16-32 spp and behind from
64 spp. Nothing checks the quality on your scene: compare a few frames with a high-spp render, or fine-tune
(`alloy.camera.denoiser.finetune`, minutes on one GPU).

Test scenes (fewest spp at which every frame passes, at 128 px / 640x480 / 1280x720). Scene weights were trained
on other frames of the same scene:

| Scene | `default` (`v3`) | Scene weights | OIDN |
|---|---|---|---|
| C2 water poured into a glass | 32 / 64 / 64 | 32 / 32 / 32 (`c2_640`, `c2_1280`) | 128 / 64 / 64 |
| C3 connector latch | 32 / 32 / 64 | 8 / 16 / 16 (`c3_128`, `c3_640`, `c3_1280`) | 64 / 32 / 64 |
| C4 clear plastic lid | 16 / 32 / 128 | 8 / 16 / 16 (`c4_128`, `c4_640`, `c4_1280`) | 128 / 64 / 64 |
| T12 two arms and blocks (128 px) | fails | 40 (`t12_128`) | 64 |

The C3 and C4 numbers at 128 px and 640x480 come from inputs with white-noise sampling (a reused Cycles session).
On blue-noise inputs, rendered at 1 / 4 / 16 / 64 / 128 spp only, `c3_128` and `c4_128` pass from 16 spp.

## Files

`default` is `v3` at every size. Scene-tuned files (`c2_*`, `c3_*`, `c4_*`, `dc5_*`, `t12_128`) are for the test
scenes named in their notes in `weights.json`; `Camera(denoise="auto")` uses `default`.
