# Env camera images are bright and grainy

## What is wrong
- The ready-made envs' camera observations (`camera=True`: 64 x 64 pixels, 4 samples per pixel, no denoiser, the
  default studio light) come out with visible sampling noise, and bright on a light scene.

## Number
- Each env's camera image of world 0 at reset (CPU), against a 256-sample image of the same pose: mean brightness of
  255, RMS noise in grey levels at 4 / 16 / 64 samples, and the pixels within 5 levels at 4 samples:
  - `GripperLiftCamera-v0`: brightness 212; noise 4.2 / 1.7 / 0.9; 85 % within 5 levels.
  - `PickPlace-v0` (varied looks, no texture images on disk): 124; 4.8 / 1.9 / 0.9; 85 %.
  - `DrawerOpen-v0` (a cabinet drawn with its texture image): 176; 16.8 / 6.4 / 2.9; 45 %. Fine texture detail is
    what 4 samples per pixel average least well.
- More samples cut the noise as above; each step costs about that factor in render time.

## Why
- The env camera favours images per second for training (4 samples, no denoiser) and uses the studio light of the
  quick-look renders.

## Workaround
- Build your own `alloy.camera.Camera(..., spp=16)` over the env's sim for cleaner images, and pass
  `render(..., exposure=-1)` (one stop darker) when the images look washed out.

## Planned fix
- Expose the env camera's samples, denoiser and exposure as env options (after the launch).
