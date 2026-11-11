# Env camera images are bright and grainy

## What is wrong
- The ready-made envs' camera observations (`camera=True`: 64 x 64 pixels, 4 samples per pixel, no denoiser, the
  default studio light) come out bright and with visible sampling noise.

## Number
- On a table, cube and ball scene seen like the env camera: mean brightness 214 of 255, and 3.9 grey levels RMS noise
  against a 256-sample image (95 % of pixels within 5 levels).
- More samples cut the noise: 1.5 levels at 16 samples, 0.7 at 64. Each step costs about that factor in render time.

## Why
- The env camera favours images per second for training (4 samples, no denoiser) and uses the studio light of the
  quick-look renders.

## Workaround
- Build your own `alloy.camera.Camera(..., spp=16)` over the env's sim for cleaner images, and pass
  `render(..., exposure=-1)` (one stop darker) when the images look washed out.

## Planned fix
- Expose the env camera's samples, denoiser and exposure as env options (after the launch).
