# Licences of the camera's parts and dependencies

Alloy's camera code is Apache-2.0. Parts ported from Blender Cycles (Apache-2.0) are listed in `THIRD_PARTY.md`.
This note covers what the camera uses but does not ship.

## Default backend: Warp's BVH (no NVIDIA SDK code)

The default `RenderSettings(backend="warp")` traces rays with NVIDIA Warp's own BVH (`wp.Mesh`, `wp.Bvh`). Warp is
Apache-2.0 and installs from PyPI. No NVIDIA SDK headers, OptiX code or other NVIDIA-licensed source is used.

## Optional backend: OptiX (RT cores), built on the user's machine

- **Our code** (`alloy/camera/optix/csrc`, the ctypes glue) is Apache-2.0. It calls the OptiX API.
- **The OptiX headers** come from github.com/NVIDIA/optix-dev. They are under NVIDIA's "SDK, Samples and Tools
  License Agreement (with distribution rights)", not an open-source licence. That licence:
  - does not let us redistribute the headers in source form, or let them become subject to an open-source licence
    (sections 1.1 and 2.1);
  - lets binaries built with them be distributed only under agreements binding the recipients (section 1.1(ii)), and
    asks for notice to NVIDIA for commercial use (section 1.5);
  - limits use to NVIDIA GPUs.
- **So Alloy never ships the headers or a binary built from them.** `python -m alloy.camera.optix build
  --accept-nvidia-optix-license` fetches the headers into the user's cache only after the user accepts NVIDIA's
  licence, and compiles the helper library locally (g++ and nvcc). The OptiX runtime (`libnvoptix.so.1`) is part of
  the NVIDIA driver.
- Without the build, the camera uses the Warp backend; results are the same images, about 6x slower on complex
  scenes.

## Denoisers

- **Learned denoiser** (`alloy/camera/denoiser`, OSS-6): our own model and weights; see that package.
- **Intel Open Image Denoise** (fallback): Apache-2.0. Not bundled. `python -m alloy.camera.oidn fetch` downloads the
  official release (with its CUDA module) into the user's cache, or `OIDN_LIB_DIR` points at an installed copy.

## View transform

Scene packs show through Blender's "Standard" view (the sRGB curve), implemented from the sRGB standard. Full-material
scenes built from a model (a look, scene variants) show through AgX (`agx.py`), ported from Filament's AgX tone
mapper (Apache-2.0, github.com/google/filament, `filament/src/ToneMapper.cpp`: its inset and outset matrices, log2
range and sigmoid fit). Blender's own AgX data comes from its colour-management configuration, whose licence is not
stated clearly enough to copy; Blender's AgX looks are approximated by a contrast around middle grey, so images are
close to Blender's AgX, not identical.

## Assets

Look files may name Poly Haven and ambientCG HDRIs and texture sets. Both are CC0 and are downloaded on first use
into the user's cache (`alloy.render.blender.assets.AssetCache`); none are stored in the repository.
