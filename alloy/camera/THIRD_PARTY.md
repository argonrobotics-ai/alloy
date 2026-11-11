# Third-party code in `alloy/camera`

The path tracer follows Blender Cycles 4.5 so that images match Cycles renders. Several kernels are ports of Cycles
source code; Cycles is licensed under the Apache License 2.0.

| Files | Ported from (Cycles 4.5, `intern/cycles/`) | Licence |
|---|---|---|
| `tracer/sampling.py` | `kernel/sample/pattern.h`, `sobol_burley.h`, `util.h`, `mapping.h` | Apache-2.0, Copyright 2011-2024 Blender Foundation |
| `tracer/tables.py` | `scene/film.cpp` (filter table), `util/math_cdf.cpp`, `util/hash.h`, `kernel/tables.h` | Apache-2.0, Copyright 2011-2024 Blender Foundation |
| `tracer/sky.py` | `kernel/light/distant.h`, `background.h`, `tree.h` | Apache-2.0, Copyright 2011-2024 Blender Foundation |
| `tracer/integrator.py` | `kernel/integrator/*` (path loop, MIS, Russian roulette, clamping), `kernel/closure/bsdf_diffuse.h` | Apache-2.0, Copyright 2011-2024 Blender Foundation |
| `pbr/bsdf/*` (Principled BSDF, multiscatter GGX, F82, coat, sheen; `cycles_tables.npz`) | `kernel/closure/bsdf_microfacet.h`, `bsdf_principled_*`, `bsdf_sheen.h`, `kernel/svm/closure.h`, `scene/shader.tables` | Apache-2.0, Copyright 2011-2024 Blender Foundation |
| `pbr/nodes/*` (Noise, Math, Vector Math, Bump, Bevel, image textures) | `kernel/svm/noise.h`, `fractal_noise.h`, `math_util.h`, `bump.h`, `kernel/svm/ao.h` / `bevel.h`, `image.h` | Apache-2.0, Copyright 2011-2024 Blender Foundation |
| `pbr/lights/*` (area lights, world importance map, distant lights, light tree) | `kernel/light/*`, `scene/light.cpp`, `scene/light_tree.cpp` | Apache-2.0, Copyright 2011-2024 Blender Foundation |
| `pbr/integrator.py`, `pbr/path.py` | `kernel/integrator/*`, `kernel/camera/camera.h` (thin lens, ray differentials) | Apache-2.0, Copyright 2011-2024 Blender Foundation |
| `looks/templates.py` | default values of Blender 4.5 shader nodes (data) | facts about Blender's node defaults; no code |

Cycles source: https://projects.blender.org/blender/cycles (Apache-2.0). The ported code is a re-implementation in
Warp (Python) and keeps Cycles' conventions and constants; it contains no NVIDIA SDK code.
