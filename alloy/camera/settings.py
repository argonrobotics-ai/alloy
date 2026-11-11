# SPDX-License-Identifier: Apache-2.0
"""Path-tracer settings. Every default is the Blender Cycles 4.5 default, so a scene renders like Cycles would.

Example::

    from alloy.camera import RenderSettings
    st = RenderSettings(seed=3, clamp_indirect=10.0)   # Cycles' defaults, another noise seed
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .envvars import env_str

SAMPLER_BLUE_NOISE = 0  # Cycles "Automatic" in a background render = BLUE_NOISE_PURE (Sobol-Burley, blue-noise index)
SAMPLER_SOBOL_BURLEY = 1  # one Owen-scrambled Sobol sequence per pixel (hash_iqnt2d seed), Cycles' white-noise Sobol
SAMPLER_PCG = 2  # plain hashed white noise (fast, not Cycles)


@dataclass
class RenderSettings:
    """Integrator, sampler and film settings (Cycles names). Change ``seed`` for independent noise."""

    # Film / pixel filter (properties.py: BLACKMAN_HARRIS, 1.5 px; film.cpp doubles BH width internally)
    filter_type: str = "BLACKMAN_HARRIS"  # BLACKMAN_HARRIS | GAUSSIAN | BOX
    filter_width: float = 1.5
    first_sample_centered: bool = True  # init_from_camera.h:27, sample 0 uses (0.5, 0.5)

    # Bounces (properties.py defaults; the kernel uses each + 1, scene/integrator.cpp:188-191)
    max_bounces: int = 12
    diffuse_bounces: int = 4
    glossy_bounces: int = 4
    transmission_bounces: int = 12
    transparent_max_bounces: int = 8
    min_light_bounces: int = 0  # RR starts after bounce > min + 1 (path_state.h:272)

    # Clamping (0 = off). Kernel compares the RGB sum against 3 x value (scene/integrator.cpp:271-275)
    clamp_direct: float = 0.0
    clamp_indirect: float = 10.0

    # Lights
    use_light_tree: bool = True  # with the tree, light_inv_rr_threshold = 0 (scene/integrator.cpp:310)
    world_mis: bool = True  # world sampling_method AUTOMATIC -> background light with MIS, BUT Cycles disables it
    # for a world without spatial variation (scene/light.cpp:287-296), e.g. T12's constant colour
    force_world_light: bool = False  # experiment: treat a constant world as a light anyway (not Cycles)
    sun_mis: bool = True  # light "Multiple Importance" default on (only active for angle > 0)
    sun_visible_camera: bool = True
    world_map: bool = True  # sample the world through Cycles' 1024x512 importance map (False: uniform sphere)
    light_order_bg_first: bool = True  # distant-leaf reservoir order (background light is synced before the sun)
    bg_energy_scale: float = 1.0  # experiment knob on the light-tree energy of the world (1 = Cycles)

    # Sampling
    sampler: int = SAMPLER_BLUE_NOISE
    seed: int = 0

    # Ray offset (stand-in for Cycles' self-intersection skipping): P + Ng * max(abs_eps, rel_eps * |P|_inf)
    ray_offset_abs: float = 1.0 / 65536.0
    ray_offset_rel: float = 256.0 / 8388608.0  # 256 ULP of a float near |P|

    # Ray tracing: "warp" = one megakernel with Warp's own BVH (any NVIDIA GPU, or the CPU, slowly);
    # "optix" = wavefront traced on RT cores (needs `python -m alloy.camera.optix build`, see alloy.camera.optix);
    # "wavefront-warp" = the same wavefront traced by Warp (validation); "auto" = "optix" when available, else "warp".
    # Default: $ALLOY_CAMERA_BACKEND, else "warp"
    backend: str = field(default_factory=lambda: env_str("ALLOY_CAMERA_BACKEND", "warp"))

    # Guide buffers
    normal_space: str = "camera"  # Cycles' Denoising Normal is camera space (+Z forward); "world" for the BRIEF API

    def kernel_bounce_limits(self) -> tuple[int, int, int]:  # noqa: D102
        """(max_bounce, max_diffuse_bounce, min_bounce) as Cycles' kernel stores them (+1 each)."""
        return self.max_bounces + 1, self.diffuse_bounces + 1, self.min_light_bounces + 1

    def kernel_clamps(self) -> tuple[float, float]:
        big = 3.4e38
        d = big if self.clamp_direct == 0.0 else self.clamp_direct * 3.0
        i = big if self.clamp_indirect == 0.0 else self.clamp_indirect * 3.0
        return d, i
