# SPDX-License-Identifier: Apache-2.0
"""Principled BSDF data types (node inputs, allocated closures) and path-flag constants.

Split out of principled.py to keep it small; principled.py re-exports every name here.
"""
import warp as wp

from .bssrdf import Bssrdf
from .glass import Glass
from .lobes import Diffuse, Sheen
from .microfacet import Microfacet

# path_flags bits (the caller's path state at the shaded vertex)
PATH_DIFFUSE = wp.constant(1)                 # the ray that reached this vertex was a diffuse bounce (PATH_RAY_DIFFUSE)
PATH_NO_REFLECTIVE_CAUSTICS = wp.constant(2)  # scene setting cycles.caustics_reflective = False (production C3)
PATH_NO_REFRACTIVE_CAUSTICS = wp.constant(4)  # scene setting cycles.caustics_refractive = False
PATH_DIFFUSE_ANCESTOR = wp.constant(8)        # some earlier bounce was diffuse (subsurface falls back to diffuse)

# closure order as Cycles allocates them: 0 sheen, 1 coat, 2 metal, 3 glass (transmission), 4 specular,
# 5 subsurface (BSSRDF), 6 its diffuse fallback, 7 diffuse
N_LOBES = 8
LOBE_GLASS = wp.constant(3)
LOBE_SPEC = wp.constant(4)
LOBE_SSS = wp.constant(5)
LOBE_SSS_DIFFUSE = wp.constant(6)
LOBE_DIFFUSE = wp.constant(7)


@wp.struct
class PrincipledParams:
    """Principled BSDF node inputs at the hit (after the node graph), plus hit geometry and path state."""
    base_color: wp.vec3
    metallic: float
    roughness: float
    ior: float
    specular_ior_level: float
    specular_tint: wp.vec3
    anisotropic: float
    anisotropic_rotation: float
    sheen_weight: float
    sheen_roughness: float
    sheen_tint: wp.vec3
    coat_weight: float
    coat_roughness: float
    coat_ior: float
    coat_tint: wp.vec3
    transmission_weight: float
    subsurface_weight: float
    subsurface_radius: wp.vec3
    subsurface_scale: float
    subsurface_anisotropy: float
    coat_normal: wp.vec3     # Coat Normal socket; zero = unlinked (Cycles then uses ns)
    ng: wp.vec3              # sd->Ng: true geometric (face) normal, facing wi
    ns: wp.vec3              # sd->N: interpolated smooth normal before Bump/Bevel (= ng on flat faces)
    blur_roughness: float    # filter-glossy blur at this vertex: blur_roughness(...) below, 0 = none
    single_scatter_ggx: int  # 0 = Multiscatter GGX (Blender default, C3), 1 = GGX
    no_bump_correction: int  # 0 = material use_bump_map_correction on (Blender default, C3)
    backfacing: int          # SD_BACKFACING: the ray hit the back of the surface (it leaves the object)


@wp.struct
class PrincipledClosures:
    """The closures Cycles allocates for one shading point (after setup and blur)."""
    ns: wp.vec3
    ng: wp.vec3
    bump_correction: int
    sheen: Sheen
    coat: Microfacet
    metal: Microfacet
    glass: Glass
    spec: Microfacet
    sss: Bssrdf
    sss_diffuse: Diffuse
    diffuse: Diffuse

