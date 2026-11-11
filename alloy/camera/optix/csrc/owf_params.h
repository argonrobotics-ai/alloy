// SPDX-License-Identifier: Apache-2.0
// The NVIDIA OptiX headers this file builds against are obtained separately (github.com/NVIDIA/optix-dev, tag
// v9.0.0, NVIDIA SDK licence) by `python -m alloy.camera.optix build`; they are never shipped with Alloy.
// Launch parameters shared by the OptiX programs (owf_programs.cu) and the host library (owf.cpp).
// Mirrored in Python by alloy.camera.optix.lib.LaunchParams (ctypes): keep the field order and types in sync.
#pragma once
#include <optix_types.h>

enum OwfKind { OWF_CLOSEST = 0, OWF_SHADOW = 1, OWF_LOCAL = 2, OWF_BEVEL = 3, OWF_BEVEL_ANY = 4, OWF_NUM_KINDS = 5 };

#define OWF_BEVEL_MAX_SAMPLES 8  // ray slots per path (Bevel node samples)
#define OWF_BEVEL_MAX_HITS 4     // Cycles LOCAL_MAX_HITS: hits kept per sample (reservoir)
#define OWF_BEVEL_MAX_STEPS 24   // march steps per sample (nodes.bevel_normal)

struct OwfParams {
    const OptixTraversableHandle* handles;  // per world (closest, shadow) or per mesh GAS (local)
    const int* queue;                       // ray i of the launch = path queue[i]; null: path i
    const float* ray_o;                     // float3 per path
    const float* ray_d;                     // float3 per path (need not be unit length; t is in units of |d|)
    const float* ray_tmax;                  // per path
    const int* ray_world;                   // per path index into handles; null: handles[0]
    const unsigned int* world_mask;         // per ray_world visibility mask (worlds sharing an IAS); null: 0xFF
    float* hit_t;                           // closest / local: t (unchanged on a miss)
    float* hit_u;                           // Warp mesh_query_ray convention: P = u p0 + v p1 + (1 - u - v) p2
    float* hit_v;
    int* hit_prim;                          // triangle index in its mesh
    int* hit_inst;                          // OptixInstance::instanceId (GpuScene instance index), -1 = miss
    int* occluded;                          // shadow: 1 blocked, 0 free
    // Bevel (kind OWF_BEVEL, one thread per path): ray slots p * 8 + s in ray_o / ray_d / ray_tmax, ray_world[p] =
    // mesh (GAS index), bv_n[p] samples, LCG state bv_state[p], march step bv_eps[p]; outputs per slot: hit count and
    // up to 4 kept hits (bv_face / bv_u / bv_v at slot * 4 + j), as nd_bevel.bevel_normal's march + reservoir
    const int* bv_n;
    const unsigned int* bv_state;
    const float* bv_eps;
    int* bv_nhits;
    int* bv_face;
    float* bv_u;
    float* bv_v;
    const int* n_dev;                       // if set: rays i >= *n_dev of the launch exit (device-side queue size)
    const int* inst_tie;                    // closest: per instance, 1 = transmissive (loses distance ties); null: none
    int n;                                  // number of rays in the launch
    float tmin;                             // ray tmin (0 = offset origins, no prim skipping)
};
