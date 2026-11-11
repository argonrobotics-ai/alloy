// SPDX-License-Identifier: Apache-2.0
// The NVIDIA OptiX headers this file builds against are obtained separately (github.com/NVIDIA/optix-dev, tag
// v9.0.0, NVIDIA SDK licence) by `python -m alloy.camera.optix build`; they are never shipped with Alloy.
// OptiX device programs for Alloy's wavefront renderer: trace ray buffers written by Warp, write hit records back.
// One raygen per ray kind; one closest-hit (triangles), one miss per ray type. Any-hit only for Bevel's collection and
// the distance-tie trace (opaque surfaces win ties with transmissive ones, see trace_closest).
#include <optix.h>

#include "owf_params.h"

extern "C" {
__constant__ OwfParams params;
}

static __forceinline__ __device__ int path_of(unsigned int i) { return params.queue ? params.queue[i] : (int)i; }

static __forceinline__ __device__ bool out_of_range(unsigned int i) {
    return (int)i >= params.n || (params.n_dev && (int)i >= *params.n_dev);
}

static __forceinline__ __device__ float3 load3(const float* a, int p) {
    return make_float3(a[3 * p], a[3 * p + 1], a[3 * p + 2]);
}

static __forceinline__ __device__ unsigned int mask_of(int w) { return params.world_mask ? params.world_mask[w] : 0xFFu; }

// tracer.geometry.tie_slack / TIE_EXACT: the near-tie window, and plane distances (metres) that count as a tie
#define OWF_TIE_ABS (1.0f / 65536.0f)
#define OWF_TIE_REL (256.0f / 8388608.0f)
#define OWF_TIE_EXACT 1.0e-9
#define OWF_TIE_P_TRANSMISSIVE 0.25f  // tracer.geometry.TIE_P_TRANSMISSIVE: exact ties the transmissive side wins

// tracer.sampling.hash_hp_uint and tracer.geometry.tie_coin (bit-identical)
static __forceinline__ __device__ unsigned int hash_hp_uint(unsigned int i) {
    i ^= i >> 16;
    i *= 0x21F0AAADu;
    i ^= i >> 15;
    i *= 0xD35A2D97u;
    i ^= i >> 15;
    return i ^ 0xE6FE3BEBu;
}

static __forceinline__ __device__ unsigned int ray_bits(float x) {
    return (unsigned int)(int)(fminf(fmaxf(x, -1024.0f), 1024.0f) * 1048576.0f);
}

static __forceinline__ __device__ float tie_coin(float3 o, float3 d) {
    unsigned int h = 0u;
    const float os[3] = {o.x, o.y, o.z}, ds[3] = {d.x, d.y, d.z};
    for (int k = 0; k < 3; ++k) {
        h = hash_hp_uint(h ^ ray_bits(os[k]));
        h = hash_hp_uint(h ^ ray_bits(ds[k]));
    }
    return (float)(h >> 8) * (1.0f / 16777216.0f);
}

static __forceinline__ __device__ float tie_slack(float3 p) {
    return fmaxf(OWF_TIE_ABS, OWF_TIE_REL * fmaxf(fmaxf(fabsf(p.x), fabsf(p.y)), fabsf(p.z)));
}

static __forceinline__ __device__ double as_double(unsigned int lo, unsigned int hi) {
    return __longlong_as_double((long long)(((unsigned long long)hi << 32) | lo));
}

struct Hit5 {
    unsigned int t, b1, b2, prim, inst;
};

// The closest hit within [lo, hi] among the opaque (group 3) or transmissive (group 4) instances, with the float64
// distance to its triangle's plane (__closesthit__plane).
static __forceinline__ __device__ Hit5 trace_side(OptixTraversableHandle h, unsigned int mask, float3 o, float3 d,
                                                 float lo, float hi, unsigned int group, double* plane) {
    Hit5 r = {0u, 0u, 0u, 0u, 0xFFFFFFFFu};
    unsigned int p_lo = 0, p_hi = 0;
    optixTrace(h, o, d, lo, hi, 0.0f, mask, OPTIX_RAY_FLAG_ENFORCE_ANYHIT, group, 1, 0, r.t, r.b1, r.b2, r.prim, r.inst,
               p_lo, p_hi);
    *plane = as_double(p_lo, p_hi);
    return r;
}

// Closest hit; with ``tie``, a hit on a transmissive instance or on an opaque one touching glass (params.inst_tie 1 or
// 2) is checked against the other kind within tie_slack of it: the two hits' exact plane distances decide, and an
// exact tie goes to the transmissive surface for OWF_TIE_P_TRANSMISSIVE of the rays (tie_coin). The rule of
// tracer.geometry.wins_tie (a glass standing exactly on a table renders as in Cycles).
static __forceinline__ __device__ void trace_closest(OptixTraversableHandle h, unsigned int mask, int p, bool tie) {
    const float3 o = load3(params.ray_o, p);
    const float3 d = load3(params.ray_d, p);
    const float tmax = params.ray_tmax[p];
    Hit5 r = {__float_as_uint(tmax), 0u, 0u, 0u, 0xFFFFFFFFu};
    optixTrace(h, o, d, params.tmin, tmax, 0.0f, mask, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, r.t, r.b1, r.b2, r.prim,
               r.inst);
    if (r.inst == 0xFFFFFFFFu) {
        params.hit_inst[p] = -1;
        return;
    }
    const int kind = (tie && params.inst_tie) ? params.inst_tie[r.inst] : 0;
    if (kind != 0) {
        const float tg = __uint_as_float(r.t);
        const float s = tie_slack(make_float3(o.x + d.x * tg, o.y + d.y * tg, o.z + d.z * tg));
        const float lo = fmaxf(params.tmin, tg - s), hi = fminf(tmax, tg + s);
        const bool glass_first = kind == 1;
        double t_other = 0.0, t_self = 0.0;
        const Hit5 other = trace_side(h, mask, o, d, lo, hi, glass_first ? 3u : 4u, &t_other);
        if (other.inst != 0xFFFFFFFFu) {
            const Hit5 self = trace_side(h, mask, o, d, lo, hi, glass_first ? 4u : 3u, &t_self);
            if (self.inst != 0xFFFFFFFFu) {
                const double dt = glass_first ? t_self - t_other : t_other - t_self;  // transmissive - opaque
                const bool glass_wins =
                    dt < -OWF_TIE_EXACT || (dt <= OWF_TIE_EXACT && tie_coin(o, d) < OWF_TIE_P_TRANSMISSIVE);
                r = glass_wins == glass_first ? self : other;
            }
        }
    }
    unsigned int t = r.t, b1 = r.b1, b2 = r.b2, prim = r.prim, inst = r.inst;
    // OptiX: P = (1 - b1 - b2) p0 + b1 p1 + b2 p2; Warp: P = u p0 + v p1 + (1 - u - v) p2
    const float fb1 = __uint_as_float(b1), fb2 = __uint_as_float(b2);
    params.hit_t[p] = __uint_as_float(t);
    params.hit_u[p] = 1.0f - fb1 - fb2;
    params.hit_v[p] = fb1;
    params.hit_prim[p] = (int)prim;
    params.hit_inst[p] = (int)inst;
}

extern "C" __global__ void __raygen__closest() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (out_of_range(i)) return;
    const int p = path_of(i);
    const int w = params.ray_world ? params.ray_world[p] : 0;
    trace_closest(params.handles[w], mask_of(w), p, true);
}

// Local rays (Bevel): handles[ray_world[p]] is a mesh GAS, traced alone (no instance level, hit_inst = 0).
extern "C" __global__ void __raygen__local() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (out_of_range(i)) return;
    const int p = path_of(i);
    const int w = params.ray_world ? params.ray_world[p] : 0;
    trace_closest(params.handles[w], 0xFFu, p, false);
}

// Bevel: march every sample ray of the path's bevel call through its mesh GAS (closest hit, restart past each hit by
// eps, skip a re-hit of the same face), keeping a uniform subset of 4 hits with Cycles' LCG reservoir; the LCG state
// runs through the samples in order, as in nd_bevel.bevel_normal.
extern "C" __global__ void __raygen__bevel() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (out_of_range(i)) return;
    const int p = path_of(i);
    const OptixTraversableHandle h = params.handles[params.ray_world[p]];
    const int ns = params.bv_n[p];
    unsigned int state = params.bv_state[p];
    const float eps = params.bv_eps[p];
    for (int s = 0; s < ns; ++s) {
        const int k = p * OWF_BEVEL_MAX_SAMPLES + s;
        const float3 start = load3(params.ray_o, k);
        const float3 dir = load3(params.ray_d, k);
        const float tmax = params.ray_tmax[k];
        int nhits = 0, last = -1;
        float t0 = 0.0f;
        for (int step = 0; step < OWF_BEVEL_MAX_STEPS; ++step) {
            const float3 o = make_float3(start.x + dir.x * t0, start.y + dir.y * t0, start.z + dir.z * t0);
            unsigned int t = 0, b1 = 0, b2 = 0, prim = 0, inst = 0xFFFFFFFFu;
            optixTrace(h, o, dir, 0.0f, tmax - t0, 0.0f, 0xFF, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, t, b1, b2, prim,
                       inst);
            if (inst == 0xFFFFFFFFu) break;
            t0 += __uint_as_float(t);
            if ((int)prim == last) {
                t0 += eps;
                continue;
            }
            last = (int)prim;
            t0 += eps;
            int slot = -1;
            if (nhits < OWF_BEVEL_MAX_HITS) {
                slot = nhits;
            } else {
                state = 1103515245u * state + 12345u;
                const int rr = (int)(state % (unsigned int)(nhits + 1));
                if (rr < OWF_BEVEL_MAX_HITS) slot = rr;
            }
            nhits += 1;
            if (slot >= 0) {
                const float fb1 = __uint_as_float(b1), fb2 = __uint_as_float(b2);
                params.bv_face[k * OWF_BEVEL_MAX_HITS + slot] = (int)prim;
                params.bv_u[k * OWF_BEVEL_MAX_HITS + slot] = 1.0f - fb1 - fb2;
                params.bv_v[k * OWF_BEVEL_MAX_HITS + slot] = fb1;
            }
        }
        params.bv_nhits[k] = nhits;
    }
}

// Bevel, one traversal per sample ray: the any-hit program sees every triangle the ray crosses inside tmax once and
// keeps a uniform subset of 4 with the same LCG reservoir (Cycles' own local intersection does this too). Hits come
// in traversal order, not distance order, so which 4 are kept when there are more differs from the march (equally
// uniform). Payload: slot base k, hit count, LCG state.
extern "C" __global__ void __raygen__bevel_any() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (out_of_range(i)) return;
    const int p = path_of(i);
    const OptixTraversableHandle h = params.handles[params.ray_world[p]];
    const int ns = params.bv_n[p];
    unsigned int state = params.bv_state[p];
    for (int s = 0; s < ns; ++s) {
        const int k = p * OWF_BEVEL_MAX_SAMPLES + s;
        unsigned int kk = (unsigned int)k, nhits = 0u;
        optixTrace(h, load3(params.ray_o, k), load3(params.ray_d, k), 0.0f, params.ray_tmax[k], 0.0f, 0xFF,
                   OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 2, 1, 2, kk, nhits, state);
        params.bv_nhits[k] = (int)nhits;
    }
}

extern "C" __global__ void __anyhit__bevel() {
    const int k = (int)optixGetPayload_0();
    const unsigned int nhits = optixGetPayload_1();
    unsigned int state = optixGetPayload_2();
    int slot = -1;
    if (nhits < OWF_BEVEL_MAX_HITS) {
        slot = (int)nhits;
    } else {
        state = 1103515245u * state + 12345u;
        const int rr = (int)(state % (nhits + 1u));
        if (rr < OWF_BEVEL_MAX_HITS) slot = rr;
    }
    if (slot >= 0) {
        const float2 b = optixGetTriangleBarycentrics();
        params.bv_face[k * OWF_BEVEL_MAX_HITS + slot] = (int)optixGetPrimitiveIndex();
        params.bv_u[k * OWF_BEVEL_MAX_HITS + slot] = 1.0f - b.x - b.y;
        params.bv_v[k * OWF_BEVEL_MAX_HITS + slot] = b.x;
    }
    optixSetPayload_1(nhits + 1u);
    optixSetPayload_2(state);
    optixIgnoreIntersection();
}

// Distance-tie traces: hit group 3 counts only opaque instances, hit group 4 only transmissive ones.
extern "C" __global__ void __anyhit__opaque() {
    if (params.inst_tie[optixGetInstanceId()] == 1) optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__transmissive() {
    if (params.inst_tie[optixGetInstanceId()] != 1) optixIgnoreIntersection();
}

extern "C" __global__ void __miss__noop() {}

extern "C" __global__ void __raygen__shadow() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (out_of_range(i)) return;
    const int p = path_of(i);
    const int w = params.ray_world ? params.ray_world[p] : 0;
    const float3 o = load3(params.ray_o, p);
    const float3 d = load3(params.ray_d, p);
    unsigned int occ = 1u;
    optixTrace(params.handles[w], o, d, params.tmin, params.ray_tmax[p], 0.0f, mask_of(w),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               1, 1, 1, occ);
    params.occluded[p] = (int)occ;
}

static __forceinline__ __device__ void set_closest_payload() {
    const float2 b = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(__float_as_uint(b.x));
    optixSetPayload_2(__float_as_uint(b.y));
    optixSetPayload_3(optixGetPrimitiveIndex());
    optixSetPayload_4(optixGetTransformListSize() > 0 ? optixGetInstanceId() : 0u);
}

extern "C" __global__ void __closesthit__closest() { set_closest_payload(); }

// Distance-tie traces: the closest hit plus the distance to its triangle's plane in float64 (payloads 5, 6), from the
// triangle's vertices and the ray in the instance's frame.
extern "C" __global__ void __closesthit__plane() {
    set_closest_payload();
    float3 v[3];
    optixGetTriangleVertexData(v);
    const float3 o = optixTransformPointFromWorldToObjectSpace(optixGetWorldRayOrigin());
    const float3 d = optixTransformVectorFromWorldToObjectSpace(optixGetWorldRayDirection());
    const double ax = v[0].x, ay = v[0].y, az = v[0].z;
    const double ux = v[1].x - ax, uy = v[1].y - ay, uz = v[1].z - az;
    const double wx = v[2].x - ax, wy = v[2].y - ay, wz = v[2].z - az;
    const double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    const double t = (nx * (ax - o.x) + ny * (ay - o.y) + nz * (az - o.z)) / (nx * d.x + ny * d.y + nz * d.z);
    const unsigned long long bits = (unsigned long long)__double_as_longlong(t);
    optixSetPayload_5((unsigned int)bits);
    optixSetPayload_6((unsigned int)(bits >> 32));
}

extern "C" __global__ void __miss__closest() { optixSetPayload_4(0xFFFFFFFFu); }

extern "C" __global__ void __miss__shadow() { optixSetPayload_0(0u); }
