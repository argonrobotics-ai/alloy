// SPDX-License-Identifier: Apache-2.0
// The NVIDIA OptiX headers this file builds against are obtained separately (github.com/NVIDIA/optix-dev, tag
// v9.0.0, NVIDIA SDK licence) by `python -m alloy.camera.optix build`; they are never shipped with Alloy.
// owf: a small OptiX 9 host library for Alloy's wavefront renderer (ctypes C API, see alloy/camera/optix/scene.py).
//  - runs in the caller's CUDA context (Warp's), on the caller's stream; device memory comes from cuMemAlloc
//  - one compacted GAS per unique mesh (built once, from Warp's wp.Mesh point / index arrays); deformables'
//    GAS are updatable and refitted in place each render (owf_gas.h)
//  - one IAS per world over an OptixInstance array the caller fills (Warp kernel); per-step update or rebuild of all
//    world IAS is submitted by a few host threads (one stream + OptiX context each)
//  - trace launches of three kinds (closest, shadow, local) over ray / hit buffers owned by the caller
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "owf_params.h"

namespace {

thread_local std::string g_err;

#define OWF_CU(call)                                                                       \
    do {                                                                                   \
        CUresult r_ = (call);                                                              \
        if (r_ != CUDA_SUCCESS) {                                                          \
            const char* s_ = nullptr;                                                      \
            cuGetErrorString(r_, &s_);                                                     \
            g_err = std::string(#call) + ": " + (s_ ? s_ : "?");                           \
            return -1;                                                                     \
        }                                                                                  \
    } while (0)

#define OWF_OX(call)                                                                       \
    do {                                                                                   \
        OptixResult r_ = (call);                                                           \
        if (r_ != OPTIX_SUCCESS) {                                                         \
            g_err = std::string(#call) + ": " + optixGetErrorName(r_) + " " + optixGetErrorString(r_); \
            return -1;                                                                     \
        }                                                                                  \
    } while (0)

struct SbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct Gas {
    CUdeviceptr buf = 0;
    OptixTraversableHandle handle = 0;
    size_t bytes = 0;
    // updatable GAS (deformables): the build input's buffers and the update scratch, kept for owf_refit_gas
    CUdeviceptr verts = 0, idx = 0, temp_update = 0;
    unsigned n_verts = 0, n_tris = 0;
    size_t temp_update_bytes = 0;
    bool rebuild = false;  // rebuilt in place every render (liquids): temp_update is the build scratch, n_tris the cap
};

struct Worlds {
    int n_worlds = 0, n_threads = 1;  // n_worlds = number of IAS (a group of worlds may share one, see alloy/camera/optix/scene.py)
    std::vector<int> off, cnt;  // instance range of each IAS
    CUdeviceptr instances = 0;  // caller-owned OptixInstance array
    CUdeviceptr out = 0;        // n_worlds IAS, stride out_stride
    size_t out_stride = 0, temp_stride = 0;
    CUdeviceptr temp = 0;       // one temp buffer per thread
    CUdeviceptr d_handles = 0;  // OptixTraversableHandle[n_worlds]
    std::vector<OptixTraversableHandle> handles;
    std::vector<CUstream> streams;            // one per thread
    std::vector<CUevent> join;                // one per thread
    CUevent fork = nullptr;
    std::vector<OptixDeviceContext> tctx;     // per-thread OptiX contexts (index 0 = the main context)
    int allow_update = 1;
};

constexpr int OWF_NUM_PG = 13;  // program groups: 5 raygen, 3 miss, 5 hit groups (see owf_create)

struct Owf {
    CUcontext cu = nullptr;
    OptixDeviceContext ctx = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup pg[OWF_NUM_PG] = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr sbt_mem = 0;
    OptixShaderBindingTable sbt[OWF_NUM_KINDS] = {};
    std::vector<Gas> gas;
    CUdeviceptr d_gas_handles = 0;
    std::vector<CUstream> gas_streams;  // owf_rebuild_gas: side streams the per-world builds spread over
    std::vector<CUevent> gas_join;
    CUevent gas_fork = nullptr;
    Worlds worlds;
    CUdeviceptr d_params = 0;  // ring of OwfParams
    OwfParams* h_params = nullptr;  // pinned ring
    int ring = 256, ring_pos = 0;
    std::string log;
};

struct CtxGuard {
    bool ok;
    explicit CtxGuard(CUcontext c) { ok = cuCtxPushCurrent(c) == CUDA_SUCCESS; }
    ~CtxGuard() {
        CUcontext d;
        if (ok) cuCtxPopCurrent(&d);
    }
};

void log_cb(unsigned int level, const char* tag, const char* msg, void* data) {
    auto* o = static_cast<Owf*>(data);
    if (level <= 2) fprintf(stderr, "[optix %u][%s] %s\n", level, tag, msg);
    if (o) o->log += std::string(tag) + ": " + msg + "\n";
}

int make_pipeline(Owf* o, const char* ptx, size_t ptx_len) {
    OptixModuleCompileOptions mco = {};
    mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pco = {};
    pco.usesMotionBlur = 0;
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pco.numPayloadValues = 7;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char lg[4096];
    size_t lgs = sizeof(lg);
    OWF_OX(optixModuleCreate(o->ctx, &mco, &pco, ptx, ptx_len, lg, &lgs, &o->module));

    // program groups: raygen closest, shadow, local, bevel, bevel_any | miss closest, shadow, noop |
    // hitgroup closest, shadow (none), bevel any-hit, distance ties (plane distance + opaque-only or
    // transmissive-only any-hit)
    constexpr int NPG = OWF_NUM_PG, NRG = 5, NMS = 3, NHG = 5;
    static_assert(NRG + NMS + NHG == NPG, "program group count");
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc d[NPG] = {};
    const char* rg[NRG] = {"__raygen__closest", "__raygen__shadow", "__raygen__local", "__raygen__bevel",
                           "__raygen__bevel_any"};
    for (int k = 0; k < NRG; ++k) {
        d[k].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d[k].raygen.module = o->module;
        d[k].raygen.entryFunctionName = rg[k];
    }
    const char* ms[NMS] = {"__miss__closest", "__miss__shadow", "__miss__noop"};
    for (int k = 0; k < NMS; ++k) {
        d[NRG + k].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        d[NRG + k].miss.module = o->module;
        d[NRG + k].miss.entryFunctionName = ms[k];
    }
    d[NRG + NMS].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    d[NRG + NMS].hitgroup.moduleCH = o->module;
    d[NRG + NMS].hitgroup.entryFunctionNameCH = "__closesthit__closest";
    d[NRG + NMS + 1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;  // shadow: no programs
    d[NRG + NMS + 2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;  // bevel any-hit collection
    d[NRG + NMS + 2].hitgroup.moduleAH = o->module;
    d[NRG + NMS + 2].hitgroup.entryFunctionNameAH = "__anyhit__bevel";
    const char* tie_ah[2] = {"__anyhit__opaque", "__anyhit__transmissive"};
    for (int k = 0; k < 2; ++k) {  // distance ties
        OptixProgramGroupDesc& g = d[NRG + NMS + 3 + k];
        g.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        g.hitgroup.moduleCH = o->module;
        g.hitgroup.entryFunctionNameCH = "__closesthit__plane";
        g.hitgroup.moduleAH = o->module;
        g.hitgroup.entryFunctionNameAH = tie_ah[k];
    }
    lgs = sizeof(lg);
    OWF_OX(optixProgramGroupCreate(o->ctx, d, NPG, &pgo, lg, &lgs, o->pg));

    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    lgs = sizeof(lg);
    OWF_OX(optixPipelineCreate(o->ctx, &pco, &plo, o->pg, NPG, lg, &lgs, &o->pipeline));
    OptixStackSizes ss = {};
    for (int k = 0; k < NPG; ++k) OWF_OX(optixUtilAccumulateStackSizes(o->pg[k], &ss, o->pipeline));
    unsigned int dc_trav, dc_state, cont;
    OWF_OX(optixUtilComputeStackSizes(&ss, 1, 0, 0, &dc_trav, &dc_state, &cont));
    OWF_OX(optixPipelineSetStackSize(o->pipeline, dc_trav, dc_state, cont, 2));

    SbtRecord rec[NPG];
    for (int k = 0; k < NPG; ++k) OWF_OX(optixSbtRecordPackHeader(o->pg[k], &rec[k]));
    OWF_CU(cuMemAlloc(&o->sbt_mem, sizeof(rec)));
    OWF_CU(cuMemcpyHtoD(o->sbt_mem, rec, sizeof(rec)));
    for (int k = 0; k < OWF_NUM_KINDS; ++k) {
        OptixShaderBindingTable& s = o->sbt[k];
        s.raygenRecord = o->sbt_mem + k * sizeof(SbtRecord);
        s.missRecordBase = o->sbt_mem + NRG * sizeof(SbtRecord);
        s.missRecordStrideInBytes = sizeof(SbtRecord);
        s.missRecordCount = NMS;
        s.hitgroupRecordBase = o->sbt_mem + (NRG + NMS) * sizeof(SbtRecord);
        s.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
        s.hitgroupRecordCount = NHG;
    }
    return 0;
}

OptixBuildInput ias_input(const Worlds& W, int w) {
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    bi.instanceArray.instances = W.instances + (CUdeviceptr)W.off[w] * sizeof(OptixInstance);
    bi.instanceArray.numInstances = (unsigned)W.cnt[w];
    return bi;
}

OptixAccelBuildOptions ias_options(const Worlds& W, bool update) {
    OptixAccelBuildOptions ao = {};
    static const bool fast_build = getenv("ALLOY_CAMERA_IAS_FAST_BUILD") != nullptr;
    ao.buildFlags = (fast_build ? OPTIX_BUILD_FLAG_PREFER_FAST_BUILD : OPTIX_BUILD_FLAG_PREFER_FAST_TRACE) |
                    (W.allow_update ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0);
    ao.operation = update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
    return ao;
}

// enqueue the builds of worlds [w0, w1) with context ctx on stream st (temp buffer `slot`)
int enqueue_range(Owf* o, OptixDeviceContext ctx, CUstream st, bool update, int w0, int w1, int slot) {
    Worlds& W = o->worlds;
    for (int w = w0; w < w1; ++w) {
        OptixBuildInput bi = ias_input(W, w);
        OptixAccelBuildOptions ao = ias_options(W, update);
        OptixTraversableHandle h = 0;
        OWF_OX(optixAccelBuild(ctx, st, &ao, &bi, 1, W.temp + (CUdeviceptr)slot * W.temp_stride, W.temp_stride,
                               W.out + (CUdeviceptr)w * W.out_stride, W.out_stride, &h, nullptr, 0));
        if (!update) W.handles[w] = h;
    }
    return 0;
}

// All world builds, ordered after the work already on s and before anything enqueued on s afterwards. The host cost
// of optixAccelBuild (~9 us per call) dominates a 70-instance IAS, so the worlds are split over n_threads host
// threads, each with its own stream, temp buffer and OptiX context. (CUDA graph capture does not work: OptiX runs
// builds on internal streams, so a captured graph comes out empty.)
int submit_worlds(Owf* o, CUstream s, bool update) {
    Worlds& W = o->worlds;
    const int T = W.n_threads;
    OWF_CU(cuEventRecord(W.fork, s));
    std::vector<int> rc(T, 0);
    std::vector<std::string> err(T);
    auto job = [&](int t) {
        if (t > 0 && cuCtxSetCurrent(o->cu) != CUDA_SUCCESS) {
            rc[t] = -1;
            err[t] = "cuCtxSetCurrent";
            return;
        }
        const int w0 = (int)((long long)W.n_worlds * t / T), w1 = (int)((long long)W.n_worlds * (t + 1) / T);
        if (cuStreamWaitEvent(W.streams[t], W.fork, 0) != CUDA_SUCCESS) {
            rc[t] = -1;
            err[t] = "cuStreamWaitEvent";
            return;
        }
        rc[t] = enqueue_range(o, W.tctx[t], W.streams[t], update, w0, w1, t);
        if (rc[t] != 0) err[t] = g_err;
        if (cuEventRecord(W.join[t], W.streams[t]) != CUDA_SUCCESS) rc[t] = -1;
    };
    std::vector<std::thread> th;
    for (int t = 1; t < T; ++t) th.emplace_back(job, t);
    job(0);
    for (auto& x : th) x.join();
    for (int t = 0; t < T; ++t) {
        if (rc[t] != 0) {
            g_err = "world builds (thread " + std::to_string(t) + "): " + err[t];
            return -1;
        }
        OWF_CU(cuStreamWaitEvent(s, W.join[t], 0));
    }
    return 0;
}

}  // namespace

extern "C" {

const char* owf_last_error() { return g_err.c_str(); }

// Create the OptiX context on an existing CUDA context (Warp's) and build the pipeline from PTX / OptiX-IR.
void* owf_create(void* cu_ctx, const char* ptx, size_t ptx_len, int log_level) {
    auto* o = new Owf();
    o->cu = static_cast<CUcontext>(cu_ctx);
    CtxGuard g(o->cu);
    auto fail = [&]() -> void* { delete o; return nullptr; };
    if (optixInit() != OPTIX_SUCCESS) {
        g_err = "optixInit failed (driver does not support this OptiX ABI?)";
        return fail();
    }
    OptixDeviceContextOptions opt = {};
    opt.logCallbackFunction = log_cb;
    opt.logCallbackData = o;
    opt.logCallbackLevel = log_level;
    if (optixDeviceContextCreate(o->cu, &opt, &o->ctx) != OPTIX_SUCCESS) {
        g_err = "optixDeviceContextCreate failed";
        return fail();
    }
    if (make_pipeline(o, ptx, ptx_len) != 0) return fail();
    if (cuMemAlloc(&o->d_params, sizeof(OwfParams) * o->ring) != CUDA_SUCCESS ||
        cuMemHostAlloc((void**)&o->h_params, sizeof(OwfParams) * o->ring, 0) != CUDA_SUCCESS) {
        g_err = "params alloc failed";
        return fail();
    }
    return o;
}

const char* owf_log(void* h) { return static_cast<Owf*>(h)->log.c_str(); }

#include "owf_gas.h"  // owf_add_gas, owf_add_gas_dynamic, owf_refit_gas, GAS handle getters

// Allocate and build n_ias IAS over the caller's OptixInstance array (already filled): IAS j holds instances
// [off[j], off[j] + cnt[j]). n_threads host threads submit the per-step refresh (owf_update_worlds).
int owf_setup_worlds(void* h, int n_ias, const int* off, const int* cnt, CUdeviceptr instances, int allow_update,
                     int n_threads, CUstream s) {
    auto* o = static_cast<Owf*>(h);
    CtxGuard g(o->cu);
    Worlds& W = o->worlds;
    W.n_worlds = n_ias;
    W.off.assign(off, off + n_ias);
    W.cnt.assign(cnt, cnt + n_ias);
    W.instances = instances;
    W.allow_update = allow_update;
    W.n_threads = n_threads < 1 ? 1 : (n_threads > n_ias ? n_ias : n_threads);
    const int n_worlds = n_ias;
    int jmax = 0;
    for (int j = 0; j < n_ias; ++j)
        if (W.cnt[j] > W.cnt[jmax]) jmax = j;
    OptixBuildInput bi = ias_input(W, jmax);
    OptixAccelBuildOptions ao = ias_options(W, false);
    OptixAccelBufferSizes sz;
    OWF_OX(optixAccelComputeMemoryUsage(o->ctx, &ao, &bi, 1, &sz));
    auto align = [](size_t x) { return (x + OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT - 1) / OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT * OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT; };
    W.out_stride = align(sz.outputSizeInBytes);
    W.temp_stride = align(sz.tempSizeInBytes > sz.tempUpdateSizeInBytes ? sz.tempSizeInBytes : sz.tempUpdateSizeInBytes);
    OWF_CU(cuMemAlloc(&W.out, W.out_stride * n_worlds));
    OWF_CU(cuMemAlloc(&W.temp, W.temp_stride * W.n_threads));
    OWF_CU(cuMemAlloc(&W.d_handles, sizeof(OptixTraversableHandle) * n_worlds));
    W.handles.assign(n_worlds, 0);
    W.streams.resize(W.n_threads);
    W.join.resize(W.n_threads);
    for (auto& x : W.streams) OWF_CU(cuStreamCreate(&x, CU_STREAM_NON_BLOCKING));
    for (auto& e : W.join) OWF_CU(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
    OWF_CU(cuEventCreate(&W.fork, CU_EVENT_DISABLE_TIMING));
    W.tctx.assign(W.n_threads, o->ctx);
    const bool own_ctx = getenv("ALLOY_CAMERA_SHARED_CTX") == nullptr;
    for (int t = 1; t < W.n_threads && own_ctx; ++t) {
        OptixDeviceContextOptions opt = {};
        opt.logCallbackFunction = log_cb;
        opt.logCallbackData = o;
        opt.logCallbackLevel = 2;
        OWF_OX(optixDeviceContextCreate(o->cu, &opt, &W.tctx[t]));
    }
    if (submit_worlds(o, s, false) != 0) return -1;
    OWF_CU(cuStreamSynchronize(s));
    OWF_CU(cuMemcpyHtoD(W.d_handles, W.handles.data(), sizeof(OptixTraversableHandle) * n_worlds));
    return 0;
}

CUdeviceptr owf_world_handles_ptr(void* h) { return static_cast<Owf*>(h)->worlds.d_handles; }
unsigned long long owf_ias_handle(void* h, int j) { return static_cast<Owf*>(h)->worlds.handles[j]; }
unsigned long long owf_world_bytes(void* h) {
    auto& W = static_cast<Owf*>(h)->worlds;
    return (unsigned long long)W.out_stride * W.n_worlds;
}

// Refresh every world IAS after the instance array changed: rebuild = 0 refit (update), 1 full rebuild. The handles
// stay the same (same output buffers); a rebuild that changed one is an error.
int owf_update_worlds(void* h, int rebuild, CUstream s) {
    auto* o = static_cast<Owf*>(h);
    CtxGuard g(o->cu);
    Worlds& W = o->worlds;
    if (!rebuild && !W.allow_update) {
        g_err = "worlds built without allow_update";
        return -1;
    }
    std::vector<OptixTraversableHandle> keep;
    if (rebuild) keep = W.handles;
    if (submit_worlds(o, s, !rebuild) != 0) return -1;
    if (rebuild && keep != W.handles) {
        g_err = "IAS rebuild changed a traversable handle";
        return -1;
    }
    return 0;
}

// Trace n rays of one kind. p: host OwfParams (handles may be null: closest/shadow use the world IAS handles,
// local the GAS handles).
int owf_trace(void* h, int kind, const OwfParams* p, CUstream s) {
    auto* o = static_cast<Owf*>(h);
    if (p->n <= 0) return 0;
    CtxGuard g(o->cu);
    const int slot = o->ring_pos;
    o->ring_pos = (o->ring_pos + 1) % o->ring;
    OwfParams& hp = o->h_params[slot];
    hp = *p;
    if (!hp.handles)
        hp.handles = reinterpret_cast<const OptixTraversableHandle*>(
            (kind == OWF_LOCAL || kind == OWF_BEVEL || kind == OWF_BEVEL_ANY) ? o->d_gas_handles : o->worlds.d_handles);
    CUdeviceptr dp = o->d_params + slot * sizeof(OwfParams);
    OWF_CU(cuMemcpyHtoDAsync(dp, &hp, sizeof(OwfParams), s));
    OWF_OX(optixLaunch(o->pipeline, s, dp, sizeof(OwfParams), &o->sbt[kind], (unsigned)p->n, 1, 1));
    return 0;
}

int owf_params_size() { return (int)sizeof(OwfParams); }
int owf_instance_size() { return (int)sizeof(OptixInstance); }

void owf_destroy(void* h) {
    auto* o = static_cast<Owf*>(h);
    if (!o) return;
    {
        CtxGuard g(o->cu);
        cuCtxSynchronize();
        Worlds& W = o->worlds;
        for (auto x : W.streams) cuStreamDestroy(x);
        for (auto e : W.join) cuEventDestroy(e);
        if (W.fork) cuEventDestroy(W.fork);
        for (size_t t = 1; t < W.tctx.size(); ++t)
            if (W.tctx[t] != o->ctx) optixDeviceContextDestroy(W.tctx[t]);
        if (W.out) cuMemFree(W.out);
        if (W.temp) cuMemFree(W.temp);
        if (W.d_handles) cuMemFree(W.d_handles);
        for (auto x : o->gas_streams) cuStreamDestroy(x);
        for (auto e : o->gas_join) cuEventDestroy(e);
        if (o->gas_fork) cuEventDestroy(o->gas_fork);
        for (auto& x : o->gas) {
            cuMemFree(x.buf);
            if (x.temp_update) cuMemFree(x.temp_update);
        }
        if (o->d_gas_handles) cuMemFree(o->d_gas_handles);
        if (o->sbt_mem) cuMemFree(o->sbt_mem);
        if (o->d_params) cuMemFree(o->d_params);
        if (o->h_params) cuMemFreeHost(o->h_params);
        if (o->pipeline) optixPipelineDestroy(o->pipeline);
        for (auto x : o->pg)
            if (x) optixProgramGroupDestroy(x);
        if (o->module) optixModuleDestroy(o->module);
        if (o->ctx) optixDeviceContextDestroy(o->ctx);
    }
    delete o;
}

}  // extern "C"
