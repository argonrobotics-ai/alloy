// SPDX-License-Identifier: Apache-2.0
// Geometry acceleration structures of owf (included once, by owf.cpp, inside its extern "C" block).
//  - owf_add_gas: one compacted GAS over a static triangle mesh
//  - owf_add_gas_dynamic: an updatable GAS over a deformable's vertex buffer (not compacted; keeps its update scratch)
//  - owf_refit_gas: OPTIX_BUILD_OPERATION_UPDATE of GAS [k0, k1) after their vertex buffers changed in place; the
//    traversable handles stay the same, so the world IAS only need their usual refit afterwards
//  - owf_add_gas_rebuild / owf_rebuild_gas: a GAS whose topology changes every render (liquid surfaces): buffers
//    sized for a triangle capacity, rebuilt in place over the first n triangles (same buffer, same handle), the
//    builds spread over side streams

// The triangle build input over a wp.Mesh's point and index arrays (flags must outlive the build call).
static OptixBuildInput gas_input(CUdeviceptr* verts, unsigned n_verts, CUdeviceptr idx, unsigned n_tris,
                                 unsigned* flags) {
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = 12;
    bi.triangleArray.numVertices = n_verts;
    bi.triangleArray.vertexBuffers = verts;
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    bi.triangleArray.indexStrideInBytes = 12;
    bi.triangleArray.numIndexTriplets = n_tris;
    bi.triangleArray.indexBuffer = idx;
    // closest-hit and shadow rays never run any-hit; the any-hit bevel (OWF_BEVEL_ANY, ray flag ENFORCE_ANYHIT) needs
    // each crossed triangle reported once (no duplicates from split BVH references): ALLOY_CAMERA_SINGLE_ANYHIT=0 drops that
    *flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    const char* single = getenv("ALLOY_CAMERA_SINGLE_ANYHIT");
    if (!single || single[0] != '0') *flags |= OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    bi.triangleArray.flags = flags;
    bi.triangleArray.numSbtRecords = 1;
    return bi;
}

// Rebuild the device copy of all GAS handles (for local traces) after a GAS was added.
static int upload_gas_handles(Owf* o) {
    if (o->d_gas_handles) OWF_CU(cuMemFree(o->d_gas_handles));
    std::vector<OptixTraversableHandle> hs;
    for (auto& x : o->gas) hs.push_back(x.handle);
    OWF_CU(cuMemAlloc(&o->d_gas_handles, hs.size() * sizeof(OptixTraversableHandle)));
    OWF_CU(cuMemcpyHtoD(o->d_gas_handles, hs.data(), hs.size() * sizeof(OptixTraversableHandle)));
    return 0;
}

// Build one compacted GAS over a triangle mesh (float3 vertices, int3 indices, device pointers). Returns its index.
int owf_add_gas(void* h, CUdeviceptr verts, int n_verts, CUdeviceptr idx, int n_tris, CUstream s) {
    auto* o = static_cast<Owf*>(h);
    CtxGuard g(o->cu);
    unsigned flags = 0;
    OptixBuildInput bi = gas_input(&verts, (unsigned)n_verts, idx, (unsigned)n_tris, &flags);
    OptixAccelBuildOptions ao = {};
    // random vertex access: the distance-tie programs read the hit triangle's vertices
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sz;
    OWF_OX(optixAccelComputeMemoryUsage(o->ctx, &ao, &bi, 1, &sz));
    CUdeviceptr temp, out, d_csize;
    OWF_CU(cuMemAlloc(&temp, sz.tempSizeInBytes + 8));
    OWF_CU(cuMemAlloc(&out, sz.outputSizeInBytes));
    d_csize = temp + ((sz.tempSizeInBytes + 7) / 8) * 8;
    OptixAccelEmitDesc em = {};
    em.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    em.result = d_csize;
    Gas G;
    OWF_OX(optixAccelBuild(o->ctx, s, &ao, &bi, 1, temp, sz.tempSizeInBytes, out, sz.outputSizeInBytes, &G.handle, &em, 1));
    OWF_CU(cuStreamSynchronize(s));
    size_t csize = 0;
    OWF_CU(cuMemcpyDtoH(&csize, d_csize, sizeof(size_t)));
    if (csize < sz.outputSizeInBytes) {
        OWF_CU(cuMemAlloc(&G.buf, csize));
        OWF_OX(optixAccelCompact(o->ctx, s, G.handle, G.buf, csize, &G.handle));
        OWF_CU(cuStreamSynchronize(s));
        OWF_CU(cuMemFree(out));
        G.bytes = csize;
    } else {
        G.buf = out;
        G.bytes = sz.outputSizeInBytes;
    }
    OWF_CU(cuMemFree(temp));
    o->gas.push_back(G);
    if (upload_gas_handles(o) != 0) return -1;
    return (int)o->gas.size() - 1;
}

// Build one updatable GAS over a deformable's vertex buffer; its vertices may then change in place between
// owf_refit_gas calls (same counts, same buffers). Returns its index.
int owf_add_gas_dynamic(void* h, CUdeviceptr verts, int n_verts, CUdeviceptr idx, int n_tris, CUstream s) {
    auto* o = static_cast<Owf*>(h);
    CtxGuard g(o->cu);
    Gas G;
    G.verts = verts;
    G.idx = idx;
    G.n_verts = (unsigned)n_verts;
    G.n_tris = (unsigned)n_tris;
    unsigned flags = 0;
    OptixBuildInput bi = gas_input(&G.verts, G.n_verts, G.idx, G.n_tris, &flags);
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sz;
    OWF_OX(optixAccelComputeMemoryUsage(o->ctx, &ao, &bi, 1, &sz));
    CUdeviceptr temp;
    OWF_CU(cuMemAlloc(&temp, sz.tempSizeInBytes));
    OWF_CU(cuMemAlloc(&G.buf, sz.outputSizeInBytes));
    G.bytes = sz.outputSizeInBytes;
    G.temp_update_bytes = sz.tempUpdateSizeInBytes > 0 ? sz.tempUpdateSizeInBytes : 8;
    OWF_CU(cuMemAlloc(&G.temp_update, G.temp_update_bytes));
    OWF_OX(optixAccelBuild(o->ctx, s, &ao, &bi, 1, temp, sz.tempSizeInBytes, G.buf, G.bytes, &G.handle, nullptr, 0));
    OWF_CU(cuStreamSynchronize(s));
    OWF_CU(cuMemFree(temp));
    o->gas.push_back(G);
    if (upload_gas_handles(o) != 0) return -1;
    return (int)o->gas.size() - 1;
}

// Refit GAS k0..k1-1 (built by owf_add_gas_dynamic) to their current vertices, queued on stream s.
int owf_refit_gas(void* h, int k0, int k1, CUstream s) {
    auto* o = static_cast<Owf*>(h);
    CtxGuard g(o->cu);
    if (k0 < 0 || k1 > (int)o->gas.size()) {
        g_err = "owf_refit_gas: GAS index out of range";
        return -1;
    }
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    ao.operation = OPTIX_BUILD_OPERATION_UPDATE;
    for (int k = k0; k < k1; ++k) {
        Gas& G = o->gas[k];
        if (!G.temp_update) {
            g_err = "owf_refit_gas: GAS " + std::to_string(k) + " was not built updatable";
            return -1;
        }
        unsigned flags = 0;
        OptixBuildInput bi = gas_input(&G.verts, G.n_verts, G.idx, G.n_tris, &flags);
        OptixTraversableHandle handle = G.handle;
        OWF_OX(optixAccelBuild(o->ctx, s, &ao, &bi, 1, G.temp_update, G.temp_update_bytes, G.buf, G.bytes, &handle,
                               nullptr, 0));
        if (handle != G.handle) {
            g_err = "owf_refit_gas: an update changed a traversable handle";
            return -1;
        }
    }
    return 0;
}

static OptixAccelBuildOptions rebuild_options() {
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    return ao;
}

// A GAS over vertex / index buffers of capacity n_verts / n_tris, rebuilt by owf_rebuild_gas. Returns its index.
int owf_add_gas_rebuild(void* h, CUdeviceptr verts, int n_verts, CUdeviceptr idx, int n_tris, CUstream s) {
    auto* o = static_cast<Owf*>(h);
    CtxGuard g(o->cu);
    Gas G;
    G.verts = verts;
    G.idx = idx;
    G.n_verts = (unsigned)n_verts;
    G.n_tris = (unsigned)n_tris;
    G.rebuild = true;
    unsigned flags = 0;
    OptixBuildInput bi = gas_input(&G.verts, G.n_verts, G.idx, 1, &flags);
    bi.triangleArray.numIndexTriplets = G.n_tris;  // sizes for the capacity
    OptixAccelBuildOptions ao = rebuild_options();
    OptixAccelBufferSizes sz;
    OWF_OX(optixAccelComputeMemoryUsage(o->ctx, &ao, &bi, 1, &sz));
    G.temp_update_bytes = sz.tempSizeInBytes > 0 ? sz.tempSizeInBytes : 8;
    OWF_CU(cuMemAlloc(&G.temp_update, G.temp_update_bytes));
    OWF_CU(cuMemAlloc(&G.buf, sz.outputSizeInBytes));
    G.bytes = sz.outputSizeInBytes;
    bi.triangleArray.numIndexTriplets = 1;  // the first (placeholder) triangle until the first rebuild
    OWF_OX(optixAccelBuild(o->ctx, s, &ao, &bi, 1, G.temp_update, G.temp_update_bytes, G.buf, G.bytes, &G.handle,
                           nullptr, 0));
    OWF_CU(cuStreamSynchronize(s));
    o->gas.push_back(G);
    if (upload_gas_handles(o) != 0) return -1;
    return (int)o->gas.size() - 1;
}

// Side streams for owf_rebuild_gas (ALLOY_CAMERA_GAS_STREAMS, default 8; 1 = all builds on the caller's stream).
static int gas_side_streams(Owf* o) {
    if (o->gas_fork) return (int)o->gas_streams.size();
    const char* env = getenv("ALLOY_CAMERA_GAS_STREAMS");
    int n = env ? atoi(env) : 8;
    n = n < 1 ? 1 : (n > 32 ? 32 : n);
    OWF_CU(cuEventCreate(&o->gas_fork, CU_EVENT_DISABLE_TIMING));
    for (int i = 0; i < n; ++i) {
        CUstream st;
        CUevent ev;
        OWF_CU(cuStreamCreate(&st, CU_STREAM_NON_BLOCKING));
        OWF_CU(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
        o->gas_streams.push_back(st);
        o->gas_join.push_back(ev);
    }
    return n;
}

// Rebuild GAS k0..k1-1 (owf_add_gas_rebuild) over their first n_tris[k - k0] triangles (host array, each 1..cap).
// The builds are small (one world's liquid each) and use their own scratch, so they spread over the side streams,
// which wait for the caller's stream s first and which s waits for afterwards: the same order of work as one stream.
int owf_rebuild_gas(void* h, int k0, int k1, const int* n_tris, CUstream s) {
    auto* o = static_cast<Owf*>(h);
    CtxGuard g(o->cu);
    if (k0 < 0 || k1 > (int)o->gas.size()) {
        g_err = "owf_rebuild_gas: GAS index out of range";
        return -1;
    }
    int ns = gas_side_streams(o);
    if (ns < 0) return -1;
    OWF_CU(cuEventRecord(o->gas_fork, s));
    for (int i = 0; i < ns; ++i) OWF_CU(cuStreamWaitEvent(o->gas_streams[i], o->gas_fork, 0));
    OptixAccelBuildOptions ao = rebuild_options();
    for (int k = k0; k < k1; ++k) {
        Gas& G = o->gas[k];
        int n = n_tris[k - k0];
        if (!G.rebuild || n < 1 || (unsigned)n > G.n_tris) {
            g_err = "owf_rebuild_gas: GAS " + std::to_string(k) + " is not rebuildable or the count is out of range";
            return -1;
        }
        unsigned flags = 0;
        OptixBuildInput bi = gas_input(&G.verts, G.n_verts, G.idx, (unsigned)n, &flags);
        OptixTraversableHandle handle = 0;
        OWF_OX(optixAccelBuild(o->ctx, o->gas_streams[(k - k0) % ns], &ao, &bi, 1, G.temp_update,
                               G.temp_update_bytes, G.buf, G.bytes, &handle, nullptr, 0));
        if (handle != G.handle) {
            g_err = "owf_rebuild_gas: a rebuild changed a traversable handle";
            return -1;
        }
    }
    for (int i = 0; i < ns; ++i) {
        OWF_CU(cuEventRecord(o->gas_join[i], o->gas_streams[i]));
        OWF_CU(cuStreamWaitEvent(s, o->gas_join[i], 0));
    }
    return 0;
}

unsigned long long owf_gas_handle(void* h, int k) { return static_cast<Owf*>(h)->gas[k].handle; }
unsigned long long owf_gas_bytes(void* h, int k) { return static_cast<Owf*>(h)->gas[k].bytes; }
CUdeviceptr owf_gas_handles_ptr(void* h) { return static_cast<Owf*>(h)->d_gas_handles; }
