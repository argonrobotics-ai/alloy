# SPDX-License-Identifier: Apache-2.0
"""Thread vs pad self-contact gated at open holes (SCR-SC11-1 item 5; the seam_contact pattern, reviewed by WP10).

With VBD self-contact on, every thread particle collides with the pad's surface triangles, so the thread could not
follow the needle through a hole. ``HoleContactFilter`` registers, for every thread particle, the candidate pad
triangles (those whose centroid lies in ``region``) in the detector's vertex-triangle filter list, gated: an entry
filters (contact ignored) only while its triangle's centroid is within ``r_hole`` of a track anchor (a hole, or any
needle / thread / knot anchor: the skin dragged in round a hole makes a funnel along the track, whose faces would snag
the thread; along the track the anchors are the thread-tissue contact). Elsewhere the thread rests on and slides over the pad as usual.
WP10's ``GatedList`` does the merge with Newton's own list and the in-place rewrite (rows stay sorted, sizes fixed,
capturable); its gate compares comp[pa] with comp[pb]: here pb = 0 holds a constant 1 and pa = 1 + candidate index
holds that triangle's near-hole flag.
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

from alloy.topology.seam_contact import GatedList, vbd_detector

from .terms import FREE, SWAGE, TIP, Anchors, Tissue, material_point

log = logging.getLogger("alloy.puncture")


@wp.kernel(enable_backward=False)
def near_holes(cand_tri: wp.array[wp.int32], tri_indices: wp.array2d[wp.int32], tri_world: wp.array[wp.int32],
               an: Anchors, ts: Tissue, capacity: int, used: wp.array[wp.int32], x: wp.array[wp.vec3], r_hole: float,
               flag: wp.array[wp.int32]):
    """flag[1 + c] = 1 when candidate triangle c has a track anchor (needle, thread or knot) of its world within r_hole
    of its centroid."""
    c = wp.tid()
    t = cand_tri[c]
    m = (x[tri_indices[t, 0]] + x[tri_indices[t, 1]] + x[tri_indices[t, 2]]) / 3.0
    w = tri_world[c]
    hit = int(0)
    for j in range(w * capacity, w * capacity + used[w]):
        if an.state[j] != FREE and an.state[j] != SWAGE and an.state[j] != TIP:   # every track anchor: holes and
            if wp.length(material_point(ts, an.tet[j], an.w[j], x) - m) <= r_hole:   # the funnel round them
                hit = 1
    flag[1 + c] = hit


class HoleContactFilter:
    """Gates thread-vs-pad vertex-triangle contact at open holes (module doc)."""

    def __init__(self, solver, thread_ranges: list[tuple[int, int]], region: tuple[np.ndarray, np.ndarray] | None,
                 r_hole: float = 1.0e-3) -> None:
        model = solver.model
        det = vbd_detector(model, solver)
        dev, n = model.device, int(model.particle_count)
        tris = model.tri_indices.numpy()
        q = model.particle_q.numpy()
        cen = q[tris].mean(1)
        thread = np.concatenate([np.arange(a, a + c) for a, c in thread_ranges]) if thread_ranges else np.zeros(0, int)
        is_thread = np.zeros(n, bool)
        is_thread[thread] = True
        pad_tri = ~is_thread[tris].any(1)
        if region is not None:
            lo, hi = (np.asarray(v, float) for v in region)
            pad_tri &= np.all((cen >= lo) & (cen <= hi), axis=1)
        cand = np.flatnonzero(pad_tri)
        if cand.size == 0 or thread.size == 0:
            raise ValueError("hole contact filter: no thread particles or no candidate pad triangles")
        pw = model.particle_world.numpy() if n else np.zeros(1, np.int32)
        tri_w, thr_w = pw[tris[cand, 0]], pw[thread]
        rows, cols, pa = [], [], []
        for w in np.unique(thr_w):                    # a thread only meets its own world's pad
            t_w, c_w = thread[thr_w == w], np.flatnonzero(tri_w == w)
            rows.append(np.repeat(t_w, c_w.size))
            cols.append(np.tile(cand[c_w], t_w.size))
            pa.append(np.tile(1 + c_w, t_w.size))
        rows, cols, pa = np.concatenate(rows), np.concatenate(cols), np.concatenate(pa)
        pb = np.zeros_like(pa)
        self.vt = GatedList.build(n, (det.vertex_triangle_filtering_list, det.vertex_triangle_filtering_list_offsets),
                                  (rows, cols, pa, pb), dev)
        det.set_collision_filter_list(self.vt.live, self.vt.offsets, det.edge_filtering_list,
                                      det.edge_filtering_list_offsets)
        tw = np.maximum(pw[tris[cand, 0]], 0) if int(model.world_count) > 1 else np.zeros(cand.size, np.int64)
        self.cand = wp.array(cand.astype(np.int32), dtype=wp.int32, device=dev)
        self.tri_world = wp.array(tw.astype(np.int32), dtype=wp.int32, device=dev)
        flag = np.zeros(1 + cand.size, np.int32)
        flag[0] = 1
        self.flag = wp.array(flag, dtype=wp.int32, device=dev)
        self.r_hole = float(r_hole)
        self.vt.update(self.flag)
        log.info("hole contact filter: %d thread particles x %d pad triangles gated, r_hole %.3g m",
                 thread.size, cand.size, self.r_hole)

    def update(self, solver, x: wp.array) -> None:
        """Flag the candidate triangles near open holes at positions x and rewrite the list (kernels only)."""
        wp.launch(near_holes, dim=self.cand.shape[0], device=x.device,
                  inputs=[self.cand, solver.model.tri_indices, self.tri_world, solver.anchors, solver.tissue,
                          solver.tracks.capacity, solver.tracks.used, x, self.r_hole, self.flag])
        self.vt.update(self.flag)
