# SPDX-License-Identifier: Apache-2.0
"""Stable slot order for proxy soft contacts on CUDA (SCR-SC4-4d run-to-run ruling, WP10).

Newton's collision pipeline hands out soft-contact slots with ``atomic_add`` counters, so on the GPU the same contacts
land in a different slot order from run to run. VBD then sums them in a different order, and two runs of one C16
world part (notes: scr/WP4.md, "C16 run-to-run divergence").

``SoftContactOrder.apply`` puts the valid records of a ``Contacts`` in a canonical order, then gathers every
per-record array through it. The order is two stable radix passes over the capacity:
- main key (body, shape, the soft element's first particle), packed into the bits those counts need;
- ties broken by a hash of the rest of the record's identity: the other soft corners, the rigid primitive corners and
  the full-surface feature tag. These integer fields come from the record's geometry, not its slot.

Only integer identity enters the key, so the order depends on which contacts exist, not on float noise in their
positions. Records with equal keys would keep their slot order, so the claim rests on each record's identity being
unique (tests: full-surface records on two pads, per-vertex records on an 8-geom tip). Slots past
``soft_contact_count`` key last and keep their place, so the sort needs no host read and runs inside a captured period
graph. The pattern follows Alloy's MuJoCo Warp contact order in ``alloy.precision.mjw_order`` (WP5), whose hash mix
this reuses.

What it makes reproducible: the record arrays, given the same input state. Not VBD's sums over them: SolverVBD adds
contact forces per particle with float ``atomic_add`` (newton particle_vbd_kernels.py:2477-2520) unless it runs
``deterministic=RUN_TO_RUN``, and MuJoCo Warp's GPU atomics are their own source. So a stable order does not by itself
make a C16 run bitwise (scr/WP4.md, SCR-SC4-4d run-to-run).

Cost: both passes sort the whole capacity each collide, since the count is device-side. On an L4, graph replay, that
is 0.37 ms at C16's proxy capacity (190 k records) and 0.54 ms at its fallback buffer's (331 k); it grows with
``soft_contact_max``, not with the contacts present.

- Only on CUDA: the CPU pipeline is serial, its order is already reproducible, and the CPU goldens stay as they are.
- Not reordered: ``soft_contact_tids``, the slot each backward-replay thread wrote. It is per thread, not per record,
  so ``ProxyContactPipeline`` leaves contacts that require grad in Newton's order.
- Safe after ``collide``: mesh records carry their feature tag in ``_soft_contact_mesh_features``, which moves with
  them (Newton: "explicit tags also survive the row reordering"); the first-mesh-record state is read only inside
  collide's own geometry pass.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.core.errors import CouplingError
from alloy.precision.mjw_order import _bitlen, _ibits, _mix

INVALID = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))
# every per-record array of a Contacts (length soft_contact_max) and its dtype; the mesh ones exist with full surface
RECORD_ARRAYS: dict[str, Any] = {
    "soft_contact_indices": wp.vec3i, "soft_contact_rigid_indices": wp.vec3i, "soft_contact_particle": wp.int32,
    "soft_contact_barycentric": wp.vec3, "soft_contact_shape": wp.int32, "soft_contact_body_pos": wp.vec3,
    "soft_contact_body_vel": wp.vec3, "soft_contact_normal": wp.vec3, "_soft_contact_mesh_features": wp.vec3i,
    "_soft_contact_mesh_params": wp.float32}


@wp.kernel(enable_backward=False)
def iota(perm: wp.array(dtype=wp.int32)):
    i = wp.tid()
    perm[i] = i


@wp.kernel(enable_backward=False)
def tie_keys(count: wp.array(dtype=wp.int32), perm: wp.array(dtype=wp.int32), soft: wp.array(dtype=wp.vec3i),
             rigid: wp.array(dtype=wp.vec3i), feature: wp.array(dtype=wp.vec3i), has_feature: int,
             keys: wp.array(dtype=wp.uint64)):
    """keys[i] = hash(soft corners 1-2, rigid corners, feature tag) of record perm[i], below INVALID."""
    i = wp.tid()
    r = perm[i]
    if r >= count[0]:
        keys[i] = INVALID
        return
    s = soft[r]
    g = rigid[r]
    f = wp.vec3i(-1, -1, -1)
    if has_feature != 0:
        f = feature[r]
    h = _mix(_mix(_mix(wp.uint64(0), _ibits(s[1])), _ibits(s[2])), _ibits(g[0]))
    h = _mix(_mix(_mix(h, _ibits(g[1])), _ibits(g[2])), _ibits(f[0]))
    h = _mix(_mix(h, _ibits(f[1])), _ibits(f[2]))
    keys[i] = h >> wp.uint64(1)


@wp.kernel(enable_backward=False)
def main_keys(count: wp.array(dtype=wp.int32), perm: wp.array(dtype=wp.int32), shape: wp.array(dtype=wp.int32),
              shape_body: wp.array(dtype=wp.int32), soft: wp.array(dtype=wp.vec3i), shape_bits: int,
              particle_bits: int, keys: wp.array(dtype=wp.uint64)):
    """keys[i] = (body + 1, shape + 1, first soft corner + 1) of record perm[i], packed; INVALID past the count."""
    i = wp.tid()
    r = perm[i]
    if r >= count[0]:
        keys[i] = INVALID
        return
    sh = shape[r]
    b = int(-1)
    if sh >= 0:
        b = shape_body[sh]
    k = (wp.uint64(b + 1) << wp.uint64(shape_bits)) | wp.uint64(sh + 1)
    keys[i] = (k << wp.uint64(particle_bits)) | wp.uint64(soft[r][0] + 1)


@wp.kernel(enable_backward=False)
def gather(count: wp.array(dtype=wp.int32), perm: wp.array(dtype=wp.int32), src: wp.array(dtype=Any),
           dst: wp.array(dtype=Any)):
    i = wp.tid()
    if i < count[0]:
        dst[i] = src[perm[i]]


@wp.kernel(enable_backward=False)
def copy_valid(count: wp.array(dtype=wp.int32), src: wp.array(dtype=Any), dst: wp.array(dtype=Any)):
    i = wp.tid()
    if i < count[0]:
        dst[i] = src[i]


class SoftContactOrder:
    """Canonical soft-record order for one ``Contacts`` of a model view; scratch sized to its ``soft_contact_max``."""

    def __init__(self, contacts: Any, view: Any) -> None:
        cap = int(contacts.soft_contact_max)
        dev = contacts.soft_contact_shape.device
        self.cap, self.contacts, self.shape_body = cap, contacts, view.shape_body
        self.shape_bits = _bitlen(int(view.shape_count or 0) + 1)
        self.particle_bits = _bitlen(int(view.particle_count or 0) + 1)
        self.end_bit = _bitlen(int(view.body_count or 0) + 1) + self.shape_bits + self.particle_bits
        if self.end_bit > 63:
            raise ValueError(f"soft contact order: key needs {self.end_bit} bits (> 63)")
        self.keys = wp.zeros(2 * max(cap, 1), dtype=wp.uint64, device=dev)
        self.perm = wp.zeros(2 * max(cap, 1), dtype=wp.int32, device=dev)
        # one scratch per dtype, allocated now: Newton allocates the mesh arrays in the first collide, which may be
        # inside a graph capture; the gathers run in sequence on one stream, so arrays of one dtype share it
        self.tmp = {t: wp.empty(cap, dtype=t, device=dev) for t in set(RECORD_ARRAYS.values())}

    def apply(self) -> None:
        """Order the valid records: stable sort by the tie hash, then by the main key; gather each record array."""
        c, cap, dev = self.contacts, self.cap, self.keys.device
        if cap == 0:
            return
        features = getattr(c, "_soft_contact_mesh_features", None)
        perm, count = self.perm[:cap], c.soft_contact_count
        wp.launch(iota, dim=cap, inputs=[perm], device=dev)
        wp.launch(tie_keys, dim=cap, device=dev,
                  inputs=[count, perm, c.soft_contact_indices, c.soft_contact_rigid_indices,
                          features if features is not None else c.soft_contact_indices, int(features is not None),
                          self.keys[:cap]])
        wp.utils.radix_sort_pairs(self.keys, self.perm, cap)
        wp.launch(main_keys, dim=cap, device=dev,
                  inputs=[count, perm, c.soft_contact_shape, self.shape_body, c.soft_contact_indices,
                          self.shape_bits, self.particle_bits, self.keys[:cap]])
        wp.utils.radix_sort_pairs(self.keys, self.perm, cap, end_bit=self.end_bit)
        for name in RECORD_ARRAYS:
            src = getattr(c, name, None)
            tmp = self.tmp[RECORD_ARRAYS[name]]
            if src is None:
                continue
            if src.shape[0] != cap or src.dtype != tmp.dtype:          # a record array this order cannot carry
                raise CouplingError(f"soft contact order: {name} is {src.dtype.__name__}[{src.shape[0]}], "
                                    f"expected {tmp.dtype.__name__}[{cap}]")
            wp.launch(gather, dim=cap, inputs=[count, perm, src, tmp], device=dev)
            wp.launch(copy_valid, dim=cap, inputs=[count, tmp, src], device=dev)
