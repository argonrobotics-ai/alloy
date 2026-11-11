# SPDX-License-Identifier: Apache-2.0
"""Proxy endpoint tables: the per-mapping device arrays the Scheduler adds on top of Newton's proxy mappings.

One ``EndpointSet`` per Newton ``_ProxyEntityMapping`` (body or particle). Endpoint ``i`` is the i-th source entity of
the mapping; its proxy lives in the destination entry at ``proxy_ids_local[i]`` and its feedback at
``coupling_forces[proxy_ids_global[i]]`` (newton@ac7f7830:newton/_src/solvers/coupled/solver_coupled_proxy.py:36-66,
841-890). Arrays are allocated once at construction, so every kernel launch on them is capture-safe.

``clock_masses``: the proxy masses the exchange clock and the strong-pair class read. A body proxy whose source body
has no shape that collides (no shapes, or visual ones only) never touches the destination, so it reads 0 there: no
bound. WP9 2026-11-08: a hinged wrist with an inertial only got a 0.85 g proxy on the MuJoCo Warp route (Newton's
effective mass at its endpoint, centre of mass on the hinge axis) against 0.3 kg on the C route, and set dt_x 16x
finer (j = 10 instead of 6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.coupling.aitken import WorldCSR, build_world_csr
from alloy.coupling.axis_mass import axis_mass_ratio

log = logging.getLogger("alloy.coupling")


@dataclass
class EndpointSet:
    """Device arrays for one proxy mapping; ``kind`` is "body" (spatial forces) or "particle" (vec3 forces)."""

    mapping: Any                      # Newton _ProxyEntityMapping (private, pinned)
    kind: str
    src_name: str
    dst_name: str
    count: int
    world: wp.array                   # (P,) int
    src_global: wp.array              # (P,) int, parent-model ids
    src_local: wp.array               # (P,) int, source-view ids
    proxy_local: wp.array             # (P,) int, destination-view ids
    proxy_global: wp.array            # (P,) int, index into coupling_forces
    meff: wp.array                    # (P,) float, proxy mass installed in the destination [kg]
    f_used: wp.array                  # (P,) feedback the source used in the current pass
    f_raw: wp.array                   # (P,) raw harvest of the current pass
    frozen_out: wp.array              # (P,) blended value held for frozen worlds
    r_prev: wp.array                  # (P,) previous Aitken residual
    x_begin: wp.array                 # (P,) vec3 source position at exchange start [m]
    x_prev: wp.array                  # (P,) vec3 source end position of the previous pass (staggered residual)
    v_prev: wp.array                  # (P,) vec3 source end velocity of the previous pass
    csr: WorldCSR
    aitken_stats: wp.array            # (W, 2) float64
    aitken_omega: wp.array            # (W,) float32
    aitken_has_prev: wp.array         # (W,) int32
    staggered: bool = False
    strong: bool = False
    extra: dict[str, wp.array] = field(default_factory=dict)
    m_axis: np.ndarray | None = None  # (P,) host: mass along the source body's most mobile axis [kg] (axis_mass)
    touches: np.ndarray | None = None  # (P,) host bool: the source can touch the destination (``clock_masses``)

    @property
    def force_dtype(self) -> Any:
        return wp.spatial_vector if self.kind == "body" else wp.vec3

    def state_arrays(self, prefix: str) -> dict[str, wp.array]:
        """Arrays that carry across exchanges and must be in snapshots (feedback, Aitken state)."""
        return {
            f"{prefix}.coupling_forces": self.mapping.coupling_forces,
            f"{prefix}.r_prev": self.r_prev,
            f"{prefix}.aitken_omega": self.aitken_omega,
            f"{prefix}.aitken_stats": self.aitken_stats,
        }


def _local_ids(global_to_local: wp.array, global_ids: np.ndarray, what: str) -> np.ndarray:
    table = global_to_local.numpy()
    local = table[global_ids]
    if np.any(local < 0):
        raise ValueError(f"{what}: global ids {global_ids[local < 0].tolist()} are not visible in the entry view")
    return local.astype(np.int64)


def build_endpoint_set(
    mapping: Any,
    kind: str,
    src_entry: Any,
    dst_entry: Any,
    model: Any,
    omega0: float,
) -> EndpointSet:
    """Build the endpoint arrays for one Newton proxy mapping after entry views exist."""
    device = model.device
    is_body = kind == "body"
    src_global = mapping.src_ids.numpy().astype(np.int64)
    proxy_local = mapping.proxy_ids_local.numpy().astype(np.int64)
    worlds = (model.body_world if is_body else model.particle_world).numpy()[src_global].astype(np.int64)
    src_g2l = src_entry.body_global_to_local if is_body else src_entry.particle_global_to_local
    src_local = _local_ids(src_g2l, src_global, f"proxy source {kind}s of {mapping.src_name!r}")
    mass = (dst_entry.view.body_mass if is_body else dst_entry.view.particle_mass).numpy()
    meff = mass[proxy_local].astype(np.float32)
    ratio = axis_mass_ratio(src_entry.solver) if is_body else None
    m_axis = meff.astype(np.float64) * (ratio[src_local] if ratio is not None else 1.0)
    touches = colliding_bodies(model)[src_global] if is_body else np.ones(src_global.shape, dtype=bool)
    n = int(src_global.shape[0])
    w = int(model.world_count)
    dtype = wp.spatial_vector if is_body else wp.vec3

    def arr(values: np.ndarray, dt: Any = int) -> wp.array:
        return wp.array(values, dtype=dt, device=device)
    log.debug("endpoint set %s->%s (%s): %d endpoints, %d worlds", mapping.src_name, mapping.dst_name, kind, n, w)
    return EndpointSet(
        mapping=mapping,
        kind=kind,
        src_name=mapping.src_name,
        dst_name=mapping.dst_name,
        count=n,
        world=arr(worlds),
        src_global=arr(src_global),
        src_local=arr(src_local),
        proxy_local=mapping.proxy_ids_local,
        proxy_global=mapping.proxy_ids_global,
        meff=arr(meff, float),
        f_used=wp.zeros(n, dtype=dtype, device=device),
        f_raw=wp.zeros(n, dtype=dtype, device=device),
        frozen_out=wp.zeros(n, dtype=dtype, device=device),
        r_prev=wp.zeros(n, dtype=dtype, device=device),
        x_begin=wp.zeros(n, dtype=wp.vec3, device=device),
        x_prev=wp.zeros(n, dtype=wp.vec3, device=device),
        v_prev=wp.zeros(n, dtype=wp.vec3, device=device),
        csr=build_world_csr(worlds, w, device),
        aitken_stats=wp.zeros((w, 2), dtype=wp.float64, device=device),
        aitken_omega=wp.full(w, float(omega0), dtype=float, device=device),
        aitken_has_prev=wp.zeros(w, dtype=wp.int32, device=device),
        m_axis=m_axis,
        touches=touches,
    )


COLLIDES = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)


def colliding_bodies(model: Any) -> np.ndarray:
    """(body_count,) bool: bodies with at least one shape that collides with shapes or particles."""
    out = np.zeros(int(model.body_count), dtype=bool)
    if not int(model.shape_count or 0):
        return out
    body, flags = model.shape_body.numpy(), model.shape_flags.numpy()
    out[body[(body >= 0) & ((flags.astype(np.int64) & COLLIDES) != 0)]] = True
    return out


def clock_masses(es: EndpointSet) -> np.ndarray:
    """(P,) float64 proxy masses for the exchange clock and the strong-pair class: the axis mass (else the installed
    proxy mass), 0 (no bound) where the source cannot touch the destination."""
    m = np.asarray(es.m_axis if es.m_axis is not None else es.meff.numpy(), dtype=np.float64)
    return m if es.touches is None else np.where(es.touches, m, 0.0)


@wp.kernel(enable_backward=False)
def capture_body_begin_kernel(
    src_global: wp.array[int], body_q: wp.array[wp.transform], x_begin: wp.array[wp.vec3]
):
    i = wp.tid()
    x_begin[i] = wp.transform_get_translation(body_q[src_global[i]])


@wp.kernel(enable_backward=False)
def capture_particle_begin_kernel(
    src_global: wp.array[int], particle_q: wp.array[wp.vec3], x_begin: wp.array[wp.vec3]
):
    i = wp.tid()
    x_begin[i] = particle_q[src_global[i]]


def capture_begin_positions(sets: list[EndpointSet], state_in: Any) -> None:
    """Record every endpoint's source position at exchange start from the parent input state."""
    for es in sets:
        if es.count == 0:
            continue
        if es.kind == "body":
            wp.launch(capture_body_begin_kernel, dim=es.count, inputs=[es.src_global, state_in.body_q, es.x_begin],
                      device=es.world.device)
        else:
            wp.launch(capture_particle_begin_kernel, dim=es.count,
                      inputs=[es.src_global, state_in.particle_q, es.x_begin], device=es.world.device)
