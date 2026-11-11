# SPDX-License-Identifier: Apache-2.0
"""Registers SolverDFSPH as the ``sph`` solver capability (SPEC §3.8 contract item 1, D9, SCR-WP9-2).

Imported by ``alloy.solvers.sph``; registration is idempotent (an equal capability re-registers as a no-op).
Besides the capability it registers a health adapter (iterations, density error per world) and a state adapter
(the plug-in's ``alloy_state_arrays``) so generic snapshot and health code covers SPH without special cases.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import warp as wp

from alloy.core.protocols import DetCapability
from alloy.core.registry import (
    HealthSample,
    SolverCapability,
    register_health_adapter,
    register_solver,
    register_state_adapter,
)

from .config import DFSPHConfig, register_custom_attributes
from .escape_box import auto_world_bounds
from .solver import SolverDFSPH

log = logging.getLogger("alloy.sph")

_CONFIG_FIELDS = frozenset(f.name for f in dataclasses.fields(DFSPHConfig))


def _factory(view: Any, kwargs: Mapping[str, Any]) -> SolverDFSPH:
    """SolverCoupled.Entry factory: ``config`` (a DFSPHConfig) or Config fields as kwargs, plus capacity_per_world.

    Without ``config`` and without a ``world_bounds`` kwarg (the scene assembler's call), the escape box comes from
    the scene's content (``escape_box.auto_world_bounds``, SCR-OSS3-2); an explicit ``world_bounds`` (None too) or a
    full ``config`` is kept as given."""
    kw = dict(kwargs)
    cfg = kw.pop("config", None)
    cap = kw.pop("capacity_per_world", None)
    fields = {k: kw.pop(k) for k in list(kw) if k in _CONFIG_FIELDS}
    if kw:
        raise TypeError(f"sph factory: unknown kwargs {sorted(kw)}")
    if cfg is None:
        if "world_bounds" not in fields:
            fields["world_bounds"] = auto_world_bounds(view)
            log.info("sph factory: escape box %s from the scene's content (retired particles: report().escaped)",
                     fields["world_bounds"])
        cfg = DFSPHConfig(**fields)
    elif fields:
        cfg = dataclasses.replace(cfg, **fields)
    return SolverDFSPH(view, cfg, capacity_per_world=cap)


class _SPHEmitter:
    """Emitter protocol adapter for SolverDFSPH.emit."""

    def emit(self, b: Any, obj: Any, d: Any) -> None:
        SolverDFSPH.emit(b, obj, d)


@wp.kernel(enable_backward=False)
def _health_kernel(iters_den: wp.array[wp.int32], iters_div: wp.array[wp.int32], den_err: wp.array[wp.float32],
                   iters: wp.array[wp.int32], residual: wp.array[wp.float32]):
    w = wp.tid()
    iters[w] = iters_den[w] + iters_div[w]
    residual[w] = den_err[w]


def _health_adapter(solver: SolverDFSPH, sample: HealthSample) -> None:
    """Per world: total pressure iterations of the last step and the mean density error (capture-safe)."""
    st = solver.stats
    W = solver.layout.world_count
    if W:
        wp.launch(_health_kernel, dim=W, inputs=[st.iters_den, st.iters_div, st.den_err, sample.iters,
                                                 sample.residual], device=solver.model.device)


CAPABILITY = SolverCapability(
    name="sph",
    factory=_factory,
    routes=frozenset({"liquid_sph"}),
    owns=frozenset({"particle", "body"}),  # bodies: free bodies with fluid + gravity only (D5 monolithic, SCR-WP9-2)
    coupling=True,
    uniform_worlds=False,
    per_contact_friction=False,
    determinism=DetCapability.NATIVE,
    friction_mix="none",
    restitution_path=False,
    emitter=_SPHEmitter(),
    register_attributes=register_custom_attributes,
)


def register() -> None:
    register_solver(CAPABILITY)
    register_health_adapter(SolverDFSPH, _health_adapter)
    # with the item->world maps, so a per-world rollback keeps the other worlds (without them the per-particle
    # arrays read as whole and every SPH scene re-ran all worlds: WP4's finding, 2026-11-08)
    register_state_adapter(SolverDFSPH, lambda s: s.alloy_state_arrays(None),
                           world_map=lambda s: s.alloy_state_worlds())


register()
