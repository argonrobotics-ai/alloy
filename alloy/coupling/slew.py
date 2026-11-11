# SPDX-License-Identifier: Apache-2.0
"""Static-preserving slew limit on lagged proxy feedback (SCR-SC4-4d, WP10; the pinch-cost ruling's step 2).

A lagged body proxy's feedback force reaches its source one pass or exchange late. In C16 a single pass can bring a
300-900 N harvest spike to a YAM finger (notes: docs/notes/wp10-sc4-4d.md). A static stiffness cap tight enough to
stop that softens every grip (film contact 0.16 mm deeper at c 0.1). This limit acts on the change instead.

- Per body endpoint, the feedback after each harvest may differ from the exchange-start feedback by at most
  ``dF = c_s · m_eff · v_s / dt_x`` in its linear part. The torque is scaled by the same factor.
- A force held over exchanges is never clipped, so contact equilibria (a pinch, a lifted load) keep full stiffness.
  A one-pass jump moves the source by at most ``c_s · v_s`` per exchange.
- ``v_s = SLEW_SPEED``, 0.5 m/s, the SC4-4d speed margin.
- Kinematic proxies (m_eff 0) and frozen worlds are left alone. Particle proxies are not limited.

**Opt-in, off by default** (``SchedulerPolicy.exchange_slew_c`` None; coordinator 2026-11-06). Measured costs
(docs/notes/wp10-sc4-4d.md): with c_s 0.1 + c_d 0.5, C7 W12 runs clean (WP4, SCR-SC4-7); CPL-4 conservative takes 3
rollbacks (0 without); the CPU coupling suite's first-contact onset test fails (the limit delays the contact force);
C16 latched in 4 of 4 worlds by 4.8 s.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.coupling import residuals as rs
from alloy.coupling.endpoints import EndpointSet

SLEW_SPEED = 0.5      # [m/s] the largest source speed change one exchange's feedback change may cause / c_s
FROZEN = rs.FROZEN


@wp.kernel(enable_backward=False)
def slew_limit_kernel(proxy_global: wp.array(dtype=int), endpoint_world: wp.array(dtype=int),
                      world_converged: wp.array(dtype=wp.int32), meff: wp.array(dtype=float),
                      f_begin: wp.array(dtype=wp.spatial_vector), n_per_kg: float,
                      coupling_forces: wp.array(dtype=wp.spatial_vector), peak: wp.array(dtype=float)):
    """coupling_forces[g] = f_begin + (F - f_begin) · min(1, dF / |dF_linear|), dF = n_per_kg · m_eff."""
    i = wp.tid()
    if world_converged[endpoint_world[i]] == FROZEN or meff[i] <= 0.0:
        return
    g = proxy_global[i]
    d = coupling_forces[g] - f_begin[i]
    lin = wp.length(wp.vec3(d[0], d[1], d[2]))
    lim = n_per_kg * meff[i]
    if lin <= lim:
        return
    coupling_forces[g] = f_begin[i] + d * (lim / lin)
    wp.atomic_max(peak, i, lin / lim)


class FeedbackSlewLimit:
    """The limit over a loop's body endpoint sets; ``begin`` at each exchange start, ``apply`` after each harvest."""

    def __init__(self, sets: list[EndpointSet], c_s: float, speed: float = SLEW_SPEED) -> None:
        if c_s <= 0.0:
            raise ValueError(f"feedback slew limit: c_s must be > 0 (got {c_s})")
        self.c_s, self.speed = float(c_s), float(speed)
        self.sets = [es for es in sets if es.kind == "body" and es.count]
        self.f_begin = {id(es): wp.zeros(es.count, dtype=wp.spatial_vector, device=es.meff.device) for es in self.sets}
        self.peak = {id(es): wp.zeros(es.count, dtype=float, device=es.meff.device) for es in self.sets}

    def begin(self) -> None:
        """Keep each endpoint's feedback at the exchange start (the previous exchange's accepted value)."""
        for es in self.sets:
            wp.launch(rs.gather_feedback_kernel, dim=es.count, device=es.meff.device,
                      inputs=[es.proxy_global, es.mapping.coupling_forces, self.f_begin[id(es)]])

    def apply(self, es: EndpointSet, world_converged: wp.array, dt_x: float) -> None:
        """Limit the harvest just written to ``es``'s coupling forces (no-op for untracked or particle sets)."""
        key = id(es)
        if key not in self.f_begin or dt_x <= 0.0:
            return
        wp.launch(slew_limit_kernel, dim=es.count, device=es.meff.device,
                  inputs=[es.proxy_global, es.world, world_converged, es.meff, self.f_begin[key],
                          self.c_s * self.speed / dt_x, es.mapping.coupling_forces, self.peak[key]])

    def peak_ratio(self, reset: bool = True) -> np.ndarray:
        """Largest |linear change| / dF per endpoint since the last read, over all sets (0: never limited)."""
        out = [self.peak[id(es)].numpy().copy() for es in self.sets]
        if reset:
            for es in self.sets:
                self.peak[id(es)].zero_()
        return np.concatenate(out) if out else np.zeros(0)
