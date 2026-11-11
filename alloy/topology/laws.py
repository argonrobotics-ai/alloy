# SPDX-License-Identifier: Apache-2.0
"""WP10 bench laws: TOP-2 knife force vs depth (rig suite; SPEC §5.2). Real data are not_run until a rig exists.

The law is registered with ``alloy.bench.law.register_law`` when WP6's bench library is installed. Until then,
importing this module only defines the spec (logged).
"""

from __future__ import annotations

import logging

from alloy.bench.types import LawSpec, Tolerance
from alloy.topology.twin import DEPTHS, LAW_ID, KnifeForceTwin

log = logging.getLogger("alloy.topology")

_METRICS = tuple(f"force_d{int(d * 1e3)}mm" for d in DEPTHS)

TOP2_KNIFE_FORCE = LawSpec(
    id=LAW_ID, suite="rig",
    metrics={k: Tolerance(rel=0.15, source="plan") for k in _METRICS},         # SPEC §5.2 TOP-2: ≤ 15 %
    stat={k: "mean" for k in _METRICS},
    dt=5.0e-4, substeps=1, duration_s=0.225,           # 4.5 mm of descent at 20 mm/s
    perturb={"youngs": 0.1, "softness": 0.25, "cut_spring_ke": 0.25},
    worlds_per_condition=4, solvers=("vbd",), production_dt=False, producer="WP10",
    quick={"worlds": 2, "duration_s": 0.225},
)


def register() -> bool:
    """Register TOP-2 with the bench library; False if WP6's ``alloy.bench.law`` is not installed yet."""
    try:
        from alloy.bench.law import register_law
    except ImportError:
        log.info("alloy.bench.law not installed; law %s not registered", LAW_ID)
        return False
    register_law(TOP2_KNIFE_FORCE, KnifeForceTwin)
    return True


register()
