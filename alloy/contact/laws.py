# SPDX-License-Identifier: Apache-2.0
"""WP3 bench laws (SPEC §5.3, bench §4): A7 hydroelastic pressure law, A8 clearance geometry and the 50 N push, A5.speed
(Coulomb sliding force independent of speed, ``alloy.contact.law_speed``), A8.squeeze_wedge / _face (BUL-0014 squeeze
penetration, ``alloy.contact.law_squeeze``).

Registered with ``alloy.bench.law.register_law`` on import (``alloy.bench.law.discover`` imports this module). The
twins live in ``alloy.contact.law_twins`` (A7.hydroelastic, A8.push) and ``alloy.contact.law_clearance``
(A8.clearance). Texture SDFs need CUDA: on another device these laws report ``not_run`` with the reason
(``alloy.contact.law_device.CudaOnly``), never a pass and not an ``error``.
"""

from __future__ import annotations

from alloy.bench.types import LawSpec, Tolerance
from alloy.contact.law_clearance import ClearancePlan, ClearanceTwin
from alloy.contact.law_device import cuda_only
from alloy.contact.law_twins import (A7_DT, A7_DURATION_S, DELTA_OVER_R, HydroelasticSphereTwin, PegPushTwin, PushPlan,
                                     push_dt)
from alloy.core.log import get_logger

log = get_logger("contact.laws")

_A7 = ("bench §4 A7 hydroelastic branch: closed form of the configured linear pressure law, F = pi R k_eff delta^2; "
       "A7's exponent (+-0.05) and prefactor (10 %) tolerances")
_A8 = "bench §4 A8 + SPEC §5.3 amendment"
# D2 for the 12.0 / 12.2 mm pair, min(50 um, radial / 2): the law's own bound, a literal. Audit AB-laws-2: read from
# push_pair().resolved_max_penetration() it moved with the engine rule that also sets the pair's stiffness
# (K = 2 F / max_pen), so a rule returning twice the depth halved K and doubled the bound together.
_MAX_PEN_M = 5.0e-5
_QUICK_PLAN = ClearancePlan.for_run(quick=True)

A7_HYDROELASTIC = LawSpec(
    id="A7.hydroelastic", suite="analytic",
    metrics={"exponent_err": Tolerance(abs=0.05, source=f"{_A7}: fitted exponent 2 +- 0.05"),
             "prefactor_rel_err": Tolerance(abs=0.10, source=f"{_A7}: every load's prefactor within 10 %"),
             "settle_rel_err": Tolerance(abs=0.01, source="derived: a static law needs a settled pose; delta changes "
                                                          "<= 1 % between the halves of the averaging window")},
    stat={"exponent_err": "mean", "prefactor_rel_err": "mean", "settle_rel_err": "mean"},
    dt=A7_DT, substeps=1, duration_s=A7_DURATION_S, worlds_per_condition=1, solvers=("mujoco",), producer="WP3",
    quick={"worlds": len(DELTA_OVER_R), "duration_s": A7_DURATION_S},
)

A8_CLEARANCE = LawSpec(
    id="A8.clearance", suite="analytic",
    metrics={"gap_err_m": Tolerance(abs=1.0e-6, source=f"{_A8}: centred gap within 1 um of the radial clearance"),
             "tilt_err_deg": Tolerance(abs=0.01, source=f"{_A8}: max tilt within 0.01 deg of d cos(t) + l sin(t) = D")},
    stat={"gap_err_m": "mean", "tilt_err_deg": "mean"},
    dt=1.0e-3, substeps=1, duration_s=0.0, worlds_per_condition=1, solvers=("none",), producer="WP3",
    quick={"worlds": len(_QUICK_PLAN.centred_depths) + _QUICK_PLAN.directions * _QUICK_PLAN.tilts, "duration_s": 0.0},
)

A8_PUSH = LawSpec(
    id="A8.push", suite="analytic",
    metrics={"pen_max_m": Tolerance(abs=_MAX_PEN_M, source=f"{_A8}: the 50 N ramp leaves <= the pair's "
                                                          f"max_penetration (D2), measured by GapMonitor")},
    stat={"pen_max_m": "mean"},
    dt=push_dt(), substeps=1, duration_s=PushPlan().duration_s, perturb={"tilt_deg": 0.05, "lateral_m": 1.0e-5},
    worlds_per_condition=16, solvers=("mujoco",), producer="WP3",
    quick={"worlds": 4, "duration_s": PushPlan().duration_s},
)


_CUDA_ONLY = (cuda_only(HydroelasticSphereTwin), cuda_only(ClearanceTwin), cuda_only(PegPushTwin))


def register() -> bool:
    """Register the WP3 laws; False if WP6's ``alloy.bench.law`` is not installed."""
    try:
        from alloy.bench.law import register_law
    except ImportError:
        log.info("alloy.bench.law not installed; WP3 laws not registered")
        return False
    register_law(A7_HYDROELASTIC, _CUDA_ONLY[0])
    register_law(A8_CLEARANCE, _CUDA_ONLY[1])
    register_law(A8_PUSH, _CUDA_ONLY[2])
    from alloy.contact.rig_twins import R1_LAW, R1Twin  # noqa: PLC0415  (R1 rig twin, synthetic data in rig_synth)

    register_law(R1_LAW, R1Twin)
    from alloy.contact.thread_rig import R2_LAW, ThreadR2Twin  # noqa: PLC0415  (R2, synthetic data)
    from alloy.contact.thread_twins import A9_LAW, ThreadA9Twin  # noqa: PLC0415  (analytic thread pair)

    register_law(A9_LAW, ThreadA9Twin)
    register_law(R2_LAW, ThreadR2Twin)
    from alloy.contact.law_speed import LAW_SPEED, SlidingSpeedTwin  # noqa: PLC0415  (A5.speed, sliding speed)

    register_law(LAW_SPEED, SlidingSpeedTwin)
    from alloy.contact.law_squeeze import LAW_FACE, LAW_WEDGE, SqueezeFaceTwin, SqueezeTwin  # noqa: PLC0415 (BUL-0014)

    register_law(LAW_WEDGE, SqueezeTwin)
    register_law(LAW_FACE, SqueezeFaceTwin)
    return True


register()
