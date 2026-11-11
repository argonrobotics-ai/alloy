# SPDX-License-Identifier: Apache-2.0
"""Shared long box-on-cloth runs for the CPL metric tests (computed once per session, marked slow)."""

from __future__ import annotations

import pytest

from alloy.testing.coupling import ClothSceneSpec, PeriodRunner, cloth_scheduler, coupling_box_on_cloth

CONTROL_DT = 0.02


def _cloth_run(box_mass: float, seconds: float, **spec_kw):  # noqa: D103
    spec = ClothSceneSpec(box_mass=box_mass, **spec_kw)
    model = coupling_box_on_cloth(1, spec, device="cpu")
    sched = cloth_scheduler(model, spec)                     # default SchedulerPolicy: adaptive clock, lagged x2
    runner = PeriodRunner(model, sched, CONTROL_DT, log_exchanges=True)
    reports = runner.run(int(round(seconds / CONTROL_DT)))
    return {"spec": spec, "model": model, "sched": sched, "runner": runner, "reports": reports}


@pytest.fixture(scope="session")
def cloth_run_02():
    """0.2 kg box on the coupling-time cloth, default policy, 2 s (the CPL-1/2/3/4/10 run)."""
    return _cloth_run(0.2, 2.0)


@pytest.fixture(scope="session")
def cloth_run_conservative():
    """Same scene with μ = 0 and no cloth/contact damping (CPL-4 conservative scene), 2 s."""
    return _cloth_run(0.2, 2.0, contact_mu=0.0, damped=False)


@pytest.fixture(scope="session")
def cloth_run_heavy_cloth():
    """0.2 kg box on a 5 kg cloth: the cloth sags well below its reset height, so E is clearly < 0 (D20), 1 s."""
    return _cloth_run(0.2, 1.0, cloth_mass=5.0)
