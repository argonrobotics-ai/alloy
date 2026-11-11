# SPDX-License-Identifier: Apache-2.0
"""STRAIN_RANGE (coordinator ruling 2026-11-06 14:17Z): a world whose soft tets pass the FEM model's strain range is
flagged DEGRADED, not INVALID; the measure is WP10's ``alloy.topology.soft_strain.SoftStrain`` (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags, Severity
from alloy.topology.soft_strain import SoftStrain

H = 0.02   # tet edge [m]


def _tets(worlds: int = 2) -> newton.Model:
    b = newton.ModelBuilder()
    ids = [b.add_particle(p, (0.0, 0.0, 0.0), 0.01) for p in ((0, 0, 0), (H, 0, 0), (0, H, 0), (0, 0, H))]
    b.add_tetrahedron(*ids, k_mu=1.0e4, k_lambda=1.0e4, k_damp=0.0)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    return s.finalize(device="cpu")


def _squeezed(model: newton.Model, world: int, height: float) -> newton.State:
    """World ``world``'s apex pushed down to ``height`` x H: principal compression 1 - height."""
    s = model.state()
    q = s.particle_q.numpy()
    q[4 * world + 3, 2] = height * H
    s.particle_q.assign(q)
    return s


def _report(model: newton.Model, state: newton.State):
    monitor = HealthMonitor(model, CheckPolicy(skip=frozenset({"energy"})))
    monitor.set_soft_strain(SoftStrain(model))
    monitor.update(0, state, state, None)
    return monitor.poll()


@pytest.mark.parametrize("height, flagged", [(0.3, True), (0.5, False)])   # 70 % / 50 % compression vs 60 %
def test_soft_strain_past_the_range_is_degraded_in_that_world_only(height: float, flagged: bool) -> None:
    model = _tets()
    rep = _report(model, _squeezed(model, 1, height))
    got = [bool(int(f) & int(HealthFlags.STRAIN_RANGE)) for f in rep.flags]
    assert got == [False, flagged]
    assert int(rep.severity[1]) == int(Severity.DEGRADED if flagged else Severity.OK)
    assert rep.invalid_worlds == ()


def test_a_cut_scene_s_health_gets_the_strain_check() -> None:
    from alloy.integration.wiring import _health_topology

    seen = []
    health = SimpleNamespace(model=SimpleNamespace(tet_count=4), set_soft_strain=seen.append)
    notes: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("alloy.topology.soft_strain.SoftStrain", lambda model: ("strain", model))
        _health_topology(health, SimpleNamespace(), notes)
    assert seen == [("strain", health.model)] and any("STRAIN_RANGE" in n for n in notes)
    seen.clear()
    _health_topology(SimpleNamespace(model=SimpleNamespace(tet_count=0), set_soft_strain=seen.append),
                     SimpleNamespace(), [])
    assert seen == []                                                 # no soft tets: no check
    np.testing.assert_equal(int(HealthFlags.STRAIN_RANGE), 1 << 18)
