# SPDX-License-Identifier: Apache-2.0
"""DFSPHConfig.fast() (SCR-WP8-7): the resting-jitter rule off and CFL 0.4, documented trade-offs."""

from __future__ import annotations

import numpy as np

from alloy.solvers.sph.config import DFSPHConfig
from alloy.testing.sph import run, water_column


def test_fast_preset_fields_and_overrides():
    f = DFSPHConfig.fast()
    assert f.rest_jitter_speed is None and f.cfl == 0.4
    f.validate()
    g = DFSPHConfig.fast(cfl=0.3, particle_radius=1.0e-3)
    assert g.cfl == 0.3 and g.particle_radius == 1.0e-3 and g.rest_jitter_speed is None
    d = DFSPHConfig()
    assert d.cfl == 0.2 and d.rest_jitter_speed is not None  # the defaults are unchanged


def test_fast_preset_takes_fewer_density_iterations_at_rest():
    """A primed resting column: the rule tightens the density tolerance at rest; fast() keeps max_density_error."""
    its = []
    for cfg in (DFSPHConfig(particle_radius=1.0e-3), DFSPHConfig.fast(particle_radius=1.0e-3)):
        sc = water_column(6, 6, 20, cfg=cfg, device="cpu")
        s = sc.make_solver()
        a, b = sc.model.state(), sc.model.state()
        s.prime(a, 1.25e-3)
        n = []
        for _ in range(20):
            a, b = run(s, a, b, 1, 1.25e-3)
            n.append(int(s.report().iters_density[0]))
        its.append(np.mean(n[5:]))
    assert its[1] < its[0]
