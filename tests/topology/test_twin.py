# SPDX-License-Identifier: Apache-2.0
"""TOP-2 twin on synthetic data (real rig data: not_run, SPEC §5).

A synthetic dataset is generated from the twin at known parameters, with 1 % noise (``synthetic_rig``). Then:
- the twin at the same parameters matches it within the 15 % TOP-2 bound;
- the twin with a clearly wrong cut parameter (``OFF``) misses it, so the comparison can tell parameters apart.
Forces are positive, and identical samples give bitwise-identical forces (multi-world repeatability on CPU).
"""

from __future__ import annotations

import numpy as np

from alloy.testing.fakes import synthetic_rig
from alloy.testing.topology import run_twin
from alloy.topology.laws import TOP2_KNIFE_FORCE
from alloy.topology.twin import LAW_ID, KnifeForceTwin

# The true and a clearly wrong set of cut parameters (``KnifeForceTwin`` CUT_KEYS). OFF cuts 4x faster: the twin
# must tell damage rates apart (docs/notes/cleanroom-cut-log.md D10).
TRUTH: dict[str, float] = {"softness": 1.0, "cut_spring_ke": 1000.0}
OFF: dict[str, float] = {"softness": 4.0, "cut_spring_ke": 1000.0}


def _rel_errors(sim: dict[str, np.ndarray], world: int, real: dict[str, float]) -> dict[str, float]:
    return {k: abs(sim[k][world] - real[k]) / abs(real[k]) for k in real}


def test_twin_matches_synthetic_data_and_rejects_wrong_params(device):
    twin = KnifeForceTwin()
    cond = twin.conditions(quick=True)[0]
    steps = int(TOP2_KNIFE_FORCE.duration_s / TOP2_KNIFE_FORCE.dt)
    sim = run_twin(twin, cond, [TRUTH, TRUTH, OFF], dt=TOP2_KNIFE_FORCE.dt, steps=steps, device=device)
    truth = {k: float(v[0]) for k, v in sim.items()}
    data = synthetic_rig(LAW_ID, TRUTH, {k: 0.01 * abs(v) for k, v in truth.items()}, conditions=[cond],
                         model=lambda p, c: truth, seed=3)
    valid = data.split("valid")
    real = {k: float(np.mean([t.scalars[k] for t in valid])) for k in truth}
    same, off = _rel_errors(sim, 1, real), _rel_errors(sim, 2, real)
    bound = TOP2_KNIFE_FORCE.metrics["force_d1mm"].rel
    print(f"TOP-2 synthetic: truth={ {k: round(v, 4) for k, v in truth.items()} } "
          f"same_err={ {k: round(v, 4) for k, v in same.items()} } off_err={ {k: round(v, 3) for k, v in off.items()} }")
    assert all(v > 0.0 for v in truth.values())
    for k in truth:
        np.testing.assert_array_equal(sim[k][0], sim[k][1])
    assert max(same.values()) <= bound
    assert max(off.values()) > bound
