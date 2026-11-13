# SPDX-License-Identifier: Apache-2.0
"""MPM entries project particles out of colliders after every step (``alloy.integration.mpm_project``; WP12-SOIL-1).

Through 0.1.2 a sand grain that got under the ground plane was never pushed back and fell for good: on one L4 the
grains soil_dig spilled went under the ground in every tree and mode, ``sim.check()`` ok until one passed 20 m/s.
CPU laws:
- a grain put half a voxel under the ground comes back to the surface within one control period; with
  ``ALLOY_MPM_PROJECT=off`` (Newton's step alone) it stays under;
- every MPM entry gets the wrapper once (a build note), and ``install`` never wraps a solver twice.

    PYTHONPATH=. python -m pytest tests/integration/test_mpm_project.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.integration import mpm_project as P

pytestmark = pytest.mark.integration


def _slab() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("sand", size=(0.02, 0.02, 0.01), pos=(0.0, 0.0, 0.005), material="sand_silica_dry")
    s.add_box("cube", size=(0.02, 0.02, 0.02), pos=(0.3, 0.0, 0.01), mass=0.1)
    return s


def _bury_one(sim: alloy.Sim) -> tuple[int, float]:
    """Move the slab's outermost corner grain half a voxel under the ground, at rest, in both of the engine's
    states; returns (its particle index, the depth put in, m)."""
    eng = sim.engine
    pids = np.asarray(sorted(i for e in eng.built.entries if e.capability == "mpm" for i in e.particles), np.int64)
    q = eng.state.particle_q.numpy()
    i = int(pids[np.argmax(np.linalg.norm(q[pids, :2], axis=1))])
    solver = next(s for s in eng.parts.solvers.values() if hasattr(s, "_mpm_model"))
    depth = 0.5 * float(solver._mpm_model.voxel_size)
    for st in (eng.state, getattr(eng, "_next", None)):
        if st is None:
            continue
        sq, sqd = st.particle_q.numpy(), st.particle_qd.numpy()
        sq[i, 2], sqd[i] = -depth, 0.0
        st.particle_q.assign(sq)
        st.particle_qd.assign(sqd)
    return i, depth


@pytest.mark.timeout(300)
@pytest.mark.parametrize("project", ["on", "off"])
def test_a_grain_under_the_ground_comes_back_to_the_surface(project: str, monkeypatch) -> None:
    monkeypatch.setenv(P.ENV, project)
    sim = alloy.build(_slab(), device="cpu", seed=0)
    assert any("projected out of colliders" in n for n in sim.engine.parts.notes) == (project == "on")
    i, depth = _bury_one(sim)
    sim.step(1)
    z = float(sim.engine.state.particle_q.numpy()[i, 2])
    if project == "on":
        assert z >= -0.25 * depth, (z, depth)
    else:
        assert z < -0.5 * depth, (z, depth)                       # Newton's step alone leaves it under


def test_install_wraps_a_solver_once() -> None:
    calls: list[str] = []

    class Fake:
        def step(self, state_in, state_out, control, contacts, dt):
            calls.append("step")

        def project_outside(self, state_in, state_out, dt):
            calls.append("project")

    solver = Fake()
    assert P.install(solver) and not P.install(solver)
    solver.step(None, "out", None, None, 0.01)
    assert calls == ["step", "project"]
    assert not P.install(object())                                 # nothing to project with: left alone
