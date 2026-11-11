# SPDX-License-Identifier: Apache-2.0
"""Surface-flow drops on the GPU (design §4.4, §4.7; WP9; light-test rule: one smoke). The harness check of a film that
sheds ballistic drops, with CUDA graphs; and a film shedding SPH particles at an open edge in 16 worlds, the film and
SPH steps captured in one CUDA graph and replayed, with every ledger exact in every world."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.host import PluginHost
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.solvers.sph.build import add_held_spares
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH

R, G, DT, RHO = 1.0e-3, 9.81, 5.0e-4, 998.2
OPEN_X = ("open", "wall", "wall", "wall")


def shedding_flow() -> SurfaceFlow:
    """The harness scene's tilted plate (gravity -1 m/s² along x) with its x- edge open: the film sheds 0.5 mm drops
    that fall to z = -0.5 m."""
    return SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 20), edges=OPEN_X),), material="water",
                       drops={"floor_z": -0.5, "drop_radius": 5.0e-4})


@pytest.mark.gpu
def test_harness_check_of_a_shedding_film_on_gpu():
    report = ap.check(shedding_flow(), device="cuda:0", laws=False)
    print(report.summary())
    assert report.passed, report.summary()


def _sph_scene(worlds: int, device: str):
    cfg = DFSPHConfig(particle_radius=R, world_bounds=((-0.2, -0.2, -0.1), (0.2, 0.2, 0.3)))
    b = newton.ModelBuilder(gravity=(-4.0, 0.0, -G))
    register_custom_attributes(b)
    for _ in range(worlds):
        b.begin_world()
        b.add_shape_box(-1, hx=0.05, hy=0.05, hz=0.01, label="plate")
        add_held_spares(b, cfg, 400, rest_density=RHO, park=(0.0, 0.0, -0.05))
        b.end_world()
    m = b.finalize(device=device)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 20), edges=OPEN_X),), material="water", sph=True,
                       drops={}, max_depth=3.0e-3)
    PluginHost(m, [flow], control_dt=0.01)
    solver = SolverDFSPH(m, cfg)
    flow.check_dt(DT)
    flow.set_depth(1.5e-3, chart=0)
    return m, flow, solver


def _two_steps(flow, solver, a, b) -> None:
    solver.step(a, b, None, None, DT)
    flow.update(a, b, DT)
    solver.step(b, a, None, None, DT)
    flow.update(b, a, DT)


@pytest.mark.gpu
def test_sph_shedding_replays_in_a_cuda_graph():
    """D-1, D-2 at 16 worlds: two eager steps (binding the port), then two steps captured (capture runs nothing) and
    replayed 99 times: 200 steps, equal to 200 eager steps. Each world sheds particles over its x- edge and books them
    exactly (film, drops, SPH's mass ledger)."""
    W, dev = 16, "cuda:0"
    m0, flow0, solver0 = _sph_scene(W, dev)
    a0, b0 = m0.state(), m0.state()
    for _ in range(100):
        _two_steps(flow0, solver0, a0, b0)
    m, flow, solver = _sph_scene(W, dev)
    a, b = m.state(), m.state()
    _two_steps(flow, solver, a, b)
    with wp.ScopedCapture(device=dev) as cap:
        _two_steps(flow, solver, a, b)
    for _ in range(99):
        wp.capture_launch(cap.graph)
    r, d, c = flow.read(), flow.drops.read(), solver.port.counts()
    assert np.array_equal(c["emitted"], solver0.port.counts()["emitted"])
    assert np.array_equal(r["spilled_volume"], flow0.read()["spilled_volume"])
    assert np.array_equal(a.particle_q.numpy(), a0.particle_q.numpy())
    vp, mp = solver.port.particle_volume, float(m.particle_mass.numpy()[0])
    made = c["emitted"]
    print("emitted per world", made.tolist())
    assert (made > 10).all()
    assert np.allclose(d["to_sph_volume"], made * vp, rtol=1e-12, atol=0.0)
    assert np.allclose(r["spilled_volume"], d["to_sph_volume"] + d["pending_volume"], rtol=1e-12, atol=0.0)
    led = solver.mass_ledger()
    assert np.allclose(led["in"] / led["scale"], made * mp, rtol=1e-6, atol=0.0)
    assert np.array_equal(led["active"] + led["out"], led["initial"] + led["in"])
    live = (solver.layout.phase.numpy() == 1) & (m.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE) != 0)
    assert live.sum() == made.sum() and (a.particle_q.numpy()[live, 0] < -0.05 + 3.0 * R).all()
