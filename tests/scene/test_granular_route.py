# SPDX-License-Identifier: Apache-2.0
"""CARD-9 (SPEC §5.3): granular column-collapse runout at a = 2 vs 2.4, within 10 %, on Alloy's default grain route.

Quasi-2D collapse (Lube et al. 2005, Phys. Rev. E 72, 041301): (L_inf - L0) / L0 = 1.2 a for a < 2.3, so 2.4 at a = 2.
A sand card (ParticleSource box, phi 33 deg) of length L0 and height 2 L0 stands against a rough back wall at x = 0,
between frictionless side walls 2 voxels from its mid-plane (bench G01's slab), on a rough floor; the back wall and
floor carry the sand's own friction tan(phi). Nothing holds its open face, so it collapses at t = 0. The router picks
grain_mpm and its voxel (min(d_feature / 4, mpm_voxel_max) = 5 mm here), ``emit_mpm`` fills it and
``build_simulation`` steps it alone. The runout front is the 99.5th percentile of particle x (bench G01's convention,
so a few stray particles do not count). The thinner "toe" front, the last column of the deposit at least one voxel
thick, is recorded beside it.

Measured (CPU, wp12-results "CARD-9"): 2.371 at 5 mm and 2.330 at 2.5 mm voxels (toe 1.90 / 2.00).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.cards.types import ParticleSource
from alloy.core.registry import isolated_registry
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.testing.integration import assemble, record

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.timeout(1800)]

ASPECT, L0, PHI = 2.0, 0.05, math.radians(33.0)
VOXEL = 5.0e-3                  # the route's mpm_voxel_max, which the slab's 20 mm width selects
SETTLE_S = 0.7                  # the a = 2 deposit stops by ~0.5 s (5 mm and 2.5 mm voxels)
WALL = 0.01                     # wall half-thickness [m]
CMD = "PYTHONPATH=. python -m pytest tests/scene/test_granular_route.py -m slow"


def lube_runout(a: float) -> float:
    """Lube et al. 2005 quasi-2D normalized runout (L_inf - L0) / L0."""
    return 1.2 * a if a < 2.3 else 1.9 * a ** (2.0 / 3.0)


def collapse_scene(a: float = ASPECT, l0: float = L0, voxel: float = VOXEL, phi: float = PHI):
    """(card, lib) of the quasi-2D collapse (module docstring); 1 world."""
    h, w = a * l0, 2.0 * voxel
    lx = l0 * (1.0 + lube_runout(a)) * 1.8
    sand = T.obj("sand", ParticleSource(((0.0, -w, 0.0), (l0, w, h)), 0.5 * voxel, grain_diameter=1e-4),
                 T.material("sand", density=1600.0, E=1e7, nu=0.3, friction_angle=phi))
    rough = T.material("rough", density=600.0, E=1e10, mu=math.tan(phi))
    smooth = T.material("smooth", density=600.0, E=1e10, mu=0.0)
    objs = [T.box("floor", rough, 0.5 * lx + WALL, w + WALL, WALL, motion="static"),
            T.box("back", rough, WALL, w + WALL, 0.5 * h + WALL, motion="static"),
            T.box("side_p", smooth, 0.5 * lx, WALL, 0.5 * h + WALL, motion="static"),
            T.box("side_n", smooth, 0.5 * lx, WALL, 0.5 * h + WALL, motion="static"), sand]
    places = [S.place("floor", x=0.5 * lx - WALL, z=-WALL), S.place("back", x=-WALL, z=0.5 * h),
              S.place("side_p", x=0.5 * lx, y=w + WALL, z=0.5 * h),
              S.place("side_n", x=0.5 * lx, y=-w - WALL, z=0.5 * h), S.place("sand")]
    return S.scene(objs, places, worlds=1)


def deposit_fronts(p: np.ndarray, voxel: float, l0: float = L0) -> dict[str, float]:
    """Normalized runouts from the p99.5 front and from the toe (last x column at least one voxel thick)."""
    x, z = p[:, 0], p[:, 2]
    front = float(np.percentile(x, 99.5))
    edges = np.arange(0.0, float(x.max()) + voxel, voxel)
    top = np.full(len(edges), -1.0)
    np.maximum.at(top, np.digitize(x, edges) - 1, z)
    thick = np.nonzero(top >= voxel)[0]
    toe = float(edges[thick[-1]] + voxel) if len(thick) else 0.0
    return {"runout": (front - l0) / l0, "runout_toe": (toe - l0) / l0}


def test_card9_column_collapse_runout():
    from alloy.integration import build_simulation

    card, lib = collapse_scene()
    with isolated_registry(empty=True):
        built = assemble(card, lib, "cpu", 1)
        decision = built.report.decisions["sand"]
        assert str(decision.route) == "grain_mpm"
        assert float(decision.numbers["voxel"]) == pytest.approx(VOXEL)
        sim = build_simulation(built, "standard", graph="off")
        for _ in range(int(round(SETTLE_S / built.control_dt))):
            sim.step_period()
        speed = np.linalg.norm(sim.state.particle_qd.numpy(), axis=1)
        fronts = deposit_fronts(sim.state.particle_q.numpy(), VOXEL)
    law = lube_runout(ASPECT)
    err = fronts["runout"] / law - 1.0
    record("CARD-9", fronts["runout"], "pass" if abs(err) <= 0.10 else "fail", CMD, law=law, rel_error=err,
           runout_toe=fronts["runout_toe"], voxel=VOXEL, aspect=ASPECT, speed_p99=float(np.percentile(speed, 99)))
    print(f"CARD-9 runout {fronts['runout']:.3f} vs {law:.2f} ({err:+.1%}); toe {fronts['runout_toe']:.2f}")
    assert np.percentile(speed, 99) < 1e-2, "the deposit has not settled"
    assert abs(err) <= 0.10


def test_a_rod_beside_paste_sets_no_mpm_voxel() -> None:
    """WP13's combination sweep F11: a rod partner's middle extent is its diameter, and taken as a container inner
    size it set the paste's voxel to a quarter of it (a 4 mm lace beside 150 ml of peanut butter: 1,041,859 MPM
    particles, over the budget). A curve card is no container. Nor, since SOFT-2 (1a23be349), is a moving solid
    primitive (a free or kinematic box: a tool or a part, its middle extent is its own width); a static box (a
    container's wall) still bounds the voxel."""
    from alloy.cards.types import CurveSource  # noqa: PLC0415
    from alloy.core.registry import registered_routes  # noqa: PLC0415
    from alloy.route.router import Router  # noqa: PLC0415
    from alloy.route.types import Route  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

    paste = T.obj("paste", ParticleSource(((0, 0, 0), (0.06, 0.06, 0.04)), 0.0025),
                  T.material("pb", density=1100.0, E=1e5, yield_stress=300.0))
    lace = T.obj("lace", CurveSource(((0.0, 0.0, 0.05), (0.3, 0.0, 0.05)), 0.002), T.material("cotton", density=500.0))
    block = T.box("block", T.material("abs", density=1000.0, E=2e9), 0.004, 0.004, 0.004, motion="static")  # 8 mm
    cube = T.box("cube", T.material("abs", density=1000.0, E=2e9), 0.004, 0.004, 0.004)                   # free, 8 mm
    with isolated_registry(empty=True):
        register_builtin_solvers()
        routes, router = registered_routes(), Router()
        alone = router.route(paste, T.context(routes))
        rod = router.route(paste, T.context(routes, touching={"paste": ("lace",)}, partners={"lace": lace}))
        box = router.route(paste, T.context(routes, touching={"paste": ("block",)}, partners={"block": block}))
        free = router.route(paste, T.context(routes, touching={"paste": ("cube",)}, partners={"cube": cube}))
    assert alone.route == rod.route == Route.PASTE_MPM
    assert rod.numbers["voxel"] == alone.numbers["voxel"] == pytest.approx(0.005)
    assert box.numbers["voxel"] == pytest.approx(0.002)                 # a quarter of the static cube's 8 mm
    assert free.numbers["voxel"] == pytest.approx(0.005)                # a free cube is no container (SOFT-2)
