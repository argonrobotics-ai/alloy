# SPDX-License-Identifier: Apache-2.0
"""SCR-SOFT-2-4: cloth, soft and MPM objects take their placement's rotation (``template.rotate_particles``: Newton's
``add_builder`` translates particles only). CPU: a 20 x 10 cm towel turned 90° about z and a 60 x 20 x 20 mm MPM paste
block turned 90° about z build with their x and y extents swapped and land on the table still turned; a block pitched
30° builds tilted 30°; a soft pad stood on its side rests like one built upright (rest shapes are relative); no
rotation leaves the particles untouched, bit for bit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.cards.types import Budget, Placement
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.scene.template import rotate_particles
from alloy.sim import Simulation
from alloy.testing import route as T
from alloy.testing import scene as S

TOP = 0.02                                                     # S.table() top [m]
YAW_90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
PITCH_30 = (0.0, math.sin(math.pi / 12), 0.0, math.cos(math.pi / 12))
ROLL_90 = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))


def _towel():
    mat = T.material("cotton", **T.COTTON, beta=1e-3)
    return T.box("towel", mat, 0.1, 0.05, 2.5e-4, budget=Budget(particles=231))


def _paste():
    mat = T.material("dough", density=1200.0, E=1.0e5, nu=0.45, fluid_mu=50.0, yield_stress=200.0)
    return T.box("dough", mat, 0.03, 0.01, 0.01)


def _foam():
    return T.box("pad", T.material("foam", **T.PU_FOAM, beta=1e-3), 0.02, 0.01, 0.005)


def _built(obj, quat, z):
    card, lib = S.scene([S.table(), obj], [S.place("table"), Placement(obj.card.name, (0.0, 0.0, z, *quat), ())],
                        worlds=1)
    return SceneAssembler(lib, Router()).assemble(card, "cpu")


def _points(built, name, q=None):
    rng = built.handles[name].template.particle
    q = built.model.particle_q.numpy() if q is None else q
    return np.asarray(q[rng.start:rng.stop], np.float64)


def _land(built, periods):
    sim = Simulation(built)
    for _ in range(periods):
        sim.step_period()
    return sim.state.particle_q.numpy()


def test_no_rotation_leaves_the_particles_untouched():
    b = newton.ModelBuilder()
    for p in ((0.1, 0.2, 0.3), (-0.4, 0.5, 0.6)):
        b.add_particle(p, (1.0, 0.0, 0.0), 1.0)
    before = (list(b.particle_q), list(b.particle_qd))
    rotate_particles(b, (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
    assert (b.particle_q, b.particle_qd) == before
    rotate_particles(b, (0.0, 0.0, 0.0, *YAW_90))
    np.testing.assert_allclose(b.particle_q, [(-0.2, 0.1, 0.3), (-0.5, -0.4, 0.6)], atol=1e-12)
    np.testing.assert_allclose(b.particle_qd, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)], atol=1e-12)


def test_pitched_mpm_block_builds_tilted():
    built = _built(_paste(), PITCH_30, TOP + 0.04)
    assert built.handles["dough"].decision.route in (Route.PASTE_MPM, Route.GRAIN_MPM)
    p = _points(built, "dough")
    axis = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)[2][0]      # the block's long axis
    tilt = math.degrees(math.atan2(abs(axis[2]), math.hypot(axis[0], axis[1])))
    assert tilt == pytest.approx(30.0, abs=0.5)


@pytest.mark.compile_heavy
def test_turned_towel_lands_turned():
    built = _built(_towel(), YAW_90, TOP + 2e-3)
    assert built.handles["towel"].decision.route == Route.CLOTH_VBD
    ext = np.ptp(_points(built, "towel"), axis=0)
    assert ext[0] == pytest.approx(0.1, rel=1e-5) and ext[1] == pytest.approx(0.2, rel=1e-5)
    landed = _points(built, "towel", _land(built, 10))
    ext = np.ptp(landed, axis=0)
    assert ext[1] > 1.9 * ext[0] and landed[:, 2].min() > TOP                 # on the table, still turned


@pytest.mark.compile_heavy
def test_turned_mpm_block_lands_turned():
    built = _built(_paste(), YAW_90, TOP + 0.012)
    ext = np.ptp(_points(built, "dough"), axis=0)
    assert ext[1] > 2.5 * ext[0]                                              # 60 mm now along y
    landed = _points(built, "dough", _land(built, 10))
    ext = np.ptp(landed, axis=0)
    assert ext[1] > 2.0 * ext[0] and landed[:, 2].min() > TOP - 1e-3


def _strain(built, periods: int = 10) -> np.ndarray:
    rest = _points(built, "pad")
    landed = _points(built, "pad", _land(built, periods))
    d0, d1 = (np.linalg.norm(p[:, None] - p[None], axis=-1) for p in (rest, landed))
    return np.abs(d1 - d0)[d0 > 0] / d0[d0 > 0]


@pytest.mark.compile_heavy
def test_turned_soft_solid_rests_like_one_built_upright():
    """Tet rest shapes are relative to the particles: a 40 x 20 x 10 mm pad stood on its side (90° about x) lands and
    strains like a 40 x 10 x 20 mm pad built upright with no rotation (same mesh size; the contact under the narrow
    footprint strains both ~3 %)."""
    turned = _built(_foam(), ROLL_90, TOP + 0.0105)
    assert turned.handles["pad"].decision.route == Route.SOFT_VBD
    ext = np.ptp(_points(turned, "pad"), axis=0)
    assert ext[2] == pytest.approx(0.02, rel=1e-5) and ext[1] == pytest.approx(0.01, rel=1e-5)
    upright = T.box("pad", T.material("foam", **T.PU_FOAM, beta=1e-3), 0.02, 0.005, 0.01)
    a, b = _strain(turned), _strain(_built(upright, (0.0, 0.0, 0.0, 1.0), TOP + 0.0105))
    assert a.max() == pytest.approx(b.max(), rel=0.05) and np.median(a) == pytest.approx(np.median(b), rel=0.05)
