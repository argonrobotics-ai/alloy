# SPDX-License-Identifier: Apache-2.0
"""CARD-1 (SB-9 v1): library objects that run from their card alone and pass their analytic law or self-test, over
the object count, target ≥ 80 % (`alloy.scene.coverage`). The full measurement is the nightly GPU job; the CPU
tests cover the pieces it rests on (rest pose, Navier series, the contact stability cap, one object end to end)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.scene import coverage as C
from alloy.scene import coverage_checks as CK
from alloy.scene import coverage_pose as P
from alloy.scene import native_gains as NG
from alloy.testing import scene as S


def _lib():
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    return CardLibrary(assets=None)      # Alloy's library only: DC asset cards are tested by their owners (WP1, SCR-DC-1)


def test_navier_series_isotropic_matches_table():
    # TWK table 8: w = 0.00406·q·a⁴/D for a simply supported square plate
    assert CK.navier_centre(1.0, 1.0, 1.0, 1.0, 1.0) == pytest.approx(0.00406, rel=2e-3)
    a = CK.sheet_side(2.0, 1e-3, 0.5)
    assert CK.navier_centre(2.0, a, 0.5, 0.5, 0.5) / 1e-3 == pytest.approx(CK.SHEET_W_OVER_H, rel=3e-3)


def test_stability_cap_is_mujoco_tau_rule_at_critical_damping():
    dt = 1.25e-3
    ke = NG.stable_ke(1.0, dt)
    assert 2.0 / (2.0 * math.sqrt(ke)) == pytest.approx(2.0 * dt, rel=1e-12)       # τ = 2/kd = 2·dt
    for zeta in (0.0, 0.2155, 0.5, 1.0):                                              # inside the stability region
        k = NG.stable_ke(zeta, dt)
        assert dt * dt * k / NG.DEFAULT_IMP + 4.0 * zeta * dt * math.sqrt(k) < 4.0


def test_rest_pose_tipping_rule():
    lib = _lib()
    pencil = P.rest_pose(lib.object("pencil_wood"), lib, 0.009, C.DROP_M)      # 2.4° on its end: laid down
    assert pencil[3:] != P.UP and pencil[2] == pytest.approx(0.009 + 0.0035 + C.DROP_M)
    bolt = P.rest_pose(lib.object("bolt_m10x40"), lib, 0.009, C.DROP_M)        # stands on its head
    assert bolt[3:] == P.ORIENTATIONS[1]
    cube = P.rest_pose(lib.object("cube_beech_30"), lib, 0.009, C.DROP_M)      # ties keep the authored pose
    assert cube[3:] == P.UP and cube[2] == pytest.approx(0.009 + 0.015 + C.DROP_M)


@pytest.mark.compile_heavy
@pytest.mark.timeout(300)
def test_light_cube_rests_on_the_table():
    """Regression: a 19 g beech cube (e = 0.5) bounced off plywood until the gains were capped for stability."""
    row = C.measure_object("cube_beech_30", _lib(), "cpu", dict(CK.CHECKS), worlds=2)
    assert row.covered, row


def _pin_route(lib, name: str, route: str) -> None:
    """Make `lib` hand out object `name` with a card route override (a woven box sheet now defaults to cloth_vbd)."""
    import dataclasses

    from alloy.cards import types as ct

    for method in ("object", "peek_object"):
        get = getattr(lib, method)
        pinned = {}

        def _get(n, *a, _get=get, _pinned=pinned, **k):
            card = _get(n, *a, **k)
            if n != name:
                return card
            return _pinned.setdefault(n, dataclasses.replace(card, route=ct.RouteOverride(route, "test: pinned route")))

        setattr(lib, method, _get)


def _built(name: str, then=None, route: str | None = None):
    """Assemble `name` alone on the table (1 world); `then(built)` runs inside the same solver registry. `route`
    pins the object's route with a card override."""
    from alloy.core.registry import isolated_registry
    from alloy.route.router import Router
    from alloy.scene.assembler import SceneAssembler
    from alloy.scene.builtin_plugins import register_builtin_solvers

    lib = _lib()
    if route is not None:
        _pin_route(lib, name, route)
    card, _ = C.scene_for(name, lib, 1)
    with isolated_registry(empty=True):
        register_builtin_solvers()
        built = SceneAssembler(lib, Router()).assemble(card, "cpu", world_count=1)
        return built, (then(built) if then is not None else None)


@pytest.mark.compile_heavy
@pytest.mark.timeout(300)
def test_style3d_napkin_steps_finite():
    """Regression: the generic cloth path left style3d:edge_rest_area at 0 and the first Style3D step was NaN."""
    from alloy.sim import Simulation

    def _step(built):
        sim = Simulation(built)
        for _ in range(3):
            sim.step_period()
        return sim.state.particle_q.numpy()

    built, q = _built("napkin_cotton_aniso", _step, route="cloth_style3d")
    assert float(built.model.style3d.edge_rest_area.numpy().min()) > 0.0
    assert np.isfinite(q).all()


@pytest.mark.compile_heavy
@pytest.mark.timeout(300)
def test_mjcf_object_carries_card_surface_and_multiccd():
    """Regression: hinged_box_plywood's MJCF shapes kept MJCF's μ = 1 (readback refused the 0.3 card)."""
    built, _ = _built("hinged_box_plywood")
    lib = _lib()
    mat = lib.material(lib.object("hinged_box_plywood").material)
    h = built.handles["hinged_box_plywood"]
    mu = built.model.shape_material_mu.numpy()[h.template.shape.start:h.template.shape.stop]
    assert np.allclose(mu, float(mat.surface.mu_dynamic.value))
    assert all(e.solver_kwargs.get("enable_multiccd") for e in built.entries if e.capability == "mujoco")


@pytest.mark.compile_heavy
@pytest.mark.timeout(300)
def test_ccd_contacts_take_the_resting_floor():
    """Regression (CARD-1 on an L4): mesh nuts, a USB plug and probe cubes on drilled blocks kept ζ(e) ≈ 0.2 and
    walked 0.1-2.9 mm per 0.25 s at rest. A pair with a CCD-path object gets a row at ζ ≥ ZETA_CCD; an analytic box on
    plywood takes the multi-point floor ZETA_MULTIPOINT (WP3 2026-11-07: at ζ(e) 0.215 a landed box never settles and
    walks down slopes, tests/contact/test_resting_floor.py); friction rows stay (0.02 s, 1) (stiffer ones made a peg
    walk on GPU)."""
    zeta = {}
    for name in ("nut_m6", "cube_beech_30"):
        built, _ = _built(name)
        mj = built.model.mujoco
        zeta[name] = mj.pair_solref.numpy()[:, 1]
        assert len(zeta[name]) and np.allclose(mj.pair_solreffriction.numpy(), (0.02, 1.0))
    assert zeta["nut_m6"].min() >= NG.ZETA_CCD - 1e-6
    assert zeta["cube_beech_30"].min() >= NG.ZETA_MULTIPOINT - 1e-6   # beech e = 0.5: ζ(e) 0.215, floored
    assert NG.restitution_cap(NG.ZETA_CCD) == pytest.approx(0.163, abs=1e-3)


@pytest.mark.gpu
@pytest.mark.nightly
@pytest.mark.timeout(6 * 3600)
def test_card1_coverage():
    import warp as wp

    lib = _lib()
    device = str(wp.get_cuda_device())

    def _row(r: C.CoverageRow) -> None:
        S.record("CARD-1.object", r.value, "pass" if r.covered else "fail", object=r.object, route=r.route,
                 check=r.check, runs=r.runs, bound=r.bound, detail=r.detail, wall_s=round(r.wall_s, 1), device=device)

    rep = C.measure(lib.names("objects"), lib, device, on_row=_row)
    misses = [(r.object, r.route, r.check, r.detail) for r in rep.rows if not r.covered]
    S.record("CARD-1.coverage", rep.rate, "pass" if rep.rate >= C.TARGET else "fail", covered=rep.covered,
             n=len(rep.rows), by_route=rep.by_route(), misses=misses, device=device)
    assert len(rep.rows) == len(lib.names("objects")) and np.isfinite(rep.rate)
    assert rep.rate >= C.TARGET, f"{rep.covered}/{len(rep.rows)}; not covered: {misses}"
