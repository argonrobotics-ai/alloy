# SPDX-License-Identifier: Apache-2.0
"""Gap #4 (cloth fold springback): KES bending hysteresis on VBD's hinges (alloy/solvers/vbd/hysteresis.py).

Law (CPU), KES-FB2: a pinned strip bent along exact arcs 0 -> +2.5 -> -2.5 -> 0 /cm, one VBD step per curvature.
Each bending hinge's moment m = k_tot (theta - rest) follows Kawabata's M = B K + HB sgn(dK): rising, the slope is the
plate map's k l_h = HINGE_C B; at K = +-1 /cm the up and down branches differ by 2 hb = HINGE_C 2HB. The plate map's
factor cancels in the ratio: loop width / slope = 2HB / B, the card's own pair. Released at 2.5 /cm, the elastic
moment falls to HB, so the hinge keeps the curvature HB / B. Unmarked edges: the stock hinge, nothing allocated.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import SolverVBDAlloy
from alloy.solvers.vbd import cloth_profile as CP
from alloy.solvers.vbd import hysteresis as H

B = 1.27e-6           # N m: cotton single jersey, KES B 0.013 gf cm2/cm
HB2 = 3.5e-4          # N: KES 2HB 0.036 gf cm/cm (moment per unit width)
NX, NY, CELL = 20, 2, 0.005
K_MAX, K_STEP = 250.0, 2.5      # 1/m: KES-FB2 bends to +-2.5 /cm


def _strip(hb2: float | None):
    b = newton.ModelBuilder()
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=NX,
                     dim_y=NY, cell_x=CELL, cell_y=CELL, mass=0.0, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0)
    CP.rewrite_hinges(b, 0, 0, B, 0.0)
    if hb2:
        assert H.mark_edges(b, 0, len(b.edge_indices), CP.HINGE_C * 0.5 * hb2) > 0
    b.color()
    return b.finalize(device="cpu")


def _arc(q0: np.ndarray, kappa: float) -> np.ndarray:
    """The flat strip wrapped isometrically onto a cylinder of curvature kappa about y (x along its length)."""
    if kappa == 0.0:
        return q0.copy()
    r = 1.0 / kappa
    s = q0[:, 0] - q0[:, 0].mean()
    return np.stack([r * np.sin(s / r), q0[:, 1], r * (1.0 - np.cos(s / r))], 1).astype(np.float32)


def _angles(q: np.ndarray, ei: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3 = (q[ei[:, k]] for k in range(4))
    n1, n2 = np.cross(x2 - x0, x3 - x0), np.cross(x3 - x1, x2 - x1)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
    e = (x3 - x2) / np.linalg.norm(x3 - x2, axis=1, keepdims=True)
    return np.arctan2((np.cross(n1, n2) * e).sum(1), (n1 * n2).sum(1))


def _loop(model) -> tuple[np.ndarray, np.ndarray]:
    """(K schedule, mean hinge moment per bending hinge) through the KES loop, one VBD step per curvature."""
    solver = SolverVBDAlloy(model, iterations=1)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    q0 = model.particle_q.numpy().astype(np.float64)
    ei = model.edge_indices.numpy()
    inner = (ei[:, 0] >= 0) & (ei[:, 1] >= 0)
    up = np.arange(0.0, K_MAX + 1e-9, K_STEP)
    ks = np.concatenate([up, up[::-1][1:], -up[1:], -up[::-1][1:]])
    top = _angles(_arc(q0, K_MAX), ei)
    probe = np.abs(top) > 1e-3                                          # the hinges this bend turns
    sign = np.sign(top)                                                 # VBD's angle sign for this bend direction
    mom = []
    for k in ks:
        q = _arc(q0, float(k))
        s0.particle_q.assign(q)
        s1.particle_q.assign(q)
        solver.step(s0, s1, ctrl, None, 1.0e-3)
        th = _angles(q.astype(np.float64), ei)
        rest = model.edge_rest_angle.numpy().astype(np.float64)
        ke = model.edge_bending_properties.numpy()[:, 0].astype(np.float64)
        sel = inner & probe
        mom.append(float(np.mean(sign[sel] * ke[sel] * (th[sel] - rest[sel]))))   # + along the bend
    return ks, np.asarray(mom)


def test_unmarked_edges_keep_the_stock_hinge():
    model = _strip(None)
    assert H.BendHysteresis.for_model(model) is None and SolverVBDAlloy(model, iterations=1).hysteresis is None


def test_the_kes_loop_is_2hb_wide_and_b_steep():
    ks, m = _loop(_strip(HB2))
    n = len(ks) // 4
    rise, fall = slice(0, n + 1), slice(n, 2 * n + 1)
    at = lambda sl, k: float(np.interp(k, ks[sl][::1 if ks[sl][0] < ks[sl][-1] else -1],      # noqa: E731
                                       m[sl][::1 if ks[sl][0] < ks[sl][-1] else -1]))
    slope = (at(rise, 150.0) - at(rise, 50.0)) / 100.0                # N per (1/m): HINGE_C B l_h / l_h
    width = at(rise, 100.0) - at(fall, 100.0)                         # N: 2 hb = HINGE_C 2HB
    assert width / slope == pytest.approx(HB2 / B, rel=0.10), (width, slope)
    held = float(np.interp(0.0, ks[fall][::-1], m[fall][::-1]))      # unloaded to K = 0 from the top
    assert held == pytest.approx(-0.5 * CP.HINGE_C * HB2, rel=0.10)   # the friction element holds -HB at K = 0


def test_released_hinge_keeps_hb_over_b():
    """Zero moment on the way down sits at K = HB / B: the curvature a released fold keeps."""
    ks, m = _loop(_strip(HB2))
    n = len(ks) // 4
    fall_k, fall_m = ks[n:2 * n + 1][::-1], m[n:2 * n + 1][::-1]
    k0 = float(np.interp(0.0, fall_m, fall_k))
    assert k0 == pytest.approx(0.5 * HB2 / B, rel=0.10), k0


def test_a_card_with_2hb_routes_hysteretic_hinges_and_reads_back():
    """The route: a cloth card with elastic.bending_hysteresis gets the friction moment on every bending hinge of
    every world, its hinges at (1 + LOCK) k (the plate map's readback agrees); a card without it is unchanged."""
    import dataclasses

    from alloy.cards.types import Q
    from alloy.core.registry import isolated_registry
    from alloy.integration.wiring import build_simulation
    from alloy.route.router import Router
    from alloy.scene.assembler import SceneAssembler
    from alloy.testing import scene as S

    towel = S.cloth()
    el = dataclasses.replace(towel.material.elastic, bending_hysteresis=Q(HB2, "model", note="test"))
    knit = dataclasses.replace(towel, material=dataclasses.replace(towel.material, elastic=el))
    card, lib = S.scene([knit], [S.place(knit.card.name, z=0.3)], worlds=2)
    with isolated_registry(empty=True):
        built = SceneAssembler(lib, Router()).assemble(card, "cpu")
        sim = build_simulation(built, "standard", seed=0)
        (vbd,) = [s for s in sim.solvers if isinstance(s, SolverVBDAlloy)]
        hb = built.model.cloth.bend_hb.numpy()
        ei = built.model.edge_indices.numpy()
        inner = (ei[:, 0] >= 0) & (ei[:, 1] >= 0)
        assert vbd.hysteresis is not None and vbd.hysteresis.count == int(inner.sum())
        assert np.allclose(hb[inner], CP.HINGE_C * 0.5 * HB2) and not hb[~inner].any()
        sim.step_period()
        assert np.isfinite(sim.state.particle_q.numpy()).all()


def test_the_hinge_state_is_snapshot_state():
    """The adapters list both per-edge arrays (whole-entry and per-world, each edge's world), and restoring them
    re-runs a stretch of the KES loop bit for bit (a coupled rollback restores the crease with the particles)."""
    from alloy.coupling.adapters import VBD_ARRAYS, VBD_EDGE, vbd_world_maps

    model = _strip(HB2)
    solver = SolverVBDAlloy(model, iterations=1)
    maps = vbd_world_maps(solver)                                       # keys: "vbd.<array>"
    assert set(VBD_EDGE) <= set(VBD_ARRAYS) and {f"vbd.{k}" for k in VBD_EDGE} <= set(maps)
    assert all(maps[f"vbd.{k}"].shape[0] == model.edge_count for k in VBD_EDGE)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    q0 = model.particle_q.numpy().astype(np.float64)

    def bend(ks):
        for k in ks:
            q = _arc(q0, float(k))
            s0.particle_q.assign(q)
            s1.particle_q.assign(q)
            solver.step(s0, s1, ctrl, None, 1.0e-3)
        return model.edge_rest_angle.numpy().copy()

    bend(np.arange(0.0, 200.0, K_STEP))
    saved = {k: getattr(solver, k).numpy().copy() for k in VBD_EDGE}
    first = bend(np.arange(200.0, 50.0, -K_STEP))
    for k, v in saved.items():
        getattr(solver, k).assign(v)
    np.testing.assert_array_equal(bend(np.arange(200.0, 50.0, -K_STEP)), first)
