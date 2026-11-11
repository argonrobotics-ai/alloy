# SPDX-License-Identifier: Apache-2.0
"""SCR-DC3-5 part 2: explicit MuJoCo pair rows honour geom solimp (flat = D22) and rows touching an armature body
are resized from MuJoCo's body_invweight0 after the entry is built; SCR-DC3-8: rows of a body whose mass is drawn per
world keep the template's force-space stiffness in every world (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.mjw_pairs import use_mujoco_pair_order
from alloy.contact.pair_fit import FitRows, fit_pair_rows, refit_solref
from alloy.scene import mujoco_pairs as MP
from alloy.scene.native_gains import DEFAULT_IMP

FLAT = (0.99, 0.99, 0.001, 0.5, 2.0)
K, ZETA = 2.0e5, 1.0


def _builder() -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    return b


def _plug_with_pad(b: newton.ModelBuilder, armature: float) -> tuple[int, int, int, int]:
    """A 22 g free plug with a 0.2 g pad on an x slide (armature) and a static blade box beside the pad."""
    plug = b.add_link(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), mass=0.022)
    b.add_shape_box(plug, hx=0.02, hy=0.01, hz=0.01)
    root = b.add_joint_free(plug, parent_xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    pad = b.add_link(xform=wp.transform((0.03, 0.0, 0.1), wp.quat_identity()), mass=0.0002)
    sp = b.add_shape_box(pad, hx=0.002, hy=0.002, hz=0.002)
    j = b.add_joint_prismatic(plug, pad, parent_xform=wp.transform((0.03, 0.0, 0.0), wp.quat_identity()),
                              axis=newton.Axis.X, armature=armature)
    b.add_articulation([root, j])
    blade = b.add_shape_box(-1, xform=wp.transform((0.0375, 0.0, 0.1), wp.quat_identity()), hx=0.0025, hy=0.01,
                            hz=0.01)
    return plug, pad, sp, blade


def _row(b: newton.ModelBuilder, s0: int, s1: int, fit: bool) -> None:
    values = {"mujoco:pair_world": b.current_world, "mujoco:pair_geom1": s0, "mujoco:pair_geom2": s1,
              "mujoco:pair_condim": 3, "mujoco:pair_solref": (0.02, 1.0), "mujoco:pair_solimp": MP.DEFAULT_SOLIMP}
    if fit:
        values.update(MP._fit_values(b, K, ZETA, DEFAULT_IMP, (None, "implicitfast")))
    elif "alloy:pair_k" in b.custom_attributes:
        values.update({f"alloy:{n}": 0.0 for n in MP.FIT_ATTRS})
    b.add_custom_values(**values)


def _fit(n: int = 1, zeta: float = ZETA, cap: float = 0.0, m0: float = 0.0, tc_min: float = 0.0) -> FitRows:
    return FitRows(*(np.full(n, v) for v in (K, zeta, cap, m0, tc_min)))


def _imp(n: int = 1) -> np.ndarray:
    return np.full(n, DEFAULT_IMP)


def _ke(solref: np.ndarray) -> np.ndarray:
    return 1.0 / (solref[..., 0] * solref[..., 1]) ** 2


def test_refit_solref_realises_k_at_the_mujoco_weight():
    w = np.array([5000.0, 201.5])
    ref = refit_solref(_fit(2), _imp(2), w, np.zeros(2))
    np.testing.assert_allclose(_ke(ref), K * (1.0 - DEFAULT_IMP) * w, rtol=1e-12)
    capped = refit_solref(_fit(cap=100.0, tc_min=1.0), _imp(), w[:1], np.full(1, 0.01))
    assert capped[0, 0] == pytest.approx(0.1)          # ke = cap = 100 -> tau = 1/sqrt(100); tc_min is for drawn rows


def test_drawn_rows_keep_the_template_force_stiffness_at_the_cap():
    """SCR-DC3-8: a capped row of a drawn body takes cap·m0/m, so ke·m (force-space stiffness) is the template's;
    the timeconst stays >= tc_min (MuJoCo's 2·dt clamp)."""
    rows, w, m = _fit(3, zeta=0.5, cap=100.0, m0=0.02), np.full(3, 5000.0), np.array([0.02, 0.016, 0.04])
    ke = _ke(refit_solref(rows, _imp(3), w, m))
    np.testing.assert_allclose(ke * m, 100.0 * 0.02, rtol=1e-12)
    floored = refit_solref(_fit(3, zeta=0.5, cap=100.0, m0=0.02, tc_min=0.2), _imp(3), w, m)
    np.testing.assert_allclose(floored[:, 0], [0.2, 0.2, 0.2 * np.sqrt(2.0)], rtol=1e-12)   # 0.02/0.016: 0.179 s
    undrawn = refit_solref(_fit(3, zeta=0.5, cap=100.0, tc_min=0.2), _imp(3), w, m)
    np.testing.assert_allclose(_ke(undrawn), 100.0, rtol=1e-12)


def test_default_geoms_keep_the_default_row_solimp():
    b = _builder()
    _, _, sp, blade = _plug_with_pad(b, 0.1)
    assert MP._row_solimp(b, sp, blade) == (MP.DEFAULT_SOLIMP, DEFAULT_IMP, False)


def test_flat_geom_solimp_reaches_the_row_others_keep_the_default():
    b = _builder()
    _, _, sp, blade = _plug_with_pad(b, 0.1)
    b.custom_attributes["mujoco:geom_solimp"].values[sp] = FLAT
    b.custom_attributes["mujoco:geom_solmix"].values[blade] = MP.STATIC_SOLMIX    # static shape: the pad governs
    solimp, imp, flat = MP._row_solimp(b, sp, blade)
    assert flat and imp == pytest.approx(0.99, abs=1e-5)
    b.custom_attributes["mujoco:geom_solimp"].values[sp] = (0.99, 0.999, 1e-4, 0.5, 2.0)   # DC4 PSU faces
    assert MP._row_solimp(b, sp, blade) == (MP.DEFAULT_SOLIMP, DEFAULT_IMP, False)   # non-flat: the default row
    b.custom_attributes["mujoco:geom_priority"].values[blade] = 1                 # higher priority wins outright
    assert MP._row_solimp(b, sp, blade)[0] == pytest.approx(MP.DEFAULT_SOLIMP, abs=1e-6)


def test_armature_bodies():
    b = _builder()
    plug, pad, _, _ = _plug_with_pad(b, 0.1)
    assert MP.armature_bodies(b) == frozenset({pad})
    b2 = _builder()
    _plug_with_pad(b2, 0.0)
    assert MP.armature_bodies(b2) == frozenset()


def test_fit_resizes_only_recorded_rows_from_body_invweight0():
    b = _builder()
    plug, pad, sp, blade = _plug_with_pad(b, 0.1)
    _row(b, sp, blade, fit=True)
    _row(b, 0, blade, fit=False)                       # the plug's own box: left as assembled
    main = _builder()
    main.replicate(b, 2)
    model = main.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    use_mujoco_pair_order(solver)                      # as the router: rows where Newton's update copies them
    before = solver.mjw_model.pair_solref.numpy().copy()
    assert fit_pair_rows(solver) == 2                  # one recorded row per world
    after = solver.mjw_model.pair_solref.numpy()
    mj = solver.mj_model
    j = int(np.flatnonzero(model.alloy.pair_k.numpy()[: mj.npair] > 0.0)[0])    # rows now in MuJoCo's order
    wt = mj.body_invweight0[mj.geom_bodyid[mj.pair_geom1[j]], 0] + mj.body_invweight0[mj.geom_bodyid[mj.pair_geom2[j]], 0]
    m_pad = float(model.body_mass.numpy()[pad])
    assert 0.0 < wt < 0.1 / m_pad                      # the armature counts under the free root (not 1/m_pad)
    want = refit_solref(_fit(), _imp(), np.array([wt]), np.zeros(1))[0]
    np.testing.assert_allclose(after[:, j], np.tile(want, (after.shape[0], 1)), rtol=1e-5)
    np.testing.assert_array_equal(after[:, 1 - j], before[:, 1 - j])
    np.testing.assert_allclose(mj.pair_solref[j], want, rtol=1e-5)


def test_fit_is_a_no_op_without_recorded_rows():
    b = _builder()
    _, _, sp, blade = _plug_with_pad(b, 0.0)
    _row(b, sp, blade, fit=False)
    model = b.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    before = solver.mjw_model.pair_solref.numpy().copy()
    assert fit_pair_rows(solver) == 0
    np.testing.assert_array_equal(solver.mjw_model.pair_solref.numpy(), before)
    assert getattr(getattr(model, "alloy", None), "pair_k", None) is None    # no attribute: digests unchanged


def _scale_world_mass(model, world: int, scale: float) -> None:
    """Scale every body of `world` by `scale` (what a per-world material.density draw writes)."""
    per = model.body_count // model.world_count
    rows = slice(world * per, (world + 1) * per)
    for name, f in (("body_mass", scale), ("body_inertia", scale), ("body_inv_mass", 1.0 / scale),
                    ("body_inv_inertia", 1.0 / scale)):
        a = getattr(model, name).numpy()
        a[rows] = a[rows] * f
        getattr(model, name).assign(a)


DT, ZETA_DRAWN = 0.00125, 0.5     # C13's step and CCD damping floor (the host model's opt.timestep is not the step)


def _drawn_solver(scale1: float):
    b = _builder()
    _, pad, sp, blade = _plug_with_pad(b, 0.0)         # no armature: recorded because its mass is drawn
    fit = MP.FitBodies(drawn=frozenset({pad}))
    values = MP._row_fit_values(b, (pad, -1), fit, float(b.body_mass[pad]),
                                (K, ZETA_DRAWN, DEFAULT_IMP, (DT, "implicitfast")))
    b.add_custom_values(**{"mujoco:pair_world": b.current_world, "mujoco:pair_geom1": sp, "mujoco:pair_geom2": blade,
                           "mujoco:pair_condim": 3, "mujoco:pair_solref": (0.02, 1.0),
                           "mujoco:pair_solimp": MP.DEFAULT_SOLIMP, **values})
    main = _builder()
    main.replicate(b, 2)
    model = main.finalize(device="cpu")
    _scale_world_mass(model, 1, scale1)                # drawn before the entry is built, as at assembly
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    return model, solver, pad


def _world_row(solver, world: int) -> tuple[np.ndarray, int]:
    """(solref on MuJoCo Warp, MuJoCo pair index) of the recorded row in `world`."""
    j = int(np.flatnonzero(solver.model.alloy.pair_k.numpy()[: solver.mj_model.npair] > 0.0)[0])
    return solver.mjw_model.pair_solref.numpy()[world, j], j


def test_drawn_rows_take_each_worlds_mass_and_refit_on_notify():
    """SCR-DC3-8: the capped row realises the template's force-space stiffness in every world (ke·m = cap·m0); a
    later mass change (a reset's redraw) refits on notify, down to MuJoCo's 2·dt clamp."""
    model, solver, pad = _drawn_solver(0.8)
    use_mujoco_pair_order(solver)
    assert fit_pair_rows(solver) == 2
    cap = float(model.alloy.pair_kecap.numpy()[0])
    m0 = float(model.alloy.pair_m0.numpy()[0])
    assert cap > 0.0 and m0 == pytest.approx(float(model.body_mass.numpy()[pad]), rel=1e-6)   # template pad
    assert float(model.alloy.pair_tcmin.numpy()[0]) == pytest.approx(2.0 * DT)
    ke0, ke1 = _ke(_world_row(solver, 0)[0]), _ke(_world_row(solver, 1)[0])
    assert ke0 == pytest.approx(cap, rel=1e-5)
    assert ke1 * 0.8 * m0 == pytest.approx(cap * m0, rel=1e-5)      # 1.25x the cap: tc 4.5 ms > 2 dt
    _scale_world_mass(model, 1, 0.5 / 0.8)             # redraw: world 1 at half the template mass
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    solref1, j = _world_row(solver, 1)
    assert solref1[0] == pytest.approx(2.0 * DT, rel=1e-5)          # 2x the cap would be 3.5 ms: clamped
    assert solver.mj_model.pair_solref[j][0] == pytest.approx(_world_row(solver, 0)[0][0], rel=1e-5)   # host = world 0


def test_drawn_bodies_are_the_density_draws():
    from types import SimpleNamespace as NS

    def obj(fields, bodies):
        return NS(obj=object(), placement=NS(randomize=tuple(NS(field=f) for f in fields)), ranges=NS(body=bodies))
    objs = [obj(["material.density"], range(2, 4)), obj(["material.surface.mu_dynamic"], range(4, 6)),
            obj([], range(6, 7))]
    assert MP.drawn_bodies(objs) == frozenset({2, 3})
