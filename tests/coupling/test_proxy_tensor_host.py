# SPDX-License-Identifier: Apache-2.0
"""Host side of the anisotropic proxy mass: the per-body tensor handed to a VBD destination
(``alloy.coupling.proxy_tensor``).

Notation: W = J M^-1 J^T, the 3x3 translational block at the body's centre of mass at qpos0 (world frame); R0 the
rotation of MuJoCo's body frame (``xmat``, the frame Newton's body_q refers to); s the scalar inverse mass installed
on the destination. Unfloored tensor Minv = (R0^T W R0) * 3 s / trace(W), so trace(Minv) / 3 = s. Floor:
eigenvalues >= lambda_max / kappa, M = inverse(Minv_floored).

Laws:
- slider, 1.5 kg along a unit vector a: W = a a^T / m, installed s = trace(W)/3 / ... = 1/(3 m), so
  Minv = 3 s a a^T = a a^T / m in the body frame (rank 1); floored eigenvalues {lam, lam/k, lam/k}.
- free body: W = I / m, Minv = I / m exactly.
- hinge tip, mass m at radius r from the axis: W = t t^T / m_axis-ish with only the tangent mobile, lambda_max
  equals 1 / m_axis of ``axis_mass_ratio``.
- rotated body frame: R0 Minv_body R0^T = W * 3 s / trace(W).
- C7 two-level chain: W = diag(0, 1/0.56, 1/2.12) for the pad (jaw y, carriage z): Minv eigenvalues in the ratio
  1/0.56 : 1/2.12 : 0, the third floored.
- trace rule with any installed s (mass_scale, per-world masses): trace(Minv)/3 == s.
- Scheduler: off by default nothing is installed; on, the destination gets (N, 3, 3) tensors whose proxy rows
  follow the laws and whose other rows are identity * inv_mass.
- a destination whose proxies are all free bodies (W = I / m) keeps the scalar kernels: the switch hands it
  (None, None), so such scenes step bit for bit as with the switch off."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy.coupling.proxy_tensor import (PROXY_TENSOR_KAPPA, body_inv_mass_tensor, floor_inverse, is_isotropic,
                                         mujoco_blocks, proxy_tensor_kappa, proxy_tensors)

XML = """<mujoco><worldbody>
  <body name="free" pos="0 0 1"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1.5"/></body>
  <body name="slider" pos="1 0 1"><joint type="slide" axis="1 2 2"/>
    <geom type="box" size="0.05 0.05 0.05" mass="1.5"/></body>
  <body name="hinge" pos="2 0 1"><joint type="hinge" axis="0 1 0"/>
    <geom type="sphere" pos="0.05 0 0" size="0.002" mass="0.0012"/></body>
  <body name="rot" pos="4 0 1" euler="0 0 30"><joint type="slide" axis="1 0 0"/>
    <geom type="box" size="0.05 0.05 0.05" mass="2"/></body>
  <body name="carriage" pos="3 0 1"><joint type="slide" axis="0 0 1"/><geom type="box" size="0.05 0.05 0.02" mass="1"/>
    <body name="pad_a" pos="0 0.05 -0.05"><joint type="slide" axis="0 1 0"/>
      <geom type="box" size="0.01 0.01 0.02" mass="0.56"/></body>
    <body name="pad_b" pos="0 -0.05 -0.05"><joint type="slide" axis="0 -1 0"/>
      <geom type="box" size="0.01 0.01 0.02" mass="0.56"/></body>
  </body>
</worldbody></mujoco>"""
KAPPA = 100.0


@pytest.fixture(scope="module")
def rig():
    mj = mujoco.MjModel.from_xml_string(XML)
    return mj, mujoco_blocks(mj)


def _tensor(rig, name: str, inv_mass: float | None = None) -> tuple[np.ndarray, int]:
    mj, (w, r, _, _) = rig
    b = mj.body(name).id
    s = float(mj.body_invweight0[b, 0]) if inv_mass is None else inv_mass
    return body_inv_mass_tensor(w[b], r[b], s, rig[1][3][b]), b


def test_slider_is_rank_one_along_its_axis_and_floors_to_kappa(rig):
    minv, _ = _tensor(rig, "slider")
    s = float(rig[0].body_invweight0[rig[0].body("slider").id, 0])
    a = np.array([1.0, 2.0, 2.0]) / 3.0
    assert np.trace(minv) / 3 == pytest.approx(s, rel=1e-6)
    np.testing.assert_allclose(minv, 3 * s * np.outer(a, a), atol=1e-9)
    assert 3 * s == pytest.approx(1 / 1.5, rel=1e-6)
    floored, m = floor_inverse(minv, KAPPA)
    lam = 3 * s
    np.testing.assert_allclose(np.linalg.eigvalsh(floored), [lam / KAPPA, lam / KAPPA, lam], rtol=1e-9)
    np.testing.assert_allclose(m @ floored, np.eye(3), atol=1e-5)


def test_free_body_is_identity_over_mass(rig):
    minv, _ = _tensor(rig, "free")
    np.testing.assert_allclose(minv, np.eye(3) / 1.5, atol=1e-12)
    floored, m = floor_inverse(minv, KAPPA)
    np.testing.assert_allclose(floored, minv, atol=1e-12)
    np.testing.assert_allclose(m, np.eye(3) * 1.5, atol=1e-9)


def test_hinge_tip_mobile_direction_is_the_axis_mass(rig):
    from alloy.coupling.axis_mass import _axis_and_installed

    mj = rig[0]
    minv, b = _tensor(rig, "hinge")
    axis, inst = _axis_and_installed(mj)
    assert np.linalg.eigvalsh(minv)[-1] == pytest.approx(1.0 / axis[b] * (1.0 / inst[b]) * inst[b], rel=1e-6)
    assert np.linalg.eigvalsh(minv)[-1] == pytest.approx(1.0 / 0.0012, rel=1e-3)      # tangent: the tip's own mass
    assert np.trace(minv) / 3 == pytest.approx(1.0 / inst[b], rel=1e-6)
    t = np.linalg.eigh(minv)[1][:, -1]
    assert abs(t[2]) > 0.999                                                            # tip on +x, hinge y: tangent z


def test_rotated_body_frame_tensor_rotates_back_to_w(rig):
    mj, (w, r, _, _) = rig
    minv, b = _tensor(rig, "rot")
    assert not np.allclose(r[b], np.eye(3))
    s = float(mj.body_invweight0[b, 0])
    np.testing.assert_allclose(r[b] @ minv @ r[b].T, w[b] * s / float(mj.body_invweight0[b, 0]), atol=1e-9)
    # simple-body path: invweight0 = 1/m = W's own value, so Minv = W (1/m along the slide)
    np.testing.assert_allclose(minv, np.diag([1 / 2.0, 0, 0]), atol=1e-9)             # slide is the body's own x


def test_c7_pad_chain_ratios_and_floored_third_axis(rig):
    mj, (w, r, _, _) = rig
    minv, b = _tensor(rig, "pad_a")
    np.testing.assert_allclose(np.diag(w[b]), [0.0, 1 / 0.56, 1 / 2.12], rtol=1e-6)
    s = float(mj.body_invweight0[b, 0])
    jaw, lift = (3 * s) * (1 / 0.56) / (1 / 0.56 + 1 / 2.12), (3 * s) * (1 / 2.12) / (1 / 0.56 + 1 / 2.12)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(minv)), [0.0, lift, jaw], atol=1e-9)
    floored, m = floor_inverse(minv, KAPPA)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(floored)), [jaw / KAPPA, lift, jaw], rtol=1e-9)
    np.testing.assert_allclose(m @ floored, np.eye(3), atol=1e-5)
    assert lift / jaw == pytest.approx(0.56 / 2.12, rel=1e-6)


@pytest.mark.parametrize("scale", [0.25, 1.0, 7.0])
def test_trace_rule_holds_for_any_installed_inverse_mass(rig, scale):
    for name in ("slider", "pad_a", "hinge", "free"):   # not "rot": simple body, w0 = 1/m
        base = float(rig[0].body_invweight0[rig[0].body(name).id, 0])
        minv, _ = _tensor(rig, name, base * scale)
        assert np.trace(minv) / 3 == pytest.approx(base * scale, rel=1e-6)


def test_proxy_tensors_rows_zero_mass_and_kappa(rig, monkeypatch):
    blocks = rig[1]
    inv = np.array([0.5, 0.0, 2.0])
    minv, mass = proxy_tensors(inv, [({0: rig[0].body("slider").id, 1: rig[0].body("pad_a").id}, blocks)], KAPPA)
    assert minv.dtype == np.float32 and minv.shape == (3, 3, 3)
    assert not minv[1].any() and not mass[1].any()                                   # zero mass: zero matrices
    np.testing.assert_allclose(minv[2], np.eye(3) * 2.0)
    np.testing.assert_allclose(mass[2], np.eye(3) / 2.0, rtol=1e-6)
    monkeypatch.setenv("ALLOY_PROXY_TENSOR_KAPPA", "10")
    assert proxy_tensor_kappa() == 10.0 and proxy_tensor_kappa(5.0) == 5.0
    monkeypatch.delenv("ALLOY_PROXY_TENSOR_KAPPA")
    assert proxy_tensor_kappa() == PROXY_TENSOR_KAPPA == 100.0


def _install_stub(monkeypatch) -> list:
    from alloy._vendor import newton

    calls: list = []
    monkeypatch.setattr(newton.solvers.SolverVBD, "set_proxy_mass_tensors",
                        lambda self, minv, m: calls.append((self, None if minv is None else minv.copy(),
                                                             None if m is None else m.copy())), raising=False)
    return calls


def _gripper():
    from alloy.coupling.pinch_rigs import _cloth_scheduler, gripper_model

    return _cloth_scheduler(gripper_model(), False)


def test_off_by_default_never_calls_the_solver(monkeypatch):
    monkeypatch.delenv("ALLOY_PROXY_TENSOR", raising=False)
    calls = _install_stub(monkeypatch)
    sched = _gripper()
    sched._apply_proxy_body_effective_masses()
    assert calls == []


def test_on_installs_tensors_that_follow_the_laws(monkeypatch):
    monkeypatch.setenv("ALLOY_PROXY_TENSOR", "1")
    calls = _install_stub(monkeypatch)
    sched = _gripper()
    assert calls, "construction installs the tensors"
    n_construct = len(calls)
    sched.notify_model_changed(int(sched._effective_mass_refresh_flags()))
    assert len(calls) > n_construct                                                  # refreshes re-install
    _, minv, mass = calls[-1]
    vbd = sched._entries["vbd"].solver.model
    assert minv.shape == mass.shape == (int(vbd.body_count), 3, 3) and minv.dtype == np.float32
    inv = vbd.body_inv_mass.numpy()
    proxy = sched._proxy_mappings[0]
    rows = set(int(i) for i in proxy.proxy_ids_local.numpy())
    labels = [str(x).split("/")[-1] for x in vbd.body_label]
    for i in range(int(vbd.body_count)):
        if i in rows and inv[i] > 0:
            assert inv[i] * (1 - 1e-5) <= np.trace(minv[i]) / 3 <= inv[i] * (1 + 2.0 / PROXY_TENSOR_KAPPA)   # floor adds <= 2 lam/kappa
            np.testing.assert_allclose(mass[i] @ minv[i], np.eye(3), atol=1e-4)
        else:
            np.testing.assert_allclose(minv[i], np.eye(3) * inv[i], atol=1e-7)
    pad = labels.index("pad_a")
    vals = np.sort(np.linalg.eigvalsh(minv[pad].astype(np.float64)))
    lam, mid = vals[-1], vals[1]
    assert mid / lam == pytest.approx(0.56 / 2.12, rel=1e-3)                          # lift : jaw as W has it
    assert vals[0] == pytest.approx(lam / PROXY_TENSOR_KAPPA, rel=1e-3)


def test_the_switch_leads_the_graph_key_and_keeps_the_rest(monkeypatch):
    """On, the key gains ("proxy_tensor",) in front; every other element keeps its place from the end (AUDIT-STACK)."""
    monkeypatch.setenv("ALLOY_PROXY_TENSOR", "1")
    _install_stub(monkeypatch)
    sched = _gripper()
    on = sched.graph_key()
    sched._proxy_tensor_on = False
    assert on == ("proxy_tensor",) + sched.graph_key()


def test_lone_slide_root_has_one_over_m_along_the_slide_on_the_suite_route():
    """Law: a lone slide root of mass m has W = e e^T / m. Whatever invweight0 the install reads (w0), s = installed / w0
    keeps Minv = W * s, and in a one-world scene installed == w0 * mass_scale so s == mass_scale: the eigenvalue along
    the slide is mass_scale / m. Probed through the suite's Warp route (MuJoCo Warp w0 == the C model's here)."""
    import warp as wp

    from alloy._vendor import newton
    from alloy.coupling.pinch_rigs import _cloth_scheduler, _finalize

    xml = """<mujoco><worldbody><body name="lone" pos="0 0 1"><joint type="slide" axis="0 0 1"/>
      <geom type="box" size="0.05 0.05 0.05" mass="2"/></body></worldbody></mujoco>"""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(xml)
    b.add_cloth_grid(pos=wp.vec3(-0.05, -0.05, 0.1), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=4, dim_y=4,
                     cell_x=0.025, cell_y=0.025, mass=1.0e-3)
    sched = _cloth_scheduler(_finalize(b), False)
    src = sched._entries["mjc"].solver
    blocks = mujoco_blocks(src.mj_model, src.mjw_model)
    proxy = sched._proxy_mappings[0]
    row = int(proxy.proxy_ids_local.numpy()[0])
    inv = float(sched._entries["vbd"].solver.model.body_inv_mass.numpy()[row])
    mjc = int(np.flatnonzero(src.mjc_body_to_newton.numpy()[0] >= 0)[-1])
    scale = 2.0                                           # what mass_scale would multiply the installed value by
    for mass_scale in (1.0, scale):
        minv = body_inv_mass_tensor(blocks[0][mjc], blocks[1][mjc], inv * mass_scale, blocks[3][mjc])
        assert np.linalg.eigvalsh(minv)[-1] == pytest.approx(mass_scale / 2.0, rel=1e-6)
    assert inv / blocks[3][mjc] == pytest.approx(float(proxy.mass_scale), rel=1e-5)        # s == mass_scale
    assert np.allclose(blocks[3], np.asarray(src.mj_model.body_invweight0).reshape(-1, 2)[:, 0], rtol=1e-5)


def test_scale_outside_range_falls_back_to_identity():
    minv = body_inv_mass_tensor(np.diag([0.0, 0.0, 0.5]), np.eye(3), 1.0e6, 0.5)
    np.testing.assert_allclose(minv, np.eye(3) * 1.0e6)


def test_is_isotropic_reads_identity_times_the_scalar():
    inv = np.array([20.0, 0.0, 0.5])
    eye = np.eye(3)[None] * inv[:, None, None]
    assert is_isotropic(eye, inv)
    assert is_isotropic(eye * (1.0 + 5e-6), inv)                 # float32 rounding of W stays inside the tolerance
    skew = eye.copy()
    skew[2, 0, 0] *= 1.001
    assert not is_isotropic(skew, inv)
    off = eye.copy()
    off[0, 0, 1] = 1e-3 * 20.0
    assert not is_isotropic(off, inv)


def test_free_bodies_only_keep_the_scalar_kernels(monkeypatch):
    """A free box on cloth has W = I / m: with the switch on the destination is handed (None, None), never a tensor."""
    from alloy.testing.coupling import ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth, fixed_clock_policy

    monkeypatch.setenv("ALLOY_PROXY_TENSOR", "1")
    calls = _install_stub(monkeypatch)
    spec = ClothSceneSpec(box_mass=0.05)
    cloth_scheduler(coupling_box_on_cloth(1, spec, device="cpu"), spec, policy=fixed_clock_policy(1.0 / 240.0),
                    vbd_adaptive=False)
    assert calls and all(minv is None and mass is None for _, minv, mass in calls)


CART_POLE = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
  <body name="cart" pos="0 0 1"><joint name="slide" type="slide" axis="1 0 0"/>
    <geom type="box" size="0.05 0.05 0.02" mass="1"/>
    <body name="pole" pos="0 0 0"><joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="sphere" pos="0 0 0.3" size="0.02" mass="0.2"/></body></body>
</worldbody></mujoco>"""


def _w_now(mj, d, b: int) -> np.ndarray:
    jacp, x = np.zeros((3, mj.nv)), np.zeros((3, mj.nv))
    mujoco.mj_jacBodyCom(mj, d, jacp, None, b)
    mujoco.mj_solveM(mj, d, x, jacp)
    return jacp @ x.T


@pytest.mark.parametrize("angle", [0.0, pytest.param(np.pi / 2, marks=pytest.mark.xfail(
    strict=True, reason="known gap coupling-proxy-isotropic-mass: W is the one at qpos0, turned with the body; the "
                        "cart's slide stays in the world while the pole turns"))])
def test_cart_pole_proxy_follows_the_carts_slide_when_the_pole_turns(angle):
    """Law: along the cart's slide (world x) the pole's proxy answers as MuJoCo's W at the current pose, within
    10 %. At the start pose it does. With the pole turned 90 degrees the body-frame tensor has turned its light
    direction to world z, and world x gets the floor: the proxy is far heavier there than the body (bounded by
    kappa)."""
    mj = mujoco.MjModel.from_xml_string(CART_POLE)
    w, r, _, w0 = mujoco_blocks(mj)
    b = mj.body("pole").id
    minv_body, _ = floor_inverse(body_inv_mass_tensor(w[b], r[b], float(w0[b]), float(w0[b])), KAPPA)
    d = mujoco.MjData(mj)
    d.qpos[mj.joint("hinge").qposadr[0]] = angle
    mujoco.mj_forward(mj, d)
    r_now = d.xmat[b].reshape(3, 3)
    proxy = (r_now @ minv_body @ r_now.T)[0, 0]
    assert proxy == pytest.approx(_w_now(mj, d, b)[0, 0], rel=0.10)
