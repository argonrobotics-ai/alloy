# SPDX-License-Identifier: Apache-2.0
"""BUL-0014: box-box line contacts on MuJoCo Warp's primitive collider (``alloy.contact.mjw_boxbox_line``), on CPU.

Upstream package 2's pose: a thin box across a slab's top edge, where MuJoCo C and MuJoCo Warp's ``box_box`` return
one edge-edge contact at the box's end. The wrapper returns the overlap segment's two endpoints; face-face, crossing
edges and vertex contacts stay MuJoCo's."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
mjw = pytest.importorskip("alloy._vendor.mujoco_warp")
wp = pytest.importorskip("warp")

from alloy.contact import mjw_boxbox_line as L  # noqa: E402
from alloy.contact.mjw_boxbox import set_primitive_boxbox  # noqa: E402

SLAB = '<geom name="slab" type="box" size="0.3 0.6 0.00635" pos="0.35 0 -0.00635"/>'
XML = f"""<mujoco><worldbody>{SLAB}
  <body name="box"><freejoint/><geom name="box" type="box" size="{{size}}"/></body>
</worldbody></mujoco>"""
THIN = "0.0375 0.0041667 0.0075"
POS2 = [0.378351959, 0.599323405, 6.33253313e-05]           # upstream package 2's pose
QUAT2 = [-0.00729818454, 0.0299928649, -0.200104167, 0.979288255]


@pytest.fixture
def table():
    """Leave MuJoCo Warp's primitive table as the test found it."""
    was_on = L.line_boxbox_on()
    yield
    if was_on:
        L.install_line_boxbox()
    else:
        L.restore_mujoco_boxbox()


def _contacts(pos, quat, line: bool, size: str = THIN, nconmax: int = 16):
    """MuJoCo C's and MuJoCo Warp's (primitive box-box) contacts: lists of (dist, pos, normal), and MJW's nacon."""
    m = mujoco.MjModel.from_xml_string(XML.format(size=size))
    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7] = pos, quat
    mujoco.mj_forward(m, d)
    ref = [(c.dist, c.pos.copy(), c.frame[:3].copy()) for c in d.contact[:d.ncon]]
    if line:
        L.install_line_boxbox()
    else:
        L.restore_mujoco_boxbox()
    with wp.ScopedDevice("cpu"):
        wm, wd = mjw.put_model(m), mjw.put_data(m, d, nworld=1, nconmax=nconmax, njmax=64)
        wm.opt.disableflags = int(wm.opt.disableflags) | int(mjw.DisableBit.NATIVECCD)
        mjw.kinematics(wm, wd)
        mjw.collision(wm, wd)
        n = int(wd.nacon.numpy()[0])
        k = min(n, nconmax)
        got = [(float(wd.contact.dist.numpy()[i]), wd.contact.pos.numpy()[i], wd.contact.frame.numpy()[i][0])
               for i in range(k)]
    return ref, got, n


def _quat(axis, deg):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), np.radians(deg))
    return q


def test_package2_edge_on_face_becomes_a_line_contact(table) -> None:
    ref, base, _ = _contacts(POS2, QUAT2, line=False)
    assert len(ref) == len(base) == 1                                   # the defect: one contact at the box end
    assert base[0][1][0] == pytest.approx(0.3408, abs=1e-3)
    _, got, _ = _contacts(POS2, QUAT2, line=True)
    assert len(got) == 2
    xs = sorted(float(p[0]) for _, p, _ in got)
    assert xs[0] < 0.342 and xs[1] > 0.414                              # spans the slab edge inside the box
    for dist, _, normal in got:
        assert dist < 0.0
        assert abs(float(np.dot(normal, ref[0][2]))) > np.cos(np.radians(1.0))  # the box face, 0.13 deg from MuJoCo's
    assert min(d for d, _, _ in got) == pytest.approx(ref[0][0], abs=2e-5)    # MuJoCo's depth at the near end


def test_face_face_and_crossing_edges_stay_mujocos(table) -> None:
    flat = ([0.4, 0.0, 0.0074], [1.0, 0.0, 0.0, 0.0])                # resting flat, 0.1 mm in: face-face
    rf, got, _ = _contacts(*flat, line=True)
    _, base, _ = _contacts(*flat, line=False)
    assert len(got) == len(base) == len(rf) == 4
    np.testing.assert_allclose(sorted(d for d, _, _ in got), sorted(d for d, _, _ in base), atol=1e-7)
    # a cube balanced on one edge across the slab's top edge, crossing at 90 deg: one point is right
    q = _quat([0, 0, 1], 90.0)
    qe = np.zeros(4)
    mujoco.mju_mulQuat(qe, q, _quat([1, 0, 0], 45.0))
    cross = ([0.35, 0.6 + 0.004, 0.01 * np.sqrt(2) - 0.0005], qe)
    _, got, _ = _contacts(*cross, line=True, size="0.01 0.01 0.01")
    _, base, _ = _contacts(*cross, line=False, size="0.01 0.01 0.01")
    assert len(got) == len(base)
    np.testing.assert_allclose([d for d, _, _ in got], [d for d, _, _ in base], atol=1e-7)


@wp.kernel
def _line_kernel(p: wp.array(dtype=wp.vec3), r: wp.array(dtype=wp.mat33), h: wp.array(dtype=wp.vec3),
                 n: wp.array(dtype=wp.vec3), margin: float, out_n: wp.array(dtype=int), out_d: wp.array(dtype=float),
                 out_p: wp.array(dtype=wp.vec3), out_nrm: wp.array(dtype=wp.vec3)):
    c, d0, d1, q0, q1, nn = L.box_box_line(p[0], r[0], h[0], p[1], r[1], h[1], n[0], margin)
    out_n[0] = c
    out_d[0] = d0
    out_d[1] = d1
    out_p[0] = q0
    out_p[1] = q1
    out_nrm[0] = nn


def _line(p1, r1, h1, p2, r2, h2, normal, margin=0.0):
    with wp.ScopedDevice("cpu"):
        arr = lambda v, t: wp.array(np.asarray(v, np.float32), dtype=t)  # noqa: E731
        on, od = wp.zeros(1, dtype=int), wp.zeros(2, dtype=float)
        op, onrm = wp.zeros(2, dtype=wp.vec3), wp.zeros(1, dtype=wp.vec3)
        wp.launch(_line_kernel, 1, inputs=[arr([p1, p2], wp.vec3), arr([r1, r2], wp.mat33), arr([h1, h2], wp.vec3),
                                           arr([normal], wp.vec3), margin, on, od, op, onrm])
        return int(on.numpy()[0]), od.numpy(), op.numpy(), onrm.numpy()[0]


def test_parallel_edges_give_the_overlap() -> None:
    """The slab's top edge (along x) and a cube's edge (turned 30 deg about x), parallel and 0.2 mm into each other
    along a normal between the faces (an edge axis): the 20 mm overlap's two ends at -0.2 mm."""
    c30, s30 = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    rot = np.array([[1, 0, 0], [0, c30, -s30], [0, s30, c30]])          # columns: x, y', z'
    h = 0.01
    n = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    edge = np.array([0.35, 0.6, 0.0]) - 0.0002 * n                     # the cube's edge, 0.2 mm into the slab edge
    centre = edge + h * (rot[:, 1] + rot[:, 2])
    count, dist, pos, normal = _line([0.35, 0.0, -0.00635], np.eye(3), [0.3, 0.6, 0.00635], centre, rot, [h] * 3, n)
    assert count == 2
    np.testing.assert_allclose(sorted(pos[:, 0]), [0.34, 0.36], atol=1e-6)
    np.testing.assert_allclose(dist, [-0.0002, -0.0002], atol=1e-6)
    np.testing.assert_allclose(normal, n, atol=1e-6)
    crossed = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ rot      # the same cube turned 90 deg about z
    count, *_ = _line([0.35, 0.0, -0.00635], np.eye(3), [0.3, 0.6, 0.00635], centre, crossed, [h] * 3, n)
    assert count == 0                                                   # crossing edges keep MuJoCo's point


def test_new_contacts_count_against_nconmax(table) -> None:
    _, got, n = _contacts(POS2, QUAT2, line=True, nconmax=1)
    assert n == 2 and len(got) == 1                                     # nacon > naconmax: MuJoCo Warp's overflow


def test_switch_is_per_solver_and_env_keeps_mujocos(table, monkeypatch) -> None:
    """``use_primitive_boxbox(solver, line=True)`` puts the wrapper in the table only around that solver's MuJoCo Warp
    step; other solvers and the default keep MuJoCo's routine; ``ALLOY_BOXBOX_LINE=0`` turns it off."""
    from types import SimpleNamespace

    from alloy.contact.mjw_boxbox import use_primitive_boxbox

    m = mujoco.MjModel.from_xml_string(XML.format(size=THIN))
    L.restore_mujoco_boxbox()
    seen = []
    with wp.ScopedDevice("cpu"):
        on = SimpleNamespace(mjw_model=mjw.put_model(m), _mujoco_warp_step=lambda: seen.append(L.line_boxbox_on()))
        off = SimpleNamespace(mjw_model=mjw.put_model(m), _mujoco_warp_step=lambda: seen.append(L.line_boxbox_on()))
    assert use_primitive_boxbox(on, line=True) is None and use_primitive_boxbox(off) is None
    on._mujoco_warp_step()
    off._mujoco_warp_step()
    assert seen == [True, False] and not L.line_boxbox_on()
    monkeypatch.setenv(L.ENV, "0")
    with wp.ScopedDevice("cpu"):
        env = SimpleNamespace(mjw_model=mjw.put_model(m), _mujoco_warp_step=lambda: seen.append(L.line_boxbox_on()))
    assert not L.enable_line_boxbox(env) and not L.install_line_boxbox()


OPTED_OUT_SCENES: set[str] = set()   # the library scenes LINE_OPT_OUT reaches (none since DC3's QSFP A/B, 52b9f01a)


def _libraries():
    from alloy.testing import dcconn as D

    if D.assets_root() is None:
        pytest.skip("alloy-assets checkout not found")
    from alloy.cards.library import CardLibrary

    return CardLibrary(), D.conn_library()


def test_line_opt_out_ids_exist_and_reach_only_the_listed_scenes(monkeypatch) -> None:
    """BUL-0014 opt-out list: every id is an object card in the registry, and the library scenes it keeps on MuJoCo's
    single edge contact are exactly OPTED_OUT_SCENES (so the list cannot spread silently)."""
    libs = _libraries()
    names = set().union(*(set(lib.names("objects")) for lib in libs))
    assert set(L.LINE_OPT_OUT) <= names
    out = set()
    for lib in libs:
        for n in lib.names("scenes"):
            card = lib.peek_scene(n)   # parsed, not validated: validating every scene took ~59 s (WP5 2026-11-05)
            if not L.line_default([p.object for p in card.objects] + [r.name for r in card.robots]):
                out.add(n)
    assert out == OPTED_OUT_SCENES
    assert L.line_default(["block_3", "plywood_sheet"])
    monkeypatch.setattr(L, "LINE_OPT_OUT", ("dc_conn_qsfp_cage",))
    assert not L.line_default(["block_3", "dc_conn_qsfp_cage"]) and L.line_default(["block_3", "plywood_sheet"])
    assert L.model_objects(SimpleNamespace(body_label=["a/worldbody/a", "a/worldbody/a/b", "c/worldbody/c"])) == {"a", "c"}


def test_routed_entries_get_line_contacts() -> None:
    """Built through the route: DC3's QSFP check scene (opted out until its A/B, 52b9f01a) and its SFP+ scene both get
    the line contacts."""
    _libraries()
    import logging

    from alloy.tasks import dc_conn_qsfp_cards as QC
    from alloy.tasks import dc_conn_qsfp_sim as QS
    from alloy.tasks import dc_conn_sfp_checks as K

    logging.disable(logging.WARNING)
    try:
        qsfp = QS.mj_solver(QS.build(QC.check_scene(shelf=True)))
        sfp = K.build(K.scene(lc=False)).parts.solvers["mujoco"]
    finally:
        logging.disable(logging.NOTSET)
    assert getattr(qsfp, "_alloy_boxbox_line", False)
    assert getattr(sfp, "_alloy_boxbox_line", False)
