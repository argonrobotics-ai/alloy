# SPDX-License-Identifier: Apache-2.0
"""Box-box line contacts in C (alloy.route._mjc, SCR-WP8-6; BUL-0014, alloy.contact.mjw_boxbox_line on MuJoCo Warp).

- upstream package 2's pose (a thin box across a slab's edge): MuJoCo's one contact at the box end becomes the two
  ends of the overlap, at MuJoCo's depth;
- the hook acts only inside a tick whose config has line on: mj_forward in the same process keeps MuJoCo's contact;
- against the Warp version (MuJoCo Warp's box_box + box_box_line, float32) on seeded poses: the same contact count
  everywhere, the same line contacts to 1e-6 m.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy.route import _mjc
from mjc_scenes import c_step

pytestmark = pytest.mark.cpu_route
lib = c_step()
SLAB = '<geom name="slab" type="box" size="0.3 0.6 0.00635" pos="0.35 0 -0.00635"/>'
XML = f"""<mujoco><worldbody>{SLAB}
  <body name="box"><freejoint/><geom name="box" type="box" size="0.0375 0.0041667 0.0075"/></body>
</worldbody></mujoco>"""
POS2 = [0.378351959, 0.599323405, 6.33253313e-05]           # upstream package 2's pose
QUAT2 = [-0.00729818454, 0.0299928649, -0.200104167, 0.979288255]


def posed(pos=POS2, quat=QUAT2):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7] = pos, quat
    mujoco.mj_kinematics(m, d)
    return m, d


def boxbox(m, d, g1, g2, margin, line):
    pos, normal, dist = np.zeros(150), np.zeros(150), np.zeros(50)
    n = lib.alloy_mjc_boxbox(m._address, d._address, g1, g2, margin, int(line), np.ctypeslib.as_ctypes(pos),
                             np.ctypeslib.as_ctypes(normal), np.ctypeslib.as_ctypes(dist))
    return n, pos[:3 * n].reshape(-1, 3), normal[:3 * n].reshape(-1, 3), dist[:n]


def test_package2_edge_on_face_becomes_a_line_contact() -> None:
    m, d = posed()
    n0, p0, n_0, d0 = boxbox(m, d, 0, 1, 0.0, False)
    assert n0 == 1 and p0[0, 0] == pytest.approx(0.3408, abs=1e-3)          # the defect: one contact at the end
    n, p, normal, dist = boxbox(m, d, 0, 1, 0.0, True)
    assert n == 2
    xs = sorted(p[:, 0])
    assert xs[0] < 0.342 and xs[1] > 0.414                                   # spans the slab edge inside the box
    assert (dist < 0.0).all() and (np.abs(normal @ n_0[0]) > np.cos(np.radians(1.0))).all()
    assert dist.min() == pytest.approx(d0[0], abs=2e-5)


def test_hook_acts_only_inside_line_ticks() -> None:
    m, d = posed()
    mujoco.mj_forward(m, d)
    assert d.ncon == 1
    out = _mjc.MjcOut()
    assert lib.alloy_mjc_tick(_mjc.MjcConfig(line=True).struct, m._address, d._address, 1, out.struct) == 0
    assert d.ncon == 2 and out.struct.line_fired == 1
    for line in (None, False):                                               # plain MuJoCo, then a line-off tick
        m, e = posed()
        if line is None:
            mujoco.mj_forward(m, e)
        else:
            lib.alloy_mjc_tick(_mjc.MjcConfig(line=line).struct, m._address, e._address, 1, _mjc.MjcOut().struct)
        assert e.ncon == 1
    lib.alloy_mjc_unhook()
    m, e = posed()
    mujoco.mj_forward(m, e)
    assert e.ncon == 1
    assert lib.alloy_mjc_tick(_mjc.MjcConfig(line=True).struct, m._address, e._address, 1, out.struct) == 0
    assert e.ncon == 2                                                       # re-installed by the next line tick


def _rot(rng, deg):
    q = np.zeros(4)
    axis = rng.normal(size=3)
    mujoco.mju_axisAngle2Quat(q, axis / np.linalg.norm(axis), np.radians(rng.uniform(-deg, deg)))
    return q


def _cases(seed: int = 0, n: int = 40):
    """(kind, size1, pos1, quat1, size2, pos2, quat2): package-2 poses jittered (edge on face), boxes edge to edge
    turned 45 deg about x (parallel edges), and generic penetrating poses."""
    rng = np.random.default_rng(seed)
    out = []
    slab = (np.array([0.3, 0.6, 0.00635]), np.array([0.35, 0, -0.00635]), np.array([1.0, 0, 0, 0]))
    for _ in range(n):
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, _rot(rng, 0.3), np.array(QUAT2))
        p = np.array(POS2) + rng.uniform(-5e-4, 5e-4, 3) * [10, 1, 1]
        out.append(("edge_face", *slab, np.array([0.0375, 0.0041667, 0.0075]), p, q))
    for _ in range(n):
        s1, s2 = rng.uniform(0.02, 0.06, 3), rng.uniform(0.02, 0.06, 3)
        q1, q2, t = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(t, np.array([1.0, 0, 0]), np.pi / 4)
        mujoco.mju_mulQuat(q1, _rot(rng, 0.5), t)
        mujoco.mju_mulQuat(q2, _rot(rng, 0.5), t)
        r1, r2 = np.zeros(9), np.zeros(9)
        mujoco.mju_quat2Mat(r1, q1)
        mujoco.mju_quat2Mat(r2, q2)
        e1, e2 = r1.reshape(3, 3) @ [0, s1[1], s1[2]], r2.reshape(3, 3) @ [0, -s2[1], -s2[2]]
        p2 = e1 - e2 + [rng.uniform(-0.02, 0.02), 0, 0] - rng.uniform(-0.002, 0.0005) * e1 / np.linalg.norm(e1)
        out.append(("edges", s1, np.zeros(3), q1, s2, p2, q2))
    for _ in range(n):
        s1, s2 = rng.uniform(0.02, 0.08, 3), rng.uniform(0.02, 0.08, 3)
        p2 = rng.normal(size=3)
        p2 *= rng.uniform(0.5, 1.0) * (s1.sum() + s2.sum()) / 2 / np.linalg.norm(p2)
        out.append(("generic", s1, np.zeros(3), _rot(rng, 180), s2, p2, _rot(rng, 180)))
    return out


def _warp_contacts(cases, margin):
    wp = pytest.importorskip("warp")
    from alloy._vendor.mujoco_warp._src.collision_primitive_core import box_box

    from alloy.contact.mjw_boxbox_line import box_box_line

    @wp.kernel
    def pairs(p1: wp.array[wp.vec3], r1: wp.array[wp.mat33], s1: wp.array[wp.vec3], p2: wp.array[wp.vec3],
              r2: wp.array[wp.mat33], s2: wp.array[wp.vec3], margin: float, ncon: wp.array[int],
              dist: wp.array2d[float], pos: wp.array2d[wp.vec3], nrm: wp.array[wp.vec3]):
        i = wp.tid()
        n, dd, pp, normal = box_box(p1[i], r1[i], s1[i], p2[i], r2[i], s2[i], margin)
        if n == 1:   # box_box_line_wrapper's rule
            k, d0, d1, q0, q1, nn = box_box_line(p1[i], r1[i], s1[i], p2[i], r2[i], s2[i], normal, margin)
            if k == 2:
                n = 2
                dd[0] = d0
                dd[1] = d1
                pp[0] = q0
                pp[1] = q1
                normal = nn
        ncon[i] = n
        for j in range(2):
            dist[i, j] = dd[j]
            pos[i, j] = wp.vec3(pp[j, 0], pp[j, 1], pp[j, 2])
        nrm[i] = normal

    def mat(q):
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, q)
        return r.reshape(3, 3)

    col = lambda k: np.array([c[k] for c in cases])                          # noqa: E731
    rots = lambda k: np.array([mat(c[k]) for c in cases])                    # noqa: E731
    n = len(cases)
    with wp.ScopedDevice("cpu"):
        ins = [wp.array(col(2), dtype=wp.vec3), wp.array(rots(3), dtype=wp.mat33), wp.array(col(1), dtype=wp.vec3),
               wp.array(col(5), dtype=wp.vec3), wp.array(rots(6), dtype=wp.mat33), wp.array(col(4), dtype=wp.vec3)]
        outs = [wp.zeros(n, int), wp.zeros((n, 2), float), wp.zeros((n, 2), wp.vec3), wp.zeros(n, wp.vec3)]
        wp.launch(pairs, n, inputs=[*ins, margin], outputs=outs)
        return [o.numpy() for o in outs]


@pytest.mark.compile_heavy
def test_matches_the_warp_line_contacts() -> None:
    cases, margin = _cases(), 1e-3
    ncon, dist, pos, nrm = _warp_contacts(cases, margin)
    m = mujoco.MjModel.from_xml_string(XML.replace(SLAB, '<body name="a"><freejoint/><geom type="box" size="1 1 1"/>'
                                                         '</body>'))
    d = mujoco.MjData(m)
    fired = {}
    for i, (kind, s1, p1, q1, s2, p2, q2) in enumerate(cases):
        m.geom_size[0], m.geom_size[1] = s1, s2
        d.qpos[:] = [*p1, *q1, *p2, *q2]
        mujoco.mj_kinematics(m, d)
        base = boxbox(m, d, 0, 1, margin, False)[0]
        n, p, normal, dd = boxbox(m, d, 0, 1, margin, True)
        assert n == ncon[i], (kind, i)
        if base == 1 and n == 2:
            fired[kind] = fired.get(kind, 0) + 1
            assert np.abs(p - pos[i]).max() < 1e-6 and np.abs(dd - dist[i]).max() < 1e-6, (kind, i)
            assert np.abs(normal - nrm[i]).max() < 1e-6, (kind, i)
    assert fired.get("edge_face", 0) >= 20 and fired.get("edges", 0) >= 5, fired       # seed 0: 32 and 8
