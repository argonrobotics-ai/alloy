# SPDX-License-Identifier: Apache-2.0
"""``alloy.route.mjc_core_frames``: a per-world model's frame data from its own values, as MuJoCo compiles it.

Laws (plain MuJoCo, no build): ``refit_tree`` rebuilds every body's bounding-volume tree of a compiled model to within
1e-15 of MuJoCo's own (a rotated inertial frame, a rotated box, a capsule, a plane); after moving a world-body plane
to 0.1 m it matches the model compiled with the plane there, and a sphere resting on the moved plane makes a contact
only with the refresh (the template's flag and tree kept the plane at 0); ``refresh`` leaves a world that matches
world 0 untouched.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from alloy.route.mjc_core_frames import refit_tree, refresh

XML = """<mujoco><worldbody><geom type="plane" size="0 0 .1" pos="0 0 {z}"/>
  <geom type="box" size=".1 .1 .1" pos="1 0 .1" euler="0 0 30"/><geom type="sphere" size=".05" pos="-1 0 .2"/>
  <body pos="0 0 .5" euler="10 20 30"><freejoint/><geom type="sphere" size=".05"/>
    <geom type="box" size=".02 .03 .04" pos=".1 0 0" euler="5 40 0"/>
    <geom type="capsule" fromto="0 0 0 0 .2 .1" size=".01"/></body></worldbody></mujoco>"""


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.abs(a) < 1e9]


def test_refit_matches_mujocos_compile():
    m = mujoco.MjModel.from_xml_string(XML.format(z=0))
    c = copy.copy(m)
    for b in range(c.nbody):
        refit_tree(c, b)
    assert np.abs(_finite(c.bvh_aabb - m.bvh_aabb)).max() < 1e-15
    moved = mujoco.MjModel.from_xml_string(XML.format(z=0.1))
    c = copy.copy(m)
    c.geom_pos[0, 2] = 0.1
    refit_tree(c, 0)
    assert np.abs(_finite(c.bvh_aabb - moved.bvh_aabb)).max() < 1e-15


def _contacts(m: mujoco.MjModel) -> int:
    d = mujoco.MjData(m)
    d.qpos[:3] = (0.0, 0.0, 0.14)                   # the body's sphere 1 cm into a plane at 0.1, its box beside
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    return int(d.ncon)


def test_a_raised_plane_holds_contacts_only_with_the_refresh():
    tmpl = mujoco.MjModel.from_xml_string(XML.format(z=0))
    rows = np.stack([tmpl.geom_pos.copy(), tmpl.geom_pos.copy()])
    rows[1, 0, 2] = 0.1                              # world 1's floor raised, as a per-world robot start does
    diff = {"geom_pos": rows}
    stale = copy.copy(tmpl)
    stale.geom_pos[:] = rows[1]
    assert _contacts(stale) == 0                     # the template's flag and tree: the plane stays at 0
    fresh = copy.copy(tmpl)
    fresh.geom_pos[:] = rows[1]
    refresh(fresh, diff, 1)
    assert _contacts(fresh) >= 1
    same = copy.copy(tmpl)
    refresh(same, diff, 0)                           # world 0 matches itself: nothing changes
    assert np.array_equal(same.bvh_aabb, tmpl.bvh_aabb) and np.array_equal(same.geom_sameframe, tmpl.geom_sameframe)
