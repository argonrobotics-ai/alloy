# SPDX-License-Identifier: Apache-2.0
"""TRAIN-FIX 1b: firm contacts for the shipped arm packs (FR3, Panda, UR10e + 2F-85), applied by
:func:`alloy.tasks.emb_build.build_spec` to the robot (``firm_contacts``) and to its world (``firm_world``)."""

from __future__ import annotations

import mujoco
import numpy as np

# TRAIN-FIX 1b (WP3, 2026-11-11): the arms' geoms kept MuJoCo's default contact, solref 0.02 s. MuJoCo scales a
# contact's stiffness by the mass behind it, and on the 15 g Franka finger (invweight 3.02 /kg) that rests 5.5 mm deep
# at 91 N and 10.9 mm as the hand arrives, so an arm pressing on its table went INVALID (TA-1 F1). FIRM_SOLREF is
# 2 x the 1.25 ms robot step, MuJoCo's floor there, as the gantry's WALL_SAFE fingers. MuJoCo mixes a pair's solref
# by solmix, so FIRM_SOLMIX makes the robot geom's solref govern a pair with a default (solmix 1) geom such as the
# pack's 0.02 s table (2.52 ms, not 11 ms); friction still takes the pair's larger value (solmix does not touch it).
# FIRM_SOLIMP (the 2F-85 pads' and T14's mug's) replaces MuJoCo's default solimp on those geoms: dmax 0.99 lets a
# contact yield a fifth as far under load ((1 - d) / d), halving the steady press depth (WP3: no solve-side risk,
# +7 % solver iterations on the press; the time scale, and so stability, is solref's).
FIRM_SOLREF = (0.0025, 1.0)
FIRM_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
DEFAULT_SOLIMP_FULL = (0.9, 0.95, 0.001, 0.5, 2.0)
FIRM_SOLMIX = 1000.0
FIRM_CONTACTS = ("e1_fr3", "e2_panda", "e3_ur10e_2f85")


def firm_contacts(spec: mujoco.MjSpec) -> int:
    """Every collision geom of the robot (the spec as its loader returns it, before the world is added): solref
    ``FIRM_SOLREF``, solmix ``FIRM_SOLMIX`` and, where it is MuJoCo's default, solimp ``FIRM_SOLIMP``; friction,
    condim, margins, priority and every other solimp (the 2F-85 pads already ship ``FIRM_SOLIMP``) unchanged. Returns
    the number of geoms changed."""
    n = 0
    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            continue
        g.solref, g.solmix = list(FIRM_SOLREF), FIRM_SOLMIX
        _firm_solimp(g)
        n += 1
    return n


def _firm_solimp(g: mujoco.MjsGeom) -> None:
    if np.allclose(g.solimp, DEFAULT_SOLIMP_FULL):
        g.solimp = list(FIRM_SOLIMP)


def firm_world(spec: mujoco.MjSpec) -> int:
    """The world's geoms still at MuJoCo's default solref (0.02 s: the table, floor, box and cylinder; T14's mug ships
    its own 2.5 ms and ``FIRM_SOLIMP``): solref ``FIRM_SOLREF`` and solimp ``FIRM_SOLIMP``, solmix unchanged. With the
    robot firm and the table at 0.02 s, the hand pressed TA-1's 4 cm cube 11.8 mm into the table. Returns the number
    of geoms changed."""
    n = 0
    for g in spec.geoms:
        if (g.contype or g.conaffinity) and np.allclose(g.solref, (0.02, 1.0)):
            g.solref = list(FIRM_SOLREF)
            _firm_solimp(g)
            n += 1
    return n
