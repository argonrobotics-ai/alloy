# SPDX-License-Identifier: Apache-2.0
"""Needle puncture and suture thread for VBD tet tissue (SC11; design docs/notes/sc11-puncture-design.md).

``SolverPuncture`` = SolverVBDMomentum + embedded sliding anchors: a needle tip punctures a tissue surface when its
axial force reaches the tet's ``puncture:fp``; the track's anchors slide on the needle with friction and cutting, hand
over to the thread at the swage and hold at a stopper knot. Opt-in per tet (``puncture:*`` attributes) and per needle
(``NeedleSpec``); without them the solver steps exactly as SolverVBDMomentum.
Not registered as a capability yet (SCR-SC11-1 pending).
"""

from alloy.solvers.puncture.config import NeedleSpec, PunctureConfig, TissueMaterial, register_custom_attributes, \
    set_tet_material
from alloy.solvers.puncture.obstacles import ARC, BOX, Obstacle, ObstacleConfig
from alloy.solvers.puncture.solver import SolverPuncture

__all__ = ["ARC", "BOX", "NeedleSpec", "Obstacle", "ObstacleConfig", "PunctureConfig", "SolverPuncture", "TissueMaterial", "register_custom_attributes",
           "set_tet_material"]
