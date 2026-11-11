# SPDX-License-Identifier: Apache-2.0
"""Sports plug-ins: ``PressureBall`` (inflated-ball contact, turf, rolling), ``BallAero`` (drag, Magnus) and
``GoalNet`` (a cable net that catches the ball).

See ``README.md`` for the model, the sources and the laws; ``model.py`` holds the float64 references.
"""

from alloy.plugins.sports.aero import BallAero
from alloy.plugins.sports.ball import PressureBall
from alloy.plugins.sports.net import GoalNet

__all__ = ["BallAero", "GoalNet", "PressureBall"]
