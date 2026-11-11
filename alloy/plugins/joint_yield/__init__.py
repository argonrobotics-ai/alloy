# SPDX-License-Identifier: Apache-2.0
"""Joint springs that yield (SCR-WP7-15 P2): ``JointYield`` moves a hinge's or slide's MuJoCo spring reference by the
1-D J2 return past a yield torque (force); ``laws`` holds the closed forms. The API: ``scene.set_joint_yield``."""

from alloy.plugins.joint_yield.plugin import JointYield, YieldJoint

__all__ = ["JointYield", "YieldJoint"]
