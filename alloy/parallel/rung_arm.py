# SPDX-License-Identifier: Apache-2.0
"""L0 arm template (parallel §4.9 "arm only"): a Franka Panda with primitive hand geoms and PD joint drives.

Stand-in for the T12 arm until WP13 vendors task robots (library/robots). Kinematics and inertias from MuJoCo
Menagerie franka_emika_panda (Apache-2.0) as vendored in mujoco_warp@a9daa05f:benchmarks/franka_emika_panda/panda.xml;
link collision meshes dropped (self collision off, arm-only scene), hand capsule and finger pad boxes kept. Same
template as the baseline suite's R06 ([internal note]:1091). Drive gains, effort
limits and armature from newton@ac7f7830:newton/examples/ik/example_ik_cube_stacking.py:316-325.
"""

from __future__ import annotations

from alloy._vendor import newton

ARM_DOFS = 9
ARM_KE = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0)
ARM_KD = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0)
ARM_EFFORT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0)
ARM_ARMATURE = (0.3, 0.3, 0.3, 0.3, 0.11, 0.11, 0.11, 0.15, 0.15)
ARM_HOME = (0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785, 0.02, 0.02)  # [rad, rad, ..., m, m] inside every joint range
ARM_SWING = (0.4, 0.3, 0.4, 0.3, 0.5, 0.4, 0.5, 0.015, 0.015)  # sine amplitude per DOF
ARM_FREQ = 0.5  # [Hz]

PANDA_XML = """<mujoco model="panda">
  <compiler angle="radian" autolimits="true"/>
  <default><joint armature="0.1" damping="0" axis="0 0 1"/></default>
  <worldbody>
    <body name="link0">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974" fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
      <body name="link1" pos="0 0 0.333">
        <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762" fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
        <joint name="joint1" range="-2.8973 2.8973"/>
        <body name="link2" quat="1 -1 0 0">
          <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495" fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
          <joint name="joint2" range="-1.7628 1.7628"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2" fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
            <joint name="joint3" range="-2.8973 2.8973"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2" fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
              <joint name="joint4" range="-3.0718 -0.0698"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2" fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                <joint name="joint5" range="-2.8973 2.8973"/>
                <body name="link6" quat="1 1 0 0">
                  <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2" fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                  <joint name="joint6" range="-0.0175 3.7525"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2" fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                    <joint name="joint7" range="-2.8973 2.8973"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                      <geom name="hand_capsule" type="capsule" size="0.04 0.06" quat="1 1 0 0" pos="0 0 0.03"/>
                      <body name="left_finger" pos="0 0 0.0584">
                        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom type="box" size="0.011 0.0075 0.01" pos="0 0.0185 0.011"/>
                        <geom type="box" size="0.011 0.0044 0.0019" pos="0 0.0068 0.0022"/>
                        <geom type="box" size="0.00875 0.0035 0.01175" pos="0 0.0159 0.02835" quat="1 0.25 0 0"/>
                        <geom name="left_pad" type="box" size="0.00875 0.0076 0.00825" pos="0 0.00758 0.04525"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                        <joint name="finger_joint2" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom type="box" size="0.011 0.0075 0.01" pos="0 0.0185 0.011"/>
                        <geom type="box" size="0.011 0.0044 0.0019" pos="0 0.0068 0.0022"/>
                        <geom type="box" size="0.00875 0.0035 0.01175" pos="0 0.0159 0.02835" quat="1 0.25 0 0"/>
                        <geom name="right_pad" type="box" size="0.00875 0.0076 0.00825" pos="0 0.00758 0.04525"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""




def panda_arm(builder: newton.ModelBuilder) -> int:
    """Add the arm to ``builder`` with position+velocity drives at ``ARM_HOME``; returns its first DOF index."""
    from alloy._vendor.newton import JointTargetMode

    first, first_q = builder.joint_dof_count, builder.joint_coord_count  # 1 coord per DOF (hinges, slides)
    builder.add_mjcf(PANDA_XML, enable_self_collisions=False)
    for i in range(ARM_DOFS):
        d = first + i
        builder.joint_target_ke[d], builder.joint_target_kd[d] = ARM_KE[i], ARM_KD[i]
        builder.joint_effort_limit[d], builder.joint_armature[d] = ARM_EFFORT[i], ARM_ARMATURE[i]
        builder.joint_target_mode[d] = int(JointTargetMode.POSITION_VELOCITY)
        builder.joint_q[first_q + i] = ARM_HOME[i]
        builder.joint_target_q[first_q + i] = ARM_HOME[i]
    return first
