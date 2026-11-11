# SPDX-License-Identifier: Apache-2.0
"""The drilling plug-in's device types: the bit, its Shaw-Oxford constants, a layer's work, the tool's state, one
layer's view of the cutting front and the stack's drive, and the codes the front uses (``drilling.kernels``).

Kept apart from the kernels so the cutting-front kernels and the hole helpers can share them.
"""

from __future__ import annotations

import math

import warp as wp

from alloy.plugins.funcs import com_world, point_velocity

TWO_PI = wp.constant(2.0 * math.pi)
MIN_FACING = wp.constant(0.8)    # cos of the largest drill-axis tilt from the face normal that still cuts (~37 deg)
NONE = wp.constant(0)            # layer modes: nothing to do
CUT = wp.constant(1)             # the edges cut material ahead of the tip
FREE = wp.constant(2)            # nothing ahead of the tip (through, or in a pilot's void): the bottom follows it
RIGID = wp.constant(3)           # pressed in without cutting (tilted past MIN_FACING, or the table is full)
JAM = wp.constant(4)             # the point rests on a bar it cannot cut (rebar): held along the hole axis
IN_NONE = wp.constant(-3)        # hole codes: not in or entering a hole
IN_FULL = wp.constant(-2)        # entering, but the layer's hole table is full
IN_NEW = wp.constant(-1)         # entering where no hole is: a new hole
ST_INSIDE = wp.constant(-3)      # stage codes: inside a wider stage's void (no stage of its own)
ST_FULL = wp.constant(-2)        # every stage slot used: held at the deepest bottom


@wp.struct
class Bit:
    tip: wp.vec3                 # drill tip, tool body frame [m]
    axis: wp.vec3                # unit drill axis from the shank toward the tip, tool body frame
    tan_half: wp.float32         # tan of the point half-angle
    chisel: wp.float32           # chisel edge length / diameter (c/d)
    stiffness: wp.float32        # cutting-front contact stiffness [N/m]
    damping: wp.float32          # its Kelvin-Voigt time [s]
    radial: wp.float32           # hole wall stiffness per spring [N/m]
    radial_damping: wp.float32   # [s]
    feed_max: wp.float32         # largest feed per revolution the flutes take [m]
    spin_eps: wp.float32         # [rad/s]: the torque is viscous below this relative spin
    spin_inertia: wp.float32     # tool inertia about the drill axis [kg·m^2] (raises spin_eps for stability)
    holes: wp.int32              # hole slots per layer (H)
    stages: wp.int32             # stage slots per hole (S)


@wp.struct
class Shaw:
    lips: wp.float32             # C_T: lip thrust = lips·f^0.8·d^0.8 [N, f and d in m]
    chisel: wp.float32           # C_J: chisel thrust = chisel·d^2
    torque: wp.float32           # C_M: torque = torque·f^0.8·d^1.8 [N·m]


@wp.struct
class Work:
    thickness: wp.float32        # plate thickness under the face [m]
    density: wp.float32          # [kg/m^3]
    capture: wp.float32          # fraction of the chips the tool's extraction collects (sink); the rest stays put
    half: wp.vec2                # half extent of the face patch [m]: no material beyond it
    ev_n: wp.int32               # rotary hammer: points of the energy-per-volume table (0: twist drilling, Shaw-Oxford)
    ev_d: wp.vec4                # their bit diameters [m], ascending
    ev: wp.vec4                  # blow energy per removed volume there [J/m^3]


@wp.struct
class Tool:
    body: wp.int32
    tip: wp.vec3                 # world
    axis: wp.vec3                # world, unit, toward the tip
    com: wp.vec3
    v: wp.vec3                   # tip velocity
    omega: wp.vec3


@wp.struct
class LayerFront:
    face: wp.transform           # face frame, world (z = 0 on the face, +z out of the material)
    n: wp.vec3                   # face normal, world
    a: wp.vec3                   # hole axis, world (into the material)
    a_l: wp.vec3                 # hole axis, face frame
    entry: wp.vec3               # hole entry, world
    entry_l: wp.vec2             # hole entry, face frame
    cross: wp.vec2               # where the bit's axis crosses the face (face frame)
    bit_l: wp.vec3               # the bit's axis, face frame
    com: wp.vec3                 # layer body COM (static: 0)
    v: wp.vec3                   # tip velocity relative to the layer
    spin: wp.float32             # spin of the tool relative to the layer about the drill axis [rad/s]
    hole: wp.int32               # hole index in the layer, or IN_NEW / IN_FULL / IN_NONE
    stage: wp.int32              # stage index, -1 a new stage, -2 the hole's stages are full
    base: wp.int32               # first stage slot of the hole
    nst: wp.int32                # the hole's stage count
    formed: wp.int32             # 1 when a stage has its full diameter at the face (the hole guides the bit)
    b: wp.float64                # the stage's bottom (its start for a new stage)
    s: wp.float32                # tip along the hole axis from the entry
    s_n: wp.float32              # tip depth below the face along its normal
    tip_z: wp.float32            # tip height over the face (face frame z)
    over: wp.int32               # 1 when the tip lies over the face patch (material can be under it)
    span: wp.float32             # the layer's length along the hole axis
    r_void: wp.float32           # radius of a wider stage's (pilot's) void the edges turn in, 0: none
    e: wp.float32                # press into uncut material (RIGID: along the normal)
    lips: wp.float32
    moment: wp.float32
    chisel: wp.float32
    mode: wp.int32


@wp.struct
class Drive:
    thrust: wp.float32           # front force T [N]
    feed: wp.float32             # feed per revolution cut this step [m]
    lips: wp.float32             # sum of phi_lips C_T over the cutting layers
    chisel: wp.float32           # sum of phi_chisel C_J
    front: wp.float32            # the established layers' front compression (a new stage starts there), -1: none
    count: wp.int32              # cutting layers
    hammer: wp.int32             # of them, layers a rotary hammer cuts (their feed is the blow power's, not Shaw's)


@wp.func
def tool_state(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
               tb: wp.int32, bit: Bit) -> Tool:
    t = Tool()
    t.body = tb
    t.tip = wp.transform_point(body_q[tb], bit.tip)
    t.axis = wp.normalize(wp.transform_vector(body_q[tb], bit.axis))
    t.com = com_world(body_q, body_com, tb)
    t.v = point_velocity(body_qd, t.com, tb, t.tip)
    qd = body_qd[tb]
    t.omega = wp.vec3(qd[3], qd[4], qd[5])
    return t


__all__ = ["CUT", "FREE", "IN_FULL", "IN_NEW", "IN_NONE", "JAM", "MIN_FACING", "NONE", "RIGID", "ST_FULL", "ST_INSIDE",
           "TWO_PI", "Bit", "Drive", "LayerFront", "Shaw", "Tool", "Work", "tool_state"]
