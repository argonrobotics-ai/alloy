# SPDX-License-Identifier: Apache-2.0
"""Math, Vector Math, Mix, Separate/Combine XYZ (Cycles 4.5 svm/math_util.h, math.h, mix.h)."""
import numpy as np
import warp as wp

# ---------------------------------------------------------------------------------------------------------
# Math / Vector Math / Mix / Separate / Combine (svm_math, svm_vector_math, svm_node_mix_vector)
# ---------------------------------------------------------------------------------------------------------
# NodeMathType values (kernel/svm/types.h)
MATH_ADD, MATH_SUBTRACT, MATH_MULTIPLY, MATH_DIVIDE, MATH_SINE, MATH_COSINE, MATH_TANGENT = 0, 1, 2, 3, 4, 5, 6
MATH_ARCSINE, MATH_ARCCOSINE, MATH_ARCTANGENT, MATH_POWER, MATH_LOGARITHM, MATH_MINIMUM, MATH_MAXIMUM = \
    7, 8, 9, 10, 11, 12, 13
MATH_ROUND, MATH_LESS_THAN, MATH_GREATER_THAN, MATH_MODULO, MATH_ABSOLUTE, MATH_ARCTAN2, MATH_FLOOR = \
    14, 15, 16, 17, 18, 19, 20
MATH_CEIL, MATH_FRACTION, MATH_SQRT, MATH_INV_SQRT, MATH_SIGN, MATH_EXPONENT, MATH_RADIANS, MATH_DEGREES = \
    21, 22, 23, 24, 25, 26, 27, 28
MATH_SINH, MATH_COSH, MATH_TANH, MATH_TRUNC, MATH_SNAP, MATH_WRAP, MATH_COMPARE, MATH_MULTIPLY_ADD = \
    29, 30, 31, 32, 33, 34, 35, 36
MATH_PINGPONG, MATH_SMOOTH_MIN, MATH_SMOOTH_MAX, MATH_FLOORED_MODULO = 37, 38, 39, 40


@wp.func
def safe_divide(a: float, b: float) -> float:
    if b != 0.0:
        return a / b
    return 0.0


@wp.func
def safe_modulo(a: float, b: float) -> float:
    if b != 0.0:
        return wp.mod(a, b)
    return 0.0


@wp.func
def safe_powf(a: float, b: float) -> float:
    if a < 0.0 and b != float(int(b)):
        return 0.0
    if b == 0.0:
        return 1.0
    if a < 0.0:  # GPU path of compatible_powf
        if wp.mod(-b, 2.0) == 0.0:
            return wp.pow(-a, b)
        return -wp.pow(-a, b)
    if a == 0.0:
        return 0.0
    return wp.pow(a, b)


@wp.func
def _signf(f: float) -> float:
    if f < 0.0:
        return -1.0
    return 1.0


@wp.func
def compatible_signf(f: float) -> float:
    if f == 0.0:
        return 0.0
    return _signf(f)


@wp.func
def _wrapf(value: float, mx: float, mn: float) -> float:
    rng = mx - mn
    if rng != 0.0:
        return value - rng * wp.floor((value - mn) / rng)
    return mn


@wp.func
def _fractf(x: float) -> float:
    return x - wp.floor(x)


@wp.func
def _pingpongf(a: float, b: float) -> float:
    if b != 0.0:
        return wp.abs(_fractf((a - b) / (b * 2.0)) * b * 2.0 - b)
    return 0.0


@wp.func
def _smoothminf(a: float, b: float, k: float) -> float:
    if k != 0.0:
        h = wp.max(k - wp.abs(a - b), 0.0) / k
        return wp.min(a, b) - h * h * h * k * (1.0 / 6.0)
    return wp.min(a, b)


@wp.func
def _svm_math_basic(op: int, a: float, b: float, c: float) -> float:
    """Math node, operations ADD .. MAXIMUM (0-13)."""
    if op == 0:
        return a + b
    if op == 1:
        return a - b
    if op == 2:
        return a * b
    if op == 3:
        return safe_divide(a, b)
    if op == 4:
        return wp.sin(a)
    if op == 5:
        return wp.cos(a)
    if op == 6:
        return wp.tan(a)
    if op == 7:
        return wp.asin(wp.clamp(a, -1.0, 1.0))
    if op == 8:
        return wp.acos(wp.clamp(a, -1.0, 1.0))
    if op == 9:
        return wp.atan(a)
    if op == 10:
        return safe_powf(a, b)
    if op == 11:
        if a <= 0.0 or b <= 0.0:
            return 0.0
        return safe_divide(wp.log(a), wp.log(b))
    if op == 12:
        return wp.min(a, b)
    if op == 13:
        return wp.max(a, b)
    return 0.0


@wp.func
def _svm_math_round_cmp(op: int, a: float, b: float, c: float) -> float:
    """Math node, operations ROUND .. RADIANS (14-27): rounding, compare, modulo, roots."""
    if op == 14:
        return wp.floor(a + 0.5)
    if op == 15:
        if a < b:
            return 1.0
        return 0.0
    if op == 16:
        if a > b:
            return 1.0
        return 0.0
    if op == 17:
        return safe_modulo(a, b)
    if op == 18:
        return wp.abs(a)
    if op == 19:
        if a == 0.0 and b == 0.0:
            return 0.0
        return wp.atan2(a, b)
    if op == 20:
        return wp.floor(a)
    if op == 21:
        return wp.ceil(a)
    if op == 22:
        return a - wp.floor(a)
    if op == 23:
        return wp.sqrt(wp.max(a, 0.0))
    if op == 24:
        if a > 0.0:
            return 1.0 / wp.sqrt(a)
        return 0.0
    if op == 25:
        return compatible_signf(a)
    if op == 26:
        return wp.exp(a)
    if op == 27:
        return a * (3.14159265358979323846 / 180.0)
    return 0.0


@wp.func
def _svm_math_hyper_misc(op: int, a: float, b: float, c: float) -> float:
    """Math node, operations DEGREES .. FLOORED_MODULO (28-40): degrees, hyperbolic, TRUNC, SNAP, WRAP, COMPARE, MULTIPLY_ADD, PINGPONG, smooth min/max, floored modulo."""
    if op == 28:
        return a * (180.0 / 3.14159265358979323846)
    if op == 29:
        return wp.sinh(a)
    if op == 30:
        return wp.cosh(a)
    if op == 31:
        return wp.tanh(a)
    if op == 32:
        if a >= 0.0:
            return wp.floor(a)
        return wp.ceil(a)
    if op == 33:
        return wp.floor(safe_divide(a, b)) * b
    if op == 34:
        return _wrapf(a, b, c)
    if op == 35:
        if a == b or wp.abs(a - b) <= wp.max(c, 1.1920929e-07):
            return 1.0
        return 0.0
    if op == 36:
        return a * b + c
    if op == 37:
        return _pingpongf(a, b)
    if op == 38:
        return _smoothminf(a, b, c)
    if op == 39:
        return -_smoothminf(-a, -b, c)
    if op == 40:
        if b != 0.0:
            return a - wp.floor(a / b) * b
        return 0.0
    return 0.0


@wp.func
def svm_math(op: int, a: float, b: float, c: float) -> float:
    """Math node (clamp not applied; use saturate() on the result when use_clamp)."""
    if op <= 13:
        return _svm_math_basic(op, a, b, c)
    if op <= 27:
        return _svm_math_round_cmp(op, a, b, c)
    return _svm_math_hyper_misc(op, a, b, c)


# NodeVectorMathType values
VMATH_ADD, VMATH_SUBTRACT, VMATH_MULTIPLY, VMATH_DIVIDE, VMATH_CROSS, VMATH_PROJECT, VMATH_REFLECT = \
    0, 1, 2, 3, 4, 5, 6
VMATH_DOT, VMATH_DISTANCE, VMATH_LENGTH, VMATH_SCALE, VMATH_NORMALIZE = 7, 8, 9, 10, 11
VMATH_SNAP, VMATH_FLOOR, VMATH_CEIL, VMATH_MODULO, VMATH_FRACTION, VMATH_ABSOLUTE = 12, 13, 14, 15, 16, 17
VMATH_MIN, VMATH_MAX, VMATH_WRAP, VMATH_SIN, VMATH_COS, VMATH_TAN, VMATH_REFRACT = 18, 19, 20, 21, 22, 23, 24
VMATH_FACEFORWARD, VMATH_MULTIPLY_ADD, VMATH_POWER, VMATH_SIGN = 25, 26, 27, 28


@wp.func
def safe_divide3(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(safe_divide(a[0], b[0]), safe_divide(a[1], b[1]), safe_divide(a[2], b[2]))


@wp.func
def safe_normalize3(a: wp.vec3) -> wp.vec3:
    t = wp.length(a)
    if t != 0.0:
        return a * (1.0 / t)
    return a


@wp.func
def vec_floor(a: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.floor(a[0]), wp.floor(a[1]), wp.floor(a[2]))


@wp.func
def _svm_vector_math_basic(op: int, a: wp.vec3, b: wp.vec3, c: wp.vec3, param1: float):
    """Vector Math, operations ADD .. NORMALIZE (0-11). Returns (value, vector)."""
    z = wp.vec3(0.0, 0.0, 0.0)
    if op == 0:
        return 0.0, a + b
    if op == 1:
        return 0.0, a - b
    if op == 2:
        return 0.0, wp.cw_mul(a, b)
    if op == 3:
        return 0.0, safe_divide3(a, b)
    if op == 4:
        return 0.0, wp.cross(a, b)
    if op == 5:  # project(a, b) = b * (dot(a, b) / dot(b, b)) (zero if b is zero)
        den = wp.dot(b, b)
        if den != 0.0:
            return 0.0, b * (wp.dot(a, b) / den)
        return 0.0, z
    if op == 6:
        n = safe_normalize3(b)
        return 0.0, a - n * (2.0 * wp.dot(n, a))
    if op == 7:
        return wp.dot(a, b), z
    if op == 8:
        return wp.length(a - b), z
    if op == 9:
        return wp.length(a), z
    if op == 10:
        return 0.0, a * param1
    if op == 11:
        return 0.0, safe_normalize3(a)
    return 0.0, z


@wp.func
def _svm_vector_math_elementwise(op: int, a: wp.vec3, b: wp.vec3, c: wp.vec3, param1: float):
    """Vector Math, operations SNAP .. SIGN (12-28), plus unported REFRACT. Returns (value, vector)."""
    z = wp.vec3(0.0, 0.0, 0.0)
    if op == 12:
        return 0.0, wp.cw_mul(vec_floor(safe_divide3(a, b)), b)
    if op == 13:
        return 0.0, vec_floor(a)
    if op == 14:
        return 0.0, wp.vec3(wp.ceil(a[0]), wp.ceil(a[1]), wp.ceil(a[2]))
    if op == 15:
        return 0.0, wp.vec3(safe_modulo(a[0], b[0]), safe_modulo(a[1], b[1]), safe_modulo(a[2], b[2]))
    if op == 16:
        return 0.0, a - vec_floor(a)
    if op == 17:
        return 0.0, wp.vec3(wp.abs(a[0]), wp.abs(a[1]), wp.abs(a[2]))
    if op == 18:
        return 0.0, wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))
    if op == 19:
        return 0.0, wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))
    if op == 21:
        return 0.0, wp.vec3(wp.sin(a[0]), wp.sin(a[1]), wp.sin(a[2]))
    if op == 22:
        return 0.0, wp.vec3(wp.cos(a[0]), wp.cos(a[1]), wp.cos(a[2]))
    if op == 23:
        return 0.0, wp.vec3(wp.tan(a[0]), wp.tan(a[1]), wp.tan(a[2]))
    if op == 25:  # faceforward(N=a, I=b, Nref=c)
        if wp.dot(c, b) < 0.0:
            return 0.0, a
        return 0.0, -a
    if op == 26:
        return 0.0, wp.cw_mul(a, b) + c
    if op == 20:
        return 0.0, wp.vec3(_wrapf(a[0], b[0], c[0]), _wrapf(a[1], b[1], c[1]), _wrapf(a[2], b[2], c[2]))
    if op == 27:
        return 0.0, wp.vec3(safe_powf(a[0], b[0]), safe_powf(a[1], b[1]), safe_powf(a[2], b[2]))
    if op == 28:
        return 0.0, wp.vec3(compatible_signf(a[0]), compatible_signf(a[1]), compatible_signf(a[2]))
    return 0.0, z


@wp.func
def svm_vector_math(op: int, a: wp.vec3, b: wp.vec3, c: wp.vec3, param1: float):
    """Vector Math node. Returns (value, vector); value is used by DOT/DISTANCE/LENGTH, vector otherwise
    (the unused output is zero)."""
    if op <= 11:
        return _svm_vector_math_basic(op, a, b, c, param1)
    return _svm_vector_math_elementwise(op, a, b, c, param1)  # REFRACT (24) is not used by C3 and not ported


@wp.func
def saturatef(x: float) -> float:
    return wp.clamp(x, 0.0, 1.0)


@wp.func
def mix_vector(fac: float, a: wp.vec3, b: wp.vec3, use_clamp: int) -> wp.vec3:
    """ShaderNodeMix, data_type VECTOR, uniform factor, blend MIX: a * (1 - t) + b * t."""
    t = fac
    if use_clamp != 0:
        t = saturatef(t)
    return wp.vec3(a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t, a[2] * (1.0 - t) + b[2] * t)


@wp.func
def _saturate3(c: wp.vec3) -> wp.vec3:
    return wp.vec3(saturatef(c[0]), saturatef(c[1]), saturatef(c[2]))


@wp.func
def mix_color(fac: float, a: wp.vec3, b: wp.vec3, blend: int, clamp_factor: int, clamp_result: int) -> wp.vec3:
    """ShaderNodeMix, data_type RGBA, uniform factor (Cycles svm_mix): blend 0 = MIX, a (1 - t) + b t; blend 1 =
    MULTIPLY, a (1 - t) + a b t. ``clamp_result`` clamps each channel to [0, 1]."""
    t = fac
    if clamp_factor != 0:
        t = saturatef(t)
    r = wp.vec3(a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t, a[2] * (1.0 - t) + b[2] * t)
    if blend == 1:
        r = wp.vec3(a[0] * (1.0 - t + t * b[0]), a[1] * (1.0 - t + t * b[1]), a[2] * (1.0 - t + t * b[2]))
    if clamp_result != 0:
        r = _saturate3(r)
    return r


@wp.func
def mix_float(fac: float, a: float, b: float, use_clamp: int) -> float:
    t = fac
    if use_clamp != 0:
        t = saturatef(t)
    return a * (1.0 - t) + b * t


@wp.func
def separate_xyz(v: wp.vec3):
    return v[0], v[1], v[2]


@wp.func
def combine_xyz(x: float, y: float, z: float) -> wp.vec3:
    return wp.vec3(x, y, z)




MAP_RANGE_KINDS = {"LINEAR": 0, "STEPPED": 1, "SMOOTHSTEP": 2, "SMOOTHERSTEP": 3}


@wp.func
def _smoothstep(e0: float, e1: float, x: float) -> float:
    """util/math_float.h smoothstep (0 or 1 outside the edges)."""
    if x < e0:
        return 0.0
    if x >= e1:
        return 1.0
    t = (x - e0) / (e1 - e0)
    return (3.0 - 2.0 * t) * (t * t)


@wp.func
def _smootherstep(e0: float, e1: float, x_in: float) -> float:
    x = wp.clamp(safe_divide(x_in - e0, e1 - e0), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


@wp.func
def map_range(value: float, from_min: float, from_max: float, to_min: float, to_max: float, steps: float,
              kind: int, use_clamp: int) -> float:
    """svm_node_map_range (FLOAT) plus the Clamp node Cycles inserts for LINEAR / STEPPED with clamp on (RANGE
    clamp: the bounds may be in either order)."""
    if from_max == from_min:
        return 0.0
    f = (value - from_min) / (from_max - from_min)
    if kind == 1:
        f = wp.where(steps > 0.0, wp.floor(f * (steps + 1.0)) / steps, 0.0)
    elif kind == 2:
        f = wp.where(from_min > from_max, 1.0 - _smoothstep(from_max, from_min, value),
                     _smoothstep(from_min, from_max, value))
    elif kind == 3:
        f = wp.where(from_min > from_max, 1.0 - _smootherstep(from_max, from_min, value),
                     _smootherstep(from_min, from_max, value))
    r = to_min + f * (to_max - to_min)
    if use_clamp != 0 and kind <= 1:
        r = wp.clamp(r, wp.min(to_min, to_max), wp.max(to_min, to_max))
    return r
