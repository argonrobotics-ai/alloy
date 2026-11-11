# SPDX-License-Identifier: Apache-2.0
"""Bump node and cubic Image Texture / triplanar (Cycles 4.5 svm/displace.h, differential.h, device/cpu/image.h)."""
import numpy as np
import warp as wp

from .mathnodes import safe_divide, safe_normalize3, _signf


# ---------------------------------------------------------------------------------------------------------
# Bump (svm/bump.h, svm/displace.h; world-space normals, use_object_space = false)
# ---------------------------------------------------------------------------------------------------------
@wp.func
def make_orthonormals(N: wp.vec3):
    """util/math_float3.h make_orthonormals -> (a, b)."""
    a = wp.vec3(N[2] - N[1], N[0] - N[2], N[1] - N[0])
    if N[0] == N[1] and N[0] == N[2]:
        a = wp.vec3(N[2] - N[1], N[0] + N[2], -N[1] - N[0])
    a = wp.normalize(a)
    return a, wp.cross(N, a)


@wp.func
def differential_from_compact(Ng: wp.vec3, dP: float):
    """kernel/util/differential.h: (dPdx, dPdy) of the compact (cone) differential at a hit."""
    a, b = make_orthonormals(Ng)
    return a * dP, b * dP


@wp.func
def differential_transfer_compact(ray_dP: float, ray_dD: float, ray_t: float) -> float:
    """sd->dP at a hit: ray.dP + t * ray.dD (both scalars; see README for camera/bounce values)."""
    return ray_dP + ray_t * ray_dD


@wp.func
def bump_sample_points(P: wp.vec3, Ng: wp.vec3, dP: float, filter_width: float):
    """World-space positions at which the Bump node's height must be evaluated: (P, P_x, P_y) with
    P_x = P + dPdx * filter_width, P_y = P + dPdy * filter_width (svm_node_bump_P_dx/dy)."""
    dx, dy = differential_from_compact(Ng, dP)
    return P, P + dx * filter_width, P + dy * filter_width


@wp.func
def bump_normal(normal_in: wp.vec3, Ng: wp.vec3, dP: float, h_c: float, h_x: float, h_y: float,
                strength: float, distance: float, invert: int, filter_width: float) -> wp.vec3:
    """svm_node_set_bump. normal_in: input normal (world; sd->N if the socket is unlinked); Ng, dP: sd->Ng,
    sd->dP (compact); h_c/h_x/h_y: height at P, P_x, P_y (see bump_sample_points)."""
    dx, dy = differential_from_compact(Ng, dP)
    Rx = wp.cross(dy, normal_in)
    Ry = wp.cross(normal_in, dx)
    det = wp.dot(dx, Rx)
    surfgrad = Rx * (h_x - h_c) + Ry * (h_y - h_c)
    absdet = wp.abs(det)
    scale = distance
    if invert != 0:
        scale = -scale
    st = wp.max(strength, 0.0)
    n = safe_normalize3(normal_in * (filter_width * absdet) - surfgrad * (scale * _signf(det)))
    if n[0] == 0.0 and n[1] == 0.0 and n[2] == 0.0:
        return normal_in
    return wp.normalize(n * st + normal_in * (1.0 - st))


# ---------------------------------------------------------------------------------------------------------
# Image Texture: Cubic (B-spline), REPEAT, FLAT (kernel/device/cpu/image.h interp_cubic)
# image arrays: float32 (H, W, C), row 0 = BOTTOM row of the picture, values already /255 (8 bit) or /65535
# (16 bit) i.e. the raw encoded value. sRGB images: convert the interpolated result with srgb_to_linear.
# ---------------------------------------------------------------------------------------------------------
@wp.func
def _cubic_weights(t: float) -> wp.vec4:
    return wp.vec4(
        (((-1.0 / 6.0) * t + 0.5) * t - 0.5) * t + (1.0 / 6.0),
        ((0.5 * t - 1.0) * t) * t + (2.0 / 3.0),
        ((-0.5 * t + 0.5) * t + 0.5) * t + (1.0 / 6.0),
        (1.0 / 6.0) * t * t * t)


@wp.func
def _wrap_periodic(x: int, n: int) -> int:
    r = x % n
    if r < 0:
        r += n
    return r


@wp.func
def _img_frac(x: float):
    i = int(x)
    if x < 0.0:
        i = i - 1
    return x - float(i), i


@wp.func
def image_cubic_repeat(img: wp.array3d(dtype=float), x: float, y: float) -> wp.vec3:
    """Cubic interpolation, REPEAT. x, y in image UV units (u*width pixels). Returns channels 0..2
    (for a 1-channel image use .x). Encoded values; no colour conversion."""
    h = img.shape[0]
    w = img.shape[1]
    tx, ix = _img_frac(x * float(w) - 0.5)
    ty, iy = _img_frac(y * float(h) - 0.5)
    u = _cubic_weights(tx)
    v = _cubic_weights(ty)
    res = wp.vec3(0.0, 0.0, 0.0)
    nc = img.shape[2]
    for j in range(4):
        yy = _wrap_periodic(iy - 1 + j, h)
        row = wp.vec3(0.0, 0.0, 0.0)
        for i in range(4):
            xx = _wrap_periodic(ix - 1 + i, w)
            c0 = img[yy, xx, 0]
            c1 = c0
            c2 = c0
            if nc >= 3:
                c1 = img[yy, xx, 1]
                c2 = img[yy, xx, 2]
            row = row + wp.vec3(c0, c1, c2) * u[i]
        res = res + row * v[j]
    return res


@wp.func
def _img_rgb(img: wp.array3d(dtype=float), yy: int, xx: int) -> wp.vec3:
    c0 = img[yy, xx, 0]
    if img.shape[2] >= 3:
        return wp.vec3(c0, img[yy, xx, 1], img[yy, xx, 2])
    return wp.vec3(c0, c0, c0)


@wp.func
def image_linear_repeat(img: wp.array3d(dtype=float), x: float, y: float) -> wp.vec3:
    """Bilinear interpolation, REPEAT (Blender "Linear"). x, y in image UV units; texel (i, j) is centred at
    ((i + 0.5) / width, (j + 0.5) / height), row 0 at v = 0 (images are stored bottom row first). Returns channels
    0..2 (a 1-channel image repeated). Encoded values; no colour conversion (as image_cubic_repeat)."""
    h = img.shape[0]
    w = img.shape[1]
    tx, ix = _img_frac(x * float(w) - 0.5)
    ty, iy = _img_frac(y * float(h) - 0.5)
    x0 = _wrap_periodic(ix, w)
    x1 = _wrap_periodic(ix + 1, w)
    y0 = _wrap_periodic(iy, h)
    y1 = _wrap_periodic(iy + 1, h)
    lo = _img_rgb(img, y0, x0) * (1.0 - tx) + _img_rgb(img, y0, x1) * tx
    hi = _img_rgb(img, y1, x0) * (1.0 - tx) + _img_rgb(img, y1, x1) * tx
    return lo * (1.0 - ty) + hi * ty


@wp.func
def srgb_to_linear(c: float) -> float:
    if c < 0.04045:
        if c < 0.0:
            return 0.0
        return c * (1.0 / 12.92)
    return wp.pow((c + 0.055) * (1.0 / 1.055), 2.4)


@wp.func
def srgb_to_linear3(c: wp.vec3) -> wp.vec3:
    return wp.vec3(srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]))


@wp.func
def triplanar_weights(n: wp.vec3) -> wp.vec3:
    """The graph's weights |n|^4 / max(sum |n|^4, 1e-6) (walnut_worktop Vector Math.001-.003, Math..Math.005)."""
    a = wp.vec3(wp.abs(n[0]), wp.abs(n[1]), wp.abs(n[2]))
    a2 = wp.cw_mul(a, a)
    a4 = wp.cw_mul(a2, a2)
    s = wp.max(a4[0] + a4[1] + a4[2], 1.0e-6)
    return wp.vec3(safe_divide(a4[0], s), safe_divide(a4[1], s), safe_divide(a4[2], s))


@wp.func
def triplanar_cubic(img: wp.array3d(dtype=float), p: wp.vec3, n: wp.vec3, scale_inv: float) -> wp.vec3:
    """Encoded-value triplanar blend as built by export_pack: q = p * scale_inv; Image Texture (YZ), (XZ),
    (XY) planes weighted by triplanar_weights(n). Weights apply to the image Color output (apply the colour
    transform before blending when blending sRGB images: srgb_to_linear is per sample in Cycles)."""
    q = p * scale_inv
    w = triplanar_weights(n)
    return (image_cubic_repeat(img, q[1], q[2]) * w[0]
            + image_cubic_repeat(img, q[0], q[2]) * w[1]
            + image_cubic_repeat(img, q[0], q[1]) * w[2])


