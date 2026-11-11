# SPDX-License-Identifier: Apache-2.0
"""Pure numpy pose, interpolation and colour math for the Blender side (no bpy).

Quaternions are Warp order ``xyzw`` unless a name says ``wxyz`` (Blender order).
"""

from __future__ import annotations

import math

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b of xyzw quaternions, broadcasting over leading axes."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by xyzw quaternions q (broadcasting)."""
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def compose(pa: np.ndarray, qa: np.ndarray, pb: np.ndarray, qb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose a * pose b: b expressed in a's frame, returned in a's parent frame."""
    return pa + quat_rotate(qa, pb), quat_mul(qa, qb)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n > 0.0, n, 1.0)


def slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Shortest-path spherical interpolation of xyzw quaternions, vectorised over leading axes."""
    q0 = normalize_quat(np.asarray(q0, dtype=np.float64))
    q1 = normalize_quat(np.asarray(q1, dtype=np.float64))
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    safe = np.where(small, 1.0, sin_t)
    w0 = np.where(small, 1.0 - a, np.sin((1.0 - a) * theta) / safe)
    w1 = np.where(small, a, np.sin(a * theta) / safe)
    return normalize_quat(w0 * q0 + w1 * q1)


def frame_pair(fr: float, count: int) -> tuple[int, int, float]:
    """Bracketing recorded frames (i0, i1) and blend weight for fractional frame fr, clamped to the recording."""
    if count <= 1 or fr <= 0.0:
        return 0, 0, 0.0
    if fr >= count - 1:
        return count - 1, count - 1, 0.0
    i0 = int(math.floor(fr))
    return i0, i0 + 1, float(fr - i0)


def lerp_frames(arr: np.ndarray, fr: float) -> np.ndarray:
    """Linear interpolation of a [T, ...] array at fractional frame fr (float32 result)."""
    i0, i1, a = frame_pair(fr, arr.shape[0])
    x0 = np.asarray(arr[i0], dtype=np.float32)
    if a == 0.0:
        return x0.copy()
    return x0 + (np.asarray(arr[i1], dtype=np.float32) - x0) * np.float32(a)


def interp_poses(arr: np.ndarray, fr: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions (lerp) and xyzw quaternions (slerp) of a [T, N, 7] pose array at fractional frame fr."""
    i0, i1, a = frame_pair(fr, arr.shape[0])
    p0 = np.asarray(arr[i0], dtype=np.float64)
    if a == 0.0:
        return p0[..., :3].copy(), normalize_quat(p0[..., 3:7])
    p1 = np.asarray(arr[i1], dtype=np.float64)
    pos = p0[..., :3] + (p1[..., :3] - p0[..., :3]) * a
    return pos, slerp(p0[..., 3:7], p1[..., 3:7], a)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def pose_matrices(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """[N, 4, 4] rigid transforms from positions and xyzw quaternions."""
    q = normalize_quat(np.atleast_2d(q))
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.zeros((q.shape[0], 4, 4))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    m[:, :3, 3] = np.atleast_2d(p)
    m[:, 3, 3] = 1.0
    return m


def catmull_rom(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    """Non-uniform Catmull-Rom (Barry-Goldman) through keyed values at time t, clamped to the key range.

    Phantom end keys are reflected, so two keys give a straight line and one key a constant.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times) == 1 or t <= times[0]:
        return values[0].copy()
    if t >= times[-1]:
        return values[-1].copy()
    k = int(np.searchsorted(times, t, side="right")) - 1
    tt = np.concatenate([[2 * times[0] - times[1]], times, [2 * times[-1] - times[-2]]])
    vv = np.concatenate([[2 * values[0] - values[1]], values, [2 * values[-1] - values[-2]]])
    t0, t1, t2, t3 = tt[k:k + 4]
    p0, p1, p2, p3 = vv[k:k + 4]

    def mix(pa: np.ndarray, pb: np.ndarray, ta: float, tb: float) -> np.ndarray:
        return pa if tb == ta else ((tb - t) * pa + (t - ta) * pb) / (tb - ta)

    a1, a2, a3 = mix(p0, p1, t0, t1), mix(p1, p2, t1, t2), mix(p2, p3, t2, t3)
    b1, b2 = mix(a1, a2, t0, t2), mix(a2, a3, t1, t3)
    return mix(b1, b2, t1, t2)


def look_at(eye: np.ndarray, target: np.ndarray, up: tuple = (0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 camera-to-world matrix: camera looks down its -Z with +Y up (Blender convention)."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= max(np.linalg.norm(fwd), 1e-12)
    upv = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(fwd, upv))) > 0.999:
        upv = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, upv)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, cam_up, -fwd, eye
    return m


def srgb_to_linear(c: list | np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    """Linear RGB (max channel 1) of a black body, from a Planck spectrum through CIE 1931 fits to sRGB primaries."""
    lam = np.linspace(380e-9, 780e-9, 81)
    h, c, kb = 6.62607e-34, 2.99792e8, 1.380649e-23
    spd = 1.0 / (lam ** 5 * (np.exp(h * c / (lam * kb * max(kelvin, 1000.0))) - 1.0))
    nm = lam * 1e9

    def g(x: np.ndarray, mu: float, s1: float, s2: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)

    # Wyman, Sloan & Shirley (2013) multi-lobe fit of the CIE 1931 colour-matching functions.
    xb = 1.056 * g(nm, 599.8, 37.9, 31.0) + 0.362 * g(nm, 442.0, 16.0, 26.7) - 0.065 * g(nm, 501.1, 20.4, 26.2)
    yb = 0.821 * g(nm, 568.8, 46.9, 40.5) + 0.286 * g(nm, 530.9, 16.3, 31.1)
    zb = 1.217 * g(nm, 437.0, 11.8, 36.0) + 0.681 * g(nm, 459.0, 26.0, 13.8)
    xyz = np.array([np.sum(spd * xb), np.sum(spd * yb), np.sum(spd * zb)])
    to_rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(to_rgb @ xyz, 0.0, None)
    return rgb / max(float(rgb.max()), 1e-12)


def absorption_density(color_srgb: list, distance_m: float) -> tuple[np.ndarray, float]:
    """Volume Absorption colour and density whose transmittance after distance_m equals color_srgb.

    Cycles absorbs with sigma = density * (1 - color), so sigma_c = -ln(T_c) / d is split into a max and a colour.
    """
    trans = np.clip(srgb_to_linear(color_srgb), 1e-4, 1.0)
    sigma = -np.log(trans) / max(float(distance_m), 1e-9)
    smax = float(sigma.max())
    if smax <= 0.0:
        return np.ones(3), 0.0
    return 1.0 - sigma / smax, smax


def light_irradiance(light: dict, target: np.ndarray) -> np.ndarray:
    """Linear RGB irradiance [W/m^2] a look light puts on `target` (on axis; area lights as Lambertian emitters)."""
    kind = light.get("type", "area")
    watts = float(light.get("watts", 50.0))
    color = kelvin_to_rgb(float(light.get("kelvin", 5500.0)))
    if kind == "sun":
        return color * watts
    d2 = max(float(np.sum((np.asarray(light["eye"], float) - np.asarray(target, float)) ** 2)), 1e-6)
    return color * watts / ((math.pi if kind == "area" else 4.0 * math.pi) * d2)


def scene_illuminant(lights: list, env_mean: np.ndarray | None, env_strength: float, target) -> np.ndarray:
    """Linear RGB of the light reaching `target`: the look lights plus the environment (pi * mean radiance)."""
    total = np.zeros(3)
    for lt in lights:
        total += light_irradiance(lt, np.asarray(target, float))
    if env_mean is not None:
        total += math.pi * float(env_strength) * np.asarray(env_mean, float)
    return total


def equirect_mean(pixels: np.ndarray, height: int, width: int, stride: int = 4) -> np.ndarray:
    """Solid-angle-weighted mean RGB of an equirectangular image given as flat RGBA floats (rows bottom-up)."""
    img = pixels.reshape(height, width, 4)[::stride, ::stride, :3].astype(np.float64)
    lat = (np.arange(img.shape[0]) * stride + 0.5) / height * math.pi - math.pi / 2
    w = np.cos(lat)[:, None, None]
    return (img * w).sum(axis=(0, 1)) / (w.sum() * img.shape[1])


def rest_arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative chord length along a centreline (metres), 0 at the first point."""
    seg = np.linalg.norm(np.diff(np.asarray(points, np.float64), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])
