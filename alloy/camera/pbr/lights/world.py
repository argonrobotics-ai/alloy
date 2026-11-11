# SPDX-License-Identifier: Apache-2.0
"""HDRI world light + sun. Ports kernel/light/background.h (background_map_sample / background_map_pdf),
scene/light.cpp (background_cdf, device_update_background), kernel/camera/projection.h (equirect mapping),
kernel/light/distant.h and the distant branch of device_update_lights, at tag v4.5.0.

Importance-map resolution: scene/light.cpp:1072-1086 uses the Environment Texture's own resolution when
sampling is AUTOMATIC (map_resolution 0), and at least 4096x2048 when there is no Nishita sun; i.e. 4096x2048 for
the C3 HDRI (NOT 1024). The 1024 in the brief is the Manual-mode default."""
import numpy as np
import warp as wp

from .common import (FLT_MAX, KIND_BACKGROUND, KIND_DISTANT, M_1_2PI, M_2PI, M_PI, LightSample, safe_sqrt,
                           sample_uniform_cone)


@wp.struct
class WorldMap:
    tex: wp.array2d(dtype=wp.vec3)         # (H, W) linear RGB, ROW 0 = BOTTOM (v = 0), already x strength
    cond: wp.array2d(dtype=wp.vec2)        # (H, W+1): .x = function value, .y = CDF; [.., W].x = row total
    marg: wp.array(dtype=wp.vec2)          # (H+1)
    width: int
    height: int
    use_mis: int
    w2t: wp.mat33                          # direction -> texture-space rotation (Mapping node; identity for C3)
    t2w: wp.mat33                          # inverse
    average_radiance: float                # cdf_total * pi/2  (scene/light.cpp:1122), used by the light tree


@wp.struct
class DistantLight:
    co: wp.vec3                # direction of travel (Cycles klight->co); toward the sun is -co
    cos_angle: float           # cos of the half angle
    one_minus_cosangle: float  # 2 sin^2(half/2)
    pdf: float                 # 1/(2 pi one_minus_cos), or 1 for angle 0
    eval_fac: float            # 1/area or 1
    angle: float               # half angle [rad]
    strength: wp.vec3
    use_mis: int


# --------------------------------------------------------------------------------- equirect mapping
@wp.func
def direction_to_equirectangular(d: wp.vec3):
    """projection.h: u = (atan2(y,x) - pi)/(-2pi) = 0.5 - atan2(y,x)/2pi ; v = (acos(z) - pi)/(-pi) (v=0 at -Z)."""
    l = wp.length(d)
    if l == 0.0:
        return wp.vec2(0.0, 0.0)
    u = (wp.atan2(d[1], d[0]) - M_PI) / (-M_2PI)
    v = (wp.acos(wp.clamp(d[2] / l, -1.0, 1.0)) - M_PI) / (-M_PI)
    return wp.vec2(u, v)


@wp.func
def equirectangular_to_direction(u: float, v: float):
    phi = -M_2PI * u + M_PI
    theta = -M_PI * v + M_PI
    return wp.vec3(wp.sin(theta) * wp.cos(phi), wp.sin(theta) * wp.sin(phi), wp.cos(theta))


@wp.func
def _wrap(i: int, n: int):
    r = i % n
    if r < 0:
        r += n
    return r


@wp.func
def world_texture(w: WorldMap, uv: wp.vec2):
    """Environment Texture, Linear interpolation, REPEAT extension on both axes (shader_nodes.cpp:550,
    cpu/image.h:270-300: frac(x*W - 0.5))."""
    fx = uv[0] * float(w.width) - 0.5
    fy = uv[1] * float(w.height) - 0.5
    ixf = wp.floor(fx)
    iyf = wp.floor(fy)
    tx = fx - ixf
    ty = fy - iyf
    ix = int(ixf)
    iy = int(iyf)
    x0 = _wrap(ix, w.width)
    x1 = _wrap(ix + 1, w.width)
    y0 = _wrap(iy, w.height)
    y1 = _wrap(iy + 1, w.height)
    return ((1.0 - ty) * ((1.0 - tx) * w.tex[y0, x0] + tx * w.tex[y0, x1])
            + ty * ((1.0 - tx) * w.tex[y1, x0] + tx * w.tex[y1, x1]))


@wp.func
def background_eval(w: WorldMap, d: wp.vec3):
    """Radiance seen along world direction d (camera ray miss or indirect ray miss): Background node colour x strength."""
    return world_texture(w, direction_to_equirectangular(w.w2t * d))


# --------------------------------------------------------------------------------- importance sampling
@wp.func
def background_map_pdf(w: WorldMap, direction: wp.vec3):
    """background.h background_map_pdf (piecewise-constant pdf of the importance map), solid angle."""
    uv = direction_to_equirectangular(w.w2t * direction)
    sin_theta = wp.sin(uv[1] * M_PI)
    if sin_theta == 0.0:
        return 0.0
    iu = wp.clamp(int(uv[0] * float(w.width)), 0, w.width - 1)
    iv = wp.clamp(int(uv[1] * float(w.height)), 0, w.height - 1)
    cdf_last_u = w.cond[iv, w.width]
    cdf_last_v = w.marg[w.height]
    denom = (M_2PI * M_PI * sin_theta) * cdf_last_u[0] * cdf_last_v[0]
    if denom == 0.0:
        return 0.0
    return (w.cond[iv, iu][0] * w.marg[iv][0]) / denom


@wp.func
def _inverse_lerp(a: float, b: float, x: float):
    return (x - a) / (b - a)


@wp.func
def background_map_sample(w: WorldMap, rand: wp.vec2):
    """background.h background_map_sample. Returns (direction, pdf)."""
    # std::lower_bound over the marginal CDF .y (res_y entries)
    first = int(0)
    count = int(w.height)
    while count > 0:
        step = count >> 1
        mid = first + step
        if w.marg[mid][1] < rand[1]:
            first = mid + 1
            count -= step + 1
        else:
            count = step
    iv = wp.max(0, first - 1)
    cdf_v = w.marg[iv]
    cdf_next_v = w.marg[iv + 1]
    cdf_last_v = w.marg[w.height]
    dv = _inverse_lerp(cdf_v[1], cdf_next_v[1], rand[1])
    v = (float(iv) + dv) / float(w.height)

    first = int(0)
    count = int(w.width)
    while count > 0:
        step = count >> 1
        mid = first + step
        if w.cond[iv, mid][1] < rand[0]:
            first = mid + 1
            count -= step + 1
        else:
            count = step
    iu = wp.max(0, first - 1)
    cdf_u = w.cond[iv, iu]
    cdf_next_u = w.cond[iv, iu + 1]
    cdf_last_u = w.cond[iv, w.width]
    du = _inverse_lerp(cdf_u[1], cdf_next_u[1], rand[0])
    u = (float(iu) + du) / float(w.width)

    sin_theta = wp.sin(M_PI * v)
    denom = (M_2PI * M_PI * sin_theta) * cdf_last_u[0] * cdf_last_v[0]
    pdf = float(0.0)
    if sin_theta != 0.0 and denom != 0.0:
        pdf = (cdf_u[0] * cdf_v[0]) / denom
    return w.t2w * equirectangular_to_direction(u, v), pdf


@wp.func
def background_light_sample(w: WorldMap, rand: wp.vec2):
    """NEE sample of the world. pdf is solid angle without selection prob. ls.L = Background colour*strength."""
    ls = LightSample()
    ls.kind = KIND_BACKGROUND
    ls.id = 0
    d, pdf = background_map_sample(w, rand)
    ls.D = d
    ls.Ng = -d
    ls.P = d
    ls.t = FLT_MAX
    ls.pdf = pdf
    ls.use_mis = w.use_mis
    ls.L = background_eval(w, d)
    ls.valid = 0
    if pdf > 0.0:
        ls.valid = 1
    return ls


@wp.func
def background_light_pdf(w: WorldMap, direction: wp.vec3):
    """background.h background_light_pdf with map_weight == 1 (no portals, no Nishita sun)."""
    return background_map_pdf(w, direction)


# --------------------------------------------------------------------------------- sun
@wp.func
def distant_light_sample(l: DistantLight, rand: wp.vec2):
    ls = LightSample()
    ls.kind = KIND_DISTANT
    ng, pdf = sample_uniform_cone(l.co, l.one_minus_cosangle, rand)
    ls.Ng = ng
    ls.P = ng
    ls.D = -ng
    ls.t = FLT_MAX
    ls.pdf = pdf
    ls.L = l.strength * l.eval_fac
    ls.use_mis = l.use_mis
    ls.valid = 1
    return ls


@wp.func
def distant_light_hit(l: DistantLight, ray_D: wp.vec3):
    """BSDF ray that escapes: inside the sun disc? (distant_light_intersect + sample_from_intersection)."""
    ls = LightSample()
    ls.kind = KIND_DISTANT
    ls.valid = 0
    if l.angle == 0.0:
        return ls
    if wp.dot(-l.co, ray_D) < l.cos_angle:
        return ls
    ls.D = ray_D
    ls.t = FLT_MAX
    ls.pdf = l.pdf
    ls.L = l.strength * l.eval_fac
    ls.use_mis = l.use_mis
    ls.valid = 1
    return ls


# ---------------------------------------------------------------------------------------------- host side
def load_hdr(path):
    """Radiance .hdr (RGBE) -> float32 (H, W, 3), first row = top of the file, with the built-in reader (some
    imageio plugins decode .hdr as 8-bit). Other formats (EXR) go through imageio when it is installed."""
    if str(path).lower().endswith(".hdr"):
        return _read_rgbe(path)
    import imageio.v3 as iio
    a = iio.imread(path)
    return np.ascontiguousarray(a[..., :3], dtype=np.float32)


def _read_rgbe(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    h = w = None
    while True:
        e = data.index(b"\n", pos)
        line = data[pos:e]
        pos = e + 1
        if line.startswith(b"-Y") or line.startswith(b"+Y"):
            p = line.split()
            h, w = int(p[1]), int(p[3])
            flip_y = line.startswith(b"+Y")
            break
    rgbe = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        if data[pos] == 2 and data[pos + 1] == 2 and w >= 8 and (data[pos + 2] << 8 | data[pos + 3]) == w:
            pos += 4
            for c in range(4):
                x = 0
                while x < w:
                    n = data[pos]
                    pos += 1
                    if n > 128:
                        n -= 128
                        rgbe[y, x:x + n, c] = data[pos]
                        pos += 1
                    else:
                        rgbe[y, x:x + n, c] = np.frombuffer(data, np.uint8, n, pos)
                        pos += n
                    x += n
        else:
            rgbe[y] = np.frombuffer(data, np.uint8, w * 4, pos).reshape(w, 4)
            pos += w * 4
    e = rgbe[..., 3].astype(np.int32)
    f = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
    out = rgbe[..., :3].astype(np.float32) * f[..., None]
    return out[::-1].copy() if flip_y else out


def build_cdfs(pixels_bottom_up):
    """scene/light.cpp background_cdf + marginal CDF on a (H, W, 3) image whose row 0 is v = 0.
    Returns cond (H, W+1, 2), marg (H+1, 2), cdf_total (float32 semantics, float64 accumulation)."""
    H, W, _ = pixels_bottom_up.shape
    sin_theta = np.sin(np.float32(np.pi) * (np.arange(H, dtype=np.float32) + 0.5) / np.float32(H)).astype(np.float32)
    lum = np.abs(pixels_bottom_up).mean(axis=2).astype(np.float32) * sin_theta[:, None]      # (H, W)
    cond = np.zeros((H, W + 1, 2), np.float32)
    cond[:, :W, 0] = lum
    cum = np.concatenate([np.zeros((H, 1)), np.cumsum(lum[:, :-1].astype(np.float64) / W, axis=1)], axis=1)
    total = cum[:, -1] + lum[:, -1].astype(np.float64) / W
    cond[:, :W, 1] = np.where(total[:, None] > 0, cum / np.where(total[:, None] > 0, total[:, None], 1.0), cum)
    cond[:, W, 0] = total
    cond[:, W, 1] = 1.0
    cond[:, 0, 1] = 0.0
    marg = np.zeros((H + 1, 2), np.float32)
    marg[:H, 0] = total
    mcum = np.concatenate([[0.0], np.cumsum(total[:-1] / H)])
    cdf_total = mcum[-1] + total[-1] / H
    marg[:H, 1] = mcum / cdf_total if cdf_total > 0 else mcum
    marg[H, 0] = cdf_total
    marg[H, 1] = 1.0
    return cond, marg, float(cdf_total)


def make_world(image_top_first, strength=1.0, rotation=None, use_mis=True, device=None):
    """WorldMap from an equirect image as stored in a file (first row = top). `rotation` is the 3x3 matrix mapping
    world directions into the texture frame (Mapping node), default identity. Row order is flipped so that v = 0
    is the bottom row, as in Blender/Cycles."""
    img = np.ascontiguousarray(image_top_first[::-1], dtype=np.float32) * np.float32(strength)
    H, W, _ = img.shape
    cond, marg, total = build_cdfs(img)
    w = WorldMap()
    w.tex = wp.array(img, dtype=wp.vec3, device=device)
    w.cond = wp.array(cond, dtype=wp.vec2, device=device)
    w.marg = wp.array(marg, dtype=wp.vec2, device=device)
    w.width = W
    w.height = H
    w.use_mis = int(use_mis)
    r = np.eye(3) if rotation is None else np.asarray(rotation, np.float64).reshape(3, 3)
    w.w2t = wp.mat33(*r.flatten())
    w.t2w = wp.mat33(*np.linalg.inv(r).flatten())
    w.average_radiance = float(total * np.pi / 2.0)
    return w


def make_distant_light(matrix_world, color, strength_w_m2, angle_rad=np.deg2rad(0.526), normalize=True,
                       exposure=0.0, use_mis=True):
    """Sun. Cycles stores the DIRECTION OF TRAVEL (-Z column of matrix_world) and the FULL angle in Blender
    (`angle`); the kernel uses the half angle (scene/light.cpp:1286-1303)."""
    m = np.asarray(matrix_world, np.float64).reshape(4, 4)
    d = -m[:3, 2]
    d = d / np.linalg.norm(d)
    half = angle_rad / 2.0
    omc = 2.0 * np.sin(0.5 * half) ** 2
    area = np.pi * np.sin(half) ** 2 if half > 0 else 1.0
    s = DistantLight()
    s.co = wp.vec3(*d)
    s.cos_angle = float(np.cos(half))
    s.one_minus_cosangle = float(omc)
    s.pdf = float(M_1_2PI_NP / omc) if half > 0 else 1.0
    s.eval_fac = float(1.0 / area) if normalize else 1.0
    s.angle = float(half)
    s.strength = wp.vec3(*(np.asarray(color, np.float64) * strength_w_m2 * 2.0 ** exposure))
    s.use_mis = int(bool(use_mis and half > 0))
    return s


M_1_2PI_NP = 1.0 / (2.0 * np.pi)
