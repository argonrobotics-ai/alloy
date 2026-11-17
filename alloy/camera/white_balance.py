# SPDX-License-Identifier: Apache-2.0
"""White balance as Blender 4.5 applies it (a look's ``render.white_balance``): a scene-linear 3x3 matrix, before the
view transform, that adapts the light's white to the D65 white of the image, so a grey under that light shows grey.

- ``"auto"`` (Blender's default): the colour of the light reaching the subject, measured as Blender does: a probe
  render of a 1 cm grey diffuse ball at the subject, alone under the look's lights and world, seen from the camera's
  side (32 x 32 px, 64 samples; the mean of the centre 8 x 8 pixels). If the probe fails, Blender's analytic
  estimate (each look light by its power, colour temperature and distance, the world by its mean colour).
- ``{"kelvin": K, "tint": t}``: that white (tint 10 when left out, Blender's default); ``"off"``: none.
- Temperature and tint <-> chromaticity: Robertson's method (A. R. Robertson, "Computation of correlated color
  temperature and distribution temperature", JOSA 58, 1968) on his isotemperature lines (CIE 1960 u, v and slope at
  0-600 mired, Wyszecki & Stiles, Color Science, 2nd ed.); tint moves along the line, 1 unit = 1/3000 in u, v
  (Blender's scale). Chromatic adaptation: the Bradford cone-response matrix (Lam 1985) in CIE XYZ. Scene linear is
  Rec.709 primaries with a D65 white (IEC 61966-2-1). The render gate's display transform uses the same method.

Example::

    import numpy as np
    from alloy.camera.white_balance import view_white, white_matrix
    vs = view_white({"render": {"white_balance": {"kelvin": 3000}}}, target=(0, 0, 0))
    m = white_matrix(vs)                      # 3x3, applied to linear RGB column vectors
    assert m is not None and m[2, 2] > m[0, 0]  # a warm light's white: blue raised against red
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .errors import CameraError

log = logging.getLogger(__name__)
# Blender's probe: image px, samples, ball radius m, distance m, lens mm (alloy/render/blender/colour.py)
PROBE_PX, PROBE_SPP, PROBE_R, PROBE_DIST, PROBE_LENS = 32, 64, 0.01, 0.12, 100.0
PROBE_MAX_SIZE = 15.0  # metres: meshes larger than this (ground planes) do not move the subject, as in auto framing
DEFAULT_TINT = 10.0  # Blender's tint for a temperature given alone (its UI default)
XYZ_TO_REC709 = np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
TINT_SCALE = 3000.0
# Robertson's isotemperature lines: mired, u, v, slope t (Wyszecki & Stiles).
ROBERTSON = np.array([
    [0, .18006, .26352, -.24341], [10, .18066, .26589, -.25479], [20, .18133, .26846, -.26876],
    [30, .18208, .27119, -.28539], [40, .18293, .27407, -.30470], [50, .18388, .27709, -.32675],
    [60, .18494, .28021, -.35156], [70, .18611, .28342, -.37915], [80, .18740, .28668, -.40955],
    [90, .18880, .28997, -.44278], [100, .19032, .29326, -.47888], [125, .19462, .30141, -.58204],
    [150, .19962, .30921, -.70471], [175, .20525, .31647, -.84901], [200, .21142, .32312, -1.0182],
    [225, .21807, .32909, -1.2168], [250, .22511, .33439, -1.4512], [275, .23247, .33904, -1.7298],
    [300, .24010, .34308, -2.0637], [325, .24792, .34655, -2.4681], [350, .25591, .34951, -2.9641],
    [375, .26400, .35200, -3.5814], [400, .27218, .35407, -4.3633], [425, .28039, .35577, -5.3762],
    [450, .28863, .35714, -6.7262], [475, .29685, .35823, -8.5955], [500, .30505, .35907, -11.324],
    [525, .31320, .35968, -15.628], [550, .32129, .36011, -23.325], [575, .32931, .36038, -40.770],
    [600, .33724, .36051, -116.45]])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def whitepoint_xyz(temperature: float, tint: float) -> np.ndarray:
    """CIE XYZ (Y = 1) of a colour temperature in kelvin and a tint along its isotemperature line."""
    mired = float(np.clip(1e6 / temperature, ROBERTSON[0, 0], ROBERTSON[-1, 0]))
    i = max(int(np.searchsorted(ROBERTSON[:, 0], mired, side="left")), 1)
    lo, hi = ROBERTSON[i - 1], ROBERTSON[i]
    f = (mired - lo[0]) / (hi[0] - lo[0])
    uv = lo[1:3] + f * (hi[1:3] - lo[1:3])
    iso = _unit(_unit(np.array([1.0, lo[3]])) * (1 - f) + _unit(np.array([1.0, hi[3]])) * f)
    u, v = uv - iso * tint / TINT_SCALE
    d = 2 * u - 8 * v + 4
    x, y = 3 * u / d, 2 * v / d
    return np.array([x / y, 1.0, (1 - x - y) / y])


def temp_tint(rgb) -> tuple[float, float]:
    """(temperature K, tint) of a scene-linear Rec.709 white, by Robertson's method."""
    w = np.linalg.inv(XYZ_TO_REC709) @ np.asarray(rgb, float)
    uv = np.array([4 * w[0], 6 * w[1]]) / (w @ [1, 15, 3])

    def dist(row: np.ndarray) -> float:
        return (uv[1] - row[2]) - row[3] * (uv[0] - row[1])

    d = np.array([dist(r) for r in ROBERTSON])
    i = max(int(np.argmax(d < 0.0)), 1)
    lo, hi = ROBERTSON[i - 1], ROBERTSON[i]
    dl, dh = dist(lo) / np.hypot(1, lo[3]), dist(hi) / np.hypot(1, hi[3])
    f = dl / (dl - dh)
    uvt = lo[1:3] + f * (hi[1:3] - lo[1:3])
    tint = np.linalg.norm(uv - uvt) * TINT_SCALE * (1.0 if uv[0] < uvt[0] else -1.0)
    return float(1e6 / (lo[0] + f * (hi[0] - lo[0]))), float(tint)


def _bradford(from_xyz: np.ndarray, to_xyz: np.ndarray) -> np.ndarray:
    fl, tl = BRADFORD @ (from_xyz / from_xyz[1]), BRADFORD @ (to_xyz / to_xyz[1])
    return np.linalg.inv(BRADFORD) @ np.diag(tl / fl) @ BRADFORD


def white_point_matrix(temperature: float, tint: float) -> np.ndarray:
    """Scene-linear Rec.709 3x3 (for column vectors) that maps the given white to the image's D65 white."""
    to_xyz = np.linalg.inv(XYZ_TO_REC709)
    return XYZ_TO_REC709 @ _bradford(whitepoint_xyz(temperature, tint), to_xyz @ np.ones(3)) @ to_xyz


def _mode(look: dict):
    """The look's ``render.white_balance``, "auto" when left out (Blender's default), checked as Blender does."""
    wb = (look.get("render") or {}).get("white_balance", "auto")
    if wb in ("off", False, None) or wb == "auto":
        return "off" if wb in ("off", False, None) else "auto"
    if isinstance(wb, dict) and float(wb.get("kelvin", 6500.0)) >= 1000.0:
        return wb
    raise CameraError(f"render.white_balance is {wb!r}",
                      "use 'auto', 'off' or {'kelvin': K, 'tint': t} with K at least 1000")


def light_white(look: dict, target, world_mean) -> np.ndarray:
    """Linear RGB of the light reaching ``target`` (Blender's analytic estimate): the look lights by power, colour
    temperature and distance, plus the world (pi x its mean radiance ``world_mean`` x strength); max channel 1."""
    from alloy.render.blender.motion import scene_illuminant
    from .look_world import world_section
    light = scene_illuminant(list(look.get("lights") or []), None if world_mean is None else np.asarray(world_mean),
                             float(world_section(look)["strength"]), np.asarray(target, float))
    if not np.all(np.isfinite(light)) or float(light.max()) <= 0.0:
        return np.ones(3)
    return light / float(light.max())


def view_white(look: dict, target, world_mean=None, light=None) -> dict:
    """Blender ``view_settings`` keys for the look's white balance: ``use_white_balance`` and, when on,
    ``white_balance_temperature``, ``white_balance_tint`` and ``white_balance_whitepoint`` (linear RGB, max 1).
    "auto" takes the measured ``light`` (linear RGB, :func:`probe_light`) when given, else the analytic estimate at
    ``target`` with the world's mean radiance ``world_mean``."""
    mode = _mode(look)
    if mode == "off":
        return {"use_white_balance": False}
    if mode == "auto":
        light = None if light is None else np.asarray(light, float)
        ok = light is not None and np.all(np.isfinite(light)) and float(light.max()) > 0.0
        white = light / float(light.max()) if ok else light_white(look, target, world_mean)
        temperature, tint = temp_tint(white)
    else:
        temperature, tint = float(mode.get("kelvin", 6500.0)), float(mode.get("tint", DEFAULT_TINT))
        rgb = XYZ_TO_REC709 @ whitepoint_xyz(temperature, tint)
        white = rgb / float(rgb.max())
    return {"use_white_balance": True, "white_balance_temperature": round(temperature, 3),
            "white_balance_tint": round(tint, 4), "white_balance_whitepoint": [float(c) for c in white]}


def white_matrix(view_settings: dict | None) -> np.ndarray | None:
    """The 3x3 white-balance matrix of Blender ``view_settings`` (temperature and tint, else the white point), or
    None when ``use_white_balance`` is off."""
    vs = view_settings or {}
    if not vs.get("use_white_balance"):
        return None
    if vs.get("white_balance_temperature") is not None:
        t, tint = float(vs["white_balance_temperature"]), float(vs.get("white_balance_tint", DEFAULT_TINT))
    else:
        t, tint = temp_tint(vs.get("white_balance_whitepoint", (1.0, 1.0, 1.0)))
    return white_point_matrix(t, tint)


def subject(look: dict, meshes: list) -> tuple[np.ndarray, np.ndarray]:
    """(target, eye) "auto" measures the light at and from: the look's first camera key (as Blender does), else the
    default camera's three-quarter view of ``meshes`` (pack meshes; ground planes left out, as in auto framing)."""
    from .pose import auto_frame
    for cam in (look.get("cameras") or {}).values():
        keys = cam.get("keys") if isinstance(cam, dict) else None
        if keys and keys[0].get("target") is not None and keys[0].get("eye") is not None:
            return np.asarray(keys[0]["target"], float), np.asarray(keys[0]["eye"], float)
    boxes = [(m.v.min(0), m.v.max(0)) for m in meshes if len(m.v)]
    boxes = [b for b in boxes if float((b[1] - b[0]).max()) <= PROBE_MAX_SIZE] or [(np.zeros(3), np.ones(3))]
    lo, hi = np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)
    xf = auto_frame(lo.astype(float), hi.astype(float))
    return 0.5 * (lo + hi).astype(float), np.asarray(xf[:3], float)


def probe_light(look: dict, target, eye, device: str) -> np.ndarray:
    """Mean linear RGB of the lit side of a 1 cm grey (0.18) diffuse ball at ``target``, alone under the look's lights
    and world, seen from ``eye``'s side through a 100 mm lens at 12 cm (Blender's white-balance probe)."""
    import warp as wp
    from alloy._vendor import newton
    from .camera import Camera, camera_scene
    target, eye = np.asarray(target, float), np.asarray(eye, float)
    b = newton.ModelBuilder()
    b.add_shape_sphere(-1, xform=wp.transform(wp.vec3(*target), wp.quat_identity()), radius=PROBE_R, label="awb_probe")
    model = b.finalize(device=device)
    probe = {"render": {**(look.get("render") or {}), "white_balance": "off"}, "world": look.get("world") or {},
             "lights": list(look.get("lights") or []), "assign": [{"match": ".*", "material": "awb_probe"}],
             "materials": {"awb_probe": {"base_color": [0.18, 0.18, 0.18], "roughness": 1.0, "specular": 0.0}}}
    view = (target - eye) / max(float(np.linalg.norm(target - eye)), 1e-9)
    fov = math.degrees(2.0 * math.atan(18.0 / PROBE_LENS))  # a 36 mm sensor, square image
    cam = Camera("awb_probe", look_at=(tuple(target - view * PROBE_DIST), tuple(target)), width=PROBE_PX,
                 height=PROBE_PX, fov=fov, spp=PROBE_SPP)
    lin = cam.render(camera_scene(model, look=probe, device=device), output="linear").numpy()[0]
    c = PROBE_PX // 2
    return lin[c - 4:c + 4, c - 4:c + 4].reshape(-1, 3).mean(axis=0).astype(float)


def look_view_white(look: dict, world: dict, meshes: list, device: str = "cpu") -> dict:
    """``view_white`` of a look as the full-material camera builds it: "auto" from :func:`probe_light` at the subject
    (:func:`subject`), else, if the probe fails, the analytic estimate with the world's mean radiance from its pack
    ``world`` (``look_world``; the studio gradient's as Blender takes it when the look names no HDRI)."""
    if _mode(look) != "auto":
        return view_white(look, (0.0, 0.0, 0.0))
    target, eye = subject(look, meshes)
    try:
        return view_white(look, target, light=probe_light(look, target, eye, device))
    except Exception as exc:  # noqa: BLE001 - Blender falls back to the analytic estimate too
        log.warning("camera: white balance probe failed (%s); using the analytic estimate", exc)
    from .look_world import STUDIO_MEAN, equirect_mean, world_section
    mean = equirect_mean(world["image"]) if world_section(look)["hdri"] else np.asarray(STUDIO_MEAN, float)
    return view_white(look, target, mean)
