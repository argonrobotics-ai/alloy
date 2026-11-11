# SPDX-License-Identifier: Apache-2.0
"""White balance as Blender 4.5 applies it, from the published methods: a scene-linear 3x3 matrix that adapts the
white point of a (temperature, tint) pair to D65 with the Bradford transform.

- Temperature and tint <-> chromaticity: Robertson's method (A. R. Robertson, "Computation of correlated color
  temperature and distribution temperature", JOSA 58, 1968) on his table of isotemperature lines (CIE 1960 u, v and
  slope t at 0-600 mired, as tabulated in Wyszecki & Stiles, Color Science, 2nd ed.). Tint moves the point along the
  isotemperature line, 1 unit = 1/3000 in u, v (Blender's scale).
- Chromatic adaptation: the Bradford cone-response matrix (Lam 1985), applied in CIE XYZ.
- Scene linear is Rec.709 / sRGB primaries with a D65 white (IEC 61966-2-1 matrix).

Example:
    >>> import numpy as np
    >>> m = white_point_matrix(*temp_tint_from_whitepoint_rgb((1.0, 1.0, 1.0)))
    >>> bool(np.allclose(m, np.eye(3), atol=1e-6))
    True
"""

from __future__ import annotations

import numpy as np

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
    """CIE XYZ (Y = 1) of a correlated colour temperature in kelvin and a tint offset along its isotemperature line.

    Example:
        >>> [round(float(c), 3) for c in whitepoint_xyz(6504.0, 0.0)]
        [0.969, 1.0, 1.122]
    """
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


def temp_tint_from_whitepoint_rgb(rgb) -> tuple[float, float]:
    """(temperature K, tint) of a scene-linear Rec.709 white point, by Robertson's method.

    Example:
        >>> t, tint = temp_tint_from_whitepoint_rgb((1.0, 1.0, 1.0))
        >>> round(t), round(tint, 1)
        (6502, 9.8)
    """
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
    return 1e6 / (lo[0] + f * (hi[0] - lo[0])), float(tint)


def bradford(from_xyz: np.ndarray, to_xyz: np.ndarray) -> np.ndarray:
    """Bradford adaptation matrix (XYZ -> XYZ) from one white to another.

    Example:
        >>> import numpy as np
        >>> w = np.array([0.95, 1.0, 1.09])
        >>> bool(np.allclose(bradford(w, w), np.eye(3)))
        True
    """
    fl, tl = BRADFORD @ (from_xyz / from_xyz[1]), BRADFORD @ (to_xyz / to_xyz[1])
    return np.linalg.inv(BRADFORD) @ np.diag(tl / fl) @ BRADFORD


def white_point_matrix(temperature: float, tint: float) -> np.ndarray:
    """Scene-linear Rec.709 3x3 (applied to column vectors) that maps the given white to the scene's D65 white.

    Example:
        >>> white_point_matrix(5000.0, 0.0).shape
        (3, 3)
    """
    to_xyz = np.linalg.inv(XYZ_TO_REC709)
    return XYZ_TO_REC709 @ bradford(whitepoint_xyz(temperature, tint), to_xyz @ np.ones(3)) @ to_xyz
