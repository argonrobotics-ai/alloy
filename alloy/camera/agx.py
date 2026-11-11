# SPDX-License-Identifier: Apache-2.0
"""AgX view transform: linear scene radiance -> display, without the hard clip of the Standard view.

The formulation is Filament's AgX tone mapper (Apache-2.0, github.com/google/filament, filament/src/ToneMapper.cpp):
an inset matrix, a log2 encoding over [-12.47, 4.03] EV, a 7th-order sigmoid fit, an outset matrix and a 2.2 power
to linear, then the sRGB curve to bytes. Blender's AgX contrast looks ("AgX - Medium High Contrast", which C2/C3/C4
use, ...) are approximated as a contrast around middle grey in the log encoding (``CONTRAST``); they are not
Blender's look data, so images are close to Blender's AgX, not identical.

Example::

    import numpy as np
    from alloy.camera.agx import agx_u8
    img = agx_u8(np.full((2, 2, 3), 0.18, np.float32))          # mid grey -> about 123
    bright = agx_u8(np.full((2, 2, 3), 16.0, np.float32))        # 6 stops over: still below 255
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

# Filament's matrices, written row-major here (Filament lists them by column)
INSET = np.array([[0.856627153315983, 0.0951212405381588, 0.0482516061458583],
                  [0.137318972929847, 0.761241990602591, 0.101439036467562],
                  [0.11189821299995, 0.0767994186031903, 0.811302368396859]])
OUTSET = np.linalg.inv(np.array([[0.899796955911611, 0.0871996192028351, 0.013003424885555],
                                 [0.11142098895748, 0.875575586156966, 0.0130034248855548],
                                 [0.11142098895748, 0.0871996192028349, 0.801379391839686]]))
MIN_EV, MAX_EV = -12.47393, 4.026069
PIVOT = (math.log2(0.18) - MIN_EV) / (MAX_EV - MIN_EV)    # middle grey in the log encoding
POLY = (0.002857, -0.1718, 4.361, -28.72, 92.06, -126.7, 78.01, -17.86)   # sigmoid, x^0 .. x^7
# Blender's AgX look names -> contrast around middle grey (our approximation of each look's strength)
CONTRAST = {"None": 1.0, "AgX - Base Contrast": 1.0, "AgX - Medium High Contrast": 1.2, "AgX - High Contrast": 1.4,
            "AgX - Medium Low Contrast": 0.9, "AgX - Low Contrast": 0.8}

_INSET = wp.constant(wp.mat33(*INSET.flatten().tolist()))
_OUTSET = wp.constant(wp.mat33(*OUTSET.flatten().tolist()))
_MIN_EV, _SPAN, _PIVOT = wp.constant(MIN_EV), wp.constant(MAX_EV - MIN_EV), wp.constant(PIVOT)


def contrast_of(look: str | None) -> float:
    """The contrast of a Blender AgX look name (1.0 for None or an unknown look)."""
    return CONTRAST.get(look or "None", 1.0)


@wp.func
def _sigmoid(x: float) -> float:
    return ((((((-17.86 * x + 78.01) * x - 126.7) * x + 92.06) * x - 28.72) * x + 4.361) * x - 0.1718) * x + 0.002857


@wp.func
def agx(c: wp.vec3, contrast: float) -> wp.vec3:
    """Linear radiance -> AgX display-linear (0..1) RGB."""
    v = _INSET * wp.max(c, wp.vec3(0.0))
    s = wp.vec3(0.0)
    for k in range(3):
        e = (wp.log2(wp.max(v[k], 1.0e-10)) - _MIN_EV) / _SPAN
        e = wp.clamp(_PIVOT + (e - _PIVOT) * contrast, 0.0, 1.0)
        s[k] = _sigmoid(e)
    d = _OUTSET * s
    return wp.vec3(wp.pow(wp.max(d[0], 0.0), 2.2), wp.pow(wp.max(d[1], 0.0), 2.2), wp.pow(wp.max(d[2], 0.0), 2.2))


def agx_linear(rgb: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    """Host version of :func:`agx`: (..., 3) linear radiance -> display-linear RGB."""
    v = np.maximum(np.asarray(rgb, np.float64), 0.0) @ INSET.T
    e = (np.log2(np.maximum(v, 1e-10)) - MIN_EV) / (MAX_EV - MIN_EV)
    e = np.clip(PIVOT + (e - PIVOT) * contrast, 0.0, 1.0)
    s = sum(c * e ** k for k, c in enumerate(POLY))
    return np.power(np.maximum(s @ OUTSET.T, 0.0), 2.2).astype(np.float32)


def agx_u8(rgb: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    """(..., 3) linear radiance -> uint8 through AgX and the sRGB curve."""
    from .view import standard_u8  # noqa: PLC0415
    return standard_u8(agx_linear(rgb, contrast))
