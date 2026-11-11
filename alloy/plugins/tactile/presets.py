# SPDX-License-Identifier: Apache-2.0
"""Taxel layouts of real tactile sensors, from published sources only.

A preset fills the layout fields of ``Tactile.Config`` (``grid``, ``pitch``, ``points``, ``reach``) and
``saturation``. Every number below names its source. Where sources conflict, the conflict is written next to the
value. A sensor whose layout no source gives has no preset: pass its taxel points yourself.

Not included, and why (2026-11-06):
- **BioTac** (19 electrodes): the electrode coordinates are in Lin et al. 2013, Table 1. That PDF was not
  reachable, and no other source lists them, so there is no preset until the table is transcribed.
- **Inspire RH56E2** pads: a retailer gives the counts and ranges (17 resistive points at 0-30 N, or 5 capacitive
  at 0-20 N), but no layout.

Example::

    Tactile(bodies=("left_finger", "right_finger"), preset="xela_uskin_4x4", origins=((0, 0, 0.01),) * 2)
"""

from __future__ import annotations

from dataclasses import dataclass

GF = 9.80665e-3  # N per gram-force


@dataclass(frozen=True)
class Preset:
    """One real sensor's layout. ``points`` are taxel centres in the pad plane (m); a ``grid`` generates them."""

    name: str
    grid: tuple[int, int] = (0, 0)                       # nx, ny; (0, 0) when ``points`` are given
    pitch: tuple[float, float] = (0.0, 0.0)              # m between taxel centres
    points: tuple[tuple[float, float], ...] = ()         # m, pad-plane taxel centres
    reach: float = 0.0                                   # m; 0 = Tactile's default from the layout
    saturation: float = 0.0                              # N per channel; 0 = none
    rate_hz: float = 0.0                                 # the real sensor's sample rate (information only)
    source: str = ""


_RESKIN_S = 0.007   # "spaced 7mm apart around a central magnetometer"

PRESETS: dict[str, Preset] = {
    "xela_uskin_4x4": Preset(
        name="xela_uskin_4x4", grid=(4, 4), pitch=(0.005, 0.005), saturation=1800 * GF, rate_hz=100.0,
        source="XELA uSPa 44 / XR1944: 16 taxels, 3-axis, 22.6 x 24.6 x 5.5 mm package, sensing points about 4 x 4 "
               "mm (xelarobotics.com/products/uspa-44); pitch 5 mm = the ~20 x 20 mm sensing area of a 4x4 module "
               "(arXiv 2410.08619); 100 Hz per the XR1944 manual (the vendor's family page gives 500 Hz, the uSPa 46 "
               "page 330 Hz); range 1-1800 gf from an older academic description (core.ac.uk 201006242). The sources "
               "conflict on rate and range."),
    "reskin": Preset(
        name="reskin",
        points=((0.0, 0.0), (_RESKIN_S, 0.0), (-_RESKIN_S, 0.0), (0.0, _RESKIN_S), (0.0, -_RESKIN_S)),
        reach=0.0105, rate_hz=400.0,
        source="Bhirangi et al. 2021 (arXiv 2111.00071): 5 magnetometers (MLX90393), four 'spaced 7mm apart around "
               "a central magnetometer', 20 x 20 mm sensing area, 2-3 mm skin, about 400 Hz. Unstated, and assumed "
               "here: a plus pattern at 7 mm from the centre. reach 10.5 mm covers the 20 x 20 mm area. A magnetometer "
               "responds over an area: set ``spread`` to model it (no source gives the width)."),
}


def preset(name: str) -> Preset:
    """The preset called ``name``. Example: ``preset("reskin").points``."""
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"no tactile preset {name!r}; presets: {sorted(PRESETS)} (BioTac and Inspire have no "
                         f"sourced layout: pass points)") from None
