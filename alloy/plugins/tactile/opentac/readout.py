# SPDX-License-Identifier: Apache-2.0
"""Taxel force to 12-bit ADC counts and back: OpenTac's velostat readout profiles, ported.

Port of ``opentac/sim/readout.py`` and ``opentac/sim/profiles/v2.json``, ``v3.json`` (github.com/opentac/opentac,
commit e302861), MIT licence, Copyright (c) 2026 The OpenTac authors (``LICENSE-OPENTAC`` beside this file):

    counts = min(adc_max, gain * a * p ** b),   p = F / taxel_area  (kPa),   a = k * taxel_area_mm2 ** s

with ``taxel_area`` the pitch area (px * py), as in their ``_taxel_area_mm2``. The inverse (``forces``) zeroes counts
under 3x the read noise and returns a lower bound at ``adc_max``.

What a profile does not capture (their words, readout.py): per-taxel gain spread; the bump sheet (calibrated by a
flat press through foam, so under a TPU bump with b < 1 a taxel-mean pressure overestimates the counts); anything
above ``calibrated_kpa``; ghosting and scan timing. Their hysteresis is measured only as an unload/load ratio of the
gain (``hysteresis_unload_over_load``), not as a model, and drift is not in the profiles: neither is applied here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    """One velostat calibration. ``gain_range`` is their spread across presses + hysteresis (for randomising)."""

    name: str
    material: str
    b: float                          # pressure exponent
    k: float                          # a = k * area_mm2 ** s, counts / kPa^b
    s: float                          # area exponent of the gain
    adc_max: float                    # counts where the board's ADC tops out
    noise_counts: float               # per-taxel read noise std at rest
    gain_range: tuple[float, float]
    calibrated_kpa: tuple[float, float]
    source: str


_SRC = ("opentac/sim/profiles/{}.json at e302861 (fitted 2026-09-14: power law in taxel pressure fitted to the "
        "frame-by-frame force balance against a load cell; manual vertical press through a foam layer, no bump sheet; "
        "pitches 1.5, 2.0, 4.0 mm)")

# Values copied from the JSON at full precision. MIT, Copyright (c) 2026 The OpenTac authors.
PROFILES: dict[str, Profile] = {
    "v2": Profile(name="v2", material="LINQSTAT", b=1.2500000000000009, k=1.4502055004663603, s=1.5083984676132116,
                  adc_max=3884.0, noise_counts=4.298846000293919, gain_range=(0.7950466873468006, 2.3423950029284475),
                  calibrated_kpa=(1.154627294238359, 108.91641447314107), source=_SRC.format("v2")),
    "v3": Profile(name="v3", material="Yunjia Textiles", b=0.5500000000000003, k=15.197462663862735,
                  s=0.7849893324437843, adc_max=3884.0, noise_counts=4.2477868171159185,
                  gain_range=(0.9173827123687528, 1.7155431298884631),
                  calibrated_kpa=(0.039423394831414375, 215.13686384078557), source=_SRC.format("v3")),
}


def profile(name: str) -> Profile:
    """A shipped profile by label ("v2", "v3") or material ("LINQSTAT", "Yunjia Textiles", or its first word)."""
    q = name.strip().lower()
    for label, p in PROFILES.items():
        m = p.material.lower()
        if q in (label, m) or q == m.split()[0]:
            return p
    shipped = ", ".join(f"{k} ({v.material})" for k, v in PROFILES.items())
    raise ValueError(f"no OpenTac readout profile {name!r}; shipped: {shipped}")


def taxel_area_mm2(pitch: tuple[float, float]) -> float:
    """Pitch area in mm2 from a pitch in metres; refuses anything not plausibly metres (as their code does)."""
    px, py = pitch
    if not (1e-4 <= px <= 2e-2 and 1e-4 <= py <= 2e-2):
        raise ValueError(f"taxel pitch {px}, {py} is not in metres (expected 0.1-20 mm, given as 1e-4-2e-2)")
    return px * py / 1e-6


def gain_a(p: Profile, pitch: tuple[float, float]) -> float:
    """counts / kPa^b for one taxel of this pitch."""
    return p.k * taxel_area_mm2(pitch) ** p.s


def counts(forces_n: np.ndarray, pitch: tuple[float, float], p: Profile, gain: float = 1.0,
           quantize: bool = True) -> np.ndarray:
    """Counts above baseline for per-taxel forces (N). ``quantize`` rounds to whole counts, as the board streams."""
    if gain <= 0 or not np.isfinite(gain):
        raise ValueError("readout gain must be positive and finite")
    f = np.asarray(forces_n, dtype=float)
    kpa = np.clip(f / taxel_area_mm2(pitch) * 1e3, 0.0, None)
    c = np.clip(gain * gain_a(p, pitch) * kpa ** p.b, 0.0, p.adc_max)
    return np.round(c) if quantize else c


def forces(counts_: np.ndarray, pitch: tuple[float, float], p: Profile, gain: float = 1.0) -> np.ndarray:
    """The inverse: N per taxel from counts. Under 3x read noise reads 0; at ``adc_max`` it is a lower bound."""
    c = np.asarray(counts_, dtype=float)
    c = np.where(c > 3.0 * p.noise_counts, c, 0.0)
    kpa = (c / (gain * gain_a(p, pitch))) ** (1.0 / p.b)
    return kpa * taxel_area_mm2(pitch) * 1e-3


def saturation_force_n(pitch: tuple[float, float], p: Profile, gain: float = 1.0) -> float:
    """Force on one taxel at which it reads ``adc_max``."""
    kpa = (p.adc_max / (gain * gain_a(p, pitch))) ** (1.0 / p.b)
    return kpa * taxel_area_mm2(pitch) * 1e-3


def floor_force_n(pitch: tuple[float, float], p: Profile, gain: float = 1.0) -> float:
    """Force on one taxel at which it reads 3x the read noise: below it the decoder reads 0."""
    kpa = (3.0 * p.noise_counts / (gain * gain_a(p, pitch))) ** (1.0 / p.b)
    return kpa * taxel_area_mm2(pitch) * 1e-3
