# SPDX-License-Identifier: Apache-2.0
"""Acid-base readout of the species plug-in: pH of a strong acid and a strong base, and an indicator's colour.

The plug-in carries the totals that reactions conserve (acid and base equivalents, indicator), so neutralisation
(H+ + OH- -> H2O, fast against any step) needs no rate law: per particle, with a and b the acid and base
concentrations [mol/L] and Kw = 1e-14 at 25 C,
    [H+] = (d + sqrt(d^2 + 4 Kw)) / 2,   d = a - b,   pH = -log10 [H+]
(charge balance [H+] - [OH-] = a - b with [H+][OH-] = Kw). The indicator is a weak acid HIn (colourless) / In-
(coloured) with pKa: coloured fraction 1 / (1 + 10^(pKa - pH)). Phenolphthalein: pKa 9.3, colourless below about
pH 8.2 (fraction 0.07), pink by 10 (0.83). The indicator's own acidity is left out (it is ~1e-4 of the acid in a
titration). Weak acids, buffers, activity coefficients and temperature are out of scope.
"""

from __future__ import annotations

import numpy as np

KW = 1.0e-14  # water's ion product at 25 C [mol^2/L^2]
PHENOLPHTHALEIN_PKA = 9.3
PINK_PH = 8.2  # first faint pink


def ph(acid_mol_l: np.ndarray, base_mol_l: np.ndarray) -> np.ndarray:
    """pH of a strong acid at ``acid_mol_l`` and a strong base at ``base_mol_l`` [mol/L], elementwise."""
    d = np.asarray(acid_mol_l, np.float64) - np.asarray(base_mol_l, np.float64)
    s = np.sqrt(d * d + 4.0 * KW)
    h = np.where(d >= 0.0, 0.5 * (d + s), 2.0 * KW / (s - d))  # the second form avoids cancellation in excess base
    return -np.log10(h)


def coloured_fraction(ph_value: np.ndarray, pka: float = PHENOLPHTHALEIN_PKA) -> np.ndarray:
    """Fraction of an indicator of ``pka`` in its coloured (base) form at ``ph_value``."""
    return 1.0 / (1.0 + 10.0 ** (pka - np.asarray(ph_value, np.float64)))


def equivalence_volume(acid_mol_l: float, acid_volume: float, base_mol_l: float) -> float:
    """Base volume [same unit as ``acid_volume``] that neutralises the acid exactly."""
    return acid_mol_l * acid_volume / base_mol_l
