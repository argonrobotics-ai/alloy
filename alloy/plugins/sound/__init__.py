# SPDX-License-Identifier: Apache-2.0
"""Sound (engine item 10, part 5): ``ModalSound`` modal synthesis from contact impacts and forces, the mode tables
(``modes.py``, ``presets.py``), the Hertz pulse (``hertz.py``) and audio analysis (``analysis.py``). See
``sensor.py`` for the model and ``laws.py`` for the laws S-1..S-6."""

from alloy.plugins.sound.modes import Material, ModeSet, beam_modes, plate_modes
from alloy.plugins.sound.presets import MATERIALS, PRESETS, material, preset
from alloy.plugins.sound.sensor import ModalSound

__all__ = ["MATERIALS", "PRESETS", "Material", "ModalSound", "ModeSet", "beam_modes", "material", "plate_modes",
           "preset"]
