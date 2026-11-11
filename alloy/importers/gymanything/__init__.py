# SPDX-License-Identifier: Apache-2.0
"""Import the object assets of Gym-Anything for Robotics (MIT) as Alloy object cards, keeping their citations.

Two stages, because Gym-Anything builds most objects in Python code that needs its own Python 3.12 env:

1. **Export** (their env): ``tools/gymanything_export/ex_<sim>.py`` (outside the package: it imports
   Gym-Anything and trimesh, not Alloy) calls their builders and writes one ``gar.spec/1`` JSON + a
   watertight OBJ per object (``spec.py`` describes the format).
2. **Import** (Alloy's env): ``pipeline.import_specs`` turns specs into ``alloy.material/1`` + ``alloy.object/1``
   cards in a library root, then checks each with Alloy's card validator, the mass Newton computes, and the cited
   dimensions.

Example::

    from pathlib import Path
    from alloy.cards.library import CardLibrary, default_root
    from alloy.importers.gymanything import import_specs

    summary = import_specs(Path("work/specs"), Path("work/gar"))        # JSON-able: converted, checked_ok, failed
    lib = CardLibrary(roots=[default_root(), Path("work/gar")])
    glass = lib.object("gar_wine_pouring_glass_bordeaux")                # 0.21 kg, datasheet-cited, SDF collision

Command line (JSON out): ``python -m alloy.importers.gymanything import --specs DIR --out DIR``.
"""

from __future__ import annotations

from alloy.importers.gymanything.pipeline import import_specs
from alloy.importers.gymanything.spec import GarSpec, SpecError, load_spec, parse_spec

__all__ = ["GarSpec", "SpecError", "import_specs", "load_spec", "parse_spec"]
