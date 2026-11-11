# SPDX-License-Identifier: Apache-2.0
"""Alloy import and export: scene and object cards to MJCF, URDF and USD, and those files back into cards.

Example::

    from alloy.io import export_scene, import_file
    res = export_scene("demo_peg_d12", "out/peg")      # MJCF + URDF + USD, checked by loading them
    print(res.to_json()["files"])
    cards = import_file("robot_scene.xml", "my_library")  # objects, materials, robots and a scene card

Command line (JSON out): ``python -m alloy.io export --scene demo_peg_d12 --out out/peg`` and
``python -m alloy.io import robot_scene.xml --out my_library``.
"""

from alloy.io.api import FORMATS, ExportResult, export_object, export_scene
from alloy.io.errors import ExportError, ImportFileError
from alloy.io.importer import import_file, read_file
from alloy.io.model import IOModel, Unmapped
from alloy.io.to_cards import ImportResult

__all__ = ["FORMATS", "ExportError", "ExportResult", "IOModel", "ImportFileError", "ImportResult", "Unmapped",
           "export_object", "export_scene", "import_file", "read_file"]
