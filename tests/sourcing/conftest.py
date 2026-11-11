# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the sourcing tests: a throwaway alloy-assets tree with one importer library."""

from __future__ import annotations

from pathlib import Path

import pytest

MATERIAL = """\
schema: alloy.material/1
name: demo_ceramic
density: {value: 2300, source: model, unc: [2000, 2500], note: ESTIMATE stoneware prior}
surface:
  mu_static: {value: 0.5, source: catalog, unc: [0.4, 0.6], note: "DOC: 'glazed 0.5' https://example.org/table"}
  mu_dynamic: 0.4
  restitution: {value: 0.5, source: catalog, note: Serway Table 5.1}
"""

OBJECT = """\
schema: alloy.object/1
name: {name}
shape:
  primitive: {{kind: box, size: [0.04, 0.04, 0.05]}}
material: demo_ceramic
"""


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    """``<tmp>/imports/demo`` with a material and two objects (``mug_white``, ``bowl_red``)."""
    lib = tmp_path / "imports" / "demo"
    (lib / "materials").mkdir(parents=True)
    (lib / "objects").mkdir()
    (lib / "materials" / "demo_ceramic.yaml").write_text(MATERIAL)
    for name in ("mug_white", "bowl_red"):
        (lib / "objects" / f"{name}.yaml").write_text(OBJECT.format(name=name))
    return tmp_path
