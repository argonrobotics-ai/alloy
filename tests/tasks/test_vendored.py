# SPDX-License-Identifier: Apache-2.0
"""Every vendored file under library/robots and every saved trace is checked against its recorded sha256."""

from __future__ import annotations

import json

import mujoco
import pytest

from alloy.core.ids import sha256_hex
from alloy.tasks.msh import MANIFEST_NAME, check_manifest
from alloy.tasks.trace import TRACE_JSON, load_trace
from alloy.testing.tasks import ROBOTS, TRACE_NAMES, TRACES


def test_robots_manifest_matches_every_file() -> None:
    assert (ROBOTS / MANIFEST_NAME).is_file()
    assert check_manifest(ROBOTS) == []


@pytest.mark.parametrize("scene", sorted(p.name for p in ROBOTS.glob("task_*")))
def test_vendored_scene_compiles(scene: str) -> None:
    m = mujoco.MjModel.from_xml_path(str(ROBOTS / scene / "scene.xml"))
    assert m.nbody > 1 and m.ngeom > 0


@pytest.mark.parametrize("name", TRACE_NAMES)
def test_trace_hashes_and_scene_binding(name: str) -> None:
    tt = load_trace(TRACES / name)  # checks the npz sha256 and the inputs digest
    scene = ROBOTS.parent.parent / json.loads((TRACES / name / TRACE_JSON).read_text())["meta"]["scene"]
    assert tt.trace.scene_hash == sha256_hex(scene.read_bytes())
    assert tt.trace.periods > 0 and tt.ref.pos.shape[0] * tt.meta["ref_every"] <= tt.trace.periods
