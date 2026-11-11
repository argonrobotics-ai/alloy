# SPDX-License-Identifier: Apache-2.0
"""Every file in tools/gymanything_export has an accepted licence in its LICENCES.yaml (release builder format)."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

EXPORT = Path(__file__).resolve().parents[2] / "tools" / "gymanything_export"
ACCEPTED = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "Zlib", "ISC",
            "PSF-2.0"}

pytestmark = pytest.mark.skipif(not (EXPORT / "LICENCES.yaml").is_file(), reason="exporters not in this checkout")


def _entries() -> list[dict]:
    data = yaml.safe_load((EXPORT / "LICENCES.yaml").read_text())
    assert data["schema"] == "alloy.licences/1"
    return data["entries"]


def test_every_exporter_file_has_an_accepted_licence():
    entries = _entries()
    for f in sorted(p for p in EXPORT.iterdir() if p.is_file() and p.name != "LICENCES.yaml" and p.suffix == ".py"):
        lic = next((e["licence"] for e in entries if fnmatch(f.name, e["path"])), None)
        assert lic is not None, f"{f.name}: add an entry to LICENCES.yaml (licence of the code it derives from)"
        ids = [t for t in lic.replace("(", " ").replace(")", " ").split() if t not in ("AND", "OR", "WITH")]
        assert ids and set(ids) <= ACCEPTED, f"{f.name}: licence {lic!r} is not accepted for release"


def test_exporters_name_their_upstream_source():
    for e in _entries():
        if e["path"].startswith("ex_"):
            assert e["licence"] == "MIT" and "gym-anything/robotics @" in e["source"], e
