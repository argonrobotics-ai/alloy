# SPDX-License-Identifier: Apache-2.0
"""gar.spec/1 parsing, citations and the exporter files (CPU, no Newton)."""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

from alloy.importers.gymanything import provenance as pv
from alloy.importers.gymanything.spec import BASIS_TO_SOURCE, SpecError, parse_spec

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "alloy" / "importers" / "gymanything"
EXPORT = REPO / "tools" / "gymanything_export"


def _raw(**over):
    raw = {"schema": "gar.spec/1", "key": "mug", "sim": "wiping", "name": "Mug", "category": "container",
           "mesh": "mug.obj", "material": {"name": "stoneware"},
           "mass": {"value": 0.3, "basis": "datasheet", "cite": "params.py:12", "quote": "maker page 300 g"}}
    raw.update(over)
    return raw


def test_parse_minimal_spec():
    s = parse_spec(_raw(), "mug.json", Path("/x"))
    assert s.card_name == "gar_wiping_mug"
    assert s.mesh == Path("/x/mug.obj")
    assert s.mass.alloy_source == "catalog" and s.kind == "rigid" and s.motion == "dynamic"


@pytest.mark.parametrize("over, field", [
    ({"colour": 1}, "<top>"),
    ({"key": "Mug"}, "key"),
    ({"category": "robot"}, "category"),
    ({"mass": {"value": 0.3, "basis": "guess", "cite": "a.py:1"}}, "mass.basis"),
    ({"mass": {"value": 0.3, "basis": "estimate", "cite": "a.py:1", "unc": [0.4, 0.5]}}, "mass.unc"),
    ({"mass": {"value": -1, "basis": "estimate", "cite": "a.py:1"}}, "mass"),
    ({"mass": {"value": 0.3, "basis": "estimate"}}, "mass.cite"),
    ({"friction": {"value": 3.0, "basis": "estimate", "cite": "a.py:1"}}, "friction.value"),
    ({"kind": "cloth"}, "cloth"),
    ({"kind": "soft"}, "material"),
    ({"mesh": None}, "mesh"),
    ({"schema": "gar.spec/0"}, "schema"),
])
def test_bad_specs_name_the_field(over, field):
    with pytest.raises(SpecError) as e:
        parse_spec(_raw(**over), "mug.json")
    assert e.value.field == field and e.value.message


def test_every_basis_maps_to_an_alloy_source():
    assert set(BASIS_TO_SOURCE.values()) <= {"measured", "catalog", "fit", "model"}
    assert BASIS_TO_SOURCE["estimate"] == "model" and BASIS_TO_SOURCE["measured"] == "measured"


@pytest.mark.parametrize("cite, path", [
    ("params.py:139", "benchmarks/roboworld/environments/wine_pouring/params.py:139"),
    ("wine_pouring/geometry.py:502", "benchmarks/roboworld/environments/wine_pouring/geometry.py:502"),
    ("src/robots/catalog.py:9", "src/robots/catalog.py:9"),
    ("REPORT.pdf p.3", "benchmarks/roboworld/environments/wine_pouring/REPORT.pdf p.3"),
])
def test_cite_ref_is_pinned(cite, path):
    assert pv.cite_ref("wine_pouring", cite) == f"gym-anything/robotics@{pv.UPSTREAM_COMMIT}:{path}"


def test_q_from_keeps_their_label_and_quote():
    s = parse_spec(_raw(), "mug.json")
    q = pv.q_from(s.mass, s.sim, "object mass")
    assert q.source == "catalog" and "'datasheet'" in q.note and "maker page 300 g" in q.note


@pytest.mark.skipif(not (EXPORT / "run.py").is_file(), reason="exporters not in this checkout")
def test_runner_lists_exporters_without_their_env():
    spec = importlib.util.spec_from_file_location("gar_run", EXPORT / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "wine_pouring" in mod.available()


def _tracked(folder: Path) -> list[Path]:
    """Tracked python files of ``folder`` (work in progress beside them is not checked); every file without git."""
    if not folder.is_dir():
        return []
    try:
        out = subprocess.run(["git", "ls-files", "*.py"], cwd=folder, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return sorted(folder.rglob("*.py"))
    return sorted(folder / f for f in out.split())


def _package_files() -> list[Path]:
    return _tracked(PKG) + _tracked(EXPORT)


# Internal names that must never reach public files (pieces joined so this file does not contain them either).
_PRIVATE = ["/home/", "pran" + "ay", "arg" + "on", "alloy" + "-refs", "astral" + "-hold", r"gs://[a-z0-9]", r"\.internal\b"]
_FORBIDDEN = re.compile("|".join(_PRIVATE), re.IGNORECASE)


@pytest.mark.parametrize("path", _package_files(), ids=lambda p: p.name)
def test_package_files_are_release_safe(path):
    text = path.read_text()
    assert len(text.splitlines()) <= 500, f"{path.name}: > 500 lines"
    hits = [ln for ln in text.splitlines() if _FORBIDDEN.search(ln)]
    assert not hits, f"{path.name}: internal names/paths: {hits[:3]}"
