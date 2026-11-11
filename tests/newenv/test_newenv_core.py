# SPDX-License-Identifier: Apache-2.0
"""bench/newenv: loader, prompt, plan, overlay, Wilson interval (CPU, no network)."""
from __future__ import annotations

import json

import pytest

from bench.newenv import prompt, requests, stats


def test_loader_real_list():
    reqs = requests.load_requests()
    assert [r.id for r in reqs] == [f"E{i:02d}" for i in range(1, 25)]
    assert {r.group for r in reqs} == set(requests.GROUPS)
    assert [r.id for r in reqs if r.policy] == [f"E{i}" for i in range(19, 25)]
    e15 = {r.id: r for r in reqs}["E15"]
    assert e15.probe["liquid_points"] == ["P", 3] and e15.entry["num_envs"] == 16 and e15.checks
    assert reqs[0].checker == "push_cube" and "cube" in reqs[0].request


def test_loader_needs_probe(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"requests": [{"id": "E1", "group": "rigid", "request": "x", "checker": "c"}]}))
    with pytest.raises(requests.ConfigError, match="probe"):
        requests.load_requests(f)


def test_loader_rejects_bad_group_and_duplicates(tmp_path):
    f = tmp_path / "r.json"
    base = {"id": "E1", "group": "rigid", "request": "x", "checker": "c", "probe": {"a": []}}
    f.write_text(json.dumps({"requests": [{**base, "group": "plasma"}]}))
    with pytest.raises(requests.ConfigError, match="group"):
        requests.load_requests(f)
    f.write_text(json.dumps({"requests": [base, base]}))
    with pytest.raises(requests.ConfigError, match="duplicate"):
        requests.load_requests(f)


def test_engines_caps_and_todo():
    eng = requests.load_engines()
    assert set(eng) == {"alloy", "maniskill", "playground", "genesis"}
    assert eng["alloy"].agent_s == 2700 and eng["alloy"].rerun_s == 900
    assert requests.unfinished_engines(["alloy", "genesis"]) == {}
    assert not any(e.notice for e in eng.values())
    reqs = {r.id: r for r in requests.load_requests()}
    declines = {k: sorted(r for r in reqs if requests.is_correct_decline(e, reqs[r])) for k, e in eng.items()}
    assert declines["alloy"] == declines["genesis"] == []
    assert {"E11", "E15", "E18", "E23", "E24"} <= set(declines["maniskill"]) and "E01" not in declines["maniskill"]
    assert declines["playground"] == ["E15", "E16", "E17", "E18", "E23"] and "E11" not in declines["playground"]


def test_extra_config_adds_arms_and_fields(tmp_path, monkeypatch):
    extra = {"restricted": {"name": "Restricted", "port": 3299, "module": "restricted", "install": "x", "docs": "y",
                            "agent_s": 3600, "refs": "TODO later", "notice": "Restricted arm: test notice."},
             "genesis": {"notice": "Genesis: test notice."}}
    (tmp_path / "engines.json").write_text(json.dumps(extra))
    (tmp_path / "allowlist.json").write_text(json.dumps({"ports": {"3299": {"engine": "restricted", "allow": ["r.example"]}}}))
    monkeypatch.setenv(requests.EXTRA_ENV, str(tmp_path))
    eng = requests.load_engines()
    assert eng["restricted"].agent_s == 3600 and eng["restricted"].rerun_s == 900
    assert eng["genesis"].notice == "Genesis: test notice." and eng["genesis"].module == "genesis"
    assert "restricted" in requests.unfinished_engines(["alloy", "restricted"])
    allow = requests.read_config("allowlist.json")
    assert "3299" in allow["ports"] and "3201" in allow["ports"] and allow["common"]
    assert "restricted" not in requests.load_engines(requests.HERE / "engines.json")    # an explicit path is read alone


def test_wilson_interval():
    lo, hi = stats.wilson(3, 3)
    assert lo == pytest.approx(0.4385, abs=1e-3) and hi == 1.0
    lo, hi = stats.wilson(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = stats.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    with pytest.raises(ValueError):
        stats.wilson(4, 3)
    assert stats.median([1, None, 3]) == 2 and stats.median([]) is None
