# SPDX-License-Identifier: Apache-2.0
"""alloy.plugins registry: built-ins, entry points, targets, config parsing and describe()."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import alloy.plugins as ap
from alloy.plugins import registry
from alloy.plugins.base import ForcePlugin, make_config


def test_builtins_are_listed_with_kinds():
    rows = {r.name: r for r in ap.available()}
    assert {"magnet", "ink", "hydraulic", "hydraulic_cylinder"} <= set(rows)
    assert rows["magnet"].kind == "force" and rows["hydraulic_cylinder"].kind == "object"
    assert rows["ink"].source == "builtin"
    objects = {r.name for r in ap.available("object")}
    assert "hydraulic_cylinder" in objects and {r.kind for r in ap.available("object")} == {"object"}


def test_create_and_config_errors_name_valid_fields():
    mag = ap.create("magnet", moments=(2.0, 2.0))
    assert mag.config.moments == (2.0, 2.0) and mag.label == "magnet"
    with pytest.raises(ap.PluginError, match="valid fields"):
        ap.create("magnet", strength=1.0)
    with pytest.raises(ap.PluginError, match="one moment per body"):
        ap.create("magnet", moments=(1.0,))
    with pytest.raises(ap.PluginError, match="available"):
        ap.get("no_such_plugin")


def test_make_config_accepts_instance_dict_and_kwargs():
    @dataclass
    class Config:
        a: float = 1.0
        b: str = "x"

    assert make_config(Config, None, {}) == Config()
    assert make_config(Config, {"a": 2.0}, {"b": "y"}) == Config(2.0, "y")
    assert make_config(Config, Config(3.0), {}) == Config(3.0)


def test_describe_reads_field_comments():
    d = ap.describe(ap.get("ink"))
    fields = {f["name"]: f for f in d["config"]}
    assert fields["rate"]["doc"].startswith("µL/s")
    assert d["kind"] == "force" and "Example" in d["doc"]


def test_load_target_from_a_file(tmp_path):
    src = tmp_path / "wind.py"
    src.write_text(textwrap.dedent('''
        from alloy.plugins import ForcePlugin

        class Wind(ForcePlugin):
            """Wind."""
            name = "wind_test"
            summary = "wind"

            def update(self, state_in, state_out, dt):
                pass
    '''))
    (cls,) = ap.load_target(str(src))
    assert cls.name == "wind_test" and cls.kind == "force"


def test_entry_points_register_and_a_broken_one_is_skipped(monkeypatch, caplog):
    class Breeze(ForcePlugin):
        """Breeze."""
        name = "breeze_ep_test"
        summary = "breeze"

    good = SimpleNamespace(name="breeze", dist=SimpleNamespace(name="breeze-pkg"), load=lambda: Breeze)

    def boom():
        raise ImportError("missing dependency")

    bad = SimpleNamespace(name="bad", dist=SimpleNamespace(name="bad-pkg"), load=boom)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda group: [good, bad])
    try:
        assert registry.load_entry_points(force=True) == ["breeze_ep_test"]
        assert {r.name: r.source for r in ap.available()}["breeze_ep_test"] == "entry_point:breeze-pkg"
        assert "bad" in caplog.text and "missing dependency" in caplog.text
    finally:
        registry._REGISTRY.pop("breeze_ep_test", None)


def test_name_clash_raises():
    class Other(ForcePlugin):
        """Clash."""
        name = "magnet"

    ap.available()
    with pytest.raises(ap.PluginError, match="taken"):
        ap.register(Other)
