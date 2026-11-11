# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from bench.render_gates import layout


def test_parse_name_maps_denoiser_names() -> None:
    assert layout.parse_name("128x128_16spp_oidn.png") == ("128x128", 16, "on")
    assert layout.parse_name("128x128_16spp_noden.png") == ("128x128", 16, "off")
    assert layout.parse_name("128x128_16spp_on.exr") is None


def test_image_name_round_trips() -> None:
    assert layout.parse_name(layout.image_name("1280x720", 12, "on")) == ("1280x720", 12, "on")


def test_collect_needs_reference_engine(gate_root: Path) -> None:
    frames = layout.collect(gate_root)
    assert sorted(frames) == [("c3", "10"), ("c3", "20")]
    assert {"cycles", "cycles_s1", "alloy"} <= set(frames[("c3", "10")])
    assert layout.collect(gate_root, "missing") == {}


def test_collect_missing_root_says_what_to_pass(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="<scene>/<frame>/<engine>"):
        layout.collect(tmp_path / "nope")


def test_controls() -> None:
    assert layout.is_control("cycles_s4") and not layout.is_control("cycles") and not layout.is_control("alloy_s1x")
