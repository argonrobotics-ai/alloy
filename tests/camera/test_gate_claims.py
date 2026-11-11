# SPDX-License-Identifier: Apache-2.0
"""Per-scene claims of the render gates (ruling: the fastest setting that passes every quality gate)."""
from __future__ import annotations

from alloy.camera.gate_claims import Claim, claim_for, parse, parse_claim


def test_builtin_claims_follow_the_ruling():
    assert claim_for("c3", "128x128", spec={}) == Claim(5, "learned", 2)
    assert claim_for("c3", "640x480", spec={}) == Claim(12, "learned", 1)
    assert claim_for("c3", "1280x720", spec={}) == Claim(16, "learned", 1, "c3_1280")
    assert claim_for("t12", "128x128", spec={}) == Claim(40, "learned", 1, "t12_128")
    for res in ("640x480", "1280x720"):
        assert claim_for("t12", res, spec={}) == Claim(64, "oidn", 1)
    assert Claim(5, "learned", 2).label() == "5 spp + learned (tta 2)" and Claim(64, "oidn").label() == "64 spp + oidn"
    assert Claim(40, "learned", 1, "t12_128").label() == "40 spp + learned t12_128 (tta 1)"
    assert claim_for("c4", "1280x720", spec={}) == Claim(32, "learned", 1, "c4_1280")
    assert claim_for("towel", "640x480", spec={}) == Claim(128, "oidn")  # deformables: 128 spp + OIDN


def test_matrix_spec_wins_scene_first_then_table_then_size():
    spec = {"claims": {"128x128": {"spp": 6, "den": "on", "denoiser": "learned, tta 4"}},
            "scenes": {"t12": {"claims": {"128x128": {"spp": 32, "den": "on", "denoiser": "learned t12 weights, tta 2"}}}}}
    assert claim_for("t12", "128x128", spec) == Claim(32, "learned", 2)
    assert claim_for("t12", "640x480", spec) == Claim(64, "oidn", 1)
    assert parse({"spp": 40, "den": "on", "denoiser": "learned, tta 1", "weights": "t12_128"}).weights == "t12_128"
    assert claim_for("c3", "128x128", spec) == Claim(6, "learned", 4)
    assert parse({"spp": 64, "den": "on", "denoiser": "OIDN"}) == Claim(64, "oidn", 1)
    assert parse({"spp": 8, "kind": "oidn", "tta": 1}) == Claim(8, "oidn", 1)


def test_repo_spec_agrees_with_the_table_for_c3():
    assert claim_for("c3", "128x128") == Claim(5, "learned", 2)


def test_candidate_claims_from_the_command_line():
    import pytest

    from alloy.camera import CameraError
    assert parse_claim("16,learned,1,c3_1280") == Claim(16, "learned", 1, "c3_1280")
    assert parse_claim("64,oidn") == Claim(64, "oidn", 1)
    with pytest.raises(CameraError, match="SPP,KIND"):
        parse_claim("sixteen,learned")
