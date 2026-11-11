# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.render_gates import view, whitebalance
from bench.render_gates.pngio import read_png, write_png


def test_standard_view_is_plain_srgb() -> None:
    lin = np.array([[[0.0, 0.18, 1.0]]], np.float32)
    assert view.to_png(lin, {"view_transform": "Standard", "look": "None"}).tolist() == [[[0, 118, 255]]]
    assert view.to_png(np.full((1, 1, 3), 0.09, np.float32), view.ViewSpec(exposure=1.0)).tolist() == [[[118, 118, 118]]]


def test_white_balance_round_trip_and_neutral_point() -> None:
    rgb = whitebalance.XYZ_TO_REC709 @ whitebalance.whitepoint_xyz(5000.0, 10.0)
    t, tint = whitebalance.temp_tint_from_whitepoint_rgb(rgb)
    assert t == pytest.approx(5000.0, rel=1e-9) and tint == pytest.approx(10.0, abs=1e-9)
    m = whitebalance.white_point_matrix(*whitebalance.temp_tint_from_whitepoint_rgb((1.0, 1.0, 1.0)))
    assert np.allclose(m, np.eye(3), atol=1e-5)  # D65 white in, D65 white out


def test_white_balance_maps_the_whitepoint_to_grey() -> None:
    wp = (1.128, 0.9755, 0.8641)
    m = view.colour_matrix(view.ViewSpec(whitepoint=wp))
    out = m @ np.array(wp)
    assert np.allclose(out / out[1], 1.0, atol=2e-3)


def test_lut_path_with_an_identity_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LUT whose nodes hold their own encoded input reproduces the analytic curve at the nodes."""
    ax = view._grid("Filmic")
    enc = view.srgb_encode(ax)
    table = np.stack(np.meshgrid(enc, enc, enc, indexing="ij"), -1)
    monkeypatch.setattr(view, "lut", lambda v, look: (table, np.eye(3)))
    lin = ax[[0, 100, 200, 256]].reshape(1, 4, 1).repeat(3, axis=2)
    got = view.to_png(lin, view.ViewSpec(view="Filmic"))
    want = view.to_png(lin, view.ViewSpec())
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 1


def test_from_view_json_reads_blender_keys() -> None:
    v = view.from_view_json({"view_transform": "AgX", "look": "AgX - Medium High Contrast", "use_white_balance": True,
                             "white_balance_whitepoint": [1.1, 1.0, 0.9], "exposure": 0.5})
    assert v == view.ViewSpec("AgX", "AgX - Medium High Contrast", 0.5, 1.0, (1.1, 1.0, 0.9))


def test_missing_config_says_what_to_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLENDER_OCIO", str(tmp_path / "nope.ocio"))
    with pytest.raises(view.ViewError, match="config.ocio"):
        view.ocio_config()


def test_png_round_trip(tmp_path: Path) -> None:
    a = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    write_png(tmp_path / "a.png", a)
    assert (read_png(tmp_path / "a.png") == a).all()
