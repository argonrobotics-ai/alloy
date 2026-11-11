# SPDX-License-Identifier: Apache-2.0
"""CLI parsing and check/encode on real inputs when the other builders' modules exist."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from alloy.render import cli

SYNTH = Path(__file__).with_name("synth.py")


def load_synth():
    if not SYNTH.exists():
        pytest.skip("tests/render/synth.py not written yet")
    spec = importlib.util.spec_from_file_location("render_synth", SYNTH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parser_accepts_documented_commands():
    ap = cli.build_parser()
    a = ap.parse_args(["render", "b", "--look", "l.json", "--camera", "v", "--out", "o", "--vms", "2",
                       "--frames", "0:11", "--keep-vms"])
    assert a.vms == 2 and a.keep_vms and a.frames == "0:11"
    assert ap.parse_args(["still", "b", "--look", "l", "--camera", "v", "--frame", "3", "--vm", "x"]).frame == 3
    assert ap.parse_args(["encode", "d", "--prores"]).prores
    assert ap.parse_args(["check", "b", "--look", "l"]).camera is None


def test_check_reports_missing_look(tmp_path, capsys):
    pytest.importorskip("alloy.render.bundle")
    rc = cli.main(["check", str(tmp_path), "--look", str(tmp_path / "nope.json")])
    assert rc == 1 and "ERROR" in capsys.readouterr().out


def test_check_on_synth_bundle(tmp_path, capsys):
    import json

    synth = load_synth()
    synth.write_synth(tmp_path / "b")
    look_path = tmp_path / "look.json"
    look_path.write_text(json.dumps(synth.make_look()))
    rc = cli.main(["check", str(tmp_path / "b"), "--look", str(look_path), "--camera", "vertical"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "GPU-h" in out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_encode_png_frames(tmp_path):
    (tmp_path / "frames").mkdir()
    for k in (0, 4, 8):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=0x{k * 20:02x}0000:s=64x64",
                        "-frames:v", "1", str(tmp_path / "frames" / f"{k:05d}.png")], check=True)
    assert cli.main(["encode", str(tmp_path), "--fps", "10"]) == 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
                            str(tmp_path / "clip.mp4")], capture_output=True, text=True)
    assert probe.stdout.strip() == "3"


def test_playback_speed():
    assert cli.playback_speed(25, 2, 50.0) == 1.0
    assert cli.playback_speed(30, 2, 50.0) == 1.2
    assert cli.playback_speed(30, 1, 120.0) == 0.25
