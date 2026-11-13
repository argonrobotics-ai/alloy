# SPDX-License-Identifier: Apache-2.0
"""Gemini audit S2-043: ``python -m alloy.looks`` answers an unknown object (``--show``) or texture family
(``--family`` with ``--pool`` / ``--fetch-textures``) with one error and its fix and exit code 2, never a traceback
with exit code 1 (which reads as "a look is grey")."""

import subprocess
import sys
from pathlib import Path

import pytest

from alloy.looks.__main__ import main
from alloy.looks.schema import FAMILIES

ROOT = Path(__file__).resolve().parents[2]


def test_an_unknown_object_prints_the_error_and_its_fix(capsys):
    assert main(["--show", "nonexistent_object_card"]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert lines[0].startswith("error: nonexistent_object_card: name: no object 'nonexistent_object_card'")
    assert lines[1] == "fix: `alloy list objects` names the objects; check the spelling"


@pytest.mark.parametrize("command", ["--pool", "--fetch-textures"])
def test_an_unknown_family_is_a_usage_error_that_lists_the_families(command, capsys):
    with pytest.raises(SystemExit) as stop:           # argparse stops before any texture is read or fetched
        main([command, "--family", "nonexistent_family"])
    assert stop.value.code == 2
    out, err = capsys.readouterr()
    assert out == "" and "invalid choice: 'nonexistent_family'" in err
    assert all(f in err for f in FAMILIES)


def test_a_known_family_still_lists_its_textures(capsys):
    assert main(["--pool", "--family", "wood"]) == 0
    assert '"wood"' in capsys.readouterr().out


def test_the_module_command_never_shows_a_traceback():
    """The audit's repro: the real ``python -m alloy.looks`` entry, from the source tree."""
    for argv in (["--show", "nonexistent_object_card"], ["--pool", "--family", "nonexistent_family"],
                 ["--fetch-textures", "--family", "nonexistent_family"]):
        res = subprocess.run([sys.executable, "-m", "alloy.looks", *argv], cwd=ROOT, capture_output=True, text=True,
                             timeout=300, check=False)
        assert (res.returncode, res.stdout) == (2, ""), argv
        assert "Traceback" not in res.stderr and "error:" in res.stderr, argv
