# SPDX-License-Identifier: Apache-2.0
"""Flag values that start with "-" (``--up -y``, ``--look-at -1.2,-1.2,0.9,0,0,0.2``, ``--area -0.2,...``) stay
the flag's value on ``alloy`` and ``python -m alloy.create``; real options, booleans and ``--`` are left alone."""

from __future__ import annotations

import io
import json

import pytest

from alloy.cli import main
from alloy.cli.command import build_parser
from alloy.cli.registry import OPS
from alloy.cli.spec import dash_values, flags_to_args


@pytest.mark.parametrize(("argv", "joined"), [
    (["render", "x", "--look-at", "-1.2,-1.2,0.9,0,0,0.2"], ["render", "x", "--look-at=-1.2,-1.2,0.9,0,0,0.2"]),
    (["compose", "a mug", "--area", "-0.2,-0.2,0.2,0.2"], ["compose", "a mug", "--area=-0.2,-0.2,0.2,0.2"]),
    (["run", "x", "--seconds", "-0.5"], ["run", "x", "--seconds=-0.5"]),
    (["create", "--up", "-y", "--no-check"], ["create", "--up=-y", "--no-check"]),
    (["run", "x", "--seconds", "--pretty"], ["run", "x", "--seconds", "--pretty"]),      # a real option: untouched
    (["check", "x", "--quick", "-v"], ["check", "x", "--quick", "-v"]),                  # a boolean takes no value
    (["list", "--match", "-"], ["list", "--match", "-"]),                                # "-" is a value already
    (["run", "x", "--", "--seconds", "-1"], ["run", "x", "--", "--seconds", "-1"]),      # after --: all positional
])
def test_only_one_value_flags_take_a_dash_value(argv, joined):
    assert dash_values(build_parser(), argv) == joined


def test_dash_values_reach_the_operation():
    parser = build_parser()
    ns = parser.parse_args(dash_values(parser, ["create", "--mesh", "cup.obj", "--up", "-y"]))
    assert flags_to_args(OPS["create"], ns)["up"] == "-y"
    ns = parser.parse_args(dash_values(parser, ["render", "x", "--look-at", "-1.2,-1.2,0.9,0,0,0.2"]))
    assert flags_to_args(OPS["render"], ns)["look_at"] == [-1.2, -1.2, 0.9, 0.0, 0.0, 0.2]


def test_a_negative_number_reaches_its_check():
    out = io.StringIO()
    code = main(["run", "demo_peg_d12", "--seconds", "-0.5"], stdout=out)
    reply = json.loads(out.getvalue())
    assert code == 2 and reply["error"]["field"] == "seconds" and ">= 0" in reply["error"]["message"], reply


def test_python_m_alloy_create_takes_up_minus_y(monkeypatch, capsys):
    from alloy.create import cli

    seen = {}
    monkeypatch.setattr(cli, "_run", lambda ns: (seen.update(up=ns.up) or 0, "{}"))
    assert cli.main(["--mesh", "cup.obj", "--up", "-y", "--no-check"]) == 0 and seen["up"] == "-y"
    assert cli.main(["--mesh", "cup.obj", "--up", "-q"]) == 2                        # still checked: not an axis
