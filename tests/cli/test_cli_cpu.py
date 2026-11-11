# SPDX-License-Identifier: Apache-2.0
"""End to end on CPU with the real engine: a box dropped through build, step, run, state, check and close in one
session, and the same through one-shot commands (kernels compile on the first run, then come from the cache)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from alloy.cli import main
from alloy.cli.registry import respond
from alloy.cli.session import Session

DROP = {"worlds": 2, "dt": 0.01, "objects": [
    {"add": "ground"},
    {"add": "box", "name": "cube", "size": [0.04, 0.04, 0.04], "pos": [[0.0, 0.0, 0.1], [0.05, 0.0, 0.1]],
     "mass": 0.1}]}


@pytest.mark.compile_heavy
def test_session_drop_box_on_cpu(tmp_path):
    session = Session()
    try:
        built = respond(session, {"op": "build", "scene": DROP, "device": "cpu"})
        assert built["ok"], built
        assert built["result"]["names"]["bodies"] == ["cube"] and built["result"]["names"]["static"] == ["ground"]
        step = respond(session, {"op": "step", "steps": 10, "world": "all"})["result"]
        assert step["t"] == pytest.approx(0.1) and np.asarray(step["state"]["cube.pos"]).shape == (2, 3)
        run = respond(session, {"op": "run", "seconds": 0.3, "names": ["cube"], "out": str(tmp_path / "t.npz")})
        res = run["result"]
        assert run["ok"] and res["health"]["ok"] and res["frames"] == 31
        assert res["final"]["cube.pos"][2] == pytest.approx(0.02, abs=2e-3)          # resting on the ground
        assert np.load(res["saved"])["cube.pos"].shape == (31, 2, 3)
        check = respond(session, {"op": "check"})
        assert check["ok"] and check["result"]["mode"] == "live" and check["result"]["ok"]
        assert respond(session, {"op": "close"})["result"]["open"] == []
    finally:
        session.close_all()


@pytest.mark.compile_heavy
def test_one_shot_step_command_on_cpu():
    out = io.StringIO()
    code = main(["step", json.dumps(DROP), "--device", "cpu", "--steps", "30", "--names", "cube", "--world", "1"],
                stdout=out)
    reply = json.loads(out.getvalue())
    assert code == 0 and reply["ok"], reply
    assert reply["result"]["state"]["cube.pos"][0] == pytest.approx(0.05, abs=1e-3)


@pytest.mark.compile_heavy
def test_render_png_on_cpu(tmp_path):
    session = Session()
    try:
        reply = respond(session, {"op": "render", "scene": DROP, "device": "cpu", "out": str(tmp_path / "drop.png"),
                                  "width": 64, "height": 48, "spp": 2, "look_at": [0.4, -0.4, 0.3, 0, 0, 0.05]})
        assert reply["ok"], reply
        assert reply["result"]["width"] == 64 and (tmp_path / "drop.png").read_bytes()[:4] == b"\x89PNG"
        assert 20 < float(np.mean(reply["result"]["mean_rgb"])) < 250            # lit, not black or blown out
    finally:
        session.close_all()
