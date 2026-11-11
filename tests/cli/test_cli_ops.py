# SPDX-License-Identifier: Apache-2.0
"""Every simulation operation through ``respond`` with a fake Sim (no kernels): replies, state views, health,
sessions, batch, targets and render."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from alloy.cli import main as cli_main
from alloy.cli.registry import passed, respond
from alloy.cli.session import Session
from alloy.cli.testing import FakeSim, fake_build

BOX = {"worlds": 2, "dt": 0.01, "objects": [{"add": "ground"},
                                            {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1], "mass": 1.0}]}


@pytest.fixture()
def session():
    s = Session(builder=fake_build)
    yield s
    s.close_all()


def call(session, **request):
    reply = json.loads(json.dumps(respond(session, request)))       # what a client sees
    assert reply["ok"], reply
    return reply["result"]


def test_build_then_step_state_targets_reset_close(session):
    built = call(session, op="build", scene=BOX, device="cpu", seed=3)
    assert built["sim"] == "sim1" and built["worlds"] == 2 and built["routes"]["cube"] == "rigid (mujoco)"
    assert session.sims["sim1"].built["seed"] == 3 and "build_s" in built
    step = call(session, op="step", steps=10, targets={"lift": 0.05})
    assert step["t"] == pytest.approx(0.1) and step["physics_steps"] == 40 and step["health"]["ok"]
    st = step["state"]
    assert st["cube.pos"] == pytest.approx([0.0, 0.0, 0.99]) and st["arm/lift.q"] == 0.05 and "t" not in st
    assert st["water.count"] == 5 and st["water.centroid"] == [2.0, 2.0, 2.0] and st["water.hi"] == [4.0] * 3
    full = call(session, op="state", names=["water"], points="full", world="all")["state"]
    assert np.shape(full["water.points"]) == (2, 5, 3)
    call(session, op="targets", targets={"lift": [0.1, 0.2]})
    assert call(session, op="step", names=["lift"], world="all")["state"]["lift.q"] == [0.1, 0.2]
    assert call(session, op="reset")["t"] == 0.0
    sim = session.sims["sim1"]
    assert call(session, op="close") == {"closed": "sim1", "open": []} and sim.closed


def test_run_samples_saves_and_ramps(session, tmp_path):
    res = call(session, op="run", scene=BOX, seconds=0.5, names=["cube", "lift"], every=5, samples=3,
               targets={"lift": {"from": 0.0, "to": 0.1}}, out=str(tmp_path / "t.npz"))
    assert res["frames"] == 11 and res["keys"]["cube.pos"] == [11, 2, 3]
    assert [s["t"] for s in res["samples"]] == pytest.approx([0.0, 0.25, 0.5])
    assert res["samples"][-1]["lift.q"] == pytest.approx(0.1) and res["final"]["lift.q"] == pytest.approx(0.1)
    assert np.load(res["saved"])["cube.pos"].shape == (11, 2, 3)


def test_errors_are_replies_with_fixes(session):
    reply = respond(session, {"op": "step", "steps": 1})
    assert not reply["ok"] and reply["error"]["type"] == "SessionError" and "scene" in reply["error"]["fix"]
    call(session, op="build", scene=BOX, id="a")
    for request, kind in [({"op": "step", "sim": "b"}, "SessionError"),
                          ({"op": "build", "scene": BOX, "id": "a"}, "SessionError"),
                          ({"op": "state", "names": ["cub"]}, "UnknownNameError"),
                          ({"op": "state", "world": 5}, "SimulationError"),
                          ({"op": "step", "targets": {"lift": {"from": 0, "to": 1}}}, "RequestError"),
                          ({"op": "stpe"}, "RequestError"),
                          ({"steps": 1}, "RequestError")]:
        reply = respond(session, request)
        assert not reply["ok"] and reply["error"]["type"] == kind and reply["error"]["fix"], request


def test_session_limit_and_last_sim(session):
    session.max_sims = 2
    call(session, op="build", scene=BOX)
    call(session, op="build", scene=BOX)
    reply = respond(session, {"op": "build", "scene": BOX})
    assert not reply["ok"] and "close one first" in reply["error"]["fix"]
    assert call(session, op="close", sim="sim2")["open"] == ["sim1"] and session.last == "sim1"


def test_invalid_worlds_fail_the_result_not_the_call(session, monkeypatch):
    monkeypatch.setattr(FakeSim, "invalid_below", 0.95)
    reply = respond(session, {"op": "step", "scene": BOX, "steps": 100})        # cube z 1.0 -> 0.9
    assert reply["ok"] and not passed(reply)
    health = reply["result"]["health"]
    assert health["invalid_worlds"] == 2 and health["problems"][0]["fixes"] == ["lower Scene(dt=...)"]
    live = respond(session, {"op": "check"})
    assert live["ok"] and live["result"]["mode"] == "live" and not live["result"]["ok"]


def test_render_writes_a_png(session, tmp_path):
    res = call(session, op="render", scene=BOX, out=str(tmp_path / "a.png"), width=32, height=24, steps=2,
               look_at=[1, -1, 1, 0, 0, 0])
    assert res["width"] == 32 and res["height"] == 24 and res["mean_rgb"] == [128.0] * 3
    assert (tmp_path / "a.png").read_bytes()[:4] == b"\x89PNG" and res["t"] == pytest.approx(0.02)
    reply = respond(session, {"op": "render", "look_at": [1, 2, 3]})
    assert not reply["ok"] and reply["error"]["field"] == "look_at"


def test_batch_keeps_simulations_open(tmp_path, monkeypatch):
    monkeypatch.setattr("alloy.cli.command.Session", lambda: Session(builder=fake_build))
    requests = [{"op": "build", "scene": BOX, "id": "box"}, {"op": "step", "sim": "box", "steps": 5},
                {"op": "state", "names": ["cube"]}, {"op": "step", "sim": "nope"}, {"op": "list"}]
    path = tmp_path / "req.json"
    path.write_text(json.dumps(requests))
    out = io.StringIO()
    assert cli_main(["batch", str(path)], stdout=out) == 2
    reply = json.loads(out.getvalue())
    assert not reply["ok"] and reply["result"]["ran"] == 4 and reply["result"]["stopped_at"] == 3
    assert reply["result"]["replies"][2]["result"]["t"] == pytest.approx(0.05)
    path.write_text("\n".join(json.dumps(r) for r in requests[:3]) + "\n")      # JSON lines
    out = io.StringIO()
    assert cli_main(["batch", str(path)], stdout=out) == 0
    assert json.loads(out.getvalue())["result"]["passed"]
