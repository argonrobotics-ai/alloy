# SPDX-License-Identifier: Apache-2.0
"""A call that fails changes nothing: step and run check names and world before stepping, a failed call that built
its own scene closes it again and leaves the default simulation as it was, and an unknown id is not echoed in full
(fake Sim, no kernels)."""

from __future__ import annotations

import json

import pytest

from alloy.cli.registry import respond
from alloy.cli.session import MAX_SIMS, Session
from alloy.cli.testing import fake_build

A = {"worlds": 2, "dt": 0.01, "objects": [{"add": "ground"},
                                          {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1], "mass": 1.0}]}
B = {"worlds": 2, "dt": 0.01, "objects": [{"add": "ground"},
                                          {"add": "sphere", "name": "ball", "radius": 0.05, "mass": 0.5}]}


@pytest.fixture()
def built():
    """A session whose builder records every simulation it builds."""
    sims = []

    def builder(scene, **kw):
        sims.append(fake_build(scene, **kw))
        return sims[-1]

    session = Session(builder=builder)
    reply = respond(session, {"op": "build", "scene": A, "id": "main"})
    assert reply["ok"], reply
    yield session, sims
    session.close_all()


@pytest.mark.parametrize(("request_", "kind", "field"), [
    ({"op": "step", "steps": 25, "names": ["typo_body"]}, "UnknownNameError", None),
    ({"op": "step", "steps": 25, "world": 99}, "SimulationError", "world"),
    ({"op": "step", "steps": 25, "targets": {"lift": {"from": 0, "to": 1}}}, "RequestError", "targets.lift"),
    ({"op": "run", "seconds": 0.1, "world": 99}, "SimulationError", "world"),
    ({"op": "run", "seconds": 0.1, "names": ["typo_body"]}, "UnknownNameError", None),
    ({"op": "render", "steps": 25, "world": 5}, "SimulationError", "world"),
    ({"op": "render", "steps": 25, "look_at": [1, 2]}, "RequestError", "look_at"),
    ({"op": "state", "world": 7}, "SimulationError", "world"),
])
def test_a_failed_call_does_not_step_the_simulation(built, tmp_path, request_, kind, field):
    session, _ = built
    out = tmp_path / "traj.npz"
    reply = respond(session, {**request_, "sim": "main", **({"out": str(out)} if request_["op"] == "run" else {})})
    assert not reply["ok"] and reply["error"]["type"] == kind and reply["error"]["fix"], reply
    if field is not None:
        assert reply["error"]["field"] == field
    assert session.sims["main"].t == 0.0 and not out.exists()


def test_a_failed_call_with_its_own_scene_opens_nothing_and_keeps_the_default(built):
    session, sims = built
    for request in ({"op": "render", "scene": B, "look_at": [1, 2]}, {"op": "step", "scene": B, "names": ["nope"]},
                    {"op": "run", "scene": B, "seconds": 0.1, "world": 9}, {"op": "targets", "scene": B,
                                                                              "targets": {"nope": 0.1}}):
        reply = respond(session, request)
        assert not reply["ok"], reply
        assert list(session.sims) == ["main"] and session.last == "main", request
    assert all(sim.closed for sim in sims[1:]) and len(sims) > 1     # the ones the failed calls built are closed
    state = respond(session, {"op": "state"})["result"]
    assert state["sim"] == "main" and "cube.pos" in state["state"]
    for _ in range(MAX_SIMS + 2):                                     # failures never use up the open slots
        assert not respond(session, {"op": "render", "scene": B, "look_at": [1, 2]})["ok"]
    reply = respond(session, {"op": "build", "scene": B})
    assert reply["ok"] and reply["result"]["sim"] == "sim2", reply


def test_a_huge_unknown_id_is_not_echoed_in_full(built):
    session, _ = built
    for op in ("reset", "step", "close"):
        reply = respond(session, {"op": op, "sim": "s" * 1_000_000})
        assert reply["error"]["type"] == "SessionError" and len(json.dumps(reply)) < 2_000, op
        assert "main" in reply["error"]["fix"]
    reply = respond(session, {"op": "s" * 1_000_000})
    assert not reply["ok"] and len(json.dumps(reply)) < 10_000
