# SPDX-License-Identifier: Apache-2.0
"""``alloy check --contacts`` reaches the check run's build as ``alloy.build(contacts=)`` sets it, and without it the
check builds exactly what it built before; a check of an open simulation refuses the arguments of a check run
(scene, seconds, worlds, moving, ...) instead of dropping them."""

from __future__ import annotations

import pytest

from alloy.check.run import RunOptions
from alloy.cli.registry import respond
from alloy.cli.session import Session
from alloy.cli.testing import fake_build

BOX = {"dt": 0.02, "objects": [{"add": "ground"},
                               {"add": "box", "name": "b", "size": [0.05, 0.05, 0.05], "pos": [0, 0, 0.025]}]}


@pytest.fixture()
def builds(monkeypatch):
    """The profile overrides of every simulation a check run builds (the real build still runs)."""
    from alloy.integration import wiring

    seen, real = [], wiring.build_simulation

    def record(built, prof="standard", **kw):
        seen.append({k: v for k, v in kw.items() if k not in ("seed", "stepper", "plugins")})
        return real(built, prof, **kw)

    monkeypatch.setattr(wiring, "build_simulation", record)
    return seen


@pytest.mark.parametrize(("contacts", "overrides"), [
    (None, {}),                                         # today's check build, unchanged
    ("auto", {"contacts_auto": True}),
    ("mujoco", {"native_contacts": "mujoco"}),
    ("newton", {"native_contacts": "newton"}),
])
def test_contacts_reach_the_check_run(builds, contacts, overrides):
    request = {"op": "check", "scene": BOX, "seconds": 0.04, "worlds": 1, "device": "cpu",
               **({"contacts": contacts} if contacts else {})}
    reply = respond(Session(), request)
    assert reply["ok"] and reply["result"]["ok"], reply
    assert builds and builds[0] == overrides, builds
    assert RunOptions(contacts=contacts).overrides() == overrides


@pytest.mark.parametrize(("extra", "field"), [
    ({"scene": BOX}, "scene"), ({"seconds": 0.5}, "seconds"), ({"worlds": 4}, "worlds"), ({"moving": True}, "moving"),
    ({"quick": True}, "quick"), ({"contacts": "newton"}, "contacts"), ({"budget": 5.0}, "budget"),
])
def test_a_live_check_refuses_run_arguments(extra, field):
    session = Session(builder=fake_build)
    assert respond(session, {"op": "build", "scene": BOX})["ok"]
    for request in ({"op": "check", "sim": "sim1", **extra}, {"op": "check", **{k: v for k, v in extra.items()
                                                                                 if k != "scene"}}):
        if request.keys() == {"op"}:
            continue                                    # scene alone without sim is a scene check, not a live one
        reply = respond(session, request)
        assert not reply["ok"] and reply["error"]["field"] == field and reply["error"]["fix"], reply
    live = respond(session, {"op": "check", "sim": "sim1"})
    assert live["ok"] and live["result"]["mode"] == "live"
