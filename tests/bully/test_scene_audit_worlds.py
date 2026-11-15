# SPDX-License-Identifier: Apache-2.0
"""Gemini 0.1.2 audit S2-038: ``alloy.bully.audit`` on a scene with per-world random values (CPU).

A box drawn per world (``scene.randomize``) overlaps a wall in some worlds; another box rests alone. The audit's
seed draws the scene (as ``alloy.build(scene, seed=)``), a finding in world w > 0 is attributed and replayed with
world w's draws (it was replayed in world 0's scene: the innocent box was blamed and ``replay_finding`` found
nothing), and ``python -m alloy.bully audit`` takes ``--library`` and makes ``--out``'s folders.
"""

from __future__ import annotations

import io
import json

import pytest

import alloy
from alloy.bully import __main__ as cli
from alloy.bully.scene_audit import audit, replay_finding
from alloy.bully.scene_target import AuditOptions, scene_target

WORLDS = 8


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    d = tmp_path_factory.mktemp("s2_038")
    sc = alloy.Scene(worlds=WORLDS, dt=0.02)
    sc.add_ground()
    sc.add_box("innocent_box", size=(0.08, 0.08, 0.08), pos=(-0.40, -0.40, 0.0395), mass=0.5)
    sc.add_box("wall", size=(0.10, 0.40, 0.40), pos=(0.20, 0.0, 0.20), static=True)
    sc.add_box("bad_box", size=(0.10, 0.10, 0.10), pos=(0.0, 0.0, 0.05), mass=0.5)
    sc.randomize("bad_box", "x", -0.25, 0.16)
    return str(sc.save(d)), str(d)


def _bad_x(card, seed: int, world: int | None = None) -> list[float]:
    """bad_box's start x per world of an 8-world build at ``seed``, or of world ``world`` built alone."""
    target, _ = scene_target(card[0], AuditOptions(), [card[1]], "cpu", draw_seed=seed)
    sim = target.make_sim(WORLDS, "cpu") if world is None else target.make_sim.alone(world)(1, "cpu")
    q = sim.state.body_q.numpy()
    return [float(q[i, 0]) for i, key in enumerate(sim.model.body_label) if key == "bad_box"]


def test_the_audit_seed_draws_the_scene_and_a_world_alone_keeps_its_draws(card):
    a, b = _bad_x(card, 101), _bad_x(card, 999)
    assert a and a != b                                        # the seed reaches the draws
    for w in (0, 3, 7):
        assert _bad_x(card, 101, w) == pytest.approx([a[w]], abs=1e-6)


def test_a_finding_in_world_w_is_blamed_and_replayed_in_world_w(card):
    r = audit(card[0], episodes=WORLDS, seed=101, device="cpu", horizon_s=0.2, batch=WORLDS, library=[card[1]])
    s1 = [f for f in r.findings if f.severity == "S1" and f.cause == "pen_s1_ratio" and f.body is not None]
    f = next(f for f in s1 if f.handle.world > 0)
    assert f.body.startswith("bad_box") and f.replays_alone, (f.body, f.body_detail)
    again = replay_finding(f.handle, "cpu", library=[card[1]])
    assert again is not None and again.handle.world == f.handle.world and again.body.startswith("bad_box")
    assert again.cause == f.cause and again.first_tick == f.first_tick


def test_cli_audit_takes_library_and_makes_out_folders(card, tmp_path, monkeypatch):
    out = io.StringIO()

    class Same:
        def __enter__(self):
            return out

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "_stdout_to_stderr", lambda: Same())
    report = tmp_path / "nested" / "deeper" / "report.json"
    rc = cli.main(["--log", "WARNING", "audit", card[0], "--episodes", "8", "--batch", "8", "--device", "cpu",
                   "--horizon", "0.1", "--seed", "101", "--library", card[1], "--out", str(report)])
    rec = json.loads(out.getvalue())
    assert rc in (0, 1) and report.is_file() and json.loads(report.read_text())["episodes"] == rec["episodes"] == 8
