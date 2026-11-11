# SPDX-License-Identifier: Apache-2.0
"""bench/newenv scoring on fake run dirs (rollouts written by common's synthetic builder), levels, declines,
the summary table and the restricted-arm notice rule."""
from __future__ import annotations

import json

import numpy as np

from bench.newenv import requests
from bench.newenv import score, summarize
from bench.newenv.requests import load_engines
from bench.newenv.checkers import common as c

N, T = 4, 41
REQUESTS = {"requests": [
    {"id": "A1", "group": "rigid", "request": "r", "checker": "generic_fake", "probe": {"x": []}},
    {"id": "B1", "group": "robot", "request": "r", "checker": "generic_fake", "probe": {"x": []}, "policy": True},
    {"id": "C1", "group": "liquid", "request": "r", "checker": "no_such_checker", "probe": {"x": []}},
    {"id": "D1", "group": "soft", "request": "r", "checker": "generic_fake", "probe": {"x": []}}]}


class GenericFake:
    """Stands in for checkers/<name>.py: only the generic checks."""

    @staticmethod
    def check(rollout, request):
        return c.finish(c.generic_checks(rollout, request))


def _rollout(**meta) -> c.Rollout:
    t = np.arange(T) * 0.01
    start = np.linspace(-0.1, 0.1, N)
    x = start[None, :] + 0.01 * (np.arange(T) % 20)[:, None] * (1 + np.arange(N))[None, :]
    done = np.zeros((T, N), bool)
    done[[20, 40]] = True
    success = np.zeros((T, N), bool)
    success[-1] = True
    reward = np.broadcast_to(np.linspace(0.0, 1.0, T)[:, None], (T, N)).copy()
    rest = c.make_phase("rest", t, {"x": np.broadcast_to(start, (T, N))}, obs=np.broadcast_to(start, (T, N))[:, :, None])
    rand = c.make_phase("random", t, {"x": x}, obs=x[:, :, None], done=done, reward=reward, success=success)
    return c.synthetic([rest, rand], **meta)


def _write_out(out, rollout: c.Rollout) -> None:
    out.mkdir(parents=True, exist_ok=True)
    flat = {}
    for p in rollout.phases.values():
        flat.update({f"{p.name}/{f}": getattr(p, f) for f in c.FIELDS})
        flat.update({f"{p.name}/probe/{k}": v for k, v in p.probe.items()})
    np.savez(out / "rollout.npz", **flat)
    (out / "result.json").write_text(json.dumps(rollout.meta))


def _fake_run(root, engine, task, rep, kind="good", module=None):
    d = root / f"{engine}-{task}-r{rep}"
    (d / "task" / "out").mkdir(parents=True)
    (d / "modules").mkdir()
    out = d / "task" / "out"
    if kind == "good":
        _write_out(out, _rollout())
    elif kind == "policy":
        _write_out(out, _rollout(policy={"success_rate": 1.0}))
    elif kind == "nan":
        r = _rollout()
        r.phase("random").obs[5, 0, 0] = np.nan
        _write_out(out, r)
    elif kind == "decline":
        (out / "result.json").write_text(json.dumps({"supported": False, "reason": "no liquids"}))
    (d / "meta.json").write_text(json.dumps({"engine": engine, "task": task, "rep": str(rep),
                                             "agent_wall_s": 600.0, "rerun_wall_s": 30.0}))
    (d / "claude_rc").write_text("0")
    (d / "rerun_rc").write_text("0")
    (d / "modules" / "1.txt").write_text(f"top os sys {module or load_engines()[engine].module}\n")
    msgs = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True, "content": "x"}]}},
            {"type": "result", "num_turns": 7, "total_cost_usd": 1.5, "usage": {"input_tokens": 10,
             "output_tokens": 200, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 5}}]
    (d / "transcript.jsonl").write_text("\n".join(json.dumps(m) for m in msgs))
    (d / "DONE").touch()
    return d


def _score(tmp_path, monkeypatch):
    f = tmp_path / "requests.json"
    f.write_text(json.dumps(REQUESTS))
    real = score.load_checker

    def load(name):
        return GenericFake if name == "generic_fake" else real(name)
    monkeypatch.setattr(score, "load_checker", load)
    return {r["id"]: r for r in score.score_dir(tmp_path / "runs", f)}


def test_levels_and_verdicts(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _fake_run(runs, "genesis", "A1", 1)                            # L2 (no level-3 check for a non-robot request)
    _fake_run(runs, "genesis", "B1", 1, "policy")                  # robot: L3
    _fake_run(runs, "genesis", "B1", 2)                            # robot, policy flag missing: L2 fails at level 3 -> L2
    _fake_run(runs, "genesis", "D1", 1, "nan")                     # NaN obs: L0
    _fake_run(runs, "genesis", "A1", 2, module="numpy")            # engine never imported: verdict L2, not a success
    rows = _score(tmp_path, monkeypatch)
    assert rows["genesis-A1-r1"]["status"] == "L2" and rows["genesis-A1-r1"]["success"]
    assert not rows["genesis-A1-r1"]["solved"]
    b1, b2 = rows["genesis-B1-r1"], rows["genesis-B1-r2"]
    assert (b1["status"], b1["level"], b1["success"], b1["solved"]) == ("L3", 3, True, True)
    assert (b2["status"], b2["success"], b2["solved"]) == ("L2", True, False)
    assert "policy_success_flag" in b2["failed"]
    nan = rows["genesis-D1-r1"]
    assert (nan["status"], nan["level"], nan["success"]) == ("L0", 0, False) and "finite" in nan["failed"]
    assert rows["genesis-A1-r2"]["status"] == "L2" and not rows["genesis-A1-r2"]["success"]
    ok = rows["genesis-A1-r1"]
    assert (ok["turns"], ok["output_tokens"], ok["tokens_total"], ok["tool_errors"]) == (7, 200, 1215, 1)
    assert ok["group"] == "rigid" and ok["agent_wall_s"] == 600.0 and ok["supported"]


def test_correct_and_wrong_declines(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _fake_run(runs, "maniskill", "D1", 1, "decline")               # soft: ManiSkill's docs say no -> correct decline
    _fake_run(runs, "genesis", "D1", 1, "decline")                 # Genesis supports soft -> L0
    _fake_run(runs, "maniskill", "A1", 1, "decline")               # rigid: ManiSkill supports it -> L0
    _fake_run(runs, "maniskill", "C1", 1, "decline")               # liquid decline needs no checker
    rows = _score(tmp_path, monkeypatch)
    right = rows["maniskill-D1-r1"]
    assert (right["status"], right["declined"], right["correct_decline"], right["supported"]) == \
        ("declined", True, True, False) and not right["success"]
    wrong = rows["genesis-D1-r1"]
    assert (wrong["status"], wrong["correct_decline"], wrong["supported"]) == ("L0", False, True)
    assert rows["maniskill-A1-r1"]["status"] == "L0"
    assert rows["maniskill-C1-r1"]["status"] == "declined"


def test_missing_checker_is_an_error_row(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _fake_run(runs, "genesis", "C1", 1)
    row = _score(tmp_path, monkeypatch)["genesis-C1-r1"]
    assert row["status"] == "error" and "checker missing" in row["error"] and not row["success"]
    assert "Not scored (checker missing" in summarize.render([row])
    assert "genesis" not in summarize.render([row]).split("Not scored")[0].split("## By engine")[1].split("\n\n")[1]


def test_checker_crash_caps_at_level_1(tmp_path, monkeypatch):
    class Broken:
        check = staticmethod(lambda rollout, request: rollout.phase("act"))

    runs = tmp_path / "runs"
    _fake_run(runs, "genesis", "A1", 1)
    f = tmp_path / "requests.json"
    f.write_text(json.dumps(REQUESTS))
    monkeypatch.setattr(score, "load_checker", lambda name: Broken)
    row = score.score_dir(runs, f)[0]
    assert row["status"] == "L1" and "KeyError" in row["error"] and not row["success"]


def test_missing_outputs_are_level_0(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    d = _fake_run(runs, "genesis", "A1", 1)
    (d / "task" / "out" / "result.json").unlink()
    (d / "task" / "out" / "rollout.npz").unlink()
    assert _score(tmp_path, monkeypatch)["genesis-A1-r1"]["status"] == "L0"


def test_restricted_notice_and_public_flag(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    notice = "Restricted arm: test notice."
    (extra / "engines.json").write_text(json.dumps({"restricted": {
        "name": "Restricted", "port": 3299, "module": "restricted", "install": "x", "docs": "y", "notice": notice}}))
    monkeypatch.setenv(requests.EXTRA_ENV, str(extra))
    runs = tmp_path / "runs"
    _fake_run(runs, "genesis", "A1", 1)
    _fake_run(runs, "restricted", "A1", 1)
    rows = list(_score(tmp_path, monkeypatch).values())
    out = tmp_path / "rows.jsonl"
    score.write_rows(rows, out)
    assert out.read_text().splitlines()[0] == notice
    assert len(score.read_rows(out)) == 2
    text = summarize.render(rows)
    assert text.startswith(notice) and "restricted" in text
    pub = summarize.render(rows, public=True)
    assert "restricted" not in pub and notice not in pub
    score.write_rows(rows, out, public=True)
    assert notice not in out.read_text() and "restricted" not in out.read_text()


def test_summary_levels_supported_and_robot_columns(tmp_path, monkeypatch):
    extra = tmp_path / "extra"                                      # a made-up arm whose docs rule out soft bodies
    extra.mkdir()
    (extra / "engines.json").write_text(json.dumps({"armx": {
        "name": "Arm X", "port": 3298, "module": "armx", "install": "x", "docs": "y", "declines": ["soft"]}}))
    monkeypatch.setenv(requests.EXTRA_ENV, str(extra))
    runs = tmp_path / "runs"
    _fake_run(runs, "armx", "A1", 1)                          # L2
    _fake_run(runs, "armx", "A1", 2, "nan")                   # L0
    _fake_run(runs, "armx", "D1", 1, "decline")               # correct decline: only in "all requests"
    _fake_run(runs, "armx", "B1", 1, "policy")                # robot L3
    lost = _fake_run(runs, "armx", "A1", 3)
    (lost / "transcript.jsonl").write_text("")                     # no agent result: lost to infra
    text = summarize.render(list(_score(tmp_path, monkeypatch).values()))
    assert "| armx | 2/4 (15-85 %) | 2/3 (21-94 %) | 1/1 (21-100 %) | L0 1, L2 1, L3 1, declined 1 | " in text
    assert "| armx | robot | 1/1 (21-100 %) | 1/1 (21-100 %) | 1/1 (21-100 %) | L3 1 | " in text
    assert "| armx | soft | 0/1 (0-79 %) | - | - | declined 1 | " in text
    assert "Lost to the harness or API (not counted): armx-A1-r3" in text
    assert "L2+ (supported requests)" in text and "## By engine and group" in text
    assert summarize.cell([{"success": True}, {"success": False}]) == "1/2 (9-91 %)"
