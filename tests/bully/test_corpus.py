# SPDX-License-Identifier: Apache-2.0
"""Corpus round trip, REPEAT filing, replay records, regressions and command findings (SPEC §3.11, BUL-2)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from alloy.core.errors import BullyError
from alloy.bully.command import CrashCommand, is_command, replay_command, run_crashed
from alloy.bully.corpus import Corpus
from alloy.bully.replay import replay_trace
from alloy.bully.search import scene_hash
from alloy.bully.stress import named_stress_phases
from alloy.bully.types import BullyLimits, BullySeverity, Finding
from alloy.testing.bully import seeded_fake_target, stable_fake_target


def _finding(monitor: str = "max_speed", sev: BullySeverity = BullySeverity.S2, h: str = "h" * 64) -> Finding:
    return Finding("NEW-0001", sev, monitor, 25.0, 20.0, 12, h, "a" * 40, "n" * 40,
                   "python -m alloy.bully replay <id> --device cpu")


def test_round_trip(tmp_path):
    c = Corpus(tmp_path)
    trace = np.arange(12, dtype=np.float32).reshape(6, 2)
    path = c.add(_finding(), trace, dt=0.02, seed=7, target="fake_seeded")
    assert path.name == "BUL-0001" and c.ids() == ["BUL-0001"]
    f, t, dt, seed = c.load("BUL-0001")
    assert f.id == "BUL-0001" and f.repro.endswith("replay BUL-0001 --device cpu")
    assert f.target == "fake_seeded"
    assert dataclasses.replace(f, id="NEW-0001", repro=_finding().repro, target="") == _finding()
    np.testing.assert_array_equal(t, trace)
    assert (dt, seed) == (0.02, 7)
    rec = json.loads((path / "finding.json").read_text())
    assert rec["schema"] == "alloy.bully.finding/1" and rec["status"] == "open" and rec["target"] == "fake_seeded"
    assert c.next_id() == "BUL-0002"


def test_open_duplicate_not_filed_closed_cause_is_repeat(tmp_path):
    c = Corpus(tmp_path)
    c.add(_finding(), np.zeros((2, 2)), 0.02, 0)
    assert c.add(_finding(), np.zeros((2, 2)), 0.02, 0).name == "BUL-0001"  # same open cause
    c.set_status("BUL-0001", "closed")
    path = c.add(_finding(), np.zeros((2, 2)), 0.02, 0)
    f, *_ = c.load(path.name)
    assert path.name == "BUL-0002" and f.repeat_of == "BUL-0001"
    assert f.severity == BullySeverity.S1  # one severity higher
    other = c.add(_finding(monitor="teleport"), np.zeros((2, 2)), 0.02, 0)
    assert c.load(other.name)[0].repeat_of is None
    with pytest.raises(BullyError):
        c.set_status("BUL-0001", "fixed")
    with pytest.raises(BullyError):
        c.load("BUL-0999")


def test_replay_from_corpus_and_regressions(tmp_path):
    target = seeded_fake_target()
    trace = dict(named_stress_phases(target))["val115_4_drive_high_hold"]
    c = Corpus(tmp_path)
    f = _finding("max_speed", BullySeverity.S1, scene_hash(target))
    fid = c.add(f, trace, 0.02, 0, target="fake_seeded").name
    res = c.replay(fid, "cpu", runs=10)
    assert (res.trips, res.runs, res.counts) == (10, 10, True)  # 10/10 on CPU
    assert c.last_replay(fid)["trips"] == 10
    c.set_status(fid, "closed")
    assert c.regressions() == [fid]  # closed but still trips: a regression
    clean = replay_trace(stable_fake_target(), trace, "max_speed", "cpu", BullyLimits(), runs=3)
    assert clean.trips == 0 and not clean.counts


def _crash_cmd(code: str) -> CrashCommand:
    return CrashCommand(("{python}", "-c", code, "{device}", "{out}"), "c" * 40, "freeze.txt", "L4",
                        ("CUDA error 700",), timeout_s=60.0)


_SEGV = "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"
_MARK = "import sys; print('Warp CUDA error 700: an illegal memory access'); sys.exit(1)"


def test_command_finding_files_without_a_trace_and_replays(tmp_path):
    c = Corpus(tmp_path)
    f = Finding("NEW-0001", BullySeverity.S1, "crash", 1.0, 0.5, -1, "", "c" * 40, "n" * 40,
                "python -m alloy.bully replay <id> --device cuda:0 --workdir <checkout>")
    fid = c.add_command(f, _crash_cmd(_SEGV), {"measured_by": "test"}).name
    rec = c.record(fid)
    assert is_command(rec) and rec["scene_hash"] == _crash_cmd(_SEGV).hash() and rec["target"] == ""
    assert not (tmp_path / fid / "trace.npz").exists()
    with pytest.raises(BullyError):
        c.load(fid)
    assert c.add_command(f, _crash_cmd(_SEGV), {}).name == fid  # same argv, still open: not filed again
    res = c.replay(fid, "cpu", runs=2, workdir=tmp_path)
    assert (res.trips, res.runs, res.counts) == (2, 2, True)
    assert c.last_replay(fid)["kind"] == "command"


def test_command_run_marker_clean_and_setup_error(tmp_path):
    assert run_crashed(_crash_cmd(_MARK), tmp_path, "cpu")  # marker in the output, ordinary exit code
    assert not run_crashed(_crash_cmd("print('ok')"), tmp_path, "cpu")
    with pytest.raises(BullyError):  # a non-zero exit without a crash marker is a setup error, not a trip
        run_crashed(_crash_cmd("raise SystemExit(2)"), tmp_path, "cpu")
    rec = {"id": "BUL-9", "kind": "command", "command": _crash_cmd("print('ok')").to_json()}
    res, extra = replay_command(rec, "cpu", 3, tmp_path)
    assert (res.trips, res.counts) == (0, False) and extra["kind"] == "command"
