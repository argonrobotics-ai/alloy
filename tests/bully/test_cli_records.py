# SPDX-License-Identifier: Apache-2.0
"""Gemini 0.1.2 audit S2-040 / S2-047: the bully CLI's records and command findings (CPU, no simulation).

- Every record is strict JSON: a blown-up world's ``inf`` / ``nan`` are strings, never the bare ``Infinity``.
- ``run`` prints the findings as filed (``BUL-nnnn``, a runnable repro naming the corpus, REPEAT severities), not the
  search's ``NEW-0001`` placeholders.
- ``replay --close`` exits 0 only when every run was clean and the finding closed.
- A command finding that dies on SIGFPE (or SIGILL, SIGBUS, ...) is a crash; ``{out}`` expands inside an argument;
  an Alloy error is a JSON record with exit 2, not a traceback.
"""

from __future__ import annotations

import io
import json
import math
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.bully import __main__ as cli
from alloy.bully.command import CrashCommand, run_crashed
from alloy.bully.corpus import Corpus
from alloy.bully.jsonsafe import dumps, plain
from alloy.bully.types import BullySeverity, Finding, ReplayResult
from alloy.core.errors import BullyError


def _strict(text: str):
    def reject(token: str):
        raise ValueError(f"non-RFC 8259 token {token}")

    return json.loads(text, parse_constant=reject)


def _finding(value: float = math.inf, fid: str = "NEW-0001", corpus: str | None = None) -> Finding:
    repro = "python -m alloy.bully replay <id> --device cpu"
    return Finding(fid, BullySeverity.S3, "max_pen", value, 0.01, 4, "a" * 64, "b" * 40, "c" * 40, repro,
                   target="fake")


def _run_cli(argv: list[str], monkeypatch) -> tuple[int, dict]:
    out = io.StringIO()
    monkeypatch.setattr(cli, "_stdout_to_stderr", lambda: _Yield(out))
    rc = cli.main(["--log", "WARNING", *argv])
    return rc, _strict(out.getvalue())


class _Yield:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, *exc):
        return False


def test_non_finite_numbers_are_strings_and_read_back():
    rec = {"value": np.float32(np.inf), "nan": float("nan"), "low": -math.inf, "rows": np.array([1.0, np.inf])}
    assert plain(rec) == {"value": "inf", "nan": "nan", "low": "-inf", "rows": [1.0, "inf"]}
    assert _strict(dumps(rec))["rows"] == [1.0, "inf"]


def test_a_filed_finding_is_strict_json_and_names_its_corpus(tmp_path):
    corpus = Corpus(tmp_path / "mine")
    path = corpus.add(_finding(), np.zeros((3, 2)), dt=0.01, seed=0, target="fake")
    rec = _strict((path / "finding.json").read_text())
    assert rec["id"] == "BUL-0001" and rec["value"] == "inf"
    assert rec["repro"] == f"python -m alloy.bully replay BUL-0001 --device cpu --corpus {tmp_path / 'mine'}"
    assert corpus.load("BUL-0001")[0].value == math.inf                       # read back as a number
    default = Corpus.__new__(Corpus)
    default.root = Path("bench/bully/corpus")
    assert "--corpus" not in default._repro("python -m alloy.bully replay <id>", "BUL-0007")


def test_run_prints_the_filed_records(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "c"
    closed = Corpus(corpus_dir)
    closed.add(_finding(1.0), np.zeros((3, 2)), dt=0.01, seed=0, target="fake")
    closed.set_status("BUL-0001", "closed")                   # the same cause again: a REPEAT, one severity up
    outcome = SimpleNamespace(findings=[_finding()], traces={"NEW-0001": np.zeros((3, 2))}, generations_run=1,
                              wall_s=0.1, best_badness=math.inf, first_trip_generation=0, replays={"max_pen": (3, 3)})
    monkeypatch.setattr(cli, "get_target", lambda name: SimpleNamespace(name="fake", actions=SimpleNamespace(rate_hz=100)))
    monkeypatch.setattr(cli, "search_detailed", lambda *a, **k: outcome)
    rc, rec = _run_cli(["run", "--target", "fake", "--corpus", str(corpus_dir)], monkeypatch)
    assert rc == 0 and rec["filed"] == ["BUL-0002"] and rec["best_badness"] == "inf"
    f = rec["findings"][0]
    assert f["id"] == "BUL-0002" and f["repeat_of"] == "BUL-0001" and f["severity"] == int(BullySeverity.S2)
    assert f["repro"].endswith(f"replay BUL-0002 --device cpu --corpus {corpus_dir}")
    assert f == _strict((corpus_dir / "BUL-0002" / "finding.json").read_text())


@pytest.mark.parametrize(("trips", "close", "rc", "status"), [(3, True, 1, "open"), (0, True, 0, "closed"),
                                                              (3, False, 0, "open"), (0, False, 1, "open")])
def test_replay_close_exits_zero_only_when_it_closed(tmp_path, monkeypatch, trips, close, rc, status):
    corpus_dir = tmp_path / "c"
    Corpus(corpus_dir).add(_finding(1.0), np.zeros((3, 2)), dt=0.01, seed=0, target="fake")
    monkeypatch.setattr(Corpus, "replay", lambda self, fid, *a, **k: ReplayResult(fid, trips, 3, "cpu", trips > 0))
    argv = ["replay", "BUL-0001", "--runs", "3", "--corpus", str(corpus_dir)] + (["--close"] if close else [])
    got, rec = _run_cli(argv, monkeypatch)
    assert (got, rec["status"], rec["replay"]["trips"]) == (rc, status, trips)


def test_an_alloy_error_is_a_json_record_with_exit_2(tmp_path, monkeypatch):
    rc, rec = _run_cli(["replay", "BUL-9999", "--corpus", str(tmp_path)], monkeypatch)
    assert rc == 2 and rec["error"]["type"] == "BullyError" and "BUL-9999" in rec["error"]["message"]


def _command(*args: str) -> CrashCommand:
    return CrashCommand(("{python}", "-c", *args), "0" * 40, "pins.txt", timeout_s=60.0)


@pytest.mark.parametrize("sig", [signal.SIGFPE, signal.SIGSEGV, signal.SIGILL, signal.SIGBUS, signal.SIGABRT])
def test_a_fault_signal_is_a_crash(tmp_path, sig):
    cmd = _command(f"import os, signal; os.kill(os.getpid(), signal.{sig.name})")
    assert run_crashed(cmd, tmp_path, "cpu") is True


def test_a_kill_from_outside_or_an_error_exit_is_not_a_crash(tmp_path):
    with pytest.raises(BullyError, match="killed by SIGKILL"):
        run_crashed(_command("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"), tmp_path, "cpu")
    with pytest.raises(BullyError, match=r"rc 3\)"):
        run_crashed(_command("raise SystemExit(3)"), tmp_path, "cpu")
    assert run_crashed(_command("pass"), tmp_path, "cpu") is False


def test_placeholders_expand_inside_arguments():
    cmd = CrashCommand(("{python}", "{out}", "{out}/dump.bin", "--out={out}", "--device={device}"), "0" * 40, "p")
    assert cmd.resolve("cuda:0", Path("/x/run")) == [sys.executable, "/x/run", "/x/run/dump.bin", "--out=/x/run",
                                                      "--device=cuda:0"]
    assert cmd.hash() == CrashCommand(cmd.argv, "1" * 40, "q").hash()       # the dedup key is the argv as filed
