# SPDX-License-Identifier: Apache-2.0
"""BUL-2 (SPEC §5.2): every closed finding in the repo corpus replays clean; target 0 regressions.

Closed findings are replayed from a scratch copy of their directory, so the run never writes into the repo. Device:
``ALLOY_DEVICE`` (default cpu). The replays are ``slow`` (about 40 s per box-stack finding, 150 s per T12 finding
on CPU, hence their own timeout), so the CPU gate skips them and runs only the check that every filed finding loads
(record + trace) and names a registered target.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from alloy.bully.command import command_of, is_command
from alloy.bully.corpus import Corpus
from alloy.bully.targets import target_names

CORPUS = Corpus(Path(__file__).resolve().parents[2] / "bench" / "bully" / "corpus")
DEVICE = os.environ.get("ALLOY_DEVICE", "cpu")


def _closed() -> list[str]:
    """Closed trace findings (a command finding replays from a pinned checkout on its GPU type, not here)."""
    return [fid for fid in CORPUS.ids()
            if CORPUS.record(fid)["status"] == "closed" and not is_command(CORPUS.record(fid))]


def _trace_ids() -> list[str]:
    return [fid for fid in CORPUS.ids() if not is_command(CORPUS.record(fid))]


def test_command_findings_name_a_pinned_run() -> None:
    for fid in (i for i in CORPUS.ids() if is_command(CORPUS.record(i))):
        rec, cmd = CORPUS.record(fid), command_of(CORPUS.record(fid))
        assert rec["scene_hash"] == cmd.hash() and not (CORPUS.root / fid / "trace.npz").exists(), fid
        assert len(cmd.commit) == 40 and cmd.pins and cmd.markers and "{device}" in cmd.argv, fid
        assert rec["alloy_sha"] == cmd.commit and rec["source"].get("measured_by"), fid


def test_filed_findings_load_and_name_a_target() -> None:
    known = set(target_names())
    for fid in _trace_ids():
        finding, trace, dt, _ = CORPUS.load(fid)
        assert finding.id == fid
        assert trace.ndim == 2 and trace.shape[0] > 0 and dt > 0.0
        assert (CORPUS.record(fid)["target"] or finding.target) in known, fid


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("fid", _closed() or [pytest.param("", marks=pytest.mark.skip(reason="no closed finding"))])
def test_closed_finding_replays_clean(fid: str, tmp_path: Path) -> None:
    shutil.copytree(CORPUS.root / fid, tmp_path / fid)
    result = Corpus(tmp_path).replay(fid, DEVICE, runs=10)
    assert result.trips == 0, f"{fid} still trips {result.trips}/{result.runs} after its fix (BUL-2)"
