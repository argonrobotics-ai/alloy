# SPDX-License-Identifier: Apache-2.0
"""Board tests build SB rows from their own producer rows: rows read straight from results notes
(`alloy.bench.note_rows`) come from an empty notes directory unless a test points `NOTES` elsewhere."""

from __future__ import annotations

import pytest

from alloy.bench import note_rows


@pytest.fixture(autouse=True)
def _no_live_note_rows(tmp_path_factory, monkeypatch) -> None:
    monkeypatch.setattr(note_rows, "NOTES", tmp_path_factory.mktemp("no_notes"))
