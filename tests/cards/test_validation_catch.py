# SPDX-License-Identifier: Apache-2.0
"""CARD-5: seeded invalid cards are rejected with the field named (target 100 %)."""

from __future__ import annotations

from pathlib import Path

import pytest

from alloy.cards.library import CardLibrary
from alloy.testing.cards import run_case, seeded_bad_cases

CASES = seeded_bad_cases()


def test_corpus_size_and_unique_ids() -> None:
    assert len(CASES) >= 50
    assert len({c.id for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_seeded_card_rejected_with_field(case, tmp_library: Path) -> None:
    lib = CardLibrary([tmp_library])
    ok, named = run_case(case, lib)
    assert ok, f"{case.id} ({case.rule}): expected an error at {case.field!r}, got {named}"
