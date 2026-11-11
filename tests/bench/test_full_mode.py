# SPDX-License-Identifier: Apache-2.0
"""SB-12 counts the full-mode verdict of a law whose quick mode is outside its stated range (coordinator 2026-11-05
05:48Z): the law stays in the denominator, the board names the mode, and a missing full row never passes."""

from __future__ import annotations

from alloy.bench import full_mode as FM
from alloy.bench import note_rows as NR
from alloy.bench import scoreboard as SB

_HEADER = "| id | value | status | command | commit |\n|---|---|---|---|---|\n"


def _notes(tmp_path, body: str):
    (tmp_path / "wp9-results.md").write_text("# WP9\n\n" + _HEADER + body)
    return tmp_path


def test_the_full_mode_row_decides_and_the_note_names_it(tmp_path):
    notes = _notes(tmp_path, "| A13.sealed_column quick (CPU) | 0.06 | pass | q | abc |\n"
                             "| A13.sealed_column full (GPU) | piston 0.855 | fail | f | 3d1606e5 |\n")
    c = FM.counted("A13.sealed_column", "pass", notes)
    assert c.verdict == "fail" and c.source == "docs/notes/wp9-results.md:6"
    assert "counted at its full-mode verdict on cuda" in c.note and "quick pass not counted" in c.note
    assert FM.counted("A13.hydrostatic", "pass", notes) is None     # quick inside its range: quick decides


def test_a_missing_full_row_counts_inconclusive_never_pass(tmp_path):
    notes = _notes(tmp_path, "| A13.piston_displacement quick (CPU) | 7.1 % low | not_run (too short) | q | abc |\n")
    c = FM.counted("A13.piston_displacement", "fail", notes)
    assert (c.verdict, c.source) == ("inconclusive", "")


def test_sb12_counts_the_seal_laws_at_their_full_rows_and_keeps_them_in_the_denominator(tmp_path, monkeypatch):
    """WP9's full GPU rows decide (piston pass, sealed column fail); both stay counted, tagged with the mode."""
    rows = ("| A13.sealed_column full (GPU) | 0.855 | fail | f | x |\n"
            "| A13.piston_displacement full (GPU) | +0.18 % | pass | f | x |\n")
    monkeypatch.setattr(NR, "NOTES", _notes(tmp_path, rows))

    class R:
        def __init__(self, law: str) -> None:
            self.law, self.solver, self.device, self.provenance = law, "sph", "cpu", {"quick": "True"}

    assert SB._counted_tag(R("A13.sealed_column")) == "A13.sealed_column/sph/cuda/full"
    monkeypatch.setattr(SB, "law_verdict", lambda r: "fail")         # the piston's quick fail ...
    assert SB._counted(R("A13.piston_displacement")) == "pass"        # ... is not what SB-12 counts
    monkeypatch.setattr(SB, "law_verdict", lambda r: "pass")         # the sealed column's quick pass ...
    assert SB._counted(R("A13.sealed_column")) == "fail"              # ... neither
    note, src = SB._full_mode([("p", R("A13.sealed_column"))])
    assert src == ["docs/notes/wp9-results.md:5"] and "quick pass not counted" in note
