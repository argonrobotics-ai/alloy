# SPDX-License-Identifier: Apache-2.0
"""The rinse verdict's rules (alloy/tasks/rinse_score.py; audit AB-gym-3): a gate that is not judged is None, success
counts judged gates only, and the verdict says how many of Gym-Anything's criteria it judged."""

from __future__ import annotations

from alloy.tasks.rinse_score import FAIL_SCALE, Verdict, finish, partly


def test_partly_joins_a_criterions_parts() -> None:
    assert partly(True, True) is True
    assert partly(True, None) is None                      # a part not judged: the criterion is not judged
    assert partly(False, None) is False                    # a judged part that fails still fails it
    assert partly() is True


def test_success_counts_judged_gates_only() -> None:
    v = Verdict(criteria={"clean": (True, 0.1), "hands": (None, "not judged"), "own": (True, 0)}, extra=("own",))
    assert v.success and v.failed() == [] and v.not_judged() == ["hands"]
    assert v.judged == (1, 2)                              # the port's own gate is not one of Gym-Anything's
    assert v.summary() == "1 of 2 Gym-Anything criteria judged; not judged: hands"
    v.criteria["own"] = (False, 1)                         # the port's own gate still counts for success
    assert not v.success and v.failed() == ["own"]


def test_a_verdict_with_nothing_judged_is_not_a_success() -> None:
    v = Verdict(criteria={"hands": (None, "not judged")})
    assert not v.success and v.judged == (0, 1)
    assert not Verdict().success


def test_all_judged_reads_as_before() -> None:
    v = finish(Verdict(criteria={"a": (True, 1), "b": (False, 2)}), 0.5, True)
    assert not v.success and v.failed() == ["b"] and v.score == FAIL_SCALE * 0.5
    assert v.summary() == "2 of 2 Gym-Anything criteria judged"
    assert finish(Verdict(criteria={"a": (True, 1), "b": (None, "")}), 0.5, True).score == 1.0
