# SPDX-License-Identifier: Apache-2.0
"""User-facing API warnings reach stderr when no log handler would show them (Alloy's logger has a NullHandler)."""

from __future__ import annotations

import logging

from alloy.api import _say


def test_warning_is_printed_without_a_handler(monkeypatch, capsys):
    quiet = logging.getLogger("alloy_say_test.quiet")
    quiet.propagate = False
    quiet.addHandler(logging.NullHandler())
    monkeypatch.setattr(_say, "log", quiet)
    _say.warn("%s: %d frozen", "world 0", 1)
    assert capsys.readouterr().err == "alloy: world 0: 1 frozen\n"


def test_warning_is_not_doubled_with_a_handler(monkeypatch, capsys):
    shown = logging.getLogger("alloy_say_test.shown")
    shown.propagate = False
    shown.addHandler(logging.StreamHandler())
    monkeypatch.setattr(_say, "log", shown)
    _say.warn("hello")
    assert capsys.readouterr().err == "hello\n"                              # the handler's line only
