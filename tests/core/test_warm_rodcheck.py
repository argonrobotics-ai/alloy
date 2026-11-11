# SPDX-License-Identifier: Apache-2.0
"""SCR-OSS1-9: ``python -m alloy.core.warm`` also warms the rod self-tests via ``alloy.route.rodcheck.warm`` (WP2) when
it exists, and logs and skips while it does not."""

from __future__ import annotations

import logging

import pytest

from alloy.core import warm as W


def test_warm_calls_rodcheck_warm(monkeypatch: pytest.MonkeyPatch) -> None:
    from alloy.route import rodcheck

    calls = []
    monkeypatch.setattr(rodcheck, "warm", lambda device: calls.append(device), raising=False)
    monkeypatch.setattr(W, "import_all", lambda: [])
    monkeypatch.setattr(W, "_warm_core", lambda device: None)
    times = W.warm("cpu", solvers=())
    assert calls == ["cpu"] and "rodcheck" in times


def test_warm_skips_rodcheck_without_warm(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from alloy.route import rodcheck

    monkeypatch.delattr(rodcheck, "warm", raising=False)
    with caplog.at_level(logging.INFO, logger="alloy.core"):
        W._warm_rodcheck("cpu")
    assert "no warm() yet" in caplog.text
