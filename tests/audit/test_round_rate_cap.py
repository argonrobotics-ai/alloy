# SPDX-License-Identifier: Apache-2.0
"""Regression test: alloy.audit check rounds must count each finding at most once per round."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alloy.audit import run as audit_run
from alloy.audit.report import AuditReport


def test_audit_check_rounds_caps_rate_at_one(monkeypatch):
    """Duplicate normalized findings within one round must not push AuditFinding.rate above 1.0."""
    import alloy.check as check_pkg

    dup = SimpleNamespace(check="overlap", level="error", objects=("box",), message="box overlaps by 1.2 mm",
                          fix="lift box", value=1.2, limit=0.0, unit="mm")
    dup2 = SimpleNamespace(check="overlap", level="error", objects=("box",), message="box overlaps by 1.5 mm",
                           fix="lift box", value=1.5, limit=0.0, unit="mm")
    monkeypatch.setattr(check_pkg, "check", lambda *a, **kw: SimpleNamespace(findings=[dup, dup2], settings={}))
    report = AuditReport("s")
    src = SimpleNamespace(path=None, label="s", card=None, lib=None)
    audit_run._check_rounds(report, src, (1, 0, ()), "cpu", 1, 0.1)
    assert len(report.findings) == 1
    assert report.findings[0].seen == 1 and report.findings[0].rate == pytest.approx(1.0)
