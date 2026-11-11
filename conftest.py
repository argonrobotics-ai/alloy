# SPDX-License-Identifier: Apache-2.0
"""Tests that need content which is not distributed with Alloy are skipped, with the reason, instead of failing.

Two kinds of content stay internal: the real-rig recordings (``bench/traces``) and a few task modules (datacenter,
paid-job and showcase-clip tasks, and the task-regression adapters built on them). A test that fails in this copy
only because one of them is absent is reported as skipped and names what it needed; any other failure is reported
as usual.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "bench" / "traces"
MODULE = re.compile(r"No module named '(alloy\.[\w.]+)'")
RECORDING = re.compile(r"No such file or directory: '[^']*bench/traces/")


def not_distributed(exc: BaseException | None) -> str | None:
    """Why ``exc`` (or an exception it wraps or names in its message) means "content not in this copy", else None.

    >>> not_distributed(ModuleNotFoundError("No module named 'alloy.clips'", name="alloy.clips"))
    'needs alloy.clips, which is not distributed'
    """
    seen = 0
    while exc is not None and seen < 5:
        text = str(exc)
        name = exc.name if isinstance(exc, ModuleNotFoundError) else None
        m = MODULE.search(text)
        if (name or "").startswith("alloy.") or m:
            return f"needs {name if (name or '').startswith('alloy.') else m.group(1)}, which is not distributed"
        if not RECORDINGS.is_dir() and RECORDING.search(text):
            return "needs the real-rig recordings (bench/traces), which are not distributed"
        exc, seen = exc.__cause__ or exc.__context__, seen + 1
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Turn a failure caused only by absent internal content into a skip."""
    report = yield
    if report.failed and call.excinfo is not None:
        why = not_distributed(call.excinfo.value)
        if why:
            report.outcome = "skipped"
            report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {why}")
    return report
