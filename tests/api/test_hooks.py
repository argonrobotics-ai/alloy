# SPDX-License-Identifier: Apache-2.0
"""alloy.api._hooks.after_import: the post-import hook rules."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from alloy.api import _hooks


@pytest.fixture
def modpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A directory on sys.path; ``make(name, body)`` writes a module there. Imported test modules are dropped."""
    monkeypatch.syspath_prepend(str(tmp_path))
    made: list[str] = []

    def make(name: str, body: str = "X = 1\n") -> str:
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
        made.append(name)
        return name

    yield make
    for name in made:
        sys.modules.pop(name, None)
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _hooks._OneShotFinder)]


def test_runs_at_once_when_already_imported():
    calls = []
    _hooks.after_import("json", lambda: calls.append(1))
    assert calls == [1]


def test_runs_once_after_the_module_body(modpath):
    name = modpath("oss3_hook_a", "VALUE = 41\n")
    seen = []
    _hooks.after_import(name, lambda: seen.append(sys.modules[name].VALUE))
    _hooks.after_import(name, lambda: seen.append("second"))
    assert _hooks.pending(name) == 2 and seen == []
    __import__(name)
    assert seen == [41, "second"]
    assert _hooks.pending(name) == 0
    assert not any(isinstance(f, _hooks._OneShotFinder) for f in sys.meta_path)   # the finder removed itself
    sys.modules.pop(name)
    __import__(name)                       # a re-import does not run the callbacks again
    assert seen == [41, "second"]


def test_finder_is_gone_before_callbacks_run(modpath):
    name = modpath("oss3_hook_b")
    state = {}
    _hooks.after_import(name, lambda: state.update(finders=_hooks.pending(name)))
    __import__(name)
    assert state["finders"] == 0


def test_callback_error_propagates(modpath):
    name = modpath("oss3_hook_c")

    def boom() -> None:
        raise RuntimeError("install failed")

    _hooks.after_import(name, boom)
    with pytest.raises(RuntimeError, match="install failed"):
        __import__(name)


def test_failed_import_runs_nothing_and_keeps_waiting(modpath):
    name = modpath("oss3_hook_d", "raise ImportError('broken on purpose')\n")
    calls = []
    _hooks.after_import(name, lambda: calls.append(1))
    with pytest.raises(ImportError, match="broken on purpose"):
        __import__(name)
    assert calls == [] and _hooks.pending(name) == 1
    modpath("oss3_hook_d", "X = 2\n")
    sys.modules.pop(name, None)
    __import__(name)
    assert calls == [1]


def test_unwatched_imports_are_untouched(modpath):
    name = modpath("oss3_hook_e")
    _hooks.after_import("oss3_never_imported", lambda: None)
    mod = __import__(name)
    assert not isinstance(mod.__spec__.loader, _hooks._HookLoader)
