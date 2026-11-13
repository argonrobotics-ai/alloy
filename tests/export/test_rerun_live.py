# SPDX-License-Identifier: Apache-2.0
"""examples/export/rerun_live.py (T3-013, Gemini audit of 0.1.2): it runs with ``--worlds 1``, the setting the frame
export guide gives for a CPU, and it logs the last captured frame of every picked world (it flushes the in-flight copy
before its publisher's final poll; on a GPU a one-step run logged no frame before).

``rerun`` is replaced by a recorder and the example's cloth scene by a falling box (the bug is in ``main``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import alloy

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "export" / "rerun_live.py"


class _Rerun(ModuleType):
    """The calls rerun_live.py makes; records (world, t) of every body pose it logs."""

    def __init__(self) -> None:
        super().__init__("rerun")
        self.frames: set[tuple[str, float]] = set()
        self._t = float("nan")

    def init(self, *_a, **_k) -> None: ...
    def save(self, *_a, **_k) -> None: ...

    def set_time(self, _timeline: str, duration: float = 0.0) -> None:
        self._t = float(duration)

    def set_time_seconds(self, _timeline: str, t: float = 0.0) -> None:
        self._t = float(t)

    def log(self, path: str, *_a, static: bool = False, **_k) -> None:
        if not static and "/bodies/" in path:
            self.frames.add((path.split("/")[0], round(self._t, 6)))

    def __getattr__(self, name: str):                 # Boxes3D, Transform3D, Quaternion, ...: plain records
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: (name, a, k)


def _box(worlds: int) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("block", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.2), mass=0.2)
    return s


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str, worlds: int, seconds: float) -> _Rerun:
    spec = importlib.util.spec_from_file_location("rerun_live_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rr = _Rerun()
    build = alloy.build
    monkeypatch.setitem(sys.modules, "rerun", rr)
    monkeypatch.setattr(mod, "scene", _box)
    monkeypatch.setattr(mod.alloy, "build", lambda scene, **kw: build(scene, device=device, **kw))
    monkeypatch.setattr(sys, "argv", ["rerun_live.py", "--worlds", str(worlds), "--seconds", str(seconds),
                                      "--save", str(tmp_path / "run.rrd")])
    mod.main()
    return rr


@pytest.mark.compile_heavy                                      # cold kernel cache: over 60 s on an L4 VM
def test_one_world_runs_and_logs_its_last_frame(monkeypatch, tmp_path):
    rr = _run(monkeypatch, tmp_path, "cpu", worlds=1, seconds=0.03)    # worlds=[0, 0] raised SceneError before
    assert ("world0", 0.03) in rr.frames
    assert {w for w, _ in rr.frames} == {"world0"}


@pytest.mark.gpu
def test_gpu_one_step_logs_the_last_frame_of_each_picked_world(monkeypatch, tmp_path):
    rr = _run(monkeypatch, tmp_path, "cuda:0", worlds=4, seconds=0.01)  # no flush: the copy stayed "issued", 0 frames
    assert rr.frames == {("world0", 0.01), ("world3", 0.01)}
