# SPDX-License-Identifier: Apache-2.0
"""GEO P6 (2026-11-11): a 9x9 towel ran about 20x slower than real time on a Mac CPU and nothing said to expect it.
A build that found no GPU now writes one line when the scene holds cloth, liquid, sand or paste, or soft bodies; not
for rigid or robot scenes (about real time on a CPU), not when device="cpu" was asked for, and once per process."""

from __future__ import annotations

import logging

import pytest

from alloy.api import _cpu_note


class _Sim:
    def __init__(self, routes: list[str], device: str = "cpu") -> None:
        self.device = device
        self._routes = {f"o{i}": {"route": r} for i, r in enumerate(routes)}

    def routes(self) -> dict[str, dict[str, str]]:
        return self._routes


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(_cpu_note, "_noted", [False])


def test_one_line_for_cloth_liquid_sand_or_soft_bodies_on_the_cpu(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        assert _cpu_note.cpu_note(_Sim(["mjc", "cloth_vbd", "liquid_sph", "paste_mpm", "soft_vbd"]), None)
        assert not _cpu_note.cpu_note(_Sim(["cloth_vbd"]), None)             # once per process
    assert len(caplog.records) == 1
    text = caplog.records[0].getMessage()
    assert text.startswith("no NVIDIA GPU found") and "alloy.estimate" in text
    assert "cloth, liquid, sand or paste and soft bodies run many times slower" in text


def test_the_line_reaches_stderr_when_no_handler_shows_it(capsys, monkeypatch) -> None:
    from alloy.api import _say  # noqa: PLC0415

    monkeypatch.setattr(_say, "_shown", lambda logger: False)
    assert _cpu_note.cpu_note(_Sim(["cloth_vbd"]), None)
    err = capsys.readouterr().err
    assert err.startswith("alloy: no NVIDIA GPU found") and "where scenes with cloth run" in err
    assert err.count("\n") == 1


@pytest.mark.parametrize(("routes", "device", "sim_device"), [
    (["mjc", "rigid", "robot"], None, "cpu"),          # rigid and robot scenes run about real time on a CPU
    (["cloth_vbd"], "cpu", "cpu"),                     # asked for the CPU: no line
    (["cloth_vbd"], None, "cuda:0"),                   # a GPU was found
    (["rod_vbd"], None, "cpu"),                        # not on physics' list
])
def test_no_line(caplog, routes, device, sim_device) -> None:
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        assert not _cpu_note.cpu_note(_Sim(routes, sim_device), device)
    assert not caplog.records


def test_a_hint_never_stops_a_build() -> None:
    class Broken(_Sim):
        def routes(self):
            raise RuntimeError("no report")
    assert not _cpu_note.cpu_note(Broken([]), None)


def test_build_ends_with_the_note_and_returns_the_sim() -> None:
    sim = _Sim(["cloth_vbd"])
    assert _cpu_note.after_build(sim, "cpu") is sim and not _cpu_note._noted[0]
    assert _cpu_note.after_build(sim, None) is sim and _cpu_note._noted[0]
