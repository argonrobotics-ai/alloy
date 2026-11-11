# SPDX-License-Identifier: Apache-2.0
"""alloy.viewer sessions on the CPU: the viewer never changes the simulation (bitwise, on both CPU routes, contact
overlay on), pause holds time while frames keep drawing and ``step(n)`` advances exactly n periods, the viewer's own
Pause / Step controls, world picking, the contact overlay calls, closing, ``python -m alloy.viewer``, and one real run
of each installed file or browser backend (viser, usd). Scenes are a rigid box on the ground in 2 worlds."""

from __future__ import annotations

import socket
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

import alloy
import alloy.viewer
from alloy.viewer import _session
from alloy.viewer.__main__ import main, parser

DT = 0.01
WORLDS = 2


def _scene() -> alloy.Scene:
    """A box dropping onto the ground, from another height in each world."""
    scene = alloy.Scene(worlds=WORLDS, dt=DT)
    scene.add_ground()
    starts = np.array([[0.0, 0.0, 0.1], [0.1, 0.0, 0.2]])
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=starts, mass=0.2)
    return scene


def _build(route: str = "warp") -> alloy.Sim:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ALLOY_CPU_ROUTE", route)
        return alloy.build(_scene(), device="cpu", seed=0)


def _periods(sim: alloy.Sim) -> int:
    return round(sim.t / sim.dt)


@pytest.fixture(scope="module")
def sim() -> alloy.Sim:
    return _build()


# --- the law: read-only --------------------------------------------------------------------------------------------


@pytest.mark.compile_heavy
@pytest.mark.parametrize("route", ["warp", "mjc"])
def test_viewer_never_changes_the_simulation(route):
    """Stepping through the viewer (step, run, a world pick, the contact overlay on) equals plain sim.step bit for
    bit."""
    plain, viewed = _build(route), _build(route)
    start = np.array(plain.state()["block.pos"], copy=True)
    with alloy.viewer.launch(viewed, backend="null", contacts=True, frames=10_000) as view:
        view.step(3)
        view.run(seconds=0.05, realtime=False)
        view.show_worlds([1])
        view.step(2)
        view.sync()
    plain.step(10)
    a, b = plain.state(), viewed.state()
    assert _periods(viewed) == 10 and a.keys() == b.keys()
    for key in a:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    assert np.all(np.abs(a["block.pos"][:, 2] - start[:, 2]) > 0.01)                 # the boxes did fall


# --- pause and step -------------------------------------------------------------------------------------------------


def test_pause_holds_time_and_step_advances_exactly(sim):
    view = alloy.viewer.launch(sim, backend="null", contacts=False, paused=True, frames=25)
    assert view.paused and view.url is None
    start = _periods(sim)
    view.run(seconds=1.0, realtime=False)            # ends on the null backend's 25-frame budget, never stepping
    assert view.frames == 25 and _periods(sim) == start
    view.step(4)
    assert _periods(sim) == start + 4 and view.frames == 26
    view.close()

    view = alloy.viewer.launch(sim, backend="null", contacts=False, paused=True, frames=100000)
    before = _periods(sim)
    view.run(seconds=0.1, realtime=False)            # paused frames count 1/60 s each: 6 frames, then it returns
    assert _periods(sim) == before and view.frames == 6
    view.close()

    view = alloy.viewer.launch(sim, backend="null", contacts=False, frames=1000)
    view.run(seconds=0.03, realtime=False)
    assert _periods(sim) == start + 7 and view.frames == 3
    view.pause().resume()
    assert not view.paused
    view.close()


def test_viewer_pause_and_step_controls(sim):
    """A backend with its own Pause / Step controls (viser, gl): one flag for the program and the UI."""
    backend = _UIBackend(frames=6)
    view = _session.Viewer(sim, backend, name="ui", contacts=False)
    view.pause()
    assert backend._paused is True
    backend._paused = False                          # the UI's Pause box unticked
    assert not view.paused
    backend._paused, backend._step_requested = True, True
    start = _periods(sim)
    view.run(realtime=False)                         # paused: only the Step press advances
    assert _periods(sim) == start + 1 and backend.frames == 6


def test_ctrl_c_ends_run_cleanly(sim, monkeypatch):
    view = alloy.viewer.launch(sim, backend="null", contacts=False, frames=1000)

    def interrupt() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(view.backend, "end_frame", interrupt)
    view.run(seconds=1.0, realtime=False)
    view.close()


def test_run_arguments_and_closed_viewer(sim):
    view = alloy.viewer.launch(sim, backend="null", contacts=False)
    with pytest.raises(alloy.SimulationError):
        view.run(seconds=-1.0)
    view.close()
    view.close()
    with pytest.raises(alloy.SimulationError):
        view.step(1)


# --- worlds and contacts ----------------------------------------------------------------------------------------------


def test_world_pick(sim):
    for bad in ([2], [-1], [], [True], ["0"], 1.5):
        with pytest.raises(alloy.SceneError) as err:
            alloy.viewer.launch(sim, backend="null", worlds=bad)
        assert err.value.field == "worlds"
    view = alloy.viewer.launch(sim, backend="null", worlds=[1, 1], contacts=False)
    assert view.worlds == [1] and view.backend._visible_worlds == {1}
    assert view.show_worlds(0).worlds == [0]
    assert view.show_worlds(None).worlds is None and view.backend._visible_worlds is None
    assert view.backend.model.world_count == WORLDS
    view.close()


def test_contact_overlay_calls(sim, monkeypatch):
    fake = _FakeContacts()
    monkeypatch.setitem(__import__("sys").modules, _session.CONTACTS_MODULE, fake)
    backend = _UIBackend(frames=100)
    view = _session.Viewer(sim, backend, name="ui", worlds=[1])
    view.sync()
    assert fake.calls == [("read", [1]), ("draw", True, True)]
    assert backend.log == ["begin", "state", "contacts", "end"]
    view.show_contacts(points=True, forces=False).sync()
    assert fake.calls[-1] == ("draw", True, False)
    view.show_contacts(points=False, forces=False).sync()
    assert fake.calls[-1] == ("draw", False, False) and len(fake.calls) == 5      # one clearing draw, no read
    view.sync()
    assert len(fake.calls) == 5                                                   # then nothing while off
    assert alloy.viewer.contacts(sim, world=1) == {"world": 1}
    assert view.contacts(0) == {"world": 0}


def test_contacts_reader_missing(sim, monkeypatch):
    monkeypatch.setattr(_session, "_load_contacts", lambda: None)
    with pytest.raises(alloy.FeatureUnavailableError):
        alloy.viewer.contacts(sim, world=0)
    view = alloy.viewer.launch(sim, backend="null", frames=10)
    view.sync()                                      # overlay skipped with a warning, frame still drawn
    assert view.frames == 1
    view.close()


# --- command line ---------------------------------------------------------------------------------------------------


def test_command_line_arguments():
    args = parser().parse_args(["demo_peg_d12", "--backend", "null", "--worlds", "4", "--show", "0,3", "--seconds",
                                "0.5", "--port", "8771", "--device", "cpu", "--paused", "--no-contacts"])
    assert (args.scene, args.backend, args.worlds, args.show) == ("demo_peg_d12", "null", 4, [0, 3])
    assert (args.seconds, args.port, args.device, args.paused, args.no_contacts) == (0.5, 8771, "cpu", True, True)
    assert parser().parse_args(["x", "--show", "all"]).show is None
    for bad in (["x", "--show", "a,b"], ["x", "--backend", "vtk"]):
        with pytest.raises(SystemExit):
            parser().parse_args(bad)


def test_command_line_run(tmp_path, capsys):
    path = tmp_path / "scene.json"
    path.write_text(_scene().to_json(), encoding="utf-8")
    assert main([str(path), "--backend", "null", "--seconds", "0.02", "--device", "cpu", "--show", "1"]) == 0
    assert main([str(path), "--backend", "null", "--device", "cpu", "--show", "5"]) == 2
    assert "worlds [5]" in capsys.readouterr().err


# --- real backends (optional packages) --------------------------------------------------------------------------------


def test_viser_serves_the_scene(sim):
    pytest.importorskip("viser")
    if not alloy.viewer.backends()["viser"]["installed"]:
        pytest.skip("viser is older than 1.1.1")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with alloy.viewer.launch(sim, backend="viser", host="127.0.0.1", port=port, verbose=False) as view:
        assert view.url.endswith(f":{port}")
        view.step(2)
        with urllib.request.urlopen(view.url, timeout=10) as page:
            assert page.status == 200


def test_usd_writes_the_run(sim, tmp_path):
    pytest.importorskip("pxr")
    out = tmp_path / "run.usda"
    with alloy.viewer.launch(sim, backend="usd", path=str(out), frames=5) as view:
        view.run(seconds=1.0, realtime=False)        # ends on the 5-frame budget
        assert view.frames == 5
    assert out.stat().st_size > 0


# --- doubles ----------------------------------------------------------------------------------------------------------


class _UIBackend:
    """A viewer with Pause / Step controls like viser's and gl's, recording the frame calls."""

    def __init__(self, frames: int) -> None:
        self._paused, self._step_requested = False, False
        self.budget, self.frames, self.log = frames, 0, []

    def set_model(self, model) -> None:
        self.model = model

    def set_visible_worlds(self, worlds) -> None:
        self.visible = worlds

    def should_step(self) -> bool:
        if not self._paused:
            return True
        pressed, self._step_requested = self._step_requested, False
        return pressed

    def is_running(self) -> bool:
        return self.frames < self.budget

    def begin_frame(self, t: float) -> None:
        self.log.append("begin")

    def log_state(self, state) -> None:
        self.log.append("state")

    def end_frame(self) -> None:
        self.log.append("end")
        self.frames += 1

    def close(self) -> None:
        pass


class _FakeContacts(SimpleNamespace):
    """Stands in for alloy.viewer._contacts, recording calls."""

    def __init__(self) -> None:
        super().__init__(calls=[])

    def read(self, sim, worlds):
        self.calls.append(("read", list(worlds)))
        return {w: object() for w in worlds}

    def draw(self, backend, sets, *, points, forces):
        backend.log.append("contacts")
        self.calls.append(("draw", points, forces))

    def contacts(self, sim, world=0):
        return {"world": world}


def test_gl_off_screen_draws_contacts():
    """The desktop viewer, drawing off screen (no display needed), takes the contact overlay and gives a picture."""
    pytest.importorskip("pyglet")
    pytest.importorskip("imgui_bundle")
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.026), mass=0.2)
    sim = alloy.build(scene, device="cpu", seed=0)
    try:
        view = alloy.viewer.launch(sim, backend="gl", headless=True, width=160, height=120)
    except Exception as exc:  # no EGL / OpenGL on this machine
        pytest.skip(f"no off-screen OpenGL here: {exc}")
    with view:
        view.step(20)
        assert len(view.contacts(world=0)["depth"]) > 0
        frame = view.backend.get_frame().numpy()
        view.show_contacts(points=False, forces=False).sync()
    assert frame.shape == (120, 160, 3) and frame.std() > 0


class _Widgets:
    """Stands in for a viewer's widget set: returns queued (changed, value) answers per label."""

    def __init__(self, answers: dict[str, object]):
        self.answers = answers

    def slider_int(self, label, value, lo, hi):
        return (label in self.answers, self.answers.get(label, value))

    def checkbox(self, label, value):
        return (label in self.answers, self.answers.get(label, value))


def test_panel_controls_pick_a_world_and_clear_the_overlay():
    """The panel's own controls (browser, desktop window): a world pick and the overlay boxes apply on the next
    frame; switching the overlay off draws one clearing frame."""
    from alloy.viewer import _contact_draw, _panel

    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.026), mass=0.2)
    sim = alloy.build(scene, device="cpu", seed=0)
    logged: dict[str, object] = {}
    with alloy.viewer.launch(sim, backend="null", frames=1000) as view:
        view.backend.log_points = lambda name, points, *a, **k: logged.__setitem__("points", points)
        view.backend.log_arrows = lambda name, starts, *a, **k: logged.__setitem__("arrows", starts)
        view.step(20)
        assert logged["points"] is not None and logged["arrows"] is not None
        view._panel.render(_Widgets({_panel.WORLD_LABEL: 2}))
        view.sync()
        assert view.worlds == [2]
        view._panel.render(_Widgets({}))                        # nothing touched: nothing changes
        view.sync()
        assert view.worlds == [2]
        view._panel.render(_Widgets({_panel.POINTS_LABEL: False, _panel.FORCES_LABEL: False}))
        view.sync()
        assert logged["points"] is None and logged["arrows"] is None     # cleared, not left on screen
        view._panel.render(_Widgets({_panel.WORLD_LABEL: _panel.ALL_WORLDS}))
        view.sync()
        assert view.worlds is None
    assert _contact_draw.POINTS


def test_world_pick_from_another_thread_while_running():
    """``show_worlds`` called from a second thread while ``run`` draws (the browser viewer rebuilt its shape buffers
    mid-frame and the run loop died)."""
    pytest.importorskip("viser")
    import threading

    scene = alloy.Scene(worlds=4, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.026), mass=0.2)
    sim = alloy.build(scene, device="cpu", seed=0)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with alloy.viewer.launch(sim, backend="viser", port=port, verbose=False) as view:
        stop = threading.Event()

        def flip() -> None:
            picks = ([1], None, [0, 2], None)
            i = 0
            while not stop.is_set():
                view.show_worlds(picks[i % len(picks)])
                i += 1
                stop.wait(0.002)

        thread = threading.Thread(target=flip, daemon=True)
        thread.start()
        try:
            view.run(seconds=0.5, realtime=False)
        finally:
            stop.set()
            thread.join(timeout=10)
        assert sim.t == pytest.approx(0.5)
