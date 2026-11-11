# SPDX-License-Identifier: Apache-2.0
"""A live view of an Alloy ``Sim`` through a Newton viewer: a run loop with pause and single steps, world picking and
the contact overlay. The viewer only reads the simulation: stepping through it gives the same state, bit for bit, as
``sim.step`` (the viewer's picking forces are never applied, its reset button is not wired).

Example::

    view = alloy.viewer.launch(sim, backend="viser", worlds=[0, 3])
    print(view.url)
    view.run(seconds=5.0)              # steps while not paused; the browser's Pause / Step buttons work too
    view.close()
"""

from __future__ import annotations

import importlib
import logging
import math
import numbers
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from alloy.api.errors import FeatureUnavailableError, SceneError, SimulationError
from alloy.viewer._backends import BackendOptions, create, spec_for
from alloy.viewer._panel import ALL_WORLDS, Panel
from alloy.viewer._tools import Tools, make_tools

log = logging.getLogger("alloy.viewer")

IDLE_FRAME_S = 1.0 / 60.0      # frame period while paused in a real-time run, so a paused window does not spin a core
CONTACTS_MODULE = "alloy.viewer._contacts"


@dataclass
class ContactOverlay:
    """Which parts of the contact overlay are drawn; off when both are False."""

    points: bool = True
    forces: bool = True

    @property
    def on(self) -> bool:
        return self.points or self.forces


class _Pacer:
    """Sleeps so frames keep pace with the wall clock; a slow frame is not caught up afterwards."""

    def __init__(self) -> None:
        self._next = time.perf_counter()

    def wait(self, period: float) -> None:
        self._next += period
        now = time.perf_counter()
        if self._next > now:
            time.sleep(self._next - now)
        else:
            self._next = now


# --- helpers ------------------------------------------------------------------------------------------------------


def contacts_module() -> ModuleType:
    """The contact overlay module; ``FeatureUnavailableError`` when this install lacks it."""
    module = _load_contacts()
    if module is None:
        raise FeatureUnavailableError("contact reading is not in this install of alloy.viewer", "update Alloy")
    return module


def _load_contacts() -> ModuleType | None:
    try:
        return importlib.import_module(CONTACTS_MODULE)  # loads Warp: only when a frame or a reader needs it
    except ModuleNotFoundError as exc:
        if exc.name != CONTACTS_MODULE:
            raise
        return None


def check_worlds(worlds: Iterable[int] | int | None, count: int) -> list[int] | None:
    """Sorted unique world indices, or None for every world; ``SceneError`` for anything outside ``0..count - 1``."""
    if worlds is None:
        return None
    items = [worlds] if isinstance(worlds, numbers.Integral) else worlds
    try:
        items = list(items)
    except TypeError as exc:
        raise SceneError(f"worlds must be a list of world indices, got {worlds!r}", "pass worlds=[0, 3] or None",
                         field="worlds") from exc
    if not items:
        raise SceneError("worlds is empty", "pass None to show every world, or a list such as [0]", field="worlds")
    bad = [w for w in items if isinstance(w, bool) or not isinstance(w, numbers.Integral) or not 0 <= int(w) < count]
    if bad:
        raise SceneError(f"worlds {bad} are not world indices of this sim (it has worlds 0..{count - 1})",
                         f"pick indices from 0 to {count - 1}, or None for every world", field="worlds")
    return sorted({int(w) for w in items})


def _periods(seconds: float | None, dt: float) -> int | None:
    """Control periods covering ``seconds`` of simulated time (None: no limit)."""
    if seconds is None:
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, numbers.Real) or not math.isfinite(seconds) \
            or seconds < 0:
        raise SimulationError(f"seconds must be a number >= 0 or None, got {seconds!r}", "pass seconds=1.0",
                              field="seconds")
    return math.ceil(float(seconds) / dt - 1e-9)


def _device_scope(device: str) -> Any:
    import warp as wp  # noqa: PLC0415

    return wp.ScopedDevice(device)


def _engine_model(sim: Any) -> Any:
    """The sim's Newton model; ``FeatureUnavailableError`` when the engine lives elsewhere (``processes=``)."""
    try:
        return sim.engine.model
    except (AttributeError, FeatureUnavailableError) as exc:
        raise FeatureUnavailableError("the viewer needs a sim built in this process (this one has no engine here)",
                                      "build it without processes=, e.g. alloy.build(scene, device='cpu')",
                                      field="sim") from exc


def _particle_placer(model: Any, backend: Any) -> Any:
    """For a model with particles (cloth, soft bodies, liquid, sand): the per-world particle placer, after turning
    the viewer's particle spheres on when some particles have no mesh to show them. None for a rigid-only model."""
    if not int(model.particle_count):
        return None
    from alloy.viewer import _particles  # noqa: PLC0415 - loads Warp kernels: only for models with particles

    if _particles.has_loose_particles(model):
        backend.show_particles = True
    return _particles.ParticlePlacer(model)


# --- the session --------------------------------------------------------------------------------------------------


class Viewer:
    """A Newton viewer (``backend``) showing ``sim``. Made by ``alloy.viewer.launch``; also a context manager."""

    def __init__(self, sim: Any, backend: Any, *, name: str, worlds: list[int] | None = None,
                 contacts: bool = True, paused: bool = False, tools: Tools | None = None) -> None:
        self.sim, self.backend, self.name = sim, backend, name
        self._overlay = ContactOverlay(bool(contacts), bool(contacts))
        self._own_paused = False
        self._worlds: list[int] | None = None
        self._frames = 0
        self._closed = False
        self._overlay_missing_logged = False
        with _device_scope(sim.device):
            backend.set_model(sim.engine.model)
            self._placer = _particle_placer(sim.engine.model, backend)
        self._overlay_drawn = False
        self._lock = threading.RLock()      # show_worlds / close from another thread wait for the frame in flight
        self._panel = Panel(sim.worlds, self._overlay.points, self._overlay.forces)
        self._panel.attach(backend)
        self._tools = tools if tools is not None else Tools([])
        self._tools.attach(self)
        if worlds is not None:
            self.show_worlds(worlds)
        self._set_paused(paused)

    # --- facts ---------------------------------------------------------------------------------------------------

    @property
    def paused(self) -> bool:
        """True while stepping is paused (by ``pause()`` or the viewer's own Pause control)."""
        flag = vars(self.backend).get("_paused")
        return self._own_paused if flag is None else bool(flag)

    @property
    def worlds(self) -> list[int] | None:
        """The worlds shown, or None for every world."""
        return None if self._worlds is None else list(self._worlds)

    @property
    def shown_worlds(self) -> list[int]:
        """The worlds shown, as indices (every world when none were picked)."""
        return list(range(self.sim.worlds)) if self._worlds is None else list(self._worlds)

    @property
    def url(self) -> str | None:
        """The browser address of the viser backend; None for the others."""
        return str(self.backend.url) if self.name == "viser" else None

    @property
    def tools(self) -> list[Any]:
        """The interactive tools of this viewer (``launch(joints=True, drag=True)``), in order."""
        return list(self._tools.tools)

    @property
    def frames(self) -> int:
        """Frames drawn so far."""
        return self._frames

    # --- control -------------------------------------------------------------------------------------------------

    def pause(self) -> Viewer:
        """Stop stepping in ``run`` (frames keep drawing); the same flag as the viewer's Pause control."""
        self._set_paused(True)
        return self

    def resume(self) -> Viewer:
        """Step again in ``run``."""
        self._set_paused(False)
        return self

    def step(self, n: int = 1) -> Viewer:
        """Advance ``n`` control periods, then draw; works while paused. Example: ``view.step(10)``."""
        self._check_open()
        if not self._tools:
            self.sim.step(n)
        else:
            for _ in range(int(n)):                  # tools (a drag spring) update their push every period
                self._apply_tools()
                self.sim.step(1)
        self._draw()
        return self

    def sync(self) -> Viewer:
        """Draw the sim's current state (for loops that call ``sim.step`` themselves)."""
        self._check_open()
        self._draw()
        return self

    def run(self, seconds: float | None = None, realtime: bool = True) -> None:
        """Step and draw until the viewer closes, ``seconds`` have passed, or Ctrl-C. Each frame steps one period when
        not paused (or when the viewer's Step was pressed). ``seconds`` is simulated time stepped plus time spent
        paused (each paused frame counts 1/60 s), so a paused ``run(seconds=...)`` still returns; None: no limit.
        ``realtime``: pace frames to the wall clock. Example: ``view.run(seconds=2.0)``."""
        self._check_open()
        limit = _periods(seconds, self.sim.dt)
        pacer = _Pacer() if realtime else None
        stepped = idle = 0
        try:
            while self._keep_running(stepped, idle, limit):
                stepping = self._should_step()
                self._apply_tools()
                if stepping:
                    self.sim.step(1)
                    stepped += 1
                else:
                    idle += 1
                self._draw()
                if pacer is not None:
                    pacer.wait(self.sim.dt if stepping else IDLE_FRAME_S)
        except KeyboardInterrupt:
            log.info("viewer run stopped by Ctrl-C at t=%.4f s after %d periods", self.sim.t, stepped)

    def show_worlds(self, worlds: Iterable[int] | int | None) -> Viewer:
        """Show only these worlds (None: every world). Example: ``view.show_worlds([0, 3])``."""
        self._check_open()
        picked = check_worlds(worlds, self.sim.worlds)
        with self._lock, _device_scope(self.sim.device):
            self.backend.set_visible_worlds(picked)
            self._worlds = picked
        self._panel.show(world=picked[0] if picked is not None and len(picked) == 1 else ALL_WORLDS)
        return self

    def show_contacts(self, points: bool = True, forces: bool = True) -> Viewer:
        """Draw contact points and/or contact forces; both False turns the overlay off."""
        self._overlay = ContactOverlay(bool(points), bool(forces))
        self._panel.show(points=points, forces=forces)
        return self

    def contacts(self, world: int = 0) -> dict[str, Any]:
        """The contacts of one world as data (``alloy.viewer.contacts(sim, world)``)."""
        return contacts_module().contacts(self.sim, world)

    def close(self) -> None:
        """Close the backend (stops a server, writes a file). Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._tools.detach(self)
            self.backend.close()
        log.info("viewer backend %s closed after %d frames", self.name, self._frames)

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"Viewer(backend={self.name!r}, worlds={self._worlds}, paused={self.paused}, frames={self._frames}, "
                f"closed={self._closed})")

    # --- internals -----------------------------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise SimulationError("this viewer is closed", "launch a new one: alloy.viewer.launch(sim, ...)")

    def _set_paused(self, flag: bool) -> None:
        self._own_paused = bool(flag)
        if "_paused" in vars(self.backend):
            self.backend._paused = bool(flag)

    def _should_step(self) -> bool:
        """The backend's answer (it applies its Pause / Step controls); backends without a pause use ours."""
        with self._lock:
            backend_says = bool(self.backend.should_step())
        if "_paused" in vars(self.backend):
            return backend_says
        return backend_says and not self._own_paused

    def _keep_running(self, stepped: int, idle: int, limit: int | None) -> bool:
        """Open, backend alive, and ``seconds`` (periods stepped + paused frames, in sim periods) not used up."""
        if self._closed or not self.backend.is_running():
            return False
        return limit is None or stepped + idle * IDLE_FRAME_S / self.sim.dt < limit

    def _apply_panel(self) -> None:
        """Apply what was asked with the panel's own controls (world shown, overlay parts) since the last frame."""
        asked = self._panel.take()
        if asked is None:
            return
        shown = None if asked.world == ALL_WORLDS else [asked.world]
        if shown != self._worlds:
            self.show_worlds(shown)
        self.show_contacts(asked.points, asked.forces)

    def _apply_tools(self) -> None:
        """Let the opt-in tools make the changes the user asked for in the viewer, before the frame's step."""
        if not self._tools:
            return
        with self._lock:
            self._tools.apply(self)

    def _draw(self) -> None:
        with self._lock, _device_scope(self.sim.device):
            self._apply_panel()
            self.backend.begin_frame(self.sim.t)
            self.backend.log_state(self._state_to_draw())
            self._draw_contacts()
            self._tools.draw(self)
            self.backend.end_frame()
        self._frames += 1

    def _state_to_draw(self) -> Any:
        """The engine's state, with each world's particles placed where the viewer draws that world."""
        state = self.sim.engine.state
        if self._placer is None:
            return state
        return self._placer.state(self.backend, state, self.shown_worlds)

    def _draw_contacts(self) -> None:
        if not self._overlay.on and not self._overlay_drawn:
            return
        module = _load_contacts()
        if module is None:
            if not self._overlay_missing_logged:
                log.warning("contact overlay skipped: this install has no contact reader")
                self._overlay_missing_logged = True
            return
        sets = module.read(self.sim, self.shown_worlds) if self._overlay.on else {}     # off: one clearing draw
        module.draw(self.backend, sets, points=self._overlay.points, forces=self._overlay.forces)
        self._overlay_drawn = self._overlay.on


def launch(sim: Any, backend: str = "viser", worlds: Iterable[int] | int | None = None, contacts: bool = True,
           paused: bool = False, joints: bool = False, drag: bool = False, **options: Any) -> Viewer:
    """Open a viewer on ``sim``. ``backend``: "viser" (browser), "gl" (window), "rerun", "usd" (``path=``,
    ``frames=``) or "null"; ``worlds``: the worlds to show (None: all); ``options`` go to the backend (``port=``,
    ``host=``, ``path=``, ``frames=``, or a keyword of its Newton viewer). Example::

        with alloy.viewer.launch(sim, backend="viser", port=8080) as view:
            view.run(seconds=10.0)
    """
    spec = spec_for(backend)
    _engine_model(sim)
    picked = check_worlds(worlds, sim.worlds)
    opts = BackendOptions.from_kwargs(options, paused=paused)
    if opts.fps is None:
        opts.fps = max(1, round(1.0 / sim.dt))      # usd: one time sample per control period
    with _device_scope(sim.device):
        viewer = create(spec.name, opts)
    try:
        view = Viewer(sim, viewer, name=spec.name, worlds=picked, contacts=contacts, paused=paused,
                      tools=make_tools(bool(joints), bool(drag)))
    except BaseException:
        viewer.close()
        raise
    if view.url:
        log.info("viewer at %s", view.url)
    return view
