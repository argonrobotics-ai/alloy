# SPDX-License-Identifier: Apache-2.0
"""``alloy.Sim``: a built scene you can step, read, check, render and reset. Make one with ``alloy.build``.

One ``step()`` advances one control period of ``sim.dt`` seconds (the engine takes its own substeps inside; physics
steps so far per world: ``sim.physics_steps``). All arrays are host NumPy copies with a leading world axis, unless
``world=`` picks one world. Quaternions are named by their order: ``quat_xyzw`` (engine order) and ``quat_wxyz``.

Example::

    sim = alloy.build(scene, worlds=4)
    sim.set_targets({"gripper/finger_left": 0.01})     # joint position targets (MJCF <position> actuators)
    traj = sim.run(2.0, record=["cube"])              # (frames, worlds, 3) positions etc.
    print(sim.check()["ok"], sim.state(world=0)["cube.pos"])
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from alloy.api import _heat, _liquid, _liquid_accuracy, _robot_random_apply, _robot_random_push, _say
from alloy.api import health as H
from alloy.api._state import StateIO
from alloy.api.errors import (
    FeatureUnavailableError, SimulationError, forget_raised, raise_if_invalid, reading, wrapped,
)
from alloy.api.names import NameIndex
from alloy.api.trajectory import Trajectory

log = logging.getLogger("alloy.api")
RIGID_KINDS = ("box", "sphere", "cylinder", "capsule", "mesh")


def route_rows(report: Any) -> dict[str, dict[str, Any]]:
    """Per object of a build report: route, solver and the reasons, the rows of ``Sim.routes()`` and
    ``alloy.estimate(...)["routes"]``. Example: ``route_rows(sim.built.report)["cable"]["route"]``."""
    return {name: {"route": str(d.route), "solver": d.entry or "none (static)", "reasons": list(d.reasons)}
            for name, d in report.decisions.items()}


def _add_solver_notes(rows: dict[str, dict[str, Any]], stepper: Any) -> None:
    """Each entry solver's ``alloy_route_notes()`` ({key: text}) into the rows of the objects routed to it; a note
    never replaces a key the row already has (route, solver, reasons)."""
    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415

    for name, solver in (entry_solvers(stepper) if stepper is not None else {}).items():
        notes = getattr(solver, "alloy_route_notes", None)
        extra = notes() if callable(notes) else None
        if not extra:
            continue
        for r in rows.values():
            if r["solver"] == name:
                r.update({k: v for k, v in extra.items() if k not in r})


class Sim:
    """Wraps the engine simulation (``sim.engine``) and the built scene (``sim.built``) behind names."""

    def __init__(self, engine: Any, *, device: str, profile: str, workdir: str | None = None,
                 scene: Any = None) -> None:
        self.engine, self.built = engine, engine.built
        self.device, self.profile, self.scene = device, profile, scene
        # engine periods per step(): more than 1 when a precise fixture made the cards' period finer (fixtures.py)
        self._sub = max(1, round(float(scene.dt) / float(engine.built.control_dt))) if scene is not None else 1
        self.index = NameIndex.from_built(self.built)
        self._io = StateIO(engine, self.index)
        self._workdir = workdir
        self._warned_invalid: set[int] = set()
        self._settled = False                    # settle() ran: reset() returns to that state
        self._pushes: Any = None                 # the scene's timed pushes (_robot_random_push.attach)
        self._report_liquids()
        self._report_soft_bodies()
        H.warn_downgraded(engine, profile)       # precision on a device that cannot repeat runs: say so once

    # --- facts ---------------------------------------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Control period in seconds (one ``step()``). Example: ``sim.dt == 0.01``."""
        return float(self.built.control_dt) * self._sub

    @property
    def t(self) -> float:
        """Simulated time in seconds. Example: ``sim.t``."""
        return float(self.engine.clock.t)

    @property
    def worlds(self) -> int:
        """Number of parallel worlds. Example: ``sim.worlds``."""
        return int(self.built.world_count)

    @property
    def physics_steps(self) -> int:
        """Physics steps taken so far, per world (for world-steps/s: ``worlds * physics_steps / wall_s``)."""
        return int(self.engine.steps)

    def names(self) -> dict[str, list[str]]:
        """Every name by kind: bodies, joints, particles (cloth / liquid), static objects and actuators.

        Example: ``{"bodies": ["cube"], "joints": [], "particles": [], "static": ["ground"], "actuators": []}``."""
        return {**self.index.names(), "actuators": list(self._io.acts.names)}

    # --- stepping ------------------------------------------------------------------------------------------------

    def step(self, n: int = 1) -> Sim:
        """Advance ``n`` control periods (``n * sim.dt`` seconds). Example: ``sim.step(100)``."""
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise SimulationError(f"step(n) needs an int >= 0, got {n!r}", "pass n=1 (control periods)", field="n")
        with wrapped("step", SimulationError), _device(self.device):
            for _ in range(int(n) * self._sub):
                if self._pushes is not None:
                    self._pushes.tick()            # scene.add_push: start and end the pushes that are due
                self.engine.step_period()
                raise_if_invalid(self.engine)      # profile "precision": raise in the period a world fails
        return self

    def run(self, seconds: float, record: Iterable[str] | str | None = None, every: int = 1, *,
            targets: Mapping[str, Any] | None = None) -> Trajectory:
        """Step ``seconds`` of simulated time, recording state every ``every`` periods (frame 0 = before stepping).

        - ``record``: names to record (default: every body, joint and particle object); [] records only time;
        - ``targets``: joint position targets during the run: ``{"joint": value}`` holds a value, and
          ``{"joint": (start, end)}`` ramps linearly over the run (a smooth move instead of a jump; a large jump
          makes a stiff servo kick hard enough to shake a grasped object loose).

        Returns an ``alloy.Trajectory``; its ``meta`` has wall time, physics steps, a health summary and
        ``mpm_grid`` (the grid each MPM solver used: "sparse" or "dense", and any fallback to sparse; [] without MPM).
        Example: ``traj = sim.run(1.0, record=["cube"], targets={"lift": (0.0, 0.12)})``."""
        periods = int(round(float(seconds) / self.dt))
        if periods < 0 or every < 1:
            raise SimulationError(f"run({seconds!r}, every={every!r}) is not a forward run",
                                  "pass seconds >= 0 and every >= 1")
        names = self._io.expand(record) if record != [] else []
        ramps = _ramps(targets or {})
        frames, times = [self._io.read(names)], [self.t]
        steps0, wall0 = self.physics_steps, time.perf_counter()
        for k in range(1, periods + 1):
            if ramps:
                self._io.set_targets({n: a + (b - a) * k / periods for n, (a, b) in ramps.items()}, "position")
            self.step(1)
            if k % every == 0 or k == periods:
                frames.append(self._io.read(names))
                times.append(self.t)
        wall = time.perf_counter() - wall0
        data = {"t": np.asarray(times)}
        for key in frames[0]:
            data[key] = np.stack([f[key] for f in frames])
        meta = {"dt": self.dt, "worlds": self.worlds, "periods": periods, "wall_s": wall,
                "physics_steps": self.physics_steps - steps0, "health": self.health(),
                "mpm_grid": _liquid.mpm_grids(self.engine)}            # at the end: a dense grid may fall back
        if self.scene is not None and _liquid_accuracy.has_liquid(self.scene.items):    # the run's liquid setting
            meta["liquid_accuracy"] = self.scene.liquid_accuracy
        self._warn_invalid(meta["health"])
        return Trajectory(data, meta)

    def settle(self, seconds: float) -> Sim:
        """Run ``seconds`` with no new commands, then make the settled state the start that ``reset()`` returns to
        (piles, drapes and soft parts start a few mm off rest), also in a scene with ``scene.randomize`` (each world
        settles from its own draws). Example: ``sim.settle(0.5)``."""
        self.step(int(round(float(seconds) / self.dt)))
        self.engine._initial.save(self.engine.state)
        self._settled = True
        self._rebase()
        return self

    def _rebase(self) -> None:
        """The engine's current state is the start: its latch snapshot and health references follow it."""
        eng = self.engine
        latch = getattr(eng, "_latch_snap", None)
        if latch is not None:
            latch.save(eng.state)
        if eng.health is not None and hasattr(eng.health, "reset_reference"):
            eng.health.reset_reference(eng.state, None)
        if hasattr(eng.health, "forget_seen"):      # flags raised while settling are not this run's (seen_flags)
            eng.health.forget_seen()
        getattr(eng.stepper, "mark_start", lambda: None)()   # the coupled exchange clock too (coupling.clock_memory)

    def set_pose(self, name: str, pos: Any = None, quat_wxyz: Any = None, *, world: int | None = None) -> Sim:
        """Teleport a free or kinematic body now (zero velocity): ``pos`` (3,) or (worlds, 3), ``quat_wxyz`` (4,) or
        (worlds, 4). A robot with a free base (MJCF ``<freejoint/>``) moves with all its links, joint angles kept.
        ``pos`` is what ``"<name>.pos"`` reads: a library object placed by its anchor gets that anchor there (and
        turns about it when only ``quat_wxyz`` is given). It removes the object's ``drive``, which would write over
        the pose at the next step: an ended one, or a running one when every world is teleported (one world under a
        running drive raises ``SimulationError``).

        Example: ``sim.set_pose("tray", pos=(0, 0, 0.3))``."""
        from alloy.api import _drive, _robots  # noqa: PLC0415

        if name in self.index.robot_joints:
            _robots.set_pose(self, name, pos, quat_wxyz, world)
        else:
            _drive.release(self, name, world)
            _drive.set_pose(self, name, pos, quat_wxyz, world)
        _drive.rebase_energy(self, world)
        return self

    def drive(self, name: str, *, velocity: Any = (0.0, 0.0, 0.0), angular_velocity: Any = (0.0, 0.0, 0.0),
              pivot: Any = None, duration: float = float("inf")) -> Sim:
        """Move a kinematic object (or a kinematic fixture's parts together) from now on with a constant linear
        ``velocity`` (m/s) and ``angular_velocity`` (rad/s, world axes) about ``pivot`` (a world point; default its
        current ``.pos``, or the fixture's centre) for ``duration`` s, then hold. Contacts and liquids see the moving
        walls every physics step. A new drive of an object replaces its earlier one (it starts from where the object
        is now). Example (tilt a cup about its lip at 40 deg/s for 2 s)::

            sim.drive("cup", angular_velocity=(0, math.radians(40), 0), pivot=(0.044, 0, 0.104), duration=2.0)"""
        from alloy.api import _drive  # noqa: PLC0415

        parts = _drive.parts_of(self, name)
        pivot = _drive.fixture_centre(self, name, parts) if pivot is None else pivot   # a fixture: its parts' centre
        _drive.replace_drives(self.engine, [_drive.make_drive(self, part, velocity, angular_velocity, pivot, duration)
                                            for part in parts])
        return self

    def stop_drives(self) -> Sim:
        """Remove every ``drive``; kinematic objects hold where they are, at rest. Example: ``sim.stop_drives()``."""
        from alloy.api import _drive  # noqa: PLC0415

        _drive.stop_all(self)
        return self

    def randomized(self) -> dict[str, np.ndarray]:
        """What ``Scene.randomize`` drew for robots and gravity: ``{"<target>.<field>": (worlds,) values}``, as each
        world holds them now. Example: ``sim.randomized()["go2.mass_scale"]``."""
        return _robot_random_apply.values(self.built)

    def pushes(self) -> dict[str, np.ndarray]:
        """The ``scene.add_push`` force held on each pushed body now: ``{"go2/base": (worlds, 3) N}`` (zeros between
        pushes). Example: ``sim.pushes()["go2/base"]``."""
        return _robot_random_push.forces(self)

    def reset(self) -> Sim:
        """Back to the start state of every world (randomized values are redrawn the same way, except robot and
        gravity rows added with ``when="reset"``, which get a new draw; drives and ``apply_wrench`` loads are removed).

        Example: ``sim.reset()``."""
        self.stop_drives()
        with wrapped("reset", SimulationError), _device(self.device):
            self.engine.reset()
            forget_raised(self.engine)             # a reset world may fail, and raise, again
            if self._settled and getattr(self.built, "randomized", None) is not None:
                # the settled start, with each world's draws (AUDIT-STACK API-1); since TRAIN-FIX 3a the engine's own
                # restore keeps it too (its redraw no longer re-runs FK from the build pose), so this repeats it
                self.engine._initial.restore(self.engine.state, None)
                _robot_random_apply.S.place(self.engine)     # start rows: the restore put the robots back
                self._rebase()
            self._io.clear_wrenches()
            self._io.match_states()                # both of the engine's states: a kinematic row (API-16 class)
        return self

    # --- state and control ---------------------------------------------------------------------------------------

    def state(self, names: Iterable[str] | str | None = None, *, world: int | None = None) -> dict[str, Any]:
        """Current state: ``{"t": s, "<body>.pos": (W, 3), "<body>.quat_wxyz": ..., "<joint>.q": (W,), ...}``.
        ``"<body>.pos"`` is the body origin, except for a library object placed by its anchor
        (``Scene.add_object(..., anchor=)``): there it is that anchor point, where you put it, and
        ``"<body>.body_pos"`` is the body origin.

        ``world``: one world's values without the world axis. Example: ``sim.state(world=0)["cube.pos"]``."""
        out: dict[str, Any] = {"t": self.t}
        out.update(self._io.read(self._io.expand(names)))
        if world is not None:
            self._check_world(world)
            out = {k: (v if k == "t" else v[world]) for k, v in out.items()}
        return out

    def set_targets(self, targets: Mapping[str, Any], kind: str = "position") -> Sim:
        """Joint targets for actuated joints: ``{"joint": value}`` (all worlds) or ``{"joint": (worlds,) array}``.

        ``kind``: "position" (MJCF ``<position>`` actuators, metres or radians) or "velocity". Kept until changed.
        A position target outside the joint's range is held at the limit, each world's target on its own (also the
        ramps of ``run(targets=)``; the first one warns). A servo whose file gives it a ``ctrlrange`` wider than the
        joint's range keeps that one (a gripper that squeezes by aiming past its closed stop). A joint that no
        actuator drives raises ``SimulationError``.
        Example: ``sim.set_targets({"gripper/lift": 0.15})``."""
        if kind not in ("position", "velocity"):
            raise SimulationError(f"kind must be 'position' or 'velocity', got {kind!r}", "pass kind='position'")
        self._io.set_targets(targets, kind)
        return self

    def apply_wrench(self, name: str, force: Any = (0.0, 0.0, 0.0), torque: Any = (0.0, 0.0, 0.0), *,
                     at: str | None = None) -> Sim:
        """Hold a force (N) and a torque (N m) on a body, in world coordinates about its centre of mass, until changed;
        zeros remove it. ``at``: "start" or "end" picks a rod's end segment. A (worlds, 3) array sets one per world.
        Example: ``sim.apply_wrench("cable", torque=(0.002, 0, 0), at="end")``."""
        self._io.apply_wrench(name, force, torque, at)
        return self

    def set_controls(self, controls: Mapping[str, Any]) -> Sim:
        """Raw MuJoCo ``ctrl`` values by actuator name (``sim.names()["actuators"]``): motors take a force or torque,
        servos a position or velocity (an MJCF ``<position>`` actuator: the joint position it holds, not clamped to
        the joint's range). Kept until changed. Example: ``sim.set_controls({"arm/actuator1": 0.5})``."""
        self._io.set_controls(controls)
        return self

    def faces(self, name: str) -> np.ndarray:
        """Triangles (F, 3) of a cloth, as indices into its ``"<name>.points"``. Example: ``sim.faces("towel")``."""
        r = self.index.particle_range(name)
        tris = self.index.tris.get(name)
        if tris is None:
            raise SimulationError(f"{name!r} has no triangles (it is not a cloth)", "call faces() on a cloth object")
        tri = self.built.model.tri_indices.numpy()[tris.start:tris.stop]
        return (tri - r.start).astype(np.int64)

    def particle_volume(self, name: str) -> float:
        """Rest volume (m^3) one particle of a liquid stands for: (2 r)^3 for particle radius r, so
        ``particles x volume`` is the liquid volume the engine simulates (see ``liquid_volume``).

        Example: ``sim.particle_volume("water") * len(sim.state(world=0)["water.points"])``."""
        self.index.particle_range(name)
        radius = self.built.handles[name].decision.numbers.get("particle_radius")
        if not radius:
            raise SimulationError(f"{name!r} has no particle radius (not a liquid)", "call it on a Scene.add_liquid "
                                  "object", field=name)
        return float((2.0 * radius) ** 3)

    def liquid_volume(self, name: str) -> float:
        """Volume (m^3) of a liquid as simulated: whole particles fill the shape you asked for, so this can be a few
        percent below the box volume; enlarge the box if you need an exact amount.

        Example: ``sim.liquid_volume("water") * 1e6`` -> millilitres."""
        return len(self.index.particle_range(name)) * self.particle_volume(name)

    # --- heat ------------------------------------------------------------------------------------------------------

    def temperature(self, name: str) -> np.ndarray:
        """Temperatures (K) of a heated object's particles, ``(worlds, particles)`` in the order of
        ``"<name>.points"`` (also in ``sim.state()`` and recordings as ``"<name>.temperature"``).

        Example: ``sim.temperature("water").mean() - 273.15`` -> mean °C."""
        with wrapped("temperature", SimulationError):
            return _heat.temperature(self.engine, name)

    def heat(self, name: str) -> dict[str, np.ndarray]:
        """Per world, over the particles that take heat: ``mean_K``, ``min_K``, ``max_K``, ``melted`` (mean liquid
        share, 0-1), ``done`` (mean doneness of food, 0-1), ``evaporated`` (share of the mass boiled off) and
        ``particles``. Not-yet-poured and boiled-off particles are left out.

        Example: ``sim.heat("water")["mean_K"][0]``."""
        with wrapped("heat", SimulationError):
            return _heat.summary(self.engine, name)

    def set_heater(self, name: str, *, temperature: Any = None, power: Any = None) -> Sim:
        """Change a heater while running: ``temperature`` K or ``power`` W (as it was built), one value or one per
        world. Example: ``sim.set_heater("pan_floor", temperature=373.15)``."""
        with wrapped("set_heater", SimulationError):
            _heat.set_heater(self.engine, name, temperature, power)
        return self

    @property
    def plugins(self) -> Any:
        """The plug-ins running in this simulation (heat, and ``alloy.build(plugins=[...])``): ``sim.plugins.read()``
        gives each one's read-outs by label, ``sim.plugins["<label>"]`` the plug-in; None without plug-ins.

        Example: ``sim.plugins.read()["particle_thermal"]["heater_area_m2"]``."""
        return getattr(self.engine, "plugins", None)

    @plugins.setter
    def plugins(self, host: Any) -> None:
        """For code that installs a plug-in host itself (e.g. ``alloy.plugins.species.build``): the host it made."""
        self.engine.plugins = host

    def routes(self) -> dict[str, dict[str, Any]]:
        """Where each object went: route, solver and the reasons. Example: ``sim.routes()["towel"]["route"]``.
        With the opt-in contact gate on (``ClockPolicy.contact_gate``), each object in a gated coupled pair also has
        ``"exchange_clock"``: what the gate does and its known gap. A solver may add notes to its objects' rows
        (``alloy_route_notes()``, e.g. a film's ``"contact_skin"``)."""
        out = route_rows(self.built.report)
        _add_solver_notes(out, getattr(self.engine, "stepper", None))
        mode = _clock_mode(self.engine)
        if mode:
            for r in out.values():
                if r["solver"] in mode["gated_entries"]:
                    r["exchange_clock"] = f"{mode['exchange_clock']}; known gap: {mode['known_gap']}"
        return out

    # --- checks, render, report ----------------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Per-world health: severity OK / DEGRADED / INVALID, flags, a fix for each problem,
        ``retired_particles`` (liquid particles per world that left the scene; None without a liquid) and
        ``refined_periods`` (periods re-run at a finer coupling step for that world since the build; None without
        coupling); a world whose cloth stretched over 50 % past its rest size has ``cloth_stretch``
        (``{"percent", "cloth", "level": "warn" | "fail"}``, the run's peak; fail is over 500 %, INVALID).
        JSON-ready. Example: ``h = sim.health(); h["invalid_worlds"]``."""
        h = H.add_retired(H.summary(H.poll(self.engine), self.worlds), H.retired(self.engine))
        return H.add_refines(h, H.refines(self.engine))

    def check(self) -> dict[str, Any]:
        """Physics checks on the current run: ``{"ok": bool, "problems": [...], ...}``. Uses ``alloy.check`` when
        it is installed, else the engine's health monitor. Example: ``assert sim.check()["ok"]``."""
        return H.check(self)

    def render(self, camera: Any = None, **kwargs: Any) -> Any:
        """An image of world 0 (or ``world=``) from ``camera`` via ``alloy.camera``. Example: ``img = sim.render()``."""
        try:
            from alloy import camera as cam  # noqa: PLC0415 - optional package
        except ImportError as exc:
            raise FeatureUnavailableError("rendering needs the alloy.camera package, which this install lacks",
                                          "update Alloy to a version that ships alloy.camera") from exc
        fn = getattr(cam, "render", None)
        if fn is None:
            raise FeatureUnavailableError("alloy.camera has no render() yet", "update Alloy")
        return fn(self, camera, **kwargs)

    def report(self) -> dict[str, Any]:
        """The engine's report (steps, time, coupling, health objects). Example: ``sim.report()["steps"]``."""
        with reading(self.engine):                 # profile "precision": reading must not raise
            return dict(self.engine.report())

    def close(self) -> None:
        """Free the scene's temporary card directory (and the ``rerun_worlds`` simulation). Example: ``sim.close()``."""
        fine = getattr(getattr(self, "engine", None), "fine_rerun", None)
        if fine is not None:
            fine.close()
        if self._workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def __repr__(self) -> str:
        return f"Sim(worlds={self.worlds}, dt={self.dt}, device={self.device!r}, t={self.t:.4g})"

    # --- internals -----------------------------------------------------------------------------------------------

    def _check_world(self, world: int) -> None:
        if not 0 <= int(world) < self.worlds:
            raise SimulationError(f"world {world} is outside 0..{self.worlds - 1}", "pass world=0 for the first")

    def _report_soft_bodies(self) -> None:
        items = getattr(self.scene, "items", {}) if self.scene is not None else {}
        for name, item in items.items():
            if item.get("kind") not in RIGID_KINDS or item.get("static") or "route" in item.get("card_extra", {}):
                continue
            d = self.built.report.decisions.get(name)
            if d is not None and str(d.route) not in ("rigid", "rigid_compliant", "rigid_pile", "static"):
                _say.warn("%r became a %s body (material %r is soft at this size): much slower than a rigid body; "
                            "pass rigid=True to add_%s for a rigid part", name, d.route, item.get("material"),
                            item["kind"])

    def _report_liquids(self) -> None:
        from alloy.api._liquid import unseen_kinematic  # noqa: PLC0415

        if unseen_kinematic(self.engine):
            _say.warn("the liquid cannot touch kinematic objects in this scene: no solver owns them here, the "
                      "liquid sees only static walls, and it passes through a kinematic container; hold the "
                      "container still (static=True)")
        items = getattr(self.scene, "items", {}) if self.scene is not None else {}
        for name in self.index.particles:
            item = items.get(name, {})
            asked = (item.get("pour") or {}).get("volume") or item.get("volume") if item.get("kind") == "liquid" else None
            if not asked or not self.built.handles[name].decision.numbers.get("particle_radius"):
                continue                       # a paste (add_liquid of mayonnaise) routes to MPM: no SPH particles
            got = self.liquid_volume(name)
            log.info("%s: %d particles, %.4g ml simulated of the %.4g ml asked for", name,
                     len(self.index.particle_range(name)), got * 1e6, asked * 1e6)
            if abs(got / asked - 1.0) > 0.03:
                grow = (asked / got) ** (1.0 / 3.0)
                _say.warn("%s: the liquid is %.4g ml, not the %.4g ml you asked for (whole particles of %.3g mm "
                            "fill it); scale the sizes by about %.3f to get the full volume",
                            name, got * 1e6, asked * 1e6, 1e3 * self.particle_volume(name) ** (1 / 3), grow)

    def _warn_invalid(self, health: dict[str, Any]) -> None:
        new = [w for w in health.get("invalid_worlds", []) if w not in self._warned_invalid]
        if new:
            self._warned_invalid.update(new)
            _say.warn("%d world(s) became INVALID and were frozen (e.g. world %d: %s); see sim.health()",
                        len(new), new[0], health["worlds"][new[0]]["problem"])


def _clock_mode(engine: Any) -> dict[str, Any] | None:
    """The coupler's exchange-clock mode when it is not today's global clock (lever #2's contact gate), else None."""
    stepper = getattr(engine, "stepper", None)
    mode = getattr(getattr(stepper, "base", stepper), "clock_mode", None)
    return mode() if callable(mode) else None


def _ramps(targets: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    out = {}
    for name, value in targets.items():
        if isinstance(value, tuple) and len(value) == 2:
            out[name] = (np.asarray(value[0], float), np.asarray(value[1], float))
        else:
            out[name] = (np.asarray(value, float), np.asarray(value, float))
    return out


def _device(device: str) -> Any:
    import warp as wp  # noqa: PLC0415

    return wp.ScopedDevice(device)
