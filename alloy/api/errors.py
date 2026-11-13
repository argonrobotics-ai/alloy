# SPDX-License-Identifier: Apache-2.0
"""Errors of the public API. Every one says what went wrong (``message``) and what to change (``fix``).

Engine errors raised while building or stepping are wrapped (``raise ... from engine_error``) into one of these with a
remedy; ``to_dict()`` gives the same as JSON for tools. Example::

    import alloy
    try:
        alloy.Scene().add_box("cube", size=(0.1, 0.1))
    except alloy.SceneError as e:
        print(e.message, "->", e.fix)       # size needs 3 values -> pass size=(x, y, z) in metres
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from alloy.core.errors import AlloyError, PrecisionError


class AlloyAPIError(AlloyError):
    """Base class: ``message`` (what went wrong), ``fix`` (what to change), ``field`` (the argument or card field).

    Example: ``AlloyAPIError("mass must be > 0", "pass mass=1.0 (kg)", field="mass").to_dict()``."""

    def __init__(self, message: str, fix: str, *, field: str | None = None) -> None:
        super().__init__(f"{message}\n  fix: {fix}")
        self.message, self.fix, self.field = message, fix, field

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form. Example: ``{"error": "SceneError", "message": ..., "fix": ..., "field": "size"}``."""
        return {"error": type(self).__name__, "message": self.message, "fix": self.fix, "field": self.field}


class SceneError(AlloyAPIError, ValueError):
    """A scene description is invalid (bad argument, unknown material, clashing name). Example: a negative size."""


class BuildError(AlloyAPIError, RuntimeError):
    """The engine could not build the scene (routing, assembly, capacity, install). Example: an unknown device."""


class SimulationError(AlloyAPIError, RuntimeError):
    """Stepping failed or a world became invalid (non-finite, deep penetration). Example: ``sim.step()`` on INVALID."""


class UnknownNameError(AlloyAPIError, LookupError):
    """A body, joint or object name is not in the scene; the message lists the names that are.

    Example: ``sim.state()["cub.pos"]`` -> use ``sim.names()`` to see the names."""


class FeatureUnavailableError(AlloyAPIError, RuntimeError):
    """A feature this install does not have yet (for example cameras before ``alloy.camera`` is installed)."""


# --- engine error -> remedy ------------------------------------------------------------------------------------

_INSTALL_FIX = ("install Alloy with its pinned engine packages: `pip install alloy-engine` (import name `alloy`) in a "
                "Python 3.11 or 3.12 virtual environment; without an NVIDIA GPU pass device='cpu'")

#: (engine error class name, message regex, fix). First match wins; the regex runs on the engine message.
_REMEDIES: tuple[tuple[str, str, str], ...] = (
    ("CardError", r"inside out", "flip the mesh's triangle winding (or re-export it with outward normals)"),
    ("CardError", r"file not found", "the file named in the message is missing: pass an existing file to "
                                     "add_robot / add_mesh, and move or copy a saved card folder whole (its robots/ "
                                     "and meshes/ paths are relative to the folder)"),
    ("CardError", r"non-finite vertex", "repair or remove the mesh's NaN / inf vertices: every coordinate must be a "
                                         "finite number in metres"),
    ("CardError", r"unknown (material|object|scene)|no (material|object|scene)|not found",
     "use a name from alloy.materials() / alloy.scenes(), or add the card with Scene.add_material"),
    ("CardError", r"", "change the field named in the message (Scene.add_* argument of that object, or its card)"),
    ("UnitError", r"", "give numbers in SI units (metres, kilograms, seconds) or unit strings such as '10 mm'"),
    ("QuantityError", r"", "give finite numbers in SI units"),
    ("FloatingContainerError", r"", "make the floating container heavier (the message gives the lightest mass that "
                                    "routes), smaller, or static; lighter ones are outside the tested range "
                                    "(docs/known_gaps/liquid-floating-container-ratio.md)"),
    ("StrongPairBesideLiquidError", r"", "make the free rigid objects heavier (the message gives the mass that "
                                         "routes), static or kinematic, or build the liquid in its own scene "
                                         "(docs/known_gaps/liquid-cloth-light-rigid.md)"),
    ("RouteError", r"", "this object/material combination has no solver yet: use a rigid material, a thicker "
                        "sheet, or another shape (sim.routes() shows why each object went where)"),
    ("CapacityError", r"particle", "use fewer or larger particles (coarser spacing / smaller region) or fewer worlds"),
    ("CapacityError", r"", "lower the world count or the object budget the message names"),
    ("PrecisionSpecError", r"", "widen the clearance, or drop precision_partners for this pair"),
    ("PrecisionError", r"", "a world became INVALID: lower dt (Scene(dt=...)), remove initial overlaps, or build "
                            "with profile='standard' to freeze bad worlds instead of raising"),
    ("NewtonPinError", r"", _INSTALL_FIX),
    ("CouplingError", r"no dynamic entry", "every object in this scene is static, so nothing steps: add a dynamic "
                                           "object (to check a static fixture, place a small object on it), or run "
                                           "plug-ins on it: alloy.build(scene, plugins=[...]) steps them alone"),
    ("CouplingError", r"", "too many different solvers touch each other in one scene: split it or make some objects "
                           "static"),
    ("AssemblyError", r"world_count", "pass worlds >= 1"),
    ("AssemblyError", r"encloses no volume", "give a closed mesh with every face wound counter-clockwise seen from "
                                             "outside (trimesh.repair.fix_normals(mesh) repairs the winding), or make "
                                             "the object static"),
    ("AssemblyError", r"cloth", "a cloth needs a thin box (Scene.add_cloth) or a triangle mesh"),
    ("AssemblyError", r"", "check the object named in the message; Scene.summary() lists every object"),
    ("ImportError", r"pxr|usd", 'reading USD needs the usd extra: pip install "alloy-engine[usd]"'),
    ("ModuleNotFoundError", r"warp|newton|mujoco", _INSTALL_FIX),
    ("ImportError", r"warp|newton|mujoco", _INSTALL_FIX),
)


#: an engine message that names a missing optional extra ("needs the optional [mesh] extra"): AUDIT-STACK INSTALL-2
_EXTRA = re.compile(r"needs the optional \[(\w+)\] extra")


def remedy_for(exc: BaseException) -> str:
    """The fix text for an engine exception, by class name (its MRO) and message. Example::

        remedy_for(RouteError("no route"))   # 'this object/material combination has no solver yet: ...'
    """
    names = [c.__name__ for c in type(exc).__mro__]
    text = str(exc)
    for cls, pattern, fix in _REMEDIES:
        if cls in names and re.search(pattern, text, re.IGNORECASE):
            return fix
    return "see the message above; if it names an engine internal, report it with the scene (Scene.to_dict())"


def wrap(exc: BaseException, stage: str, cls: type[AlloyAPIError] = BuildError) -> AlloyAPIError:
    """An API error for an engine exception raised during ``stage`` (keeps the engine message, adds the fix: the
    exception's own ``fix`` when it carries one, such as an unsupported route pair's nearest alternative, else the
    remedy for its kind). A message that names a missing optional extra becomes a ``FeatureUnavailableError`` with
    the install command. Example: ``raise wrap(e, "build") from e``."""
    if isinstance(exc, AlloyAPIError):
        return exc
    extra = _EXTRA.search(str(exc))
    if extra:                                    # a feature of an extra that is not installed: name the command
        e = extra.group(1)
        return FeatureUnavailableError(f"{stage} failed: {type(exc).__name__}: {exc}", f'pip install "alloy-engine[{e}]"')
    field = getattr(exc, "field", None)
    card = getattr(exc, "card", None)
    where = f"{card}.{field}" if card and field else None
    own = getattr(exc, "fix", None)
    fix = own.strip() if isinstance(own, str) and own.strip() else remedy_for(exc)
    return cls(f"{stage} failed: {type(exc).__name__}: {exc}", fix, field=where)


@contextmanager
def wrapped(stage: str, cls: type[AlloyAPIError] = BuildError) -> Iterator[None]:
    """Re-raise engine errors in the block as ``cls`` with a remedy. Example::

        with wrapped("build"):
            built = assembler.assemble(card, device)
    """
    try:
        yield
    except AlloyAPIError:
        raise
    except (AlloyError, ImportError) as exc:
        raise wrap(exc, stage, cls) from exc


# --- the world-count limit: Warp's raw failures become a BuildError (audit WORLDS-10) --------------------------

_LIMIT_FIX = ("build fewer worlds (worlds=...) or a smaller scene, or split the batch over several builds; "
              'alloy.estimate(scene, worlds, seconds)["memory_bytes"] gives the memory before you build')
#: (exception type, message regex, what ran out)
_LIMITS: tuple[tuple[type[BaseException], str, str], ...] = (
    (ValueError, r"maximum representable value of a signed 32-bit integer",
     "an array grew past Warp's 32-bit size limit (2**31 - 1 elements)"),
    (RuntimeError, r"Failed to allocate \d+ bytes|out of memory",
     "the device ran out of memory"),
)


def limit_error(exc: BaseException, worlds: int) -> BuildError | None:
    """A ``BuildError`` naming what ran out and the world count when ``exc`` is Warp's int32 array-size ValueError or
    a device out-of-memory RuntimeError, else None: also for an Alloy error, which names its own capacity and fix
    (a BuildError is a RuntimeError too). Example: ``err = limit_error(exc, 16384)``."""
    if isinstance(exc, AlloyError):
        return None
    for kind, pattern, what in _LIMITS:
        if isinstance(exc, kind) and re.search(pattern, str(exc), re.IGNORECASE):
            return BuildError(f"build failed with {worlds:,} world(s): {what} ({type(exc).__name__}: {exc})",
                              _LIMIT_FIX, field="worlds")
    return None


@contextmanager
def world_limits(worlds: int) -> Iterator[None]:
    """Inside the block, Warp's int32 array-size and out-of-memory failures raise ``BuildError`` (see
    ``limit_error``); every other exception passes unchanged. Example: ``with world_limits(n): sim = make(n)``."""
    try:
        yield
    except (ValueError, RuntimeError) as exc:
        err = limit_error(exc, worlds)
        if err is None:
            raise
        raise err from exc


# --- profile "precision": raising when a world becomes INVALID (the health monitor's on_invalid="raise") ---------

def raise_if_invalid(engine: Any) -> None:
    """Profile "precision" (``CheckPolicy.on_invalid="raise"``): raise ``PrecisionError`` for each world that became
    INVALID since the last call, naming it (``Sim.step`` calls this after every control period and re-raises it as
    ``SimulationError``). A world raises once: it stays frozen and later steps go on, until ``sim.reset()``, after
    which it can fail and raise again. A no-op under the other profiles, where INVALID worlds freeze and
    ``sim.health()`` reports them. Costs one host read of the health record per period (on a GPU a sync per period;
    its cost there is not measured yet).
    Example: ``raise_if_invalid(sim.engine)``."""
    monitor = getattr(engine, "health", None)
    if getattr(getattr(monitor, "cfg", None), "on_invalid", "latch") != "raise":
        return
    seen = getattr(engine, "_api_raised_invalid", frozenset())
    try:
        monitor.poll()
    except PrecisionError as exc:
        report = getattr(exc, "report", None)
        invalid = frozenset(getattr(report, "invalid_worlds", ()))
        engine._api_raised_invalid = invalid
        new = sorted(invalid - seen)
        if not invalid or (new and new[0] == min(invalid)):
            raise                                     # poll's message already names the world that just failed
        if new:                                       # name the world that failed now, not the first on record
            err = PrecisionError(f"{len(new)} world(s) became INVALID; world {new[0]}: {report.failure_text(new[0])}")
            err.report = report  # type: ignore[attr-defined]
            raise err from None
    else:
        engine._api_raised_invalid = frozenset()


def forget_raised(engine: Any) -> None:
    """After a reset every world may fail, and raise, again (``raise_if_invalid``)."""
    engine._api_raised_invalid = frozenset()


@contextmanager
def reading(engine: Any) -> Iterator[None]:
    """The monitor latching inside the block: under profile "precision" a poll raises ``PrecisionError`` for an
    INVALID world, and reading checks or the report after a failed step must still work. Host state only, and not
    thread-safe: a reader thread inside the block while the stepping thread polls delays that raise by one period.
    Example::

        with reading(sim.engine):
            report = sim.engine.report()
    """
    monitor = getattr(engine, "health", None)
    cfg = getattr(monitor, "cfg", None)
    if getattr(cfg, "on_invalid", "latch") != "raise":
        yield
        return
    monitor.cfg = dataclasses.replace(cfg, on_invalid="latch")
    try:
        yield
    finally:
        monitor.cfg = cfg
