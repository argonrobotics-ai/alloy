# SPDX-License-Identifier: Apache-2.0
"""``Scene.rigid_step`` and the build's request to ``alloy.scene.rigid_step`` (SCR-WP0-RIGIDDT).

``scene.rigid_step = "auto"`` (the default): a rigid-only scene of free bodies steps its rigid bodies at the largest
stable step the scene allows (sizes, start drops, loads, materials; up to the control period ``dt``), and every other
scene at the fixed step; ``sim.routes()`` names the step and what bound it. ``"fixed"`` keeps the fixed step (1/600 s
snapped down to the grid dt / 2**j: 1.25 ms at dt 0.01) for every scene. A number in seconds pins the step (snapped
down to the grid; the build warns when it breaks a stability cap). The precision and debug profiles, and builds with
plug-ins, keep the fixed step under "auto". Scene JSON keeps the setting (``to_dict``), so worker processes get it.
``"fast"`` (SPEED-NOW S1, opt-in; coordinator ruling 2026-11-09 09:18Z): the scenes "auto" takes step at the largest
step whose predicted rest sag stays within 1 % of each body's thinnest member, so contacts can be softer than at the
fixed step (``alloy.scene.rigid_step``, Fast mode). ``profile="fast"`` turns it on when ``rigid_step`` was never set
(the build warns once, only when the fast mode really steps the scene); any value the user sets, "auto" included, wins
over the profile. ``DEFAULT`` marks the
never-set value: it equals "auto", is not saved in Scene JSON, and survives copies, pickles and ``from_dict``.
("today", the name before 0.1.0, still reads as "fixed".)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from alloy.api.errors import SceneError

if TYPE_CHECKING:
    from alloy.scene.rigid_step import RigidStepRequest

AUTO_PROFILES = ("standard", "fast")
MODES = ("auto", "fast", "fixed")
FAST_PROFILES = ("fast",)          # "auto" under these profiles is "fast"
ALIASES = {"today": "fixed"}      # the name before 0.1.0, for old Scene JSON (not advertised)


class _Unset(str):
    """``rigid_step`` never set: "auto", which ``profile="fast"`` may turn into "fast" (module docstring)."""

    __slots__ = ()


DEFAULT = _Unset("auto")


def is_default(value: Any) -> bool:
    """True for ``DEFAULT``, the never-set ``rigid_step`` (also after a copy, a pickle or ``Scene.from_dict``)."""
    return isinstance(value, _Unset)


def checked(value: Any) -> str | float:
    """"auto", "fast", "fixed" or a positive step in seconds; a SceneError naming the choices otherwise."""
    if isinstance(value, str) and ALIASES.get(value, value) in MODES:
        return ALIASES.get(value, value)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value)
    raise SceneError(f"rigid_step must be 'auto', 'fast', 'fixed' or a step in seconds, got {value!r}",
                     "pass rigid_step='auto' (the default), 'fast', 'fixed' or e.g. 0.00125", field="rigid_step")


class RigidStepMixin:
    """``Scene.rigid_step`` (module docstring); checked when set."""

    @property
    def rigid_step(self) -> str | float:
        """"auto" (default), "fast", "fixed" or a step in seconds (``alloy.api._rigid_step``); set it before
        building."""
        return getattr(self, "_rigid_step", DEFAULT)

    @rigid_step.setter
    def rigid_step(self, value: Any) -> None:
        self._rigid_step = checked(value)


def saved(value: str | float) -> dict[str, Any]:
    """The Scene JSON entry: none for the default."""
    return {} if is_default(value) else {"rigid_step": value}


def request(scene: Any, profile: str, plugins: Sequence[Any], per_world_pos: Mapping[str, Any], *,
            announce: bool = True, newton_contacts: bool | None = None) -> RigidStepRequest:
    """What the build asks ``alloy.scene.rigid_step`` for (module docstring), with every world's start rows.
    ``announce``: warn when the profile's fast mode steps the scene (False in all but one worker of a split build).
    ``newton_contacts``: the build's contact pipeline is Newton's; None: what ``contacts="auto"`` picks."""
    from alloy.scene.rigid_step import RigidStepRequest  # noqa: PLC0415 - `import alloy` stays light (no Warp)

    if newton_contacts is None:
        from alloy.api._build import _contacts  # noqa: PLC0415

        newton_contacts = _contacts(scene, "auto") == "newton"

    mode, made = scene.rigid_step, False
    if is_default(mode) and profile in FAST_PROFILES:
        mode, made = "fast", True
    if mode in ("auto", "fast") and (profile not in AUTO_PROFILES or any(p is not None for p in plugins)):
        mode = "fixed"
    return RigidStepRequest(str(mode) if isinstance(mode, str) else mode, rows=dict(per_world_pos) or None,
                            from_profile=made, announce=_announced(profile) if made and announce else None,
                            newton_contacts=newton_contacts, explicit=not is_default(scene.rigid_step),
                            warn=_warned if announce else _unheard)


def _warned(text: str) -> None:
    """A gap warning of ``alloy.scene.rigid_step_warn``, to the user (logged, and printed when no handler shows it)."""
    from alloy.api._say import warn  # noqa: PLC0415

    warn("%s", text)


def _unheard(text: str) -> None:
    """A gap warning in a build the user does not run (``estimate``, a split build's other workers): kept in
    ``report.warnings`` only."""


def _announced(profile: str):
    """The build's one warning when ``profile`` turned the fast mode on and it steps the scene."""
    def say(step: Any) -> None:
        from alloy.api._say import warn  # noqa: PLC0415

        warn("profile %r: rigid fast mode on, the rigid step %g ms (fixed step %g ms): contacts can be softer than "
             "at the fixed step (a resting body sinks up to 1 %% of its thinnest member); free bodies step on "
             "implicitfast with a capped friction pull on MuJoCo Warp (a 30 cm cube sliding at 5 m/s rocks under "
             "1 deg) and stay on rk4 on the CPU's MuJoCo C route (that cube: 6 deg, an 8 cm hop); "
             "scene.rigid_step = 'auto' keeps the standard step", profile, step.dt * 1e3, step.today * 1e3)
    return say


__all__ = ["AUTO_PROFILES", "DEFAULT", "FAST_PROFILES", "RigidStepMixin", "is_default", "checked", "request", "saved"]
