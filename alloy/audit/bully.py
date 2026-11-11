# SPDX-License-Identifier: Apache-2.0
"""Adapter to the bully agent (``alloy.bully.audit``): N random legal episodes on a scene, each finding as an
``AuditFinding``.

An episode drives the scene's actuators with random legal commands (within their ranges and rate limits), or, with
no actuators, pushes free bodies with at most 0.2 x their weight per axis, or else runs passive. Each bad episode
is a finding: its severity (S1 impossible, S2 degraded, S3 far from the origin), the cause (the monitor or health
flag that tripped), value and limit, the first tick and time, the body involved (for the first few per cause), the
health flags, and a JSON replay handle: ``alloy.bully.replay_finding(handle, library=...)`` runs that episode
alone. Findings are merged per cause and severity with the share of episodes that showed them. When this install
has no bully call, the audit says so and runs the ``alloy.check`` rounds alone. ``runner`` replaces the call.

Example::

    >>> res = run_bully(card, episodes=8, seed=0, device="cpu")          # doctest: +SKIP
    >>> res.available, len(res.findings), res.summary["counts"]           # doctest: +SKIP
    (True, 1, {'clean': 7, 'S1': 0, 'S2': 1, 'S3': 0})
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from alloy.audit.report import AuditFinding

log = logging.getLogger("alloy.audit")

#: bully severity (S1 worst) -> audit severity
SEVERITY = {"S1": "error", "1": "error", "S2": "warn", "2": "warn", "S3": "info", "3": "info"}
#: words in a bully class -> what to change (first match wins)
FIXES: tuple[tuple[str, str], ...] = (
    ("nonfinite", "a world went NaN: lower dt (control_dt / 2), remove start overlaps, or check extreme masses"),
    ("nan", "a world went NaN: lower dt (control_dt / 2), remove start overlaps, or check extreme masses"),
    ("overflow", "too many contacts: give the scene fewer touching parts, or raise the contact budget"),
    ("pen", "deep penetration: thicken thin parts, lower dt, or limit the drive force/speed that pushes through"),
    ("pass_through", "a part passed through another: thicken it or lower dt"),
    ("speed", "a body moved impossibly fast: check its mass and inertia, and the actuator limits"),
    ("teleport", "a body jumped: remove start overlaps and check welds/joints for conflicting constraints"),
    ("energy", "passive parts gained energy: lower dt, raise damping, or check restitution > 1 / stiff contacts"),
    ("separation", "a joint pulled apart: raise its stiffness / solver iterations or lower the load"),
    ("displacement", "an object the robot does not touch moved: check friction and resting contacts"),
    ("e_gain", "passive parts gained energy: lower dt, raise damping, or check restitution > 1 / stiff contacts"),
    ("capacity", "too many contacts: give the scene fewer touching parts, or raise the contact budget"),
    ("unconverged", "the solver did not converge: lower dt, raise solver iterations, or soften stiff contacts"),
    ("invalid", "health failed the world (see its flags): replay it alone and fix what the flags name"),
)
MAX_BODIES = 3
SUMMARY_KEYS = ("counts", "by_cause", "s1_upper95", "channel", "ordered_route", "horizon_s", "device", "seconds")
REPLAY_HOW = ("from alloy.bully import replay_finding; replay_finding(handle, library=...)  # handle: this "
              "replay's 'handle'; library: alloy.compose.library([your card folders]) if the scene uses them")


@dataclass
class BullyResult:
    """What the bully run gave: ``available`` (the call exists here), ``findings``, ``episodes`` run, a ``note``."""

    available: bool
    findings: list[AuditFinding] = field(default_factory=list)
    episodes: int = 0
    note: str = ""
    summary: dict[str, Any] = field(default_factory=dict)


def _call() -> Callable[..., Any] | None:
    try:
        from alloy.bully import audit  # noqa: PLC0415
    except ImportError:
        return None
    return audit if callable(audit) else None


def run_bully(card: Any, *, episodes: int, seed: int, device: str | None, runner: Callable[..., Any] | None = None,
              control_dt: float | None = None, library: Any = None, horizon_s: float | None = None) -> BullyResult:
    """Run the bully agent on ``card`` (a ``SceneCard``; ``library`` resolves its objects; ``horizon_s`` the episode
    length, default the bully's) and convert its findings."""
    fn = runner or _call()
    if fn is None:
        return BullyResult(False, note="the bully library call (alloy.bully.audit) is not in this install; "
                                       "the audit ran the alloy.check rounds only")
    try:
        extra = {"horizon_s": float(horizon_s)} if horizon_s else {}
        raw = fn(card, episodes=episodes, seed=seed, device=device, library=library, **extra)
    except Exception as exc:  # noqa: BLE001 - a crash of the bully run is itself reported, not raised
        log.exception("bully audit failed")
        return BullyResult(True, [AuditFinding("bully", "error", "crash", f"the bully run raised {type(exc).__name__}: "
                                               f"{exc}", "report it with the scene; the engine should not raise")],
                           episodes, "the bully run raised")
    items = list(getattr(raw, "findings", raw) or [])
    ran = int(getattr(raw, "episodes", 0) or episodes)
    summary = {k: getattr(raw, k) for k in SUMMARY_KEYS if getattr(raw, k, None) is not None}
    note = ("the bully agent found nothing to drive (no actuators, no free bodies): its episodes differ only by the "
            "scene's randomized values") if str(summary.get("channel", "")).startswith("passive") else ""
    return BullyResult(True, merge([convert(f, control_dt or None) for f in items], ran), ran, note, summary)


def _get(f: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        v = f.get(n) if isinstance(f, dict) else getattr(f, n, None)
        if v is not None:
            return v
    return default


def convert(f: Any, dt: float | None = None) -> AuditFinding:
    """One bully finding (``alloy.bully.AuditFinding``, or a dict / object with the same or older field names) ->
    ``AuditFinding``."""
    kind = str(_get(f, "cause", "cls", "klass", "kind", "monitor", "class_", "class", default="unknown"))
    sev_raw = _get(f, "severity", default="S2")
    sev = SEVERITY.get(getattr(sev_raw, "name", str(sev_raw)), "warn")
    tick, time_s = _get(f, "first_tick", "first_step", "tick"), _get(f, "time_s")
    body = _get(f, "body", "body_name", "object")
    value, limit = _get(f, "value"), _get(f, "threshold", "limit")
    msg = _get(f, "message", "text") or _message(kind, value, limit, body, f)
    when = float(time_s) if time_s is not None else (float(tick) * dt if tick is not None and dt else None)
    fix = _fix(kind)
    if tick is not None and int(tick) == 0:
        fix = "it tripped at the first tick, before any input mattered: fix the start pose first (see alloy.check's " \
              "overlap findings); then " + fix
    return AuditFinding("bully", sev, kind, str(msg), fix, (str(body),) if body is not None else (),
                        first_time_s=when, value=float(value) if value is not None else None,
                        limit=float(limit) if limit is not None else None, replay=_replay(f, tick))


def _message(kind: str, value: Any, limit: Any, body: Any, f: Any) -> str:
    out = kind
    if value is not None and limit is not None:
        out += f": {float(value):.3g} against a limit of {float(limit):.3g}"
    if body is not None:
        out += f" on {body}"
    if _get(f, "body_detail"):
        out += f" ({_get(f, 'body_detail')})"
    flags = _get(f, "flags") or ()
    return out + (f"; health flags {', '.join(map(str, flags))}" if flags else "")


def _replay(f: Any, tick: Any) -> dict[str, Any]:
    handle = _get(f, "replay", "replay_handle", "handle", "repro", default={})
    if isinstance(handle, dict) and not _get(f, "handle"):
        return handle                                         # an older runner's own replay dict
    if hasattr(handle, "to_dict"):
        return {"handle": handle.to_dict(), "first_tick": tick, "replays_alone": _get(f, "replays_alone"),
                "how": REPLAY_HOW}
    if isinstance(handle, dict):
        return {"handle": handle, "first_tick": tick, "how": REPLAY_HOW}
    return {"handle": str(handle), "first_tick": tick, "id": _get(f, "id")}


def _fix(kind: str) -> str:
    k = kind.lower()
    return next((fix for word, fix in FIXES if word in k), "replay it (replay handle) and change what moves wrongly")


def merge(findings: list[AuditFinding], episodes: int) -> list[AuditFinding]:
    """One finding per (cause, severity) with how many episodes showed it, the bodies named (up to ``MAX_BODIES``)
    and the first episode's replay."""
    out: dict[tuple, AuditFinding] = {}
    for f in findings:
        key = (f.kind, f.severity)
        if key not in out:
            out[key] = f
            continue
        first = out[key]
        first.seen += 1
        extra = [o for o in f.objects if o not in first.objects]
        first.objects = tuple([*first.objects, *extra][:MAX_BODIES])
    for f in out.values():
        f.rate = min(1.0, f.seen / max(episodes, 1))
    return list(out.values())


__all__ = ["BullyResult", "FIXES", "convert", "merge", "run_bully"]
