# SPDX-License-Identifier: Apache-2.0
"""``audit(scene)``: is this scene sound under many starts and many legal inputs?

Two parts, merged into one ``AuditReport``:

1. ``alloy.check`` over ``rounds`` rounds, a new seed each (randomized worlds differ): every finding with the share
   of rounds it appeared in and the seed that replays it;
2. the bully agent (``alloy.bully``, through ``alloy.audit.bully``): ``episodes`` random legal episodes; every
   finding with its class, first tick, body, the share of episodes and a replay handle.

Example::

    >>> report = audit("demo_peg_d12", rounds=2, episodes=8, device="cpu")       # doctest: +SKIP
    >>> report.ok, report.counts()                                               # doctest: +SKIP
    (True, {'error': 0, 'warn': 1, 'info': 0})
"""

from __future__ import annotations

import logging
import re
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alloy.audit.bully import run_bully
from alloy.audit.report import AuditFinding, AuditReport

log = logging.getLogger("alloy.audit")

MAX_ROUNDS = 50
MAX_EPISODES = 10_000
_NUM = re.compile(r"[-+]?\d+(\.\d+)?(e[-+]?\d+)?")


def audit(scene: Any, *, rounds: int = 3, episodes: int = 16, seed: int = 0, device: str | None = None,
          worlds: int | None = None, seconds: float | None = None, library: Any = None,
          bully: bool = True, runner: Callable[..., Any] | None = None) -> AuditReport:
    """Audit ``scene`` (a scene YAML path, library scene name, YAML text or ``SceneCard``); see the module docstring.

    ``library``: your card folders (a path or a list, searched first, e.g. where ``alloy create`` wrote) or a card
    library (default: Alloy's library with every imported library);
    ``bully=False`` skips the bully agent; ``runner`` stands in for its library call.
    Example: ``audit("my_scenes/scenes/scene.yaml", rounds=3, episodes=32).to_json()``."""
    _limits(rounds, episodes)
    t0 = time.perf_counter()
    given = _roots(library)
    src = _load(scene, list(given) if given or library is None else library)         # check adds the scene's root
    roots = _with_scene_root(given, src.path)
    report = AuditReport(scene=src.label, runs={"rounds": rounds, "episodes": episodes if bully else 0,
                                                "seed": seed, "device": device})
    _check_rounds(report, src, (rounds, seed, roots), device, worlds, seconds)
    report.timings_s["check_rounds"] = time.perf_counter() - t0
    if bully:
        t = time.perf_counter()
        res = run_bully(src.card, episodes=episodes, seed=seed, device=device, runner=runner,
                        control_dt=getattr(src.card, "control_dt", None), library=src.lib, horizon_s=seconds)
        report.runs["bully"] = res.available
        if res.summary:
            report.runs["bully_summary"] = res.summary
            report.runs["episodes"] = res.episodes
        for f in res.findings:
            if roots and isinstance(f.replay, dict) and "handle" in f.replay:
                f.replay["library"] = list(roots)                # replay_finding needs the scene's card folders
        report.findings.extend(res.findings)
        if res.note:
            report.notes.append(res.note)
        report.timings_s["bully"] = time.perf_counter() - t
    report.timings_s["total"] = time.perf_counter() - t0
    log.info("audit %s: %s in %.1f s", report.scene, "OK" if report.ok else "FAIL", report.timings_s["total"])
    return report


def _limits(rounds: int, episodes: int) -> None:
    from alloy.audit.report import AuditFinding as _  # noqa: F401, PLC0415 - keeps the import order lazy

    if not isinstance(rounds, int) or not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"rounds={rounds!r}: pass an int from 1 to {MAX_ROUNDS}")
    if not isinstance(episodes, int) or not 1 <= episodes <= MAX_EPISODES:
        raise ValueError(f"episodes={episodes!r}: pass an int from 1 to {MAX_EPISODES}")


def _roots(library: Any) -> tuple[str, ...]:
    """The card folders in ``library`` when it is a path or a list of paths (else none)."""
    if isinstance(library, (str, Path)):
        return (str(library),)
    if isinstance(library, (list, tuple)) and all(isinstance(r, (str, Path)) for r in library):
        return tuple(str(r) for r in library)
    return ()


def _load(scene: Any, library: Any):
    """``alloy.check``'s scene source; a card that does not load raises a ``SceneError`` with check's own fix."""
    from alloy.check.cards import card_error_finding  # noqa: PLC0415
    from alloy.check.source import SceneInputError, load_source  # noqa: PLC0415
    from alloy.core.errors import CardError  # noqa: PLC0415

    try:
        return load_source(scene, library)
    except CardError as exc:
        f = card_error_finding(exc)
        raise SceneInputError(f"the scene card does not load: {f.message}", f.fix, field="scene") from exc


def _with_scene_root(roots: tuple[str, ...], path: Path | None) -> tuple[str, ...]:
    """``roots`` plus the card folder a scene file sits in (``<root>/scenes/<name>.yaml``): the folder its generated
    cards and robot files are in, which every replay needs."""
    if path is None:
        return roots
    root = next((p.parent for p in path.resolve().parents if p.name == "scenes"), None)
    return roots if root is None or str(root) in roots else (*roots, str(root))


def library(roots: Any = ()) -> Any:
    """The card library an audit reads: ``roots`` (searched first), Alloy's library, every fetch cache and every
    imported library; the same roots ``alloy.check`` uses. For ``replay_finding(handle, library=...)``.

    Example: ``library(["my_scenes"]).has_object("cube_beech_30")``."""
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.cards.roots import assets_root  # noqa: PLC0415
    from alloy.check.source import default_roots  # noqa: PLC0415

    mine = [Path(r).expanduser().resolve() for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    rest = [d for d in default_roots() if all(d.resolve() != m for m in mine)]
    return CardLibrary([*mine, *rest], assets=assets_root())


def _check_rounds(report: AuditReport, src, plan: tuple[int, int, tuple[str, ...]], device: str | None,
                  worlds: int | None, seconds: float | None) -> None:
    from alloy.check import check  # noqa: PLC0415

    rounds, seed, roots = plan
    target = str(src.path or src.label)                          # a file's full path: its stem is not a scene name
    seen: dict[tuple, AuditFinding] = {}
    for r in range(rounds):
        s = seed + r
        rep = check(src.card, worlds, seconds, device=device, seed=s, library=src.lib)
        report.runs.setdefault("check_settings", rep.settings)
        for f in rep.findings:
            if f.level == "info":
                continue
            key = (f.check, f.level, tuple(sorted(f.objects)), _NUM.sub("#", f.message))
            if key in seen:
                seen[key].seen += 1
                continue
            seen[key] = AuditFinding("check", f.level, f.check, f.message, f.fix, tuple(f.objects), value=f.value,
                                     limit=f.limit, unit=f.unit, replay=_replay(target, s, rep.settings, roots))
    for f in seen.values():
        f.rate = f.seen / rounds
        report.add(f)


def _replay(target: str, seed: int, settings: dict[str, Any], roots: tuple[str, ...] = ()) -> dict[str, Any]:
    """The command that re-runs one check round. ``target``: the scene file's full path, else its library name."""
    cmd = f"python -m alloy.check {shlex.quote(target)} --seed {seed}"
    for k in ("worlds", "seconds", "device"):
        if settings.get(k) is not None:
            cmd += f" --{k} {settings[k]}"
    cmd += "".join(f" --library {shlex.quote(r)}" for r in roots)
    return {"seed": seed, "settings": {k: settings.get(k) for k in ("worlds", "seconds", "device")},
            "command": cmd + " --json"}


__all__ = ["audit", "library"]
