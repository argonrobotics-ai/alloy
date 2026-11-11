# SPDX-License-Identifier: Apache-2.0
"""``alloy.bully.audit``: N random-legal bully episodes on any scene card, returned as findings.

Every episode drives the scene's own legal inputs (``scene_target``: MuJoCo actuators, else bounded pushes on free
bodies, else nothing) with SB-7's Ornstein-Uhlenbeck traces (``stress.random_legal_traces``) and is classified by
the §4.3 table (``severity``): S1 = impossible (blow-up, penetration fail, energy created, impossible speed),
S2 = degraded, S3 = far from origin. Episodes run in batches of ``batch`` worlds; batch b uses seed ``seed + b``,
so ``(seed + b, batch, world)`` is a replay handle that rebuilds the exact trace.

Each finding names its class, the row that set it, the first control tick, a replay handle and, for the first
``attribute`` findings of each (class, cause), the body involved, found by replaying that world alone up to its first
tick with a host probe (``culprit``); ``replays_alone`` says whether the world trips the same way alone.

SB-7 itself is unchanged: it runs ``python -m alloy.bully sb7`` on the registered targets.

Example::

    >>> from alloy.bully import audit
    >>> report = audit("demo_peg_d12", episodes=4, seed=0, device="cpu", horizon_s=0.2)  # doctest: +SKIP
    >>> report.ok, report.counts                                                       # doctest: +SKIP
    (True, {'clean': 4, 'S1': 0, 'S2': 0, 'S3': 0})
    >>> print(report.to_json())                                                        # doctest: +SKIP
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alloy.bully.culprit import BODY_CAUSES, BodyProbe, culprit
from alloy.bully.rate import blowup_rate
from alloy.bully.replay import EpisodeRunner
from alloy.bully.scene_target import AuditOptions, Channel, scene_target
from alloy.bully.search import alloy_sha
from alloy.bully.severity import world_cause
from alloy.bully.stress import random_legal_traces
from alloy.bully.types import BullyLimits, BullyTarget, EpisodeResult
from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")

GPU_BATCH, CPU_BATCH = 1024, 8
LEVELS = {1: "S1", 2: "S2", 3: "S3"}


@dataclass(frozen=True)
class ReplayHandle:
    """Everything ``replay_finding`` needs to rebuild one episode's trace and run it alone."""

    scene: str  # as given: YAML path, library scene name or YAML text (a SceneCard object as its YAML)
    seed: int  # the batch's trace seed
    batch: int  # worlds in that batch (world w's trace depends on the batch size)
    world: int
    horizon_s: float
    options: dict[str, Any] = field(default_factory=dict)  # AuditOptions fields

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AuditFinding:
    severity: str  # "S1" | "S2" | "S3"
    cause: str  # the monitor row or health flag that set the severity
    value: float
    threshold: float
    first_tick: int  # control tick (0-based) at which the world first tripped
    time_s: float  # end of that tick
    handle: ReplayHandle
    body: str | None = None  # the placed object or body label involved (attributed findings only)
    body_index: int | None = None  # its body index in the one-world model
    body_detail: str | None = None  # how it was picked, e.g. "fastest body: 23.1 m/s"
    replays_alone: bool | None = None  # same class at the same tick when the world runs alone (attributed only)
    flags: tuple[str, ...] = ()  # WP5 health flags set in the world (e.g. PEN_FAIL, UNCONVERGED, LATCHED)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class AuditReport:
    scene: str
    device: str
    episodes: int
    seed: int
    horizon_s: float
    channel: str  # the command channel the episodes drove
    ordered_route: bool  # WP5's ordered MuJoCo Warp route on: GPU findings replay alone bit for bit
    counts: dict[str, int]  # clean / S1 / S2 / S3 episodes
    by_cause: dict[str, int]  # "S1:PEN_FAIL" -> episodes
    s1_rate: float
    s1_upper95: float  # one-sided 95 % upper bound on the S1 rate (3/N at zero, else Clopper-Pearson)
    findings: list[AuditFinding]
    seconds: float
    alloy_sha: str

    @property
    def ok(self) -> bool:
        """No S1 episode."""
        return self.counts.get("S1", 0) == 0

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "ok": self.ok}

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)


def audit(scene: Any, episodes: int = 256, seed: int = 0, device: str | None = None, *, horizon_s: float = 2.0,
          batch: int | None = None, options: AuditOptions | None = None, limits: BullyLimits | None = None,
          attribute: int = 3, library: Any = None) -> AuditReport:
    """Run ``episodes`` random-legal episodes of ``scene`` (YAML path, library scene name, YAML text or
    ``SceneCard``) on ``device`` (None: ``cuda:0`` if available, else ``cpu``) and return the findings
    (module doc). Episodes are rounded up to whole batches; ``report.episodes`` is the number run."""
    t0 = time.perf_counter()
    dev, opts, lim = resolve_device(device), options or AuditOptions(), limits or BullyLimits()
    if episodes < 1 or horizon_s <= 0.0:
        raise BullyError(f"episodes={episodes}, horizon_s={horizon_s}: pass episodes >= 1 and horizon_s > 0")
    target, channel = scene_target(scene, opts, library, dev)
    size = int(batch or min(episodes, GPU_BATCH if dev.startswith("cuda") else CPU_BATCH))
    base = ReplayHandle(scene_text(scene), seed, size, 0, float(horizon_s), dataclasses.asdict(opts))
    runner = EpisodeRunner(target, size, dev, lim)
    severities, findings = [], []
    for b in range(math.ceil(episodes / size)):
        res = runner.run(random_legal_traces(target, size, horizon_s, seed=seed + b))
        severities.append(res.severity)
        findings += episode_findings(res, lim, dataclasses.replace(base, seed=seed + b), 1.0 / target.actions.rate_hz)
        log.info("audit %s: batch %d (%d worlds): S1 %d, S2 %d", target.name, b, size,
                 int((res.severity == 1).sum()), int((res.severity == 2).sum()))
    findings = attribute_bodies(target, findings, dev, lim, attribute)
    report = _report(target, channel, np.concatenate(severities), findings, base, dev, time.perf_counter() - t0)
    report.ordered_route = bool(getattr(runner.sim, "ordered", False))
    return report


def resolve_device(device: str | None) -> str:
    import warp as wp  # noqa: PLC0415

    return device or ("cuda:0" if wp.is_cuda_available() else "cpu")


def scene_text(scene: Any) -> str:
    """The scene as the handle stores it: a path or name as given, a SceneCard as its canonical YAML."""
    if isinstance(scene, (str, Path)):
        return str(scene)
    from alloy.cards.loader import dump_card  # noqa: PLC0415

    return dump_card(scene)


def episode_findings(res: EpisodeResult, limits: BullyLimits, handle: ReplayHandle, tick_s: float
                     ) -> list[AuditFinding]:
    """One finding per world with a severity, named by the row that set it (``severity.world_cause``)."""
    out = []
    for w in np.nonzero(res.severity > 0)[0]:
        cause, value, thr, level = world_cause(res.health, res.monitors, limits, int(w))
        k = int(res.first_step[w])
        out.append(AuditFinding(LEVELS.get(int(level), f"S{int(level)}"), cause, float(value), float(thr), k,
                                round((k + 1) * tick_s, 9), dataclasses.replace(handle, world=int(w)),
                                flags=flag_names(int(np.asarray(res.health.flags)[w]))))
    return out


def flag_names(bits: int) -> tuple[str, ...]:
    """The ``HealthFlags`` names set in ``bits``."""
    from alloy.precision.types import HealthFlags  # noqa: PLC0415

    return tuple(f.name for f in HealthFlags if f.value and bits & int(f.value) == int(f.value))


def attribute_bodies(target: BullyTarget, findings: list[AuditFinding], device: str, limits: BullyLimits,
                     per_group: int) -> list[AuditFinding]:
    """Body attribution for the first ``per_group`` findings of each (class, cause), S1 first (module doc)."""
    if per_group <= 0 or not findings:
        return findings
    groups: dict[tuple[str, str], int] = collections.Counter()
    alone = EpisodeRunner(target, 1, device, limits)
    out = list(findings)
    for i in sorted(range(len(out)), key=lambda j: out[j].severity):
        f = out[i]
        if groups[(f.severity, f.cause)] >= per_group:
            continue
        groups[(f.severity, f.cause)] += 1
        out[i] = _attributed(alone, target, f, limits)
    return out


def _attributed(alone: EpisodeRunner, target: BullyTarget, f: AuditFinding, limits: BullyLimits) -> AuditFinding:
    h = f.handle
    trace = random_legal_traces(target, h.batch, h.horizon_s, seed=h.seed)[h.world: h.world + 1, : f.first_tick + 1]
    alone.sim.reset()
    probe = BodyProbe(alone.model, alone.sim.state)
    alone.sim.add_step_observer(probe.observer)
    try:
        res = alone.run(trace, reset=False)
    finally:
        alone.sim.observers.remove(probe.observer)
    same = bool(res.severity[0] > 0 and LEVELS.get(int(res.severity[0])) == f.severity
                and int(res.first_step[0]) == f.first_tick)
    who = culprit(probe, _cause_for_body(f), _namer(alone))
    return dataclasses.replace(f, body=who.label, body_index=who.body, body_detail=who.detail, replays_alone=same)


def _cause_for_body(f: AuditFinding) -> str:
    """The cause to attribute: an ``INVALID`` world (health failed it closed) is attributed by its first flag that
    names a body check (e.g. ENERGY_JUMP, PEN_FAIL), else stays ``INVALID``."""
    if f.cause != "INVALID":
        return f.cause
    return next((fl for fl in f.flags if fl in BODY_CAUSES), f.cause)


def _namer(runner: EpisodeRunner) -> Any:
    """(kind, index) -> the placed object's name (``alloy.check.objects.ObjectIndex``), else the body or shape
    label; kind is "body" or "shape"."""
    labels = {k: list(getattr(runner.model, f"{k}_label", None) or []) for k in ("body", "shape")}
    index = None
    try:
        from alloy.check.objects import ObjectIndex  # noqa: PLC0415

        index = ObjectIndex.build(runner.sim.built)
    except (ImportError, ValueError, AttributeError) as exc:
        log.debug("audit: no object index (%s); body labels only", exc)

    def name(kind: str, i: int) -> str | None:
        if i is None or i < 0:
            return None
        owner = index.name_of(kind, i) if index is not None else None
        label = labels[kind][i] if i < len(labels[kind]) else None
        if owner and label and owner not in str(label):
            return f"{owner}/{label}"
        return str(label or owner or f"{kind} {i}")  # a robot link's label already starts with the robot's name
    return name


def _report(target: BullyTarget, channel: Channel, sev: np.ndarray, findings: list[AuditFinding],
            base: ReplayHandle, device: str, seconds: float) -> AuditReport:
    counts = {"clean": int((sev == 0).sum()), **{name: int((sev == lvl).sum()) for lvl, name in LEVELS.items()}}
    by_cause = collections.Counter(f"{f.severity}:{f.cause}" for f in findings)
    rate = blowup_rate(counts["S1"], int(sev.size))
    report = AuditReport(target.name.removeprefix("scene:"), device, int(sev.size), base.seed, base.horizon_s,
                         channel.description, False, counts, dict(sorted(by_cause.items())), rate.rate, rate.upper95,
                         findings, round(seconds, 3), alloy_sha())
    log.info("audit %s: %d episodes, %s, S1 upper bound %.3g, %.1f s", report.scene, report.episodes, counts,
             rate.upper95, seconds)
    return report


def replay_finding(handle: ReplayHandle | dict[str, Any], device: str | None = None, *,
                   limits: BullyLimits | None = None, library: Any = None, attribute: bool = True
                   ) -> AuditFinding | None:
    """Rebuild a finding's trace from its handle and run that world alone for the whole horizon; the finding it
    produces (with the body when ``attribute``), or None when the world runs clean alone."""
    h = handle if isinstance(handle, ReplayHandle) else ReplayHandle(**handle)
    dev, lim = resolve_device(device), limits or BullyLimits()
    target, _ = scene_target(h.scene, AuditOptions(**h.options), library, dev)
    trace = random_legal_traces(target, h.batch, h.horizon_s, seed=h.seed)[h.world: h.world + 1]
    runner = EpisodeRunner(target, 1, dev, lim)
    found = episode_findings(runner.run(trace), lim, h, 1.0 / target.actions.rate_hz)
    if not found:
        return None
    f = dataclasses.replace(found[0], handle=h)  # the original world, not the alone run's world 0
    return _attributed(runner, target, f, lim) if attribute else f
