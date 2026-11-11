# SPDX-License-Identifier: Apache-2.0
"""``evaluate_plugin``: contract checks, laws, B0/B1 attack, and the ship decision (SPEC §3.11, FAC-1, FAC-2).

A plug-in ships only if every contract check passes, ``lint_atomics`` is clean for a NATIVE plug-in, every law in its
manifest passes (``not_run`` blocks shipping; the hidden valid set of rig laws is read by the bench, never by the
plug-in), and B0 + B1 leave no replay-confirmed S1 or S2 finding. ``verdict_to_json`` / ``verdict_from_json`` carry
a verdict across the process boundary the factory loop uses to keep agent-written code out of its own process.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from alloy._vendor.newton.solvers import SolverBase

from alloy.bench.types import LawResult, MetricVerdict, Quantity
from alloy.bully.search import search_detailed
from alloy.bully.targets import get_target
from alloy.bully.types import BullyLimits, BullySeverity, Finding, SearchConfig
from alloy.core.errors import LawError, PluginContractError
from alloy.core.protocols import DetCapability
from alloy.factory.contract import ContractReport, dynamic_checks, static_checks
from alloy.factory.types import FactoryVerdict, PluginManifest

log = logging.getLogger("alloy.factory")

ATTACK_CPU = SearchConfig(worlds=16, generations=3, horizon_s=1.0, budget_gpu_h=0.05)


def load_manifest(path: Path) -> PluginManifest:
    rec = json.loads(Path(path).read_text())
    return PluginManifest(rec["name"], rec["module"], tuple(rec["routes"]), tuple(rec["laws"]),
                          tuple(rec["bully_scenes"]), rec["author"])


def plugin_class(module: Any, name: str) -> type:
    """The ``SolverBase`` subclass defined inside package ``module`` whose ``plugin_name`` is ``name``."""
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, SolverBase) and getattr(obj, "plugin_name", None) == name:
            if obj.__module__ == module.__name__ or obj.__module__.startswith(f"{module.__name__}."):
                return obj
    raise PluginContractError(f"{module.__name__} exports no SolverBase subclass with plugin_name {name!r}")


def _not_run(law: str, solver: str, device: str, reason: str) -> LawResult:
    return LawResult(law, solver, device, (), {"status": "not_run", "reason": reason}, {}, 0.0, False)


def run_laws(manifest: PluginManifest, device: str) -> tuple[LawResult, ...]:
    """Each manifest law through the bench (WP6 ``run_law``), or ``not_run`` with the reason."""
    try:
        bench = importlib.import_module("alloy.bench")
        law_mod = importlib.import_module("alloy.bench.law")
        registered = {spec.id: (spec, factory) for spec, factory in law_mod.registered_laws()}
    except (ImportError, AttributeError) as exc:
        return tuple(_not_run(i, manifest.name, device, f"bench library not merged: {exc}") for i in manifest.laws)
    out = []
    for law_id in manifest.laws:
        if law_id not in registered:
            out.append(_not_run(law_id, manifest.name, device, "law not registered by the plug-in"))
            continue
        spec, factory = registered[law_id]
        if spec.suite == "rig":  # judged on the hidden valid set, which never enters a process running plug-in code
            out.append(_not_run(law_id, manifest.name, device, "rig laws need the hidden-valid judge (no rig data)"))
            continue
        try:
            out.append(bench.run_law(spec, factory(), None, device=device, solver=manifest.name))
        except LawError as exc:
            out.append(LawResult(law_id, manifest.name, device, (), {"error": str(exc)[:500]}, {}, 0.0, False))
    return tuple(out)


def lint_atomics(module: Any) -> str | None:
    """None if clean; the failure text; or a ``not_run:`` reason when WP5's linter is not merged."""
    try:
        importlib.import_module("alloy.tools.lint_atomics")
    except ImportError:
        return "not_run: alloy.tools.lint_atomics (WP5) not merged"
    pkg = Path(module.__file__).parent
    res = subprocess.run([sys.executable, "-m", "alloy.tools.lint_atomics", str(pkg)], capture_output=True, text=True)
    return None if res.returncode == 0 else (res.stdout + res.stderr)[-2000:]


def attack(manifest: PluginManifest, device: str, cfg: SearchConfig, replay_runs: int) -> tuple[Finding, ...]:
    """B0 + B1 on every B0 scene of the plug-in; replay-confirmed findings only."""
    found: list[Finding] = []
    for name in manifest.bully_scenes:
        out = search_detailed(get_target(name), cfg, BullyLimits(), device, replay_runs=replay_runs)
        found.extend(out.findings)
    return tuple(found)


def decide(contract: ContractReport, lint: str | None, laws: tuple[LawResult, ...], findings: tuple[Finding, ...],
           native: bool) -> tuple[bool, tuple[str, ...]]:
    """Ship rule; every reason it does not ship is listed."""
    reasons = [f"contract {f}" for f in contract.failures]
    if native and lint is not None:
        reasons.append(f"lint_atomics {lint}")
    reasons += [f"law {r.law}: {r.verdict}" + (f" ({r.monitors.get('reason')})" if r.monitors.get("reason") else "")
                for r in laws if r.verdict != "pass"]
    reasons += [f"finding {f.severity.name} {f.monitor} = {f.value:.4g} > {f.threshold:.4g}" for f in findings
                if f.severity in (BullySeverity.S1, BullySeverity.S2)]
    if not laws:
        reasons.append("no laws in the manifest")
    return not reasons, tuple(reasons)


def evaluate_plugin(manifest: PluginManifest, device: str, *, cfg: SearchConfig | None = None,
                    replay_runs: int = 10) -> FactoryVerdict:
    """Contract checks + laws + B0/B1 (SPEC §3.11 signature, with an optional search config)."""
    module = importlib.import_module(manifest.module)
    cls = plugin_class(module, manifest.name)
    rep = ContractReport()
    static_checks(cls, rep)
    if not manifest.bully_scenes:
        rep.record("bully_scene", "the manifest names no B0 scene (§3.8 item 7)")
    else:
        dynamic_checks(cls, get_target(manifest.bully_scenes[0]), device, rep)
    lint = lint_atomics(module)
    laws = run_laws(manifest, device)
    findings = attack(manifest, device, cfg or ATTACK_CPU, replay_runs) if rep.ok else ()
    ship, reasons = decide(rep, lint, laws, findings, cls.determinism == DetCapability.NATIVE)
    log.info("evaluate %s: contract %s, %d laws, %d findings, ship %s", manifest.name, rep.ok, len(laws),
             len(findings), ship)
    return FactoryVerdict(manifest.name, rep.ok, laws, findings, ship, reasons)


# ---------------------------------------------------------------------------------------------- JSON round trip
def verdict_to_json(v: FactoryVerdict) -> dict[str, Any]:
    rec = dataclasses.asdict(v)
    rec["findings"] = [{**dataclasses.asdict(f), "severity": int(f.severity)} for f in v.findings]
    return json.loads(json.dumps(rec, default=str))


def _law_from_json(rec: dict[str, Any]) -> LawResult:
    verdicts = []
    for mv in rec["verdicts"]:
        real = Quantity(**mv["real"]) if mv.get("real") else None
        verdicts.append(MetricVerdict(**{**mv, "real": real, "sim": Quantity(**mv["sim"])}))
    return LawResult(**{**rec, "verdicts": tuple(verdicts)})


def verdict_from_json(rec: dict[str, Any]) -> FactoryVerdict:
    findings = tuple(Finding(**{**f, "severity": BullySeverity(int(f["severity"]))}) for f in rec["findings"])
    return FactoryVerdict(rec["plugin"], bool(rec["contract_ok"]), tuple(_law_from_json(r) for r in rec["law_results"]),
                          findings, bool(rec["ship"]), tuple(rec["reasons"]))
