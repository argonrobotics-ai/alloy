# SPDX-License-Identifier: Apache-2.0
"""``factory_loop``: create -> evaluate -> attack -> score, for agent-written solver plug-ins (SPEC §3.11).

Each round: write the round brief (spec + contract + the previous verdict's reasons and findings), let the runner
write ``alloy/solvers/contrib/<name>/`` in its scratch worktree, report and reset anything it changed elsewhere,
copy the package alone into a pristine evaluation worktree of the same commit, and evaluate it **in a subprocess**
(``python -B -m alloy.factory evaluate``), so agent code never runs in the factory's process. The manifest must name
the package being evaluated. The loop stops when a round ships or after ``rounds``; a shipped package is copied to
``export_to`` if given, and promoting it onto a real route stays a coordinator decision.

Limit: the evaluation process imports the plug-in, so a plug-in written to attack the evaluator could forge its
verdict. The loop re-derives ``ship`` from the verdict's parts, which stops a flipped flag but not forged parts;
isolating the judge from plug-in code needs OS-level sandboxing and is not done here.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from alloy.bully.types import BullySeverity, SearchConfig
from alloy.core.errors import FactoryError
from alloy.factory.evaluate import ATTACK_CPU, verdict_from_json
from alloy.factory.runner import (CONTRIB, changes_outside, ensure_contrib, remove_worktree, repo_root,
                                  revert_outside, scratch_worktree, scrubbed_env, stage_package)
from alloy.factory.types import AgentRunner, FactoryVerdict

log = logging.getLogger("alloy.factory")

CONTRACT_SUMMARY = """\
## Contract (SPEC §3.8) the evaluator checks
- Start from `alloy/factory/template/` (copy it to your package and change `PLUGIN_NAME` to `{name}`). Use relative
  imports inside the package. Write nothing outside `alloy/solvers/contrib/{name}/`: only that directory is copied
  into the evaluation, and writes elsewhere are reported as sandbox violations and undone.
- `manifest.json` must say `"name": "{name}"` and `"module": "alloy.solvers.contrib.{name}"`.
- `SolverBase` + `CouplingInterface` subclass with `plugin_name`, `routes`, `determinism`, `register_custom_attributes`
  (attributes only in the `{name}:` namespace), `emit`, `alloy_state_arrays`, `alloy_state_worlds`,
  `alloy_capacity_report`, `alloy_stable_dt`, `alloy_energy`; a `SolverCapability` registered on import; route
  `contrib:{name}`.
- No allocation and no host readback inside `step`; per-world `reset(state, world_mask)`; no float atomics if
  NATIVE (`python -m alloy.tools.lint_atomics`); `alloy_state_arrays` must be enough to repeat a step bitwise.
- `manifest.json` lists your laws (registered in `laws.py`) and B0 scenes (registered in `scenes.py`); the bench
  then attacks every B0 scene with the bully (B0 stress + B1 search). Any replayed S1/S2 finding blocks shipping.
"""


@dataclass(frozen=True)
class LoopConfig:
    """Where and how hard the loop evaluates."""

    device: str = "cpu"
    timeout_s: float = 3600.0  # per agent run
    eval_timeout_s: float = 1800.0  # per evaluation subprocess
    attack: SearchConfig = ATTACK_CPU
    replay_runs: int = 10
    keep_worktree: bool = False
    repo: Path | None = None
    export_to: Path | None = None
    scratch_parent: Path | None = None  # where scratch worktrees are made (default: the system temp directory)
    python: str = field(default_factory=lambda: sys.executable)


def round_brief(spec: str, name: str, previous: FactoryVerdict | None, round_no: int) -> str:
    """The brief for one round: the task spec, the contract, and what the last evaluation said."""
    parts = [f"# Solver factory, plug-in `{name}`, round {round_no}", "", spec.strip(), "",
             CONTRACT_SUMMARY.format(name=name)]
    if previous is not None:
        parts += ["## Previous evaluation", f"- contract ok: {previous.contract_ok}; ship: {previous.ship}"]
        parts += [f"- {r}" for r in previous.reasons] or ["- no reasons recorded"]
        parts += [f"- finding {f.severity.name} {f.monitor} = {f.value:.4g} (threshold {f.threshold:.4g}), "
                  f"first tick {f.first_step}" for f in previous.findings]
    return "\n".join(parts) + "\n"


def _failed(name: str, *reasons: str) -> FactoryVerdict:
    return FactoryVerdict(name, False, (), (), False, tuple(reasons))


def manifest_problems(manifest: Path, name: str) -> list[str]:
    """The manifest must exist and name the package under evaluation (not, say, the trusted template)."""
    if not manifest.is_file():
        return [f"no manifest.json under {manifest.parent}"]
    try:
        rec = json.loads(manifest.read_text())
    except json.JSONDecodeError as exc:
        return [f"manifest.json is not JSON: {exc}"]
    module = f"{CONTRIB.as_posix().replace('/', '.')}.{name}"
    return [f"manifest {k} is {rec.get(k)!r}, expected {v!r}" for k, v in (("name", name), ("module", module))
            if rec.get(k) != v]


def _attack_args(a: SearchConfig) -> list[str]:
    return ["--worlds", str(a.worlds), "--generations", str(a.generations), "--horizon", str(a.horizon_s),
            "--knots", str(a.knots), "--sigma0", str(a.sigma0), "--seed", str(a.seed), "--budget-h", str(a.budget_gpu_h)]


def evaluate_in_subprocess(tree: Path, rel: Path, name: str, cfg: LoopConfig) -> FactoryVerdict:
    """``python -B -m alloy.factory evaluate`` on the evaluation tree; a crash becomes a failed verdict."""
    manifest = tree / rel / "manifest.json"
    problems = manifest_problems(manifest, name)
    if problems:
        return _failed(name, *problems)
    out = tree.parent / f"verdict-{name}.json"
    out.unlink(missing_ok=True)
    cmd = [cfg.python, "-B", "-m", "alloy.factory", "evaluate", "--manifest", str(manifest), "--device", cfg.device,
           "--out", str(out), "--replay-runs", str(cfg.replay_runs), *_attack_args(cfg.attack)]
    env = scrubbed_env({"PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        res = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=cfg.eval_timeout_s)
    except subprocess.TimeoutExpired:
        return _failed(name, f"evaluation timed out after {cfg.eval_timeout_s:.0f} s")
    if res.returncode != 0 or not out.is_file():
        return _failed(name, f"evaluation crashed (exit {res.returncode}): {res.stderr[-1500:]}")
    return _consistent(verdict_from_json(json.loads(out.read_text())), name)


def _consistent(v: FactoryVerdict, name: str) -> FactoryVerdict:
    """Re-derive ``ship`` from the verdict's parts; a mismatch (or a different plug-in name) never ships."""
    serious = any(f.severity in (BullySeverity.S1, BullySeverity.S2) for f in v.findings)
    derived = (v.plugin == name and v.contract_ok and bool(v.law_results) and not v.reasons and not serious
               and all(r.verdict == "pass" for r in v.law_results))
    if v.ship and not derived:
        return _with_reasons(v, ["verdict inconsistent: ship claimed but its parts do not support it"])
    return v


def _with_reasons(v: FactoryVerdict, extra: list[str]) -> FactoryVerdict:
    if not extra:
        return v
    return FactoryVerdict(v.plugin, v.contract_ok, v.law_results, v.findings, False, (*v.reasons, *extra))


@dataclass(frozen=True)
class _Run:
    """One factory run: the agent's worktree, the pristine evaluation worktree and their base commit."""

    spec: str
    name: str
    runner: AgentRunner
    work: Path
    evaluation: Path
    base: str
    cfg: LoopConfig


def _round(run: _Run, k: int, previous: FactoryVerdict | None) -> FactoryVerdict:
    spec, name, runner, cfg = run.spec, run.name, run.runner, run.cfg
    work, evaluation, base = run.work, run.evaluation, run.base
    rel = CONTRIB / name
    brief = work.parent / f"brief-round{k}.md"
    brief.write_text(round_brief(spec, name, previous, k))
    code = runner.run(brief, work, work / rel, cfg.timeout_s)
    outside = changes_outside(work, rel, base)
    if outside:
        revert_outside(work, base, rel)
    stage_package(work / rel, evaluation, rel)
    extra = [f"sandbox: wrote outside {rel}: {outside[:10]}"] if outside else []
    extra += [f"agent exited {code}"] if code != 0 else []
    return _with_reasons(evaluate_in_subprocess(evaluation, rel, name, cfg), extra)


def factory_loop(spec_brief: Path, runner: AgentRunner, name: str, rounds: int = 3, *,
                 cfg: LoopConfig | None = None) -> list[FactoryVerdict]:
    """Run up to ``rounds`` create-evaluate rounds; returns one verdict per round."""
    if not name.isidentifier() or name.startswith("_"):
        raise FactoryError(f"plug-in name must be a Python identifier, got {name!r}")
    cfg = cfg or LoopConfig()
    spec = Path(spec_brief).read_text()
    repo = cfg.repo or repo_root()
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    made: list[Path] = []
    verdicts: list[FactoryVerdict] = []
    try:
        made.append(scratch_worktree(repo, base, cfg.scratch_parent))
        made.append(scratch_worktree(repo, base, cfg.scratch_parent))
        ensure_contrib(made[0])
        run = _Run(spec, name, runner, made[0], made[1], base, cfg)
        for k in range(1, rounds + 1):
            verdict = _round(run, k, verdicts[-1] if verdicts else None)
            verdicts.append(verdict)
            log.info("factory %s round %d: contract %s, %d findings, ship %s", name, k, verdict.contract_ok,
                     len(verdict.findings), verdict.ship)
            if verdict.ship:
                _export(made[1] / CONTRIB / name, cfg.export_to)
                break
    finally:
        if not cfg.keep_worktree:
            for tree in made:
                remove_worktree(repo, tree)
    return verdicts


def _export(package: Path, dest: Path | None) -> None:
    if dest is None:
        return
    target = dest / package.name
    shutil.copytree(package, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    log.info("factory: shipped package copied to %s", target)
