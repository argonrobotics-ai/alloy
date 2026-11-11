# SPDX-License-Identifier: Apache-2.0
"""Solver-factory types (SPEC §3.11; the loop, evaluator and runner are WP7)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from alloy.bench.types import LawResult
from alloy.bully.types import Finding


@runtime_checkable
class AgentRunner(Protocol):                  # launches the plug-in-writing agent (B2-style, sandboxed)
    command_template: str                     # e.g. "codex exec --cd {workdir} -f {brief}"; {workdir}, {brief}, {out}

    def run(self, brief: Path, workdir: Path, out_package: Path, timeout_s: float) -> int:
        """workdir = a scratch git worktree; the agent may write only under out_package = alloy/solvers/contrib/<name>/."""
        ...


@dataclass(frozen=True)
class PluginManifest:
    name: str
    module: str
    routes: tuple[str, ...]
    laws: tuple[str, ...]
    bully_scenes: tuple[str, ...]
    author: str


@dataclass(frozen=True)
class FactoryVerdict:
    plugin: str
    contract_ok: bool
    law_results: tuple[LawResult, ...]
    findings: tuple[Finding, ...]
    ship: bool
    reasons: tuple[str, ...]
