# SPDX-License-Identifier: Apache-2.0
"""Sandboxed agent runs for the solver factory (SPEC §3.11 ``AgentRunner``).

The agent works in a scratch git worktree of the current commit and may write only under
``alloy/solvers/contrib/<name>/``. ``ShellAgentRunner`` launches any command line (``{workdir}``, ``{brief}``,
``{out}`` are substituted) in its own process group, with an environment scrubbed of the hidden valid set
(``ALLOY_HIDDEN_*``) and a wall-clock timeout that kills the whole group.

The agent's worktree is only a scratchpad: the loop evaluates a **pristine** worktree of the same commit into which
it copies the package alone (``stage_package``), so commits, index tricks, ignored files or stale bytecode the agent
leaves elsewhere cannot reach the evaluation. ``changes_outside`` (against the base commit, committed and ignored
files included) still reports every write outside the package as a sandbox violation.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from alloy.core.errors import FactoryError

log = logging.getLogger("alloy.factory")

CONTRIB = Path("alloy/solvers/contrib")
_HIDDEN_PREFIX = "ALLOY_HIDDEN"
_BYTECODE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise FactoryError(f"git {' '.join(args)} failed in {cwd}: {res.stderr.strip()}")
    return res.stdout


def repo_root(start: Path | None = None) -> Path:
    return Path(_git(start or Path(__file__).resolve().parent, "rev-parse", "--show-toplevel").strip())


def scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The process environment without the hidden valid set (agents and plug-ins never see it, §3.11)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_HIDDEN_PREFIX)}
    env.update(extra or {})
    return env


def scratch_worktree(repo: Path, base: str = "HEAD", parent: Path | None = None) -> Path:
    """A detached worktree of ``base`` in a fresh temporary directory (the directory is removed if the add fails)."""
    tmp = Path(tempfile.mkdtemp(prefix="alloy-factory-", dir=parent))
    work = tmp / "wt"
    try:
        _git(repo, "worktree", "add", "--detach", str(work), base)
    except FactoryError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log.info("factory: scratch worktree %s at %s", work, _git(work, "rev-parse", "--short", "HEAD").strip())
    return work


def remove_worktree(repo: Path, work: Path) -> None:
    """Remove our scratch worktree; prune (which also drops other stale entries) only if the remove failed."""
    res = subprocess.run(["git", "worktree", "remove", "--force", str(work)], cwd=repo, capture_output=True, text=True)
    shutil.rmtree(work.parent, ignore_errors=True)
    if res.returncode != 0:
        log.warning("factory: worktree remove failed (%s); pruning", res.stderr.strip())
        _git(repo, "worktree", "prune", check=False)


def ensure_contrib(work: Path) -> None:
    """``alloy/solvers/contrib/`` is a package in the scratch worktree (created there, never in the repo)."""
    pkg = work / CONTRIB
    pkg.mkdir(parents=True, exist_ok=True)
    init = pkg / "__init__.py"
    if not init.exists():
        init.write_text('"""Agent-written solver plug-ins produced by the factory loop (never hand-edited)."""\n')


def stage_package(src: Path, eval_tree: Path, rel: Path) -> None:
    """Replace the evaluation tree's package with a bytecode-free copy of the agent's (and nothing else)."""
    dest = eval_tree / rel
    shutil.rmtree(dest, ignore_errors=True)
    if src.is_dir():
        shutil.copytree(src, dest, ignore=_BYTECODE, symlinks=False)
    ensure_contrib(eval_tree)


def _inside(path: str, allowed: Path) -> bool:
    return path == str(allowed) or path.startswith(f"{allowed}/") or path == f"{CONTRIB}/__init__.py"


def changes_outside(work: Path, allowed: Path, base: str) -> list[str]:
    """Paths that differ from commit ``base`` outside ``allowed``: commits, staged and working changes, untracked
    and ignored files (bytecode caches excluded), and index flags that hide changes (skip-worktree, assume-unchanged).
    """
    names = set(_git(work, "diff", "--name-only", "-z", base).split("\0"))
    names |= set(_git(work, "ls-files", "-z", "--others", "--exclude-standard").split("\0"))
    names |= set(_git(work, "ls-files", "-z", "--others", "--ignored", "--exclude-standard").split("\0"))
    names |= {ln[2:] for ln in _git(work, "ls-files", "-v", "-z").split("\0") if ln[:1] in ("S", "h", "s")}
    names.discard("")
    return sorted(n for n in names if not _inside(n, allowed) and "__pycache__/" not in n and not n.endswith(".pyc"))


def revert_outside(work: Path, base: str, allowed: Path) -> None:
    """Return the agent's worktree to ``base`` outside ``allowed`` (keeps the package for the next round)."""
    keep = Path(tempfile.mkdtemp(prefix="alloy-factory-keep-"))
    src = work / allowed
    try:
        if src.is_dir():
            shutil.copytree(src, keep / "pkg", ignore=_BYTECODE, symlinks=False)
        flagged = [ln[2:] for ln in _git(work, "ls-files", "-v", "-z").split("\0") if ln[:1] in ("S", "h", "s")]
        if flagged:
            _git(work, "update-index", "--no-skip-worktree", "--no-assume-unchanged", "--", *flagged, check=False)
        _git(work, "reset", "--hard", base)
        _git(work, "clean", "-fdx")
        if (keep / "pkg").is_dir():
            shutil.copytree(keep / "pkg", src, dirs_exist_ok=True)
        ensure_contrib(work)
    finally:
        shutil.rmtree(keep, ignore_errors=True)
    log.warning("factory: agent worktree reset to %s outside %s", base[:12], allowed)


@dataclass
class ShellAgentRunner:
    """Runs an agent CLI, e.g. ``codex exec --cd {workdir} -f {brief}`` or ``claude -p ... --add-dir {out}``."""

    command_template: str

    def run(self, brief: Path, workdir: Path, out_package: Path, timeout_s: float) -> int:
        cmd = self.command_template.format(workdir=shlex.quote(str(workdir)), brief=shlex.quote(str(brief)),
                                           out=shlex.quote(str(out_package)))
        log.info("factory: agent run (timeout %.0f s): %s", timeout_s, cmd)
        proc = subprocess.Popen(shlex.split(cmd), cwd=workdir, env=scrubbed_env(), start_new_session=True)
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("factory: agent run timed out after %.0f s; killing its process group", timeout_s)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return 124
