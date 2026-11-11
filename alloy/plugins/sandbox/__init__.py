# SPDX-License-Identifier: Apache-2.0
"""Run the plug-in harness on untrusted plug-in code inside an OS sandbox.

Layers (each reported in the ``sandbox`` check of the report):

1. A child process in its own session, with a wall-clock timeout that kills the whole process group, and an
   environment scrubbed of credentials (``*TOKEN*``, ``*SECRET*``, ``*KEY*``, cloud and API variables, the factory's
   hidden set); ``HOME`` and ``TMPDIR`` point into the sandbox.
2. Linux user + network namespaces (``unshare -rn``) when allowed: no network at all.
3. Landlock (Linux 5.13+): the plug-in may read only Python, the installed packages, Alloy and system directories,
   and write only its sandbox folder and the sandbox kernel cache; with ABI 4 (Linux 6.7+) TCP is denied too. The
   rest of ``$HOME`` (keys, cloud credentials, other repositories) is unreadable.
4. Resource limits: CPU seconds, file size, open files, no core dumps, and address space on CPU runs.

What it does not do: it does not make the report tamper-proof (the plug-in runs in the same process as the
harness and could fake its own report; review code before trusting a pass), and it does not limit GPU memory or
time beyond the timeout. Plug-ins share one sandbox kernel cache (``~/.cache/alloy/plugin-sandbox``), never the
trusted one.

Example::

    from alloy.plugins.sandbox import sandboxed_check
    report = sandboxed_check("my_plugin/plugin.py", quick=True)
    print(report.summary())
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alloy.plugins.harness.report import CheckReport, CheckResult

log = logging.getLogger("alloy.plugins")

SECRET_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "COOKIE", "SESSION")
SECRET_PREFIXES = ("AWS_", "GOOGLE_", "GCLOUD", "CLOUDSDK_", "AZURE_", "ANTHROPIC", "OPENAI", "GITHUB", "GH_",
                   "HF_", "HUGGING", "SSH_", "GPG_", "ALLOY_HIDDEN", "CLAUDE")
SYSTEM_READ = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/sys", "/dev", "/opt", "/run")
# Newton and MuJoCo Warp are inside alloy (alloy/_vendor): a newton / mujoco_warp of the user's own is not granted.
PACKAGES = ("alloy", "warp", "mujoco", "numpy", "scipy", "trimesh")


@dataclass
class SandboxPolicy:
    """Limits and allow-lists. ``extra_read`` adds read-only paths (data your plug-in needs)."""

    timeout_s: float = 1800.0
    memory_bytes: int | None = 24 * 2**30       # address space on CPU runs (None: no limit; GPU runs skip it)
    max_file_bytes: int = 2 * 2**30
    open_files: int = 4096
    network: bool = False
    extra_read: tuple[str, ...] = ()
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "alloy" / "plugin-sandbox")
    seed_cache: bool = True                    # first use: copy the trusted Warp kernel cache (read once)


def scrubbed_env(home: Path, tmp: Path, warp_cache: Path) -> dict[str, str]:
    """The parent's environment without credentials, with HOME/TMPDIR/WARP_CACHE_PATH inside the sandbox."""
    env = {k: v for k, v in os.environ.items()
           if not any(m in k.upper() for m in SECRET_MARKERS) and not k.upper().startswith(SECRET_PREFIXES)}
    env.update(HOME=str(home), TMPDIR=str(tmp), WARP_CACHE_PATH=str(warp_cache), PYTHONDONTWRITEBYTECODE="1",
               XDG_CACHE_HOME=str(home / ".cache"), PYTHONPATH=child_pythonpath())
    return env


def child_pythonpath() -> str:
    """This process's import path as a PYTHONPATH for the child, so the child imports the same ``alloy`` (and the
    same packages) however the parent found them: installed, on PYTHONPATH, or from the folder it was started in
    (``""`` on ``sys.path``; the child starts in its sandbox folder, where a tree run with no install found no
    ``alloy`` at all). Example: ``str(Path(alloy.__file__).parents[1]) in child_pythonpath().split(os.pathsep)``."""
    dirs: list[str] = []
    for entry in sys.path:
        full = str(Path(entry or os.getcwd()).resolve())
        if os.path.isdir(full) and full not in dirs:
            dirs.append(full)
    return os.pathsep.join(dirs)


@functools.cache
def netns_prefix() -> tuple[str, ...]:
    """``unshare`` arguments for a private user + network namespace, or () if the host does not allow it."""
    exe = shutil.which("unshare")
    if exe is None:
        return ()
    cmd = (exe, "--user", "--map-root-user", "--net")
    try:
        ok = subprocess.run([*cmd, "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return cmd if ok else ()


def _package_roots() -> list[str]:
    import importlib.util  # noqa: PLC0415

    roots = []
    for name in PACKAGES:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            roots.append(str(Path(spec.origin).resolve().parent.parent))
    return roots


def read_paths(target: str, extra: tuple[str, ...]) -> list[str]:
    """What the plug-in may read: system dirs, Python, packages, sys.path (never $HOME itself) and the target."""
    home = Path.home().resolve()
    paths = [*SYSTEM_READ, sys.prefix, sys.base_prefix, *_package_roots(), *extra]
    paths += [p for p in sys.path if p and Path(p).is_dir()]
    if Path(target).is_dir():                            # a package or project folder: itself, not its parent
        paths.append(str(Path(target).resolve()))
    elif Path(target).exists():
        paths.append(str(Path(target).resolve().parent))
    keep = []
    for p in paths:
        rp = Path(p).resolve()
        if home.is_relative_to(rp):
            continue                                     # never the home directory or one of its parents
        if str(rp) not in keep:
            keep.append(str(rp))
    return keep


def _seed_cache(policy: SandboxPolicy) -> Path:
    cache = policy.cache_dir / "warp"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        trusted = Path.home() / ".cache" / "warp"
        if policy.seed_cache and trusted.is_dir():
            log.info("sandbox: seeding the sandbox kernel cache from %s (once)", trusted)
            shutil.copytree(trusted, cache, symlinks=False, ignore_dangling_symlinks=True)
        cache.mkdir(parents=True, exist_ok=True)
    return cache


def _policy_file(work: Path, target: str, config: dict, opts: dict[str, Any], policy: SandboxPolicy,
                 cache: Path) -> Path:
    pol = {"target": str(Path(target).resolve()) if Path(target).exists() else target, "config": config,
           **opts, "out": str(work / "out" / "report.json"), "network": policy.network,
           "read": read_paths(target, policy.extra_read), "write": [str(work), str(cache), "/dev"],
           "cpu_s": 4 * int(policy.timeout_s) + 60, "max_file_bytes": policy.max_file_bytes,
           "open_files": policy.open_files,
           "memory_bytes": policy.memory_bytes if str(opts["device"]).startswith("cpu") else None}
    path = work / "policy.json"
    path.write_text(json.dumps(pol, indent=2))
    return path


def _run_child(cmd: list[str], env: dict[str, str], work: Path, timeout_s: float) -> tuple[int, str]:
    err = work / "stderr.txt"
    with err.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = 124
    return code, err.read_text()[-3000:]


def sandboxed_check(target: str, *, config: dict | None = None, device: str = "cpu", quick: bool = False,
                    speed: bool = True, timeout_s: float | None = None,
                    policy: SandboxPolicy | None = None) -> CheckReport:
    """``alloy.plugins.check`` in a confined child process (module docstring); never raises for plug-in faults.
    Example: ``sandboxed_check("my_plugin/plugin.py", config={"k": 2.0}).passed``."""
    policy = policy or SandboxPolicy()
    timeout = float(timeout_s or policy.timeout_s)
    cache = _seed_cache(policy)
    work = Path(tempfile.mkdtemp(prefix="alloy-plugin-sandbox-"))
    try:
        for sub in ("home", "tmp", "out"):
            (work / sub).mkdir()
        opts = {"device": device, "quick": quick, "speed": speed}
        pol = _policy_file(work, target, dict(config or {}), opts, policy, cache)
        ns = () if policy.network else netns_prefix()
        cmd = [*ns, sys.executable, "-m", "alloy.plugins.sandbox.child", str(pol)]
        code, tail = _run_child(cmd, scrubbed_env(work / "home", work / "tmp", cache), work, timeout)
        return _collect(work / "out" / "report.json", target, device, code, tail, ns, timeout)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _collect(out: Path, target: str, device: str, code: int, tail: str, ns: tuple[str, ...],
             timeout: float) -> CheckReport:
    data = json.loads(out.read_text()) if out.exists() else {}
    if "plugin" in data:
        report = CheckReport.from_dict(data)
        info = data.get("sandbox", {})
        ll = info.get("landlock") or {}
        detail = (f"landlock ABI {ll.get('abi', 0)} (fs allow-list, tcp {'denied' if ll.get('tcp_denied') else 'open'})"
                  f", {'user+net namespace' if ns else 'no namespace'}, limits {sorted(info.get('limits', {}))}")
        report.add(CheckResult("sandbox", "sandbox", code in (0, 1), float(code), None, detail))
        return report
    why = "timed out" if code == 124 else f"exited {code}"
    report = CheckReport(Path(target).stem, "?", target, device)
    report.add(CheckResult("sandbox", "sandbox", False, float(code), None, f"sandboxed check {why} ({timeout:.0f} s)",
                           (data.get("error") or tail)[-2000:]))
    return report


__all__ = ["SandboxPolicy", "child_pythonpath", "netns_prefix", "read_paths", "sandboxed_check", "scrubbed_env"]
