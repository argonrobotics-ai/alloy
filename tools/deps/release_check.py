# SPDX-License-Identifier: Apache-2.0
"""Release regression test: build the release wheel, install it in a clean venv, run the installed-package check.

    python tools/deps/release_check.py --tree <release tree> [--json-out report.json] [--no-coexist] [--no-source]

1. The wheel (``wheels.py``): ``alloy-engine`` from ``--tree``, with Newton and MuJoCo Warp vendored inside it; it
   must carry every staged file, every package-data glob and the pinned vendored copies.
2. A new venv in a temp dir (``uv`` when present, else ``python3 -m venv``). ``alloy-engine`` is installed from the
   wheel and its dependencies (mujoco, warp-lang, numpy, ...) from PyPI.
3. ``wheel_smoke.py`` runs with ``python -I`` from an empty folder, with ``PYTHONPATH`` and every ``ALLOY_*``
   variable removed and a fresh ``ALLOY_CACHE_DIR``, so the C route compiles from the installed ``step.c``. It runs
   a robot on the CPU route, one ``alloy.camera`` render and one tactile read, and checks that Alloy imported only its
   vendored engine.
4. Coexistence (skip with ``--no-coexist``): a second venv gets a user's own ``newton`` and ``mujoco-warp`` from PyPI
   first (``USER_ENGINE``), then ``alloy-engine``. Both must still import at their versions from their own
   site-packages directories, and ``wheel_smoke.py`` must pass there too. ``coexist_probe.py`` then runs both copies
   in one process, once per import order: every submodule imported, an Alloy robot + cloth step (on the CPU's Warp
   route, so MuJoCo Warp launches) and a cloth step of the user's ``newton``, and no Warp module or kernel shared
   between the copies (SCR-WP0-COEXIST).
5. Source install (skip with ``--no-source``): ``git archive`` of the tree's HEAD, as a fresh clone has it, installed
   with a plain ``pip install <dir>`` (not editable: the build runs, as on a user's machine) into a third venv;
   ``wheel_smoke.py`` must pass there too, AGENTS.md's quick start included, so that install carries the card
   library (SCR-OSS2-2).

``tools/release/make_release.py`` runs this as a gate step after the scan; a failure blocks the release commit.
Prints one JSON document; exit 0 when the wheel built, installed and passed every check, 1 otherwise.

Example (Python):
    >>> clean_env({"PYTHONPATH": "/src", "ALLOY_ASSETS": "/a", "HOME": "/h"}, Path("/c"))["ALLOY_CACHE_DIR"]
    '/c'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wheels import DIST_NAME, BuildError, build_all  # noqa: E402

log = logging.getLogger("alloy.deps")
SMOKE_TIMEOUT_S = 1800
DROP_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "PYTHONSAFEPATH")
# A user's own engine packages, installed before alloy-engine in the coexistence venv: the PyPI releases nearest
# Alloy's vendored commits (the same MuJoCo Warp code, Newton without Alloy's commits), so every kernel that could
# be shared is. Plain newton: its [sim] extra adds only mujoco and mujoco-warp, at the versions installed here.
USER_ENGINE = ("newton==1.8.0rc1", "mujoco-warp==3.17.0")
# coexist_probe.py's row, as the report keeps it when the probe passed (the whole row when it failed).
PROBE_FIELDS = ("ok", "warp_modules", "taken_at_import", "taken_by_launch", "unshared_vendored", "borrowed", "s")


def build_wheels(tree: Path, wheel_dir: Path) -> dict[str, Any]:
    """The alloy wheel from ``tree`` in ``wheel_dir`` (the vendored engine inside it); the ``wheels.py`` report."""
    args = argparse.Namespace(out=wheel_dir, tree=tree, work=None, name=DIST_NAME, assets=None, no_imports=False)
    return build_all(args)


def make_venv(venv: Path, python: str) -> Path:
    """A new venv at ``venv``; returns its interpreter."""
    if shutil.which("uv"):
        _run(["uv", "venv", "-q", "--python", python, str(venv)])
    else:
        _run([python, "-m", "venv", str(venv)])
    return venv / "bin" / "python"


def install(py: Path, reqs: list[str], wheel_dir: Path | None = None) -> float:
    """Install ``reqs`` into ``py``'s venv (``wheel_dir`` as extra find-links, else PyPI); the seconds taken."""
    links = ["--find-links", str(wheel_dir)] if wheel_dir else []
    t0 = time.perf_counter()
    if shutil.which("uv"):
        _run(["uv", "pip", "install", "-q", "--python", str(py), *links, *reqs])
    else:
        _run([str(py), "-m", "pip", "install", "-q", *links, *reqs])
    return round(time.perf_counter() - t0, 1)


PROBE = """import importlib.metadata as md, json, sys
import newton, mujoco_warp
import alloy, alloy._vendor.newton as vn, alloy._vendor.mujoco_warp as vm
print(json.dumps({"newton": [md.version("newton"), newton.__file__], "mujoco-warp": [md.version("mujoco-warp"),
    mujoco_warp.__file__], "vendored": [vn.__file__, vm.__file__], "same_module": newton is vn or mujoco_warp is vm}))
"""


def engine_versions(py: Path) -> dict[str, str | None]:
    """Installed newton / mujoco-warp / mujoco / warp-lang versions in ``py``'s venv (None when absent)."""
    code = ("import importlib.metadata as md, json\n"
            "def v(d):\n    try: return md.version(d)\n    except md.PackageNotFoundError: return None\n"
            "print(json.dumps({d: v(d) for d in ('newton', 'mujoco-warp', 'mujoco', 'warp-lang')}))")
    return json.loads(subprocess.run([str(py), "-I", "-c", code], capture_output=True, text=True, check=True).stdout)


def git_archive(tree: Path, dest: Path) -> Path:
    """``tree``'s committed HEAD unpacked in ``dest`` (what a clone checks out; the build writes ``build/`` there)."""
    if subprocess.run(["git", "-C", str(tree), "rev-parse", "--git-dir"], capture_output=True, check=False).returncode:
        raise BuildError(f"{tree} is not a git checkout, so there is no clone to install from: pass a release tree "
                         "made by tools/release/make_release.py, or a clone, or use --no-source")
    dest.mkdir(parents=True)
    with subprocess.Popen(["git", "-C", str(tree), "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE) as git:
        subprocess.run(["tar", "-x", "-C", str(dest)], stdin=git.stdout, check=True)
    if git.returncode:
        raise BuildError(f"git archive of {tree} failed ({git.returncode}): commit the tree first")
    return dest


def missing_files(src: Path, site: Path) -> list[str]:
    """Files of ``src``'s ``alloy`` package (dotfiles aside) that are not under ``site``'s ``alloy``.

    >>> import tempfile; d = Path(tempfile.mkdtemp())
    >>> for f in ("src/alloy/a.py", "src/alloy/w.json", "src/alloy/.gitignore", "site/alloy/a.py"):
    ...     (d / f).parent.mkdir(parents=True, exist_ok=True); _ = (d / f).write_text("")
    >>> missing_files(d / "src", d / "site")
    ['alloy/w.json']
    """
    def files(root: Path) -> set[str]:
        return {p.relative_to(root).as_posix() for p in (root / "alloy").rglob("*")
                if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)}

    return sorted(files(src) - files(site))


def purelib(py: Path) -> Path:
    """The site-packages directory of ``py``'s venv."""
    code = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    return Path(subprocess.run([str(py), "-I", "-c", code], capture_output=True, text=True, check=True).stdout.strip())


def source_install(work: Path, python: str, tree: Path) -> dict[str, Any]:
    """``pip install`` (not editable) of a git archive of ``tree`` into a new venv: every file of its ``alloy``
    package must be installed, and the smoke must pass there (what a user gets from ``git clone`` + ``pip install .``)."""
    src = git_archive(tree, work / "source")
    py = make_venv(work / "venv-source", python)
    install_s = install(py, [str(src)])
    missing = missing_files(src, purelib(py))
    smoke = run_smoke(py, work, "run-source")
    failed = [r["check"] for r in smoke.get("checks", []) if not r["ok"]]
    ok = bool(smoke.get("ok")) and not missing
    row = {"ok": ok, "install_s": install_s, "smoke_ok": bool(smoke.get("ok")), "failed": failed,
           "missing_files": len(missing), "missing": missing[:25]}
    if not ok:
        row["fix"] = ("a plain `pip install .` of the tree leaves out files the release wheel ships (see missing and "
                      "failed): pyproject.toml's package data must take all of alloy/ (\"alloy\" = [\"**/*\"]), and "
                      "setup.py's build_py step must add library/ at alloy/_data/library")
        row["smoke"] = smoke
    return row


def coexist(work: Path, python: str, wheel_dir: Path, version: str) -> dict[str, Any]:
    """A user's own newton and mujoco-warp, then alloy-engine, in one venv: theirs must import unchanged, from their
    own directories, beside Alloy's vendored copies, and the smoke must pass with them installed."""
    py = make_venv(work / "venv-coexist", python)
    install(py, list(USER_ENGINE))
    before = engine_versions(py)
    install_s = install(py, [f"{DIST_NAME}=={version}"], wheel_dir)
    after = engine_versions(py)
    probe = subprocess.run([str(py), "-I", "-c", PROBE], capture_output=True, text=True, cwd=work,
                           env=clean_env(dict(os.environ), work / "alloy-cache-coexist"), check=False)
    seen = json.loads(probe.stdout) if probe.returncode == 0 else {"error": probe.stderr[-1500:]}
    kept = all(before[d] == after[d] for d in ("newton", "mujoco-warp"))
    smoke = run_smoke(py, work, "run-coexist")
    launch = {f"{first}_first": run_tool(py, work / "launch-coexist", "coexist_probe.py", ["--first", first],
                                         work / "alloy-cache-coexist", f"{first}-first.json")
              for first in ("alloy", "user")}
    apart = all(r.get("ok") for r in launch.values())
    ok = kept and "error" not in seen and not seen.get("same_module") and bool(smoke.get("ok")) and apart
    row = {"ok": ok, "user_engine": list(USER_ENGINE), "before": before, "after": after, "user_packages_kept": kept,
           "imports": seen, "install_s": install_s, "smoke_ok": bool(smoke.get("ok")),
           "one_process": {k: {f: r.get(f) for f in PROBE_FIELDS if f in r} for k, r in launch.items()}}
    if not ok:
        row["fix"] = ("installing alloy-engine changed or shadowed the user's newton / mujoco-warp: alloy-engine must "
                      "not require them, and Alloy must import only alloy._vendor.* (see the smoke's vendored row)"
                      if apart else next(r.get("fix") or r.get("error") for r in launch.values() if not r.get("ok")))
        row["smoke"], row["one_process"] = smoke, launch
    return row


def clean_env(env: dict[str, str], cache: Path) -> dict[str, str]:
    """``env`` without ``PYTHONPATH``-like and ``ALLOY_*`` variables, with ``ALLOY_CACHE_DIR`` set to ``cache``."""
    out = {k: v for k, v in env.items() if k not in DROP_ENV and not k.startswith("ALLOY_")}
    out["ALLOY_CACHE_DIR"] = str(cache)
    return out


def run_smoke(py: Path, work: Path, name: str = "run") -> dict[str, Any]:
    """``wheel_smoke.py`` under ``py -I`` from an empty folder; its JSON report (``ok`` False on a crash)."""
    return run_tool(py, work / name, "wheel_smoke.py", [], work / f"alloy-cache-{name}")


def run_tool(py: Path, run_dir: Path, script: str, args: list[str], cache: Path,
             out_name: str = "result.json") -> dict[str, Any]:
    """``tools/deps/<script>`` copied into ``run_dir`` (an otherwise empty folder, so no checkout is importable) and
    run under ``py -I`` with ``args`` and ``--out``, in ``clean_env``; its JSON report (``ok`` False on a crash)."""
    run_dir.mkdir(exist_ok=True)
    shutil.copy2(HERE / script, run_dir / script)
    out = run_dir / out_name
    proc = subprocess.run([str(py), "-I", str(run_dir / script), *args, "--out", str(out)], cwd=run_dir,
                          capture_output=True, text=True, env=clean_env(dict(os.environ), cache),
                          timeout=SMOKE_TIMEOUT_S, check=False)
    if out.is_file():
        return json.loads(out.read_text())
    return {"ok": False, "error": f"{script} exited {proc.returncode} without a report", "stderr": proc.stderr[-2000:]}


def _run(cmd: list[str]) -> None:
    log.info("run: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise BuildError(f"{' '.join(cmd)} failed ({proc.returncode}):\n{(proc.stderr or proc.stdout)[-1500:]}")


def check(tree: Path, python: str, work: Path, with_coexist: bool = True, with_source: bool = True) -> dict[str, Any]:
    """Build, install and smoke-test in ``work`` (then the coexistence and source-install venvs); the JSON report."""
    t0 = time.perf_counter()
    wheel_dir = work / "wheels"
    built = build_wheels(tree, wheel_dir)
    py = make_venv(work / "venv", python)
    install_s = install(py, [f"{DIST_NAME}=={built['version']}"], wheel_dir)
    smoke = run_smoke(py, work)
    both = coexist(work, python, wheel_dir, built["version"]) if with_coexist else {"ok": None, "skipped": True}
    source = source_install(work, python, tree) if with_source else {"ok": None, "skipped": True}
    ok = bool(smoke.get("ok")) and both["ok"] is not False and source["ok"] is not False
    return {"ok": ok, "tree": str(tree), "version": built["version"],
            "wheels": [{k: w[k] for k in ("file", "bytes", "sha256")} for w in built["wheels"]],
            "vendored": built["vendored"], "install_s": install_s, "smoke": smoke, "coexist": both, "source": source,
            "seconds": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry: one JSON document on stdout (and in ``--json-out``); exit 0 passed, 1 failed."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, default=HERE.parents[1], help="release (or source) tree to build")
    ap.add_argument("--python", default="3.12", help="venv interpreter: a uv version request or a path")
    ap.add_argument("--keep", type=Path, help="work here and keep it (default: a temp dir, removed on exit)")
    ap.add_argument("--json-out", type=Path, help="also write the JSON report here")
    ap.add_argument("--no-coexist", action="store_true", help="skip the venv with a user's own newton / mujoco-warp")
    ap.add_argument("--no-source", action="store_true", help="skip the plain `pip install .` of a git archive")
    args = ap.parse_args(argv)
    try:
        if args.keep:
            args.keep.mkdir(parents=True, exist_ok=False)
            doc = check(args.tree.resolve(), args.python, args.keep.resolve(), not args.no_coexist, not args.no_source)
        else:
            with tempfile.TemporaryDirectory(prefix="alloy-release-check-") as tmp:
                doc = check(args.tree.resolve(), args.python, Path(tmp), not args.no_coexist, not args.no_source)
    except (BuildError, OSError, subprocess.SubprocessError, ValueError) as exc:
        doc = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    text = json.dumps(doc, indent=1)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    sys.exit(main())
