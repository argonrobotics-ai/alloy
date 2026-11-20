# SPDX-License-Identifier: Apache-2.0
"""CI check on the oldest Python that pyproject.toml's requires-python allows (3.11; the other jobs run 3.12).

Python 3.12 accepts syntax that 3.11 rejects (PEP 701 f-strings: the same quotes nested inside the braces, a
backslash or a line break there), so a file can pass the 3.12 jobs and still raise SyntaxError for a 3.11 user.

- ``--compile``: from the checkout, every tracked .py file compiles under this interpreter, which must be the version
  requires-python names as its lower bound (any other version is an error, never a skip).
- ``--installed``: from outside the checkout, with ``python -I``, the installed Alloy imports every one of its
  modules (a module whose only missing piece is an optional third-party package counts as imported; not the
  vendored packages' examples and tests) and runs the README quick start.

Prints what it checked and every failure; exit 0 when the check passes, 1 otherwise.

Example::

    python -I .github/scripts/oldest_python_check.py --compile
    cd /tmp && <venv>/bin/python -I <checkout>/.github/scripts/oldest_python_check.py --installed \\
        --readme <checkout>/README.md
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import re
import subprocess
import sys
import traceback
from pathlib import Path


# Not imported: a package's command line (``__main__``), and the vendored packages' own examples and tests, which
# import upstream's test package (not shipped: alloy/_vendor/VENDORED.md).
UPSTREAM_EXTRAS = re.compile(r"alloy\._vendor\.[^.]+\.(examples|tests)(\.|$)")


def floor_version(pyproject: Path) -> str:
    """The X.Y lower bound of requires-python (``>=X.Y``)."""
    m = re.search(r'^\s*requires-python\s*=\s*"[^"]*>=\s*(\d+\.\d+)', pyproject.read_text(), re.M)
    if not m:
        raise SystemExit(f"no '>=X.Y' requires-python in {pyproject}")
    return m.group(1)


def check_compile(root: Path) -> list[str]:
    """Every tracked .py under ``root`` compiled by this interpreter; one line per file it rejects."""
    want, have = floor_version(root / "pyproject.toml"), "%d.%d" % sys.version_info[:2]
    if have != want:
        return [f"this is Python {have}; requires-python's lower bound is {want}: run this check on {want}"]
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "*.py"], capture_output=True, check=True)
    files = [p for p in out.stdout.decode().split("\0") if p and (root / p).is_file()]
    bad = []
    for p in files:
        try:
            compile((root / p).read_bytes(), p, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as e:
            bad.append(f"{p}:{getattr(e, 'lineno', 0) or 0}: {type(e).__name__}: {e}")
    print(f"Python {have}: compiled {len(files)} tracked .py files, {len(bad)} rejected")
    return bad


def check_installed(readme: Path) -> list[str]:
    """Import every module of the installed Alloy, then run the README quick start; one line per failure."""
    import alloy

    if Path(alloy.__file__).resolve().is_relative_to(readme.resolve().parent / "alloy"):
        return [f"imported the checkout's alloy ({alloy.__file__}), not the installed one: run with -I from outside"]
    walk = pkgutil.walk_packages(alloy.__path__, "alloy.", onerror=lambda _n: None)
    names = [m.name for m in walk if not m.name.endswith(".__main__") and not UPSTREAM_EXTRAS.match(m.name)]
    bad, argv = [], sys.argv
    sys.argv = [argv[0]]                                                  # a module that reads argv sees none
    for name in names:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] == "alloy":
                bad.append(f"{name}: {type(e).__name__}: {e}")
        except (Exception, SystemExit) as e:  # noqa: BLE001 - every import failure is reported, not only syntax
            bad.append(f"{name}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=-3)}")
    sys.argv = argv
    print(f"Python {'%d.%d' % sys.version_info[:2]}: imported {len(names) - len(bad)} of {len(names)} alloy modules")
    found = re.search(r"## Quick start.*?```python\n(.*?)```", readme.read_text(), re.S)
    if found is None:
        return [*bad, f"{readme} has no '## Quick start' section with a python block"]
    try:
        exec(compile(found.group(1), "README.md quick start", "exec"), {"__name__": "__main__"})
    except Exception as e:  # noqa: BLE001
        bad.append(f"README quick start: {type(e).__name__}: {e}")
    return bad


def main(argv: list[str] | None = None) -> int:
    """Run the chosen check and print its failures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile", action="store_true", help="every tracked .py compiles (run in the checkout)")
    mode.add_argument("--installed", action="store_true", help="the installed Alloy imports and runs")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="the checkout (--compile)")
    ap.add_argument("--readme", type=Path, help="the checkout's README.md (--installed)")
    args = ap.parse_args(argv)
    if args.installed and args.readme is None:
        ap.error("--installed needs --readme <checkout>/README.md")
    bad = check_compile(args.root) if args.compile else check_installed(args.readme)
    for line in bad:
        print("FAIL", line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
