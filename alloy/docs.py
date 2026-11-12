# SPDX-License-Identifier: Apache-2.0
"""Where Alloy's docs, runnable examples and AGENTS.md are.

``python -m alloy.docs`` prints the docs folder, the examples folder and AGENTS.md (the quick start for AI agents);
``python -m alloy.docs agents`` prints AGENTS.md itself. A wheel or a plain ``pip install`` carries them inside the
package, at ``alloy/_data/docs``, ``alloy/_data/examples`` and ``alloy/_data/AGENTS.md`` (no top-level folders in
site-packages); a source checkout uses its own. Example::

    >>> import alloy.docs
    >>> (alloy.docs.docs_root() / "public" / "index.md").is_file(), alloy.docs.agents_md().name
    (True, 'AGENTS.md')
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
FIX = "install Alloy with pip (pip install alloy-engine)"
USAGE = "usage: python -m alloy.docs [agents]   (no argument: the paths; agents: print AGENTS.md)"


def _in_checkout(package: Path) -> bool:
    """True when ``package`` sits in a source checkout (pyproject.toml and library/ beside it)."""
    return (package.parent / "pyproject.toml").is_file() and (package.parent / "library").is_dir()


def _find(name: str, package: Path = _PACKAGE) -> Path:
    """``<checkout>/<name>`` in a source checkout, else the copy installed with the package (``alloy/_data/<name>``);
    a folder or a file."""
    if _in_checkout(package) and (package.parent / name).exists():
        return package.parent / name
    installed = package / "_data" / name
    if installed.exists():
        return installed
    raise FileNotFoundError(f"no {name} beside or inside the alloy package at {package}: {FIX}")


def _root(name: str, package: Path = _PACKAGE) -> Path:
    """The folder ``name`` (``docs``, ``examples``), checkout or installed copy."""
    found = _find(name, package)
    if not found.is_dir():
        raise FileNotFoundError(f"{found} is not a folder: {FIX}")
    return found


def docs_root() -> Path:
    """The docs folder: ``public/`` (the pages, from ``public/index.md``) and ``known_gaps/`` (a page per known
    limit, each linked from ``public/keep-in-mind.md``)."""
    return _root("docs")


def examples_root() -> Path:
    """The runnable examples folder (``basics/``, ``api/``, ``envs/``, ...)."""
    return _root("examples")


def agents_md() -> Path:
    """AGENTS.md: the whole quick start for AI agents, in one file."""
    return _find("AGENTS.md")


def main(argv: list[str] | None = None) -> int:
    """No argument: print ``docs <path>``, ``examples <path>`` and ``agents <path>``; ``agents``: print AGENTS.md.
    Exit 1 with the fix when something is missing, 2 on an unknown argument."""
    args = sys.argv[1:] if argv is None else argv
    if args not in ([], ["agents"]):
        print(USAGE, file=sys.stderr)
        return 2
    try:
        if args == ["agents"]:
            sys.stdout.write(agents_md().read_text(encoding="utf-8"))
            return 0
        print(f"docs      {docs_root()}")
        print(f"examples  {examples_root()}")
        print(f"agents    {agents_md()}")
    except FileNotFoundError as exc:
        print(f"alloy.docs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
