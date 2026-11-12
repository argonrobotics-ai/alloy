# SPDX-License-Identifier: Apache-2.0
"""Build hook (SCR-OSS2-2, coordinator ruling 2026-11-08 08:50Z): a non-editable build (``pip install .`` from a clone,
the wheel) ships the card library ``library/`` inside the package at ``alloy/_data/library``, where
``alloy.cards.library.default_root()`` finds it when there is no source checkout. Editable installs read
``<repo>/library`` directly, so they skip the copy. The docs and examples go beside it, at ``alloy/_data/docs`` and
``alloy/_data/examples``, where ``python -m alloy.docs`` finds them (TRAIN-AUDIT fix 1, 2026-11-11). Everything else
about the build is in pyproject.toml."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

log = logging.getLogger("alloy.build")
ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "library"
# Staged beside the library when the source tree has them (a clone, the sdist): examples/ whole, AGENTS.md, and of
# docs/ ONLY the public set (an allow-list: never docs/notes, oss, research, ENV.md or SPEC.md; 2026-11-11). Same
# lists in tools/deps/stage.py, tools/release/stage_release.py and MANIFEST.in (tests/core/test_packaging.py).
# public/ pages, the known_gaps/ pages keep-in-mind.md links, and guide/ as reference (coordinator 17:5xZ; physics
# ruling (A): AGENTS.md and the code's errors point at it).
DOCS_PUBLIC = ("public", "known_gaps", "guide")
DATA_FILES = ("AGENTS.md", "llms.txt")   # alloy.docs agents prints AGENTS.md; llms.txt beside it (GEO P3)
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def stage_public_docs(root: Path, data: Path) -> list[str]:
    """Copy ``root``'s examples/, AGENTS.md and the public part of docs/ into ``data`` (``alloy/_data``); the names
    copied."""
    pairs = [(root / n, data / n) for n in ("examples", *DATA_FILES)]
    pairs += [(root / "docs" / n, data / "docs" / n) for n in DOCS_PUBLIC]
    done = []
    for src, dst in pairs:
        if not src.exists():
            continue
        if dst.is_dir():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)
        done.append(src.relative_to(root).as_posix())
    return done


class BuildPyWithLibrary(build_py):
    """build_py that also stages the card library into the built package."""

    def run(self) -> None:
        super().run()
        if getattr(self, "editable_mode", False):
            return
        if not LIBRARY.is_dir():
            raise FileNotFoundError(f"{LIBRARY}: the card library is missing from the source tree (an sdist must graft "
                                    "it: MANIFEST.in)")
        data = Path(self.build_lib) / "alloy" / "_data"
        if (data / "library").exists():
            shutil.rmtree(data / "library")
        shutil.copytree(LIBRARY, data / "library", ignore=IGNORE)
        log.info("staged the card library and %s at %s", stage_public_docs(ROOT, data), data)


setup(cmdclass={"build_py": BuildPyWithLibrary})
