# SPDX-License-Identifier: Apache-2.0
"""Offline wheelhouse: the full wheel closure of one requirement, with ``requirements.txt`` and ``SHA256SUMS``.

Run it under the target interpreter and platform (e.g. inside a clean ``python:3.12-slim`` container) so pip picks
the same wheels a user's box would:

    python tools/deps/closure.py --links <dir with the alloy and engine wheels> --out wheels "alloy-engine==<version>"
    python tools/deps/closure.py --verify wheels         # re-check SHA256SUMS and requirements.txt

The output directory then holds every wheel the requirement needs (wheels only; a package without a wheel for this
platform is an error naming it), ``requirements.txt`` (``name==version`` for each wheel) and ``SHA256SUMS``
(``sha256sum -c`` format), so ``pip install --no-index --find-links wheels -r wheels/requirements.txt`` installs
offline and no index package can shadow the pinned engine. Prints one JSON document; exit 0 ok, 1 on a failure.

Example (Python):
    >>> wheel_pin("absl_py-2.5.1-py3-none-any.whl")
    'absl-py==2.5.1'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("alloy.deps")


class ClosureError(RuntimeError):
    """A closure could not be built or does not verify; the message says what to change."""


def wheel_pin(filename: str) -> str:
    """``name==version`` for a wheel file name (PEP 503 normalised name, local labels kept).

    >>> wheel_pin("mujoco_warp-3.16.0.post1+g86e2fae0-py3-none-any.whl")
    'mujoco-warp==3.16.0.post1+g86e2fae0'
    """
    if not filename.endswith(".whl") or filename.count("-") < 4:
        raise ClosureError(f"{filename}: not a wheel file name (name-version-python-abi-platform.whl)")
    name, version = filename.split("-")[:2]
    return f"{re.sub(r'[-_.]+', '-', name).lower()}=={version}"


def sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(requirement: str, links: Path, dest: Path) -> list[Path]:
    """``pip download`` the requirement's closure (wheels only) into ``dest``, preferring wheels from ``links``."""
    cmd = [sys.executable, "-m", "pip", "download", "--disable-pip-version-check", "--only-binary", ":all:",
           "--find-links", str(links), "--dest", str(dest), requirement]
    log.info("run: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise ClosureError(f"pip download failed for {requirement}; a package may lack a wheel for this platform:\n"
                           f"{res.stderr[-2000:]}")
    wheels = sorted(dest.glob("*.whl"))
    others = sorted(p.name for p in dest.iterdir() if p.suffix != ".whl")
    if others:
        raise ClosureError(f"non-wheel downloads {others}; build wheels for them first")
    return wheels


def write_closure(wheels: list[Path], out: Path) -> dict[str, Any]:
    """Copy ``wheels`` into ``out`` (dropping stale ``*.whl`` there), write requirements.txt and SHA256SUMS."""
    out.mkdir(parents=True, exist_ok=True)
    keep = {w.name for w in wheels}
    for stale in out.glob("*.whl"):
        if stale.name not in keep:
            log.info("removing stale %s", stale.name)
            stale.unlink()
    rows = []
    for w in wheels:
        shutil.copy2(w, out / w.name)
        rows.append({"file": w.name, "pin": wheel_pin(w.name), "bytes": w.stat().st_size, "sha256": sha256(w)})
    (out / "requirements.txt").write_text("".join(f"{r['pin']}\n" for r in rows))
    (out / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['file']}\n" for r in rows))
    return {"wheels": rows, "total_mb": round(sum(r["bytes"] for r in rows) / 1e6, 1)}


def verify(out: Path) -> dict[str, Any]:
    """Check SHA256SUMS against the files and that requirements.txt names exactly the wheels present."""
    sums = out / "SHA256SUMS"
    if not sums.is_file():
        raise ClosureError(f"{sums} missing; rebuild with closure.py --out {out}")
    bad = []
    listed = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name] = digest
        if not (out / name).is_file() or sha256(out / name) != digest:
            bad.append(name)
    present = {p.name for p in out.glob("*.whl")}
    reqs = set((out / "requirements.txt").read_text().split())
    problems = {"sha_mismatch_or_missing": bad, "unlisted_wheels": sorted(present - listed.keys()),
                "requirements_mismatch": sorted(reqs ^ {wheel_pin(n) for n in listed})}
    return {"ok": not any(problems.values()), "wheels": len(listed), **problems}


def main(argv: list[str] | None = None) -> int:
    """CLI entry: one JSON document on stdout; exit 0 ok, 1 failure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("requirement", nargs="?", help='e.g. "alloy-engine==0.1.0+g1234abcd"')
    ap.add_argument("--links", type=Path, help="directory holding the alloy and engine wheels")
    ap.add_argument("--out", type=Path, help="wheelhouse directory to fill")
    ap.add_argument("--verify", type=Path, help="only verify an existing wheelhouse")
    args = ap.parse_args(argv)
    try:
        if args.verify:
            doc = verify(args.verify)
        elif args.requirement and args.links and args.out:
            with tempfile.TemporaryDirectory() as tmp:
                doc = write_closure(download(args.requirement, args.links, Path(tmp)), args.out)
            doc = {"ok": True, "requirement": args.requirement, "python": platform.python_version(),
                   "machine": platform.machine(), **doc, "verify": verify(args.out)}
        else:
            ap.error("give a requirement with --links and --out, or --verify DIR")
    except ClosureError as exc:
        doc = {"ok": False, "error": str(exc)}
    json.dump(doc, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    sys.exit(main())
