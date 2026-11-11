# SPDX-License-Identifier: Apache-2.0
"""Alloy's engine dependency pins: what is pinned, what is installed, and what upstream has released.

Every command prints one JSON document on stdout and exits 0 when everything matches, 1 when something does not
(the JSON says what to change), 2 on bad usage.

    python tools/deps/pins.py check                 # vendored newton / mujoco-warp, installed mujoco / warp-lang
    python tools/deps/pins.py releases              # newest PyPI releases of each pinned package (network)
    python tools/deps/pins.py upstream              # state of the upstream PRs that carry Alloy's Newton patches
    python tools/deps/pins.py requirements          # the alloy-engine wheel's core requirements

Newton and MuJoCo Warp are vendored (``kind: vendored``): their pinned commits ship inside Alloy as
``alloy/_vendor/newton`` and ``alloy/_vendor/mujoco_warp``, listed in ``alloy/_vendor/VENDORED.md``. The wheel
requires only PyPI packages (``kind: pypi``), so a user's own newton or mujoco-warp is never replaced.

Example (Python):
    >>> requirements(load_pins())[:2]
    ['mujoco==3.17.0', 'warp-lang==1.18.0']
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("alloy.deps")
PINS_PATH = Path(__file__).with_name("pins.json")
PYPI = "https://pypi.org/pypi/{name}/json"
GITHUB_PR = "https://api.github.com/repos/newton-physics/newton/pulls/{n}"

REPO = Path(__file__).resolve().parents[2]
VENDORED_MD = "_vendor/VENDORED.md"  # inside the alloy package
COMMON = ("numpy", "scipy", "pyyaml")


def load_pins(path: Path = PINS_PATH) -> dict[str, Any]:
    """Read and validate ``pins.json``.

    >>> sorted(d["name"] for d in load_pins()["deps"])[:2]
    ['mujoco', 'mujoco-warp']
    """
    data = json.loads(path.read_text())
    if data.get("schema") != "alloy.deps/1":
        raise ValueError(f"{path}: schema must be 'alloy.deps/1', got {data.get('schema')!r}")
    for dep in data["deps"]:
        missing = {"name", "module", "kind", "pin", "why", "released"} - dep.keys()
        if missing:
            raise ValueError(f"{path}: dep {dep.get('name', '?')} lacks {sorted(missing)}; add them")
    return data


def requirements(pins: dict[str, Any]) -> list[str]:
    """Core requirements of the alloy-engine wheel: each PyPI pin outside an extra, then ``COMMON``. Vendored pins
    are not requirements (they ship inside the wheel). Extras come from the tree's pyproject (``wheels.tree_meta``).

    >>> any(r.startswith(("newton", "mujoco-warp")) for r in requirements(load_pins()))
    False
    """
    pinned = [f"{d['name']}=={d['pin']}" for d in pins["deps"] if d["kind"] == "pypi" and not d.get("extra")]
    return pinned + list(COMMON)


def installed(dist: str) -> dict[str, str | None]:
    """Version and git commit (pip ``direct_url.json``) of an installed distribution; None when absent."""
    try:
        meta = md.distribution(dist)
    except md.PackageNotFoundError:
        return {"version": None, "commit": None}
    commit = None
    text = meta.read_text("direct_url.json")
    if text:
        commit = json.loads(text).get("vcs_info", {}).get("commit_id")
    return {"version": meta.version, "commit": commit}


def _matches(dep: dict[str, Any], got: dict[str, str | None]) -> bool:
    """A PyPI pin matches on version; a git pin on commit, or on a wheel version carrying its ``+g<pin8>`` label."""
    if got["version"] is None:
        return False
    if dep["kind"] == "pypi":
        return got["version"] == dep["pin"]
    return got["commit"] == dep["pin"] or got["version"].endswith(f"+g{dep['pin'][:8]}")


def alloy_root() -> Path:
    """The ``alloy`` package directory: the importable one, else this checkout's."""
    spec = importlib.util.find_spec("alloy")
    if spec is not None and spec.origin:
        return Path(spec.origin).parent
    return REPO / "alloy"


def vendored(dep: dict[str, Any], root: Path) -> dict[str, Any]:
    """Whether ``dep``'s vendored copy is under ``root`` (the alloy package): its ``__init__.py``, a LICENSE file in
    its directory, and a line of ``_vendor/VENDORED.md`` naming both its path and its pinned commit (8 characters),
    so a commit listed elsewhere in the file (an earlier patch) does not count.

    >>> import tempfile; r = Path(tempfile.mkdtemp())
    >>> vendored({"path": "alloy/_vendor/newton", "pin": "4f3b41ef8430"}, r)["match"]
    False
    """
    pkg = root.parent / dep["path"]
    notes = root / VENDORED_MD
    text = notes.read_text(errors="replace") if notes.is_file() else ""
    row = {"path": str(pkg), "package": (pkg / "__init__.py").is_file(),
           "licence": any(pkg.glob("LICENSE*")),
           "pin_in_vendored_md": any(dep["path"] in ln and dep["pin"][:8] in ln for ln in text.splitlines())}
    row["match"] = row["package"] and row["licence"] and row["pin_in_vendored_md"]
    return row


def check(pins: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Compare the installed engine packages with the pins (extras are skipped when not installed)."""
    rows, ok, root = [], True, alloy_root()
    for dep in pins["deps"]:
        if dep["kind"] == "vendored":
            row = {"name": dep["name"], "pin": dep["pin"], **vendored(dep, root),
                   "separate_install": installed(dep["name"])["version"]}  # a user's own copy: never Alloy's
            ok &= row["match"]
            rows.append(row if row["match"] else row | {"fix": _fix_hint(dep)})
            continue
        got = installed(dep["name"])
        if dep.get("extra") and got["version"] is None:
            continue
        match = _matches(dep, got)
        ok &= match
        row = {"name": dep["name"], "pin": dep["pin"], "installed": got, "match": match}
        if not match:
            row["fix"] = _fix_hint(dep)
        rows.append(row)
    return {"command": "check", "ok": ok, "deps": rows}, ok


def _fix_hint(dep: dict[str, Any]) -> str:
    """What to run to get the pinned build of ``dep``."""
    if dep["kind"] == "vendored":
        return (f"{dep['path']} must hold the pinned commit {dep['pin'][:8]} with its LICENSE, and alloy/{VENDORED_MD} "
                "must name it: re-vendor it as alloy/_vendor/VENDORED.md describes")
    if dep["kind"] == "git":
        return f"pip install 'mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@{dep['pin']}'"
    return f"pip install '{dep['name']}=={dep['pin']}'"


def _get_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "alloy-deps"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https hosts)
        return json.loads(resp.read())


def releases(pins: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Newest final and pre-release on PyPI per pinned package, and whether a release now covers the pin."""
    rows = []
    for dep in pins["deps"]:
        try:
            info = _get_json(PYPI.format(name=dep["name"]))
        except (urllib.error.URLError, TimeoutError) as exc:
            log.warning("PyPI lookup for %s failed: %s", dep["name"], exc)
            rows.append({"name": dep["name"], "error": str(exc)})
            continue
        versions = sorted(info["releases"], key=lambda v: _upload_time(info["releases"][v]))
        finals = [v for v in versions if v.replace(".", "").isdigit()]
        covered = dep["pin"] if dep["kind"] == "pypi" else (dep.get("in_release") or dep.get("base_in_release"))
        rows.append({"name": dep["name"], "pin": dep["pin"], "newest": versions[-1] if versions else None,
                     "newest_final": finals[-1] if finals else None, "covering_release": covered,
                     "covering_release_on_pypi": bool(covered) and covered in info["releases"]})
    return {"command": "releases", "deps": rows}, all("error" not in r for r in rows)


def _upload_time(files: list[dict[str, Any]]) -> str:
    return min((f["upload_time"] for f in files), default="")


def upstream(pins: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Live state of each upstream PR named in the Newton patches; ``changed`` lists PRs whose state moved."""
    newton = next(d for d in pins["deps"] if d["name"] == "newton")
    prs = sorted({p["pr"] for p in newton["patches"] if p["pr"]})
    rows, changed = [], []
    for n in prs:
        try:
            pr = _get_json(GITHUB_PR.format(n=n))
        except (urllib.error.URLError, TimeoutError) as exc:
            rows.append({"pr": n, "error": str(exc)})
            continue
        state = "merged" if pr.get("merged_at") else ("pr-open" if pr["state"] == "open" else "closed")
        recorded = {p["upstream"] for p in newton["patches"] if p["pr"] == n}
        rows.append({"pr": n, "title": pr["title"], "state": state, "recorded": sorted(recorded)})
        if recorded != {state}:
            changed.append(n)
    unposted = [p["commit"][:8] for p in newton["patches"] if p["upstream"] == "not-posted"]
    doc = {"command": "upstream", "prs": rows, "changed": changed, "not_posted": unposted}
    return doc, not changed and all("error" not in r for r in rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: one JSON document on stdout; exit 0 ok, 1 mismatch, 2 usage."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("check", "releases", "upstream", "requirements"))
    ap.add_argument("--pins", type=Path, default=PINS_PATH)
    args = ap.parse_args(argv)
    pins = load_pins(args.pins)
    if args.command == "requirements":
        doc, ok = {"command": "requirements", "requires": requirements(pins)}, True
    else:
        doc, ok = {"check": check, "releases": releases, "upstream": upstream}[args.command](pins)
    json.dump(doc, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
