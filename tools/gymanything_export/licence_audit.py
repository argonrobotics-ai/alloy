# SPDX-License-Identifier: Apache-2.0
"""Licence audit of the exporters: run each one with an import + file-open audit hook (Gym-Anything's env) and
summarise which third-party packages, upstream modules and data files it touched, with their licences.

Usage (Gym-Anything env python, from the repo root)::

    python tools/gymanything_export/licence_audit.py trace --repo <gym-anything> --sim wine_pouring --out audit/
    python tools/gymanything_export/licence_audit.py table --repo <gym-anything> --audit audit/ --out audit/table.json
    python tools/gymanything_export/licence_audit.py md --repo <gym-anything> --audit audit/ --out audit/table.md

``trace`` writes ``<out>/<sim>.json`` (modules imported, files opened for reading); ``table`` maps every module to
its distribution's licence metadata, flags upstream files whose header names another copyright holder, and lists
data files by origin (upstream repo, package data, user cache, other). One sim per process, so imports do not leak.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import re
import sys
import sysconfig
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HDR = re.compile(r"(SPDX-License-Identifier:\s*\S+|Copyright[^\n]{0,80}|Licensed under[^\n]{0,60})", re.I)
SKIP_PREFIX = ("/proc", "/dev", "/sys", "/etc", "/tmp", "/usr/share/zoneinfo", "/usr/lib/python3")


def trace(repo: Path, sim: str, out: Path) -> dict:
    """Run one exporter with the audit hook; returns and writes the trace."""
    opened: set[str] = set()

    def hook(event: str, args: tuple) -> None:
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            if not any(m in mode for m in "wax"):
                opened.add(os.path.abspath(os.fsdecode(args[0])))

    before = set(sys.modules)
    sys.path[:0] = [str(HERE), str(repo / "src"), str(repo / "benchmarks/roboworld/environments")]
    sys.addaudithook(hook)
    import run  # noqa: PLC0415 - after sys.path is set

    with tempfile.TemporaryDirectory() as td:
        res = run.run_one(sim, Path(td))
    mods = {n: os.path.abspath(f) for n in set(sys.modules) - before
            if (f := getattr(sys.modules.get(n), "__file__", None))}
    data = {"sim": sim, "written": len(res.get("written", [])), "crashed": res.get("crashed"), "modules": mods,
            "files": sorted(opened)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sim}.json").write_text(json.dumps(data, indent=0))
    return data


def _dist(top: str, dists: dict) -> tuple[str, str]:
    names = dists.get(top, [])
    if not names:
        return top, "unknown"
    meta = md.distribution(names[0]).metadata
    lic = meta.get("License-Expression") or ""
    if not lic:
        cls = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
        lic = "; ".join(cls) or ((meta.get("License") or "unknown").splitlines() or ["unknown"])[0][:60]
    return f"{names[0]} {md.distribution(names[0]).version}", lic


def _module_row(path: Path, repo: Path, stdlib: Path, dists: dict) -> tuple[str, str, str] | None:
    if path.is_relative_to(stdlib) and "site-packages" not in path.parts:
        return None
    if path.is_relative_to(HERE):
        return None
    if path.is_relative_to(repo):
        head = path.read_text(errors="replace")[:1500] if path.suffix == ".py" else ""
        other = "; ".join(sorted({m.strip() for m in HDR.findall(head)} - {""}))
        return "upstream", str(path.relative_to(repo)), other
    if "site-packages" in path.parts:
        name, lic = _dist(path.parts[path.parts.index("site-packages") + 1].split(".")[0], dists)
        return "package", name, lic
    return "other", str(path), ""


def _file_row(path: Path, repo: Path) -> tuple[str, str] | None:
    if path.suffix in (".py", ".pyc", ".so", ".pth") or "__pycache__" in path.parts or str(path).startswith(SKIP_PREFIX):
        return None
    if path.is_relative_to(HERE) or not path.is_file():      # our files; directories and import-hook probes
        return None
    if path.is_relative_to(repo):
        return "upstream data", str(path.relative_to(repo))
    if "site-packages" in path.parts:
        i = path.parts.index("site-packages")
        return None if path.parts[i + 1].endswith(".dist-info") else ("package data", "/".join(path.parts[i + 1:]))
    if path.is_relative_to(Path.home() / ".cache"):
        return "user cache", str(path.relative_to(Path.home()))
    return "other file", str(path)


def table(repo: Path, audit: Path) -> dict:
    """Per sim: packages with licences, upstream modules (and any non-upstream copyright header), data files."""
    stdlib, dists, out = Path(sysconfig.get_paths()["stdlib"]).resolve(), md.packages_distributions(), {}
    for f in sorted(p for p in audit.glob("*.json") if not p.name.startswith("_") and p.name != "table.json"):
        d = json.loads(f.read_text())
        rows = [r for p in d["modules"].values() if (r := _module_row(Path(p).resolve(), repo, stdlib, dists))]
        files: dict[str, list[str]] = {}
        for p in d["files"]:
            if (r := _file_row(Path(p), repo)):
                files.setdefault(r[0], []).append(r[1])
        out[d["sim"]] = {"written": d["written"], "crashed": d["crashed"],
                         "packages": dict(sorted((r[1], r[2]) for r in rows if r[0] == "package")),
                         "upstream_modules": sorted(r[1] for r in rows if r[0] == "upstream"),
                         "upstream_other_headers": {r[1]: r[2] for r in rows if r[0] == "upstream" and r[2]
                                                    and "cmu-l3" not in r[2]},
                         "files": {k: sorted(set(v)) for k, v in files.items()}}
    return out


COMMON = ("numpy", "scipy", "trimesh", "manifold3d", "shapely", "networkx", "pillow", "charset-normalizer")


def markdown(t: dict) -> str:
    """The licences table (one row per exporter) and the package list, as markdown."""
    pk: dict[str, str] = {}
    lines = ["| Exporter | Objects | Upstream modules run (MIT) | Other copyright headers | Data files read | "
             "Packages beyond the common set |", "|---|---|---|---|---|---|"]
    for sim, r in sorted(t.items()):
        pk.update(r["packages"])
        extra = sorted(k.split()[0] for k in r["packages"] if k.split()[0] not in COMMON)
        data = "; ".join(f"{k}: {len(v)} ({', '.join(sorted({Path(x).parent.name or x for x in v})[:3])})"
                         for k, v in r["files"].items()) or "none"
        hdr = "; ".join(f"{k}: {v}" for k, v in r["upstream_other_headers"].items()) or "none"
        lines.append(f"| {sim} | {r['written']} | {len(r['upstream_modules'])} | {hdr} | {data} | "
                     f"{', '.join(extra) or '-'} |")
    lines += ["", "| Package (as installed in the export env) | Licence (distribution metadata) |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(pk.items())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("trace", "table", "md"))
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--sim")
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "trace":
        res = trace(a.repo.resolve(), a.sim, a.out)
        print(json.dumps({"sim": a.sim, "written": res["written"], "crashed": res["crashed"]}))
        return 0 if not res["crashed"] else 1
    t = table(a.repo.resolve(), a.audit)
    a.out.write_text(markdown(t) if a.cmd == "md" else json.dumps(t, indent=1))
    print(json.dumps({a.cmd: str(a.out), "sims": len(t)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
