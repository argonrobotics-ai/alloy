# SPDX-License-Identifier: Apache-2.0
"""Fetch objects of the library from the Hub into Alloy's asset cache, lazily: only files not already there.

Objects land in ``cache_root(<library>)`` (``$ALLOY_CACHE/imports/<library>/``, default
``~/.cache/alloy/imports/<library>/``) at the paths their cards expect, with the library's shared files (manifest,
shared material cards, looks) beside them, so Alloy's card libraries find them like an imported library. Every file is
checked against the index's sha256 before it is moved into place. Objects whose models use another object's files
(RoboCasa's composed cabinets use a door panel's and a handle's meshes) bring those objects too (``required``).

Several fetches can fill one cache at the same time (processes or threads): each downloads into a folder of its own
(``<hub_dir>/staging-<random>``) and moves a checked file into place in one step.

Example::

    >>> fetch(names=["ycb_025_mug"])                         # doctest: +SKIP
    {'objects': 1, 'downloaded': 6, 'bytes': 4318812, 'kept': 0, 'roots': ['~/.cache/alloy/imports/ycb_gso'], ...}
    >>> fetch(library="ycb_gso", category="ycb")             # doctest: +SKIP
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from alloy.assets.hub.layout import LIBRARY_FILES, library_dir, object_dir, sha256_file
from alloy.assets.hub.mjcf import named_files
from alloy.assets.hub.source import HubError, download, load_index, repo_id
from alloy.compose.fetch import cache_base, cache_root

log = logging.getLogger("alloy.assets.hub")

WORKERS = 8
#: a download folder untouched for this long was left by a fetch that was killed: the next fetch removes it
STALE_S = 2 * 86400.0


def hub_dir(repo: str | None = None) -> Path:
    """Where the index and in-flight downloads of ``repo`` live: ``<alloy cache>/hub/<owner>--<name>``."""
    return cache_base() / "hub" / (repo or repo_id()).replace("/", "--")


def select(rows: list[dict[str, Any]], names: list[str] | None = None, library: str | None = None,
           category: str | None = None) -> list[dict[str, Any]]:
    """Rows matching every filter given (names, library, category). Raises ``HubError`` naming close matches when a
    name, library or category is unknown. Example: ``select(rows, library="robocasa", category="blenders")``."""
    by_name = {r["name"]: r for r in rows}
    if names:
        unknown = [n for n in names if n not in by_name]
        if unknown:
            close = difflib.get_close_matches(unknown[0], list(by_name), n=3)
            raise HubError(f"not in the object library: {', '.join(unknown[:5])}",
                           f"check the name (close: {', '.join(close) or 'none'}); `alloy fetch --list` lists them",
                           field="names")
    out = [by_name[n] for n in names] if names else list(rows)
    for key, want in (("library", library), ("category", category)):
        if want is None:
            continue
        have = sorted({str(r.get(key)) for r in out})
        if want not in have:
            raise HubError(f"no {key} '{want}' among the selected objects",
                           f"use one of: {', '.join(have[:40])}", field=key)
        out = [r for r in out if r.get(key) == want]
    return out


def _valid(path: Path, digest: str) -> bool:
    return path.is_file() and sha256_file(path) == digest


def _get(job: tuple[str, Path, str], staging: Path, repo: str) -> int:
    """Download one dataset file, check its sha256, move it to its place; returns its size."""
    remote, dest, digest = job
    got = download(remote, staging, repo)
    if sha256_file(got) != digest:
        got.unlink(missing_ok=True)
        raise HubError(f"{remote} arrived with a different sha256 than the index lists",
                       "retry; if it repeats, the dataset revision and index disagree: report it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(got, dest)
    return dest.stat().st_size


def _jobs(rows: list[dict[str, Any]], prefix: str, base: Path) -> tuple[list[tuple[str, Path, str]], int]:
    """(remote, dest, sha256) of every listed file not already valid in ``base``; and how many were kept."""
    jobs, kept = [], 0
    for f in rows:
        dest = base / f["path"]
        if _valid(dest, f["sha256"]):
            kept += 1
        else:
            jobs.append((f"{prefix}/{f['file']}", dest, f["sha256"]))
    return jobs, kept


def _clear_stale(hub: Path) -> None:
    """Remove download folders in ``hub`` that a killed fetch left (nothing in them changed for ``STALE_S``)."""
    for d in hub.glob("staging*"):
        try:
            newest = max(p.stat().st_mtime for p in (d, *d.rglob("*")))
        except OSError:                                # another fetch is removing it right now
            continue
        if time.time() - newest > STALE_S:
            shutil.rmtree(d, ignore_errors=True)


def _run(jobs: list[tuple[str, Path, str]], repo: str, workers: int) -> int:
    """Download ``jobs`` through a folder only this call uses, so a fetch running at the same time (another process
    filling the same cache) cannot remove files in flight; returns the bytes moved into place."""
    if not jobs:
        return 0
    hub = hub_dir(repo)
    hub.mkdir(parents=True, exist_ok=True)
    _clear_stale(hub)
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=hub))
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return sum(pool.map(lambda j: _get(j, staging, repo), jobs))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def library_files(library: str, repo: str) -> list[tuple[str, Path, str]]:
    """Download jobs for a library's shared files (its ``files.json`` is always re-read)."""
    listing = download(f"{library_dir(library)}/{LIBRARY_FILES}", hub_dir(repo) / "lists", repo)
    doc = json.loads(listing.read_text(encoding="utf-8"))
    return _jobs(doc["files"], library_dir(library), cache_root(library))[0]


def _download(rows: list[dict[str, Any]], jobs: list[tuple[str, Path, str]], repo: str,
              workers: int) -> tuple[int, int, int]:
    """Download ``jobs`` and every file of ``rows`` not already valid in the cache: (files, bytes, kept)."""
    jobs, kept = list(jobs), 0
    for r in rows:
        js, k = _jobs(r["files"], object_dir(r["library"], r["name"]), cache_root(r["library"]))
        jobs += js
        kept += k
    log.info("fetching %d files of %d objects from %s (%d already in the cache)", len(jobs), len(rows), repo, kept)
    return len(jobs), _run(jobs, repo, workers), kept


def owners(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """(library, path under its cache root) -> the index row of the object that ships that file."""
    return {(r["library"], f["path"]): r for r in rows for f in r["files"]}


def required(rows: list[dict[str, Any]], owner: dict[tuple[str, str], dict[str, Any]]) -> tuple[
        list[dict[str, Any]], list[str]]:
    """The rows of other objects that ship files the cached MJCF models of ``rows`` name (a composed cabinet needs
    its door panel and handle: ``alloy.assets.hub.mjcf``), and the named files no object ships and the cache lacks.
    Example: ``required(rows, owners(load_index()))``."""
    have = {r["name"] for r in rows}
    more: dict[str, dict[str, Any]] = {}
    lost: set[str] = set()
    for r in rows:
        base = cache_root(r["library"])
        for rel in named_files(base, [f["path"] for f in r["files"]]):
            o = owner.get((r["library"], rel))
            if o is None and not (base / rel).is_file():
                lost.add(f"{r['library']}/{rel}")
            elif o is not None and o["name"] not in have:
                more[o["name"]] = o
    return [more[n] for n in sorted(more)], sorted(lost)


def fetch(names: list[str] | None = None, library: str | None = None, category: str | None = None,
          repo: str | None = None, workers: int = WORKERS) -> dict[str, Any]:
    """Put the selected objects (and their libraries' shared files) into the cache, with the objects their models
    use files of (``required``, e.g. a composed cabinet's door panel and handle); files already there with the right
    sha256 are kept. At least one filter is needed. Returns counts and the cache roots written. Example: see the
    module docstring."""
    if not (names or library or category):
        raise HubError("say what to fetch", "pass object names, library=<name> or category=<name>", field="names")
    repo = repo or repo_id()
    index = load_index(repo=repo)
    rows = select(index, names, library, category)
    libs = sorted({r["library"] for r in rows})
    n, size, kept = _download(rows, [j for lib in libs for j in library_files(lib, repo)], repo, workers)
    owner, deps = owners(index), []
    more, lost = required(rows, owner)
    while more:                                   # a fetched model can include another object's model in turn
        deps += more
        dn, ds, dk = _download(more, [], repo, workers)
        n, size, kept = n + dn, size + ds, kept + dk
        more, lost = required(rows + deps, owner)
    if lost:
        log.warning("%d file(s) the fetched models name are not in the object library %s: %s", len(lost), repo,
                    ", ".join(lost[:5]))
    return {"repo": repo, "objects": len(rows) + len(deps), "downloaded": n, "bytes": size, "kept": kept,
            "roots": [str(cache_root(lib)) for lib in libs], "names": [r["name"] for r in rows][:50],
            "required": [r["name"] for r in deps][:50], "unresolved": lost[:50]}


def list_objects(library: str | None = None, category: str | None = None, repo: str | None = None,
                 limit: int = 200) -> dict[str, Any]:
    """Libraries, categories and object names in the dataset (filtered), without downloading objects.
    Example: ``list_objects(library="objaverse")["categories"]``."""
    rows = select(load_index(repo=repo), None, library, category)
    cats: dict[str, int] = {}
    for r in rows:
        key = f"{r['library']}/{r.get('category')}"
        cats[key] = cats.get(key, 0) + 1
    return {"repo": repo or repo_id(), "objects": len(rows), "bytes": sum(r.get("size", 0) for r in rows),
            "libraries": sorted({r["library"] for r in rows}), "categories": dict(sorted(cats.items())),
            "names": [r["name"] for r in rows][:limit]}
