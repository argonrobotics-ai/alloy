# SPDX-License-Identifier: Apache-2.0
"""Download Objaverse 1.0 GLBs from the public Hugging Face mirror (stdlib only).

``object-paths.json.gz`` maps each uid to its GLB path (``glbs/000-023/<uid>.glb``); a GLB is fetched once, written
atomically and verified by size. Network errors are retried with backoff.

Example:
    >>> from alloy.importers.objaverse.fetch import glb_url
    >>> glb_url("glbs/000-023/abc.glb")
    'https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-023/abc.glb'
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from alloy.importers.objaverse.errors import ImportRejected

log = logging.getLogger("alloy.importers.objaverse")

BASE_URL = os.environ.get("ALLOY_OBJAVERSE_URL", "https://huggingface.co/datasets/allenai/objaverse/resolve/main")
PATHS_FILE = "object-paths.json.gz"


def glb_url(rel: str, base: str = BASE_URL) -> str:
    """URL of a GLB given its path relative to the dataset root."""
    return f"{base.rstrip('/')}/{rel.lstrip('/')}"


def _get(url: str, dest: Path, tries: int = 4, timeout: float = 120.0) -> None:
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = 2.0 ** attempt
            log.warning("fetch %s failed (%s); retry %d/%d in %.0f s", url, e, attempt + 1, tries, wait)
            time.sleep(wait)
    raise ImportRejected(dest.stem, "fetch", f"{url} failed {tries} times; check the network or ALLOY_OBJAVERSE_URL")


def object_paths(cache: Path) -> dict[str, str]:
    """uid -> GLB path, from ``object-paths.json.gz`` (downloaded into ``cache`` on first use)."""
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / PATHS_FILE
    if not local.exists():
        _get(glb_url(PATHS_FILE), local)
    with gzip.open(local, "rt") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_glb(uid: str, paths: dict[str, str], out_dir: Path) -> Path:
    """The GLB of ``uid`` under ``out_dir`` (downloaded if missing). Raises :class:`ImportRejected` (stage fetch).

    Example:
        >>> fetch_glb("not-a-uid", {}, Path("/tmp"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ImportRejected: not-a-uid: rejected at fetch: uid not in object-paths.json.gz
    """
    rel = paths.get(uid)
    if rel is None:
        raise ImportRejected(uid, "fetch", f"uid not in {PATHS_FILE}")
    dest = out_dir / f"{uid}.glb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    out_dir.mkdir(parents=True, exist_ok=True)
    _get(glb_url(rel), dest)
    return dest
