# SPDX-License-Identifier: Apache-2.0
"""Download and unpack one upstream archive safely (no path traversal, no links, size-capped).

Archives are untrusted data: members are extracted only under the target directory, links and devices are skipped,
and only the members a job names are written.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from alloy.importers.common.errors import ImporterError

log = logging.getLogger("alloy.importers")

MAX_ARCHIVE_BYTES = 2 << 30          # 2 GiB: no YCB or GSO archive is near this
USER_AGENT = "alloy-importer/1 (+https://alloy.argonrobotics.ai)"


def download(url: str, dest: Path, retries: int = 4, timeout: float = 120.0) -> Path:
    """Fetch ``url`` to ``dest`` with retries (exponential back-off); returns ``dest``.

    Example::

        download("https://fuel.gazebosim.org/1.0/GoogleResearch/models/X/1/X.zip", Path("/tmp/x.zip"))
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh, 1 << 20)
            if dest.stat().st_size > MAX_ARCHIVE_BYTES:
                raise ImporterError(url, "download", f"archive over {MAX_ARCHIVE_BYTES} bytes; refusing it")
            return dest
        except (OSError, ValueError) as exc:
            log.warning("download %s failed (%s), attempt %d/%d", url, exc, attempt + 1, retries)
            time.sleep(2.0 * 2**attempt)
    raise ImporterError(url, "download", f"failed after {retries} attempts; check the URL or the network")


def _safe_rel(name: str) -> PurePosixPath | None:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        return None
    return rel


def _wanted(rel: PurePosixPath, prefixes: tuple[str, ...]) -> bool:
    return not prefixes or any(str(rel) == p or str(rel).startswith(p.rstrip("/") + "/") for p in prefixes)


def _extract_tar(path: Path, dest: Path, prefixes: tuple[str, ...]) -> int:
    n = 0
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            rel = _safe_rel(m.name)
            if rel is None or not m.isfile() or not _wanted(rel, prefixes):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(target, "wb") as fh:
                shutil.copyfileobj(src, fh)
            n += 1
    return n


def _extract_zip(path: Path, dest: Path, prefixes: tuple[str, ...]) -> int:
    n = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            rel = _safe_rel(info.filename)
            if rel is None or info.is_dir() or not _wanted(rel, prefixes):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as fh:
                shutil.copyfileobj(src, fh)
            n += 1
    return n


def is_archive(path: Path) -> bool:
    """True for a zip or tar file (what ``extract`` unpacks); False for a bare mesh such as ``.glb``.

    Example::

        is_archive(Path("model.glb"))   # False
    """
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def extract(archive: Path, dest: Path, prefixes: tuple[str, ...] = ()) -> int:
    """Extract regular files under ``prefixes`` (all if empty) from a .tgz/.tar/.zip into ``dest``; returns count.

    Example::

        extract(Path("025_mug.tgz"), Path("src"), ("025_mug/google_16k",))
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        n = _extract_zip(Path(archive), dest, prefixes)
    elif tarfile.is_tarfile(archive):
        n = _extract_tar(Path(archive), dest, prefixes)
    else:
        raise ImporterError(str(archive), "extract", "not a zip or tar archive; check the download")
    if n == 0:
        raise ImporterError(str(archive), "extract", f"no files under {list(prefixes)}; check the member paths")
    return n
