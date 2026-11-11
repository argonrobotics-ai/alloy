# SPDX-License-Identifier: Apache-2.0
"""Sparse local mirror of a bundle that lives in GCS (`render gs://.../bundle`).

Everything except the frame arrays is copied. Each frames/*.npy above SMALL_BYTES gets only its .npy header and is then
extended to its full size as a sparse file, so local checks (shapes, frame count, labels) work while the mirror uses
almost no disk. The farm never ships the mirror: its VMs pull the real bundle from GCS (FarmJob.bundle_gcs).
"""
from __future__ import annotations

import logging
import os
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

SMALL_BYTES = 64 << 10  # frame arrays below this are copied whole (t.npy); per-world curves are not small
HEAD_BYTES = 4096
MIRRORS = Path.home() / ".cache" / "alloy_render" / "mirrors"

Run = Callable[..., subprocess.CompletedProcess]


def npy_header_len(head: bytes) -> int:
    """Bytes of the .npy preamble + header at the start of `head` (format versions 1-3)."""
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    if head[6] == 1:
        return 10 + struct.unpack("<H", head[8:10])[0]
    return 12 + struct.unpack("<I", head[8:12])[0]


def list_sizes(uri: str, run: Run = subprocess.run) -> dict[str, int]:
    """Object path (relative to uri) -> size for every object under uri."""
    res = run(["gcloud", "storage", "ls", "-l", "-r", f"{uri}/**"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cannot list {uri}: {res.stderr.strip()[-300:]}")
    out = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1].startswith(uri + "/") and parts[0].isdigit():
            out[parts[-1][len(uri) + 1:]] = int(parts[0])
    return out


def _sparse_frame(uri: str, rel: str, size: int, dest: Path, run: Run) -> None:
    path = dest / rel
    if path.is_file() and path.stat().st_size == size:
        return
    res = run(["gcloud", "storage", "cat", "-r", f"0-{HEAD_BYTES - 1}", f"{uri}/{rel}"], capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"cannot read the header of {uri}/{rel}")
    head = res.stdout[:npy_header_len(res.stdout)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(head)
    os.truncate(path, size)


def mirror_gcs_bundle(uri: str, dest: Path | None = None, run: Run = subprocess.run, workers: int = 32) -> Path:
    """Make (or refresh) the sparse mirror of gs:// bundle `uri`; returns its directory."""
    uri = uri.rstrip("/")
    dest = dest or MIRRORS / uri.rsplit("/", 1)[-1]
    dest.mkdir(parents=True, exist_ok=True)
    sizes = list_sizes(uri, run)
    big = {r: n for r, n in sizes.items() if r.startswith("frames/") and r.endswith(".npy") and n > SMALL_BYTES}
    res = run(["gcloud", "storage", "rsync", "-r", "-x", r"^frames/.*\.npy$", uri, str(dest)],
              capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cannot copy {uri}: {res.stderr.strip()[-300:]}")
    small_frames = sorted(r for r in sizes if r.startswith("frames/") and r.endswith(".npy") and r not in big)
    if small_frames:
        (dest / "frames").mkdir(exist_ok=True)
        run(["gcloud", "storage", "cp", "-I", str(dest / "frames") + "/"], input="\n".join(
            f"{uri}/{r}" for r in small_frames), capture_output=True, text=True, check=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda kv: _sparse_frame(uri, kv[0], kv[1], dest, run), big.items()))
    log.info("mirrored %s to %s (%d files, %d sparse frame arrays)", uri, dest, len(sizes), len(big))
    return dest
