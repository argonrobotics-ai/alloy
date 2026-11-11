# SPDX-License-Identifier: Apache-2.0
"""MuJoCo ``.msh`` mesh files and the sha256 manifest of vendored files.

The ``.msh`` format is MuJoCo's legacy binary mesh: four int32 counts (vertices, normals, texcoords, faces) followed
by float32 vertex, normal and texcoord arrays and int32 faces. Both MuJoCo and Newton's MJCF importer read it
natively (newton@ac7f7830:newton/_src/utils/import_mjcf.py:918-922), it stores float32 vertices exactly, and it
needs no mesh library, so the converted scenes reproduce the compiled mesh data bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("alloy.tasks")

MANIFEST_NAME = "SHA256SUMS.json"


@dataclass(frozen=True)
class MeshData:
    """Vertices [n, 3] float32 and triangle faces [m, 3] int32 of one mesh."""

    vertices: np.ndarray
    faces: np.ndarray


def msh_bytes(mesh: MeshData) -> bytes:
    """``mesh`` encoded as a MuJoCo ``.msh`` file (no normals, no texcoords)."""
    vert = np.ascontiguousarray(mesh.vertices, dtype="<f4").reshape(-1, 3)
    face = np.ascontiguousarray(mesh.faces, dtype="<i4").reshape(-1, 3)
    header = np.array([len(vert), 0, 0, len(face)], dtype="<i4")
    return header.tobytes() + vert.tobytes() + face.tobytes()


def write_msh(path: Path, mesh: MeshData) -> None:
    """Write ``mesh`` as a MuJoCo ``.msh`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(msh_bytes(mesh))


def store_msh(store: Path, name: str, mesh: MeshData) -> str:
    """Content-addressed write: ``store/<name>-<sha12>.msh`` (kept if present); returns the file name.

    The converted scenes share most meshes (the arms): 1,453 mesh files of 15.1 MB over five traces hold 3.95 MB of
    distinct data, so one store under ``library/robots/meshes`` is vendored once."""
    data = msh_bytes(mesh)
    fname = f"{name}-{hashlib.sha256(data).hexdigest()[:12]}.msh"
    path = store / fname
    if not path.is_file():
        store.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return fname


def read_msh(path: Path) -> MeshData:
    """Read a ``.msh`` file written by :func:`write_msh` (normals and texcoords are skipped)."""
    raw = path.read_bytes()
    nv, nn, nt, nf = np.frombuffer(raw[:16], dtype="<i4")
    off = 16
    vert = np.frombuffer(raw[off : off + 12 * nv], dtype="<f4").reshape(-1, 3)
    off += 12 * nv + 12 * nn + 8 * nt
    face = np.frombuffer(raw[off : off + 12 * nf], dtype="<i4").reshape(-1, 3)
    return MeshData(vertices=vert.copy(), faces=face.copy())


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path) -> dict[str, str]:
    """Hash every file under ``root`` (except the manifest) into ``root/SHA256SUMS.json``; returns the map."""
    sums = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST_NAME
    }
    (root / MANIFEST_NAME).write_text(json.dumps(sums, indent=1, sort_keys=True) + "\n")
    log.info("manifest %s: %d files", root, len(sums))
    return sums


def check_manifest(root: Path) -> list[str]:
    """Return a list of problems (missing, changed or unlisted files); empty means every hash matches."""
    sums = json.loads((root / MANIFEST_NAME).read_text())
    problems = []
    for rel, want in sorted(sums.items()):
        path = root / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != want:
            problems.append(f"sha256 mismatch: {rel}")
    listed = set(sums)
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and p.name != MANIFEST_NAME and rel not in listed:
            problems.append(f"unlisted: {rel}")
    return problems
