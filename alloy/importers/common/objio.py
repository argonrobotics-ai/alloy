# SPDX-License-Identifier: Apache-2.0
"""Deterministic OBJ text for importer outputs: the watertight ``physics.obj`` and the ``<stem>.hulls.obj`` sidecar.

Sidecar format (read by the rigid emitter for ``collision: decompose``): first line ``# source_sha256: <sha256 of <stem>.obj>``, then one ``o hull_<k>`` block per
convex hull, k = 0..n-1. Floats are written with 7 significant digits (0.1 µm at the 0.1 m scale), so the same
arrays always give the same bytes and the same sha256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from alloy.importers.common.errors import ImporterError

SHA_TAG = "# source_sha256:"
HULL_PREFIX = "hull_"


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data``.

    Example::

        sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    """
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    """Hex sha256 of a file's bytes.

    Example::

        sha256_path(Path("physics.obj"))
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vertex_lines(v: np.ndarray) -> list[str]:
    return [f"v {x:.7g} {y:.7g} {z:.7g}" for x, y, z in np.asarray(v, dtype=np.float64).tolist()]


def _face_lines(f: np.ndarray, base: int) -> list[str]:
    return [f"f {a + base} {b + base} {c + base}" for a, b, c in np.asarray(f, dtype=np.int64).tolist()]


def mesh_obj_text(vertices: np.ndarray, faces: np.ndarray, header: str = "") -> str:
    """OBJ text of one triangle mesh with optional ``#`` header lines.

    Example::

        text = mesh_obj_text(np.eye(3), np.array([[0, 1, 2]]), header="unit triangle")
    """
    lines = [f"# {h}" for h in header.splitlines() if h]
    lines += _vertex_lines(vertices) + _face_lines(faces, 1)
    return "\n".join(lines) + "\n"


def hulls_obj_text(hulls: list[tuple[np.ndarray, np.ndarray]], source_sha256: str, header: str = "") -> str:
    """Sidecar OBJ text: the source-sha line, then ``o hull_<k>`` blocks with 1-based global vertex indices.

    Example::

        text = hulls_obj_text([(verts, faces)], source_sha256=sha256_path(Path("physics.obj")))
    """
    lines = [f"{SHA_TAG} {source_sha256}"] + [f"# {h}" for h in header.splitlines() if h]
    base = 1
    for k, (v, f) in enumerate(hulls):
        lines.append(f"o {HULL_PREFIX}{k}")
        lines += _vertex_lines(v) + _face_lines(f, base)
        base += len(v)
    return "\n".join(lines) + "\n"


def read_hulls_obj(path: Path) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """``(source_sha256, [(vertices, faces), ...])`` of a sidecar; faces re-indexed per hull.

    Example::

        sha, hulls = read_hulls_obj(Path("physics.hulls.obj"))
    """
    text = Path(path).read_text(encoding="utf-8").splitlines()
    if not text or not text[0].startswith(SHA_TAG):
        raise ImporterError(str(path), "hulls", f"first line must be '{SHA_TAG} <sha256>'; regenerate the sidecar")
    sha = text[0][len(SHA_TAG):].strip()
    verts: list[list[float]] = []
    blocks: list[tuple[int, list[list[int]]]] = []
    for line in text[1:]:
        tok = line.split()
        if not tok or tok[0].startswith("#"):
            continue
        if tok[0] == "o":
            blocks.append((len(verts), []))
        elif tok[0] == "v":
            verts.append([float(x) for x in tok[1:4]])
        elif tok[0] == "f" and blocks:
            blocks[-1][1].append([int(t.split("/")[0]) - 1 for t in tok[1:4]])
    return sha, _split_blocks(np.asarray(verts, dtype=np.float64).reshape(-1, 3), blocks)


def _split_blocks(v: np.ndarray, blocks: list[tuple[int, list[list[int]]]]) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    ends = [b[0] for b in blocks[1:]] + [len(v)]
    for (start, faces), end in zip(blocks, ends):
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3) - start
        out.append((v[start:end].copy(), f))
    return out
