# SPDX-License-Identifier: Apache-2.0
"""Convex parts for ``Scene.add_mesh(..., collision="decompose")``.

The rigid emitter collides a ``decompose`` mesh as the convex hulls in a sidecar file ``<stem>.hulls.obj`` beside the
card mesh (first line ``# source_sha256: <sha256 of the mesh file>``, then one ``o hull_<k>`` block per hull) and
never runs CoACD itself. The API writes that sidecar when it turns a scene into cards:

1. a hull file shipped beside the source mesh (``part.obj`` -> ``part.hulls.obj``) whose sha line matches is copied;
2. else a cached decomposition of the same mesh bytes and settings (``$ALLOY_CACHE_DIR/hulls``, default
   ``~/.cache/alloy/hulls``) is reused;
3. else CoACD (Wei et al., SIGGRAPH 2022; MIT; the ``import`` extra) decomposes the mesh once, with the importers'
   settings (``alloy.importers.common.hulls``) plus CoACD's own manifold repair, and the result is cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from alloy.api import _say
from alloy.api.errors import FeatureUnavailableError, SceneError

log = logging.getLogger("alloy.api")

SUFFIX = ".hulls.obj"
#: CoACD's manifold check (and remesh only when it fails): API meshes need not be watertight; importer meshes are.
PREPROCESS = "auto"
_OTHER = ('or pass collision="convex" (one hull; fast) or "sdf" (exact; Newton contacts), or ship the hulls beside '
          "the mesh file as <stem>.hulls.obj")


def _cache_dir() -> Path:
    return Path(os.environ.get("ALLOY_CACHE_DIR", Path.home() / ".cache" / "alloy")) / "hulls"


def _sha_line(path: Path) -> str:
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().split()
    return head[2] if len(head) == 3 and head[:2] == ["#", "source_sha256:"] else ""


def _volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def write_sidecar(name: str, card_mesh: Path, mesh: tuple[np.ndarray, np.ndarray], source: Path | None) -> Path:
    """Write ``<stem>.hulls.obj`` beside ``card_mesh`` (the scene's copy of the mesh) and return its path.

    ``mesh`` is ``(vertices, faces)`` in the mesh file's own units (the card's ``scale`` applies to the hulls when they
    load); ``source`` is the user's mesh file, if any, where a shipped hull file is looked for.

    Example::

        write_sidecar("ell", root / "meshes/ell.obj", (v, f), None)   # -> root / "meshes/ell.hulls.obj"
    """
    sha = hashlib.sha256(card_mesh.read_bytes()).hexdigest()
    out = card_mesh.with_name(card_mesh.stem + SUFFIX)
    shipped = source.with_name(source.stem + SUFFIX) if source is not None else None
    if shipped is not None and shipped.is_file():
        if _sha_line(shipped) == sha:
            shutil.copyfile(shipped, out)
            log.info("%s: convex parts from %s", name, shipped)
            return out
        _say.warn("%s: %s does not belong to this mesh (its source_sha256 line differs); decomposing the mesh "
                  "again", name, shipped)
    out.write_text(_hull_text(name, sha, *mesh), encoding="utf-8")
    return out


def _hull_text(name: str, sha: str, v: np.ndarray, f: np.ndarray) -> str:
    from alloy.importers.common.hulls import HullParams  # noqa: PLC0415

    params = HullParams(preprocess=PREPROCESS)
    key = hashlib.sha256(f"{sha}|{json.dumps(asdict(params), sort_keys=True)}".encode()).hexdigest()[:32]
    cached = _cache_dir() / f"{key}{SUFFIX}"
    if cached.is_file() and _sha_line(cached) == sha:
        log.info("%s: convex parts from the cache %s", name, cached)
        return cached.read_text(encoding="utf-8")
    text = _decompose(name, sha, v, f, params)
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(cached)
    except OSError as exc:
        log.info("%s: could not cache the convex parts (%s)", name, exc)
    return text


def _decompose(name: str, sha: str, v: np.ndarray, f: np.ndarray, params) -> str:
    from alloy.importers.common.errors import ImporterError, MissingExtraError  # noqa: PLC0415
    from alloy.importers.common.hulls import decompose  # noqa: PLC0415
    from alloy.importers.common.objio import hulls_obj_text  # noqa: PLC0415

    volume = _volume(v, f)
    if not volume > 0:
        raise SceneError(f"mesh {name!r}: collision='decompose' needs a closed mesh with outward normals (volume "
                         f"{volume:.3g})", "close the mesh (faces counter-clockwise seen from outside), " + _OTHER,
                         field="collision")
    log.info("%s: splitting the mesh into convex parts with CoACD (seconds to minutes; cached after)", name)
    try:
        hulls, stats = decompose(v, f, volume, params)
    except MissingExtraError as exc:
        raise FeatureUnavailableError("collision='decompose' needs CoACD and trimesh, which are not installed",
                                      'pip install "alloy-engine[import]", ' + _OTHER, field="collision") from exc
    except ImporterError as exc:
        raise SceneError(f"mesh {name!r}: convex decomposition failed: {exc}", "repair the mesh, " + _OTHER,
                         field="collision") from exc
    log.info("%s: %d convex parts, hull volume / mesh volume %.2f, %.1f s", name, stats.n_hulls, stats.volume_ratio,
             stats.seconds)
    return hulls_obj_text(hulls, sha, header=f"CoACD {json.dumps(stats.params, sort_keys=True)}")
