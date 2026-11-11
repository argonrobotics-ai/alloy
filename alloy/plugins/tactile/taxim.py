# SPDX-License-Identifier: Apache-2.0
"""Taxim calibration data (github.com/CMURoboTouch/Taxim, MIT licence), fetched on demand and sha256-checked.

The data is pinned at commit 4936657a (2022-11-02). It is cached under ``$ALLOY_TAXIM_DIR``, or
``~/.cache/alloy/taxim/<commit8>``. The GelSensor ``kernel="taxim"`` mode uses the marker FEM tensor
(``calibs/femCalib.npz``, 900 x 900 offsets x 3 x 3). The image side (OSS-4) uses the optical files of the
GelSight and DIGIT sensors; OSS-21 ships those converted in ``alloy/camera/tactile/data``.

Taxim's convention, checked in ``MarkerMotionSimulation/compose/superposition.py``:
- the displacement at pixel q is Σ_k T[q - k + 450] @ load_k;
- the rows of T are displacement components and its columns load components;
- x is the image column, y the image row, and z is positive out of the gel.

Taxim scales the shear-to-shear block by 0.07 (``shear_friction``), because its loads are prescribed displacements.
GelSensor's loads are engine forces, so it does not apply that factor (docs/notes/wp13-tactile.md).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

COMMIT = "4936657a42e00f09a94301111830250ef2c96a89"
FEM_PIXMM = 0.0295e-3     # m per px of the FEM tensor grid (Basics/sensorParams.py, pixmm = 0.0295 mm)
FILES = {"calibs/femCalib.npz": "ddc39487a5c181646b743416f4ca75688443fb1642b528634e47888bb532f767"}


def cache_dir() -> Path:
    """Where fetched files live: ``$ALLOY_TAXIM_DIR`` or ``~/.cache/alloy/taxim/<commit8>``."""
    env = os.environ.get("ALLOY_TAXIM_DIR")
    return Path(env) if env else Path.home() / ".cache" / "alloy" / "taxim" / COMMIT[:8]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel: str) -> Path:
    """The local path of Taxim's file ``rel`` (one of ``FILES``), downloading and checking it if needed."""
    if rel not in FILES:
        raise ValueError(f"unknown Taxim file {rel!r}; known: {sorted(FILES)}")
    path = cache_dir() / rel
    if path.exists() and _sha256(path) == FILES[rel]:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/CMURoboTouch/Taxim/{COMMIT}/{rel}"
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL at a pinned commit
    except OSError as e:
        raise RuntimeError(f"could not fetch Taxim's {rel} from {url} ({e}); download it to {path} by hand, or set "
                           f"ALLOY_TAXIM_DIR to a folder holding it") from e
    got = _sha256(tmp)
    if got != FILES[rel]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Taxim's {rel}: sha256 {got} != pinned {FILES[rel]}")
    tmp.replace(path)
    return path


def fem_tensor() -> np.ndarray:
    """Taxim's marker FEM tensor as float32 [900, 900, 3, 3], in Taxim's axes (cached as .npy after the first load)."""
    npy = cache_dir() / "femCalib_tensor_f32.npy"
    if npy.exists():
        return np.load(npy)
    t = np.load(fetch("calibs/femCalib.npz"))["tensorMap"].astype(np.float32)
    np.save(npy, t)
    return t


def fem_tensor_sensor_axes() -> np.ndarray:
    """Taxim's tensor in GelSensor's axes: x right, y up (Taxim's rows run down), z into the gel (Taxim's out).
    That is S·T·S with S = diag(1, -1, -1). Index it at [450 - dy, 450 + dx] for an offset (dx, dy) in px, x right
    and y up."""
    s = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    return np.ascontiguousarray(np.einsum("ab,ijbc,cd->ijad", s, fem_tensor(), s))
