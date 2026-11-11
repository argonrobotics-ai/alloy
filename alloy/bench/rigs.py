# SPDX-License-Identifier: Apache-2.0
"""Rig datasets on disk (bench §4.4, SPEC §3.10 `load_rig`): immutable, sha256-checked.

Layout of `bench/rigs/<law>/<dataset-id>/`:
- `rig_card.json`: the `RigCard` fields;
- `conditions.json`: `[{"id", "params"}]`;
- `trials.json`: `[{"file", "condition_id", "index", "split"}]`;
- `trials/<cond>__<k>.npz`: one SI array per series channel ("t" first) and `scalar__<name>` 0-d arrays;
- `MANIFEST.sha256`: `<sha256>  <relative path>` for every file above; the dataset sha256 is the hash of this file.
`load_rig` recomputes every hash and refuses a dataset whose files differ from the manifest (edited trials).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from alloy.bench.types import Condition, RigCard, RigDataset, RigTrial
from alloy.core.errors import LawError

log = logging.getLogger("alloy.bench")

MANIFEST = "MANIFEST.sha256"
_SCALAR = "scalar__"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _npz_bytes(trial: RigTrial) -> bytes:
    arrays: dict[str, Any] = {k: np.asarray(v, dtype=np.float64) for k, v in trial.series.items()}
    arrays.update({_SCALAR + k: np.asarray(float(v)) for k, v in trial.scalars.items()})
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def write_rig(data: RigDataset, root: Path) -> Path:
    """Write `data` under `root/<law>/<dataset-id>/` with its manifest; refuses to overwrite an existing dataset."""
    out = Path(root) / data.card.law_id / data.card.dataset_id
    if out.exists():
        raise LawError(f"rig dataset {out} exists; datasets are immutable (write a new dataset id)")
    files: dict[str, bytes] = {
        "rig_card.json": _json_bytes(asdict(data.card)),
        "conditions.json": _json_bytes([{"id": c.id, "params": dict(c.params)} for c in data.conditions]),
    }
    index = []
    for t in data.trials:
        rel = f"trials/{t.condition_id}__{t.index}.npz"
        files[rel] = _npz_bytes(t)
        index.append({"file": rel, "condition_id": t.condition_id, "index": t.index, "split": t.split})
    files["trials.json"] = _json_bytes(index)
    for rel, blob in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(blob)
    manifest = "".join(f"{_sha(blob)}  {rel}\n" for rel, blob in sorted(files.items()))
    (out / MANIFEST).write_text(manifest)
    log.info("wrote rig dataset %s (%d trials)", out, len(data.trials))
    return out


def _verify(root: Path) -> str:
    """Check every manifest entry; return the dataset sha256 (hash of the manifest)."""
    mpath = root / MANIFEST
    if not mpath.exists():
        raise LawError(f"{root}: no {MANIFEST}")
    text = mpath.read_text()
    listed = set()
    for line in text.splitlines():
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        f = root / rel
        if not f.exists() or _sha(f.read_bytes()) != digest:
            raise LawError(f"{root}: {rel} does not match {MANIFEST} (edited or missing); datasets are immutable")
    extra = {str(p.relative_to(root)) for p in (root / "trials").glob("*.npz")} - listed
    if extra:
        raise LawError(f"{root}: trial files not in the manifest: {sorted(extra)}")
    return _sha(text.encode())


def _trial(root: Path, entry: Mapping[str, Any]) -> RigTrial:
    with np.load(root / entry["file"]) as z:
        series = {k: np.array(z[k]) for k in z.files if not k.startswith(_SCALAR)}
        scalars = {k[len(_SCALAR):]: float(z[k]) for k in z.files if k.startswith(_SCALAR)}
    return RigTrial(entry["condition_id"], int(entry["index"]), entry["split"], series, scalars)


def load_rig(root: Path) -> RigDataset:
    """Load a dataset after verifying its manifest; raises LawError on any edited, missing or extra trial."""
    root = Path(root)
    sha = _verify(root)
    card = RigCard(**json.loads((root / "rig_card.json").read_text()))
    conds = tuple(Condition(c["id"], c["params"]) for c in json.loads((root / "conditions.json").read_text()))
    trials = tuple(_trial(root, e) for e in json.loads((root / "trials.json").read_text()))
    known = {c.id for c in conds}
    bad = sorted({t.condition_id for t in trials} - known)
    if bad:
        raise LawError(f"{root}: trials reference unknown conditions {bad}")
    return RigDataset(card, conds, trials, sha)


def datasets_for(law_id: str, root: Path = Path("bench/rigs")) -> list[Path]:
    """Dataset directories for a law (`bench/rigs/<law_id>/*/` holding a manifest)."""
    base = Path(root) / law_id
    return sorted(p for p in base.glob("*") if (p / MANIFEST).exists()) if base.is_dir() else []


def is_synthetic(data: RigDataset) -> bool:
    return data.card.protocol_version.startswith("synthetic")


__all__ = ["load_rig", "write_rig", "datasets_for", "is_synthetic", "MANIFEST"]
