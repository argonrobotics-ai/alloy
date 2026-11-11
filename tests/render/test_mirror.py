# SPDX-License-Identifier: Apache-2.0
"""Sparse mirror of a gs:// bundle and GCS-sourced stage items (no network: gcloud is faked)."""
from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from alloy.render import mirror
from alloy.render.staging import StageItem, Stager


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def fake_gcloud(objects: dict[str, bytes]):
    """subprocess.run stand-in for the gcloud storage calls mirror.py makes."""
    def run(cmd, input=None, capture_output=False, text=False, check=False):
        verb = cmd[2]
        if verb == "ls":
            out = "\n".join(f"  {len(b)}  2026-11-05T00:00:00Z  {u}" for u, b in objects.items())
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if verb == "rsync":
            src, dest = cmd[-2], Path(cmd[-1])
            for u, b in objects.items():
                rel = u[len(src) + 1:]
                if not (rel.startswith("frames/") and rel.endswith(".npy")):
                    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                    (dest / rel).write_bytes(b)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if verb == "cp":
            dest = Path(cmd[-1])
            for u in input.split():
                (dest / u.rsplit("/", 1)[-1]).write_bytes(objects[u])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if verb == "cat":
            lo, hi = map(int, cmd[4].split("-"))
            return subprocess.CompletedProcess(cmd, 0, objects[cmd[5]][lo:hi + 1], b"")
        raise AssertionError(cmd)
    return run


def test_sparse_mirror_keeps_shapes_without_data(tmp_path):
    uri = "gs://b/bundles/grid"
    big = np.arange(40_000, dtype=np.float32).reshape(100, 400)
    objects = {f"{uri}/scene.json": b'{"arb": 1}', f"{uri}/meshes/m.npz": b"npz",
               f"{uri}/frames/t.npy": npy_bytes(np.arange(5, dtype=np.float64)),
               f"{uri}/frames/body_q.npy": npy_bytes(big)}
    dest = mirror.mirror_gcs_bundle(uri, tmp_path / "m", run=fake_gcloud(objects), workers=2)
    assert (dest / "scene.json").read_bytes() == b'{"arb": 1}'
    assert np.load(dest / "frames" / "t.npy").tolist() == [0, 1, 2, 3, 4]       # small arrays copied whole
    q = np.load(dest / "frames" / "body_q.npy", mmap_mode="r")
    assert q.shape == (100, 400) and q.dtype == np.float32 and float(q[-1, -1]) == 0.0  # header only, sparse
    assert (dest / "frames" / "body_q.npy").stat().st_size == len(objects[f"{uri}/frames/body_q.npy"])


def test_gcs_sourced_stage_item_is_never_uploaded_or_rsynced(tmp_path):
    calls = []
    be = SimpleNamespace(ensure_bucket=lambda b: None, stage_up=lambda *a: calls.append(("up", a)),
                         stage_down=lambda vm, uri, remote, is_dir: calls.append(("down", uri, is_dir)))
    item = StageItem("bundle", tmp_path, "alloy_render/j/bundle", source="gs://b/bundles/grid")
    st = Stager(be, "gs://b/stage/j-1", [item])
    st.start()
    assert st.fetch(SimpleNamespace(name="vm"), item)
    assert calls == [("down", "gs://b/bundles/grid", True)]
