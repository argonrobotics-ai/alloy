#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pack one teacher-render job (EXR folders from ``blender/render_bundle.py``) into training arrays.

    python tools/denoiser/pack.py JOB_DIR OUT_DIR [--oidn] [--workers 8] [--delete-exr]

``JOB_DIR`` holds ``view.json``, ``lut.png`` (+ .json), ``wb.json``, optionally ``job.json`` (the job's own record)
and ``<W>x<H>/<frame>/{in_s<seed>_<spp>,ref}.exr``. ``OUT_DIR`` gets, per resolution ``<W>x<H>/``:

- ``noisy.npy``, ``albedo.npy``, ``normal.npy``: (F, K, H, W, 3) float16, K inputs as listed in ``meta.json`` keys
  (seed, spp), sorted by seed then spp; ``normal`` is Cycles' camera-space Denoising Normal;
- ``oidn.npy`` (with ``--oidn``): Cycles' own OIDN output of each input, the baseline for evaluation;
- ``ref.npy``: (F, H, W, 3) float16, the reference (high spp + OIDN);

and once per job ``lut.npy`` ((3, n, n, n) float16 display LUT, axes [c, b, g, r]) and ``meta.json`` (frames,
keys, view settings, white-balance matrix, LUT grid, job record). Frames missing any input are skipped and counted.
Prints one JSON summary line.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger("denoiser.pack")
LAYER = "ViewLayer"
PASSES = {"noisy": ("Noisy Image", "RGB"), "oidn": ("Combined", "RGB"), "albedo": ("Denoising Albedo", "RGB"),
          "normal": ("Denoising Normal", "XYZ"), "depth": ("Denoising Depth", "Z")}
IN_RX = re.compile(r"in_s(\d+)_(\d+)\.exr$")


def read_exr(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Multilayer teacher EXR -> {key: (H, W, C) float32}, rows top to bottom.

    Example: ``read_exr(Path("ref.exr"), ("oidn",))["oidn"].shape == (128, 128, 3)``."""
    import OpenEXR
    with OpenEXR.File(str(path), separate_channels=True) as f:
        ch = f.parts[0].channels
        out = {}
        for k in keys:
            layer, comps = PASSES[k]
            out[k] = np.stack([ch[f"{LAYER}.{layer}.{c}"].pixels.astype(np.float32) for c in comps], axis=-1)
    return out


def read_png16(path: Path) -> np.ndarray:
    """Decode a non-interlaced 16-bit RGB PNG (as Blender writes the display LUT) -> (H, W, 3) float32 in [0, 1].

    Example: ``read_png16(Path("lut.png")).dtype == np.float32``."""
    data = path.read_bytes()
    pos, chunks, hdr = 8, [], None
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            hdr = body
        elif kind == b"IDAT":
            chunks.append(body)
        pos += 12 + n
    w, h = int.from_bytes(hdr[0:4], "big"), int.from_bytes(hdr[4:8], "big")
    if hdr[8] != 16 or hdr[9] != 2 or hdr[12] != 0:
        raise ValueError(f"{path}: expected a non-interlaced 16-bit RGB PNG; re-bake it with teacher_common.bake_lut")
    raw = np.frombuffer(zlib.decompress(b"".join(chunks)), np.uint8).reshape(h, 1 + w * 6)
    return _unfilter(raw, w, h, bpp=6).view(">u2").reshape(h, w, 3).astype(np.float32) / 65535.0


def _unfilter(raw: np.ndarray, w: int, h: int, bpp: int) -> np.ndarray:
    """Undo PNG scanline filters (types 0-4) -> (h, w * bpp) uint8."""
    out = np.zeros((h, w * bpp), np.uint8)
    prev = np.zeros(w * bpp, np.int32)
    for y in range(h):
        ft, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        if ft == 2:
            line = (line + prev) & 255
        elif ft in (1, 3, 4):
            line = _unfilter_seq(line, prev, ft, bpp)
        out[y] = line
        prev = line
    return out


def _unfilter_seq(line: np.ndarray, prev: np.ndarray, ft: int, bpp: int) -> np.ndarray:
    """Sub / Average / Paeth filters, which depend on the already-decoded left neighbour."""
    cur = line.copy()
    for i in range(len(cur)):
        a = cur[i - bpp] if i >= bpp else 0
        b, c = prev[i], (prev[i - bpp] if i >= bpp else 0)
        if ft == 1:
            pred = a
        elif ft == 3:
            pred = (a + b) >> 1
        else:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
        cur[i] = (cur[i] + pred) & 255
    return cur


def load_lut(path: Path) -> tuple[np.ndarray, dict]:
    """Display LUT PNG + its JSON -> ((3, n, n, n) float32 [c, b, g, r], grid meta)."""
    meta = json.loads(path.with_suffix(".json").read_text())
    n = meta["n"]
    px = read_png16(path)[::-1].reshape(-1, 3)[: n ** 3]  # undo Blender's bottom-up pixel order
    return px.reshape(n, n, n, 3).transpose(3, 0, 1, 2).copy(), meta


def keys_of(d: Path) -> list[tuple[int, int]]:
    """The (seed, spp) inputs present in one frame folder, sorted."""
    return sorted((int(m.group(1)), int(m.group(2))) for p in d.glob("in_s*.exr") if (m := IN_RX.search(p.name)))


def load_frame(d: Path, keys: list[tuple[int, int]], oidn: bool) -> dict[str, np.ndarray] | None:
    """All inputs + reference of one frame as float16 arrays, or None if anything is missing or unreadable."""
    if not (d / "ref.exr").exists() or not set(keys) <= set(keys_of(d)):
        return None
    k = ("noisy", "albedo", "normal") + (("oidn",) if oidn else ())
    try:
        ref = read_exr(d / "ref.exr", ("oidn",))["oidn"]
        ins = [read_exr(d / f"in_s{s}_{n}.exr", k) for s, n in keys]
    except Exception as exc:  # noqa: BLE001 (a broken file skips the frame, it is counted)
        log.warning("skip %s: %s", d, exc)
        return None
    out = {n: np.stack([x[n] for x in ins]).astype(np.float16) for n in k}
    out["ref"] = ref.astype(np.float16)
    return out


def pack_res(rdir: Path, out: Path, oidn: bool, workers: int) -> dict:
    """Pack one resolution folder; returns its summary (frames, keys, skipped)."""
    frames = sorted(d for d in rdir.iterdir() if d.is_dir())
    keys = max((keys_of(d) for d in frames[:20]), key=len, default=[])
    with ThreadPoolExecutor(workers) as pool:
        loaded = list(pool.map(lambda d: load_frame(d, keys, oidn), frames))
    ok = [(d, x) for d, x in zip(frames, loaded) if x is not None]
    if not ok:
        return {"frames": [], "keys": keys, "skipped": len(frames)}
    out.mkdir(parents=True, exist_ok=True)
    for name in ok[0][1]:
        np.save(out / f"{name}.npy", np.stack([x[name] for _, x in ok]))
    return {"frames": [int(d.name) if d.name.isdigit() else d.name for d, _ in ok], "keys": keys,
            "skipped": len(frames) - len(ok)}


def pack_job(job: Path, out: Path, oidn: bool = False, workers: int = 8) -> dict:
    """Pack every resolution of one job folder into ``out``; returns the meta dict also written to meta.json.

    Example: ``pack_job(Path("render/c3_v001_wide"), Path("packed/c3_v001_wide"))``."""
    res = {}
    for rdir in sorted(p for p in job.iterdir() if p.is_dir() and re.fullmatch(r"\d+x\d+", p.name)):
        res[rdir.name] = pack_res(rdir, out / rdir.name, oidn, workers)
    lut, lut_meta = load_lut(job / "lut.png")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "lut.npy", lut.astype(np.float16))
    meta = {"res": res, "view": json.loads((job / "view.json").read_text()),
            "wb": json.loads((job / "wb.json").read_text())["matrix"], "lut": lut_meta,
            "job": json.loads((job / "job.json").read_text()) if (job / "job.json").exists() else {}}
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job")
    p.add_argument("out")
    p.add_argument("--oidn", action="store_true", help="also pack Cycles' OIDN output of each input (eval sets)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--delete-exr", dest="delete_exr", action="store_true", help="remove the EXR folders afterwards")
    a = p.parse_args(argv)
    meta = pack_job(Path(a.job), Path(a.out), a.oidn, a.workers)
    n = sum(len(r["frames"]) for r in meta["res"].values())
    if a.delete_exr and n:
        for r in meta["res"]:
            shutil.rmtree(Path(a.job) / r)
    print(json.dumps({"out": a.out, "frames": n, "skipped": sum(r["skipped"] for r in meta["res"].values())}))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
