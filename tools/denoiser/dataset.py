# SPDX-License-Identifier: Apache-2.0
"""Training data from packed teacher jobs (``pack.py`` output), streamed through a GPU frame pool.

    from dataset import PackedIndex, GpuPool
    index = PackedIndex.scan(["/data/packed"], split="train")
    pool = GpuPool(index, res="128x128", capacity=12000, device="cuda")
    batch = pool.sample(48, crop=128, g=torch.Generator("cuda"))     # alloy.camera.denoiser.training.Batch

A teacher set (~100k frames, ~300 GB) does not fit on one GPU. ``GpuPool`` holds ``capacity`` random frames and
``refresh(n)`` swaps ``n`` of them for new random frames read from the memory-mapped arrays (a background thread
prefetches them into host memory). Inputs of every job are mapped onto one global list of (seed, spp) keys; a
frame only offers the keys its job rendered. Seed mixing averages two seeds' inputs into a valid input with the
summed spp.
"""
from __future__ import annotations

import json
import logging
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from alloy.camera.denoiser.training import Batch, DisplayLUT

log = logging.getLogger("denoiser.dataset")
ARRAYS = ("noisy", "albedo", "normal")


@dataclass
class JobRes:
    """One job at one resolution: its array folder, keys and LUT index. ``arrays`` maps the files on each access
    (a cached memory map keeps a file descriptor open, and thousands of jobs would exhaust the process limit)."""

    path: Path
    keys: list[tuple[int, int]]
    lut: int
    weight: float
    frames: int = 0
    names: tuple[str, ...] = ()

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {n: np.load(self.path / f"{n}.npy", mmap_mode="r") for n in self.names}

    def open(self) -> JobRes:
        """Check the files and read the frame count. Raises OSError / ValueError for a half-written job."""
        self.names = ARRAYS + ("ref",) + (("oidn",) if (self.path / "oidn.npy").exists() else ())
        a = self.arrays
        self.frames = int(a["ref"].shape[0])
        if any(a[n].shape[0] != self.frames for n in ARRAYS):
            raise ValueError(f"{self.path}: arrays disagree on the frame count")
        return self


@dataclass
class PackedIndex:
    """All packed jobs of one split, grouped by resolution, with their display LUTs stacked."""

    by_res: dict[str, list[JobRes]]
    luts: list[np.ndarray]
    lut_meta: dict
    keys: list[tuple[int, int]]
    roots: list[Path] = field(default_factory=list)
    split: str | None = None
    exclude: str | None = None
    seen: set[str] = field(default_factory=set)
    weight_rules: dict[str, float] = field(default_factory=dict)  # job-id substring -> sampling weight multiplier

    @classmethod
    def scan(cls, roots: list[str | Path], split: str | None = None, exclude: str | None = None,
             weight_rules: dict[str, float] | None = None) -> PackedIndex:
        """Find every ``<root>/<job>/meta.json``; keep jobs of ``split`` (None: all) whose id lacks ``exclude``.
        ``weight_rules`` ({"c2v2": 3, "_v000": 2}) multiply the sampling weight of jobs whose id contains the key."""
        index = cls({}, [], {}, [], [Path(r) for r in roots], split, exclude, weight_rules=dict(weight_rules or {}))
        if not index.update():
            raise FileNotFoundError(f"no packed jobs under {roots} (split={split}); run run_jobs.py first")
        index.keys = sorted({k for jobs in index.by_res.values() for j in jobs for k in j.keys})
        return index

    def update(self) -> int:
        """Add jobs that appeared since the last scan (``keys`` stays fixed); returns how many were added."""
        added = 0
        for meta_path in sorted(p for r in self.roots for p in r.glob("*/meta.json")):
            if str(meta_path) in self.seen:
                continue
            self.seen.add(str(meta_path))
            added += self._add(meta_path)
        return added

    def _add(self, meta_path: Path) -> int:
        meta = json.loads(meta_path.read_text())
        job = meta.get("job", {})
        if (self.split and job.get("split", "train") != self.split) or (
                self.exclude and self.exclude in meta_path.parent.name):
            return 0
        idx = len(self.luts)
        self.luts.append(np.load(meta_path.parent / "lut.npy").astype(np.float16))
        self.lut_meta = meta["lut"]
        w = float(job.get("weight", 1.0))
        for key, mult in self.weight_rules.items():
            w *= mult if key in meta_path.parent.name else 1.0
        try:
            opened = [(res, JobRes(meta_path.parent / res, [tuple(k) for k in info["keys"]], idx, w).open())
                      for res, info in meta["res"].items() if info["frames"]]
        except (OSError, ValueError) as exc:  # a job still being unpacked or copied: retry at the next update()
            log.warning("skipping %s for now: %s", meta_path.parent.name, exc)
            self.luts.pop()
            self.seen.discard(str(meta_path))
            return 0
        for res, jr in opened:
            self.by_res.setdefault(res, []).append(jr)
        return 1

    def display(self, device: str | torch.device) -> DisplayLUT:
        """The stacked display LUTs as a :class:`DisplayLUT` on ``device``."""
        stack = torch.from_numpy(np.stack(self.luts)).to(device).float()
        return DisplayLUT(stack, self.lut_meta["lo"], self.lut_meta["hi"])

    def frames(self, res: str) -> int:
        return sum(j.frames for j in self.by_res.get(res, []))


def _finite(t: torch.Tensor) -> torch.Tensor:
    """float32 copy with inf (fp16 overflow of a firefly) clamped and nan zeroed."""
    return torch.nan_to_num(t.float(), nan=0.0, posinf=6.0e4, neginf=0.0)


def _seed_pairs(keys: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Index pairs of inputs from different seeds (any spp): their spp-weighted mean is a valid input."""
    return [(a, b) for a in range(len(keys)) for b in range(len(keys)) if keys[a][0] < keys[b][0]]


class GpuPool:
    """A GPU-resident random subset of one resolution's frames; see the module docstring.

    Example: ``pool = GpuPool(index, "640x480", capacity=300); b = pool.sample(16, 128, g)``."""

    def __init__(self, index: PackedIndex, res: str, capacity: int, device: str = "cuda", seed: int = 0,
                 spp_weights: dict[int, float] | None = None) -> None:
        self.index, self.res, self.device = index, res, torch.device(device)
        self.jobs = index.by_res[res]
        self.rng = random.Random(seed)
        self.keys = index.keys
        self.kpos = {k: i for i, k in enumerate(self.keys)}
        j0 = self.jobs[0].arrays["ref"]
        self.h, self.w = int(j0.shape[1]), int(j0.shape[2])
        self.cap, self.n = capacity, 0  # slots allocated / filled (the pool grows as packed jobs arrive)
        shape = (self.cap, len(self.keys), self.h, self.w, 3)
        self.data = {n: torch.zeros(shape, dtype=torch.float16, device=self.device) for n in ARRAYS}
        self.ref = torch.zeros((self.cap, self.h, self.w, 3), dtype=torch.float16, device=self.device)
        self.valid = torch.zeros((self.cap, len(self.keys)), dtype=torch.bool, device=self.device)
        self.lut = torch.zeros(self.cap, dtype=torch.long, device=self.device)
        self.spp = torch.tensor([k[1] for k in self.keys], dtype=torch.float32, device=self.device)
        self.key_w = torch.tensor([(spp_weights or {}).get(k[1], 1.0) for k in self.keys], device=self.device)
        self.pairs = torch.tensor(_seed_pairs(self.keys) or [(0, 0)], device=self.device)
        self._next: list | None = None
        self._thread: threading.Thread | None = None
        self.fill()

    def _pick(self) -> tuple[JobRes, int]:
        """A random (job, frame), jobs weighted by frames x weight."""
        w = [j.frames * j.weight for j in self.jobs]
        job = self.rng.choices(self.jobs, w)[0]
        return job, self.rng.randrange(job.frames)

    def _read(self, n: int) -> list:
        """Read ``n`` random frames into host memory (runs in the prefetch thread)."""
        out = []
        for _ in range(n):
            job, f = self._pick()
            a = job.arrays
            out.append((job, {k: np.ascontiguousarray(a[k][f]) for k in ARRAYS + ("ref",)}))
        return out

    def _write(self, slot: int, job: JobRes, x: dict) -> None:
        own = [i for i, k in enumerate(job.keys) if k in self.kpos]  # inputs outside the pool's keys are skipped
        cols = torch.tensor([self.kpos[job.keys[i]] for i in own], device=self.device)
        self.valid[slot] = False
        self.valid[slot, cols] = True
        for n in ARRAYS:
            self.data[n][slot, cols] = torch.from_numpy(x[n][own]).to(self.device, non_blocking=True)
        self.ref[slot] = torch.from_numpy(x["ref"]).to(self.device, non_blocking=True)
        self.lut[slot] = job.lut

    def fill(self) -> None:
        """Fill up to ``capacity`` slots, at most one per frame on disk (at start)."""
        for slot, (job, x) in enumerate(self._read(min(self.cap, self.index.frames(self.res)))):
            self._write(slot, job, x)
            self.n = slot + 1
        log.info("pool %s: %d of %d slots, %d keys", self.res, self.n, self.cap, len(self.keys))

    def _slot(self) -> int:
        """The next free slot while the pool grows (up to the frames on disk), else a random one to replace."""
        if self.n < min(self.cap, self.index.frames(self.res)):
            self.n += 1
            return self.n - 1
        return self.rng.randrange(self.n)

    def refresh(self, n: int) -> None:
        """Replace ``n`` random slots with the prefetched frames, then start prefetching the next ``n``."""
        if self._thread is not None:
            self._thread.join()
            for job, x in self._next or []:
                self._write(self._slot(), job, x)
        box: list = []
        self._thread = threading.Thread(target=lambda: box.extend(self._read(n)), daemon=True)
        self._next = box
        self._thread.start()

    def sample(self, b: int, crop: int, g: torch.Generator, seedmix: float = 0.3) -> Batch:
        """``b`` random (frame, input) items, one random ``crop`` window for the whole batch."""
        dev = self.device
        i = torch.randint(0, self.n, (b,), device=dev, generator=g)
        k = torch.multinomial(self.valid[i].float() * self.key_w, 1, generator=g)[:, 0]
        c = min(crop, self.h, self.w)
        y0 = int(torch.randint(0, self.h - c + 1, (1,), device=dev, generator=g))
        x0 = int(torch.randint(0, self.w - c + 1, (1,), device=dev, generator=g))
        get = lambda t, kk: _finite(t[i, kk, y0:y0 + c, x0:x0 + c])
        rgb, alb, nrm = (get(self.data[n], k) for n in ARRAYS)
        spp = self.spp[k]
        if seedmix > 0 and len(self.pairs) > 1:
            rgb, alb, nrm, spp = self._seedmix(i, get, (rgb, alb, nrm), spp, seedmix, g)
        return Batch(rgb, alb, nrm, _finite(self.ref[i, y0:y0 + c, x0:x0 + c]), spp, self.lut[i])

    def _seedmix(self, i, get, imgs, spp, rate: float, g: torch.Generator):
        """Replace a ``rate`` share of items by the spp-weighted mean of two seeds' inputs (when both exist)."""
        b = i.shape[0]
        pi = self.pairs[torch.randint(0, len(self.pairs), (b,), device=self.device, generator=g)]
        ka, kb = pi[:, 0], pi[:, 1]
        ok = self.valid[i, ka] & self.valid[i, kb] & (torch.rand(b, device=self.device, generator=g) < rate)
        wa = (self.spp[ka] / (self.spp[ka] + self.spp[kb])).view(b, 1, 1, 1)
        sel = ok.view(b, 1, 1, 1)
        out = [torch.where(sel, wa * get(self.data[n], ka) + (1 - wa) * get(self.data[n], kb), x)
               for n, x in zip(ARRAYS, imgs)]
        return (*out, torch.where(ok, self.spp[ka] + self.spp[kb], spp))
