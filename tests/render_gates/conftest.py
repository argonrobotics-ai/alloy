# SPDX-License-Identifier: Apache-2.0
"""Synthetic gate roots for the render-gate tests: noise that shrinks with spp, and a cheap metric (no FLIP/LPIPS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.render_gates.layout import image_name
from bench.render_gates.metrics import Scorer
from bench.render_gates.pngio import write_png

SPPS = (1, 4, 16, 64, 128)


def fake_metric(ref: np.ndarray, test: np.ndarray) -> dict:
    """Mean absolute difference as FLIP, a tenth of it as LPIPS."""
    d = float(np.mean(np.abs(ref.astype(np.float64) - test.astype(np.float64)))) / 255.0
    return {"flip_mean": d, "lpips": d / 10}


def noisy(base: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)


def write(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, a)


def make_frame(root: Path, scene: str, frame: str, engines: dict[str, float], res: str = "32x32", seed: int = 0) -> None:
    """engines: name -> noise scale (1.0 = Cycles-like); noise sigma = 40 * scale / sqrt(spp), denoise on halves it."""
    w, h = (int(v) for v in res.split("x"))
    base = np.random.default_rng(seed).uniform(40, 200, (h, w, 3))
    fd = root / scene / frame
    write(fd / "cycles" / image_name(res, 4096, "on"), base.astype(np.uint8))
    for k, (name, scale) in enumerate(engines.items()):
        for spp in SPPS:
            for den, f in (("on", 0.5), ("off", 1.0)):
                sigma = 40 * scale * f / np.sqrt(spp)
                write(fd / name / image_name(res, spp, den), noisy(base, sigma, seed * 1000 + k * 100 + spp))


@pytest.fixture
def scorer() -> Scorer:
    return Scorer(cache_dir=None, metric_fn=fake_metric)


@pytest.fixture
def gate_root(tmp_path: Path) -> Path:
    """c3 with two frames: Cycles controls s1-s3 at scale 1, engine 'alloy' at 1.0 (passes), 'bad' at 3.0 (fails)."""
    root = tmp_path / "root"
    for i, frame in enumerate(("10", "20")):
        make_frame(root, "c3", frame, {"cycles_s1": 1.0, "cycles_s2": 1.0, "cycles_s3": 1.0, "alloy": 1.0, "bad": 3.0},
                   seed=i + 1)
    return root
