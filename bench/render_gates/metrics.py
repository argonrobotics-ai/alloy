# SPDX-License-Identifier: Apache-2.0
"""Image metrics for the render gates: FLIP (LDR mean) and LPIPS (AlexNet) on 8-bit sRGB, plus a disk cache.

FLIP comes from ``flip-evaluator`` (BSD-3-Clause) and LPIPS from ``lpips`` (BSD-2-Clause); both are imported only when
an image pair is scored, so the gate logic and its tests run without them. Install them with
``pip install flip-evaluator lpips torch torchvision``.

Example:
    >>> import numpy as np
    >>> from bench.render_gates.metrics import Scorer, psnr
    >>> a = np.zeros((4, 4, 3), np.uint8)
    >>> psnr(a, a)
    99.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray], dict]
KEYS = ("flip_mean", "lpips")
_LPIPS: dict[str, object] = {}


class MetricsUnavailable(ImportError):
    """Raised when FLIP or LPIPS is missing; the message says what to install."""


def load_rgb(path: Path) -> np.ndarray:
    """Read an image as ``(H, W, 3) uint8`` sRGB.

    Example:
        >>> load_rgb(Path("ref.png")).shape  # doctest: +SKIP
        (128, 128, 3)
    """
    try:
        from PIL import Image
    except ImportError:
        from .pngio import read_png

        return read_png(path)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _lpips_model():
    if "m" not in _LPIPS:
        try:
            import lpips
            import torch
        except ImportError as e:
            raise MetricsUnavailable("LPIPS needs `pip install lpips torch torchvision`") from e
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _LPIPS["m"] = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    return _LPIPS["m"]


def lpips_alex(ref: np.ndarray, test: np.ndarray) -> float:
    """LPIPS (AlexNet, v0.1) between two uint8 sRGB images; 0 means identical.

    Example:
        >>> lpips_alex(img, img)  # doctest: +SKIP
        0.0
    """
    import torch

    model = _lpips_model()
    dev = next(model.parameters()).device

    def tensor(a: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(a).copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)

    with torch.no_grad():
        return float(model(tensor(ref), tensor(test)).item())


def flip_ldr(ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """(mean, 99th percentile) of the LDR FLIP error map between two uint8 sRGB images.

    Example:
        >>> flip_ldr(img, img)  # doctest: +SKIP
        (0.0, 0.0)
    """
    try:
        import flip_evaluator as flip
    except ImportError as e:
        raise MetricsUnavailable("FLIP needs `pip install flip-evaluator`") from e
    err, mean, _ = flip.evaluate(ref.astype(np.float32) / 255, test.astype(np.float32) / 255, "LDR",
                                 inputsRGB=True, applyMagma=False, computeMeanError=True)
    return float(mean), float(np.percentile(np.asarray(err), 99))


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """PSNR in dB on 8-bit values; 99.0 for identical images.

    Example:
        >>> import numpy as np
        >>> round(psnr(np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 255, np.uint8)), 1)
        0.0
    """
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def pair_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    """The gate's metric row for one pair: ``flip_mean``, ``flip_p99``, ``lpips``, ``psnr``.

    Example:
        >>> pair_metrics(ref, test)["lpips"]  # doctest: +SKIP
        0.0031
    """
    fm, fp = flip_ldr(ref, test)
    return {"flip_mean": fm, "flip_p99": fp, "lpips": lpips_alex(ref, test), "psnr": psnr(ref, test)}


def hist_match(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-channel monotone remap of ``test`` so its value distribution equals ``ref``'s (reported, never gated).

    Example:
        >>> import numpy as np
        >>> a = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        >>> bool((hist_match(a, a) == a).all())
        True
    """
    out = np.empty_like(test)
    for c in range(3):
        t, r = test[..., c].ravel(), ref[..., c].ravel()
        order = np.argsort(t, kind="stable")
        matched = np.empty_like(t)
        matched[order] = np.sort(r)
        out[..., c] = matched.reshape(test.shape[:2])
    return out


class Scorer:
    """Scores image pairs; caches reference pixels in memory and metric rows on disk (keyed by path, size, mtime).

    ``metric_fn`` defaults to :func:`pair_metrics`; tests inject a cheap function instead.

    Example:
        >>> s = Scorer(cache_dir=None, metric_fn=lambda r, t: {"flip_mean": 0.0, "lpips": 0.0})
        >>> s.score(Path("ref.png"), Path("test.png"), "raw")  # doctest: +SKIP
        {'flip_mean': 0.0, 'lpips': 0.0}
    """

    def __init__(self, cache_dir: Path | None = None, metric_fn: MetricFn | None = None):
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.metric_fn = metric_fn or pair_metrics
        self._pix: dict[Path, np.ndarray] = {}
        self.computed = self.hits = 0

    def pixels(self, p: Path) -> np.ndarray:
        if p not in self._pix:
            if len(self._pix) > 64:
                self._pix.clear()
            self._pix[p] = load_rgb(p)
        return self._pix[p]

    def _key(self, ref: Path, test: Path, mode: str) -> Path | None:
        if not self.cache_dir:
            return None
        fn = getattr(self.metric_fn, "__qualname__", "fn")
        sig = "|".join(f"{q.resolve()}:{q.stat().st_size}:{q.stat().st_mtime_ns}" for q in (ref, test)) + mode + fn
        return self.cache_dir / (hashlib.sha1(sig.encode()).hexdigest() + ".json")

    def score(self, ref: Path, test: Path, mode: str = "raw") -> dict:
        """Metric row for ``test`` against ``ref``; ``mode`` "matched" histogram-matches first. Shape mismatch -> error row."""
        f = self._key(ref, test, mode)
        if f and f.exists():
            self.hits += 1
            return json.loads(f.read_text())
        r, t = self.pixels(ref), self.pixels(test)
        if r.shape != t.shape:
            return {"error": f"{test.name}: shape {t.shape} != reference {r.shape}; render at the reference size"}
        row = self.metric_fn(r, hist_match(t, r) if mode == "matched" else t)
        self.computed += 1
        if f:
            f.write_text(json.dumps(row))
        return row
