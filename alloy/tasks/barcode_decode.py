# SPDX-License-Identifier: Apache-2.0
"""The scanner's decoder: zxing-cpp (Apache-2.0; the optional ``barcode`` extra) on each 8-bit frame, run the way
Gym-Anything's barcode sim runs it (their ``decoder.py``, MIT), so a read here means what a read there means.

Strategy: zxing's three binarizers (LocalAverage, GlobalHistogram, FixedThreshold), each on the frame and on a 2x
bicubic upsample, on a frame padded by 16 px (zxing misses symbols at the border). A 1-D symbol counts only when at
least two of the six passes read the same text; 2-D symbols (Data Matrix, QR) on one pass. Among several symbols the
one whose centre is nearest the aim point (default: the image centre) is reported, the DS2208 default with picklist
off (the real firmware's order is unpublished; this rule is authored and disclosed there).

EAN / UPC readings are normalised to GTIN-14 (``gtin14``), so a UPC-A item read as UPC-A or as EAN-13 compares equal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORMATS = {
    "retail": ("EAN13", "UPCA", "UPCE", "EAN8"),
    "inventory": ("EAN13", "UPCA", "UPCE", "EAN8", "Code128", "ITF", "DataMatrix", "QRCode"),
    "library": ("EAN13", "UPCA", "Codabar", "Code128"),
}
ONE_PASS = ("DataMatrix", "QRCode")
RETAIL = ("EAN13", "UPCA", "UPCE", "EAN8")
PAD_PX = 16
INSTALL = "pip install 'alloy-engine[barcode]' (or: pip install zxing-cpp)"


class DecoderUnavailable(ImportError):
    """zxing-cpp is not installed (the optional ``barcode`` extra)."""


def _zxing():
    try:
        import zxingcpp
    except ImportError as exc:
        raise DecoderUnavailable(f"the barcode decoder needs zxing-cpp: {INSTALL}") from exc
    return zxingcpp


def available() -> bool:
    """True when zxing-cpp can be imported."""
    try:
        _zxing()
        return True
    except DecoderUnavailable:
        return False


def gtin14(text: str) -> str:
    """An EAN / UPC reading as GTIN-14 (digits only, zero-padded)."""
    return "".join(c for c in text if c.isdigit()).zfill(14)


def _fmt(name: str) -> str:
    return str(name).replace("-", "").replace(" ", "").replace("BarcodeFormat.", "")


def normalise(fmt: str, text: str) -> str:
    """The comparable text of a reading: GTIN-14 for EAN / UPC, the data otherwise."""
    return gtin14(text) if _fmt(fmt) in RETAIL else text


@dataclass(frozen=True)
class Read:
    """One decoded symbol: normalised ``text``, zxing ``format``, ``center`` (x, y) px in the frame, ``passes``."""

    text: str
    format: str
    center: tuple[float, float]
    passes: int


def _passes(frame: np.ndarray, site: str) -> list[tuple[str, str, tuple[float, float]]]:
    """(format, normalised text, centre px) of every symbol of every pass."""
    from scipy.ndimage import zoom

    zx = _zxing()
    fmts = zx.barcode_formats_from_str(",".join(FORMATS[site]))
    padded = np.pad(frame, PAD_PX, mode="edge")
    images = ((padded, 1.0), (np.clip(zoom(padded.astype(np.float32), 2.0, order=3), 0, 255).astype(np.uint8), 2.0))
    out = []
    for binarizer in (zx.Binarizer.LocalAverage, zx.Binarizer.GlobalHistogram, zx.Binarizer.FixedThreshold):
        for img, scale in images:
            for r in zx.read_barcodes(img, formats=fmts, try_rotate=True, try_downscale=True, try_invert=False,
                                      binarizer=binarizer):
                p = r.position
                cx = (p.top_left.x + p.top_right.x + p.bottom_right.x + p.bottom_left.x) / 4.0 / scale - PAD_PX
                cy = (p.top_left.y + p.top_right.y + p.bottom_right.y + p.bottom_left.y) / 4.0 / scale - PAD_PX
                out.append((_fmt(r.format), normalise(str(r.format), r.text), (cx, cy)))
    return out


def decode(frame: np.ndarray, site: str = "inventory", aim: tuple[float, float] | None = None) -> Read | None:
    """The symbol a scanner reports for one (h, w) uint8 frame, or None."""
    if frame.ndim != 2 or frame.dtype != np.uint8:
        raise ValueError(f"decode takes one (h, w) uint8 frame, got {frame.shape} {frame.dtype}")
    if site not in FORMATS:
        raise ValueError(f"site {site!r}: use one of {sorted(FORMATS)}")
    seen: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for fmt, text, c in _passes(frame, site):
        seen.setdefault((fmt, text), []).append(c)
    good = [(k, cs) for k, cs in seen.items() if len(cs) >= (1 if k[0] in ONE_PASS else 2)]
    if not good:
        return None
    ax, ay = aim if aim is not None else (frame.shape[1] / 2.0, frame.shape[0] / 2.0)

    def dist(item):
        cx, cy = np.mean(item[1], axis=0)
        return (cx - ax) ** 2 + (cy - ay) ** 2

    (fmt, text), cs = min(good, key=dist)
    cx, cy = np.mean(cs, axis=0)
    return Read(text, fmt, (float(cx), float(cy)), len(cs))


__all__ = ["DecoderUnavailable", "FORMATS", "Read", "available", "decode", "gtin14", "normalise"]
