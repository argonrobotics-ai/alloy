# SPDX-License-Identifier: Apache-2.0
"""Barcode symbols as images, for label textures: EAN-13 / UPC-A (retail items) and Code 128 (location and library
labels), encoded here from the standards' tables (GS1 General Specifications section 5.2; ISO/IEC 15417), so label
artwork needs no extra package. Decoding is ``barcode_decode`` (zxing-cpp, the optional ``barcode`` extra).

- ``ean13_modules("590123412345")``: the 95 modules (1 = bar) of the EAN-13 symbol, check digit added; 12 digits
  starting with 0 are a UPC-A item. ``code128_modules("A-01-03")``: start, data (subset C for runs of digits, else
  B), check character and stop.
- ``Symbol``: the data, its symbology, the module (X dimension, m) and bar height; ``Symbol.image()`` rasterises it
  with its quiet zones; ``label_image`` puts symbols on a white label (row 0 = top, sRGB uint8, ready as a texture).

Sizes follow GS1: an EAN-13 at 100 % magnification has X = 0.330 mm, bars 22.85 mm high, quiet zones of 11 modules
left and 7 right (a 37.29 mm wide symbol); Code 128 quiet zones are 10 modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_L = ("0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011")
_R = tuple(code.translate(str.maketrans("01", "10")) for code in _L)
_G = tuple(code[::-1] for code in _R)
_PARITY = ("LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL")

# Code 128 bar/space widths of values 0..106 (103-105: start A/B/C, 106: stop), ISO/IEC 15417 table 1
_C128 = (
    "212222 222122 222221 121223 121322 131222 122213 122312 132212 221213 221312 231212 112232 122132 122231 113222 "
    "123122 123221 223211 221132 221231 213212 223112 312131 311222 321122 321221 312212 322112 322211 212123 212321 "
    "232121 111323 131123 131321 112313 132113 132311 211313 231113 231311 112133 112331 132131 113123 113321 133121 "
    "313121 211331 231131 213113 213311 213131 311123 311321 331121 312113 312311 332111 314111 221411 431111 111224 "
    "111422 121124 121421 141122 141221 112214 112412 122114 122411 142112 142211 241211 221114 413111 241112 134111 "
    "111242 121142 121241 114212 124112 124211 411212 421112 421211 212141 214121 412121 111143 111341 131141 114113 "
    "114311 411113 411311 113141 114131 311141 411131 211412 211214 211232 2331112").split()
START_B, START_C, CODE_B, CODE_C, STOP = 104, 105, 100, 99, 106
EAN_X = 0.330e-3                 # m, EAN-13 module at 100 % magnification
EAN_BAR_H = 22.85e-3             # m
QUIET = {"EAN13": (11, 7), "Code128": (10, 10)}


def ean13_check(digits12: str) -> int:
    """The EAN-13 check digit of 12 digits (weights 1, 3, 1, ... from the left)."""
    if len(digits12) != 12 or not digits12.isdigit():
        raise ValueError(f"EAN-13 needs 12 digits before the check digit, got {digits12!r}")
    s = sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(digits12))
    return (10 - s % 10) % 10


def ean13_text(data: str) -> str:
    """The 13-digit EAN-13 of ``data``: 12 digits (check added), 13 (check verified) or an 11-digit UPC-A item."""
    if len(data) == 11:
        data = "0" + data
    if len(data) == 13:
        if ean13_check(data[:12]) != int(data[12]):
            raise ValueError(f"EAN-13 {data}: check digit should be {ean13_check(data[:12])}")
        return data
    return data + str(ean13_check(data))


def ean13_modules(data: str) -> np.ndarray:
    """The 95 modules (uint8, 1 = bar) of the EAN-13 symbol of ``data`` (see ``ean13_text``)."""
    t = ean13_text(data)
    parity = _PARITY[int(t[0])]
    left = "".join((_L if p == "L" else _G)[int(c)] for p, c in zip(parity, t[1:7]))
    right = "".join(_R[int(c)] for c in t[7:])
    return np.frombuffer(("101" + left + "01010" + right + "101").encode(), np.uint8) - ord("0")


def _code128_values(text: str) -> list[int]:
    """Symbol values (start .. data, no check or stop): subset C for even runs of 4+ digits, else subset B."""
    if not text or any(not 32 <= ord(c) < 127 for c in text):
        raise ValueError(f"Code 128 here takes printable ASCII, got {text!r}")
    out: list[int] = []
    sub, i = None, 0
    while i < len(text):
        run = len(text[i:]) - len(text[i:].lstrip("0123456789"))
        if run >= 4 or (run >= 2 and run == len(text) - i and sub == "C"):
            n = run - run % 2
            out += [START_C if sub is None else CODE_C] if sub != "C" else []
            sub = "C"
            out += [int(text[i + k:i + k + 2]) for k in range(0, n, 2)]
            i += n
            continue
        out += [START_B if sub is None else CODE_B] if sub != "B" else []
        sub = "B"
        out.append(ord(text[i]) - 32)
        i += 1
    return out


def code128_modules(text: str) -> np.ndarray:
    """The modules (uint8, 1 = bar) of the Code 128 symbol of ``text``, check character and stop included."""
    vals = _code128_values(text)
    check = (vals[0] + sum(k * v for k, v in enumerate(vals[1:], 1))) % 103
    widths = "".join(_C128[v] for v in vals + [check, STOP])
    bits = [1 - (k % 2) for k, w in enumerate(widths) for _ in range(int(w))]
    return np.asarray(bits, np.uint8)


@dataclass(frozen=True)
class Symbol:
    """A printed symbol: ``data`` in ``kind`` ("EAN13" or "Code128"), module ``x`` (m) and bar height (m)."""

    data: str
    kind: str = "EAN13"
    x: float = EAN_X
    bar_h: float = EAN_BAR_H

    def modules(self) -> np.ndarray:
        if self.kind == "EAN13":
            return ean13_modules(self.data)
        if self.kind == "Code128":
            return code128_modules(self.data)
        raise ValueError(f"symbology {self.kind!r}: use EAN13 or Code128")

    def text(self) -> str:
        """What a correct decode reads: the 13 EAN digits, or the Code 128 data."""
        return ean13_text(self.data) if self.kind == "EAN13" else self.data

    def size(self) -> tuple[float, float]:
        """(width, height) in m, quiet zones included."""
        left, right = QUIET[self.kind]
        return (len(self.modules()) + left + right) * self.x, self.bar_h

    def image(self, px_per_module: int = 4) -> np.ndarray:
        """(H, W) uint8, 0 = bar, 255 = space, quiet zones included."""
        left, right = QUIET[self.kind]
        row = np.concatenate([np.zeros(left, np.uint8), self.modules(), np.zeros(right, np.uint8)])
        row = np.repeat(255 - 255 * row, px_per_module).astype(np.uint8)
        h = max(1, round(self.bar_h / self.x * px_per_module))
        return np.tile(row, (h, 1))


def label_image(symbol: Symbol, size: tuple[float, float], px_per_module: int = 4,
                paper: tuple[int, int, int] = (246, 244, 238)) -> np.ndarray:
    """A label ``size`` = (width, height) m with ``symbol`` centred: (H, W, 3) sRGB uint8, row 0 = top. The label
    must hold the symbol with its quiet zones."""
    sw, sh = symbol.size()
    if sw > size[0] + 1e-9 or sh > size[1] + 1e-9:
        raise ValueError(f"label {size} m is smaller than the symbol {sw:.4f} x {sh:.4f} m")
    ppm = px_per_module / symbol.x
    w, h = max(1, round(size[0] * ppm)), max(1, round(size[1] * ppm))
    out = np.empty((h, w, 3), np.uint8)
    out[:] = paper
    sym = symbol.image(px_per_module)
    y0, x0 = (h - sym.shape[0]) // 2, (w - sym.shape[1]) // 2
    out[y0:y0 + sym.shape[0], x0:x0 + sym.shape[1]] = np.where(sym[..., None] > 0, np.asarray(paper, np.uint8), 0)
    return out


__all__ = ["EAN_BAR_H", "EAN_X", "Symbol", "code128_modules", "ean13_check", "ean13_modules", "ean13_text",
           "label_image"]
