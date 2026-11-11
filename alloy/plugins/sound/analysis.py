# SPDX-License-Identifier: Apache-2.0
"""Host-side audio analysis for ``ModalSound`` output and recordings: WAV I/O, spectra, peaks, decay and centroid.

Ported from DC6's click sound (``alloy/tasks/dc_click_sound.py``), plus:
- ``peak_near``: sub-bin peak frequency, by a parabola on the log magnitude;
- ``mode_decay_rate``: one mode's envelope decay rate, from an FFT band-pass and a Hilbert envelope (law S-2);
- ``centroid``: the magnitude-weighted spectral centroid (law S-5).
Every function takes plain float arrays and the sample rate ``fs`` [Hz].
"""

from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

BAND_EDGES_HZ = (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


@dataclass(frozen=True)
class SpectrumFeatures:
    """Summary of one recording or synthesis."""

    peaks_hz: list[float]
    centroid_hz: float
    decay_20db_s: float
    band_db: dict[str, float]


def write_wav(path: str, samples: np.ndarray, fs: int, normalise: bool = True) -> float:
    """16-bit PCM mono WAV (stdlib ``wave``). Normalised to peak 0.9 unless ``normalise`` is False (then clipped
    to ±1). Returns the scale applied."""
    x = np.asarray(samples, float)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    scale = 0.9 / peak if (normalise and peak > 0.0) else 1.0
    pcm = np.clip(x * scale, -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(np.round(pcm * 32767).astype("<i2").tobytes())
    return scale


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Mono 16-bit WAV as floats in [-1, 1], and its sample rate."""
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, "<i2").astype(np.float64) / 32767.0, wf.getframerate()


def magnitude_spectrum(samples: np.ndarray, fs: int, nfft: int = 1 << 16) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed one-sided magnitude spectrum: (freqs_hz, mag)."""
    x = np.asarray(samples, float)
    nfft = max(nfft, 1 << int(np.ceil(np.log2(max(x.size, 2)))))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), nfft))
    return np.fft.rfftfreq(nfft, 1.0 / fs), spec


def top_peaks(freqs: np.ndarray, mag: np.ndarray, k: int = 5, min_sep_hz: float = 200.0,
              fmin: float = 100.0) -> list[float]:
    """The ``k`` strongest local maxima at least ``min_sep_hz`` apart, above ``fmin`` (greedy)."""
    loc = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
    loc = loc[freqs[loc] >= fmin]
    chosen: list[float] = []
    for i in loc[np.argsort(mag[loc])[::-1]]:
        if all(abs(freqs[i] - c) >= min_sep_hz for c in chosen):
            chosen.append(float(freqs[i]))
        if len(chosen) == k:
            break
    return chosen


def peak_near(freqs: np.ndarray, mag: np.ndarray, f0: float, rel_window: float = 0.05) -> float:
    """The spectral peak within ±``rel_window``·f0 of ``f0``, refined by a parabola through log magnitudes."""
    sel = np.where(np.abs(freqs - f0) <= rel_window * f0)[0]
    if sel.size < 3:
        return float("nan")
    i = int(sel[np.argmax(mag[sel])])
    if i <= 0 or i >= len(mag) - 1:
        return float(freqs[i])
    a, b, c = np.log(np.maximum(mag[i - 1:i + 2], 1e-300))
    den = a - 2.0 * b + c
    shift = 0.0 if den == 0.0 else 0.5 * (a - c) / den
    return float(freqs[i] + shift * (freqs[1] - freqs[0]))


def mode_decay_rate(samples: np.ndarray, fs: int, f0: float, rel_band: float = 0.1, skip_s: float = 0.005,
                    floor_db: float = 40.0) -> float:
    """Envelope decay rate [1/s] (σ in exp(-σt)) of the component near ``f0``. Band-pass ±``rel_band``·f0 in the
    FFT domain, Hilbert envelope, then a least-squares line on the log envelope from ``skip_s`` after its peak
    until it is ``floor_db`` below the peak."""
    x = np.asarray(samples, float)
    n = x.size
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spec[np.abs(f - f0) > rel_band * f0] = 0.0
    env = np.abs(hilbert(np.fft.irfft(spec, n)))
    ip = int(np.argmax(env))
    start = ip + int(skip_s * fs)
    tail = np.where(env[start:] < env[ip] * 10 ** (-floor_db / 20.0))[0]
    stop = start + (int(tail[0]) if tail.size else n - start - int(0.05 * n))
    if stop - start < 16:
        return float("nan")
    t = np.arange(start, stop) / fs
    slope = np.polyfit(t, np.log(np.maximum(env[start:stop], 1e-300)), 1)[0]
    return float(-slope)


def centroid(samples: np.ndarray, fs: int, fmin: float = 0.0) -> float:
    """Magnitude-weighted spectral centroid [Hz] above ``fmin``."""
    freqs, mag = magnitude_spectrum(samples, fs)
    m = np.where(freqs >= fmin, mag, 0.0)
    return float(np.sum(freqs * m) / max(np.sum(m), 1e-300))


def rms_envelope(samples: np.ndarray, fs: int, window_s: float = 1e-3) -> np.ndarray:
    """Centred moving RMS (window clipped at the edges)."""
    x2 = np.asarray(samples, float) ** 2
    half = max(1, int(round(window_s * fs / 2)))
    cs = np.concatenate([[0.0], np.cumsum(x2)])
    i = np.arange(x2.size)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, x2.size)
    return np.sqrt((cs[hi] - cs[lo]) / (hi - lo))


def decay_time_20db(samples: np.ndarray, fs: int) -> float:
    """Time from the envelope peak to its first fall below -20 dB [s]; nan if never."""
    env = rms_envelope(samples, fs)
    ip = int(np.argmax(env))
    below = np.where(env[ip:] < 0.1 * env[ip])[0]
    return float(below[0] / fs) if below.size else float("nan")


def band_levels_db(freqs: np.ndarray, mag: np.ndarray, edges: tuple[float, ...] = BAND_EDGES_HZ) -> dict[str, float]:
    """Band energies in dB relative to the total in-band energy."""
    e = mag ** 2
    pairs = list(zip(edges[:-1], edges[1:]))
    bands = [e[(freqs >= lo) & (freqs < hi)].sum() for lo, hi in pairs]
    total = max(sum(bands), 1e-300)
    return {f"{lo:g}-{hi:g}": float(10 * np.log10(max(b, 1e-300) / total)) for (lo, hi), b in zip(pairs, bands)}


def spectrum_features(samples: np.ndarray, fs: int) -> SpectrumFeatures:
    freqs, mag = magnitude_spectrum(samples, fs)
    cen = float(np.sum(freqs * mag) / max(np.sum(mag), 1e-300))
    return SpectrumFeatures(top_peaks(freqs, mag), cen, decay_time_20db(samples, fs), band_levels_db(freqs, mag))


def compare(sim: SpectrumFeatures, real: SpectrumFeatures) -> dict:
    """Peak matches, centroid ratio, decay ratio and band level differences (sim against a recording)."""
    matches = []
    for f in sim.peaks_hz:
        near = min(real.peaks_hz, key=lambda r: abs(r - f))
        matches.append({"sim_hz": f, "nearest_real_hz": near, "err_hz": f - near,
                        "err_pct": 100.0 * (f - near) / near})
    return {"peak_matches": matches,
            "centroid_ratio_sim_over_real": sim.centroid_hz / real.centroid_hz,
            "decay_ratio_sim_over_real": sim.decay_20db_s / real.decay_20db_s,
            "band_diff_db_sim_minus_real": {k: sim.band_db[k] - real.band_db[k] for k in sim.band_db}}


__all__ = ["BAND_EDGES_HZ", "SpectrumFeatures", "band_levels_db", "centroid", "compare", "decay_time_20db",
           "magnitude_spectrum", "mode_decay_rate", "peak_near", "read_wav", "rms_envelope", "spectrum_features",
           "top_peaks", "write_wav"]
