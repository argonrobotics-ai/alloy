# SPDX-License-Identifier: Apache-2.0
"""Encode rendered frames into h264 mp4 or ProRes 4444 mov with ffmpeg."""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


class EncodeError(RuntimeError):
    """ffmpeg is missing, there are no frames, or it failed."""


def frame_files(out: Path) -> list[Path]:
    """Sorted PNG frames in OUT/frames (EXR cannot be encoded directly)."""
    frames = sorted((out / "frames").glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if not frames:
        raise EncodeError(f"no .png frames in {out / 'frames'} (EXR output must be converted first)")
    return frames


def ffmpeg_args(list_file: Path, dest: Path, fps: float, crf: int, prores: bool) -> list[str]:
    """ffmpeg argument vector for the concat list."""
    base = ["ffmpeg", "-y", "-loglevel", "error", "-r", str(fps), "-f", "concat", "-safe", "0", "-i", str(list_file)]
    if prores:
        return base + ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-r", str(fps), str(dest)]
    return base + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow",
                   "-movflags", "+faststart", "-r", str(fps), str(dest)]


def encode(out: Path, fps: float = 30.0, crf: int = 16, prores: bool = False) -> Path:
    """Encode OUT/frames/*.png (in bundle-frame order) to OUT/clip.mp4 or OUT/clip.mov."""
    if shutil.which("ffmpeg") is None:
        raise EncodeError("ffmpeg not found on PATH")
    frames = frame_files(out)
    dest = out / ("clip.mov" if prores else "clip.mp4")
    with tempfile.TemporaryDirectory() as tmp:
        list_file = Path(tmp) / "frames.txt"
        list_file.write_text("".join(f"file '{p.resolve()}'\nduration {1.0 / fps:.9f}\n" for p in frames))
        res = subprocess.run(ffmpeg_args(list_file, dest, fps, crf, prores), capture_output=True, text=True)
    if res.returncode != 0:
        raise EncodeError(f"ffmpeg failed (rc {res.returncode}): {res.stderr.strip()[-500:]}")
    log.info("encoded %d frames to %s", len(frames), dest)
    return dest


GCS_BATCH = 50


def encode_gcs(prefix: str, dest: Path, fps: float = 30.0, crf: int = 16,
               lister: Callable[[str], list[str]] | None = None) -> Path:
    """Encode PNG frames stored under a gs:// prefix to `dest` without writing them to disk: the objects are
    streamed in frame order (`gcloud storage cat`, GCS_BATCH at a time) into ffmpeg's image2pipe input."""
    if shutil.which("ffmpeg") is None:
        raise EncodeError("ffmpeg not found on PATH")
    from alloy.render.farm_frames import gcs_frame_names
    from alloy.render.farm_util import frame_index
    names = sorted((n for n in (lister or gcs_frame_names)(prefix) if n.endswith(".png") and frame_index(n) is not None),
                   key=frame_index)
    if not names:
        raise EncodeError(f"no .png frames under {prefix}")
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "png",
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow",
            "-movflags", "+faststart", str(dest)]
    base = prefix.rstrip("/")
    ff = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(0, len(names), GCS_BATCH):
            urls = [f"{base}/{n}" for n in names[i:i + GCS_BATCH]]
            cat = subprocess.run(["gcloud", "storage", "cat", *urls], stdout=ff.stdin, stderr=subprocess.PIPE)
            if cat.returncode != 0:
                raise EncodeError(f"gcloud storage cat failed at {urls[0]}: {cat.stderr.decode()[-300:]}")
    finally:
        ff.stdin.close()
        err = ff.stderr.read().decode()
        rc = ff.wait()
    if rc != 0:
        raise EncodeError(f"ffmpeg failed (rc {rc}): {err.strip()[-500:]}")
    got = video_frames(dest)
    if got is not None and got != len(names):
        raise EncodeError(f"{dest} has {got} frames, expected {len(names)} from {prefix} (a damaged frame object?)")
    log.info("encoded %d frames from %s to %s", len(names), prefix, dest)
    return dest


def video_frames(path: Path) -> int | None:
    """Frame count of a video's first stream (ffprobe container metadata); None when ffprobe cannot tell."""
    if shutil.which("ffprobe") is None:
        return None
    res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    out = res.stdout.strip()
    return int(out) if res.returncode == 0 and out.isdigit() else None
