# SPDX-License-Identifier: Apache-2.0
"""H.264 video of the picked worlds' policy camera, encoded on the GPU (NVENC through PyNvVideoCodec), off the
training thread: all picked worlds of a worker in ONE mosaic stream.

The training thread only hands over the camera batch and the picked worlds (:meth:`VideoTap.offer`: one device-side
gather and a CUDA event, no host sync). An encoder thread runs on its own clock at ``fps``: every tick it lays the
newest images out as tiles (4 x 2 tiles of the camera's own size, 512 x 256 for a 128 px camera) and encodes that
one frame, repeating the last images when training has produced none (cheap for NVENC). So the stream has a steady
frame rate and NVENC's fixed 3-frame output delay stays 3/fps seconds (0.2 s at 15 fps). Each packet carries the
worlds of its tiles and the capture time of its images; the browser crops the open world's tile. Packets are
Annex-B access units with an IDR every ``gop`` frames (SPS/PPS repeated), so a new viewer or a dropped packet
recovers within one second.

Without NVENC (a CPU test) the tap falls back to the ``ffmpeg`` libx264 encoder through a pipe when ``ffmpeg`` is on
PATH, else video is off. The fallback exists for CPU tests only.

Example::

    tap = VideoTap(send=link_offer, fps=15, tile=128)
    tap.offer(obs["front"], picked_worlds)      # (W, H, W, 3) uint8 CUDA tensor, list of world indices
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from alloy.live.gate import GATE

log = logging.getLogger(__name__)
Send = Callable[[list[int], float, bool, bytes], bool]   # (tile worlds, t, key, annexb) -> accepted


class _Encoder:
    """One NVENC session for one world."""

    def __init__(self, width: int, height: int, fps: int, gop: int, bitrate: int) -> None:
        import PyNvVideoCodec as nvc  # noqa: PLC0415

        self.enc = nvc.CreateEncoder(width, height, "ABGR", False, codec="h264", preset="P1",
                                     tuning_info="ultra_low_latency", rc="cbr", bitrate=bitrate, fps=fps, gop=gop,
                                     idrperiod=gop, bf=0, repeatspspps=1)

    def encode(self, rgba: Any) -> list[tuple[bool, bytes]]:
        return [(p.get("picture_type", 0) == 3 or _has_idr(p["data"]), bytes(p["data"])) for p in self.enc.Encode(rgba)]

    def close(self) -> None:
        try:
            self.enc.EndEncode()
        except Exception:  # noqa: BLE001 - closing a session that never encoded can raise
            pass


class _X264:
    """CPU fallback: ffmpeg libx264 through pipes (tests without a GPU)."""

    def __init__(self, width: int, height: int, fps: int, gop: int, bitrate: int) -> None:
        self.p = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r",
             str(fps), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", str(gop),
             "-bf", "0", "-b:v", str(bitrate), "-x264-params", "repeat-headers=1", "-f", "h264", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._buf = bytearray()
        self._lock = threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        while True:
            chunk = self.p.stdout.read1(1 << 16)
            if not chunk:
                return
            with self._lock:
                self._buf += chunk

    def encode(self, rgba: Any) -> list[tuple[bool, bytes]]:
        self.p.stdin.write(rgba.cpu().numpy().tobytes())
        self.p.stdin.flush()
        with self._lock:
            data, self._buf = bytes(self._buf), bytearray()
        return [(_has_idr(data), data)] if data else []

    def close(self) -> None:
        self.p.kill()


def _has_idr(data: bytes) -> bool:
    """True when an Annex-B access unit holds an IDR slice (NAL type 5)."""
    i = data.find(b"\x00\x00\x01")
    while i >= 0 and i + 3 < len(data):
        if data[i + 3] & 0x1F == 5:
            return True
        i = data.find(b"\x00\x00\x01", i + 3)
    return False


def nvenc_available() -> bool:
    try:
        import PyNvVideoCodec  # noqa: F401, PLC0415
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a missing libnvidia-encode raises at import
        return False


class VideoTap:
    """Newest images of the picked worlds in, ONE steady H.264 mosaic stream out (``cols`` x ``rows`` tiles of the
    policy camera's own size; a world's tile index travels with every packet, the browser crops its tile).

    One encode per tick for all picked worlds keeps the encoder's share of the GPU, the GIL and the GPU gate small
    (eight per-world encoders cost the training thread about 7 %; see docs/notes/live-test.md)."""

    def __init__(self, send: Send, *, fps: int = 15, tile: int = 128, cols: int = 4, rows: int = 2,
                 gop: int | None = None, bitrate: int = 1_500_000) -> None:
        self.send, self.fps, self.tile, self.cols, self.rows = send, int(fps), int(tile), int(cols), int(rows)
        self.gop, self.bitrate = int(gop or fps), int(bitrate)
        self.backend = "nvenc" if nvenc_available() else ("x264" if shutil.which("ffmpeg") else "off")
        self._latest: tuple[Any, Any, float, list[int]] | None = None   # (images, ready event, capture t, worlds)
        self._pending: list[tuple[float, list[int]]] = []    # frames inside NVENC's 3-frame delay
        self._lock = threading.Lock()
        self._enc: Any = None
        self._mosaic: Any = None
        self._stop = threading.Event()
        self.frames = self.dropped = 0
        self.paused = False
        self._thread = None
        if self.backend != "off":
            self._thread = threading.Thread(target=self._run, name="alloy-live-video", daemon=True)
            self._thread.start()
        log.info("live video: %s, %d fps, %dx%d tiles of %d px", self.backend, self.fps, self.cols, self.rows,
                 self.tile)

    @property
    def slots(self) -> int:
        return self.cols * self.rows

    # --- training thread -------------------------------------------------------------------------------------------

    def offer(self, images: Any, worlds: list[int]) -> None:
        """The (W, H, W, 3) uint8 camera batch and the picked worlds: one gather on the device, one event."""
        if self.backend == "off" or not worlds:
            return
        import torch  # noqa: PLC0415

        worlds = list(worlds)[: self.slots]
        idx = torch.as_tensor(worlds, device=images.device)
        picked = images.index_select(0, idx)
        ev = None
        if picked.is_cuda:
            ev = torch.cuda.Event()
            ev.record()
        with self._lock:
            self._latest = (picked, ev, time.time(), worlds)

    def keep(self, worlds: set[int]) -> None:
        """Picks changed: the next offer carries the new list (nothing to tear down: one stream)."""

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._enc is not None:
            with GATE.background():
                self._enc.close()

    # --- encoder thread --------------------------------------------------------------------------------------------

    def _run(self) -> None:
        import torch  # noqa: PLC0415

        stream = torch.cuda.Stream() if self.backend == "nvenc" else None
        period, nxt = 1.0 / self.fps, time.monotonic()
        while not self._stop.is_set():
            nxt += period
            with self._lock:
                latest = None if self.paused else self._latest
            if latest is not None:
                self._encode(latest, stream)
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()        # fell behind: skip ticks rather than burst

    def _encode(self, latest: tuple[Any, Any, float, list[int]], stream: Any) -> None:
        import torch  # noqa: PLC0415

        images, ev, t_cap, worlds = latest
        try:
            with GATE.background():           # every CUDA call here stays out of the engine's graph capture
                if ev is not None:
                    ev.synchronize()
                if self._enc is None:
                    w, h = self.cols * self.tile, self.rows * self.tile
                    self._enc = (_Encoder if self.backend == "nvenc" else _X264)(w, h, self.fps, self.gop,
                                                                                self.bitrate)
                ctx = torch.cuda.stream(stream) if stream is not None else _Null()
                with ctx:
                    rgba = self._compose(images)
                    if stream is not None:
                        stream.synchronize()
                packets = self._enc.encode(rgba)
            self._pending.append((t_cap, worlds))      # packets leave in input order: stamp each with its input
            for key, data in packets:
                self.frames += 1
                t, tiles = self._pending.pop(0) if self._pending else (t_cap, worlds)
                if not self.send(tiles, t, key, data):
                    self.dropped += 1
        except Exception:  # noqa: BLE001 - video must never take training down
            log.exception("live video: encode failed; stream reset")
            self._pending.clear()
            bad, self._enc = self._enc, None
            if bad is not None:
                with GATE.background():
                    bad.close()

    def _compose(self, images: Any) -> Any:
        """(k, H, W, 3) -> the (rows*T, cols*T, 4) RGBA mosaic, tiles in row-major order, unused tiles black."""
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        T, k = self.tile, images.shape[0]
        x = images
        if x.shape[1] != T or x.shape[2] != T:
            x = F.interpolate(x.permute(0, 3, 1, 2).float(), size=(T, T), mode="bilinear",
                              align_corners=False).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        if self._mosaic is None:
            self._mosaic = torch.zeros(self.slots, T, T, 4, dtype=torch.uint8, device=images.device)
            self._mosaic[..., 3] = 255
        self._mosaic[:k, ..., :3] = x
        self._mosaic[k:, ..., :3] = 0
        m = self._mosaic.view(self.rows, self.cols, T, T, 4).permute(0, 2, 1, 3, 4)
        return m.reshape(self.rows * T, self.cols * T, 4).contiguous()


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None
