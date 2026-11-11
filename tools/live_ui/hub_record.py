# SPDX-License-Identifier: Apache-2.0
"""The hub's recorder: everything streamed is saved, indexed by time, so the timeline can show any past moment.

Layout of one run directory:
- ``numbers.jsonl``: every worker's per-world numbers, one line per sample (``t``, worker, offset, fields).
- ``state/<gw>.jsonl``: 3D frames of global world ``gw`` while it was picked.
- ``video/<worker>.h264`` + ``video/<worker>.idx``: each worker's mosaic stream (Annex-B) and one
  ``t offset length key tile cols worlds`` line per packet (``worlds``: the global world of each tile).
- ``commands.jsonl``: every applied command (the worker's log entry plus the hub's receive time).
- ``geometry/<gw>.jsonl``: world ``gw``'s topology from the frame export, one line per change.
Indexes are rebuilt from the files when the hub restarts, so a recording outlives the hub process.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _Index:
    """Times and byte offsets of one append-only file."""
    t: list[float] = field(default_factory=list)
    off: list[int] = field(default_factory=list)
    size: list[int] = field(default_factory=list)
    key: list[bool] = field(default_factory=list)
    meta: list[dict] = field(default_factory=list)

    def add(self, t: float, off: int, size: int, key: bool = True, meta: dict | None = None) -> None:
        self.t.append(t)
        self.off.append(off)
        self.size.append(size)
        self.key.append(key)
        self.meta.append(meta or {})

    def at(self, t: float) -> int:
        """Index of the last entry at or before ``t`` (-1 when none)."""
        return bisect.bisect_right(self.t, t) - 1


class Recorder:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for sub in ("state", "video", "geometry"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.numbers_idx = _Index()
        self.state_idx: dict[int, _Index] = {}
        self.video_idx: dict[str, _Index] = {}             # per worker stream
        self._files: dict[str, Any] = {}
        self._load()

    # --- writing ---------------------------------------------------------------------------------------------------

    def _append(self, rel: str, data: bytes) -> int:
        f = self._files.get(rel)
        if f is None:
            f = self._files[rel] = open(self.root / rel, "ab")  # noqa: SIM115 - kept open for the hub's life
        off = f.tell()
        f.write(data)
        f.flush()
        return off

    def numbers(self, row: dict[str, Any]) -> None:
        line = (json.dumps(row, separators=(",", ":")) + "\n").encode()
        self.numbers_idx.add(row["t"], self._append("numbers.jsonl", line), len(line))

    def state(self, gw: int, t: float, frame: dict[str, Any]) -> None:
        line = (json.dumps({"t": t, **frame}, separators=(",", ":")) + "\n").encode()
        idx = self.state_idx.setdefault(gw, _Index())
        idx.add(t, self._append(f"state/{gw}.jsonl", line), len(line))

    def video(self, stream: str, t: float, key: bool, data: bytes, meta: dict) -> None:
        idx = self.video_idx.setdefault(stream, _Index())
        off = self._append(f"video/{stream}.h264", data)
        idx.add(t, off, len(data), key, meta)
        worlds = ",".join(str(x) for x in meta["tiles"]) or "-"
        self._append(f"video/{stream}.idx",
                     f"{t:.4f} {off} {len(data)} {int(key)} {meta['tile']} {meta['cols']} {worlds}\n".encode())

    def command(self, row: dict[str, Any]) -> None:
        self._append("commands.jsonl", (json.dumps(row, separators=(",", ":")) + "\n").encode())

    def topology(self, gw: int, t: float, topo: dict[str, Any]) -> None:
        """A world's topology from the frame export (names, shapes, statics, mesh faces), on change."""
        self._append(f"geometry/{gw}.jsonl", (json.dumps({"t": t, **topo}, separators=(",", ":")) + "\n").encode())

    def topologies(self) -> dict[int, dict[str, Any]]:
        """The newest recorded topology of every world (after a hub restart)."""
        out = {}
        for p in (self.root / "geometry").glob("*.jsonl"):
            lines = p.read_text().splitlines()
            if lines:
                topo = json.loads(lines[-1])
                topo.pop("t", None)
                out[int(p.stem)] = topo
        return out

    # --- reading (timeline) ----------------------------------------------------------------------------------------

    def span(self) -> tuple[float, float] | None:
        t = self.numbers_idx.t
        return (t[0], t[-1]) if t else None

    def _read(self, rel: str, off: int, size: int) -> bytes:
        f = self._files.get(rel)
        if f is not None:
            f.flush()
        with open(self.root / rel, "rb") as r:
            r.seek(off)
            return r.read(size)

    def numbers_at(self, t: float, window: float = 2.5) -> list[dict[str, Any]]:
        """The newest sample of each worker at or before ``t`` (within ``window`` seconds)."""
        i, out, seen = self.numbers_idx.at(t), [], set()
        while i >= 0 and self.numbers_idx.t[i] >= t - window:
            row = json.loads(self._read("numbers.jsonl", self.numbers_idx.off[i], self.numbers_idx.size[i]))
            if row["worker"] not in seen:
                seen.add(row["worker"])
                out.append(row)
            i -= 1
        return out

    def state_at(self, gw: int, t: float) -> dict[str, Any] | None:
        idx = self.state_idx.get(gw)
        if idx is None:
            return None
        i = idx.at(t)
        if i < 0:
            return None
        return json.loads(self._read(f"state/{gw}.jsonl", idx.off[i], idx.size[i]))

    def video_window(self, stream: str, t0: float, t1: float) -> list[tuple[float, bool, bytes, dict]]:
        """Packets from the last key frame at or before ``t0`` up to ``t1`` (enough to show any frame in between)."""
        idx = self.video_idx.get(stream)
        if idx is None:
            return []
        i = idx.at(t0)
        while i > 0 and not idx.key[i]:
            i -= 1
        i = max(i, 0)
        out = []
        while i < len(idx.t) and idx.t[i] <= t1:
            out.append((idx.t[i], idx.key[i], self._read(f"video/{stream}.h264", idx.off[i], idx.size[i]),
                        idx.meta[i]))
            i += 1
        return out

    def recorded_worlds(self) -> list[int]:
        return sorted(self.state_idx)

    # --- restart ---------------------------------------------------------------------------------------------------

    def _load(self) -> None:
        nums = self.root / "numbers.jsonl"
        if nums.exists():
            off = 0
            with nums.open("rb") as f:
                for line in f:
                    try:
                        self.numbers_idx.add(json.loads(line)["t"], off, len(line))
                    except (ValueError, KeyError):
                        pass
                    off += len(line)
        for p in (self.root / "state").glob("*.jsonl"):
            idx, off = self.state_idx.setdefault(int(p.stem), _Index()), 0
            with p.open("rb") as f:
                for line in f:
                    try:
                        idx.add(json.loads(line)["t"], off, len(line))
                    except (ValueError, KeyError):
                        pass
                    off += len(line)
        for p in (self.root / "video").glob("*.idx"):
            idx = self.video_idx.setdefault(p.stem, _Index())
            for line in p.read_text().splitlines():
                t, off, size, key, tile, cols, worlds = line.split()
                tiles = [] if worlds == "-" else [int(x) for x in worlds.split(",")]
                idx.add(float(t), int(off), int(size), key == "1", {"tile": int(tile), "cols": int(cols), "tiles": tiles})
