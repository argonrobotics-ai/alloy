# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for the render farm: frame splitting, zone cooldowns, timing parse, cost log."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

FRAME_EXTS = (".png", ".exr")
ZONE_COOLDOWN_S = 600.0


@dataclass
class CostLog:
    """Accumulates VM billing, preemptions and timings for render_log.json."""

    vm_seconds: float = 0.0
    zones: list[str] = field(default_factory=list)
    preemptions: int = 0
    timings: list[float] = field(default_factory=list)
    blender_version: str = ""
    stockouts: int = 0
    quota_hits: int = 0
    regions: dict = field(default_factory=dict)
    setup_s: list = field(default_factory=list)
    push_s: list = field(default_factory=list)
    stage_up_s: float = 0.0
    late_joiners: int = 0

    def add_vm(self, zone: str, seconds: float) -> None:
        self.vm_seconds += max(0.0, seconds)
        if zone not in self.zones:
            self.zones.append(zone)


class RegionBook:
    """Placement state: zone preference order, per-region VM counts and cap, zone and region cooldowns.

    Stockout cools one zone; a quota error cools the whole region (quota is per region). `reserve_next` picks the
    region with the fewest of our VMs first, then that region's zones in list order, and reserves the slot under a
    lock, so parallel creations spread instead of piling into the first region.
    """

    def __init__(self, zones: tuple[str, ...], region_cap: int = 8, cooldown_s: float = ZONE_COOLDOWN_S,
                 clock: Callable[[], float] = time.time) -> None:
        self.zones, self.region_cap, self.cooldown_s, self.clock = zones, max(1, region_cap), cooldown_s, clock
        self.regions: list[str] = []
        for z in zones:
            if _region(z) not in self.regions:
                self.regions.append(_region(z))
        self.counts: dict[str, int] = {r: 0 for r in self.regions}
        self._zone_until: dict[str, float] = {}
        self._region_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _zone_free_at(self, zone: str) -> float:
        return max(self._zone_until.get(zone, 0.0), self._region_until.get(_region(zone), 0.0))

    def ordered(self, probe: int = 3) -> list[str]:
        """Candidate zones now: fewest-VMs region first. If everything is cooling, probe the soonest to recover."""
        now = self.clock()
        open_regions = [r for r in self.regions if self.counts[r] < self.region_cap]
        open_regions.sort(key=lambda r: (self.counts[r], self.regions.index(r)))
        free = [z for r in open_regions for z in self.zones if _region(z) == r and self._zone_free_at(z) <= now]
        if free or not open_regions:
            return free
        cooling = [z for z in self.zones if _region(z) in open_regions]
        return sorted(cooling, key=lambda z: (self._zone_free_at(z), self.zones.index(z)))[:probe]

    def reserve_next(self, tried: set[str]) -> str | None:
        """Reserve a slot in the best untried zone; None when nothing is available."""
        with self._lock:
            for zone in self.ordered():
                if zone not in tried:
                    self.counts[_region(zone)] += 1
                    return zone
        return None

    def adopt(self, zone: str) -> None:
        with self._lock:
            self.counts[_region(zone)] = self.counts.get(_region(zone), 0) + 1

    def release(self, zone: str) -> None:
        with self._lock:
            r = _region(zone)
            self.counts[r] = max(0, self.counts.get(r, 0) - 1)

    def cool_zone(self, zone: str) -> None:
        self._zone_until[zone] = self.clock() + self.cooldown_s

    def cool_region(self, region: str) -> None:
        self._region_until[region] = self.clock() + self.cooldown_s

    def used_regions(self) -> list[str]:
        return [r for r in self.regions if self.counts[r] > 0]


def _region(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def split_chunks(frames: list[int], n: int) -> list[list[int]]:
    """Split a sorted frame list into at most n contiguous, near-equal chunks."""
    n = max(1, min(n, len(frames)))
    size, extra = divmod(len(frames), n)
    chunks, pos = [], 0
    for i in range(n):
        take = size + (1 if i < extra else 0)
        chunks.append(frames[pos:pos + take])
        pos += take
    return [c for c in chunks if c]


def to_runs(frames: list[int], step: int = 1) -> list[tuple[int, int]]:
    """Group sorted frames into (first, last) runs; consecutive means spaced by exactly step."""
    runs: list[tuple[int, int]] = []
    for f in sorted(frames):
        if runs and f - runs[-1][1] == step:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def frame_index(name: str) -> int | None:
    """Frame number of a finished frame file name ("00012.png" -> 12); None for anything else, e.g. the
    ".00012.part.png" Blender writes while a frame is still rendering."""
    stem, dot, ext = name.rpartition(".")
    return int(stem) if dot and "." + ext in FRAME_EXTS and stem.isdigit() else None


def existing_frames(out: Path) -> set[int]:
    """Frame indices already present in OUT/frames."""
    found = set()
    for p in (out / "frames").glob("*"):
        if p.suffix in FRAME_EXTS and p.stem.isdigit():
            found.add(int(p.stem))
    return found


def parse_timing(lines: list[str]) -> list[float]:
    """Per-frame render seconds from timing.jsonl lines ({"frame": k, "seconds": s}; other rows skipped)."""
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("summary") or "frame" not in rec:
            continue
        for key in ("seconds", "render_s"):
            if isinstance(rec.get(key), (int, float)):
                out.append(float(rec[key]))
                break
    return out


def batch_size(pool_left: int, workers: int, minimum: int = 2, maximum: int = 32) -> int:
    """Frames per claim (guided): half of an even share of what is left, so the tail is split finely and every
    worker stays busy to the end, but never below `minimum` (Blender start-up cost) or above `maximum`."""
    share = -(-pool_left // (2 * max(1, workers)))
    return max(1, min(pool_left, max(minimum, min(maximum, share))))


def render_summary(c: "CostLog", stager: Any, total: int, rendered: int, wall_s: float) -> dict:
    """The render_log.json content for a finished farm run."""
    from alloy.render.look import SPOT_USD_PER_HOUR

    hours = c.vm_seconds / 3600.0
    per_frame = (sum(c.timings) / len(c.timings)) if c.timings else (c.vm_seconds / rendered if rendered else 0.0)
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else 0.0  # noqa: E731
    return {"frames": total, "frames_rendered_this_run": rendered, "seconds_per_frame": round(per_frame, 2),
            "gpu_hours": round(hours, 3), "spot_usd": round(hours * SPOT_USD_PER_HOUR, 3), "zones": c.zones,
            "regions": c.regions, "preemptions": c.preemptions, "stockouts": c.stockouts,
            "quota_hits": c.quota_hits, "late_joiners": c.late_joiners, "staged": stager is not None,
            "stage_upload_s": round(stager.upload_s, 1) if stager else 0.0,
            "vm_setup_s_mean": mean(c.setup_s), "vm_input_load_s_mean": mean(c.push_s),
            "blender": c.blender_version, "wall_s": round(wall_s, 1)}
