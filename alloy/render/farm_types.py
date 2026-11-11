# SPDX-License-Identifier: Apache-2.0
"""Data types of the spot render farm: the job, a VM serving batches, and one poll's outcome."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alloy.render.vm import ZONES, VmInfo


class FarmError(RuntimeError):
    """The farm could not finish the job."""


@dataclass
class FarmJob:
    """Everything a render run needs."""

    bundle: Path
    look: Path
    camera: str
    out: Path
    name: str
    frames: list[int]
    samples: int | None = None
    fmt: str = "png"
    vms: int = 4
    zones: tuple[str, ...] = ZONES
    region_cap: int = 8
    stage: str | None = None  # None = auto (on when vms > 8), "off", or a gs:// base
    keep_vms: bool = False
    blender_dir: Path | None = None
    assets_dir: Path | None = None
    cycles_dir: Path | None = None
    reuse_vms: tuple[str, ...] = ()
    poll_s: float = 30.0
    retry_s: float = 120.0
    no_vm_timeout_s: float = 1200.0
    ready_timeout_s: float = 1500.0
    vm_name: str | None = None  # name for the first created VM (still)
    frames_gcs: str | None = None  # gs:// prefix: VMs upload frames there and nothing is pulled here
    bundle_gcs: str | None = None  # the bundle lives here; `bundle` is only a sparse local mirror of it


@dataclass
class Worker:
    """A VM serving batches from the frame pool."""

    vm: VmInfo
    reused: bool = False
    batch: list[int] = field(default_factory=list)
    timing_seen: int = 0
    pulled: set[str] = field(default_factory=set)
    idle_since: float = 0.0
    start_fails: int = 0


@dataclass
class PollResult:
    """Outcome of polling one busy worker."""

    state: str  # running | done | gone
    first_frames: bool = False
    timings: list[float] = field(default_factory=list)
    rc: str = ""
