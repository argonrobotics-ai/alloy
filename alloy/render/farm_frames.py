# SPDX-License-Identifier: Apache-2.0
"""Where farm frames end up: pulled to OUT/frames on this machine, or uploaded by each VM straight to GCS.

In GCS mode (FarmJob.frames_gcs) the VMs push every finished frame to the prefix themselves and mark it under
OUT/uploaded on the VM; the farm only counts those marks, so no frame ever lands on the host. The clip is then
encoded by streaming the objects (encode.encode_gcs).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable

from alloy.render.farm_util import existing_frames, frame_index

log = logging.getLogger(__name__)


def gcs_frame_names(prefix: str) -> list[str]:
    """Frame object names (e.g. 00012.png) under a gs:// prefix; [] when it does not exist yet."""
    res = subprocess.run(["gcloud", "storage", "ls", prefix.rstrip("/") + "/"], capture_output=True, text=True)
    if res.returncode != 0:
        return []
    names = (line.rsplit("/", 1)[-1] for line in res.stdout.split())
    return [n for n in names if frame_index(n) is not None]


class FrameStore:
    """Frames already done, and how a VM's finished frames are collected."""

    def __init__(self, out: Path, gcs: str | None = None,
                 lister: Callable[[str], list[str]] = gcs_frame_names) -> None:
        self.out, self.gcs = out, gcs.rstrip("/") if gcs else None
        self._gcs_have: set[int] = set()
        if self.gcs:
            self._gcs_have = {i for i in map(frame_index, lister(self.gcs)) if i is not None}
            log.info("frames go to %s (%d already there)", self.gcs, len(self._gcs_have))

    @property
    def remote_subdir(self) -> str:
        """The VM directory the poll lists: frames (pulled here) or uploaded (marks of frames already in GCS)."""
        return "uploaded" if self.gcs else "frames"

    def have(self) -> set[int]:
        return set(self._gcs_have) if self.gcs else existing_frames(self.out)

    def collect(self, pull: Callable[..., None], vm, remote_frames: str, names: list[str]) -> None:
        """Take finished frames `names` from a VM: pull them here, or (GCS mode) just count them."""
        if not names:
            return
        if self.gcs:
            self._gcs_have |= {i for i in map(frame_index, names) if i is not None}
            return
        pull(vm, remote_frames, self.out / "frames", names)
