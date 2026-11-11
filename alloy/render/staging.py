# SPDX-License-Identifier: Apache-2.0
"""GCS staging: upload job inputs once, let every VM pull them from the bucket (rsync from the controller is the
fallback)."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from alloy.render.vm import Backend, RenderConfig, VmError, VmInfo

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageItem:
    """One input: label under the stage prefix, local source, home-relative remote destination."""

    label: str
    local: Path
    remote: str
    source: str | None = None  # already in GCS (a gs:// bundle): never uploaded, never rsynced from here

    @property
    def is_dir(self) -> bool:
        return self.source is not None or self.local.is_dir()


def stage_prefix(arg: str | None, job: str, config: RenderConfig | None = None) -> str:
    """``<arg>/<job>`` for a gs:// base, else ``gs://$ALLOY_RENDER_BUCKET/stage/<job>``.

    With no base and no bucket configured it returns "" (staging off: inputs go by rsync) and logs which variable
    to set.

    >>> stage_prefix("gs://b/stage/", "j"), stage_prefix(None, "j", RenderConfig())
    ('gs://b/stage/j', '')
    """
    if arg:
        return f"{arg.rstrip('/')}/{job}"
    try:
        return f"{(config or RenderConfig.load()).bucket_uri('stage')}/{job}"
    except VmError as exc:
        log.warning("not staging inputs through GCS (%s); VMs get them by rsync from this machine", exc)
        return ""


def bucket_of(uri: str) -> str:
    return uri.removeprefix("gs://").split("/", 1)[0]


class Stager:
    """Uploads inputs in the background; VMs wait for them (`wait`) then pull what was staged."""

    def __init__(self, backend: Backend, prefix: str, items: list[StageItem]) -> None:
        self.be, self.prefix, self.items = backend, prefix, items
        self.staged: set[str] = set()
        self.ready = threading.Event()
        self.upload_s = 0.0
        self.started = False
        self._thread = threading.Thread(target=self._upload_all, name="stage-upload", daemon=True)

    def start(self) -> None:
        self.started = True
        self._thread.start()

    def _uri(self, label: str) -> str:
        return f"{self.prefix}/{label}"

    def _upload_all(self) -> None:
        t0 = time.time()
        try:
            if self.prefix:
                self.be.ensure_bucket(bucket_of(self.prefix))
            for item in self.items:
                self.upload_one(item)
        except VmError as exc:
            log.warning("staging unavailable (%s); VMs will use rsync from the controller", exc)
        finally:
            self.upload_s = time.time() - t0
            self.ready.set()

    def upload_one(self, item: StageItem) -> None:
        """Upload one item; on failure it simply stays un-staged so VMs fall back to rsync."""
        if item.source:
            self.staged.add(item.label)
            return
        if not self.prefix:  # no bucket configured: this item goes by rsync
            return
        try:
            self.be.stage_up(item.local, self._uri(item.label))
            self.staged.add(item.label)
        except VmError as exc:
            log.warning("could not stage %s: %s", item.label, exc)

    def fetch(self, vm: VmInfo, item: StageItem, timeout_s: float = 3 * 3600) -> bool:
        """Download one staged item to the VM; False means the caller should rsync it instead."""
        self.ready.wait(timeout_s)
        if item.label not in self.staged:
            return False
        try:
            self.be.stage_down(vm, item.source or self._uri(item.label), item.remote, item.is_dir)
            return True
        except VmError as exc:
            if item.source:  # the local copy is a sparse mirror: rsyncing it would ship zeros
                raise VmError(f"{vm.name}: download of {item.source} failed: {exc}") from exc
            log.warning("%s: staged download of %s failed (%s); falling back to rsync", vm.name, item.label, exc)
            return False

    def cleanup(self) -> None:
        if not self.started or not self.prefix:
            return
        self.ready.wait(3600)
        try:
            self.be.stage_delete(self.prefix)
            log.info("deleted stage %s", self.prefix)
        except VmError as exc:
            log.error("could not delete stage %s: %s (delete it by hand)", self.prefix, exc)
