# SPDX-License-Identifier: Apache-2.0
"""gcloud + ssh layer for spot L4 render VMs, behind a small Backend interface (fakeable in tests)."""
from __future__ import annotations

import contextlib
import getpass
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

log = logging.getLogger(__name__)

ZONES = (
    "us-central1-a", "us-central1-b", "us-central1-c", "us-east1-b", "us-east1-c", "us-east1-d", "us-east4-a",
    "us-east4-c", "us-west1-a", "us-west1-b", "us-west1-c", "us-west4-a", "us-west4-c",
    "europe-west1-b", "europe-west1-c", "europe-west2-a", "europe-west2-b", "europe-west3-a", "europe-west3-b",
    "europe-west4-a", "europe-west4-b", "europe-west4-c",
    "asia-east1-a", "asia-east1-c", "asia-northeast1-a", "asia-northeast1-c", "asia-southeast1-a",
    "asia-southeast1-b", "asia-southeast1-c", "asia-south1-a", "asia-south1-b",
)
IMAGE_FAMILY = "common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT = "deeplearning-platform-release"
# Prebuilt by scripts/render_image.sh: DLVM + OptiX libs + Blender + a warm kernel cache (~2.5 min faster boot).
RENDER_IMAGE_FAMILY = "alloy-render-blender45"
MACHINE = "g2-standard-8"
STARTUP_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_vm_startup.sh"
SSH_KEY = Path.home() / ".ssh" / "google_compute_engine"


class VmError(RuntimeError):
    """A gcloud or ssh operation failed."""


ENV_PROJECT = "ALLOY_RENDER_PROJECT"  # cloud project (else the config file, else `gcloud config` default)
ENV_BUCKET = "ALLOY_RENDER_BUCKET"    # GCS bucket for staging and frames; the VMs' service account must write it
ENV_LABELS = "ALLOY_RENDER_LABELS"    # extra VM labels, "key=value,key=value"
ENV_ROLE = "ALLOY_RENDER_ROLE"        # the role label the farm finds its own VMs by
ENV_CONFIG = "ALLOY_RENDER_CONFIG"    # JSON file with any of: project, bucket, labels, role
CONFIG_PATH = Path("~/.config/alloy/render.json")
DEFAULT_ROLE = "alloy-render"


@dataclass(frozen=True)
class RenderConfig:
    """Cloud settings for render VMs: environment variables first, then a JSON config file; nothing built in.

    Example::

        >>> cfg = RenderConfig.load(env={"ALLOY_RENDER_BUCKET": "gs://my-renders"}, path=Path("/nonexistent"))
        >>> cfg.bucket_uri("stage"), cfg.vm_labels()
        ('gs://my-renders/stage', 'role=alloy-render')
    """

    project: str | None = None
    bucket: str | None = None
    labels: str = ""
    role: str = DEFAULT_ROLE

    @classmethod
    def load(cls, env: dict[str, str] | None = None, path: Path | None = None) -> RenderConfig:
        """Read the settings; ``env`` defaults to ``os.environ``, ``path`` to $ALLOY_RENDER_CONFIG or CONFIG_PATH."""
        env = dict(os.environ) if env is None else env
        file = _read_config(Path(env.get(ENV_CONFIG) or path or CONFIG_PATH).expanduser())

        def pick(var: str, key: str, default: str | None = None) -> str | None:
            return env.get(var) or file.get(key) or default

        bucket = (pick(ENV_BUCKET, "bucket") or "").removeprefix("gs://").strip("/") or None
        return cls(pick(ENV_PROJECT, "project"), bucket, pick(ENV_LABELS, "labels", "") or "",
                   pick(ENV_ROLE, "role", DEFAULT_ROLE) or DEFAULT_ROLE)

    def vm_labels(self) -> str:
        """The ``--labels`` value for a new VM: the configured labels plus ``role=<role>``."""
        extra = [x.strip() for x in self.labels.split(",") if x.strip() and not x.strip().startswith("role=")]
        return ",".join([*extra, f"role={self.role}"])

    def bucket_uri(self, suffix: str) -> str:
        """``gs://<bucket>/<suffix>``; raises VmError naming the variable to set when no bucket is configured."""
        if not self.bucket:
            raise VmError(f"no render bucket configured: set {ENV_BUCKET} to a GCS bucket the render VMs can write "
                          f"(or \"bucket\" in {CONFIG_PATH}), or pass an explicit gs:// prefix")
        return f"gs://{self.bucket}/{suffix}"


def _read_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VmError(f"render config {path} is not valid JSON ({exc}); fix it or unset {ENV_CONFIG}") from exc
    if not isinstance(data, dict):
        raise VmError(f"render config {path} must be a JSON object with project/bucket/labels/role")
    return {k: str(v) for k, v in data.items() if v is not None}


class StockoutError(VmError):
    """A zone has no spot capacity right now."""


class QuotaError(VmError):
    """The project's GPU quota is exhausted in the whole region of the zone."""


class VmGoneError(VmError):
    """The VM disappeared (preempted or deleted) while we were using it."""


@dataclass(frozen=True)
class VmInfo:
    """A render VM."""

    name: str
    zone: str
    ip: str
    status: str = "RUNNING"


@dataclass(frozen=True)
class CmdResult:
    """Result of a remote command."""

    rc: int
    out: str


class Backend(Protocol):
    """Everything the farm needs from the cloud; implemented by GcloudBackend and by test fakes."""

    def create(self, name: str, zone: str) -> VmInfo: ...
    def describe(self, name: str) -> VmInfo | None: ...
    def delete(self, vm: VmInfo) -> None: ...
    def list_owned(self) -> list[VmInfo]: ...
    def wait_ready(self, vm: VmInfo, timeout_s: float) -> None: ...
    def run(self, vm: VmInfo, cmd: str, timeout_s: float = 120) -> CmdResult: ...
    def push(self, vm: VmInfo, local: Path, remote: str) -> None: ...
    def pull(self, vm: VmInfo, remote: str, local: Path, names: Sequence[str] | None = None) -> None: ...
    def ensure_bucket(self, bucket: str) -> None: ...
    def stage_up(self, local: Path, uri: str) -> None: ...
    def stage_down(self, vm: VmInfo, uri: str, remote: str, is_dir: bool) -> None: ...
    def stage_delete(self, uri: str) -> None: ...


def region_of(zone: str) -> str:
    """us-central1-a -> us-central1."""
    return zone.rsplit("-", 1)[0]


STOCKOUT_RE = re.compile(r"STOCKOUT|RESOURCE_POOL_EXHAUSTED|does not have enough resources", re.I)
QUOTA_RE = re.compile(r"QUOTA_EXCEEDED|Quota '[A-Z0-9_]+' exceeded|quota .*exceeded", re.I)
CREATE_SLOTS = 8  # concurrent `gcloud compute instances create` calls (API rate limits)
SNAPSHOT_TTL_S = 8.0


def classify_create_error(text: str) -> type[VmError]:
    """Quota is checked first: a quota message can also mention resources."""
    if QUOTA_RE.search(text):
        return QuotaError
    return StockoutError if STOCKOUT_RE.search(text) else VmError


def rsync_remote_path(remote: str) -> str:
    """Home-relative form of a remote path for rsync.

    rsync >= 3.2.4 protects remote args, so ``$HOME`` / ``~`` are no longer expanded by a shell; a relative path
    resolves against the login home on every rsync version.
    """
    for prefix in ("$HOME/", "${HOME}/", "~/"):
        if remote.startswith(prefix):
            return remote[len(prefix):]
    return remote


@contextlib.contextmanager
def dereferenced(path: Path):
    """``path`` itself, or a temporary copy with every symlink replaced by its target when it contains links.

    gcloud storage rsync does not follow symlinked directories, so a bundle that links meshes/ from another bundle
    would reach the VMs without its meshes (every Blender run then fails).
    """
    path = Path(path)
    if not path.is_dir() or not any(p.is_symlink() for p in path.rglob("*")):
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="alloy_stage_") as tmp:
        copy = Path(tmp) / path.name
        shutil.copytree(path, copy, symlinks=False)
        log.info("staging a dereferenced copy of %s (it contains symlinks)", path)
        yield copy


def sanitise_name(text: str) -> str:
    """GCE-safe lowercase fragment for VM names (letters, digits, dashes)."""
    s = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return s[:36].strip("-") or "bundle"


def vm_name(bundle: Path, k: int | str) -> str:
    """alloy-render-<bundle dirname>-<k>."""
    return f"alloy-render-{sanitise_name(Path(bundle).resolve().name)}-{k}"


def _exec(argv: Sequence[str], timeout_s: float, input_text: str | None = None) -> CmdResult:
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s, input=input_text)
    except subprocess.TimeoutExpired as exc:
        raise VmError(f"timed out after {timeout_s:.0f}s: {' '.join(argv[:6])}") from exc
    except FileNotFoundError as exc:
        raise VmError(f"{argv[0]} not found on PATH") from exc
    return CmdResult(proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


class GcloudBackend:
    """Backend over `gcloud compute` and ssh/rsync with ~/.ssh/google_compute_engine."""

    def __init__(self, project: str | None = None, user: str | None = None, key: Path = SSH_KEY,
                 exec_fn: Callable[..., CmdResult] = _exec, config: RenderConfig | None = None) -> None:
        self._exec = exec_fn
        self.user = user or getpass.getuser()
        self.key = key
        self.config = config or RenderConfig.load()
        self.project = project or self.config.project or self._default_project()
        self._slots = threading.BoundedSemaphore(CREATE_SLOTS)
        self._snap_lock = threading.Lock()
        self._snap: dict[str, VmInfo] = {}
        self._snap_at = 0.0
        self._image: list[str] | None = None
        self._fresh: dict[str, tuple[VmInfo, float]] = {}

    def _default_project(self) -> str:
        res = self._exec(["gcloud", "config", "get-value", "project"], 30)
        project = res.out.strip().splitlines()[-1] if res.out.strip() else ""
        if res.rc != 0 or not project or project == "(unset)":
            raise VmError(f"no cloud project: set {ENV_PROJECT} or `gcloud config set project ...` "
                          f"({res.out.strip()})")
        return project

    def _gcloud(self, args: Sequence[str], timeout_s: float = 300) -> CmdResult:
        return self._exec(["gcloud", "compute", *args, "--project", self.project], timeout_s)

    def image_args(self) -> list[str]:
        """The prebuilt render image family when it exists in the project, else the stock DLVM image."""
        if self._image is None:
            res = self._gcloud(["images", "describe-from-family", RENDER_IMAGE_FAMILY, "--format", "value(name)"], 60)
            if res.rc == 0 and res.out.strip():
                self._image = ["--image-family", RENDER_IMAGE_FAMILY, "--image-project", self.project]
                log.info("render image: %s (prebuilt)", res.out.strip())
            else:
                self._image = ["--image-family", IMAGE_FAMILY, "--image-project", IMAGE_PROJECT]
                log.info("render image: stock DLVM %s (no %s yet)", IMAGE_FAMILY, RENDER_IMAGE_FAMILY)
        return self._image

    def create(self, name: str, zone: str) -> VmInfo:
        args = ["instances", "create", name, "--zone", zone, "--machine-type", MACHINE,
                "--provisioning-model", "SPOT", "--instance-termination-action", "DELETE",
                "--max-run-duration", "8h", "--maintenance-policy", "TERMINATE", "--labels", self.config.vm_labels(),
                *self.image_args(), "--boot-disk-size", "50GB",
                "--scopes", "storage-rw",  # VMs push frames to GCS themselves (FarmJob.frames_gcs)
                "--metadata", "install-nvidia-driver=True",
                "--metadata-from-file", f"startup-script={STARTUP_SCRIPT}", "--format", "json"]
        with self._slots:
            res = self._gcloud(args)
        if res.rc != 0:
            text = res.out.strip()
            raise classify_create_error(text)(f"gcloud create {name} in {zone} failed: {text[-600:]}")
        vm = _created_vm(res.out)
        if vm is None or not vm.ip:
            vm = self._describe_now(name)
        if vm is None:
            raise VmError(f"{name} was created in {zone} but is not visible")
        self._fresh[name] = (vm, time.time())
        return vm

    def _describe_now(self, name: str) -> VmInfo | None:
        res = self._gcloud(["instances", "list", "--filter", f"name={name} AND status=RUNNING", "--format", "json"], 60)
        if res.rc != 0:
            raise VmError(f"gcloud instances list failed: {res.out.strip()[-400:]}")
        return _first_vm(res.out)

    def _snapshot(self, max_age: float) -> dict[str, VmInfo]:
        """All wp20-render VMs, shared by every thread and refreshed at most every max_age seconds."""
        with self._snap_lock:
            if time.time() - self._snap_at > max_age:
                self._snap = {v.name: v for v in self.list_owned() if v.status == "RUNNING"}
                self._snap_at = time.time()
            return self._snap

    def describe(self, name: str) -> VmInfo | None:
        vm = self._snapshot(SNAPSHOT_TTL_S).get(name)
        if vm is None:
            vm = self._snapshot(2.0).get(name)
        if vm is None:
            known = self._fresh.get(name)
            if known and time.time() - known[1] < 90:
                return known[0]
        return vm

    def list_owned(self) -> list[VmInfo]:
        res = self._gcloud(["instances", "list", "--filter", f"labels.role={self.config.role}", "--format", "json"],
                           120)
        if res.rc != 0:
            raise VmError(f"gcloud instances list failed: {res.out.strip()[-400:]}")
        return _all_vms(res.out)

    def delete(self, vm: VmInfo) -> None:
        res = self._gcloud(["instances", "delete", vm.name, "--zone", vm.zone, "--quiet"])
        self._fresh.pop(vm.name, None)
        if res.rc != 0 and "was not found" not in res.out:
            raise VmError(f"gcloud delete {vm.name} failed: {res.out.strip()[-400:]}")

    def _ssh_opts(self) -> list[str]:
        return ["-i", str(self.key), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]

    def run(self, vm: VmInfo, cmd: str, timeout_s: float = 120) -> CmdResult:
        return self._exec(["ssh", *self._ssh_opts(), f"{self.user}@{vm.ip}", cmd], timeout_s)

    def wait_ready(self, vm: VmInfo, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        probe = ("test -x /opt/blender/blender && ls /usr/lib/x86_64-linux-gnu/libnvoptix.so.1 >/dev/null 2>&1 "
                 "&& nvidia-smi -L >/dev/null 2>&1 && echo ready")
        while time.time() < deadline:
            if self.describe(vm.name) is None:
                raise VmGoneError(f"{vm.name} disappeared while waiting for ssh/Blender")
            try:
                if "ready" in self.run(vm, probe, 30).out:
                    return
            except VmError:
                pass
            time.sleep(10)
        raise VmError(f"{vm.name}: ssh/Blender/driver not ready after {timeout_s:.0f}s")

    def _rsync(self, args: Sequence[str], timeout_s: float, stdin: str | None = None) -> None:
        ssh = "ssh " + " ".join(self._ssh_opts())
        flags = "-azL"  # follow symlinks: bundles made with links (e.g. shared meshes/) must arrive as files
        res = self._exec(["rsync", flags, "-e", ssh, *args], timeout_s) if stdin is None else \
            _exec(["rsync", flags, "-e", ssh, *args], timeout_s, stdin)
        if res.rc != 0:
            raise VmError(f"rsync {' '.join(args[-2:])} failed (rc {res.rc}): {res.out.strip()[-400:]}")

    def push(self, vm: VmInfo, local: Path, remote: str) -> None:
        remote = rsync_remote_path(remote)
        src = str(local).rstrip("/") + ("/" if Path(local).is_dir() else "")
        self.run(vm, f"mkdir -p {_remote_parent(remote, Path(local).is_dir())}", 60)
        self._rsync([src, f"{self.user}@{vm.ip}:{remote}"], 6 * 3600)

    def pull(self, vm: VmInfo, remote: str, local: Path, names: Sequence[str] | None = None) -> None:
        Path(local).mkdir(parents=True, exist_ok=True)
        base = [f"{self.user}@{vm.ip}:{rsync_remote_path(remote).rstrip('/')}/", str(local) + "/"]
        if names is None:
            self._rsync(base, 1800)
            return
        if names:
            self._rsync(["--files-from=-", *base], 1800, "\n".join(names) + "\n")


    # ---- GCS staging ---------------------------------------------------------
    def _storage(self, args: Sequence[str], timeout_s: float) -> CmdResult:
        return self._exec(["gcloud", "storage", *args, "--project", self.project], timeout_s)

    def ensure_bucket(self, bucket: str) -> None:
        if self._storage(["buckets", "describe", f"gs://{bucket}"], 60).rc == 0:
            return
        res = self._storage(["buckets", "create", f"gs://{bucket}", "--location=US",
                             "--uniform-bucket-level-access"], 120)
        if res.rc != 0 and "already" not in res.out.lower():
            raise VmError(f"cannot create bucket {bucket}: {res.out.strip()[-300:]}")

    def stage_up(self, local: Path, uri: str) -> None:
        local = Path(local)
        with dereferenced(local) as src:
            args = ["rsync", "--recursive", str(src), uri] if src.is_dir() else ["cp", str(src), uri]
            res = self._storage(args, 3 * 3600)
        if res.rc != 0:
            raise VmError(f"stage upload of {local} to {uri} failed: {res.out.strip()[-300:]}")

    def stage_down(self, vm: VmInfo, uri: str, remote: str, is_dir: bool) -> None:
        dest = f"$HOME/{rsync_remote_path(remote)}"
        if is_dir:
            cmd = f"mkdir -p {dest} && gcloud storage rsync --recursive {uri} {dest}"
        else:
            cmd = f"mkdir -p $(dirname {dest}) && gcloud storage cp {uri} {dest}"
        res = self.run(vm, cmd, 3 * 3600)
        if res.rc != 0:
            raise VmError(f"{vm.name}: staged download of {uri} failed: {res.out.strip()[-300:]}")

    def stage_delete(self, uri: str) -> None:
        res = self._storage(["rm", "--recursive", uri], 600)
        if res.rc != 0 and "matched no objects" not in res.out and "No URLs matched" not in res.out:
            raise VmError(f"stage delete {uri} failed: {res.out.strip()[-300:]}")


def _remote_parent(remote: str, is_dir: bool) -> str:
    return remote if is_dir else os.path.dirname(remote) or "."


def _vm_from(item: dict) -> VmInfo:
    nic = (item.get("networkInterfaces") or [{}])[0]
    ip = ((nic.get("accessConfigs") or [{}])[0]).get("natIP", "")
    return VmInfo(item["name"], str(item.get("zone", "")).rsplit("/", 1)[-1], ip, item.get("status", "RUNNING"))


def _all_vms(text: str) -> list[VmInfo]:
    try:
        return [_vm_from(i) for i in json.loads(text or "[]")]
    except (json.JSONDecodeError, KeyError) as exc:
        raise VmError(f"unparseable gcloud output: {exc}: {text[:200]}") from exc


def _created_vm(text: str) -> VmInfo | None:
    """The VM in `instances create --format json` output (warnings may precede the JSON list)."""
    m = re.search(r"^\[\s*$|^\[\{", text, re.M)
    try:
        return _first_vm(text[m.start():]) if m else None
    except VmError:
        return None


def _first_vm(text: str) -> VmInfo | None:
    vms = _all_vms(text)
    return vms[0] if vms else None
