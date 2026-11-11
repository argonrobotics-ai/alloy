# SPDX-License-Identifier: Apache-2.0
"""Spot render farm: a frame pool served by whatever VMs exist; VMs join late, preempted work is re-queued."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from alloy.render.farm_frames import FrameStore
from alloy.render.farm_remote import (BLENDER_ERR_CMD, CYCLES_REMOTE, POLL_CMD, REMOTE_ROOT, RemoteLayout, parse_poll,  # noqa: F401
                                      poll_cmd, remote_script)
from alloy.render.farm_types import FarmError, FarmJob, PollResult, Worker  # noqa: F401 - re-exported
from alloy.render.farm_util import (FRAME_EXTS, CostLog, RegionBook, batch_size, existing_frames,  # noqa: F401
                                    parse_timing, render_summary, split_chunks, to_runs)
from alloy.render.staging import StageItem, Stager, stage_prefix
from alloy.render.vm import (Backend, QuotaError, StockoutError, VmError, VmGoneError, VmInfo,
                             region_of)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
BLENDER_FAIL_PENALTY = 2  # a Blender failure counts double: a bad look should not loop for ever
CYCLES_PULL_TRIES = 3
IDLE_GRACE_S = 90.0  # an idle VM waits this long for re-queued frames before it is deleted
MAX_START_FAILS = 3  # an idle VM that refuses this many batches in a row is dropped (and replaced)
STAGE_AUTO_VMS = 8  # stage through GCS when more VMs than this are requested


class Farm:
    """Runs a FarmJob on a Backend."""

    def __init__(self, backend: Backend, job: FarmJob, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, frames: FrameStore | None = None) -> None:
        self.be, self.job, self.clock, self.sleep = backend, job, clock, sleep
        self.frames = frames or FrameStore(job.out, job.frames_gcs)
        self.cost = CostLog()
        self.book = RegionBook(job.zones, job.region_cap, clock=clock)
        self.workers: list[Worker] = []
        self.created: dict[str, VmInfo] = {}
        self.attempted: set[str] = set()
        self.pool: list[int] = []
        self.attempts: dict[int, int] = {}
        self.prov: list[Future] = []
        self.errors: list[str] = []
        self._counter = 0
        self._lock = threading.Lock()
        self._t0: dict[str, float] = {}
        self._step = job.frames[1] - job.frames[0] if len(job.frames) > 1 else 1
        self.lay = RemoteLayout(job.name, int(clock()))
        # Per-run tag in VM names, so two jobs on the same bundle never adopt each other's VMs.
        self._tag = format(int(clock() * 1000) % 46656, "x")[-3:].rjust(3, "0") + "-"
        cyc = job.cycles_dir
        self._cycles_have = bool(cyc and cyc.is_dir() and any(cyc.iterdir()))
        self._cycles_tries = 0
        self._stop = threading.Event()
        self._started_any = False
        n = job.vms + len(job.reuse_vms)
        self._prov_exec = ThreadPoolExecutor(max_workers=max(1, job.vms))
        self._poll_exec = ThreadPoolExecutor(max_workers=max(8, n))
        self.stager = self._make_stager()

    def _make_stager(self) -> Stager | None:
        j = self.job
        if not j.bundle_gcs and (j.stage == "off" or (j.stage is None and j.vms <= STAGE_AUTO_VMS)):
            return None
        base = j.stage if j.stage and j.stage.startswith("gs://") else None
        return Stager(self.be, stage_prefix(base, f"{j.name}-{self.lay.run_dir.rsplit('/', 1)[-1]}"),
                      self._stage_items())

    def _stage_items(self) -> list[StageItem]:
        j, lay = self.job, self.lay
        items = [StageItem("bundle", j.bundle, lay.bundle, source=j.bundle_gcs), StageItem("look.json", j.look, lay.look)]
        if j.blender_dir:
            items.append(StageItem("blender", j.blender_dir, f"{REMOTE_ROOT}/blender"))
        if j.assets_dir and j.assets_dir.is_dir():
            items.append(StageItem("assets", j.assets_dir, f"{REMOTE_ROOT}/assets"))
        if self._cycles_have and j.cycles_dir:
            items.append(StageItem("cycles", j.cycles_dir, CYCLES_REMOTE))
        return items

    # ---- VM creation (runs in threads) ------------------------------------
    def _next_name(self) -> str:
        with self._lock:
            k = self._counter
            self._counter += 1
        return self.job.vm_name if (k == 0 and self.job.vm_name) else f"{self.job.name}-{self._tag}{k}"

    def _create_vm(self, name: str) -> VmInfo:
        existing = self.be.describe(name)
        if existing:
            log.info("reusing VM %s in %s", name, existing.zone)
            self.book.adopt(existing.zone)
            return existing
        self.attempted.add(name)
        tried: set[str] = set()
        while not self._stop.is_set() and (zone := self.book.reserve_next(tried)) is not None:
            tried.add(zone)
            vm = self._try_create(name, zone)
            if vm:
                return vm
        raise VmError(f"no zone with capacity for {name} (tried {len(tried)}; regions at cap or cooling down)")

    def _try_create(self, name: str, zone: str) -> VmInfo | None:
        try:
            vm = self.be.create(name, zone)
        except QuotaError as exc:
            self._failed(zone, f"quota in {region_of(zone)}: {str(exc)[-160:]}")
            self.cost.quota_hits += 1
            self.book.cool_region(region_of(zone))
            return None
        except StockoutError as exc:
            self._failed(zone, f"stockout in {zone}: {str(exc)[-120:]}")
            self.cost.stockouts += 1
            self.book.cool_zone(zone)
            return None
        except VmError as exc:
            self._failed(zone, f"create failed in {zone}: {str(exc)[-200:]}")
            self.book.cool_zone(zone)
            return None
        log.info("created %s in %s", name, zone)
        return vm

    def _failed(self, zone: str, msg: str) -> None:
        self.book.release(zone)
        self.errors.append(msg)
        log.warning("%s", msg)

    def _provision(self, name: str) -> Worker:
        """Create a VM, wait for ssh/Blender/OptiX, load the inputs."""
        t0 = self.clock()
        vm = self._create_vm(name)
        self.created[vm.name] = vm
        self._t0[vm.name] = t0
        try:
            if self._stop.is_set():
                raise VmError("farm finished before this VM was needed")
            self.be.wait_ready(vm, self.job.ready_timeout_s)
            t1 = self.clock()
            self._push_inputs(vm)
            self.cost.push_s.append(self.clock() - t1)
        except VmError as exc:
            self._discard(vm, preempted=isinstance(exc, VmGoneError) and not self._stop.is_set())
            raise
        self.cost.setup_s.append(self.clock() - t0)
        return Worker(vm, idle_since=self.clock())

    def _prepare_reused(self, name: str) -> Worker:
        vm = self.be.describe(name)
        if vm is None:
            raise FarmError(f"--reuse-vm {name}: no running VM with that name")
        self.be.wait_ready(vm, 120)
        self._push_inputs(vm)
        return Worker(vm, reused=True, idle_since=self.clock())

    def _push_inputs(self, vm: VmInfo) -> None:
        for item in self._stage_items():
            if self.stager and self.stager.fetch(vm, item):
                continue
            self.be.push(vm, item.local, item.remote)
        if not self.cost.blender_version:
            self.cost.blender_version = self.be.run(vm, "/opt/blender/blender --version | head -1").out.strip()

    def _discard(self, vm: VmInfo, preempted: bool) -> None:
        self.cost.add_vm(vm.zone, self.clock() - self._t0.get(vm.name, self.clock()))
        self.cost.regions[region_of(vm.zone)] = self.cost.regions.get(region_of(vm.zone), 0) + 1
        if preempted:
            self.cost.preemptions += 1
        if self.job.keep_vms and not preempted:
            return
        self.book.release(vm.zone)
        self.created.pop(vm.name, None)
        try:
            self.be.delete(vm)
        except VmError as exc:
            log.error("could not delete %s: %s (delete it by hand)", vm.name, exc)

    def delete_all(self) -> None:
        """Stop provisioning, delete every VM this farm created (never reused ones), then the stage prefix."""
        self._stop.set()
        if not self.job.keep_vms:
            self._delete_ours()
        self._wait_provisioning()
        if not self.job.keep_vms:
            self._delete_ours()
        if self.stager:
            self.stager.cleanup()

    def _discard_many(self, vms: list[VmInfo]) -> None:
        """Delete VMs in parallel (a sequential `gcloud delete` takes ~20 s each)."""
        if vms:
            with ThreadPoolExecutor(max_workers=min(16, len(vms))) as pool:
                list(pool.map(lambda vm: self._discard(vm, preempted=False), vms))

    def _delete_ours(self) -> None:
        self._discard_many(list(self.created.values()))
        try:
            owned = self.be.list_owned()
        except VmError:
            return
        for vm in owned:
            if vm.name in self.attempted and vm.name not in self.job.reuse_vms:
                log.warning("deleting leftover VM %s", vm.name)
                self._discard_leftover(vm)

    def _discard_leftover(self, vm: VmInfo) -> None:
        try:
            self.be.delete(vm)
        except VmError as exc:
            log.error("could not delete %s: %s", vm.name, exc)

    def _wait_provisioning(self) -> None:
        for fut in self.prov:
            try:
                fut.result()
            except Exception:  # noqa: BLE001 - only waiting for the VM to settle
                pass
        self.prov = []

    # ---- batches --------------------------------------------------------
    def _missing(self, frames: list[int]) -> list[int]:
        have = self.frames.have()
        return [f for f in frames if f not in have]

    def _requeue(self, frames: list[int], why: str, penalty: int = 1) -> None:
        missing = self._missing(frames)
        if not missing:
            return
        for f in missing:
            self.attempts[f] = self.attempts.get(f, 0) + penalty
            if self.attempts[f] >= MAX_ATTEMPTS:
                raise FarmError(f"frame {f} failed {self.attempts[f]} times; last: {why}")
        log.warning("re-queueing %d frames: %s", len(missing), why)
        self.pool = sorted(set(self.pool) | set(missing))

    def _remote_start(self, w: Worker) -> bool:
        try:
            res = self.be.run(w.vm, remote_script(self.job, self.lay, to_runs(w.batch, self._step)))
        except VmError as exc:
            log.warning("start on %s failed: %s", w.vm.name, exc)
            return False
        if "started" not in res.out:
            log.warning("start on %s failed: %s", w.vm.name, res.out.strip()[-300:])
            return False
        self._started_any = True
        return True

    # ---- polling (threads) ---------------------------------------------
    def _poll_worker(self, w: Worker) -> PollResult:
        try:
            if self.be.describe(w.vm.name) is None:
                return PollResult("gone")
            return self._poll_alive(w)
        except VmGoneError:
            return PollResult("gone")
        except VmError as exc:
            log.warning("poll %s failed: %s", w.vm.name, exc)
            return PollResult("running")

    def _poll_alive(self, w: Worker) -> PollResult:
        local, remote = self.job.out / "frames", f"{self.lay.out}/frames"
        names, done = self._collect(w, remote)
        if not done:
            return PollResult("running", first_frames=bool(names))
        if self.frames.gcs:
            names = self._collect(w, remote)[0] or names  # marks written just before `done`
        elif names:  # a batch whose Blender died before its first frame has nothing to pull
            self.be.pull(w.vm, remote, local)
        rows = self.be.run(w.vm, f"cat {self.lay.out}/timing.jsonl 2>/dev/null").out.splitlines()
        return PollResult("done", first_frames=True, timings=parse_timing(rows)[w.timing_seen:], rc=done)

    def _collect(self, w: Worker, remote: str) -> tuple[list[str], str]:
        """Poll a VM once and take its newly finished frames; returns (all listed names, exit code or '')."""
        names, done = parse_poll(self.be.run(w.vm, poll_cmd(self.lay.out, self.frames.remote_subdir)).out)
        names = [n for n in names if n.endswith(FRAME_EXTS)]
        fresh = [n for n in names if n not in w.pulled]
        self.frames.collect(self.be.pull, w.vm, remote, fresh)
        w.pulled.update(fresh)
        return names, done

    def _pull_cycles(self, w: Worker) -> None:
        cyc = self.job.cycles_dir
        if self._cycles_have or cyc is None or self._cycles_tries >= CYCLES_PULL_TRIES:
            return
        self._cycles_tries += 1
        try:
            self.be.pull(w.vm, f"$HOME/{CYCLES_REMOTE}", cyc)
        except VmError as exc:
            log.warning("could not pull the Cycles cache: %s", exc)
            return
        self._cycles_have = any(cyc.iterdir())
        log.info("saved Cycles kernel cache from %s to %s", w.vm.name, cyc)
        if self._cycles_have and self.stager:
            self.stager.upload_one(StageItem("cycles", cyc, CYCLES_REMOTE))

    def _handle(self, w: Worker, res: PollResult) -> bool:
        """Apply a poll result; returns False when the worker must be dropped."""
        if res.first_frames:
            self._pull_cycles(w)
        if res.state == "gone":
            if not w.reused:
                self._discard(w.vm, preempted=True)
            else:
                self.cost.preemptions += 1
            self._requeue(w.batch, "VM preempted")
            return False
        if res.state == "done":
            self.cost.timings += res.timings
            w.timing_seen += len(res.timings)
            failed = res.rc != "0"
            why = f"Blender exit code {res.rc}" + (self._blender_error(w) if failed else "")
            self._requeue(w.batch, why, BLENDER_FAIL_PENALTY if failed else 1)
            w.batch, w.pulled, w.idle_since = [], set(), self.clock()
        return True

    def _blender_error(self, w: Worker) -> str:
        """': <last error lines of blender.log>' for a failed run, or '' when they cannot be read."""
        try:
            text = self.be.run(w.vm, BLENDER_ERR_CMD.format(out=self.lay.out)).out
        except VmError:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return f" ({w.vm.name}: {' | '.join(lines)})" if lines else ""

    # ---- main loop -------------------------------------------------------
    def _created_workers(self) -> int:
        return sum(1 for w in self.workers if not w.reused)

    def _spawn_missing(self, next_try: float) -> float:
        """Submit VM creations up to --vms while frames are waiting; returns the next allowed attempt time."""
        wanted = self.job.vms - self._created_workers() - len(self.prov)
        if wanted <= 0 or not self.pool or self.clock() < next_try or not self._worth_spawning():
            return next_try
        for _ in range(wanted):
            self.prov.append(self._prov_exec.submit(self._provision, self._next_name()))
        return self.clock() + self.job.retry_s

    def _worth_spawning(self) -> bool:
        """A new VM needs minutes to boot; skip it when the running crew will finish before it is ready."""
        if not self.workers:
            return True
        frames_left = len(self.pool) + sum(len(w.batch) for w in self.workers)
        t = self.cost.timings
        per_frame = sum(t) / len(t) if t else 20.0
        setup = sum(self.cost.setup_s) / len(self.cost.setup_s) if self.cost.setup_s else 180.0
        return frames_left * per_frame / len(self.workers) > setup

    def _collect_provisioned(self) -> None:
        still = []
        for fut in self.prov:
            if not fut.done():
                still.append(fut)
                continue
            try:
                self.workers.append(fut.result())
                self.cost.late_joiners += 1 if self._started_any else 0
            except FarmError:
                raise
            except VmError as exc:
                self.errors.append(str(exc)[-200:])
                log.warning("VM did not come up: %s", exc)
        self.prov = still

    def _step_workers(self) -> None:
        busy = [w for w in self.workers if w.batch]
        results = list(self._poll_exec.map(self._poll_worker, busy)) if busy else []
        for w, res in zip(busy, results):
            if not self._handle(w, res):
                self.workers.remove(w)
        self._start_idle()

    def _start_idle(self) -> None:
        crew = max(1, len(self.workers) + len(self.prov))
        starting, retiring = [], []
        for w in [w for w in self.workers if not w.batch]:
            if not self.pool:
                if self._should_retire(w):
                    retiring.append(w)
                continue
            size = batch_size(len(self.pool), crew)
            w.batch, self.pool = self.pool[:size], self.pool[size:]
            starting.append(w)
        for w, ok in zip(starting, self._poll_exec.map(self._remote_start, starting)):
            w.start_fails = 0 if ok else w.start_fails + 1
            if not ok:
                self.pool = sorted(set(self.pool) | set(w.batch))
                w.batch = []
                self._drop_if_unstartable(w)
        for w in retiring:
            self.workers.remove(w)
        self._discard_many([w.vm for w in retiring if not w.reused])

    def _drop_if_unstartable(self, w: Worker) -> None:
        """Drop an idle VM whose start failed when it is gone (reclaimed between batches) or failed
        MAX_START_FAILS times; idle workers are never polled, so without this the farm retries it for ever."""
        gone = self.be.describe(w.vm.name) is None
        if not gone and w.start_fails < MAX_START_FAILS:
            return
        log.warning("dropping %s: %s", w.vm.name, "gone" if gone else f"{w.start_fails} failed starts")
        self.workers.remove(w)
        if not w.reused:
            self._discard(w.vm, preempted=gone)

    def _should_retire(self, w: Worker) -> bool:
        busy_elsewhere = any(x.batch for x in self.workers)
        return not busy_elsewhere or self.clock() - w.idle_since >= IDLE_GRACE_S

    def run(self) -> dict:
        """Render every frame in the job; returns the render_log dict."""
        j = self.job
        have = self.frames.have()
        self.pool = [f for f in j.frames if f not in have]
        rendered, t_start = len(self.pool), self.clock()
        if self.pool:
            try:
                self._serve(t_start)
            finally:
                self._stop.set()
                self._prov_exec.shutdown(wait=False)
                self._poll_exec.shutdown(wait=False)
        return self.summary(len(j.frames), rendered, self.clock() - t_start)

    def _serve(self, t_start: float) -> None:
        j = self.job
        if self.stager:
            self.stager.start()
        for name in j.reuse_vms:
            self.workers.append(self._prepare_reused(name))
        next_try, last_vm = t_start, t_start
        while self.pool or any(w.batch for w in self.workers):
            self._collect_provisioned()
            next_try = self._spawn_missing(next_try)
            self._step_workers()
            if self.workers:
                last_vm = self.clock()
            elif self.clock() - last_vm > j.no_vm_timeout_s:
                raise FarmError(f"no VM could be started in {j.no_vm_timeout_s / 60:.0f} min; "
                                f"last errors: {'; '.join(self.errors[-3:]) or 'no capacity in any region'}")
            if self.pool or any(w.batch for w in self.workers):
                self.sleep(j.poll_s)
        leaving = list(self.workers)
        self.workers.clear()
        self._discard_many([w.vm for w in leaving if not w.reused])

    def summary(self, total: int, rendered: int, wall_s: float) -> dict:
        """The render_log.json content."""
        return render_summary(self.cost, self.stager, total, rendered, wall_s)
