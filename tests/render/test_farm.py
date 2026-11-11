# SPDX-License-Identifier: Apache-2.0
"""Farm logic against a fake backend: pool, stockout of k of n, late joiners, reuse, preemption, resume, costs."""
from __future__ import annotations

import re
import shlex
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from alloy.render.farm import Farm, FarmError, FarmJob
from alloy.render.farm_util import RegionBook, batch_size, existing_frames, parse_timing, split_chunks, to_runs
from alloy.render.vm import (ZONES, CmdResult, QuotaError, StockoutError, VmError, VmInfo, classify_create_error,
                             region_of)


def bare(name: str) -> str:
    """A farm VM name without its per-run tag: clip-f00-3 -> clip-3."""
    return re.sub(r"-[0-9a-f]{3}-(\d+)$", r"-\1", name)


class FakeBackend:
    """In-memory cloud. A batch started on a VM shows half its frames on the first poll and all on the second."""

    def __init__(self, create_ok=lambda be, name, zone: True, preempt_at=None, existing=(), quota_regions=(),
                 stage_fail=False):
        self.vms = {n: VmInfo(n, "us-west1-b", "9.9.9.9") for n in existing}
        self.create_ok, self.preempt_at = create_ok, preempt_at or {}
        self.create_calls: list[tuple[str, str]] = []
        self.created, self.deleted, self.pushes, self.pulls = [], [], [], []
        self.scripts: list[tuple[str, str]] = []
        self.batch: dict[str, list[int]] = {}
        self.polls: dict[str, int] = {}
        self.files: dict[str, dict[str, bytes]] = {}
        self.done: set[str] = set()
        self.total_polls = 0
        self.quota_regions = set(quota_regions)
        self.stage_fail = stage_fail
        self.staged_up: list[str] = []
        self.staged_down: list[tuple[str, str]] = []
        self.stage_deleted: list[str] = []
        self.buckets: list[str] = []

    def create(self, name, zone):
        self.create_calls.append((name, zone))
        if region_of(zone) in self.quota_regions:
            raise QuotaError(f"Quota 'PREEMPTIBLE_NVIDIA_L4_GPUS' exceeded. Limit: 8.0 in region {region_of(zone)}.")
        if not self.create_ok(self, name, zone):
            raise StockoutError(f"stockout {zone}")
        self.vms[name] = VmInfo(name, zone, "1.2.3.4")
        self.created.append((name, zone))
        return self.vms[name]

    def describe(self, name):
        return self.vms.get(name)

    def delete(self, vm):
        self.vms.pop(vm.name, None)
        self.deleted.append(vm.name)

    def list_owned(self):
        return [v for v in self.vms.values() if v.name != "alloy-render-1"]

    def wait_ready(self, vm, timeout_s):
        pass

    def push(self, vm, local, remote):
        self.pushes.append((vm.name, remote))

    def ensure_bucket(self, bucket):
        self.buckets.append(bucket)

    def stage_up(self, local, uri):
        if self.stage_fail:
            raise VmError("no network")
        self.staged_up.append(uri)

    def stage_down(self, vm, uri, remote, is_dir):
        self.staged_down.append((vm.name, uri))

    def stage_delete(self, uri):
        self.stage_deleted.append(uri)

    def run(self, vm, cmd, timeout_s=120):
        if "nohup" in cmd:
            self._start(vm.name, cmd)
            return CmdResult(0, "started")
        if "::DONE" in cmd:
            rc = self._poll(vm.name)
            return CmdResult(0, " ".join(self.files.get(vm.name, {})) + "\n::DONE\n" + rc)
        if "blender.log" in cmd:
            return CmdResult(0, getattr(self, "blender_log", ""))
        if "timing.jsonl" in cmd:
            rows = [f'{{"frame": {i}, "seconds": 10}}' for i in range(len(self.files.get(vm.name, {})))]
            return CmdResult(0, "\n".join(rows + ['{"summary": true, "seconds": 99}']))
        return CmdResult(0, "Blender 4.5.9")

    def _start(self, name, cmd):
        frames = []
        for a, b in re.findall(r"--frames (\d+):(\d+)", cmd):
            frames += list(range(int(a), int(b)))
        self.scripts.append((name, cmd))
        self.batch[name], self.polls[name] = frames, 0
        self.done.discard(name)

    def _poll(self, name):
        self.total_polls += 1
        self.polls[name] += 1
        frames = self.batch.get(name, [])
        shown = frames[: (len(frames) + 1) // 2] if self.polls[name] == 1 else frames
        self.files.setdefault(name, {}).update({f"{f:05d}.png": b"png" for f in shown})
        if self.polls[name] >= 2:
            self.done.add(name)
        if self.preempt_at.get(bare(name)) == self.polls[name]:
            self.vms.pop(name, None)
            return ""
        return "0" if name in self.done else ""

    def pull(self, vm, remote, local, names=None):
        self.pulls.append((vm.name, remote))
        Path(local).mkdir(parents=True, exist_ok=True)
        if "cycles" in remote:
            (Path(local) / "kernel.bin").write_bytes(b"k")
            return
        for n in names if names is not None else list(self.files.get(vm.name, {})):
            (Path(local) / n).write_bytes(self.files[vm.name][n])


class Clock:
    """Fake clock: sleep advances it by the requested amount (plus a tiny real pause for threads)."""

    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        time.sleep(0.02)


def make_job(tmp_path, frames, **kw):
    (tmp_path / "frames").mkdir(exist_ok=True)
    kw.setdefault("poll_s", 30)
    return FarmJob(bundle=tmp_path, look=tmp_path / "look.json", camera="cam", out=tmp_path, name="clip",
                   frames=frames, **kw)


def run_farm(be, job, clock=None):
    c = clock or Clock()
    return Farm(be, job, clock=c.now, sleep=c.sleep).run()


def test_helpers():
    assert split_chunks(list(range(10)), 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert to_runs([0, 4, 8, 20, 24], 4) == [(0, 8), (20, 24)]
    assert batch_size(450, 48) == 5 and batch_size(450, 4) == 32 and batch_size(30, 48) == 2
    assert batch_size(1, 48) == 1 and batch_size(3, 48) == 2
    rows = ['{"frame": 1, "seconds": 3.5}', "junk", '{"x": 1}', '{"summary": true, "seconds": 99}']
    assert parse_timing(rows) == [3.5]


def test_renders_all_frames_and_logs_cost(tmp_path):
    be = FakeBackend()
    log = run_farm(be, make_job(tmp_path, list(range(12)), vms=2))
    assert existing_frames(tmp_path) == set(range(12))
    assert len(be.created) == 2 and sorted(be.deleted) == sorted(n for n, _ in be.created)
    assert log["frames"] == 12 and log["seconds_per_frame"] == 10.0 and log["preemptions"] == 0
    assert log["zones"] and log["blender"] == "Blender 4.5.9"


def test_zone_stockout_falls_back_and_cools_down(tmp_path):
    bad = ("us-central1-a", "us-central1-b")
    be = FakeBackend(create_ok=lambda b, n, z: z not in bad)
    run_farm(be, make_job(tmp_path, list(range(8)), vms=1))
    assert be.created[0][1] == "us-central1-c"
    assert [z for _, z in be.create_calls].count("us-central1-a") == 1


def test_k_of_n_vms_stocked_out_still_renders_then_late_joiner(tmp_path):
    be = FakeBackend(create_ok=lambda b, n, z: n.endswith("-0") or b.total_polls >= 6)
    job = make_job(tmp_path, list(range(60)), vms=3, retry_s=60)
    log = run_farm(be, job)
    assert existing_frames(tmp_path) == set(range(60))
    names = {bare(n) for n, _ in be.created}
    assert "clip-0" in names and len(names) >= 2
    assert log["stockouts"] > 0 and log["late_joiners"] >= 1


def test_all_vms_down_for_20_minutes_aborts(tmp_path):
    be = FakeBackend(create_ok=lambda b, n, z: False)
    job = make_job(tmp_path, [0, 1, 2], vms=2, retry_s=120, no_vm_timeout_s=1200)
    with pytest.raises(FarmError, match="no VM could be started"):
        run_farm(be, job)
    assert be.created == []


def test_reuse_vm_is_used_and_never_deleted(tmp_path):
    be = FakeBackend(existing=("alloy-render-1",))
    job = make_job(tmp_path, list(range(8)), vms=0, reuse_vms=("alloy-render-1",))
    run_farm(be, job)
    assert existing_frames(tmp_path) == set(range(8))
    assert be.created == [] and be.deleted == [] and "alloy-render-1" in be.vms


def test_reuse_vm_missing_is_an_error(tmp_path):
    with pytest.raises(FarmError, match="no running VM"):
        run_farm(FakeBackend(), make_job(tmp_path, [0, 1], vms=0, reuse_vms=("alloy-render-1",)))


def test_reuse_plus_new_vms_share_the_pool(tmp_path):
    be = FakeBackend(existing=("alloy-render-1",))
    run_farm(be, make_job(tmp_path, list(range(24)), vms=1, reuse_vms=("alloy-render-1",)))
    assert {bare(n) for n, _ in be.scripts} == {"alloy-render-1", "clip-0"}


def test_preemption_requeues_missing_frames(tmp_path):
    be = FakeBackend(preempt_at={"clip-0": 1})
    log = run_farm(be, make_job(tmp_path, list(range(8)), vms=1))
    assert existing_frames(tmp_path) == set(range(8))
    assert log["preemptions"] == 1 and len(be.created) == 2


class LostBetweenBatches(FakeBackend):
    """clip-0 renders one batch, then its next start fails: reclaimed (gone) or listed but unreachable."""

    def __init__(self, gone: bool):
        super().__init__()
        self.gone = gone

    def run(self, vm, cmd, timeout_s=120):
        if "nohup" in cmd and bare(vm.name) == "clip-0" and vm.name in self.batch:
            if self.gone:
                self.vms.pop(vm.name, None)
            raise VmError(f"ssh: connect to host {vm.ip} port 22: Connection timed out")
        return super().run(vm, cmd, timeout_s)


@pytest.mark.parametrize("gone", [True, False])
def test_vm_lost_between_batches_is_replaced(tmp_path, gone):
    be = LostBetweenBatches(gone)
    log = run_farm(be, make_job(tmp_path, list(range(8)), vms=1))
    assert existing_frames(tmp_path) == set(range(8))
    assert [bare(n) for n, _ in be.created] == ["clip-0", "clip-1"] and log["preemptions"] == int(gone)
    assert any(bare(n) == "clip-0" for n in be.deleted)


def test_resume_skips_existing_frames(tmp_path):
    (tmp_path / "frames").mkdir(exist_ok=True)
    for f in range(6):
        (tmp_path / "frames" / f"{f:05d}.png").write_bytes(b"x")
    be = FakeBackend()
    run_farm(be, make_job(tmp_path, list(range(8)), vms=2))
    assert all("--frames 0:" not in c and "--frames 1:" not in c for _, c in be.scripts)
    assert existing_frames(tmp_path) == set(range(8))


def test_keep_vms(tmp_path):
    be = FakeBackend()
    run_farm(be, make_job(tmp_path, [0, 1, 2], vms=1, keep_vms=True))
    assert be.deleted == []


def test_cycles_cache_pulled_once_and_pushed_to_late_vm(tmp_path):
    cyc = tmp_path / "cycles"
    be = FakeBackend(create_ok=lambda b, n, z: n.endswith("-0") or b.total_polls >= 4)
    job = make_job(tmp_path, list(range(60)), vms=3, retry_s=60, cycles_dir=cyc)
    run_farm(be, job)
    assert [(bare(n), r) for n, r, *_ in be.pulls if "cycles" in r] == [("clip-0", "$HOME/.cache/cycles")]
    assert (cyc / "kernel.bin").exists()
    late = [bare(n) for n, r in be.pushes if r == ".cache/cycles"]
    assert late and "clip-0" not in late


def test_zone_list_is_wide_and_us_first():
    assert len(ZONES) >= 30 and ZONES[0].startswith("us-central1")
    assert [z for z in ZONES if z.startswith("us-")] == list(ZONES[:13])


def test_rsync_remote_paths_are_home_relative():
    """rsync >= 3.2.4 does not expand $HOME in remote args (the test host has 3.2.7)."""
    from alloy.render.vm import rsync_remote_path

    assert rsync_remote_path("$HOME/alloy_render/job/out/frames") == "alloy_render/job/out/frames"
    assert rsync_remote_path("~/x") == "x" and rsync_remote_path("${HOME}/y") == "y"
    assert rsync_remote_path("alloy_render/bundle") == "alloy_render/bundle"


# ---- regions, quota, spread, cap, staging -------------------------------------------------------------------
def test_error_classification_uses_real_message_text():
    quota = ("ERROR: (gcloud.compute.instances.create) Could not fetch resource:\n"
             " - Quota 'PREEMPTIBLE_NVIDIA_L4_GPUS' exceeded.")
    assert classify_create_error(quota) is QuotaError
    assert classify_create_error("code: QUOTA_EXCEEDED") is QuotaError
    pool = "ZONE_RESOURCE_POOL_EXHAUSTED: The zone does not have enough resources available"
    assert classify_create_error(pool) is StockoutError
    assert classify_create_error("permission denied") is VmError


def test_spread_order_picks_fewest_vms_region_then_zone_order():
    book = RegionBook(ZONES, region_cap=4)
    regions = [region_of(book.reserve_next(set())) for _ in range(5)]
    assert regions[:5] == ["us-central1", "us-east1", "us-east4", "us-west1", "us-west4"]
    nxt = book.reserve_next(set())
    assert region_of(nxt) == "europe-west1" and nxt == "europe-west1-b"
    assert book.reserve_next({"us-central1-a"}) is not None
    book.counts["us-central1"] = 0
    book.counts["us-east1"] = 0
    assert book.ordered()[0].startswith("us-central1-a")  # ties: list order, zones in order


def test_region_cap_is_never_exceeded():
    book = RegionBook(ZONES, region_cap=2)
    for _ in range(2 * len(book.regions)):
        assert book.reserve_next(set()) is not None
    assert book.reserve_next(set()) is None
    assert max(book.counts.values()) == 2
    book.release("us-east4-a")
    assert region_of(book.reserve_next(set())) == "us-east4"


def test_quota_cools_the_whole_region_but_stockout_only_the_zone():
    c = Clock()
    book = RegionBook(("a-1-x", "a-1-y", "b-2-x"), region_cap=4, clock=c.now)
    book.cool_zone("a-1-x")
    assert book.ordered() == ["a-1-y", "b-2-x"] or book.ordered() == ["b-2-x", "a-1-y"]
    book.cool_region("a-1")
    assert book.ordered() == ["b-2-x"]
    c.t += 601
    assert set(book.ordered()) == {"a-1-x", "a-1-y", "b-2-x"}


def test_farm_quota_error_skips_rest_of_region(tmp_path):
    be = FakeBackend(quota_regions={"us-central1"})
    log = run_farm(be, make_job(tmp_path, list(range(8)), vms=1))
    tried = [z for _, z in be.create_calls]
    assert tried[0] == "us-central1-a" and sum(z.startswith("us-central1") for z in tried) == 1
    assert log["quota_hits"] == 1 and existing_frames(tmp_path) == set(range(8))


def test_farm_spreads_over_regions_and_respects_cap(tmp_path):
    be = FakeBackend()
    log = run_farm(be, make_job(tmp_path, list(range(60)), vms=12, region_cap=4))
    per_region: dict[str, int] = {}
    for _, z in be.created:
        per_region[region_of(z)] = per_region.get(region_of(z), 0) + 1
    assert len(per_region) >= 3 and max(per_region.values()) <= 4 and sum(per_region.values()) >= 12
    assert existing_frames(tmp_path) == set(range(60)) and len(log["regions"]) >= 3


def test_staging_off_uses_rsync_and_on_uses_bucket(tmp_path):
    off = FakeBackend()
    run_farm(off, make_job(tmp_path, list(range(8)), vms=2))  # <= 8 VMs: auto off
    assert off.staged_up == [] and any(r.endswith("/bundle") for _, r in off.pushes)
    on = FakeBackend()
    (tmp_path / "frames2").mkdir()
    job = make_job(tmp_path / "frames2", list(range(8)), vms=2, stage="gs://b/stage")
    (tmp_path / "frames2" / "look.json").write_text("{}")
    log = run_farm(on, job)
    assert log["staged"] and on.buckets == ["b"]
    assert any(u.endswith("/bundle") for u in on.staged_up) and on.staged_down
    assert not any(r.endswith("/bundle") for _, r in on.pushes)
    assert on.stage_deleted == []  # removed by delete_all, not by run()


def test_stage_auto_on_above_eight_vms_and_deleted_at_the_end(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOY_RENDER_BUCKET", "renders")  # the bucket comes from the environment or a config file
    be, job, c = FakeBackend(), make_job(tmp_path, list(range(30)), vms=9), Clock()
    farm = Farm(be, job, clock=c.now, sleep=c.sleep)
    assert farm.stager is not None and farm.stager.prefix.startswith("gs://renders/stage/")
    farm.run()
    farm.delete_all()
    assert be.stage_deleted == [farm.stager.prefix] and sorted(be.deleted) == sorted(n for n, _ in be.created)


def test_stage_failure_falls_back_to_rsync(tmp_path):
    be = FakeBackend(stage_fail=True)
    run_farm(be, make_job(tmp_path, list(range(8)), vms=2, stage="gs://b/stage"))
    assert be.staged_down == [] and any(r.endswith("/bundle") for _, r in be.pushes)
    assert existing_frames(tmp_path) == set(range(8))


def test_guided_batches_keep_every_worker_busy_to_the_end(tmp_path):
    be = FakeBackend()
    run_farm(be, make_job(tmp_path, list(range(100)), vms=10))
    sizes = [sum(int(b) - int(a) for a, b in re.findall(r"--frames (\d+):(\d+)", c)) for _, c in be.scripts]
    assert sizes[0] > sizes[-1] and min(sizes) >= 1 and sum(sizes) >= 100


def test_no_speculative_vms_when_the_crew_will_finish_first(tmp_path):
    be = FakeBackend(create_ok=lambda b, n, z: n.endswith("-0"))
    job = make_job(tmp_path, list(range(6)), vms=3, retry_s=30)
    run_farm(be, job)
    assert len(be.created) == 1  # 6 frames x 20 s / 1 worker < 180 s set-up: do not start more VMs


def test_prebuilt_image_is_used_when_present():
    from alloy.render.vm import IMAGE_FAMILY, RENDER_IMAGE_FAMILY, CmdResult, GcloudBackend

    def fake(found):
        def run(cmd, timeout_s, stdin=None):
            if "describe-from-family" in cmd:
                return CmdResult(0 if found else 1, "alloy-render-blender45-20261103-1800\n" if found else "not found")
            return CmdResult(0, "")
        return run

    assert RENDER_IMAGE_FAMILY in GcloudBackend(project="p", user="u", exec_fn=fake(True)).image_args()
    assert IMAGE_FAMILY in GcloudBackend(project="p", user="u", exec_fn=fake(False)).image_args()


def test_remote_script_creates_frames_dir():
    """A batch whose Blender dies before frame 1 must still leave out/frames, or every poll fails forever."""
    from alloy.render.farm_remote import RemoteLayout, remote_script

    job = SimpleNamespace(camera="v", fmt="png", samples=0)
    assert "mkdir -p $HOME/alloy_render/j/1/out/frames" in remote_script(job, RemoteLayout("j", 1), [(0, 3)])


def test_dereferenced_copies_symlinked_dirs(tmp_path):
    from alloy.render.vm import dereferenced

    real = tmp_path / "real_meshes"
    real.mkdir()
    (real / "m.npz").write_bytes(b"x")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "meshes").symlink_to(real, target_is_directory=True)
    with dereferenced(bundle) as src:
        assert src != bundle and (src / "meshes" / "m.npz").is_file() and not (src / "meshes").is_symlink()
    plain = tmp_path / "plain"
    plain.mkdir()
    with dereferenced(plain) as src:
        assert src == plain


class BrokenBlender(FakeBackend):
    """Blender exits 1 before writing any frame (e.g. a bundle whose meshes never arrived); pulling the empty
    frames dir errors like rsync does on a missing directory."""

    blender_log = "ERROR build: DeviceError: --device OPTIX requested but Cycles lists no OPTIX device"

    def _poll(self, name):
        self.total_polls += 1
        self.polls[name] += 1
        return "1"

    def pull(self, vm, remote, local, names=None):
        if "frames" in remote and not names:
            raise VmError("rsync: change_dir failed: No such file or directory (rc 23)")
        super().pull(vm, remote, local, names)


def test_blender_failing_everywhere_stops_the_job(tmp_path):
    """Regression: a job whose Blender always died before frame 1 polled forever and churned VMs."""
    be = BrokenBlender()
    with pytest.raises(FarmError, match="failed.*DeviceError"):
        run_farm(be, make_job(tmp_path, list(range(8)), vms=2))
    assert len(be.created) <= 6


def test_remote_script_gcs_mode_uploads_before_done():
    from alloy.render.farm_remote import RemoteLayout, poll_cmd, remote_script

    job = SimpleNamespace(camera="v", fmt="png", samples=0, frames_gcs="gs://b/run/frames/")
    script = remote_script(job, RemoteLayout("j", 1), [(0, 3)])
    inner = next(t for t in shlex.split(script) if "gcloud storage rsync" in t)  # the nohup bash -c body
    assert "gcloud storage rsync -q -x '(^|/)[.]' $HOME/alloy_render/j/1/out/frames gs://b/run/frames " in inner
    assert script.index("/out/rc") < script.index("wait $up") < script.rindex("/out/done")
    assert "out/uploaded" in poll_cmd("$HOME/x/out", "uploaded") and "-mmin" not in poll_cmd("$HOME/x/out", "uploaded")


def test_gcs_mode_pulls_no_frames(tmp_path):
    """With frames_gcs the VMs push frames themselves: the farm counts their upload marks and pulls nothing."""
    from alloy.render.farm_frames import FrameStore

    be = FakeBackend()
    job = make_job(tmp_path, list(range(8)), vms=2)
    job.frames_gcs = "gs://b/run/frames"
    store = FrameStore(job.out, job.frames_gcs, lister=lambda prefix: ["00001.png"])
    c = Clock()
    Farm(be, job, clock=c.now, sleep=c.sleep, frames=store).run()
    assert store.have() == set(range(8))
    assert not [r for _, r in be.pulls if r.endswith("/frames")]
    assert not any((job.out / "frames").iterdir())
    assert all("--frames 1:2" not in cmd for _, cmd in be.scripts)  # frame 1 was already in GCS


def test_partial_frames_are_never_frames():
    from alloy.render.farm_frames import FrameStore
    from alloy.render.farm_remote import uploader
    from alloy.render.farm_util import frame_index
    assert frame_index("00012.png") == 12 and frame_index("00012.exr") == 12
    assert frame_index(".00291.part.png") is None and frame_index("00291.part.png") is None
    assert frame_index("clip.mp4") is None and frame_index("00012") is None
    store = FrameStore(Path("/tmp/x"), "gs://b/frames", lister=lambda prefix: ["00001.png", ".00002.part.png"])
    assert store.have() == {1}
    assert "-x '(^|/)[.]'" in uploader("/o", "gs://b/frames")
