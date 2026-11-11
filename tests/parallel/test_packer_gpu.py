# SPDX-License-Identifier: Apache-2.0
"""PAR-10 (planned throughput vs best hand-swept N, >= 90 %) and PAR-11 (memory prediction ±15 %, 0 OOMs), SPEC §5.2.

Per rung: probe N = 1, 64, 1024 -> WorldProfile; plan a 4,096-world request on this GPU -> pack size N*; hand-sweep
N over the ladder plus N*; PAR-10 = Θ(N*) / max_N Θ(N). PAR-11 compares ``predicted_bytes(profile, N)`` (from the
probe's linear memory fit) with the mempool bytes each measured pack added, at every swept N, and counts OOMs while
building the planned pack. Timing counts only from an exclusive run on an idle GPU (``gpu_run.sh --exclusive``).
"""

from __future__ import annotations

import os

import pytest

from alloy.core.newton_compat import NEWTON_PIN
from alloy.parallel import costmodel as cm
from alloy.parallel.packer import device_free_memory
from alloy.parallel.probe import ProfileMeta, TimingOptions, probe_targets, run_ladder_point, structure_of
from alloy.parallel.rungs import RawConfig, RawRung
from alloy.parallel.types import PackPolicy
from alloy.testing.parallel import record_metric

pytestmark = [pytest.mark.gpu, pytest.mark.nightly, pytest.mark.timeout(3600)]
RUNGS = [("box_pile", 16), ("cable_chain", 20), ("arm", 0)]
PROBE_N = (1, 64, 1024)
SWEEP_N = (1, 16, 64, 256, 1024, 2048, 4096)
REQUEST = 4096
OPTS = TimingOptions(window_s=float(os.environ.get("ALLOY_WINDOW_S", "2.0")), windows=5)
CMD = "scripts/gpu_run.sh WP8 'tests/parallel/test_packer_gpu.py' --vm <gpu-vm> --exclusive"


@pytest.mark.parametrize("scene,param", RUNGS)
def test_par10_par11(device, scene, param):
    if not device.startswith("cuda"):
        pytest.skip("GPU metric")

    def make(n: int):
        return RawRung(scene, param, n, device, RawConfig()).target()

    meta = ProfileMeta(f"raw:{scene}:{param}", "raw_mujoco", NEWTON_PIN, device)
    structure = structure_of(RawRung(scene, param, 1, device).model)
    prof, probe_runs = probe_targets(make, PROBE_N, device, meta, OPTS, structure)
    n_star = cm.pack_size(prof, REQUEST, device_free_memory(device), PackPolicy())
    runs = {r.n: r for r in probe_runs}
    ooms = 0
    for n in sorted(set(SWEEP_N) | {n_star}):
        if n in runs:
            continue
        try:
            runs[n] = run_ladder_point(make, n, device, OPTS)
        except Exception as exc:  # noqa: BLE001 - an OOM at a planned size is the PAR-11 failure we count
            if n == n_star:
                ooms += 1
            record_metric("PAR-11-oom", str(exc)[:300], "fail" if n == n_star else "info", CMD, scene=scene, n=n)
    theta = {n: n * r.sim_dt / r.step_s for n, r in runs.items()}
    best_n = max(theta, key=theta.get)
    par10 = theta[n_star] / theta[best_n]
    mem_err = {n: (cm.predicted_bytes(prof, n) - r.mem_bytes) / r.mem_bytes for n, r in runs.items()
               if r.mem_bytes and n >= 64}
    worst = max(mem_err.values(), key=abs) if mem_err else None
    ctx = dict(scene=scene, param=param, n_star=n_star, best_n=best_n, n_knee=cm.n_knee(prof),
               t_floor_s=prof.t_floor_s, c_w_s=prof.c_w_s, bytes_fixed=prof.bytes_fixed,
               bytes_per_world=prof.bytes_per_world, theta={str(k): v for k, v in sorted(theta.items())},
               mem_err={str(k): v for k, v in sorted(mem_err.items())})
    record_metric("PAR-10", par10, "pass" if par10 >= 0.9 else "fail", CMD, **ctx)
    ok11 = worst is not None and abs(worst) <= 0.15 and ooms == 0
    record_metric("PAR-11", {"worst_mem_err": worst, "ooms": ooms}, "pass" if ok11 else "fail", CMD, **ctx)
    assert par10 >= 0.9, ctx
    assert ok11, ctx
