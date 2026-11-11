# SPDX-License-Identifier: Apache-2.0
"""CPL-5 (SPEC §5.2): randomized box-on-cloth seeds, mass ratio 0.01-100, contact k 1e3-1e6 N/m, impacts <= 3 m/s;
0 NaN / ejections and every unstable configuration caught by the watchdog before 20 m/s. 256 seeds on GPU
(8 stiffness groups x 32 worlds; stiffness is model-wide in Newton)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from alloy.testing.coupling import ClothSceneSpec, PeriodRunner, cloth_scheduler, cloth_seed_model, record_metric

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
CONTROL_DT = 0.02
GROUPS, PER_GROUP = 8, 32
SECONDS = float(os.environ.get("ALLOY_CPL5_SECONDS", "0.4"))
EJECT_SPEED, EJECT_Z = 10.0, 2.0          # m/s, m: "ejected" = impossible motion (D12 bully S1) or thrown clear


def _seeds(rng):
    k = 10.0 ** rng.uniform(3.0, 6.0, GROUPS)
    ratio = 10.0 ** rng.uniform(-2.0, 2.0, (GROUPS, PER_GROUP))
    speed = rng.uniform(0.0, 3.0, (GROUPS, PER_GROUP))
    return k, ratio, speed


def _run_group(k, ratio, speed, spec):
    model = cloth_seed_model(ratio * spec.cloth_mass, speed, k, spec, device=DEV)
    sched = cloth_scheduler(model, spec)
    runner = PeriodRunner(model, sched, CONTROL_DT, log_exchanges=True, log_speed_only=True)
    for _ in range(int(round(SECONDS / CONTROL_DT))):
        rep = runner.run_period()
    tripped = runner.seen_escalated.copy()
    qd = np.array(runner.log.body_qd + runner.discarded.body_qd)[:, :, :3]   # (accepted + rolled-back exchanges, W, 3)
    speed_max = np.nanmax(np.linalg.norm(qd, axis=-1), axis=0)
    final_q = runner.state.body_q.numpy()[:, :3]
    nan = ~np.all(np.isfinite(qd), axis=(0, 2)) | ~np.all(np.isfinite(final_q), axis=1)
    final_v = np.linalg.norm(runner.state.body_qd.numpy()[:, :3], axis=1)
    ejected = (final_v > EJECT_SPEED) | (final_q[:, 2] > EJECT_Z)
    fell = final_q[:, 2] < 0.7                                 # tunnelled through the cloth (not a coupler event)
    for w in np.flatnonzero(nan | ejected):
        print(f"  world {w}: mass ratio {ratio[w]:.4g}, impact {speed[w]:.3g} m/s, k {k:.3g} N/m, nan {nan[w]}, "
              f"final z {final_q[w, 2]:.3f} m, final v {final_v[w]:.2f} m/s, max v {speed_max[w]:.2f} m/s, "
              f"escalated {tripped[w]}", flush=True)
    return {"nan": nan, "ejected": ejected, "fell": fell, "speed_max": speed_max, "tripped": tripped,
            "rollbacks": runner.rollbacks, "dt_x": float(rep.dt_x), "failed": rep.watch_state == 4}


@pytest.mark.gpu
@pytest.mark.timeout(5400)   # one L4: 2,781 s with SCR-WP11-4's force-jump re-runs off, 4,188 s on (8 jumps; WP4 2026-11-05)
def test_cpl5_seeds():
    rng = np.random.default_rng(2026)
    k, ratio, speed = _seeds(rng)
    spec = ClothSceneSpec()
    rows = []
    groups = [int(x) for x in os.environ.get("ALLOY_CPL5_GROUPS", ",".join(map(str, range(GROUPS)))).split(",")]
    for g in groups:
        r = _run_group(float(k[g]), ratio[g], speed[g], spec)
        rows.append(r)
        print(f"group {g}: k {k[g]:.3g} N/m, dt_x {r['dt_x']:.3g} s, rollbacks {r['rollbacks']}, nan {r['nan'].sum()}, "
              f"ejected {r['ejected'].sum()}, fell {r['fell'].sum()}, tripped {r['tripped'].sum()}, "
              f"max speed {np.nanmax(r['speed_max']):.2f} m/s", flush=True)
    nan = int(sum(r["nan"].sum() for r in rows))
    ejected = int(sum(r["ejected"].sum() for r in rows))
    fell = int(sum(r["fell"].sum() for r in rows))
    tripped = np.concatenate([r["tripped"] for r in rows])
    smax = np.concatenate([r["speed_max"] for r in rows])
    caught = int(np.sum(tripped & (smax < 20.0)))
    record_metric("CPL-5", seeds=len(groups) * PER_GROUP, groups=groups, seconds=SECONDS, nan=nan, ejected=ejected, fell_through=fell,
                  unstable_tripped=int(tripped.sum()), caught_below_20=caught,
                  max_speed=float(np.nanmax(smax)), rollbacks=int(sum(r["rollbacks"] for r in rows)), device=DEV)
    assert nan == 0 and ejected == 0
    assert caught == int(tripped.sum())
