# SPDX-License-Identifier: Apache-2.0
"""SCR-DC5-4 (i): PRC-13 classification of VBD, MuJoCo + VBD under the Scheduler and MuJoCo + SPH, with and without
Alloy's ordered route on the MuJoCo entry, on the GPU. Measurement only: no solver or route code changes.

Each case runs ``alloy.tools.coupled_det_probe``, which builds the scene the way its users run it:
- R1: two fresh builds in one process, world digests after every control period;
- R2: the same run in a second process;
- R4: world 0 and world N-1 run alone (``--worlds 1 --world-ids k``), against the same world in the batch
  (per-world digests). ``sph_pour``'s worlds are identical, so its one-world run is compared with every world.
- claim: ``effective_determinism`` under the standard preset, as built.
- agreement: (claim >= RUN_TO_RUN) == (R1 and R2). A silent downgrade (RUN_TO_RUN claimed, digests differ) fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.testing.precision import compare_digests, probe_npz, write_row

pytestmark = [pytest.mark.gpu, pytest.mark.det, pytest.mark.slow]
CMD = "PYTHONPATH=. pytest tests/precision/test_coupled_det_gpu.py -q (spot L4 <gpu-vm>, one session, nohup)"
MODULE = "alloy.tools.coupled_det_probe"
# (case, worlds, control periods). MuJoCo + VBD runs at 64 worlds: the Scheduler's ADMM step takes 5-9 s per
# period there on the L4 (2026-11-03 smoke), about 20x the other cases. Its graph-profile runs segfaulted in
# IntegratedSimulation's period-graph replay until WP12's node budget (d0a3319c, 2355e5c2). Since then they run, but
# each exchange replays one at a time (~190 s per period at 64 worlds), so the graph cases use 6 periods: 3 eager,
# the capture, and 2 replays.
CASES = [
    ("mjc_sph_route", 1024, 10), ("mjc_sph", 1024, 10),
    ("mjc_sph_pour_route", 1024, 25), ("mjc_sph_pour", 1024, 25),
    ("vbd_cloth", 1024, 25), ("vbd_soft", 1024, 25),
    ("mjc_vbd_route_eager", 64, 20), ("mjc_vbd_eager", 64, 20),
    ("mjc_vbd_route", 64, 6), ("mjc_vbd", 64, 6),
]
IDENTICAL = {"mjc_sph", "mjc_sph_route", "mjc_sph_pour", "mjc_sph_pour_route"}  # sph_pour replicates one template
_SUMMARY: list[dict] = []


def _probe(tmp, tag: str, case: str, worlds: int, periods: int, *extra: str) -> tuple[dict, dict]:
    return probe_npz(tmp, tag, MODULE, "--case", case, "--worlds", str(worlds), "--periods", str(periods), *extra,
                     timeout=5400)


def _r4(tmp, case: str, batch: np.ndarray, periods: int) -> tuple[float, int]:
    """(share of compared worlds equal to their alone run at every period, first period with a difference or -1)."""
    worlds = batch.shape[1]
    if case in IDENTICAL:
        alone, _ = _probe(tmp, f"{case}-alone", case, 1, periods)
        eq = (batch == alone["digests_world"][0]).all(axis=-1)          # [periods + 1, worlds]
    else:
        cols = []
        for k in sorted({0, worlds - 1}):
            alone, _ = _probe(tmp, f"{case}-alone{k}", case, 1, periods, "--world-ids", str(k))
            cols.append((batch[:, k] == alone["digests_world"][0][:, 0]).all(axis=-1))
        eq = np.stack(cols, axis=1)
    first = -1 if eq.all() else int(np.argmax(~eq.all(axis=1)))
    return float(eq.all(axis=0).mean()), first


def _measured(r: tuple[float, int, float]) -> bool:
    return r[0] == 1.0 and r[1] == -1


@pytest.mark.timeout(7200)  # four probe processes per case; the driven pour at 1,024 worlds takes about 25 min
@pytest.mark.parametrize("case, worlds, periods", CASES)
def test_coupled_classification(case: str, worlds: int, periods: int, tmp_path) -> None:
    a, meta = _probe(tmp_path, f"{case}-a", case, worlds, periods, "--repeat", "2")
    assert meta["error"] is None, meta["error"]
    b, _ = _probe(tmp_path, f"{case}-b", case, worlds, periods)
    r1 = compare_digests(a["digests"][0], a["digests"][1])
    r2 = compare_digests(a["digests"][0], b["digests"][0])
    r4 = _r4(tmp_path, case, a["digests_world"][0], periods)
    measured = _measured(r1) and _measured(r2)
    claimed = meta["claim"] in ("RUN_TO_RUN", "GPU_TO_GPU")
    silent, agree = claimed and not measured, claimed == measured
    row = {"case": case, "worlds": worlds, "periods": periods, "claim": meta["claim"], "reasons": meta["reasons"],
           "r1": {"share": r1[0], "first": r1[1], "all": r1[2]}, "r2": {"share": r2[0], "first": r2[1], "all": r2[2]},
           "r4": {"share": r4[0], "first": r4[1]}, "agree": agree, "silent_downgrade": silent,
           "solvers": meta["solvers"], "stepper": meta["stepper"], "routes": meta["routes"],
           "use_graph": meta["use_graph"], "native_share": meta["native_share"],
           "period_ms_median": float(np.median(a["period_s"][0]) * 1e3), "mem_high_gib": meta["mem_high_bytes"] / 2**30}
    _SUMMARY.append(row)
    value = (f"{case}@{worlds}: claim {meta['claim']}; R1 {r1[0]:.1%} (first {r1[1]}), R2 {r2[0]:.1%} "
             f"(first {r2[1]}), R4 {r4[0]:.1%} (first {r4[1]}); {meta['stepper']} over {meta['solvers']}")
    write_row("PRC-13 case (DC5-4)", value, "fail" if silent else ("pass" if agree else "conservative"), CMD, **row)
    assert not silent, row


def test_coupled_summary() -> None:
    if len(_SUMMARY) < len(CASES):
        pytest.skip("run with the case tests in the same session")
    agree = sum(r["agree"] for r in _SUMMARY)
    silent = sum(r["silent_downgrade"] for r in _SUMMARY)
    write_row("PRC-13 (DC5-4 coupled cases)", f"agreement {agree}/{len(_SUMMARY)}; silent downgrades {silent}",
              "pass" if silent == 0 and agree == len(_SUMMARY) else ("fail" if silent else "partial"), CMD,
              cases=[(r["case"], r["claim"], r["r1"]["share"], r["r2"]["share"], r["r4"]["share"]) for r in _SUMMARY])
