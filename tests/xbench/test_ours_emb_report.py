# SPDX-License-Identifier: Apache-2.0
"""P3-2b report: per-pack SB-7 / SB-11 / SB-5 / SB-6 rows are recomputed from the results tree (CPU, no sim)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OURS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours"
sys.path.insert(0, str(OURS))

import report_emb  # noqa: E402

PACK = "emb_e1_fr3"


def _batch(d: Path, seed: int, worlds: int, s1: dict[int, str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    row = {"worlds": worlds, "s1_worlds": sorted(s1), "episodes": [[seed, i] for i in sorted(s1)],
           "s1_causes": {str(i): c for i, c in s1.items()}}
    (d / f"batch_{seed}_{worlds}.s1.json").write_text(json.dumps(row))


def _repeat(d: Path, setting: str, digests: list[list[str]]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, ours in enumerate(digests, 1):
        runs = {"a": {"ours": ours, "step_ms": 1.0}, "b_rerun": {"ours": ours}, "n1": {"ours": ours[:1]}}
        (d / f"{PACK}_{setting}_p{i}.json").write_text(json.dumps({"engine": "e", "worlds": len(ours), "runs": runs}))


def _tree(root: Path) -> Path:
    p = root / "eng" / PACK
    _batch(p / "sb7" / "recommended" / "BATCHES", 100000, 4, {1: "pen_s1_ratio"})
    _batch(p / "sb7" / "tuned", 100000, 4, {})
    _batch(p / "sb7" / "alt", 100000, 4, {0: "max_speed", 2: "max_speed"})
    (p / "sb7" / "alt" / "batch_100001_4.s1.json").touch()       # a scorer killed mid-write: left out, named
    _batch(p / "sb7_alloy", 100000, 4, {})
    _repeat(p / "repeat", "default", [["a", "b"], ["a", "b"], ["a", "c"]])
    _repeat(p / "repeat_alloy", "ordered", [["a", "b"]] * 3)
    (p / "repeat" / f"{PACK}_default_p4.json").touch()            # cut off mid-write: left out
    eng = [{"row": "recommended", "n": 1, "episodes_per_hour": 3600.0, "steps_per_s": 200.0, "real_time_factor": 2.0},
           {"row": "recommended", "n": 1024, "episodes_per_hour": 5e5, "steps_per_s": 1.0},
           {"row": "recommended", "n": 4096, "error": "CUDA error 700"},
           {"row": "tuned", "n": 1024, "episodes_per_hour": 1.0, "steps_per_s": 1.0}]
    (p / "throughput.jsonl").write_text("\n".join(json.dumps(r) for r in eng) + "\n")
    for n, sps in ((1, 1.3), (1024, 124.4)):
        recs = [{"nworld": n, "status": "ok", "sim_seconds_per_gpu_second": sps, "single_world_realtime_factor":
                 sps if n == 1 else None, "ms_per_step": 0.5}, {"nworld": 0, "status": "not_run"}]
        (p / f"throughput_alloy_{n}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return root


def test_pack_rows_recompute_sb7_sb11_and_speed(tmp_path):
    r = report_emb.pack_rows(_tree(tmp_path), "eng", PACK)
    assert (r["sb7"]["recommended"]["s1"], r["sb7"]["recommended"]["episodes"]) == (1, 4)
    assert r["sb7"]["tuned"]["s1"] == 0 and r["sb7"]["alloy"]["s1"] == 0 and r["sb7"]["alt"]["s1"] == 2
    assert r["sb7"]["alt"]["episodes"] == 4 and r["sb7"]["alt"]["unscored"] == ["batch_100001_4.s1.json"]
    assert r["sb11"]["default"]["r1_share"] == 1.0 and r["sb11"]["default"]["r2_share"] == 0.5
    assert r["sb11_alloy"]["ordered"]["sb11_share"] == 1.0
    assert r["speed"][1]["rtf"] == 2.0 and r["speed"][1]["ms_per_step"] == 5.0
    assert r["speed"][1024]["episodes_per_hour"] == 5e5 and "failed" in r["speed"][4096]
    assert abs(r["speed_alloy"][1024]["episodes_per_hour"] - 124.4 * 1800.0) < 1e-6 and 4096 not in r["speed_alloy"]


def test_tables_name_every_pack_and_keep_beside_rows_apart(tmp_path):
    rows = [report_emb.pack_rows(_tree(tmp_path), "eng", PACK)]
    sb7 = report_emb.sb7_table(rows, ["eng"])
    assert "| E1 fr3 | 25 % (1/4) | 0 % (0/4, up95 " in sb7 and sb7.count("not run") == 3 * (len(report_emb.PACKS) - 1)
    assert "| eng | E1 fr3 | alt | 50 % (2/4) |" in report_emb.beside_table(rows)
    assert "0.50 / failed; RTF 2.00" in report_emb.speed_table(rows, ["eng"])
    assert "default 0.875 (R1 1, R2 0.5)" in report_emb.sb11_table(rows, ["eng"])


def test_unclamped_port_cells_are_marked_and_a_clamp_rerun_replaces_them(tmp_path, monkeypatch):
    monkeypatch.setattr(report_emb, "UNCLAMPED", {("eng", PACK)})
    root = _tree(tmp_path)
    r = report_emb.pack_rows(root, "eng", PACK)
    assert r["actuator_clamp"] == "no"
    sb7 = report_emb.sb7_table([r], ["eng"])
    assert "| E1 fr3 | 25 % (1/4) † |" in sb7 and sb7.endswith(report_emb.FOOT)
    _batch(root / "eng" / f"{PACK}_clamp" / "sb7" / "recommended", 100000, 4, {})
    r = report_emb.pack_rows(root, "eng", PACK)
    assert r["actuator_clamp"] == "yes" and r["sb7"]["recommended"]["s1"] == 0
    assert r["sb7"]["recommended (unclamped)"]["s1"] == 1 and "†" not in report_emb.sb7_table([r], ["eng"])
