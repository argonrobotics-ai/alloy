# SPDX-License-Identifier: Apache-2.0
"""P3-2b tables: SB-7, SB-11, SB-5 and SB-6 per embodiment pack, for each rival engine and Alloy beside it.

Reads the results tree ``p3-2b/<engine>/<pack>/`` laid out as in ``P3-2b-engines.md`` and recomputes every number
with Alloy's own code: ``sb7_watch.summary`` over the scored batches and ``score_repeat.shares`` over the repeat
files, so all engines are judged by the same checker.

    python bench/xbench/ours/report_emb.py [internal note] \
        --rows [internal note] \
        --doc [internal note]

The report and its rows stay outside this repo (rival-engine results that may not be published).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

import score_repeat  # noqa: E402
import sb7_watch  # noqa: E402
from report import fill_doc  # noqa: E402

log = logging.getLogger("xbench.report_emb")

PACKS = ("emb_e1_fr3", "emb_e2_panda", "emb_e3_ur10e_2f85", "emb_e4_fr3_bimanual", "emb_e5_fr3_duo", "emb_e8_go2",
         "emb_e9_leap", "emb_e6_g1", "emb_e7_h1_2")
VERDICT, TUNED = "recommended", "tuned"
EPISODE_S = 2.0                      # B0 horizon (P3-2.md): episodes per hour = sim-seconds per hour / 2 s
WORLDS = (1, 1024, 4096)
# XB4-4: ports that drove a pack without a joint actuatorfrcrange tighter than the actuator's own forcerange. Only E5
# has one (franka_spine_vertical_joint 100 N vs 670 N).
# Two rival ports used the actuator forcerange there (UNCLAMPED below).
# A pack directory's settings.json "actuator_clamp" overrides; a ``<pack>_clamp`` rerun replaces the engine's rows.
UNCLAMPED = {("maniskill", "emb_e5_fr3_duo"), ("isaaclab", "emb_e5_fr3_duo")}


def label(pack: str) -> str:
    """``emb_e1_fr3`` -> ``E1 fr3``."""
    _, num, *rest = pack.split("_")
    return f"{num.upper()} {'_'.join(rest)}"


def sb7_row(d: Path) -> dict | None:
    """Summary of one SB-7 row directory (its batches sit in ``BATCHES/`` or in the row directory itself)."""
    src = d / "BATCHES" if any((d / "BATCHES").glob("batch_*.s1.json")) else d
    if not any(src.glob("batch_*.s1.json")):
        return None
    s = sb7_watch.summary(src)
    return {k: s[k] for k in ("batches", "episodes", "s1", "s1_raw", "causes", "rate", "upper95", "passes", "unscored")}


def sb7_rows(pack_dir: Path) -> dict[str, dict]:
    """Every SB-7 row of a pack (verdict, tuned and beside rows), plus ``alloy`` for Alloy beside."""
    out = {d.name: r for d in sorted((pack_dir / "sb7").glob("*")) if d.is_dir() and (r := sb7_row(d))}
    if (alloy := sb7_row(pack_dir / "sb7_alloy")) is not None:
        out["alloy"] = alloy
    return out


def sb11_rows(d: Path, pack: str) -> dict[str, dict]:
    """SB-11 shares per setting from ``<pack>_<setting>_p<i>.json`` (three fresh processes each)."""
    groups: dict[str, list] = collections.defaultdict(list)
    for f in sorted(d.glob(f"{pack}_*_p*.json")):
        if f.stat().st_size == 0:                    # a process cut off mid-write (e.g. a preempted VM)
            log.warning("%s: empty repeat file left out", f)
            continue
        groups[f.stem[len(pack) + 1:].rsplit("_p", 1)[0]].append(json.loads(f.read_text()))
    return {s: score_repeat.shares(p) for s, p in groups.items()}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def engine_speed(pack_dir: Path) -> dict[int, dict]:
    """SB-5 / SB-6 of the engine's verdict row: episodes per hour per N, and ms per step + real-time factor at N=1."""
    out: dict[int, dict] = {}
    for r in _jsonl(pack_dir / "throughput.jsonl"):
        if r.get("row", VERDICT) != VERDICT or r.get("n") not in WORLDS or r.get("valid") is False:
            continue
        if r.get("error") or not r.get("episodes_per_hour"):
            out.setdefault(r["n"], {"failed": str(r.get("error", "no rate"))[:80]})
            continue
        out[r["n"]] = {"episodes_per_hour": r["episodes_per_hour"], "rtf": r.get("real_time_factor"),
                       "ms_per_step": 1e3 / r["steps_per_s"] if r.get("steps_per_s") else None}
    return out


def alloy_speed(pack_dir: Path) -> dict[int, dict]:
    """The same for Alloy beside, from the ladder records ``throughput_alloy_<N>.jsonl``."""
    out: dict[int, dict] = {}
    for n in WORLDS:
        rows = [r for r in _jsonl(pack_dir / f"throughput_alloy_{n}.jsonl") if r.get("nworld") == n]
        if not rows:
            continue
        ok = [r for r in rows if r.get("status") == "ok"]
        if not ok:
            out[n] = {"failed": str(rows[-1].get("error") or rows[-1].get("status"))[:80]}
            continue
        r = ok[-1]
        sps = r["sim_seconds_per_gpu_second"]
        out[n] = {"episodes_per_hour": sps * 3600.0 / EPISODE_S, "rtf": r.get("single_world_realtime_factor"),
                  "ms_per_step": r.get("ms_per_step")}
    return out


def clamp_tag(d: Path, engine: str, pack: str) -> str:
    f = d / "settings.json"
    said = json.loads(f.read_text()) if f.exists() else {}
    said = said.get("actuator_clamp") if isinstance(said, dict) else None
    return said or ("no" if (engine, pack) in UNCLAMPED else "yes")


def pack_rows(root: Path, engine: str, pack: str) -> dict:
    d, c = root / engine / pack, root / engine / f"{pack}_clamp"
    out = {"engine": engine, "pack": pack, "sb7": sb7_rows(d), "sb11": sb11_rows(d / "repeat", pack),
           "sb11_alloy": sb11_rows(d / "repeat_alloy", pack), "speed": engine_speed(d), "speed_alloy": alloy_speed(d),
           "actuator_clamp": clamp_tag(d, engine, pack)}
    if c.is_dir():                    # the clamped rerun: its rows replace the engine's, the old ones stay, labelled
        for k, v in sb7_rows(c).items():
            if k in out["sb7"]:
                out["sb7"][f"{k} (unclamped)"] = out["sb7"].pop(k)
            out["sb7"][k] = v
        out["sb11"] = sb11_rows(c / "repeat", pack) or out["sb11"]
        out["actuator_clamp"] = "yes"
    return out


def _dag(r: dict | None) -> str:
    return " †" if r and r.get("actuator_clamp") == "no" else ""


FOOT = "\n† Ported without the pack's joint actuator clamp (XB4-4): kept for the record, not compared.\n"


def _foot(table: str) -> str:
    return table + FOOT if "†" in table else table


def _pct(x: float) -> str:
    return f"{100 * x:.3g} %"


def sb7_cell(r: dict | None) -> str:
    if r is None:
        return "not run"
    up = f", up95 {_pct(r['upper95'])}" if r["rate"] < 0.01 else ""
    return f"{_pct(r['rate'])} ({r['s1']:,}/{r['episodes']:,}{up})"


def sb7_table(rows: list[dict], engines: list[str]) -> str:
    head = ["pack"] + [f"{e} {c}" for e in engines for c in ("verdict", "tuned 0.625 ms", "Alloy beside")]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for pack in PACKS:
        cells = [label(pack)]
        for e in engines:
            r = next((x for x in rows if x["engine"] == e and x["pack"] == pack), {"sb7": {}})
            cells += [sb7_cell(r["sb7"].get(k)) + (_dag(r) if k != "alloy" and k in r["sb7"] else "")
                      for k in (VERDICT, TUNED, "alloy")]
        lines.append("| " + " | ".join(cells) + " |")
    return _foot("\n".join(lines) + "\n")


def beside_table(rows: list[dict]) -> str:
    """SB-7 rows other than verdict / tuned / Alloy (e.g. alt settings, diagnostic rows)."""
    lines = ["| engine | pack | row | S1 rate | causes |", "|---|---|---|---|---|"]
    for r in rows:
        for name, s in r["sb7"].items():
            if name not in (VERDICT, TUNED, "alloy"):
                lines.append(f"| {r['engine']} | {label(r['pack'])} | {name} | {sb7_cell(s)} | {s['causes']} |")
    return "\n".join(lines) + "\n"


def _share(s: dict) -> str:
    v = s.get("sb11_share")
    return "-" if v is None else f"{v:.3g} (R1 {s['r1_share']:.2g}, R2 {s['r2_share']:.2g})"


def sb11_table(rows: list[dict], engines: list[str]) -> str:
    head = ["pack"] + [f"{e} {c}" for e in engines for c in ("engine", "Alloy beside")]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for pack in PACKS:
        cells = [label(pack)]
        for e in engines:
            r = next((x for x in rows if x["engine"] == e and x["pack"] == pack), None)
            for key in ("sb11", "sb11_alloy"):
                got = (r or {}).get(key) or {}
                cell = "; ".join(f"{k} {_share(v)}" for k, v in got.items()) or "not run"
                cells.append(cell + (_dag(r) if key == "sb11" and got else ""))
        lines.append("| " + " | ".join(cells) + " |")
    return _foot("\n".join(lines) + "\n")


def _speed_cell(sp: dict) -> str:
    def eph(n: int) -> str:
        x = sp.get(n)
        return "not run" if x is None else ("failed" if "failed" in x else f"{x['episodes_per_hour'] / 1e6:.2f}")
    one = sp.get(1) or {}
    rtf = f"{one['rtf']:.2f}" if one.get("rtf") else "-"
    return f"{eph(1024)} / {eph(4096)}; RTF {rtf}"


def speed_table(rows: list[dict], engines: list[str]) -> str:
    head = ["pack"] + [f"{e} {c}" for e in engines for c in ("engine", "Alloy beside")]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for pack in PACKS:
        cells = [label(pack)]
        for e in engines:
            r = next((x for x in rows if x["engine"] == e and x["pack"] == pack), {})
            cells += [_speed_cell(r.get("speed", {})) + (_dag(r) if r.get("speed") else ""),
                      _speed_cell(r.get("speed_alloy", {}))]
        lines.append("| " + " | ".join(cells) + " |")
    return _foot("\n".join(lines) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--engines", default="genesis,maniskill,isaaclab")
    p.add_argument("--rows", type=Path)
    p.add_argument("--doc", type=Path, action="append", default=[])
    a = p.parse_args()
    engines = [e for e in a.engines.split(",") if (a.root / e).is_dir()]
    rows = [pack_rows(a.root, e, k) for e in engines for k in PACKS if (a.root / e / k).is_dir()]
    if a.rows:
        a.rows.write_text(json.dumps(rows, indent=1, default=str))
    blocks = {"emb_sb7": sb7_table(rows, engines), "emb_beside": beside_table(rows),
              "emb_sb11": sb11_table(rows, engines), "emb_speed": speed_table(rows, engines)}
    for doc in a.doc:
        fill_doc(doc, blocks)
    log.info("%s", "\n".join(blocks.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
