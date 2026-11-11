# SPDX-License-Identifier: Apache-2.0
"""SB-7 scoring service for an engine's campaign on the same VM (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/sb7_watch.py watch BATCH_DIR [--device cuda:0] [--keep]
    PYTHONPATH=. python bench/xbench/ours/sb7_watch.py summary BATCH_DIR

The engine runner writes one contract npz per batch, ``batch_<first seed>_<worlds>.npz``, frames every physics step,
world ``i`` = episode ``(seeds[i // 2048], i % 2048)`` with ``meta.seeds`` the batch's seeds (or, for B < 2048,
``(seeds[0], meta.row0 + i)``: ``batch_<seed>r<row0>_<B>.npz``), and then an empty
``<name>.done`` marker. ``watch`` scores each finished batch with Alloy's SB-7 checker (``score_s1.Replay``,
``alloy.bully.severity.classify``), writes ``<name>.s1.json``, saves every S1 episode's trajectory as
``trips/trip_<seed>_<k>.npz`` (replayable: its trace is ``b0.applied_traces(space, 2048, seed)[k]``), and deletes the
batch npz unless ``--keep``. It exits when ``STOP`` exists in BATCH_DIR and nothing is left. An engine-reported
capacity overflow (``meta.overflow_worlds``, or ``meta.overflow`` for the whole batch) is S1 ``capacity_overflow``.
An S1 world whose tripped S1 rows are all ones a robot contact can excuse (``e_gain_cum``, ``impossible_speed``) is
re-scored with robot touch from the pre- OR post-step frame (``score_s1 --touch union``) and counts only if an
excusable row still trips (WP7: at a coarse dt a fingertip that reaches a block during the step is not excused by
the pre-step contacts alone; the union only adds excuses).
``summary`` pools every scored batch: episodes, confirmed and raw S1 counts and causes, rate, one-sided 95 % upper
bound (Alloy's ``alloy.bully.rate.blowup_rate``).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

import b0  # noqa: E402
from packs_io import DEFAULT_TARGET  # noqa: E402
from score_s1 import Replay, _order, load, verdict  # noqa: E402

log = logging.getLogger("xbench.ours")


def _episode(meta: dict, i: int) -> tuple[int, int]:
    """(seed, row) of batch world ``i``: B / 2048 whole seeds, or rows ``row0 ..`` of one seed (B < 2048)."""
    seeds = meta["seeds"]
    if "row0" in meta:
        return int(seeds[0]), int(meta["row0"]) + i
    return int(seeds[i // b0.SEED_BATCH]), i % b0.SEED_BATCH


def _overflow(meta: dict, worlds: int) -> np.ndarray:
    out = np.zeros(worlds)
    if meta.get("overflow"):
        out[:] = 1.0
    for w in meta.get("overflow_worlds", []):
        out[int(w)] = 1.0
    return out


EXCUSED_ROWS = frozenset({"e_gain_cum", "impossible_speed"})   # S1 rows a robot-contact excuse can clear


def _s1_rows(extra: dict[str, np.ndarray], limits, i: int) -> list[str]:
    from alloy.bully.severity import MONITORS, S1

    return sorted(n for n, v in extra.items() if MONITORS[n].severity == S1
                  and not float(np.asarray(v)[i]) <= float(MONITORS[n].threshold(limits)))


def _post_touch(q: np.ndarray, qd: np.ndarray, t: np.ndarray, replays: dict, device: str,
                target: str = DEFAULT_TARGET) -> dict:
    """Re-score one world with robot touch from the pre- OR post-step frame (WP7: coarse-dt excuses); it confirms
    the S1 only if an excusable row still trips."""
    key = (target, "union", 1)
    if key not in replays:
        replays[key] = Replay(1, device, "union", target)
    rep = replays[key]
    extra = rep.run(q, qd, t)
    rows = [r for r in _s1_rows(extra, rep.limits, 0) if r in EXCUSED_ROWS]
    return {"union_touch_s1": bool(rows), "union_rows": rows}


def _confirm(res: dict, extra: dict, q, qd, t, limits, replays: dict, device: str, target: str) -> None:
    """Per S1 world: its tripped S1 rows; if every one is excusable by robot contact, the post-touch pass decides."""
    res["checks"] = {}
    for i in res["s1_worlds"]:
        rows = _s1_rows(extra, limits, i)
        chk: dict = {"s1_rows": rows, "confirmed": True}
        if rows and set(rows) <= EXCUSED_ROWS:
            chk.update(_post_touch(q[:, i:i + 1], qd[:, i:i + 1], t, replays, device, target))
            chk["confirmed"] = chk["union_touch_s1"]
        res["checks"][str(i)] = chk


def score_batch(path: Path, replays: dict, device: str) -> dict:
    z = load(path)
    meta = json.loads(str(z["meta"]))
    q, qd, t = z["body_q"], z["body_qd"], z["t"].astype(np.float64)
    w, target = q.shape[1], meta.get("alloy_target", DEFAULT_TARGET)
    if (target, "pre", w) not in replays:
        replays[(target, "pre", w)] = Replay(w, device, "pre", target)
    rep = replays[(target, "pre", w)]
    idx = _order([str(n) for n in z["body_names"]], rep.names)
    q, qd = q[:, :, idx], qd[:, :, idx]
    extra = rep.run(q, qd, t)
    extra["capacity_overflow"] = _overflow(meta, w)
    res = verdict(extra, rep.limits)
    res["episodes"] = [list(_episode(meta, i)) for i in res["s1_worlds"]]
    _confirm(res, extra, q, qd, t, rep.limits, replays, device, target)
    trips = path.parent / "trips"
    trips.mkdir(exist_ok=True)
    for i, (seed, k) in zip(res["s1_worlds"], res["episodes"], strict=True):
        np.savez_compressed(trips / f"trip_{seed}_{k}.npz", meta=np.array(json.dumps(dict(meta, worlds=1, seed=seed,
                            world=k))), body_names=np.array(rep.names), t=t, body_q=q[:, i:i + 1],
                            body_qd=qd[:, i:i + 1])
    res.update({"file": path.name, "worlds": w, "seeds": meta["seeds"], "row": meta.get("row"),
                "scorer_contacts": rep.pm.contact_stats(),
                "frame_dt": float(np.median(np.diff(t)))})
    res.pop("monitors")   # per-world values are large; keep the S1 worlds' values only
    res["s1_values"] = {str(i): {k: float(v[i]) for k, v in extra.items()} for i in res["s1_worlds"]}
    return res


def _score_exclusive(p: Path, replays: dict, device: str, gpu_lock: Path | None) -> dict:
    """Score under an flock shared with the engine's batch job (one GPU job at a time), then free the scorer's GPU
    memory."""
    if gpu_lock is None:
        return score_batch(p, replays, device)
    import fcntl
    import gc

    import warp as wp

    wp.set_mempool_release_threshold(device, 0)  # freed scorer arrays go back to the driver at the next sync
    with open(gpu_lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            return score_batch(p, replays, device)
        finally:
            replays.clear()
            gc.collect()
            wp.synchronize()


def _scored(f: Path) -> bool:
    """A score file counts only when it has content: a scorer killed mid-write leaves an empty one."""
    return f.exists() and f.stat().st_size > 0


def _write_atomic(f: Path, text: str) -> None:
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(f)


def watch(d: Path, device: str, keep: bool, gpu_lock: Path | None = None) -> None:
    replays: dict = {}
    while True:
        todo = sorted(p for p in d.glob("batch_*.npz") if p.with_suffix(".done").exists()
                      and not _scored(p.with_suffix(".s1.json")))
        if not todo:
            if (d / "STOP").exists():
                return
            time.sleep(10)
            continue
        for p in todo:
            t0 = time.perf_counter()
            res = _score_exclusive(p, replays, device, gpu_lock)
            res["score_wall_s"] = time.perf_counter() - t0
            _write_atomic(p.with_suffix(".s1.json"), json.dumps(res, indent=1))
            log.info("%s: S1 %s (%.1f s)", p.name, res["episodes"], res["score_wall_s"])
            if not keep:
                p.unlink(missing_ok=True)


def summary(d: Path) -> dict:
    from alloy.bully.rate import blowup_rate

    files = sorted(d.glob("batch_*.s1.json"))
    unscored = [p.name for p in files if not _scored(p)]
    if unscored:
        log.warning("%s: %d empty score file(s) left out of the summary: %s", d, len(unscored), unscored)
    rows = [json.loads(p.read_text()) for p in files if _scored(p)]
    episodes = sum(r["worlds"] for r in rows)
    trips = [e for r in rows for e in r["episodes"]]
    causes = collections.Counter(c for r in rows for c in r["s1_causes"].values())
    conf = [(e, r["s1_causes"][str(i)]) for r in rows for i, e in zip(r["s1_worlds"], r["episodes"], strict=True)
            if r.get("checks", {}).get(str(i), {}).get("confirmed", True)]
    rate, raw = blowup_rate(len(conf), episodes), blowup_rate(len(trips), episodes)
    return {"batches": len(rows), "episodes": episodes, "s1": len(conf), "s1_raw": len(trips),
            "causes": dict(collections.Counter(c for _, c in conf)), "causes_raw": dict(causes), "unscored": unscored,
            "rate": rate.rate, "upper95": rate.upper95, "method": rate.method, "passes": rate.passes,
            "upper95_raw": raw.upper95, "trips": [e for e, _ in conf], "unconfirmed": [e for e in trips
            if e not in [c for c, _ in conf]], "rows": sorted({r.get("row") for r in rows} - {None}),
            "rule": "an S1 whose tripped S1 rows are all robot-contact-excusable (e_gain_cum, impossible_speed) "
                    "counts only if a re-score with robot touch from the pre- or post-step frame still trips one (WP7)"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("watch", "summary"))
    p.add_argument("dir", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--gpu-lock", type=Path,
                   help="flock file shared with the engine job: score only while holding it, free GPU memory after")
    a = p.parse_args()
    if a.cmd == "watch":
        watch(a.dir, a.device, a.keep, a.gpu_lock)
    else:
        print(json.dumps(summary(a.dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
