# SPDX-License-Identifier: Apache-2.0
"""Analytic baseline and the no-regression merge gate (SPEC D23, §3.10 `compare_to_baseline`).

`bench/goldens/analytic_baseline.json` holds one entry per (law, solver, device, mode) with every metric's verdict,
error and bound, failures included. A regression is a verdict that got worse (rank pass < inconclusive < fail <
not_run < error) or a metric error above the baseline error plus its bound. A result the baseline has no key for is
judged by its own verdict: anything but a pass is a regression too (audit AB-laws-1: a keyless law that failed used
to pass the gate), so a law enters the gate passing or with its known state recorded under a key. A run adds keys the
baseline does not have; changing an existing key needs an explicit update with a reason, recorded in `history`.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from alloy.bench.types import LawResult

log = logging.getLogger("alloy.bench")

SCHEMA = "alloy.bench.baseline/1"
DEFAULT_PATH = Path("bench/goldens/analytic_baseline.json")
RANK = {"pass": 0, "inconclusive": 1, "fail": 2, "not_run": 3, "error": 4}


def mode_of(result: LawResult) -> str:
    return "quick" if str(result.provenance.get("quick", "False")) == "True" else "full"


def key_of(result: LawResult) -> str:
    """`law|solver|device|mode`; a rig result names its dataset (`law@<dataset_id>`): two arms are two results."""
    ds = (result.data or {}).get("dataset_id", "")
    return f"{result.law}{'@' + ds if ds else ''}|{result.solver}|{result.device}|{mode_of(result)}"


def law_verdict(result: LawResult) -> str:
    """Worst metric verdict of a result (the law-level verdict)."""
    if not result.verdicts:
        return "error"
    return max((v.verdict for v in result.verdicts), key=lambda v: RANK[v])


def _finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def entry_of(result: LawResult) -> dict[str, Any]:
    metrics = {v.name: {"verdict": v.verdict, "error": _finite_or_none(v.error), "bound": _finite_or_none(v.bound),
                        "unit": v.unit} for v in result.verdicts}
    return {"law": result.law, "solver": result.solver, "device": result.device, "mode": mode_of(result),
            "verdict": law_verdict(result), "metrics": metrics, "commit": result.provenance.get("alloy", ""),
            "date": result.provenance.get("date", "")}


def load_baseline(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"schema": SCHEMA, "entries": {}, "history": []}
    data = json.loads(p.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {data.get('schema')!r} != {SCHEMA}")
    return data


def _metric_regressions(key: str, new: Mapping[str, Any], base: Mapping[str, Any]) -> list[str]:
    out = []
    for name, b in base.get("metrics", {}).items():
        n = new["metrics"].get(name)
        if n is None:
            out.append(f"{key}: metric {name} missing (baseline {b['verdict']})")
            continue
        if RANK[n["verdict"]] > RANK[b["verdict"]]:
            out.append(f"{key}: {name} verdict {b['verdict']} -> {n['verdict']}")
        elif b["verdict"] == "fail" and b["bound"] == 0:
            continue    # a zero-tolerance metric that already fails (a GPU MuJoCo run does not repeat: D1 / D3 on
            # cuda): its size is one run's sample, so growth past the stored sample is spread, not a regression
        elif None not in (n["error"], b["error"], b["bound"]) and n["error"] > b["error"] + b["bound"]:
            out.append(f"{key}: {name} error {n['error']:.6g} > baseline {b['error']:.6g} + bound {b['bound']:.6g}")
    return out


def _keyless(key: str, result: LawResult) -> list[str]:
    """The regression line of a result with no baseline key whose own verdict is not a pass (none when it passes)."""
    verdict = law_verdict(result)
    if verdict == "pass":
        return []
    worst = max(result.verdicts, key=lambda v: RANK[v.verdict], default=None)
    detail = f" ({worst.name} error {worst.error:.6g}, bound {worst.bound:.6g})" if worst is not None else ""
    return [f"{key}: no baseline key and its own verdict is {verdict}{detail}; fix the law, or record a known "
            "failure under a key (--add-new-keys, reviewed with the baseline file)"]


def compare_to_baseline(results: Sequence[LawResult], baseline: Path = DEFAULT_PATH) -> list[str]:
    """Regressions of `results` against the baseline file (empty list = no regression). A result without a key is
    held to its own verdict: a new key that passes is not a regression, any other verdict is (`_keyless`)."""
    entries = load_baseline(baseline)["entries"]
    out: list[str] = []
    for r in results:
        key = key_of(r)
        base = entries.get(key)
        if base is None:
            log.info("baseline: new key %s (%s)", key, law_verdict(r))
            out += _keyless(key, r)
            continue
        new = entry_of(r)
        if RANK[new["verdict"]] > RANK[base["verdict"]]:
            out.append(f"{key}: law verdict {base['verdict']} -> {new['verdict']}")
        out += _metric_regressions(key, new, base)
    return out


def missing_from_run(results: Sequence[LawResult], baseline: Path = DEFAULT_PATH, *, device: str, mode: str,
                     in_scope: Callable[[str], bool] = lambda law: True) -> list[str]:
    """Regressions for baseline keys of this `device` and `mode` (laws `in_scope` accepts) that a whole-suite run has
    no result for: a law that stopped running (removed, renamed, its producer broken) must not pass by being absent.
    A law retired on purpose leaves the baseline with `drop_keys` and a reason."""
    have = {key_of(r) for r in results}
    return [f"{key}: in the baseline ({e['verdict']}) but this run has no result for it; restore the law, or retire "
            "the key (alloy.bench.baseline.drop_keys, with a reason)"
            for key, e in load_baseline(baseline)["entries"].items()
            if key not in have and e["device"] == device and e["mode"] == mode and in_scope(e["law"])]


def drop_keys(keys: Iterable[str], path: Path = DEFAULT_PATH, *, reason: str) -> list[str]:
    """Remove retired keys from the baseline (a reason is required and recorded in `history`). Returns those removed."""
    if not reason:
        raise ValueError("dropping baseline entries needs a reason")
    data = load_baseline(path)
    gone = [k for k in keys if data["entries"].pop(k, None) is not None]
    if gone:
        data["history"].append({"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reason": reason, "dropped": True,
                                "keys": gone})
        Path(path).write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    return gone


def update_baseline(results: Iterable[LawResult], path: Path = DEFAULT_PATH, *, reason: str = "",
                    overwrite: bool = False) -> list[str]:
    """Add keys the baseline lacks (and replace existing ones when `overwrite`, which needs a reason).

    Returns the keys written. The history records date, reason and keys, so every change is traceable.
    """
    if overwrite and not reason:
        raise ValueError("overwriting baseline entries needs a reason")
    data = load_baseline(path)
    written = []
    for r in results:
        key = key_of(r)
        if key in data["entries"] and not overwrite:
            continue
        data["entries"][key] = entry_of(r)
        written.append(key)
    if written:
        data["entries"] = dict(sorted(data["entries"].items()))
        data["history"].append({"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reason": reason or "new keys",
                                "overwrite": overwrite, "keys": written})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    return written


__all__ = ["compare_to_baseline", "missing_from_run", "update_baseline", "drop_keys", "load_baseline", "key_of",
           "entry_of", "law_verdict", "RANK", "DEFAULT_PATH"]
