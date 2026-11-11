# SPDX-License-Identifier: Apache-2.0
"""The calibrated G1 gate checks, as pure functions over metric tables (no image I/O, no GPU).

A *table* is ``{(spp, den): {"raw": {"flip_mean": f, "lpips": l, ...}, "matched": {...}}}`` for one scene, frame and
size: every image of one engine scored against the Cycles 4,096-spp reference.

Checks (gated on raw 8-bit sRGB; histogram-matched numbers are reported only):

- **parity**: the engine's best converged image (highest spp; of its denoise states, the lower LPIPS) has FLIP mean
  and LPIPS each <= 2 x the gap of Cycles seed 1 at 128 spp + denoise (same scene, frame, size).
- **G1 (end to end)**: the same limit applied to the image at the setting an engine claims (low spp + denoiser).
- **equal-spp rows**: at every (spp, denoise) the engine and a control share, ratio = max over FLIP/LPIPS of
  engine gap / worst control gap; a row passes at ratio <= 1.1.
- **equal-spp verdict** (statistical rule): PASS when the engine's row pass rate is >= the worst control seed's pass
  rate (each control judged against the others) and no row exceeds 1.25.
- **speed**: images/s rows against a bar (see :func:`check_speed`).

Example:
    >>> t = {(128, "on"): {"raw": {"flip_mean": 0.01, "lpips": 0.002}, "matched": {"flip_mean": 0.01, "lpips": 0.002}}}
    >>> g1_limit(t)
    {'flip_mean': 0.02, 'lpips': 0.004}
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

KEYS = ("flip_mean", "lpips")
PARITY_FACTOR = 2.0
ROW_FACTOR = 1.1
ROW_CAP = 1.25
BASE_KEY = (128, "on")

Table = Mapping[tuple[int, str], Mapping[str, Mapping[str, float]]]


def ratio(value: float, base: float) -> float:
    """value / base, with 0/0 = 1 and x/0 = inf.

    Example:
        >>> ratio(2.0, 1.0), ratio(0.0, 0.0)
        (2.0, 1.0)
    """
    if base > 0:
        return value / base
    return float("inf") if value > 0 else 1.0


def g1_limit(s1: Table) -> dict[str, float] | None:
    """The G1 limit for one scene, frame and size: 2 x Cycles seed 1's 128 spp + denoise gap; None without that image.

    Example:
        >>> g1_limit({}) is None
        True
    """
    base = s1.get(BASE_KEY)
    if base is None or "error" in base["raw"]:
        return None
    return {k: PARITY_FACTOR * base["raw"][k] for k in KEYS}


def g1_ratio(raw: Mapping[str, float], limit: Mapping[str, float]) -> float:
    """Worst of FLIP/limit and LPIPS/limit; <= 1 passes G1.

    Example:
        >>> g1_ratio({"flip_mean": 0.01, "lpips": 0.003}, {"flip_mean": 0.02, "lpips": 0.002})
        1.5
    """
    return max(ratio(raw[k], limit[k]) for k in KEYS)


def check_parity(eng: Table, s1: Table) -> dict:
    """Parity record for one engine at one scene, frame and size (status PASS/FAIL/n/a).

    Example:
        >>> check_parity({}, {})["status"]
        'n/a'
    """
    limit = g1_limit(s1)
    top = max((k[0] for k in eng), default=None)
    if limit is None or top is None:
        return {"status": "n/a", "reason": "needs cycles_s1 128spp_on and at least one engine image"}
    best = min((k for k in eng if k[0] == top), key=lambda k: eng[k]["raw"]["lpips"])
    got = {k: eng[best]["raw"][k] for k in KEYS}
    r = g1_ratio(got, limit)
    return {"status": "PASS" if r <= 1.0 else "FAIL", "best_image": f"{best[0]}spp_{best[1]}", "value": got,
            "threshold": limit, "ratio": r, "matched_value": {k: eng[best].get("matched", {}).get(k) for k in KEYS}}


def g1_images(eng: Table, s1: Table) -> dict[str, dict]:
    """End-to-end G1 for every image of an engine: ``{"<spp>spp_<den>": {"ratio", "pass"}}`` (empty without a limit).

    Example:
        >>> g1_images({}, {})
        {}
    """
    limit = g1_limit(s1)
    if limit is None:
        return {}
    out = {}
    for (spp, den), v in sorted(eng.items()):
        r = g1_ratio(v["raw"], limit)
        out[f"{spp}spp_{den}"] = {"ratio": r, "pass": r <= 1.0}
    return out


def yardstick(controls: Mapping[str, Table]) -> dict[tuple[int, str], dict[str, float]]:
    """Worst (max) control gap per (spp, den) and metric over the given control seeds.

    Example:
        >>> yardstick({})
        {}
    """
    keys = set().union(*[set(t) for t in controls.values()]) if controls else set()
    out = {}
    for key in keys:
        rows = [t[key]["raw"] for t in controls.values() if key in t and "error" not in t[key]["raw"]]
        if rows:
            out[key] = {m: max(r[m] for r in rows) for m in KEYS}
    return out


def equal_spp_rows(eng: Table, yard: Mapping[tuple[int, str], Mapping[str, float]]) -> list[dict]:
    """One row per (spp, den) shared by the engine and the yardstick; ratio = worst metric ratio to the yardstick.

    Example:
        >>> equal_spp_rows({}, {})
        []
    """
    rows = []
    for (spp, den) in sorted(eng):
        y = yard.get((spp, den))
        if y is None or "error" in eng[(spp, den)]["raw"]:
            continue
        raw = eng[(spp, den)]["raw"]
        r = {k: ratio(raw[k], y[k]) for k in KEYS}
        worst = max(r.values())
        rows.append({"spp": spp, "den": den, "ratio": worst, "ratio_by_metric": r, "pass": worst <= ROW_FACTOR,
                     "value": {k: raw[k] for k in KEYS}, "yardstick": dict(y)})
    return rows


def rows_for(eng_name: str, eng: Table, controls: Mapping[str, Table]) -> list[dict]:
    """Equal-spp rows of one engine against every control except itself.

    Example:
        >>> rows_for("ours", {}, {})
        []
    """
    others = {n: t for n, t in controls.items() if n != eng_name}
    return equal_spp_rows(eng, yardstick(others))


def pass_rate(rows: Iterable[dict]) -> float | None:
    """Fraction of rows that pass; None for no rows.

    Example:
        >>> pass_rate([{"pass": True}, {"pass": False}])
        0.5
    """
    rows = list(rows)
    return sum(r["pass"] for r in rows) / len(rows) if rows else None


def equal_spp_verdict(rows: list[dict], control_rows: Mapping[str, list[dict]]) -> dict:
    """Statistical equal-spp verdict over all rows of one engine in one cell (scene x size, all frames).

    PASS when (a) the engine's row pass rate >= the worst control seed's rate and (b) no row exceeds 1.25. Without at
    least one control judged against another, (a) needs every row to pass.

    Example:
        >>> equal_spp_verdict([{"pass": True, "ratio": 1.0}], {})["status"]
        'PASS'
    """
    if not rows:
        return {"status": "n/a", "reason": "no spp/denoise state shared with a control seed"}
    rates = {n: pass_rate(r) for n, r in control_rows.items() if r}
    bar = min(rates.values()) if rates else 1.0
    rate = pass_rate(rows) or 0.0
    worst = max(r["ratio"] for r in rows)
    ok = rate >= bar - 1e-12 and worst <= ROW_CAP
    return {"status": "PASS" if ok else "FAIL", "rows_pass": sum(r["pass"] for r in rows), "rows": len(rows),
            "pass_rate": rate, "control_rates": rates, "bar_rate": bar, "worst_ratio": worst, "cap": ROW_CAP}


def check_speed(rows: Iterable[Mapping], bar: Mapping) -> dict:
    """Best images/s among rows matching ``bar["match"]`` (every key equal) vs ``bar["min_images_per_s"]``.

    A bar without ``min_images_per_s`` is report-only. No matching row -> "pending".

    Example:
        >>> check_speed([{"res": "128x128", "images_per_s": 1200}], {"match": {"res": "128x128"}, "min_images_per_s": 1000})
        {'status': 'PASS', 'value': 1200, 'threshold': 1000}
    """
    match = bar.get("match", {})
    hits = [r for r in rows if all(r.get(k) == v for k, v in match.items()) and "images_per_s" in r]
    if not hits:
        return {"status": "pending", "reason": f"no speed row matching {dict(match)}"}
    best = max(r["images_per_s"] for r in hits)
    thr = bar.get("min_images_per_s")
    if thr is None:
        return {"status": "report", "value": best}
    return {"status": "PASS" if best >= thr else "FAIL", "value": best, "threshold": thr}
