# SPDX-License-Identifier: Apache-2.0
"""GA's weld-task judgements on the sections (SCR-WP8-14): a port of Gym-Anything's welding evaluator (GA
``benchmarks/roboworld/environments/welding/task.py``: ``_eval_fillet_2f``, ``fillet_progress``, ``_craters``,
``judge_half_fillet``, ``judge_tack_and_weld``), on the rows of ``weld_ga_measures.measure`` (GA's keys).

- fillet-2f: continuous fused metal from <= 5 mm of each end; size >= 6.35 mm on 90 % of 5-145 mm and >= 3.95 mm
  everywhere; median throat >= 0.707 x 6.35 mm; convexity within D1.1 Fig 5.4 on every section holding metal; overlap
  one cell (0.25 mm) over the span and every crater, two cells in a start zone; macro sections at 6, 37.5, 75, 112.5
  and 144 mm fused (root, a cell into each member); every crater (10 mm about where a pool froze last, run on to the
  pass's front, less a bead's 1 mm end face) at 6.35 mm; undercut, porosity; solid.
- Progress: the length-weighted share of 5-145 mm that is sound, fused weld (min(1, size / 6.35 mm) per section),
  less what is judged outside the span, net of the start state's (the fixture's tacks); score 1 on success, else
  0.9 x progress.
- half-fillet: the same over 5-75 mm (macro sections at 6, 37.5, 70 mm), the craters touching it or laid by the
  episode, the metal beyond it sound, 100-120 mm bare (progress less 0.5 x its share holding metal).
- tack-and-weld: fillet-2f on the final weld, and the assembly (bonded, square, placed, released), which holds by
  construction here (the bracket is a fixed part of the grid); progress net of the loose start (no weld).
Undercut and porosity read 0 (heat v3: parent metal never leaves, no shielding gas), so their checks pass whenever a
weld covers the span.
"""

from __future__ import annotations

import numpy as np

L_BRACKET, NOMINAL_LEG, UNDERSIZE = 0.150, 6.35e-3, 2.4e-3
END_ALLOWANCE, COVERAGE_FOR_QUALITY = 5e-3, 0.9
OVERLAP_RESOLUTION, START_OVERLAP_MAX, PENETRATION_MIN, CELL_TOL = 2.5e-4, 5e-4, 2.5e-4, 1e-9
UNDERCUT_LIMIT, UNDERCUT_MAX, UNDERCUT_OVER_1MM_LENGTH, POROSITY_MAX = 1e-3, 2e-3, 50e-3, 0.015
CRATER_LEN, CRATER_END_FACE, EPISODE_GAP_S = 10e-3, 1e-3, 2.0
STOP_RADIUS, FRONT_REACH = CRATER_LEN / 2, 2 * CRATER_LEN
FAIL_SCORE_SCALE, FREE_ZONE_PENALTY = 0.9, 0.5
FUSION_X = (6e-3, 0.25 * L_BRACKET, 0.5 * L_BRACKET, 0.75 * L_BRACKET, 144e-3)   # GA: off its tacks (8-23, 127-142)
HALF_SPAN, HALF_FUSION_X, HALF_LEFT_FREE = (END_ALLOWANCE, L_BRACKET / 2), (6e-3, 0.25 * L_BRACKET, 70e-3), (0.1, 0.12)


def _key(r):
    return round(r["x_m"] * 1e6)


def _in(r, lo, hi):
    return lo - 1e-9 <= r["x_m"] <= hi + 1e-9


def _span(r):
    return _in(r, END_ALLOWANCE, L_BRACKET - END_ALLOWANCE)


def _joint(r):
    return 0.0 <= r["x_m"] <= L_BRACKET


def _overlap(r):
    return max(r["overlap_base_m"], r["overlap_bracket_m"])


def _undercut(r):
    return max(r["undercut_base_m"], r["undercut_bracket_m"])


def _convexity_limit(r):
    """D1.1 Fig 5.4: 1/16 in for a face up to 5/16 in wide, else 1/8 in."""
    return 3.2e-3 if r["face_width_m"] > 7.9e-3 else 1.6e-3


def _fused(r):
    """An etched macro section shows it fused: the root melted and both members entered by a cell."""
    return bool(r["root_fused"] and r["penetration_base_m"] >= PENETRATION_MIN - CELL_TOL
                and r["penetration_bracket_m"] >= PENETRATION_MIN - CELL_TOL)


def undercut_beyond(rows) -> set:
    """Keys of sections whose 1-2 mm undercut lies beyond the 50 mm D1.1 allows (shallowest kept first)."""
    mid = sorted((r for r in rows if _joint(r) and UNDERCUT_LIMIT + CELL_TOL < _undercut(r) <= UNDERCUT_MAX + CELL_TOL),
                 key=lambda r: (_undercut(r), r["x_m"]))
    used, out = 0.0, set()
    for r in mid:
        if used + r["dx_m"] <= UNDERCUT_OVER_1MM_LENGTH + CELL_TOL:
            used += r["dx_m"]
        else:
            out.add(_key(r))
    return out


def defects(r, beyond=frozenset(), overlap_limit=OVERLAP_RESOLUTION) -> list[str]:
    """The conditions that reject one section."""
    out = []
    if _overlap(r) > overlap_limit + CELL_TOL:
        out.append("overlap")
    if _undercut(r) > UNDERCUT_MAX + CELL_TOL or _key(r) in beyond:
        out.append("undercut")
    if r["porosity"] > POROSITY_MAX:
        out.append("porosity")
    if r["convexity_m"] > _convexity_limit(r) + CELL_TOL:
        out.append("convexity")
    return out


def credit(r, beyond=frozenset(), overlap_limit=OVERLAP_RESOLUTION) -> tuple[float, str | None, list]:
    """One section's credit: min(1, size / 6.35 mm) if solid, fused and sound, else 0; why not; its defects."""
    if r.get("liquid"):
        return 0.0, "liquid", []
    if r["weld_area_m2"] <= 0 or r["size_m"] <= 0:
        return 0.0, "no_fillet", []
    if not _fused(r):
        return 0.0, "unfused", []
    bad = defects(r, beyond, overlap_limit)
    if bad:
        return 0.0, "defective", bad
    return min(1.0, r["size_m"] / NOMINAL_LEG), None, []


def beads(rows) -> list[list[dict]]:
    """Contiguous sections holding weld metal, each cut to its fillet (first to last section with size > 0)."""
    runs, cur = [], []
    for r in rows:
        if r["weld_area_m2"] > 0:
            cur.append(r)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        idx = [i for i, r in enumerate(run) if r["size_m"] > 0]
        if idx:
            out.append(run[idx[0]:idx[-1] + 1])
    return out


def episodes(bead) -> list[list[int]]:
    """[a, b) of a bead's freeze episodes: adjacent sections that froze within EPISODE_GAP_S (outside the joint never
    splits one); a section still freezing belongs to none."""
    eps, cur = [], None
    for i, r in enumerate(bead):
        t = r.get("freeze_t")
        if t is None:
            if cur:
                eps.append(cur)
            cur = None
        elif cur and (abs(t - bead[i - 1]["freeze_t"]) <= EPISODE_GAP_S or not (_joint(r) and _joint(bead[i - 1]))):
            cur[1] = i + 1
        else:
            if cur:
                eps.append(cur)
            cur = [i, i + 1]
    if cur:
        eps.append(cur)
    return eps


def _stops(bead, a, b, edge):
    """The sections of episode [a, b) that froze last within STOP_RADIUS (not at a start that a neighbour outlived),
    and whether one is the last section read (the pass ran on beyond it)."""
    later = lambda q, r: q.get("freeze_t") is None or q["freeze_t"] > r["freeze_t"]  # noqa: E731
    ep, stops, ran_on = bead[a:b], [], []
    for j, r in enumerate(ep):
        if any(q["freeze_t"] > r["freeze_t"] for q in ep if abs(q["x_m"] - r["x_m"]) <= STOP_RADIUS + 1e-9):
            continue
        if (j == 0 and a > 0 and later(bead[a - 1], r)) or (j == len(ep) - 1 and b < len(bead) and later(bead[b], r)):
            continue
        (ran_on if r["x_m"] in edge else stops).append(r)
    return stops, ran_on


def _window(ep, group, lo, hi):
    """A crater's sections: within CRATER_LEN / 2 of its stops, run on to the nearer end of its episode within
    FRONT_REACH, inside the joint, less a bead's own end face."""
    x0, x1 = group[0]["x_m"], group[-1]["x_m"]
    e0, e1 = ep[0]["x_m"] - ep[0]["dx_m"] / 2, ep[-1]["x_m"] + ep[-1]["dx_m"] / 2
    up = e1 - x1 <= x0 - e0 and e1 - x1 <= FRONT_REACH + 1e-9
    down = x0 - e0 < e1 - x1 and x0 - e0 <= FRONT_REACH + 1e-9

    def near(r):
        return (any(abs(r["x_m"] - s["x_m"]) <= CRATER_LEN / 2 + 1e-9 for s in group)
                or (up and r["x_m"] >= x1) or (down and r["x_m"] <= x0))
    return [r for r in ep if near(r) and _joint(r) and r["x_m"] - lo > CRATER_END_FACE
            and hi - r["x_m"] > CRATER_END_FACE]


def craters(rows) -> list[dict]:
    """The weld's craters, wherever a pool froze last (D1.1:2000 Table 6.1(3)): {rows, stop_mm, min_size_mm, ok}."""
    edge = (rows[0]["x_m"], rows[-1]["x_m"]) if rows else ()
    out = []
    for bead in beads(rows):
        lo, hi = bead[0]["x_m"] - bead[0]["dx_m"] / 2, bead[-1]["x_m"] + bead[-1]["dx_m"] / 2
        eps = episodes(bead)
        found = []
        for a, b in eps:
            stops, ran_on = _stops(bead, a, b, edge)
            out += [dict(rows=[], how="ran on past the sections read", stop_mm=[round(r["x_m"] * 1e3, 2)],
                         min_size_mm=None, ok=True) for r in ran_on]
            found.append((bead[a:b], stops, "last to freeze"))
        if not eps:
            found.append((bead, [bead[0], bead[-1]], "bead end (no freeze times)"))
        for ep, stops, how in found:
            groups = []                      # stops within CRATER_LEN of each other are one crater
            for s in sorted(stops, key=lambda r: r["x_m"]):
                if groups and s["x_m"] - groups[-1][-1]["x_m"] <= CRATER_LEN + 1e-9:
                    groups[-1].append(s)
                else:
                    groups.append([s])
            for g in groups:
                sel = _window(ep, g, lo, hi)
                low = min(sel, key=lambda r: r["size_m"]) if sel else None
                out.append(dict(rows=sel, how=how, stop_mm=[round(s["x_m"] * 1e3, 2) for s in g],
                                min_size_mm=round(low["size_m"] * 1e3, 4) if low else None,
                                at_x_mm=round(low["x_m"] * 1e3, 2) if low else None,
                                ok=bool(low is None or low["size_m"] >= NOMINAL_LEG - 1e-9)))
    return out


def fillet_progress(rows, start_raw=None, found=None) -> dict:
    """fillet-2f's progress: the sound share of 5-145 mm less what is judged outside it (a crater's size deficit, a
    start zone's rejected metal), net of the start state's raw: (raw - start_raw) / (1 - start_raw) in [0, 1]."""
    found = craters(rows) if found is None else found
    beyond = undercut_beyond(rows)
    in_crater = {_key(r) for c in found for r in c["rows"]}
    req = [r for r in rows if _span(r)]
    length = sum(r["dx_m"] for r in req)
    total = sum(r["dx_m"] * credit(r, beyond)[0] for r in req)
    outside = 0.0
    for r in rows:
        if not _joint(r) or _span(r) or r["weld_area_m2"] <= 0:
            continue
        if _key(r) in in_crater:
            outside += r["dx_m"] * (1.0 - credit(r, beyond)[0])
        elif defects(r, beyond, START_OVERLAP_MAX):
            outside += r["dx_m"]
    raw = (total - outside) / length if length > 0 else 0.0
    net = (raw - start_raw) / (1.0 - start_raw) if start_raw is not None and start_raw < 1.0 else raw
    progress = float(np.clip(net, 0.0, 1.0)) if start_raw is None or start_raw < 1.0 else float(raw >= 1.0)
    return dict(progress=progress, net_progress=float(net), raw=raw, start_raw=start_raw)


def continuous(rows) -> tuple[float, float] | None:
    """The longest run of sections holding fused weld metal: (from, to) [m]."""
    best, cur, start, bs = 0, 0, 0, 0
    for k, r in enumerate(rows):
        if r["weld_area_m2"] > 0 and r["root_fused"]:
            start = k if cur == 0 else start
            cur += 1
            if cur > best:
                best, bs = cur, start
        else:
            cur = 0
    if not best:
        return None
    run = rows[bs:bs + best]
    return run[0]["x_m"] - run[0]["dx_m"] / 2, run[-1]["x_m"] + run[-1]["dx_m"] / 2


def _fusion(rows, xs) -> list[dict]:
    out = []
    for xq in xs:
        r = min(rows, key=lambda r_: abs(r_["x_m"] - xq)) if rows else None
        out.append(dict(x_mm=round(xq * 1e3, 3), ok=bool(r and _fused(r))))
    return out


def fillet_2f(rows) -> tuple[dict, list[dict]]:
    """fillet-2f's checks {name: {"ok": bool, ...}} on the rows (GA ``_eval_fillet_2f``), and the craters."""
    joint = [r for r in rows if _joint(r)]
    inside = [r for r in rows if _span(r)]
    metal = [r for r in joint if r["weld_area_m2"] > 0]
    length = sum(r["dx_m"] for r in inside) or 1.0
    size = np.array([r["size_m"] for r in inside]) if inside else np.zeros(1)
    dxs = np.array([r["dx_m"] for r in inside]) if inside else np.ones(1)
    frac = float(np.sum(dxs * (size >= NOMINAL_LEG - 1e-9)) / length)
    found = craters(rows)
    judged = [c for c in found if c["rows"]]
    in_crater = {_key(r) for c in judged for r in c["rows"]}
    profiled = [r for r in joint if _span(r) or _key(r) in in_crater]
    start_zone = [r for r in joint if not _span(r) and _key(r) not in in_crater and r["weld_area_m2"] > 0]
    conv_bad = [round(r["x_m"] * 1e3, 2) for r in metal if r["convexity_m"] > _convexity_limit(r) + CELL_TOL]
    ov_bad = [round(r["x_m"] * 1e3, 2) for r in profiled if _overlap(r) > OVERLAP_RESOLUTION + CELL_TOL] + \
        [round(r["x_m"] * 1e3, 2) for r in start_zone if _overlap(r) > START_OVERLAP_MAX + CELL_TOL]
    uc = [_undercut(r) for r in joint]
    uc_over = sum(r["dx_m"] for r, u in zip(joint, uc, strict=True) if u > UNDERCUT_LIMIT + CELL_TOL)
    covered = float(np.sum(dxs * (size > 0)) / length) if inside else 0.0
    present = covered >= COVERAGE_FOR_QUALITY
    run = continuous(rows)
    throat = float(np.median([r["throat_m"] for r in inside])) if inside else 0.0
    fus = _fusion(rows, FUSION_X)
    worst = min(judged, key=lambda c: c["min_size_mm"]) if judged else None
    freezing = sum(1 for r in rows if r["weld_area_m2"] > 0 and r.get("freeze_t") is None)
    checks = {
        "continuous_length": dict(from_to_mm=[round(v * 1e3, 2) for v in run] if run else None,
                                  ok=bool(run and run[0] <= END_ALLOWANCE + 1e-9
                                          and run[1] >= L_BRACKET - END_ALLOWANCE - 1e-9)),
        "leg_size": dict(fraction_size_ge_6_35mm=frac, min_size_mm=float(size.min() * 1e3),
                         ok=bool(frac >= 0.9 - 1e-9 and size.min() >= NOMINAL_LEG - UNDERSIZE)),
        "throat": dict(median_mm=throat * 1e3, ok=bool(inside and throat >= 0.707 * NOMINAL_LEG)),
        "convexity": dict(max_mm=max((r["convexity_m"] for r in metal), default=0.0) * 1e3, over_mm=conv_bad[:40],
                          ok=bool(present and not conv_bad)),
        "undercut": dict(max_mm=max(uc, default=0.0) * 1e3, ok=bool(present and max(uc, default=0.0) <= UNDERCUT_MAX
                                                                   + CELL_TOL and uc_over <= UNDERCUT_OVER_1MM_LENGTH)),
        "overlap": dict(max_mm=max((_overlap(r) for r in profiled), default=0.0) * 1e3, over_mm=ov_bad[:40],
                        ok=bool(present and not ov_bad)),
        "fusion_sections": dict(sections=fus, ok=all(f["ok"] for f in fus)),
        "porosity": dict(max_fraction=max((r["porosity"] for r in metal), default=0.0),
                         ok=bool(present and max((r["porosity"] for r in metal), default=0.0) <= POROSITY_MAX)),
        "crater_filled": dict(min_size_mm=worst["min_size_mm"] if worst else 0.0,
                              at_x_mm=worst["at_x_mm"] if worst else None,
                              craters=[{k: v for k, v in c.items() if k != "rows"} for c in found],
                              ok=bool((judged or any(c["how"].startswith("ran on") for c in found))
                                      and all(c["ok"] for c in judged))),
        "solidified": dict(liquid_sections=sum(1 for r in rows if r["liquid"]), freezing_sections=freezing,
                           ok=bool(present and not any(r["liquid"] for r in rows) and freezing == 0)),
    }
    return checks, found


def _scored(checks: dict, progress: dict) -> dict:
    success = all(c["ok"] for c in checks.values())
    return dict(success=success, score=1.0 if success else FAIL_SCORE_SCALE * progress["progress"],
                progress=progress["progress"], reward=progress, checks=checks)


def judge_fillet(rows, start_rows=None) -> dict:
    """fillet-2f (and set-up-and-weld's weld): every check; progress net of the start state (its tacks)."""
    checks, found = fillet_2f(rows)
    start_raw = fillet_progress(start_rows)["raw"] if start_rows else None
    return _scored(checks, fillet_progress(rows, start_raw, found))


def judge_tack_and_weld(rows) -> dict:
    """tack-and-weld: fillet-2f's checks on the final weld; the assembly (bonded, square within 1°, placed, released)
    holds by construction (the bracket is a fixed part of the grid); progress net of the loose start (no weld)."""
    checks, found = fillet_2f(rows)
    for name in ("bonded", "square", "position", "hand_off"):
        checks[name] = dict(by_construction=True, ok=True)
    return _scored(checks, fillet_progress(rows, 0.0, found))


def free_share(rows, lo, hi) -> float:
    """The share of [lo, hi] holding weld metal."""
    sec = [r for r in rows if _in(r, lo, hi)]
    length = sum(r["dx_m"] for r in sec)
    return float(sum(r["dx_m"] for r in sec if r["weld_area_m2"] > 0) / length) if length > 0 else 0.0


def episode_metal(start_rows):
    """True where a section holds weld metal the episode laid or melted again (not the start's, untouched)."""
    old = {_key(r): r.get("freeze_t") for r in (start_rows or []) if r["weld_area_m2"] > 0}

    def new(r):
        if r["weld_area_m2"] <= 0:
            return False
        if _key(r) not in old:
            return True
        t, t0 = r.get("freeze_t"), old[_key(r)]
        return t is None or t0 is None or abs(t - t0) > 1e-9
    return new


def _window_checks(rows, lo, hi, fusion_x, beyond) -> dict:
    """fillet-2f's span checks over [lo, hi]: continuous fused metal, size, median throat, every section sound."""
    sec = [r for r in rows if _in(r, lo, hi)]
    dxs = np.array([r["dx_m"] for r in sec]) if sec else np.ones(1)
    size = np.array([r["size_m"] for r in sec]) if sec else np.zeros(1)
    frac = float(np.sum(dxs * (size >= NOMINAL_LEG - 1e-9)) / dxs.sum())
    covered = float(np.sum(dxs * (size > 0)) / dxs.sum())
    unfused = [round(r["x_m"] * 1e3, 2) for r in sec if not (r["weld_area_m2"] > 0 and r["root_fused"])]
    bad = [(round(r["x_m"] * 1e3, 2), defects(r, beyond)) for r in sec if r["weld_area_m2"] > 0 and defects(r, beyond)]
    throat = float(np.median([r["throat_m"] for r in sec])) if sec else 0.0
    fus = _fusion(rows, fusion_x)
    return {"continuous": dict(unfused_or_bare_mm=unfused[:40], ok=bool(sec and not unfused)),
            "leg_size": dict(fraction_size_ge_6_35mm=frac, min_size_mm=float(size.min() * 1e3),
                             ok=bool(frac >= 0.9 - 1e-9 and size.min() >= NOMINAL_LEG - UNDERSIZE)),
            "throat": dict(median_mm=throat * 1e3, ok=bool(sec and throat >= 0.707 * NOMINAL_LEG)),
            "sound": dict(rejected=bad[:40], ok=bool(covered >= COVERAGE_FOR_QUALITY and not bad)),
            "fusion_sections": dict(sections=fus, ok=all(f["ok"] for f in fus))}


def _half_reading(rows, new) -> tuple[list, set, set, float]:
    """half-fillet's craters (touching the half, or of the episode's metal) and its raw progress."""
    lo, hi = HALF_SPAN
    beyond = undercut_beyond(rows)
    found = [c for c in craters(rows) if c["rows"] and any(_in(r, lo, hi) or new(r) for r in c["rows"])]
    in_crater = {_key(r) for c in found for r in c["rows"]}
    sec = [r for r in rows if _in(r, lo, hi)]
    length = sum(r["dx_m"] for r in sec)
    charge = 0.0
    for r in rows:
        if not _joint(r) or _in(r, lo, hi) or r["weld_area_m2"] <= 0:
            continue
        if _key(r) in in_crater:
            charge += r["dx_m"] * (1.0 - credit(r, beyond)[0])
        elif r["x_m"] < lo and defects(r, beyond, START_OVERLAP_MAX):
            charge += r["dx_m"]
    window = sum(r["dx_m"] * credit(r, beyond)[0] for r in sec) / length if length > 0 else 0.0
    return found, in_crater, beyond, window - (charge / length if length > 0 else 0.0)


def judge_half_fillet(rows, start_rows=None) -> dict:
    """half-fillet: 5-75 mm as fillet-2f judges its span; beyond it the craters' metal at one cell of overlap, a start
    zone's at two, the episode's other metal for convexity and porosity; craters filled; 100-120 mm bare; solid."""
    lo, hi = HALF_SPAN
    new = episode_metal(start_rows)
    found, in_crater, beyond, raw = _half_reading(rows, new)
    checks = _window_checks(rows, lo, hi, HALF_FUSION_X, beyond)
    outside = []
    for r in rows:
        if not _joint(r) or _in(r, lo, hi) or r["weld_area_m2"] <= 0:
            continue
        if _key(r) in in_crater:
            bad = defects(r, beyond)
        elif r["x_m"] < lo:
            bad = defects(r, beyond, START_OVERLAP_MAX)
        else:
            bad = [d for d in defects(r, beyond) if d in ("convexity", "porosity")] if new(r) else []
        if bad:
            outside.append((round(r["x_m"] * 1e3, 2), bad))
    worst = min(found, key=lambda c: c["min_size_mm"]) if found else None
    checks["sound_beyond_half"] = dict(rejected=outside[:40], ok=not outside)
    checks["crater_filled"] = dict(min_size_mm=worst["min_size_mm"] if worst else 0.0,
                                   craters=[{k: v for k, v in c.items() if k != "rows"} for c in found],
                                   ok=bool(found and all(c["ok"] for c in found)))
    joint = [r for r in rows if _joint(r)]
    checks["undercut"] = dict(ok=max((_undercut(r) for r in joint), default=0.0) <= UNDERCUT_MAX + CELL_TOL)
    share = free_share(rows, *HALF_LEFT_FREE)
    checks["left_half_unwelded"] = dict(metal_share=share, ok=share == 0.0)
    freezing = sum(1 for r in rows if r["weld_area_m2"] > 0 and r.get("freeze_t") is None)
    checks["solidified"] = dict(ok=not any(r["liquid"] for r in rows) and freezing == 0)
    raw0 = _half_reading(start_rows, episode_metal(start_rows))[3] if start_rows else 0.0
    net = (raw - raw0) / (1.0 - raw0) if raw0 < 1.0 else float(raw >= 1.0)
    return _scored(checks, dict(progress=float(np.clip(net - FREE_ZONE_PENALTY * share, 0.0, 1.0)), raw=raw,
                                start_raw=raw0, net_progress=float(net), left_half_metal_share=share))


__all__ = ["craters", "fillet_2f", "fillet_progress", "judge_fillet", "judge_half_fillet", "judge_tack_and_weld"]
