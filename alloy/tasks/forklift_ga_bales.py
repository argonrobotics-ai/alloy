# SPDX-License-Identifier: Apache-2.0
"""load-bales (F/evaluator.py:674-929) and its own reward (F/evaluator.py:880-1052): two cardboard bales set side by
side on dunnage strips against a trailer's nose wall, the truck backed out onto the dock, the trailer held in place.

GA scores this task not by the fraction of criteria met (an untouched scene would read 0.909) but by
``1 if success, 0 if the world faulted, else 0.9 x progress x workmanship`` (F/evaluator.py:276-280); ``load_bales``
returns
that as ``Verdict.score`` with the breakdown in ``details['reward']``. Success and the reward's gates run over the
judged criteria only (a criterion that is not judged is never counted as passed; ``Verdict.summary()`` says how many).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.tasks.forklift_ga_geom import box_pose, clip, hull, in_convex, plan_gap, truck_frame, yaw_of, BoxPose
from alloy.tasks.forklift_ga_latches import onset_normal, trailer_max_move
from alloy.tasks.forklift_ga_shared import crit, episode_load, lean_limit, shared_criteria
from alloy.tasks.forklift_ga_types import (BOX_SIGNS, ContactEpisode, Episode, Verdict, judged_success,
                                           make_verdict)
from alloy.tasks.forklift_site import SITE, ForkliftSite, TrailerFrame, trailer_frame_from_pose, trailer_rear


@dataclass(frozen=True)
class RewardSpec:
    """load-bales-reward-v1 (every number authored; F/evaluator.py:913-915)."""

    fail_scale: float = 0.9  # F/evaluator.py:913
    exit_share: float = 0.1  # F/evaluator.py:913
    exit_gate_from: float = 0.9  # F/evaluator.py:913
    carry_cap: float = 0.8  # F/evaluator.py:914
    short_decay_m: float = 0.5  # F/evaluator.py:914
    carry_tilt_deg: float = 20.0  # F/evaluator.py:914
    carry_deadband_m: float = 0.02  # F/evaluator.py:914
    strike_merge_s: float = 0.25  # F/evaluator.py:914
    floor_merge_s: float = 0.3  # F/evaluator.py:915
    strand_factor: float = 0.8  # F/evaluator.py:915
    brush_factor: float = 0.9  # F/evaluator.py:915
    brush_decay_s: float = 1.0  # F/evaluator.py:915
    speed_floor: float = 0.3  # F/evaluator.py:915


REWARD = RewardSpec()
NOSE_WALL_MOVERS = ("load:bale_", "load:nose_dunnage_", "load:dunnage_")  # F/evaluator.py:827


@dataclass(frozen=True)
class Strip:
    """A dunnage strip's top face in the target trailer's frame."""

    name: str
    top: float
    rect: np.ndarray


@dataclass(frozen=True)
class BaleState:
    """One bale against the work order, in the target trailer's frame (F/evaluator.py:768-814)."""

    name: str
    tilt: float
    nose_gap: float
    front_edge_gaps: list
    inside: bool
    on_strips: bool
    strips: list
    placed: bool
    on_floor: bool
    in_trailer: bool
    centre_trailer: list
    xy: list
    plan: np.ndarray


def nose_wall_rule(ep: Episode, site: ForkliftSite, door: str):
    """A bale's (or strip's) contact with the target trailer's nose wall is a strike only if it approached at the impact
    speed or pressed harder than its weight, and then only above the strike force (F/evaluator.py:817-833)."""
    lim = site.limits

    def rule(mover: str, cls: str, e: ContactEpisode):
        if (cls != "trailer_wall" or not mover.startswith(NOSE_WALL_MOVERS) or len(e.parts) < 2
                or str(e.parts[1]) != f"{door}:nose"):
            return None
        if onset_normal(e) < lim.impact_speed and e.peak_n <= lean_limit(ep, site, episode_load(mover, e)):
            return False
        return e.peak_n > lim.strike_n
    return rule


def trailer_frames(ep: Episode, site: ForkliftSite, door: str) -> dict[str, TrailerFrame]:
    """Final frames of every docked trailer, the target first."""
    out = {door: trailer_frame_from_pose(site, ep.trailer_pos[-1], ep.trailer_quat_xyzw[-1])}
    for name, tr in ep.other_trailers.items():
        out[name] = trailer_frame_from_pose(site, tr.pos[-1], tr.quat_xyzw[-1])
    return out


def dunnage_tops(ep: Episode, frame: TrailerFrame) -> list[Strip]:
    """Every dunnage strip's top face in the target trailer's frame (F/evaluator.py:741-751)."""
    out = []
    for n in sorted(ep.loads):
        if "dunnage" in n:
            loc = frame.to_local(ep.loads[n][-1])
            top = loc[np.argsort(loc[:, 2])[4:]]
            out.append(Strip(n, float(top[:, 2].mean()), hull(top[:, :2])))
    return out


def floor_z(site: ForkliftSite, bed_z: dict[str, float], x: float, y: float) -> float:
    """The dock floor under a point: 0, or a deployed leveler deck rising to the trailer bed
    (F/evaluator.py:757-765)."""
    Ld, Wd = site.leveler.length, site.leveler.width
    for dn, bz in bed_z.items():
        dx, face = site.dock.door(dn)
        if abs(x - dx) <= Wd / 2 and face - Ld <= y <= face:
            return float(bz) * (y - (face - Ld)) / Ld
    return 0.0


def _under_strips(base: np.ndarray, low4: np.ndarray, nb: np.ndarray, strips: list[Strip], tol: float):
    def plane(q):
        return low4[0, 2] - (nb[0] * (q[:, 0] - low4[0, 0]) + nb[1] * (q[:, 1] - low4[0, 1])) / nb[2]
    clipped = [(s.name, s.top, clip(base, s.rect)) for s in strips]
    return [(n, a) for n, top, a in clipped if len(a) >= 3 and abs(float(plane(a).min()) - top) < tol]


def _front_edge_gaps(frame: TrailerFrame, loc: np.ndarray, R: np.ndarray, L: float) -> list[float]:
    """The face turned toward the nose, at both of its vertical edges, each at its nearer corner
    (F/evaluator.py:790-795)."""
    ax = frame.R.T @ R[:, :2]
    k, s = max(((k, s) for k in (0, 1) for s in (-1, 1)), key=lambda ks: ks[1] * ax[0, ks[0]])
    return [float(L - loc[(BOX_SIGNS[:, k] == s) & (BOX_SIGNS[:, 1 - k] == e), 0].max()) for e in (-1, 1)]


def _on_floor_now(site, frames, bed_z, pose: BoxPose) -> bool:
    low = pose.corners[int(np.argmin(pose.corners[:, 2]))]
    tol = site.bale.rest_tol
    for fr in frames.values():
        cc = fr.to_local(pose.centre)
        if 0.0 < cc[0] < fr.spec.L and abs(cc[1]) < fr.spec.W / 2:
            return float(fr.to_local(pose.corners)[:, 2].min()) < tol
    return float(low[2]) - floor_z(site, bed_z, float(low[0]), float(low[1])) < tol


def bale_state(ep: Episode, site: ForkliftSite, name: str, strips: list[Strip], floored: set, frames, door: str,
               bed_z: dict[str, float]) -> BaleState:
    frame, bs = frames[door], site.bale
    tr = frame.spec
    pose = box_pose(ep.loads[name][-1])
    loc, c = frame.to_local(pose.corners), frame.to_local(pose.centre)
    tilt = math.degrees(math.acos(min(1.0, abs(float(frame.R[:, 2] @ pose.R[:, 2])))))
    low4 = loc[np.argsort(loc[:, 2])[:4]]
    base = hull(low4[:, :2])
    nb = frame.R.T @ pose.R[:, 2]
    nb = nb if nb[2] > 0 else -nb
    under = _under_strips(base, low4, nb, strips, bs.rest_tol)
    on_strips = bool(under) and in_convex(c[:2], hull(np.vstack([a for _, a in under])))
    edges = _front_edge_gaps(frame, loc, pose.R, tr.L)
    inside = bool(np.all((loc[:, 0] > 0) & (loc[:, 0] < tr.L + 0.005) & (np.abs(loc[:, 1]) < tr.W / 2)))
    at_wall = all(bs.nose_gap[0] <= g <= bs.nose_gap[1] for g in edges)
    placed = inside and on_strips and tilt <= bs.upright_deg and at_wall
    on_floor = bool(name in floored and not on_strips and _on_floor_now(site, frames, bed_z, pose))
    return BaleState(name, tilt, float(tr.L - loc[:, 0].max()), edges, inside, on_strips, [n for n, _ in under],
                     bool(placed), on_floor, bool(0.0 < c[0] < tr.L and abs(c[1]) < tr.W / 2), [float(c[0]),
                                                                                                float(c[1])],
                     [float(pose.centre[0]), float(pose.centre[1])], hull(loc[:, :2]))


def truck_past_face(ep: Episode, site: ForkliftSite, door: str) -> float:
    """How far the truck's farthest point (footprint corners and both blade tips) stands past the door's face (m)."""
    p, R = truck_frame(ep)
    ys = [c[1] for c in site.truck.corners(float(p[0]), float(p[1]),
                                           yaw_of(R))] + [float(t[1]) for t in ep.fork_tips[-1]]
    return float(max(ys) - site.dock.door(door)[1])


# ------------------------------------------------------------------------------------------------ reward
def to_nose_row(site: ForkliftSite, frames: dict[str, TrailerFrame], door: str, xy) -> float:
    """Plan distance from a world point to the target trailer's nose row (F/evaluator.py:932-955): inside that trailer
    straight; from outside, to its rear opening and on; from inside another trailer, out through its rear sill first."""
    fr = frames[door]
    tr = fr.spec
    lx, ly = site.bale.size[:2]
    xg, yg = tr.L - lx / 2, (tr.W - ly) / 2

    def local(name, p):
        f = frames[name]
        return f.to_local(np.array([p[0], p[1], f.origin[2]]))[:2]

    def inside(name, r):
        t_ = frames[name].spec
        return 0.0 <= r[0] <= t_.L and abs(r[1]) <= t_.W / 2
    p, d = np.asarray(xy[:2], float), 0.0
    for name in frames:
        if name != door and inside(name, local(name, p)):
            d, p = d + float(np.linalg.norm(local(name, p))), frames[name].origin[:2]
    r = local(door, p)
    if not inside(door, r):
        q = np.array([0.0, float(np.clip(r[1], -tr.W / 2, tr.W / 2))])
        d, r = d + float(np.linalg.norm(r - q)), q
    return d + float(np.hypot(abs(xg - r[0]), max(0.0, abs(r[1]) - yg)))


def strike_events(strikes: list[ContactEpisode], merge_s: float, strike_n: float, impact: float) -> list[dict]:
    """Strike episodes as contact events, merged per (mover, structure) when < merge_s apart
    (F/evaluator.py:958-973)."""
    out: list[dict] = []
    for s in sorted(strikes, key=lambda s: (s.mover, s.structure, s.t or 0.0)):
        v = onset_normal(s)
        sev = (v / impact) ** 2 if v >= impact else max(1.0, s.peak_n / strike_n)
        t0 = float(s.t or 0.0)
        t1 = math.inf if s.open or s.t_end is None else float(s.t_end)
        last = out[-1] if out else None
        if last and (last["mover"], last["structure"]) == (s.mover, s.structure) and t0 - last["t_end"] < merge_s:
            last.update(severity=max(last["severity"], sev), t_end=max(last["t_end"], t1),
                        episodes=last["episodes"] + 1)
        else:
            out.append(dict(mover=s.mover, structure=s.structure, t=t0, t_end=t1, severity=sev, episodes=1))
    return out


def floor_events(dropped: list, onsets: dict, strikes: list[ContactEpisode], merge_s: float) -> tuple[dict, set]:
    """Each bale's floor contacts as events (entries < merge_s apart are one); an event's approach speed is the largest
    onset latched in it, folding in the bale's trailer_floor strikes that began in it (F/evaluator.py:976-999)."""
    out: dict[str, list[dict]] = {}
    folded: set[int] = set()
    for name, t, _ in sorted(dropped, key=lambda r: (r[0], r[1])):
        evs = out.setdefault(name, [])
        if evs and t - evs[-1]["t1"] < merge_s:
            evs[-1]["t1"] = float(t)
        else:
            evs.append(dict(t0=float(t), t1=float(t), v=0.0))
    for name, evs in out.items():
        for e in evs:
            e["v"] = max([float(v) for t, v in onsets.get(name,
                                                          []) if e["t0"] - merge_s <= t <= e["t1"] + merge_s] or [0.0])
        for i, s in enumerate(strikes):
            t = float(s.t or 0.0)
            hit = [e for e in evs if s.structure == "trailer_floor" and s.mover == "load:" + name
                   and e["t0"] - merge_s <= t <= e["t1"] + merge_s]
            if hit:
                hit[0]["v"] = max(hit[0]["v"], onset_normal(s))
                folded.add(i)
    return out, folded


def bale_progress(site: ForkliftSite, frames, door: str, bales: list[BaleState], start_xy: dict,
                  rw: RewardSpec) -> dict:
    """Each bale's placement progress from its current state only (F/evaluator.py:1008-1019)."""
    bs, p = site.bale, {}
    for b in bales:
        lean = 1.0 - max(0.0, b.tilt - bs.upright_deg) / (rw.carry_tilt_deg - bs.upright_deg)
        if b.on_floor or lean <= 0.0:
            p[b.name] = 0.0
        elif b.placed:
            p[b.name] = 1.0
        elif b.inside and b.on_strips:
            short = max(0.0, max(b.front_edge_gaps) - bs.nose_gap[1]) / rw.short_decay_m
            p[b.name] = (1.0 - (1.0 - rw.carry_cap) * min(1.0, short)) * min(1.0, lean)
        else:
            d0 = to_nose_row(site, frames, door, start_xy[b.name]) - rw.carry_deadband_m
            p[b.name] = rw.carry_cap * max(0.0, (d0 - to_nose_row(site, frames, door, b.xy)) / d0)
    return p


def _workmanship(ep, site, by, sr, folded_strikes, events, floor_evs, lying, rw: RewardSpec) -> dict[str, float]:
    lim = site.limits
    impact = lim.set_down_speed

    def speed(v):
        return 1.0 if v <= impact else max(rw.speed_floor, 1.0 / (1.0 + math.log10(v / impact)))

    def brush(e):
        return rw.strand_factor + (rw.brush_factor - rw.strand_factor) / (1.0 + (e["t1"] - e["t0"]) / rw.brush_decay_s)

    def floor_factor(name, evs):
        f = float(np.prod([speed(e["v"]) for e in evs]))
        return f * (rw.strand_factor if name in lying else float(np.prod([brush(e) for e in evs])))

    def m(name):
        return by[name].value
    grade = {
        "no_strike": lambda: float(np.prod([1.0 / (1.0 + 0.5 * math.log10(e["severity"])) for e in events])),
        "no_dropped_load": lambda: float(np.prod([floor_factor(n, evs) for n, evs in floor_evs.items()])),
        "no_wheel_lift": lambda: 1.0 / (1.0 + max(0.0, m("no_wheel_lift") - lim.wheel_lift_s)),
        "roll_under_6deg": lambda: 0.0 if sr.overturn else 1.0 / (1.0 + max(0.0,
                                                                            sr.max_roll - lim.roll_deg) / lim.roll_deg),
        "speed_limit": lambda: 1.0 / (1.0 + sr.over_limit_s / 10.0 + 2.0 * max(0.0,
                                                                               m("speed_limit") / lim.limit_kmh - 1.0)),
        "operator_seated": lambda: 1.0 / (1.0 + max(0.0, m("operator_seated") - lim.seat_open_s)),
        "head_clear_of_guard": lambda: min(1.0, lim.head_contact_n / max(m("head_clear_of_guard"), 1e-9)),
        "trailer_movement": lambda: 1.0 / (1.0 + max(0.0, m("trailer_movement") - site.trailer.max_move)
                                           / site.trailer.max_move),
    }
    return {k: float(f()) for k, f in grade.items() if by[k].passed is False}


def bale_reward(ep, site, door, sr, by, bales, frames, start_xy, rw: RewardSpec = REWARD) -> dict:
    """load-bales-reward-v1 (F/evaluator.py:1002-1052): progress x workmanship, the factors only of rules that broke."""
    floor_evs, folded = floor_events(sr.dropped, sr.floor_onsets, sr.strikes, rw.floor_merge_s)
    p = bale_progress(site, frames, door, bales, start_xy, rw)
    placement = float(np.mean(sorted(p.values())[-2:]))
    gate = min(1.0, max(0.0, (placement - rw.exit_gate_from) / (1.0 - rw.exit_gate_from)))
    out_ = float(by["truck_back_on_dock"].passed)
    progress = (1.0 - rw.exit_share) * placement + rw.exit_share * out_ * gate
    events = strike_events([s for i, s in enumerate(sr.strikes) if i not in folded], rw.strike_merge_s,
                           site.limits.strike_n, site.limits.impact_speed)
    lying = {b.name for b in bales if b.on_floor or b.tilt > rw.carry_tilt_deg}
    factors = _workmanship(ep, site, by, sr, folded, events, floor_evs, lying, rw)
    return dict(contract="load-bales-reward-v1",
                rule="1 if success, 0 if the world faulted, else 0.9 x progress x workmanship",
                progress=progress, placement=placement, truck_out=out_, exit_gate=gate,
                workmanship=float(np.prod(list(factors.values()) or [1.0])), factors=factors, bales=p,
                strike_events=events, floor_events=floor_evs, fail_scale=rw.fail_scale)


# ------------------------------------------------------------------------------------------------ the task
def load_bales(ep: Episode, site: ForkliftSite = SITE, door: str = "door3", bed_z: float | None = None,
               bale_start_xy: dict | None = None) -> Verdict:
    """load-bales (F/evaluator.py:844-929). door: 'door3' or 'door4' (the target trailer; the other is in
    ``ep.other_trailers``). bed_z: the spotted floor height of the docked trailers (default: the target's first sample).
    bale_start_xy: each bale's start centre (default: its first sample). Preset: dock."""
    frames = trailer_frames(ep, site, door)
    bz = float(ep.trailer_pos[0][2]) if bed_z is None else float(bed_z)
    beds = {door: bz, **{n: bz for n in ep.other_trailers}}
    sr = shared_criteria(ep, site, "load-bales", "dock", dest_floor=lambda n, xy: not n.startswith("bale"),
                         strike_rule=nose_wall_rule(ep, site, door))
    criteria = list(sr.criteria)
    strips = dunnage_tops(ep, frames[door])
    floored = {n for n, _, _ in sr.dropped}
    bales = [bale_state(ep, site, n, strips, floored, frames, door,
                        beds) for n in sorted(ep.loads) if n.startswith("bale_")]
    at_nose = [b for b in bales if b.placed]
    loaded = [b.name for b in bales if b.in_trailer]
    gaps = [plan_gap(a.plan, b.plan) for i, a in enumerate(at_nose) for b in at_nose[i + 1:]]
    clear = max(gaps) if gaps else None
    ok = clear is not None and clear >= -site.bale.overlap_tol and len(loaded) == 2
    criteria.append(crit("two_bales_at_nose", dict(at_nose=[b.name for b in at_nose], bales_in_trailer=loaded,
                                                   clearance_m=clear),
                         ok, "-",
                         "exactly two bales in the trailer, each upright (<= 5 deg) on dunnage strips, its front "
                         "face <= 75 mm from the front wall at both front edges; the two not overlapping (<= 5 mm)",
                         "F/evaluator.py:858-866"))
    past = truck_past_face(ep, site, door)
    criteria.append(crit("truck_back_on_dock", past, past < 0.0, "m",
                         "the whole truck, counterweight to fork tips, behind the door face", "F/evaluator.py:867-869"))
    mv = trailer_max_move(ep, trailer_rear(site, door))
    criteria.append(crit("trailer_movement", mv, mv <= site.trailer.max_move, "m", "<= 25 mm from its spotted position",
                         "F/evaluator.py:870-872"))
    by = {c.name: c for c in criteria}
    start = bale_start_xy or {b.name: ep.loads[b.name][0].mean(axis=0)[:2] for b in bales}
    reward = bale_reward(ep, site, door, sr, by, bales, frames, start)
    success = judged_success(criteria, ep.fault)
    own = reward["fail_scale"] * reward["progress"] * reward["workmanship"]
    score = 1.0 if success else 0.0 if ep.fault is not None else own
    details = dict(strikes=sr.strikes, dropped=sr.dropped, bales=[_public(b, reward) for b in bales], reward=reward)
    return make_verdict("load-bales", criteria, ep.fault, details, score)


def _public(b: BaleState, reward: dict) -> dict:
    d = {k: v for k, v in b.__dict__.items() if k != "plan"}
    return dict(d, progress=reward["bales"][b.name])

