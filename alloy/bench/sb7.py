# SPDX-License-Identifier: Apache-2.0
"""SB-7 (bully S1 rate per scene class) feeds, and the label the board puts on task_t12's S2 count.

Coordinator ruling 2026-11-05 14:58Z (scr/WP3.md, "T12 S2: accepted as arrival penetration"): T12's S2 episodes are
transient arrival depth (out in about 6 ms), not resting sink-in, which SB-3 measures; the board labels the count with
WP3's depth percentiles beside it. Correction 18:54Z: WP3's L1 rows show that smaller steps do not cut the depth
(> 1 mm in 46 / 54 / 59 % of episodes at dt, dt/2, dt/4), so the label drops "speed-proportional" (approach speed
times one substep) and reads "arrival penetration (transient), cause under re-derivation".

WP3 measured the percentiles on 8,192 fresh task_t12 episodes on aeac0ab1 ([internal note]);
the label says so, so a later campaign on another commit shows where its percentiles came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alloy.bench.headline_types import Feed

RULING = "coordinator 2026-11-05 14:58Z, label corrected 18:54Z"
ARRIVAL = "arrival penetration (transient), cause under re-derivation"
DEPTHS = ("p50 / p95 / p99 0.97 / 2.02 / 2.76 mm (WP3, 8,192 task_t12 episodes on aeac0ab1, "
          "[internal note])")
_NUM = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_BOUND = r"bound(?: 3/N =)? " + _NUM
_UNIT = "S1 per episode (95 % upper bound)"
_S2 = re.compile(r"S2 in ([\d,]+) episodes \((\d+(?:\.\d+)?) ?%")
_CLASS = r"\b[a-z]\w*_\w+\b"        # first snake_case name in an SB-7 label: xpbd_box_stack, task_t12
_T12 = r"\btask_t12\b"
_TRIAGE = r"\btriage\b"             # "SB-7 triage (...) -> BUL-0014": WP7's replay of a campaign's S1 episodes


# Embodiment classes (coordinator ruling 2026-11-05 21:04Z, end of scr/WP13.md): the source scenes ask for soft contact
# (solref 20 ms on every geom, 5 ms in the _stiff variants), so their impacts reach about v·τ/e by design and measure
# the scene, not Alloy (WP13: 26 of 26 traced S1, [internal note]). SB-7 counts each embodiment's
# `emb_*_route` variant instead (every geom's solref at 2x the route step, 1.25 ms; valid at Alloy's step only). The
# source rows stay as reported rows, not counted; the headline reads fail until every counted class passes.
# Ruling 2026-11-05 22:33Z: a class whose S1s MuJoCo C float64 reproduces on the same model is reported, not counted
# (the trace is the evidence; 10 sampled S1 per class when the signature is uniform). `MC_REPRODUCED` lists them.
EMB_RULING = "coordinator 2026-11-05 21:04Z"
MC_RULING = "coordinator 2026-11-05 22:33Z"
MC_REPRODUCED: dict[str, str] = {
    "emb_e7_h1_2_route": "fails on the source hand (Inspire mimic links, armature 0), reproduced in MuJoCo C",
}
SOURCE_CLASSES: tuple[str, ...] = (
    "emb_e1_fr3", "emb_e2_panda", "emb_e3_ur10e_2f85", "emb_e3_ur10e_2f85_stiff", "emb_e4_fr3_bimanual",
    "emb_e4_fr3_bimanual_stiff", "emb_e5_fr3_duo", "emb_e5_fr3_duo_stiff", "emb_e6_g1", "emb_e6_g1_stiff",
    "emb_e7_h1_2", "emb_e7_h1_2_stiff", "emb_e8_go2", "emb_e8_go2_stiff")
_REPORTED_EMB = {c.split("_")[1] for c in MC_REPRODUCED}                  # e7: its route variant is reported too
ROUTE_EMBODIMENTS: tuple[str, ...] = tuple(e for e in dict.fromkeys(c.split("_")[1] for c in SOURCE_CLASSES)
                                           if e not in _REPORTED_EMB)    # the counted route classes: e1-e6, e8
_SOURCE = r"\b(?:" + "|".join(SOURCE_CLASSES) + r")\b"
_MC = r"\b(?:" + "|".join(MC_REPRODUCED) + r")\b"
_ROUTE = r"\bemb_\w+_route\b"


def source_tau_ms(cls: str) -> int:
    """The source scene's solref time constant: 5 ms on every geom of a _stiff variant, 20 ms otherwise (WP13)."""
    return 5 if cls.endswith("_stiff") else 20


def reported_aside(label: str, text: str) -> str:
    """The board's text on a reported embodiment row (a source scene, or a class MuJoCo C reproduces): why it is
    reported and not counted, and what SB-7 counts instead; a reproduced class also shows its reproduced-S1 count
    and rate (23:27Z)."""
    if (mc := re.search(_MC, label)) is not None:
        repro = _REPRO.search(text)
        count = f": {repro.group(1)} S1 (rate {repro.group(2)})" if repro else ""
        return f"{MC_REPRODUCED[mc.group(0)]}{count}; reported, not counted ({MC_RULING})"
    m = re.search(_SOURCE, label)
    if m is None:
        return ""
    emb = m.group(0).split("_")[1]
    instead = (f"its route variant is reported too ({MC_RULING})" if emb in _REPORTED_EMB
               else f"SB-7 counts the emb_{emb} route variant (solref 1.25 ms)")
    return (f"fail: source contact parameters (τ = {source_tau_ms(m.group(0))} ms); reported, not counted as an "
            f"Alloy failure ({EMB_RULING}): {instead}")


# Ruling 2026-11-05 23:27Z (end of scr/WP7.md): SB-7 measures the failures Alloy adds over MuJoCo C float64 on the same
# model and step. In a counted class an S1 that MuJoCo C reproduces is excluded from N and listed; the class still
# needs 300k clean episodes and 0 counted S1s. Each class row shows its reproduced-S1 count and rate, read from WP7's
# value cell ("reproduced in MuJoCo C: <k> S1 (rate <r>)"); a row with S1s that states none says so.
REPRO_RULING = "coordinator 2026-11-05 23:27Z"
_REPRO = re.compile(r"reproduced in MuJoCo C(?: float64)?: ([\d,]+) S1 \(rate ([^)]+)\)", re.IGNORECASE)
_NO_S1 = re.compile(r"^\**0 S1 in ")


def repro_aside(label: str, text: str) -> str:
    """A counted class row's reproduced-S1 count and rate (MuJoCo C float64, same model and step); "" for a row that
    names no scene class (e.g. WP5's T12 overflow check)."""
    if re.search(_CLASS, label) is None:
        return ""
    m = _REPRO.search(text)
    if m is not None:
        return f"reproduced in MuJoCo C float64: {m.group(1)} S1 (rate {m.group(2)}), listed, not counted"
    if _NO_S1.match(text.strip()):
        return "reproduced in MuJoCo C float64: 0 S1 (no S1)"
    return "reproduced-S1 count and rate: not stated in the row"


def t12_aside(label: str, text: str) -> str:
    return "; ".join(x for x in (repro_aside(label, text), s2_aside(label, text)) if x)


def s2_aside(_label: str, text: str) -> str:
    """The board's label for the S2 count in a task_t12 campaign's value cell ("" when the cell states none)."""
    m = _S2.search(text)
    if m is None:
        return ""
    return (f"S2 {m.group(1)} episodes ({m.group(2)} %) = {ARRIVAL}: {DEPTHS}; not resting sink-in, which SB-3 "
            f"measures ({RULING})")


FEEDS: tuple[Feed, ...] = (
    Feed("SB-7", _BOUND, _UNIT, newest=_CLASS, exclude=f"{_TRIAGE}|{_T12}|{_SOURCE}|{_ROUTE}", aside=repro_aside),
    Feed("SB-7", _BOUND, _UNIT, label=_T12, exclude=_TRIAGE, newest=_CLASS, aside=t12_aside),
    *(Feed("SB-7", _BOUND, _UNIT, label=rf"\bemb_{e}_\w*route\b", exclude=_TRIAGE, newest=_CLASS, aside=repro_aside,
           required=f"emb_{e} route variant (emb_{e}_*_route)") for e in ROUTE_EMBODIMENTS),
    Feed("SB-7", _BOUND, _UNIT, label=f"{_SOURCE}|{_MC}", exclude=_TRIAGE, newest=_CLASS, decides=False,
         aside=reported_aside),
    Feed("SB-7", _BOUND, _UNIT, label=_TRIAGE, decides=False),
)


@dataclass(frozen=True)
class PopulationChange:
    """One change to what SB-7 counts (audit AB-laws-7): the S1 thresholds did not move, what is counted did. Every
    published glitch rate carries this list or its count (`population_line`). WP7 and WP8 reviewed their rows
    2026-11-10."""

    date: str        # UTC day of the commit
    commit: str      # on main
    kind: str        # "counted": which episodes / classes count; "rule": what trips S1; "board": which row decides;
                     # "scorer": the cross-engine scorer (bench/xbench/ours), not the campaign rate on the board
    widens: bool     # True: more is counted; False: less
    what: str
    removed: str     # what it removed (or added) when made, as far as a number was recorded
    ruling: str      # who ruled, and why


_NONE = "not recorded"
POPULATION: tuple[PopulationChange, ...] = (
    PopulationChange("2026-11-01", "a51fa6d36, ed6031dac", "rule", True, "capacity_overflow is an S1 row, read every "
                     "substep (task_t12's contact capacities doubled)", "njmax 512 -> 1,024 moved 29 S1 to 25",
                     "WP7 (no coordinator ruling): an overflowed buffer drops contacts"),
    PopulationChange("2026-11-02", "a173d4220", "rule", False, "a body in contact-free flight no longer trips "
                     "impossible_speed at a flat 10 m/s: flight is held to a ballistic envelope, max(10 m/s, exit "
                     "speed + g t + 0.5 m/s); in contact, or one substep after, its own speed still counts",
                     "1 of 3 S1 on the 303,104-episode task_t12 campaign (100109:258, a free fall at 11.57 m/s) and 1 "
                     "of 1 on 20,480 routed episodes; the old rule's rate is still printed as rate_d12",
                     "SCR-WP7-7, accepted by WP0 with the envelope: T12 has no ground under the plywood, so a block "
                     "knocked off its edge passes 10 m/s by gravity alone"),
    PopulationChange("2026-11-02", "72c5ed18e", "rule", False, "the energy ledger credits the work robot shapes do on "
                     "a solver's particles (the driven liquid cup), so it is not read as energy gain", _NONE,
                     "SCR-WP9-6"),
    PopulationChange("2026-11-02", "97cfdec9b", "board", False, "per scene class only the newest campaign row "
                     "decides; earlier rows of the class, failures included, are shown beside as superseded",
                     "the headline moved from 6.87e-3 to 2.08e-5", "WP6's feed rule (no coordinator ruling)"),
    PopulationChange("2026-11-03", "6f4929b7e", "scorer", False, "an S1 tripped only by e_gain_cum or "
                     "impossible_speed is re-scored with robot touch from the pre- or the post-step frame and counts "
                     "only if it still trips (rival engines' replayed frames and Alloy-beside runs)",
                     "1 S1 in the 12 T12 rows of bench/xbench/OUR_TESTS.md (ManiSkill tuned: 23 raw, 22 confirmed, an "
                     "impossible_speed trip); the other 11 rows are equal",
                     "WP8 with WP7's one-scorer rule (no coordinator ruling): at a coarse step a fingertip reaching a "
                     "block during the step is not excused by pre-step contacts alone"),
    PopulationChange("2026-11-03", "4d010db43", "rule", False, "a soft contact at a cut front is held to its own "
                     "limit (2 x blade half-thickness + radius, growth capped), not to PEN_FAIL", _NONE,
                     "SCR-SC5-3 (coordinator 2026-11-03 19:05Z, 2026-11-04 11:30Z)"),
    PopulationChange("2026-11-05", "4d9c6db94", "rule", False, "the thin-part rule reads the bodies' solids along "
                     "the contact normal, so task_t12's 3.9-4.7 mm contacts are S2, not S1", _NONE,
                     "coordinator 2026-11-05 02:00Z"),
    PopulationChange("2026-11-05", "42283f1d8", "rule", True, "S1 rows of the cutting and cloth classes "
                     "(cut_pool_overflow, inverted_tets, cloth_pass_through, cloth_crossings)", "adds rows",
                     "coordinator 2026-11-05 13:55Z: SB-7 per scene class"),
    PopulationChange("2026-11-05", "52c82f4df", "board", False, "triage rows (replays of a campaign's S1 episodes) "
                     "never decide", "no S1: the campaign row keeps them", "WP6's feed rule (no coordinator ruling)"),
    PopulationChange("2026-11-05", "20a7132b3", "counted", False, "the 14 embodiment source scenes (contact solref "
                     "20 ms, 5 ms _stiff) are reported, not counted; their route variants (solref rewritten to 2x the "
                     "route step) count", "14 failing rows, upper bounds 1.67e-5 to 1.0",
                     f"{EMB_RULING}: the source scenes ask for soft contact, so impacts sink about v tau / e by "
                     "design, which measures the scene (26 of 26 traced S1)"),
    PopulationChange("2026-11-05", "db0e58259", "counted", False, "a class whose S1s MuJoCo C float64 reproduces on "
                     "the same model is reported, not counted: emb_e7_h1_2_route",
                     "153 S1 in 2,048 episodes (rate 7.47e-2); 21 of the 153 were traced and reproduced",
                     f"{MC_RULING}: the failures are the source hand's (mimic links, armature 0), in MuJoCo C too"),
    PopulationChange("2026-11-06", "ef7339732", "counted", False, "inside a counted class, an S1 that MuJoCo C "
                     "float64 reproduces leaves N and is listed", "1 S1 so far (emb_e5_fr3_duo_route, rate 3.49e-5)",
                     f"{REPRO_RULING}: SB-7 measures the failures Alloy adds over MuJoCo C on the same model and "
                     "step"),
    PopulationChange("2026-11-06", "86addd7f9", "scorer", False, "the scorer collides MJCF mesh geoms as convex "
                     "hulls, as MuJoCo does, not as triangle meshes", "1 S1 of 512 on the CPU route; rival rows "
                     "rescored from their saved trips on 2026-11-09 (25 rows): Genesis T12 tuned 15 -> 14, ManiSkill "
                     "T12 tuned 22 -> 21, five Isaac Lab embodiment tuned rows unchanged; rows over 50 S1 are rescored "
                     "as samples only; Isaac Lab's T12 rows, the Alloy-beside T12 rows in OUR_TESTS.md and the "
                     "Genesis / ManiSkill embodiment rows are still on the mesh path",
                     "WP8 on WP3's pair evidence (no coordinator ruling): a mesh-mesh "
                     "depth of 14 mm where MuJoCo's own distance is +1.08 mm"),
    PopulationChange("2026-11-06", "a10f7a04d", "rule", True, "a soft body's particle inside another soft body is "
                     "PEN_FAIL, an S1", "made cut_block_stroke's broken faces S1", "SCR-SOFT-3-1 (coordinator "
                     "2026-11-06 18:07Z)"),
    PopulationChange("2026-11-07", "224224c7b", "rule", False, "liquid and granular particles' soft contacts are "
                     "not judged for penetration", _NONE, "WP5 (no coordinator ruling): liquid at rest in a mesh cup "
                     "read PEN_FAIL"),
    PopulationChange("2026-11-09", "d0a527682", "rule", False, "a step in which the robot touches particles is "
                     "excused from the energy rows, like a rigid robot contact, unless that work is booked",
                     f"{_NONE}; WP15's CPU re-score of 10 cut_block_stroke worlds: S1 6 -> 2", "WP7 on WP15's "
                     "finding (no coordinator ruling): the ledger does not measure a blade's work on a soft solid, "
                     "so elastic recoil read as energy gain"),
    PopulationChange("2026-11-09", "d5d2c5874", "board", False, "a row its own results note marks superseded, with "
                     "the re-run pending, no longer decides: its class reads 'no current row' and the headline shows "
                     "no value", "cut_block_stroke's 1,372 S1 in 2,048 (bound 0.687) and the superseded emb_e2 / "
                     "emb_e8 route rows (1 S1 each)", "audit AB-inv-reports-12: no headline from a superseded row"),
)


def population_line() -> str:
    """The count to put beside any published glitch rate (the table: `population_md`, on the board)."""
    wider = sum(c.widens for c in POPULATION)
    return (f"SB-7's S1 thresholds are fixed, but what it counts changed {len(POPULATION)} times between "
            f"{POPULATION[0].date} and {POPULATION[-1].date} ({len(POPULATION) - wider} narrowed the count, {wider} "
            "widened it); the rate is the failures Alloy adds over MuJoCo C on scenes at the route's contact "
            "settings (table: 'SB-7 counted population' in bench/SCOREBOARD.md)")


def population_md() -> list[str]:
    out = ["", "## SB-7 counted population", "", population_line() + ".", "",
           "| date | commit | kind | what changed | removed (or added) when made | ruling and reason |",
           "|---|---|---|---|---|---|"]
    return out + [f"| {c.date} | {c.commit} | {c.kind}, {'widens' if c.widens else 'narrows'} | {c.what} | "
                  f"{c.removed} | {c.ruling} |" for c in POPULATION]


def note() -> str:
    clauses = (
        f"SB-7 measures the failures Alloy adds over MuJoCo C float64 on the same model and step ({REPRO_RULING}): in "
        "a counted class an S1 that MuJoCo C reproduces is excluded from N and listed, the class needs 300k clean "
        "episodes and 0 counted S1s, and each class row shows its reproduced-S1 count and rate",
        "bully random-legal rollouts per scene class; per class WP7's newest row decides (its note lists a class "
        "newest first), earlier rows of the class are shown beside as superseded",
        "a triage row (the replay of a campaign's S1 episodes) is shown beside: the campaign row it explains carries "
        "those S1",
        f"task_t12's S2 count is {ARRIVAL}, with WP3's depths beside it ({RULING})",
        "embodiment classes count on their emb_*_route variants (every geom's solref at 2x the route step, 1.25 ms, "
        f"valid at Alloy's route step only; {EMB_RULING}): the 14 source scenes (solref 20 ms, 5 ms _stiff) are "
        "reported beside as 'fail: source contact parameters', not counted, and the headline reads fail until every "
        "counted class passes",
        f"a class whose S1s MuJoCo C float64 reproduces on the same model is reported, not counted ({MC_RULING}): "
        + ", ".join(MC_REPRODUCED),
        population_line(),
    )
    return "; ".join(clauses)


__all__ = ["ARRIVAL", "DEPTHS", "EMB_RULING", "FEEDS", "ROUTE_EMBODIMENTS", "RULING", "SOURCE_CLASSES", "note",
           "s2_aside", "reported_aside", "source_tau_ms", "MC_REPRODUCED", "MC_RULING", "REPRO_RULING",
           "repro_aside", "t12_aside", "POPULATION", "PopulationChange", "population_line", "population_md"]
