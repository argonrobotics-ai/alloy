# SPDX-License-Identifier: Apache-2.0
"""B2 agent audit brief (SPEC §3.11, bench §4.7; the Gym-Anything legal-action auditor role).

``audit_prompt`` writes the brief an auditing agent receives: the rules (legal actions only, in-range scene edits
only, never edit product code), the target and its action box, the §4.3 thresholds, what B0/B1 already found, the
attack catalogue and the submission format. It returns text; it runs nothing and edits nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alloy.bully.search import scene_hash
from alloy.bully.severity import MONITOR_ROWS
from alloy.bully.types import BullyLimits, BullyTarget, Finding

ATTACKS: tuple[str, ...] = (
    "a light part driven against a stiff wall at the maximum legal speed (contact stiffness vs mass ratio)",
    "a 1000:1 mass ratio across a coupling interface (heavy body resting on cloth, light body on a heavy one)",
    "a sleeping island woken by an impact at speed",
    "a reset in the middle of a contact (reset one world while its neighbours keep stepping)",
    "a cut or tear through a node or a pre-split line end (topology scenes)",
    "grasp, release and regrasp at the actuator rate limit (warm starts and contact matching)",
    "pushing an object into a corner so it is wedged between two supports (jamming)",
    "commands that sit exactly on a joint limit while a contact pushes back (limit + contact constraints)",
)


def _action_table(target: BullyTarget) -> list[str]:
    sp = target.actions
    rows = ["| actuator | low | high | max change per tick |", "|---|---|---|---|"]
    for a in range(sp.dim):
        rows.append(f"| {a} | {sp.low[a]:.6g} | {sp.high[a]:.6g} | {np.asarray(sp.max_delta_per_tick)[a]:.6g} |")
    return rows


def _threshold_table(limits: BullyLimits) -> list[str]:
    rows = ["| monitor | severity | threshold | condition |", "|---|---|---|---|"]
    for row in MONITOR_ROWS:
        level = row.severity.name if row.severity is not None else "diagnostic"
        rows.append(f"| {row.name} | {level} | {row.threshold(limits):.6g} | {row.condition} |")
    return rows


def _findings_table(findings: Sequence[Finding]) -> list[str]:
    if not findings:
        return ["None yet. B0 and B1 found nothing that replayed."]
    rows = ["| id | severity | monitor | value | threshold | first tick | repeat of |", "|---|---|---|---|---|---|---|"]
    for f in findings:
        rows.append(f"| {f.id} | {f.severity.name} | {f.monitor} | {f.value:.6g} | {f.threshold:.6g} | "
                    f"{f.first_step} | {f.repeat_of or '-'} |")
    return rows


def audit_prompt(target: BullyTarget, findings: Sequence[Finding], limits: BullyLimits | None = None) -> str:
    """The B2 brief for ``target`` given the findings so far (markdown)."""
    limits = limits or BullyLimits()
    sp = target.actions
    lines = [
        f"# Bully B2 audit: target `{target.name}`",
        "",
        "You are auditing a physics simulation for defects. Your job is to find legal inputs that make it do",
        "something physically impossible, and to hand each one over as a replayable trace.",
        "",
        "## Rules (binding)",
        "- Never edit product code: nothing under `alloy/`, Newton, MuJoCo Warp or Warp. Write only under your",
        "  output directory. A fix is not your job; a reproducible trace is.",
        f"- Actions must be legal: inside the box below, at {sp.rate_hz:g} Hz, changing by at most the listed amount",
        "  per tick. Commands outside it are clipped in code and the clips are logged; a clipped attack is void.",
        "- Scene edits only inside each card field's validated range (material values, masses, poses). Say which.",
        "- A finding counts only after replay trips the same monitor 10/10 on CPU or at least 9/10 on the same GPU",
        "  type. A failed precondition is inconclusive, not a finding (playbook VAL-03).",
        "- Read `bench/bully/FINDINGS.md` (lessons from earlier rounds) before you start. At most 3 rounds per pin.",
        "",
        "## Target",
        f"- scene hash `{scene_hash(target)}`; non-target bodies (must not move > {limits.rest_disp_max:g} m): "
        f"{list(target.non_target_bodies) or 'none'}; thin parts: {dict(target.thin_parts) or 'none'}",
        "",
        *_action_table(target),
        "",
        "## Monitors and thresholds (SPEC §4.3)",
        *_threshold_table(limits),
        "",
        "## Already found (do not resubmit; a new trace for the same cause is a REPEAT)",
        *_findings_table(findings),
        "",
        "## Attack ideas",
        *[f"- {a}" for a in ATTACKS],
        "",
        "## Submit",
        "For each attack write `<out>/attacks/<short_name>.npz` with `actions` (float32 [T, A], SI, one row per",
        "tick), `hypothesis` (one sentence: which code path you expect to fail and why) and `scene_edits` (a JSON",
        f"string, `{{}}` if none). Check it with `python -m alloy.bully try --target {target.name} --trace <file>`.",
    ]
    return "\n".join(lines) + "\n"
