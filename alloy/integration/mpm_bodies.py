# SPDX-License-Identifier: Apache-2.0
"""MPM entries carry their proxy bodies to the step's end (speed-now-1109 S5, robot + sand).

Newton's ``SolverImplicitMPM`` reads its colliders' bodies from ``state_in`` (``body_q``, and ``body_qd`` in the
default "forward" collider velocity mode) and never writes ``state_out.body_q`` / ``body_qd``. A coupled MPM entry
that is the destination of a body proxy pair (a robot or a free body beside sand or paste) therefore ends every step
with stale proxy bodies in its output state: the coupler's lagged residual compares the source's end pose with them.
On a Panda swinging at about 1 m/s far from a sand tray, the gap read 6.6, 18 and 30 mm in the first three periods
(the arm's travel since the buffers last met), so the watchdog's kinematic rule (``res_pos_escalate``) tripped,
re-ran those periods and then raised the clock floor for the rest of the run: 16 exchanges per 10 ms instead of 8.
SPH's solver copies its bodies through for the same reason
(``alloy/solvers/sph/solver.py``, ``step``).

By default (``ALLOY_MPM_BODY_CARRY=copy``) each such entry's step ends by copying its bodies' input poses and
velocities into ``state_out``, as SPH does: the lagged residual then reads the source's travel over one exchange
(``v dt``, 1.3 mm at 1 m/s and 1.25 ms, under the 10 mm escalation gap), and the power bond books no destination-side
proxy motion. ``advance`` moves every body over the step as the forward-mode MPM collider did (centre of mass by
``v dt``, orientation by ``ω dt``, velocities kept): the gap is then the step's curvature (um) and the bond books the
collider's motion, so the watchdog sees a lagged pair's true injection. On a heavy body resting on sand
(tests/integration/test_mpm_contact_free.py's cup) that injection (0.3-0.4 mJ per 10 ms) escalated the clock to
0.16 ms, so ``advance`` stays opt-in. A step in place (``state_in is state_out``) is left as is, so a restart pass that
steps again from the same input never carries a body twice. ``off`` restores the stale output (before 2026-11-09).

Robot + sand (a Panda plunging and dragging its fingers through a 4 cm dry-sand tray, L4, CUDA graphs, per 10 ms period
in contact): 1 world 1.336 -> 0.365 s (3.7x), 64 worlds 18.84 -> 13.08 s (1.44x); the clock at 1.25 ms throughout,
no re-run (measured with ``advance``; ``copy`` takes the same clock and passes). Laws: tests/integration/test_mpm_bodies.py.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import warp as wp

from alloy.core.errors import SolverOptionError

log = logging.getLogger("alloy.integration")

__all__ = ["ENV", "MODES", "install", "install_for_pairs", "mode"]

ENV = "ALLOY_MPM_BODY_CARRY"
MODES = ("off", "copy", "advance")
DEFAULT = "copy"


def mode() -> str:
    """The carry mode from ``ALLOY_MPM_BODY_CARRY`` (unset: ``DEFAULT``); an unknown value is refused."""
    raw = os.environ.get(ENV, "").strip().lower() or DEFAULT
    if raw not in MODES:
        raise SolverOptionError(f"{ENV}={raw!r}: the MPM body carry must be one of {', '.join(MODES)}. Fix: unset "
                                f"it (the default, copy), or set {ENV}=advance or {ENV}=off")
    return raw


@wp.kernel(enable_backward=False)
def _advance_kernel(q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3], dt: float,
                    q_out: wp.array[wp.transform], qd_out: wp.array[wp.spatial_vector]):
    i = wp.tid()
    x = q[i]
    v = qd[i]
    lin = wp.spatial_top(v)
    ang = wp.spatial_bottom(v)
    r = wp.transform_get_rotation(x)
    c = wp.transform_point(x, com[i])
    w = wp.length(ang)
    r1 = r
    if w * dt > 1.0e-12:
        r1 = wp.normalize(wp.quat_from_axis_angle(ang / w, w * dt) * r)
    c1 = c + lin * dt
    q_out[i] = wp.transform(c1 - wp.quat_rotate(r1, com[i]), r1)
    qd_out[i] = v


def install(solver: Any, how: str = "advance") -> bool:
    """Wrap ``solver.step`` to carry its bodies into ``state_out`` (``how``: "copy" or "advance", module doc); False
    when already wrapped or the solver's model has no bodies."""
    if getattr(solver, "_alloy_body_carry", False):
        return False
    model = solver.model
    n = int(getattr(model, "body_count", 0) or 0)
    if n == 0 or getattr(model, "body_com", None) is None:
        return False
    prev = solver.step

    def step(state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float, **kw: Any) -> None:
        prev(state_in, state_out, control, contacts, dt, **kw)
        if state_out is state_in or state_in.body_q is None or state_out.body_q is None:
            return
        if how == "copy":
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
            return
        wp.launch(_advance_kernel, dim=n, inputs=[state_in.body_q, state_in.body_qd, model.body_com, float(dt)],
                  outputs=[state_out.body_q, state_out.body_qd], device=model.device)

    solver.step = step
    solver._alloy_body_carry = True
    return True


def install_for_pairs(built: Any, solvers: Mapping[str, Any]) -> list[str]:
    """Unless ``mode()`` is "off": ``install`` (copy, the default, or advance) on every MPM entry that is the
    destination of a two-way pair; one note each. Off: nothing (the stale output)."""
    how = mode()
    if how == "off":
        return []
    cap = {e.name: e.capability for e in built.entries}
    notes = []
    for pair in getattr(built.coupler, "pairs", ()) or ():
        solver = solvers.get(pair.destination)
        if pair.one_way or cap.get(pair.destination) != "mpm" or solver is None or not install(solver, how):
            continue
        line = f"mpm entry {pair.destination}: carries its proxy bodies to the step's end ({ENV}={how})"
        log.info("%s", line)
        notes.append(line)
    return notes
