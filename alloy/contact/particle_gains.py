# SPDX-License-Identifier: Apache-2.0
"""Contact gains a particle solver reads for MuJoCo-owned card bodies (C3 box_on_pad, XB4-5, 2026-11-08).

**What was wrong: units.** ``alloy.scene.native_gains.apply_native_gains`` writes SolverMuJoCo's native gains into
the template's ``shape_material_ke`` / ``shape_material_kd`` of every shape on a MuJoCo-owned dynamic card body. These
are acceleration-space values: ke = K·(1−imp)/m [1/s²] and kd = 2ζ·√ke [1/s], so that MuJoCo's ``convert_solref``
realises the card's force-space K on that body.

A particle destination reads the same arrays as force-space values. VBD mixes them per body-particle record with the
soft globals, ``avg_ke = (soft_contact_ke + shape ke) / 2`` and ``avg_kd`` likewise (newton
vbd/rigid_vbd_kernels.py ``_average_contact_material``), and applies ``avg_kd`` in N·s/m. A static shape of the same
material carries the card's own values instead: ke = K, kd = 0 (``emitters.common.shape_config``). For C3's platen,
kd = 2·0.7·√6.75e4 = 364 (ζ 0.7 is the multi-point floor), so each record got 187 N·s/m.

**What was wrong: many records under one body.** VBD's lagged feedback to the body
(``coupling_harvest_proxy_wrenches``) is the sum of the records' force at VBD's last iterate. That is not the impulse
the particles took, and the error grows with the records' summed stiffness and damping against the body's mass and
the exchange interval:
- The damping part is one-sided (it acts only while the particle approaches).
- Every record touching the body carries a full material K (2e6 N/m, D22's default), while the exchange clock bounds
  by one record (``_pair_stiffness_mass``).

C3's platen lists 1,457 records on 226 pad particles; its ~265 touching ones carry about 5e8 N/m together. Two
failures followed:
- Card values with Newton's soft mix (kd 5). Mid-air, the platen lost 50x the pad's impulse per step. At rest, it
  hovered 0.2 s before sinking.
- The material's ke on the Warp CPU route (the suite's; MuJoCo Warp steps the platen). The pad's first rebound
  injected energy: the health check latched ENERGY_JUMP at 18 ms, and without the latch, at kd 0 the platen left at
  3.4 mm with the pad off the ground. The MuJoCo C route stayed bounded.

A heavy body on a coarse cloth touches only a few records, so it needs the opposite: stiff, damped records.

**What this does.** ``force_gains(built)`` names the shapes ``apply_native_gains`` rewrote (same rule: a card object,
not a robot, on a dynamic body of the MuJoCo entry). For each it gives the material's force-space K, the pair table's
own-material stiffness, which is what a static shape of that material carries.

``particle_view`` is a ``SolverCoupled.Entry.configure_view`` that writes K and the card's kd 0 into one destination's
view only, so no acceleration-space value is left there.

``attach(solver, gains, bound)`` then sizes the records on those shapes per body each time VBD rebuilds its record
materials (``_refresh_body_particle_contact_state``, the only writer of the per-record arrays that the solve and the
feedback both read). The body's n records share a spring and a damper held to the exchange clock's own bounds for
that body's mass m at the exchange interval dt (``bound`` = safety · omega_dt_max, 0.5 by default; ``clocks.py``):
- summed stiffness: ω·dt ≤ bound, so each record gets ke = min(VBD's mix, bound² · m / (dt² · n));
- summed damping: c·dt/m ≤ bound, so each record gets kd = bound · m / (dt · n).

Here n counts every record VBD lists for the body (``body_particle_contact_counts``, inside the contact gap), so the
budget is spread over at least the touching ones. C3's platen gets ~3.6e3 N/m and 1.1 N·s/m per record. A 0.5 kg
ball on four records of a coarse cloth gets the material's ke and ~400 N·s/m. ``soft_contact_kd`` and
``soft_contact_ke`` stay as they are: they also set cloth self-contact and records on static shapes.

**Measurements.** All on CPU; probes in [internal note]. C3: a 0.515 kg platen on a 40 x 40 x 10 mm
Neo-Hookean pad of 900 particles, 7-9 mg each. The settle runs use ``alloy.Scene`` with tissue_fat_silicone (E 30 kPa,
ν 0.45), whose analytic stretch is 0.9029.

| case | native gains (main) | fixed 1e4 N/m, 1 N·s/m | per-body (this) |
|---|---|---|---|
| C3, platen at rest | unloaded, 0.998 | 0.9035 (Warp), 0.9038 (C) at 0.8 s | 0.9035-0.9040 at 0.74-0.8 s |
| C3, 1 mm drop | thrown off, 1.9 m/s | rings, 0.905-0.911 at 0.44-0.5 s | rings, 0.905-0.911 at 0.44-0.5 s |
| 0.5 kg ball on a 3 cm cloth | rests 0.04 mm in | hovers 0.65-0.97 mm off at 0.15-0.3 s | rests 0.05 mm in by 0.1 s |
| 2 kg ball, same cloth | | sinks 0.2-0.9 mm, moving at 0.3 s | rests 0.10 mm in by 0.15 s |
| 2 kg ball hits that cloth in mid-air, 0.31 m/s | impulse 0.08 % over | | impulse 0.15 % over, no launch |
| 2 kg ball lands on it on the ground, 0.34 m/s | rebounds, e 0.71 | | rebounds, e 0.58-0.69 (cloth on ground) |

Health stays OK in every case of the last column (Warp route; C3 also on the C route). Two other options were
tried:
- More VBD iterations do not fix the impact (40 at kd 5: 2-80x off per step).
- Capping a record's kd at m_particle / dt threw the platen off from rest.

**Not covered.** The feedback is still the last iterate's force, so landings are softer than the pair's
restitution. Robot shapes keep their importer's values (not measured). The exchange clock still bounds by one record.
All three are in known gap coupled-record-damping. A view whose shape ids are not the Model's keeps VBD's mix, with a
warning (``_global_shapes``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

log = logging.getLogger("alloy.contact")

ROBOT_KEY = "robot:"     # material key prefix of robots (SceneAssembler.plan): native gains skip them


@dataclass(frozen=True)
class ForceGains:
    """Model shape ids (every world) on MuJoCo-owned dynamic card bodies, with their card's force-space K."""

    shapes: np.ndarray      # int64
    ke: np.ndarray          # float32 [N/m]
    model_shapes: int       # the Model's shape count: a view must keep shape ids global (``_global_shapes``)


def own_material_k(stiffness: np.ndarray) -> np.ndarray:
    """Each material's own K from the pair table's series stiffness (D22: K·K/(K+K) = K/2 on the diagonal)."""
    return 2.0 * np.diag(np.asarray(stiffness, np.float64))


def force_gains(built: Any) -> ForceGains | None:
    """The shapes whose Model ke/kd hold MuJoCo native gains, with their card's force-space K; None if none."""
    m = built.model
    alloy = getattr(m, "alloy", None)
    tagged = getattr(alloy, "shape_material", None) if alloy is not None else None
    table = getattr(built, "pair_table", None)
    if tagged is None or table is None or not int(m.shape_count or 0) or not int(m.body_count or 0):
        return None
    mujoco = np.zeros(int(m.body_count), bool)
    for e in built.entries:
        if e.capability == "mujoco":
            mujoco[np.asarray(e.bodies, np.int64)] = True
    body, mat = m.shape_body.numpy(), tagged.numpy()
    dynamic = (m.body_mass.numpy() > 0.0) & ((m.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC)) == 0)
    card = np.array([not str(k).startswith(ROBOT_KEY) for k in table.materials] + [False], bool)   # [-1]: untagged
    b = np.where(body >= 0, body, 0)
    shapes = np.flatnonzero((body >= 0) & mujoco[b] & dynamic[b] & card[np.where(mat >= 0, mat, -1)])
    if not shapes.size:
        return None
    k = own_material_k(table.stiffness.numpy())[mat[shapes]]
    ke = np.where(np.isfinite(k) & (k > 0.0), k, float(m.soft_contact_ke)).astype(np.float32)
    return ForceGains(shapes.astype(np.int64), ke, int(m.shape_count))


def _global_shapes(gains: ForceGains, view: Any) -> np.ndarray | None:
    """``gains.shapes`` as ids of ``view``, or None. Newton's entry views keep shape ids global: compaction renumbers
    bodies, joints and particles, not shapes, and detaches the shapes an entry cannot see (pin d15a836c,
    solver_coupled.py: a compact view's ``shape_count`` is the Model's). A view with another shape count would put
    these gains on the wrong shapes, so it keeps VBD's own mix, with a warning (fails soft; WP4's review)."""
    n = int(view.shape_count or 0)
    if n == gains.model_shapes:
        return gains.shapes
    log.warning("particle view %r has %d shapes, the Model %d: shape ids are not global, so its records against "
                "MuJoCo card bodies keep VBD's mix (alloy.contact.particle_gains; known gap coupled-record-damping)",
                getattr(view, "name", "?"), n, gains.model_shapes)
    return None


def particle_view(gains: ForceGains, then: Callable[[Any], None] | None = None) -> Callable[[Any], None]:
    """A ``configure_view`` for one particle destination: on ``gains.shapes`` its view reads the card's K and kd 0;
    the Model and the other views keep theirs. ``then``: an earlier configure_view, run first."""

    def configure(view: Any) -> None:
        if then is not None:
            then(view)
        ids = _global_shapes(gains, view)
        if ids is None:
            return
        ke, kd = view.shape_material_ke.numpy().copy(), view.shape_material_kd.numpy().copy()
        ke[ids], kd[ids] = gains.ke, 0.0
        view.shape_material_ke = wp.array(ke, dtype=float, device=view.device)
        view.shape_material_kd = wp.array(kd, dtype=float, device=view.device)
        log.info("particle view %s: %d MuJoCo-body shapes read the card's K; their records sized per body (native "
                 "gains stay in MuJoCo)", getattr(view, "name", "?"), ids.size)

    return configure


@wp.kernel(enable_backward=False)
def _record_gains(count: wp.array[int], shape: wp.array[int], marked: wp.array[wp.int32],
                  shape_body: wp.array[wp.int32], records: wp.array[wp.int32], inv_mass: wp.array[float],
                  inv_dt: float, bound: float, kd: wp.array[float], penalty_k: wp.array[float],
                  material_ke: wp.array[float]):
    """Per-body share of the clock-bounded spring and damper on each marked record (module docstring)."""
    i = wp.tid()
    if i >= wp.min(count[0], kd.shape[0]):
        return
    s = shape[i]
    if s < 0 or s >= marked.shape[0]:
        return
    if marked[s] == 0:
        return
    b = shape_body[s]
    if b < 0 or inv_mass[b] <= 0.0:
        return
    share = (1.0 / inv_mass[b]) * inv_dt / float(wp.max(records[b], 1))       # m / (dt n)
    ke = wp.min(material_ke[i], bound * bound * share * inv_dt)
    material_ke[i] = ke
    penalty_k[i] = wp.min(penalty_k[i], ke)
    kd[i] = bound * share


def attach(solver: Any, gains: ForceGains, bound: float) -> bool:
    """Size ``solver``'s records against ``gains.shapes`` per body to the exchange clock's ``bound`` (module
    docstring); False when the solver is not VBD-family. The interval is the exchange's, from the coupler's input
    notification (each sub-step's when the solver is stepped alone). Capture-safe: one launch after each refresh, its
    interval a constant of the captured exchange (graphs are keyed by dt_x)."""
    inner = getattr(solver, "_refresh_body_particle_contact_state", None)
    needed = ("body_particle_contact_material_kd", "body_particle_contact_counts", "body_inv_mass_effective")
    if inner is None or not all(hasattr(solver, a) for a in needed):
        return False
    ids = _global_shapes(gains, solver.model)
    if ids is None:
        return False
    mark = np.zeros(gains.model_shapes, np.int32)
    mark[ids] = 1
    marked = wp.array(mark, dtype=wp.int32, device=solver.device)
    interval = {"exchange": 0.0, "step": 0.0}
    inner_step, inner_notify = solver.step, getattr(solver, "coupling_notify_input_state_update", None)

    def step(*args: Any, **kwargs: Any) -> Any:
        interval["step"] = float(kwargs["dt"] if "dt" in kwargs else args[4])
        return inner_step(*args, **kwargs)

    def notify(*args: Any, **kwargs: Any) -> Any:
        if float(kwargs.get("dt", 0.0) or 0.0) > 0.0:
            interval["exchange"] = float(kwargs["dt"])
        return inner_notify(*args, **kwargs)

    def refresh(contacts: Any, refresh: bool) -> None:
        inner(contacts, refresh)
        kd = solver.body_particle_contact_material_kd
        dt = interval["exchange"] or interval["step"]
        if contacts is None or not int(contacts.soft_contact_max) or kd.shape[0] == 0 or dt <= 0.0:
            return
        wp.launch(_record_gains, dim=min(int(contacts.soft_contact_max), kd.shape[0]), device=solver.device,
                  inputs=[contacts.soft_contact_count, contacts.soft_contact_shape, marked, solver.model.shape_body,
                          solver.body_particle_contact_counts, solver.body_inv_mass_effective, 1.0 / dt,
                          float(bound), kd, solver.body_particle_contact_penalty_k,
                          solver.body_particle_contact_material_ke])

    solver.step = step
    if inner_notify is not None:
        solver.coupling_notify_input_state_update = notify
    solver._refresh_body_particle_contact_state = refresh
    return True


__all__ = ["ForceGains", "attach", "force_gains", "own_material_k", "particle_view"]
