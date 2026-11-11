# SPDX-License-Identifier: Apache-2.0
"""Precision pairs in a coupled scene under ``native="mujoco"`` (SCR-DC5-3, coordinator ruling 2026-11-03).

``wire`` accepts precision pairs with more than one dynamic entry only when the model's collision filters prove, at
assembly time, that no non-MuJoCo entry can contact any precision-pair shape (``check_isolated``). Otherwise it
refuses, naming the pair. A shape counts as reachable unless one of these filters (newton@ac7f7830) rules it out:

- shape-shape (the entry's shapes against a pair shape): COLLIDE_SHAPES on both
  (``geometry/broad_phase_common.py`` ``precompute_world_map``), compatible worlds and colliding groups
  (``test_world_and_group_pair`` / ``test_group_pair``), not in ``shape_collision_filter_pairs``, not one body
  (``is_shape_pair_same_body_filtered``), not both static (``is_shape_pair_immovable_filtered``);
- particle-shape (the entry's particles against a pair shape): COLLIDE_PARTICLES on the pair shape (soft contacts,
  cloth edge/face contacts, MPM's default colliders) and a particle world compatible with the shape's
  (``sim/collide.py`` ``_build_soft_particle_rigid_contact_pairs``). Alloy SPH entries also skip shapes marked
  ``sph:shape_boundary == IGNORE`` (``alloy/solvers/sph/layout.py`` ``shape_table``).

These are the same filters the Scheduler's proxy pipelines see: they collide a ``ModelView`` of the same Model.
Hydroelastic is a narrow-phase mode behind the same broad-phase filters (WP3). Not covered: colliders an MPM solver is
given later through ``setup_collider(collider_meshes=..., collider_body_ids=...)``, which bypass the shape flags
(Alloy's MPM factory never does), and WP3's ``ColliderField`` once it lands (it exposes every rigid collider).

A contact-free destination (``CONTACT_FREE_DESTINATIONS``: SolverDFSPH, SolverImplicitMPM) never adds contact rows and
never reads the fallback buffer: it sees a shape only as a boundary, and its load reaches the shape's body as a proxy
wrench that MuJoCo applies before it solves, so the precision rows see that load (SCR-JOB2-1, ruling 2026-11-04). Its
particles reaching a pair shape that sits on a body it receives as a proxy from the MuJoCo entry are reported as
"boundary only (proxy wrench)", and ``check_isolated`` accepts them when the pair's GapMonitor is on and fails closed
(``gap_fail_closed``); it returns them for the plan report. Every other reach still refuses (VBD, film, rigid).

When the check passes, ``CoupledPrecisionContacts`` runs two streams. The precision layer keeps its own buffer, and
WP3's binding appends it into the SolverMuJoCo entry (D10); the pair table writes its friction scale there
(``precision_external=True``, as on the single-entry path), which the append reads. WP2's Newton fallback buffer
(``NewtonContacts``, the one coupled scenes without pairs get) is the solver-facing ``contacts()`` for the non-MuJoCo
entries, with the table applied as on the fallback path (``precision_external=False``); MuJoCo runs its own contacts
and ignores it. The layer requires every pair in ``shape_collision_filter_pairs``, so the fallback buffer holds no
precision rows. It does hold Newton rows between a pair shape and other shapes (geometry only; MuJoCo collides those
natively). Health's rigid-contact check reads the fallback buffer, and the precision rows are checked by the
GapMonitor (exact geometry, per-pair limits, D11).

Lagged coupling restarts the MuJoCo entry from the exchange's start state every iteration (``SchedulerPolicy.
lagged_iterations`` = 2), while the binding appends once per ``arm``. ``rearm_each_step`` arms it before every
MuJoCo step with the rows posed at the exchange start, which is exact for those restarts. A MuJoCo entry that
substeps (rung > 1) would get rows posed at the exchange start in later substeps, so ``check_isolated`` refuses it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.integration.stepper import CONTACT_FREE_DESTINATIONS, proxy_bodies
from alloy.scene.types import BuiltScene, EntrySpec

log = logging.getLogger("alloy.integration")

__all__ = ["BOUNDARY_ONLY", "CoupledPrecisionContacts", "Reach", "check_isolated", "describe_reach", "find_reach",
           "rearm_each_step"]

COLLIDE_SHAPES = int(newton.ShapeFlags.COLLIDE_SHAPES)
COLLIDE_PARTICLES = int(newton.ShapeFlags.COLLIDE_PARTICLES)
SHOWN = 3                     # reaches listed in the refusal
BOUNDARY_ONLY = "boundary only (proxy wrench)"
PARTICLES = "its particles (COLLIDE_PARTICLES)"


@dataclass(frozen=True)
class Reach:
    """One way a non-MuJoCo entry can contact a precision-pair shape."""

    pair: int                 # index into built.precision_pairs
    shapes: tuple[int, int]   # the pair's global shape ids in ``world``
    shape: int                # the reached one
    world: int
    entry: str
    capability: str
    channel: str              # "shape <label>" (the entry's shape) or "particles"


@dataclass(frozen=True)
class _Shapes:
    flags: np.ndarray
    group: np.ndarray
    world: np.ndarray
    body: np.ndarray
    filt: np.ndarray          # sorted keys min * S + max of shape_collision_filter_pairs
    sph_ignore: np.ndarray | None
    labels: Sequence[str] | None


def _shapes(model: Any) -> _Shapes:
    S = int(model.shape_count)
    pairs = np.asarray(sorted(tuple(sorted((int(a), int(b)))) for a, b in model.shape_collision_filter_pairs),
                       dtype=np.int64).reshape(-1, 2)
    sb = getattr(getattr(model, "sph", None), "shape_boundary", None)
    ignore = None
    if sb is not None:
        from alloy.solvers.sph.config import BOUNDARY_IGNORE  # noqa: PLC0415 - WP9

        ignore = sb.numpy() == BOUNDARY_IGNORE
    return _Shapes(model.shape_flags.numpy().astype(np.int64), model.shape_collision_group.numpy().astype(np.int64),
                   model.shape_world.numpy().astype(np.int64), model.shape_body.numpy().astype(np.int64),
                   np.sort(pairs[:, 0] * S + pairs[:, 1]), ignore, getattr(model, "shape_label", None))


def _groups_collide(a: np.ndarray, b: int) -> np.ndarray:
    """``test_group_pair`` (0: none; positive: own group and negatives; negative: all but itself)."""
    if b == 0:
        return np.zeros(a.shape, bool)
    return (a != 0) & np.where(a > 0, (a == b) | (b < 0), a != b)


def _shape_reach(t: _Shapes, own: np.ndarray, p: int) -> np.ndarray:
    """The entry's shapes (``own``) that Newton would collide with pair shape ``p``."""
    if not t.flags[p] & COLLIDE_SHAPES or own.size == 0:
        return own[:0]
    own = own[(t.flags[own] & COLLIDE_SHAPES) != 0]
    w, body = t.world[p], t.body[p]
    ok = (t.world[own] < 0) | (w < 0) | (t.world[own] == w)
    ok &= _groups_collide(t.group[own], int(t.group[p]))
    ok &= ~((t.body[own] >= 0) & (t.body[own] == body))
    ok &= ~((t.body[own] < 0) & (body < 0))
    S = t.flags.size
    keys = np.minimum(own, p) * S + np.maximum(own, p)
    ok &= ~np.isin(keys, t.filt)
    return own[ok]


def _particle_reach(t: _Shapes, worlds: np.ndarray, p: int, capability: str) -> bool:
    """Whether particles in ``worlds`` (the entry's particle worlds) can contact pair shape ``p``."""
    if worlds.size == 0 or not t.flags[p] & COLLIDE_PARTICLES:
        return False
    if capability == "sph" and t.sph_ignore is not None and t.sph_ignore[p]:
        return False
    w = t.world[p]
    return bool(w < 0 or np.any(worlds < 0) or np.any(worlds == w))


def _label(t: _Shapes, s: int) -> str:
    return str(t.labels[s]) if t.labels is not None and s < len(t.labels) else f"shape {s}"


def _channels(t: _Shapes, own: np.ndarray, owned: set[int], pworlds: np.ndarray, p: int, capability: str,
              proxied: set[int]) -> list[str]:
    """How an entry (its shapes ``own``, the shapes on its bodies ``owned``, its particle worlds, the bodies it gets
    as MuJoCo proxies) reaches ``p``."""
    if p in owned:
        return ["ownership (the shape is on the entry's body)"]
    out = [f"its shape {_label(t, int(s))}" for s in _shape_reach(t, own, p)[:1]]
    if _particle_reach(t, pworlds, p, capability):
        boundary = capability in CONTACT_FREE_DESTINATIONS and int(t.body[p]) in proxied
        out.append(BOUNDARY_ONLY if boundary else PARTICLES)
    return out


def _entry_reach(model: Any, t: _Shapes, e: EntrySpec, wpairs: np.ndarray, proxied: set[int]) -> list[Reach]:
    """Every (pair, world) shape the non-MuJoCo entry ``e`` can contact, or owns."""
    own = np.asarray(e.shapes, dtype=np.int64)
    on_bodies = np.nonzero(np.isin(t.body, np.asarray(e.bodies, np.int64)))[0]
    owned = {int(s) for s in e.shapes} | {int(s) for s in on_bodies}
    parts = np.asarray(e.particles, dtype=np.int64)
    pworlds = np.unique(model.particle_world.numpy()[parts]) if parts.size else np.zeros(0, np.int64)
    out: list[Reach] = []
    for i, w in np.ndindex(wpairs.shape[0], wpairs.shape[1]):
        ab = (int(wpairs[i, w, 0]), int(wpairs[i, w, 1]))
        for p in ab:
            out.extend(Reach(int(i), ab, p, int(t.world[p]), e.name, e.capability, ch)
                       for ch in _channels(t, own, owned, pworlds, p, e.capability, proxied))
    return out


def find_reach(built: BuiltScene, entries: Sequence[EntrySpec]) -> list[Reach]:
    """Every way a non-MuJoCo entry of ``entries`` can contact a shape of ``built.precision_pairs``."""
    from alloy.contact.pipeline import world_shape_pairs  # noqa: PLC0415 - WP3's [P, W, 2] layout

    model = built.model
    wpairs = world_shape_pairs(model, tuple(built.precision_pairs))
    t = _shapes(model)
    proxied = proxy_bodies(built)
    out: list[Reach] = []
    for e in entries:
        if e.capability != "mujoco":
            out.extend(_entry_reach(model, t, e, wpairs, proxied.get(e.name, set())))
    return out


def _describe(t: _Shapes, reach: Sequence[Reach]) -> list[str]:
    """One line per (pair, entry, reached shape, channel), with the number of worlds."""
    worlds: dict[tuple, set[int]] = {}
    for r in reach:
        a, b = (_label(t, s) for s in r.shapes)
        worlds.setdefault((r.pair, a, b, r.entry, r.capability, _label(t, r.shape), r.channel), set()).add(r.world)
    return [f"precision pair {i} ({a} / {b}): entry {e!r} ({cap}) can contact {s} through {ch} in {len(w)} world(s)"
            for (i, a, b, e, cap, s, ch), w in worlds.items()]


def describe_reach(built: BuiltScene, reach: Sequence[Reach]) -> list[str]:
    """One line per (pair, entry, reached shape, channel), with the number of worlds (plan report, refusals)."""
    return _describe(_shapes(built.model), reach)


def check_isolated(built: BuiltScene, entries: Sequence[EntrySpec], *, gap_fail_closed: bool = False) -> list[Reach]:
    """Raise ``AssemblyError`` unless exactly one entry is MuJoCo and no other entry can contact a precision-pair
    shape, except boundary-only reaches when ``gap_fail_closed`` (module docstring). The message names the offending
    pair(s). Returns the accepted boundary-only reaches."""
    names = [e.name for e in entries]
    mj = [e for e in entries if e.capability == "mujoco"]
    head = f"precision pairs under native_contacts='mujoco' in a scene with entries {names}"
    if len(mj) != 1:
        raise AssemblyError(f"{head}: the precision stream is appended into exactly one SolverMuJoCo entry, found "
                            f"{[e.name for e in mj]}; run this scene with native_contacts='newton'")
    rung = max(int(mj[0].substeps), int(mj[0].max_substeps))
    if rung > 1:
        raise AssemblyError(f"{head}: MuJoCo entry {mj[0].name!r} substeps (rung {rung}), and its later substeps would "
                            "get precision rows posed at the exchange start; give it one step per exchange or run "
                            "this scene with native_contacts='newton'")
    reach = find_reach(built, entries)
    boundary = [r for r in reach if r.channel == BOUNDARY_ONLY]
    refused = [r for r in reach if r.channel != BOUNDARY_ONLY]
    if refused:
        _refuse(built, head, refused, "Keep non-MuJoCo entries off precision-pair shapes (collision group, "
                "shape_collision_filter_pairs, COLLIDE_PARTICLES off) or run this scene with native_contacts='newton'")
    if boundary and not gap_fail_closed:
        _refuse(built, head, boundary, "A boundary-only reach needs the pair's GapMonitor on and failing closed "
                "(profile gap_monitor not 'never', health on)")
    for line in describe_reach(built, boundary):
        log.info("precision pair reach accepted (SCR-JOB2-1): %s", line)
    log.info("precision pairs isolated from non-MuJoCo entries %s (SCR-DC5-3)", [n for n in names if n != mj[0].name])
    return boundary


def _refuse(built: BuiltScene, head: str, reach: Sequence[Reach], advice: str) -> None:
    lines = describe_reach(built, reach)
    shown = "; ".join(lines[:SHOWN])
    more = f" (+{len(lines) - SHOWN} more)" if len(lines) > SHOWN else ""
    raise AssemblyError(f"{head}: the precision stream reaches only SolverMuJoCo, and {shown}{more}. {advice}")


class CoupledPrecisionContacts:
    """``ContactLayerLike`` for a coupled scene with precision pairs under ``native="mujoco"`` (module docstring):
    ``contacts()`` is the Newton fallback buffer; the precision layer (``layer``) collides into its own buffer, gets
    the pair table's friction scale, and its MuJoCo binding appends it. Other attributes (``pairs``, ``overflow``,
    ``monitor``, ...) are the layer's."""

    precision_external = False    # the caller's pair-table pass over contacts(): the fallback path's setting

    def __init__(self, layer: Any, model: Any, pair_table: Any, *, bulk: Any = None) -> None:
        """``bulk``: the fallback buffer (wiring passes it with the profile's budget, SCR-WP4-6/7); default Newton's
        auto-sized rigid + soft buffer."""
        from alloy.scene.steppers import NewtonContacts  # noqa: PLC0415 - WP2's fallback path

        self.layer, self.pair_table = layer, pair_table
        self.bulk = bulk if bulk is not None else NewtonContacts(model)
        self.armed_state: newton.State | None = None    # the exchange's start state (rows posed there)

    def contacts(self) -> newton.Contacts:
        return self.bulk.contacts()

    def sidecar(self) -> Any:
        """The precision stream's sidecar (rows of ``layer.contacts()``, not of ``contacts()``)."""
        return self.layer.sidecar()

    def collide(self, state: newton.State, contacts: newton.Contacts, *, dt: float) -> None:
        """Precision stream (with the pair table's friction scale, which the MuJoCo append reads), then the
        fallback buffer ``contacts``; the caller applies the pair table to ``contacts``."""
        from alloy.scene.pairs import apply_pair_table  # noqa: PLC0415 - WP2

        prec = self.layer.contacts()
        self.armed_state = state
        self.layer.collide(state, prec, dt=dt)
        if self.pair_table is not None and prec.rigid_contact_friction is not None:
            apply_pair_table(self.pair_table, prec, state, precision_external=True)
        self.bulk.collide(state, contacts, dt=dt)

    def __getattr__(self, name: str) -> Any:
        if name in ("layer", "bulk", "pair_table", "armed_state"):
            raise AttributeError(name)
        return getattr(self.layer, name)


def rearm_each_step(solver: Any, contacts: CoupledPrecisionContacts, binding: Any) -> None:
    """Arm ``binding`` before every step of the MuJoCo entry ``solver`` with the rows posed at the exchange start
    (``contacts.armed_state``), so each lagged-coupling iteration gets the precision rows (module docstring)."""
    inner = solver.step

    def step(state_in: Any, state_out: Any, control: Any, step_contacts: Any, dt: float) -> Any:
        if contacts.armed_state is not None:
            binding.arm(contacts.armed_state)
        return inner(state_in, state_out, control, step_contacts, dt)

    solver.step = step
