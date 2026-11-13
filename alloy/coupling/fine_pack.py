# SPDX-License-Identifier: Apache-2.0
"""#10b P1, the fine pack ([internal note] notes/wp4-perworld-clock/DESIGN.md sections 10-11): slice S1, moving one world.

Today an escalated world re-runs its period with every other world at the finer clock (``IntegratedSimulation._roll_
back``). P1 re-runs the escalated worlds alone in a second simulation of the same scene with few worlds, the fine pack.
S1 is what that needs first: ``WorldTransfer`` copies world ``e`` of one simulation into slot ``s`` of another and back.
- The arrays are ``WorldSnapshot``'s: every State array, the stepper's ``state_arrays`` (its entry solvers' state
  adapters, the coupler's carry and Aitken, the watchdog), the health monitor's (record, inbox, latched, the D20
  energy reference: as ``_end_snapshot``) and the Control's (joint targets, joint_f, MuJoCo ctrl: the step reads
  them), with their item -> world maps (WP12's review). They are matched by name, each world's items paired in order.
- A simulation with step observers (``sim.drive``'s Drive, a pour's FluxEnergy) is refused: their clocks are not
  moved (WP12).
- A transfer is refused (``TransferError``; the caller re-runs as today) when an array has no per-world axis (checked
  first, on both sides), when the two sides do not have the same slot names, or when a world's item count differs
  from the slot's. VBD's per-contact-row history (``_prev_contact_*``) has no world map, so a pack that carries it is
  refused as an array without a per-world axis: its rows follow the pack's contact build, not the world (WP10).
- Per-world Model parameters (a randomized scene's masses, frictions, ...) move with the world (S1b:
  ``world_clock_params.WorldParams``, which also refreshes what the solvers and the coupler derive from them and
  names what it refuses). The way back uses ``params="same"``: the rows must already agree, nothing is written. The
  model's start state (body_q, joint_q, particle_q and their velocities) is not a parameter: only a reset reads it,
  and the fine pack refuses reset (WP12).
- The start checks (``health.start``: START_PEN level, spin peak and body, judged bodies) move too (WP12).
- Not moved here (S2): the pack counters (clock ticks, steps, ``health._step``), the host-side period indices
  (OnsetRefiner.last, PeriodLadder.last / per_world) and the contact gate's rows (DESIGN sections 11, 13).
Laws: tests/coupling/test_fine_pack.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import CouplingError
from alloy.core.snapshot import WorldSnapshot, _state_arrays, _state_item_world
from alloy.core.worldops import item_world_for, world_iota

_NO_STATE = SimpleNamespace()       # control_snapshot's State: none, its arrays are all extras
START = frozenset({"body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"})   # reset-only
# StepHealth fields of health.record that hold a global item index: re-pointed at the slot's items. The inbox's are
# never written (neutral_record leaves them 0) and are moved as they are.
HEALTH_INDEX = {"argmax_body": "body_world", "fail_body": "body_world", "fail_shape_a": "shape_world",
                "fail_shape_b": "shape_world", "fail_particle": "particle_world"}
_ELEMENTS = {"TRIANGLE": ("tri_indices", 0), "EDGE": ("edge_indices", 2), "TETRAHEDRON": ("tet_indices", 0),
             "SPRING": ("spring_indices", None)}     # an element's world: its first particle's (edge: its own vertex)


class TransferError(CouplingError):
    """A world cannot be moved between these two simulations (the caller falls back to the whole re-run)."""


def snapshot(sim: Any) -> WorldSnapshot:
    """The per-world snapshot of what a period reads besides Control: State, the stepper's arrays, the health
    monitor's (``health/*``) and their world maps."""
    from alloy.integration.simulation import _health_arrays  # noqa: PLC0415  (integration imports coupling)

    extra, maps = dict(sim.state_arrays()), dict(sim.state_world_maps())
    health = getattr(sim, "health", None)
    if health is not None:
        extra.update({f"health/{k}": v for k, v in _health_arrays(health).items()})
        extra.update(_start_arrays(sim.model, health, maps))
        extra.update(_gain_arrays(sim.model, health, maps))
    return WorldSnapshot(sim.model, sim.state, extra or None, extra_world=maps or None)


def _start_arrays(model: Any, health: Any, maps: dict[str, wp.array]) -> dict[str, wp.array]:
    """The start checks' per-world state (``health.start``, WP12): the sticky START_PEN level, the run's fastest spin
    and its body ([W]) and the bodies judged for spin ([body]); their maps go into ``maps``."""
    start, out = getattr(health, "start", None), {}
    for name in ("_start", "peak", "peak_body", "_judged"):
        arr = getattr(start, name, None)
        if arr is None:
            continue
        world = model.body_world if name == "_judged" else world_iota(int(model.world_count), model.device)
        if world is not None and world.shape[0] == arr.shape[0]:
            out[f"health/start.{name}"] = arr
            maps[f"health/start.{name}"] = world
    return out


def _gain_arrays(model: Any, health: Any, maps: dict[str, wp.array]) -> dict[str, wp.array]:
    """The energy-gain tier's per-world block state (``health.gain.state``, PHYSA-1: time, block, its maximum, the
    floor, ready, lowers; the flag and the peak are in ``state_arrays``): set from the world's own start energy, so a
    moved world takes its own, not the slot's (test_world_clock_params law iii)."""
    state, out = getattr(getattr(health, "gain", None), "state", None), {}
    world = world_iota(int(model.world_count), model.device)
    for name in ("t", "block", "block_max", "floor", "ready", "lowers"):
        arr = getattr(state, name, None)
        if arr is not None and arr.shape[0] == int(model.world_count):
            out[f"health/gain.{name}"], maps[f"health/gain.{name}"] = arr, world
    return out


def item_world(model: Any, freq: Any) -> wp.array | None:
    """``item_world_for`` plus the particle elements (triangles, edges, tets, springs: their first vertex's world)."""
    mapped = item_world_for(model, freq)
    name = getattr(freq, "name", None)
    if mapped is not None or name not in _ELEMENTS or model.particle_world is None:
        return mapped
    attr, col = _ELEMENTS[name]
    idx = getattr(model, attr, None)
    if idx is None or idx.size == 0:
        return None
    first = idx.numpy().reshape(-1, 2)[:, 0] if col is None else idx.numpy()[:, col]
    return wp.array(model.particle_world.numpy()[first].astype(np.int32), dtype=wp.int32, device=model.device)


def _control_world(model: Any, path: str, arr: wp.array) -> wp.array | None:
    """A Control array's item -> world map: the State rule (frequency, then name prefix), and for an articulation-scoped
    custom frequency (MuJoCo's actuators: ``mujoco:ctrl``) each row's articulation's world (owner -1: global)."""
    try:
        freq = model.get_attribute_frequency(path)
    except (KeyError, AttributeError):
        freq = None
    owners = model.custom_frequency_articulation.get(freq) if isinstance(freq, str) else None
    if owners is not None and owners.shape[0] == arr.shape[0] and model.articulation_world is not None:
        o, aw = owners.numpy(), model.articulation_world.numpy()
        world = np.where(o >= 0, aw[np.clip(o, 0, None)] if aw.size else -1, -1).astype(np.int32)
        return wp.array(world, dtype=wp.int32, device=model.device)
    mapped = item_world(model, freq) if freq is not None else None
    return mapped if mapped is not None and mapped.shape[0] == arr.shape[0] else _state_item_world(model, path, arr)


def control_snapshot(sim: Any) -> WorldSnapshot:
    """The per-world snapshot of the step's Control (joint targets, joint_f, MuJoCo ctrl), as extra arrays with
    explicit world maps (``_control_world``)."""
    model, arrays = sim.model, list(_state_arrays(sim.control))
    extra = {f"control:{path}": arr for _, path, arr in arrays}
    maps = {f"control:{path}": m for _, path, arr in arrays if (m := _control_world(model, path, arr)) is not None}
    return WorldSnapshot(model, _NO_STATE, extra or None, extra_world=maps or None)


def _model_arrays(model: Any) -> list[tuple[str, wp.array]]:
    out = []
    for name, value in sorted(vars(model).items()):
        if isinstance(value, wp.array):
            out.append((name, value))
        elif isinstance(value, newton.Model.AttributeNamespace):
            out += [(f"{name}:{sub}", arr) for sub, arr in sorted(vars(value).items()) if isinstance(arr, wp.array)]
    return out


def model_differences(src: Any, dst: Any, e: int, s: int) -> list[str]:
    """The float Model arrays (per-world parameters) whose world ``e`` rows in ``src`` differ from slot ``s``'s in
    ``dst``; the start state (``START``) is left out."""
    other = dict(_model_arrays(dst))
    out = []
    for name, arr in _model_arrays(src):
        if name in START or arr.size == 0 or name not in other:
            continue
        try:
            freq = src.get_attribute_frequency(name)
        except (KeyError, AttributeError):
            continue
        ma, mb = item_world(src, freq), item_world(dst, freq)
        if ma is None or mb is None or ma.shape[0] != arr.shape[0] or mb.shape[0] != other[name].shape[0]:
            continue
        a, b = arr.numpy(), other[name].numpy()
        if a.dtype.kind != "f":
            continue
        ra, rb = a[ma.numpy() == e], b[mb.numpy() == s]
        if ra.shape != rb.shape or not np.array_equal(ra, rb, equal_nan=True):
            out.append(name)
    return out


class WorldTransfer:
    """Copies world ``e`` of ``src`` into slot ``s`` of ``dst``, both built from the same scene (module docstring)."""

    def __init__(self, src: Any, dst: Any, skip: Iterable[str] = (), params: str = "move") -> None:
        from alloy.coupling.world_clock_params import WorldParams  # noqa: PLC0415  (it imports this module)

        self.src, self.dst = src, dst
        self.params, self._move = WorldParams(src, dst), params == "move"
        watched = [name for name, sim in (("source", src), ("destination", dst)) if getattr(sim, "_observers", None)]
        if watched:
            raise TransferError(f"step observers on the {' and '.join(watched)} (drives, pour energy): their clocks "
                                "are not moved")
        self.a, self.b = snapshot(src), snapshot(dst)
        self.ca, self.cb = control_snapshot(src), control_snapshot(dst)
        skip = set(skip)
        sa = {s.name: s for s in self.a.slots if s.name not in skip}
        sb = {s.name: s for s in self.b.slots if s.name not in skip}
        sa.update({s.name: s for s in self.ca.slots if s.name not in skip})
        sb.update({s.name: s for s in self.cb.slots if s.name not in skip})
        whole = sorted({n for side in (sa, sb) for n, s in side.items() if s.item_world is None})
        if whole:                        # first: a lazily allocated whole array (VBD contact history) on one side only
            raise TransferError(f"arrays without a per-world axis: {whole}")
        if set(sa) != set(sb):
            raise TransferError(f"slot names differ: only in the source {sorted(set(sa) - set(sb))}, only in the "
                                f"destination {sorted(set(sb) - set(sa))}")
        bad = sorted(n for n, s in sa.items() if (s.words, s.byte_mode) != (sb[n].words, sb[n].byte_mode))
        if bad:
            raise TransferError(f"item sizes differ: {bad}")
        self.pairs = [(n, sa[n], sb[n]) for n in sorted(sa)]
        self._iw = {n: (sa[n].item_world.numpy(), sb[n].item_world.numpy()) for n in sa}

    def items(self, e: int, s: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per slot name, world ``e``'s item indices in the source and slot ``s``'s in the destination (same length)."""
        out = {}
        for name, _, _ in self.pairs:
            ia, ib = (np.flatnonzero(iw == w) for iw, w in zip(self._iw[name], (e, s), strict=True))
            if ia.shape != ib.shape:
                raise TransferError(f"{name}: world {e} has {ia.size} items, slot {s} has {ib.size}")
            out[name] = (ia, ib)
        return out

    def check(self, e: int, s: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Raises ``TransferError`` when world ``e`` cannot move into slot ``s`` (item counts; per-world Model
        parameters that cannot move with it, ``WorldParams.refusal``); no side effects, so a caller can refuse before
        it changes anything (WP12, S2). The item rows."""
        rows = self.items(e, s)
        if self._move:
            self.params.check(e, s)
        return rows

    def copy(self, e: int, s: int) -> None:
        """World ``e`` of the source -> slot ``s`` of the destination, bit for bit; the destination's other slots and
        its global items are untouched."""
        rows = self.check(e, s)
        if self._move:
            self.params.move(e, s)       # before the state: a solver's refresh never sees half a world
            from alloy.coupling.world_clock_params import move_watch_reference  # noqa: PLC0415  (imports this module)

            move_watch_reference(self.src, self.dst, e, s)   # the watchdog judges the world on its own scale
        elif self.params.differing(e, s):
            raise TransferError(f"world {e} -> slot {s}: per-world model parameters differ on a transfer that does "
                                f"not move them: {self.params.differing(e, s)[:8]}")
        self.a.save(self.src.state)
        self.b.save(self.dst.state)
        self.ca.save(_NO_STATE)
        self.cb.save(_NO_STATE)
        for name, a, b in self.pairs:
            ia, ib = rows[name]
            if not ia.size:
                continue
            src = a.store.numpy().reshape(-1, a.words)
            dst = b.store.numpy().reshape(-1, b.words).copy()
            dst[ib] = src[ia]
            b.store.assign(dst.reshape(-1))
        mask = np.zeros(int(self.dst.model.world_count), np.int32)
        mask[s] = 1
        m = wp.array(mask, dtype=wp.int32, device=self.dst.model.device)
        self.b.restore(self.dst.state, m)
        self.cb.restore(_NO_STATE, m)
        self._translate_health(e, s)

    def copy_controls(self, e: int, s: int) -> None:
        """Only world ``e``'s Control rows (joint targets, joint_f, MuJoCo ctrl) -> slot ``s``: a resident's inputs,
        every period (world_clock_run)."""
        rows = self.items(e, s)
        self.ca.save(_NO_STATE)
        self.cb.save(_NO_STATE)
        for name, a, b in self.pairs:
            ia, ib = rows[name]
            if not name.startswith("extra:control:") and not name.startswith("control:") or not ia.size:
                continue
            dst = b.store.numpy().reshape(-1, b.words).copy()
            dst[ib] = a.store.numpy().reshape(-1, a.words)[ia]
            b.store.assign(dst.reshape(-1))
        mask = np.zeros(int(self.dst.model.world_count), np.int32)
        mask[s] = 1
        self.cb.restore(_NO_STATE, wp.array(mask, dtype=wp.int32, device=self.dst.model.device))

    def _translate_health(self, e: int, s: int) -> None:
        """The moved health rows (and the start checks' spin body) name bodies, shapes and particles by global index:
        re-point them at slot ``s``'s."""
        from alloy.integration.simulation import _health_arrays  # noqa: PLC0415

        health = getattr(self.dst, "health", None)
        arr = (_health_arrays(health) if health is not None else {}).get("health.record")
        if arr is not None:
            rec = arr.numpy()
            fields = [f for f in HEALTH_INDEX if rec.dtype.names and f in rec.dtype.names]
            if not fields or rec.shape[0] != int(self.dst.model.world_count):
                return
            for f in fields:
                rec[f][s] = _lookup(self.src.model, self.dst.model, HEALTH_INDEX[f], e, s, int(rec[f][s]))
            arr.assign(rec)
        peak = getattr(getattr(health, "start", None), "peak_body", None)
        if peak is not None and peak.shape[0] == int(self.dst.model.world_count):
            host = peak.numpy()
            host[s] = _lookup(self.src.model, self.dst.model, "body_world", e, s, int(host[s]))
            peak.assign(host)


def _lookup(src: Any, dst: Any, attr: str, e: int, s: int, i: int) -> int:
    """Global item ``i`` of world ``e`` (or a global item) in ``src`` -> the same item of slot ``s`` in ``dst``; any
    other value (-1, a default 0 naming another world's item) is kept as it is."""
    a, b = getattr(src, attr, None), getattr(dst, attr, None)
    if i < 0 or a is None or b is None or i >= a.shape[0]:
        return i
    wa, wb = a.numpy(), b.numpy()
    w = int(wa[i])
    if w not in (e, -1):
        return i
    ia, ib = np.flatnonzero(wa == w), np.flatnonzero(wb == (s if w == e else w))
    k = int(np.searchsorted(ia, i))
    return int(ib[k]) if k < ib.size else i


__all__ = ["HEALTH_INDEX", "START", "TransferError", "WorldTransfer", "control_snapshot", "item_world",
           "model_differences", "snapshot"]
