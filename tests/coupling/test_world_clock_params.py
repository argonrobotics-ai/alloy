# SPDX-License-Identifier: Apache-2.0
"""#10b S1b laws (alloy.coupling.world_clock_params; [internal note] notes/gap-10b/s1b_params.md): a randomized world
moves into a fine-pack slot with its parameters. Before S1b ``WorldTransfer.check`` refused it ("per-world model
parameters differ"), so every law below that moves a world failed there.
(i) parity: the moved world steps bit for bit as in its own pack;
(ii) round trip: out and back is the identity, Model rows included, and the slot gets its own parameters back;
(iii) derived state: after the move the slot is, array for array, a simulation built as that world alone;
(iv) what cannot move is refused by name.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.api._build import _with_worlds, build_block
from alloy.coupling.fine_pack import (HEALTH_INDEX, TransferError, WorldTransfer, _model_arrays, control_snapshot,
                                      snapshot)
from alloy.coupling.world_clock_params import HOME, WorldParams
from alloy.integration.simulation import _health_arrays

W, E = 3, 2                         # the coarse pack's worlds; the world moved
SEED = 3
PROFILE = ("standard", None)
SKIP = {"workdir", "_workdir", "scene", "_scene", "timings", HOME}
# What a slot may still hold that its world's own build does not (law iii), by path fragment, each with its reason:
ALLOWED = {
    "world_id_offset": "the slot's build id: only a build or a reset reads it (WP12), and a fine pack refuses reset",
    "model_digest": "a digest of the build's Model",
    ".id": "Warp handles (grids, BVHs)", "_cache": "the card library's cache (the scene's world count)",
    "geomcollisionid": "contact scratch", "_particle_contact_next": "contact scratch",
    "qLDiagInv": "MuJoCo data recomputed by the next forward pass", "qfrc_bias": "as qLDiagInv",
    "qfrc_smooth": "as qLDiagInv", "_alloy_pair_refitting": "pair_fit's re-entry flag (None until a first notify)",
    "_e_scratch": "health energy scratch", "_energy.partial": "health energy scratch",
    "_watchdog.energy": "watchdog scratch (energy_ref is moved)", "inputs.e_part": "watchdog scratch",
    "latch_log._host": "the latch log's host copy",
    "shape_material_ke": "UNVERIFIED (s1b_params.md, left to do 2): the proxy view's record gains after a SHAPE notify",
    "shape_material_kd": "as shape_material_ke",
}
# Build-time copies S1b does not move yet (s1b_params.md, left to do 1): the law pins the list, so a new one fails.
OPEN = {"sim.health._scale_floor", "sim.stepper._census_used[0]"}


def _scene(worlds: int, rand: tuple[tuple[str, str, float, float], ...]) -> alloy.Scene:
    """A box dropped on a cloth (MuJoCo + VBD under the Scheduler) with per-world draws ``rand``."""
    sc = alloy.Scene(worlds=worlds, dt=0.01)
    sc.add_ground()
    sc.add_cloth("sheet", size=(0.3, 0.3), pos=(0.0, 0.0, 0.05), resolution=8, mass=0.05, self_contact=False)
    sc.add_box("box", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.10), mass=0.2)
    for name, field, lo, hi in rand:
        sc.randomize(name, field, lo, hi)
    return sc


BOX = (("box", "density", 500.0, 1500.0), ("box", "friction", 0.3, 0.9))


def _build(sc: alloy.Scene, worlds: int, block: tuple[int, int] | None = None):
    return build_block(_with_worlds(sc, worlds, trim=True), worlds, "cpu", SEED, profile=PROFILE, plugins=[],
                       block=block)


def _local(model, attr: str, i: int) -> int:
    """A global body / shape / particle index as its index among its own world's items (-1 stays)."""
    wa = getattr(model, attr).numpy()
    if not 0 <= i < wa.size:
        return i
    return int(np.searchsorted(np.flatnonzero(wa == wa[i]), i))


def _words(sim, w: int) -> dict[str, np.ndarray]:
    """Every slot's words of world ``w`` (copies): State, the stepper's arrays, health (the record's item indices as
    world-local ones, as test_fine_pack._world_words) and Control."""
    out = {}
    for snap, src in ((snapshot(sim), sim.state), (control_snapshot(sim), None)):
        snap.save(src)
        for s in snap.slots:
            out[s.name] = s.store.numpy().reshape(-1, s.words)[s.item_world.numpy() == w].copy()
    row = _health_arrays(sim.health)["health.record"].numpy()[w:w + 1].copy()
    for f, attr in HEALTH_INDEX.items():
        row[f] = _local(sim.model, attr, int(row[f][0]))
    out["extra:health/health.record"] = np.frombuffer(row.tobytes(), np.uint8)
    out.pop("extra:health/start.peak_body", None)        # a global body index (test_fine_pack localizes and checks it)
    return out


def _differ(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    return sorted(k for k in a if not np.array_equal(a[k], b[k]))


def _model_bytes(sim) -> dict[str, bytes]:
    out = {n: a.numpy().tobytes() for n, a in _model_arrays(sim.model) if a.size}
    table = sim.built.pair_table
    out.update({f"pair_table.{n}": getattr(table, n).numpy().tobytes() for n in ("world_table", "world_stiffness")
                if getattr(table, n, None) is not None})
    out.update({f"randomized.{k}": np.asarray(v).tobytes() for k, v in sim.built.randomized.items()})
    for i, es in enumerate(getattr(sim.stepper, "_sets", [])):
        out[f"set{i}.meff"] = es.meff.numpy().tobytes()
        out[f"set{i}.m_axis"] = b"" if es.m_axis is None else np.asarray(es.m_axis).tobytes()
    out["pair_mass"] = repr(sorted(sim.stepper._pair_mass.items())).encode()
    return out


def _walk(a, b, path: str, out: list[str], seen: set[int], depth: int = 0) -> None:
    """Every array or number under ``a`` that differs from its counterpart under ``b`` (same attribute path)."""
    if depth > 9 or a is None or id(a) in seen or isinstance(a, (str, bytes, type, types.ModuleType,
                                                                 types.FunctionType, types.MethodType)):
        return
    if isinstance(a, (wp.array, np.ndarray)):
        seen.add(id(a))
        x = a.numpy() if isinstance(a, wp.array) and a.size else np.asarray(a if isinstance(a, np.ndarray) else [])
        y = b.numpy() if isinstance(b, wp.array) and b.size else np.asarray(b if isinstance(b, np.ndarray) else [])
        if x.dtype != object and (x.shape != y.shape or x.tobytes() != y.tobytes()):
            out.append(path)
        return
    if isinstance(a, (bool, int, float, np.generic)):
        if not (a == b or (a != a and b != b)):
            out.append(path)
        return
    seen.add(id(a))
    if isinstance(a, dict):
        for k in (a if isinstance(b, dict) else ()):
            if k in b and str(k) not in SKIP:
                _walk(a[k], b[k], f"{path}[{k!r}]", out, seen, depth + 1)
    elif isinstance(a, (list, tuple)):
        if isinstance(b, (list, tuple)) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                _walk(x, y, f"{path}[{i}]", out, seen, depth + 1)
    elif type(a) is type(b) and hasattr(a, "__dict__"):
        for k, v in vars(a).items():
            if k not in SKIP:
                _walk(v, getattr(b, k, None), f"{path}.{k}", out, seen, depth + 1)


def _unlike(a, b) -> list[str]:
    out: list[str] = []
    _walk(a, b, "sim", out, set())
    return [p for p in out if not any(key in p for key in ALLOWED)]


def _clock_of(sim, control_dt: float, rep) -> tuple[float, dict[str, int]]:
    """The coarse pack's dt_x and substeps for ``sim``'s period; the real ``choose_clock`` also opens the watchdog's
    period (window push, cleared watch levels), which this stand-in must do or the slot keeps last period's levels."""
    sim.stepper._watchdog.begin_period(control_dt)
    return float(rep.dt_x), dict(rep.substeps)


@pytest.fixture(scope="module")
def packs():
    """The coarse pack (3 worlds, each its own box density and friction), a 1-slot fine pack (built as world 0) and
    world E built alone (the reference of law iii). ``facts`` holds what was read before any step."""
    sc = _scene(W, BOX)
    coarse, fine, alone = _build(sc, W), _build(sc, 1), build_block(sc, W, "cpu", SEED, profile=PROFILE, plugins=[],
                                                                    block=(E, 1))
    try:
        a, b = coarse.engine, fine.engine
        facts = {"home": _model_bytes(b), "coarse": _model_bytes(a), "before": _unlike(b, alone.engine),
                 "differing": WorldParams(a, b).differing(E, 0)}
        WorldTransfer(a, b).copy(E, 0)
        facts["after"] = _unlike(b, alone.engine)
        yield a, b, facts
    finally:
        for s in (coarse, fine, alone):
            s.close()


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_moved_slot_is_the_world_built_alone(packs) -> None:
    """Law (iii). Before the move the slot (world 0's draws) differs from world E built alone in its Model rows and in
    what the solvers and the coupler derived from them; after it, in nothing (``ALLOWED`` aside): no build-time copy
    of a randomized parameter is left behind."""
    _, _, facts = packs
    assert {"body_mass", "body_inertia", "shape_material_mu"} <= set(facts["differing"]), facts["differing"]
    assert any("meff" in p for p in facts["before"]), facts["before"]        # the coupler's proxy masses were baked
    assert set(facts["after"]) <= OPEN, sorted(set(facts["after"]) - OPEN)


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_randomized_world_out_and_back_is_itself_and_the_slot_gets_its_own_back(packs) -> None:
    """Law (ii). Out: the slot holds world E's state. Back (``params="same"``): every world of the coarse pack and
    every Model row, pair-table row and proxy mass of it is bit for bit as before. Restored, the slot's parameters
    and what is derived from them are bit for bit its build's."""
    a, b, facts = packs
    before = {w: _words(a, w) for w in range(W)}
    there, back = WorldTransfer(a, b), WorldTransfer(b, a, params="same")
    there.copy(E, 0)
    assert _differ(_words(b, 0), before[E]) == []
    assert there.params.differing(E, 0) == []
    back.copy(0, E)
    for w in range(W):
        assert _differ(_words(a, w), before[w]) == [], w
    assert _model_bytes(a) == facts["coarse"]
    assert set(there.params.restore(0)) >= {"body_mass", "body_inertia"}
    now = _model_bytes(b)
    assert sorted(k for k in now if now[k] != facts["home"][k]) == []
    assert there.params.restore(0) == []                                      # nothing kept: a second call is a no-op


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_moved_randomized_world_steps_as_in_its_own_pack(packs) -> None:
    """Law (i). World E (its own box mass and friction) moved into the slot after the box has landed, then stepped at
    the coarse pack's dt_x and substeps, ends every period bit for bit as world E stepped in the coarse pack."""
    a, b, _ = packs
    for _ in range(12):
        a.step_period()
    WorldTransfer(a, b).copy(E, 0)
    b.clock.ticks, b.steps, b.periods = a.clock.ticks, a.steps, a.periods
    wa, wb = a.stepper._watchdog, b.stepper._watchdog                      # the pack's own window clock, as FineRerun sets
    wp.copy(wb.elapsed_ns, wa.elapsed_ns)
    wb.head = wa.head
    for _ in range(4):
        a.step_period()
        rep = a.stepper.report()
        assert int(rep.rollbacks) == 0
        b.stepper.choose_clock = lambda control_dt, last, rep=rep: _clock_of(b, control_dt, rep)
        b.step_period()
        assert _differ(_words(b, 0), _words(a, E)) == []
    assert a.state.body_q.numpy()[E, 2] < 0.10                               # on the cloth: contact in play
    mass = a.model.body_mass.numpy()
    assert len({float(m) for m in mass}) == W                                # the worlds really differ


@pytest.mark.law
@pytest.mark.timeout(900)
def test_what_cannot_move_is_refused_by_name(packs) -> None:
    """Law (iv). A randomized cloth (particle parameters), the contact gate beside randomized parameters, and a
    transfer told not to move parameters each refuse with their reason, before anything is written."""
    a, b, facts = packs
    sc = _scene(2, (("sheet", "density", 200.0, 600.0),))
    c, d = _build(sc, 2), _build(sc, 1)
    try:
        rows = _model_bytes(d.engine)
        with pytest.raises(TransferError, match="randomized particle parameters.*particle_mass"):
            WorldTransfer(c.engine, d.engine).copy(1, 0)
        assert _model_bytes(d.engine) == rows
    finally:
        c.close()
        d.close()
    params = WorldParams(a, b)
    params.restore(0)
    assert params.differing(E, 0)
    b.stepper._gate = object()
    try:
        with pytest.raises(TransferError, match="contact gate"):
            params.move(E, 0)
    finally:
        b.stepper._gate = None
    with pytest.raises(TransferError, match="does not move them.*body_mass"):
        WorldTransfer(a, b, params="same").copy(E, 0)
    assert _model_bytes(b) == facts["home"]
