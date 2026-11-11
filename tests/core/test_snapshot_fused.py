# SPDX-License-Identifier: Apache-2.0
"""WorldSnapshot's fused copy (SCR-WP0-RIGIDDT-2): word slots go 32 to a launch instead of one launch each, bit for
bit the per-slot ``_copy_words`` path (still used for byte slots), with and without a world mask (CPU, light)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core import snapshot as S
from alloy.core.snapshot import WorldSnapshot, mask_from_worlds

DEV, W = "cpu", 3
EXTRA_DTYPES = (wp.float32, wp.vec3, wp.int8, wp.uint16, wp.float64, wp.transform)   # int8 / uint16: byte slots


def _model():
    one = newton.ModelBuilder()
    body = one.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    one.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    b = newton.ModelBuilder()
    b.replicate(one, W)
    return b.finalize(device=DEV)


def _extras(n: int) -> tuple[dict[str, wp.array], dict[str, wp.array]]:
    """n world-major extras of mixed dtypes (more than one fused group), plus one item->world mapped extra with a
    global item (-1: copied only without a mask)."""
    extra = {f"x{k:02d}": wp.zeros((W, 1 + k % 4), dtype=EXTRA_DTYPES[k % len(EXTRA_DTYPES)], device=DEV)
             for k in range(n)}
    extra["mapped"] = wp.zeros(6, dtype=wp.vec2, device=DEV)
    worlds = {"mapped": wp.array([0, 0, 1, 2, 2, -1], dtype=wp.int32, device=DEV)}
    return extra, worlds


def _fill(views, rng) -> None:
    """Random bits, with a NaN payload and -0.0 at the start of every word array."""
    for view in views:
        if view.dtype == wp.uint8:
            view.assign(rng.integers(0, 256, view.size, dtype=np.uint8))
            continue
        bits = rng.integers(0, 2 ** 32, view.size, dtype=np.uint64).astype(np.uint32)
        bits[:2] = np.array([0x7FC0BEEF, 0x80000000], dtype=np.uint32)[:min(2, view.size)]
        view.assign(bits)


def _words(views) -> list[np.ndarray]:
    return [v.numpy().copy() for v in views]


def _setup(seed: int = 0, n_extra: int = 40):
    model = _model()
    state = model.state()
    extra, worlds = _extras(n_extra)
    snap = WorldSnapshot(model, state, extra, extra_world=worlds)
    sources = [snap._source(slot, state) for slot in snap.slots]      # flat u32 / u8 views of every source
    rng = np.random.default_rng(seed)
    _fill(sources, rng)
    return model, state, snap, sources, rng


def _reference(snap: WorldSnapshot, state, mask, into_store: bool) -> None:
    """The pre-fusion path: one ``_copy_words`` / ``_copy_bytes`` launch per slot."""
    for slot in snap.slots:
        arr = snap._source(slot, state)
        src, dst = (arr, slot.store) if into_store else (slot.store, arr)
        snap._launch(slot, src, dst, mask)


def test_the_fixture_spans_two_fused_groups_and_byte_slots():
    _, _, snap, _, _ = _setup()
    word = [s for s in snap.slots if not s.byte_mode]
    assert len(word) > S.FUSED and any(s.byte_mode for s in snap.slots)


@pytest.mark.parametrize("worlds", [None, [1], [0, 2]])
def test_restore_matches_the_per_slot_copy_bitwise(worlds):
    _, state, snap, sources, rng = _setup()
    snap.save(state)
    saved = _words(sources)
    _fill(sources, rng)                                             # scramble, then restore some worlds
    scrambled = _words(sources)
    mask = mask_from_worlds(worlds, W, DEV) if worlds is not None else None
    snap.restore(state, mask)
    fused = _words(sources)
    for view, words in zip(sources, scrambled):
        view.assign(words)
    _reference(snap, state, mask, into_store=False)
    for got, want in zip(fused, _words(sources)):
        assert np.array_equal(got, want)
    if worlds is None:
        assert all(np.array_equal(a, b) for a, b in zip(fused, saved))
    else:
        assert any(not np.array_equal(a, b) for a, b in zip(fused, saved))   # unmasked worlds kept their words


@pytest.mark.parametrize("worlds", [None, [2]])
def test_save_matches_the_per_slot_copy_bitwise(worlds):
    stores = []
    for fused in (True, False):                                     # two identical set-ups, the same draws
        _, state, snap, sources, rng = _setup(seed=1)
        snap.save(state)
        _fill(sources, rng)
        mask = mask_from_worlds(worlds, W, DEV) if worlds is not None else None
        if fused:
            snap.save(state, mask)
        else:
            _reference(snap, state, mask, into_store=True)
        stores.append([slot.store.numpy().copy() for slot in snap.slots])
    assert all(np.array_equal(a, b) for a, b in zip(*stores))


def test_one_launch_per_fused_group(monkeypatch):
    _, state, snap, _, _ = _setup()
    calls = []
    real = wp.launch
    monkeypatch.setattr(wp, "launch", lambda kernel, *a, **k: (calls.append(kernel.key), real(kernel, *a, **k))[1])
    snap.save(state)
    words = sum(not s.byte_mode for s in snap.slots)
    groups = -(-words // S.FUSED)
    assert calls.count(S._copy_slots.key) == groups
    assert calls.count(S._copy_bytes.key) == sum(s.byte_mode for s in snap.slots)
    assert S._copy_words.key not in calls


def test_restore_selected_copies_nothing_for_an_empty_mask(monkeypatch):
    """restore_selected (the latch's restore, SCR-WP0-RIGIDDT-2): no world selected, no copy launched and the state
    kept; a selected world, exactly restore()'s result."""
    _, state, snap, sources, rng = _setup(seed=2)
    snap.save(state)
    _fill(sources, rng)
    scrambled = _words(sources)
    calls = []
    real = wp.launch
    monkeypatch.setattr(wp, "launch", lambda kernel, *a, **k: (calls.append(kernel.key), real(kernel, *a, **k))[1])
    snap.restore_selected(state, mask_from_worlds([], W, DEV))
    assert calls == [] and all(np.array_equal(a, b) for a, b in zip(_words(sources), scrambled))
    mask = mask_from_worlds([1], W, DEV)
    snap.restore_selected(state, mask)
    selected = _words(sources)
    for view, words in zip(sources, scrambled):
        view.assign(words)
    snap.restore(state, mask)
    assert all(np.array_equal(a, b) for a, b in zip(selected, _words(sources)))


def test_a_latched_world_stays_frozen_across_cpu_graph_replays(monkeypatch):
    """WP5's review: while a CPU graph records, the latch's mask is not computed yet, so restore_selected must record
    the copy (no host check). A world sent past SPEED_FAIL mid-run then latches and stays frozen across replays."""
    import alloy
    from alloy.integration import build_simulation

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05), density=500.0)
    built = alloy.build(scene, device="cpu").built
    sim = build_simulation(built, "standard", graph="on")
    for _ in range(3):
        sim.step_period()
    launches = sim.graph_stats()["launches"]
    jqd = sim.state.joint_qd.numpy()
    jqd[sim.model.joint_qd_start.numpy()[1] + 2] = 60.0               # world 1's cube: 60 m/s up (> SPEED_FAIL)
    sim.state.joint_qd.assign(jqd)
    sim.step_period()
    frozen = sim.state.body_q.numpy()[1].copy()
    for _ in range(3):
        sim.step_period()
    assert sim.graph_stats()["launches"] > launches                    # the periods replayed graphs
    report = sim.health.poll()
    assert report.invalid_worlds == (1,)
    assert np.array_equal(sim.state.body_q.numpy()[1], frozen)          # latched: frozen across the replays
