# SPDX-License-Identifier: Apache-2.0
"""Self-contact rows grow between periods; a period that dropped pairs re-runs (GAP-13B, ``row_growth``).

Fast host-side laws on a stand-in solver: the growth rule, the two words, the bytes cap, and what ``regrow_rows``
restores. The scene laws (a stack that overflows 32 / 64) are in ``tests/solvers/vbd/test_self_contact_rows.py``."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.integration import grid_growth
from alloy.integration import row_growth as G
from alloy.integration.period_snapshots import save_period_start
from alloy.solvers.vbd import self_contact_rows as R

VERTS, EDGES = 6, 9


def _info(vertex_size: int = 32, edge_size: int = 64) -> SimpleNamespace:
    def family(n: int, size: int) -> dict[str, wp.array]:
        return {"": wp.zeros(2 * n * size, dtype=wp.int32, device="cpu"),
                "_count": wp.zeros(n, dtype=wp.int32, device="cpu"),
                "_buffer_sizes": wp.full(n, size, dtype=wp.int32, device="cpu"),
                "_offsets": wp.array(np.arange(n + 1) * size, dtype=wp.int32, device="cpu")}
    fields = {f"{field}{k}": v for field, (n, size) in (("vertex_colliding_triangles", (VERTS, vertex_size)),
                                                        ("edge_colliding_edges", (EDGES, edge_size)))
              for k, v in family(n, size).items()}
    return SimpleNamespace(**fields)


def _state(bytes_cap: int | None = None) -> R.RowState:
    solver = SimpleNamespace(model=SimpleNamespace(device=wp.get_device("cpu")), trimesh_collision_info=_info())
    return R.RowState(solver, bytes_cap)


def _detect(state: R.RowState, vertex: list[int], edge: list[int]) -> None:
    info = state.solver.trimesh_collision_info
    info.vertex_colliding_triangles_count.assign(np.array(vertex, np.int32))
    info.edge_colliding_edges_count.assign(np.array(edge, np.int32))
    state.after_detection()


def test_grown_sizes_never_shrink_hold_the_need_and_stop_at_the_cap() -> None:
    need = np.array([0, 16, 17, 24, 33, 100, 400])
    size = np.full(7, 32)
    out = R.grown_sizes(need, size)
    assert (out >= size).all() and (out <= R.ROW_CAP).all() and (out % 8 == 0).all()
    assert out[:2].tolist() == [32, 32]                              # at or under half full: untouched
    assert (out[2:6] >= np.ceil(R.HEADROOM * need[2:6])).all() and (out[2:6] >= 64).all()
    assert out[6] == R.ROW_CAP                                       # more than a row can hold: the cap


def test_rows_under_the_fill_mark_keep_their_arrays() -> None:
    state = _state()
    info = state.solver.trimesh_collision_info
    before = (info.vertex_colliding_triangles, info.edge_colliding_edges_offsets)
    _detect(state, [24] * VERTS, [48] * EDGES)                       # exactly 75 %: not over the mark
    assert state.grow() is None and state.grown == []
    assert (info.vertex_colliding_triangles, info.edge_colliding_edges_offsets) == before


def test_a_row_over_the_mark_grows_before_it_drops_a_pair() -> None:
    state = _state()
    _detect(state, [25, 17, 3, 0, 0, 0], [0] * EDGES)
    g = state.grow()
    assert g is not None and not g.dropped and g.rows["vertex"] == (2, 64) and g.rows["edge"] == (0, 64)
    info = state.solver.trimesh_collision_info
    sizes = info.vertex_colliding_triangles_buffer_sizes.numpy()
    assert sizes.tolist() == [64, 64, 32, 32, 32, 32]                # the asking row and the one over half full
    assert info.vertex_colliding_triangles_offsets.numpy().tolist() == [0, 64, 128, 160, 192, 224, 256]
    assert info.vertex_colliding_triangles.shape[0] == 2 * 256 and state.bytes() == g.bytes == 8 * (256 + EDGES * 64)
    assert not info.vertex_colliding_triangles_count.numpy().any()   # empty rows until the next detection
    assert state.grow() is None                                      # the words were cleared


def test_dropped_pairs_are_said_and_capped_rows_do_not_ask() -> None:
    state = _state()
    _detect(state, [0] * VERTS, [70] + [0] * (EDGES - 1))
    g = state.grow()
    assert g is not None and g.dropped and g.rows["edge"] == (1, 128)
    _detect(state, [300] + [0] * (VERTS - 1), [0] * EDGES)
    assert state.grow().rows["vertex"] == (1, R.ROW_CAP)
    _detect(state, [300] + [0] * (VERTS - 1), [0] * EDGES)           # at the cap: pairs drop, no growth is asked
    assert state.grow() is None and not state.exhausted


def test_bytes_cap_stops_growth_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    state = _state(bytes_cap=8 * (VERTS * 32 + EDGES * 64) + 8)
    _detect(state, [40] * VERTS, [0] * EDGES)
    with caplog.at_level(logging.WARNING, logger="alloy.solvers.vbd"):
        assert state.grow() is None and state.exhausted
        _detect(state, [40] * VERTS, [0] * EDGES)
        assert state.grow() is None
    said = [r for r in caplog.records if "over the cap" in r.getMessage()]
    assert len(said) == 1 and R.BYTES_KEY in said[0].getMessage()
    assert state.sizes("vertex_colliding_triangles").tolist() == [32] * VERTS


def _sim(state: R.RowState, calls: list[str]) -> SimpleNamespace:
    snap = SimpleNamespace(restore=lambda s: calls.append("restore"), save=lambda s: None)
    graphs = SimpleNamespace(invalidate=lambda: calls.append("graphs"))
    stepper = SimpleNamespace(replay_period=lambda: calls.append("replay"))
    state.solver._alloy_self_contact_watch = SimpleNamespace(dropped=wp.zeros(2, dtype=wp.int32, device="cpu"))
    return SimpleNamespace(solvers=(state.solver,), _good=snap, _health_snap=None, stepper=stepper, state="s",
                           clock=SimpleNamespace(ticks=9), steps=9, _graphs=graphs, exchange_nodes={1: 1},
                           _warm={1}, periods=3)


def test_regrow_rows_reruns_only_a_period_that_dropped_pairs() -> None:
    state, calls = _state(), []
    state.solver.alloy_rows = state
    sim = _sim(state, calls)
    watch = state.solver._alloy_self_contact_watch
    assert not G.regrow_rows(sim, (4, 5)) and calls == []            # nothing asked: nothing touched
    _detect(state, [25] + [0] * (VERTS - 1), [0] * EDGES)
    assert not G.regrow_rows(sim, (4, 5))                            # grown before a drop: the period stands
    assert calls == ["graphs"] and sim.exchange_nodes == {} and sim._warm == set()
    assert (sim.clock.ticks, sim.steps) == (9, 9)
    watch.dropped.assign(np.array([0, 3], np.int32))                 # an accepted period whose capped rows dropped
    assert not G.regrow_rows(sim, (4, 5))
    G.row_guard(sim).keep_dropped()                                  # the next period starts (save_period_start)
    _detect(state, [0] * VERTS, [70] + [0] * (EDGES - 1))
    watch.dropped.assign(np.array([6, 9], np.int32))                 # what health's watch saw in the attempt
    assert G.regrow_rows(sim, (4, 5))
    assert calls[1:] == ["graphs", "restore", "replay"] and (sim.clock.ticks, sim.steps, sim.row_regrows) == (4, 5, 1)
    assert watch.dropped.numpy().tolist() == [0, 3]                  # back to the period start, not to zero
    assert [name for name, _ in G.row_guard(sim).grown] == ["0", "0"]
    watch.dropped.assign(np.array([8, 8], np.int32))                 # another re-run (an MPM grid grew) abandons an
    grid_growth.restore_period_start(sim, (4, 5))                    # attempt that dropped pairs (WP4's review)
    assert watch.dropped.numpy().tolist() == [0, 3]


def test_the_dropped_baseline_is_the_period_start_so_a_reset_between_periods_holds() -> None:
    """WP12's review: drop pairs, reset by mask (health zeroes the watch), drop again in the first period after it."""
    state, calls = _state(), []
    state.solver.alloy_rows = state
    sim = _sim(state, calls)
    sim._latch_snap = sim._good
    watch = state.solver._alloy_self_contact_watch
    save_period_start(sim)
    watch.dropped.assign(np.array([5, 7], np.int32))                 # an accepted period of the old episode
    assert not G.regrow_rows(sim, (4, 5))
    watch.dropped.assign(np.array([5, 0], np.int32))                 # reset(mask) of world 1
    save_period_start(sim)
    _detect(state, [0] * VERTS, [70] + [0] * (EDGES - 1))
    watch.dropped.assign(np.array([6, 4], np.int32))
    assert G.regrow_rows(sim, (4, 5))
    assert watch.dropped.numpy().tolist() == [5, 0]                  # the fresh episode's, not the old 7


def test_a_deterministic_solver_keeps_its_fixed_rows(caplog: pytest.LogCaptureFixture) -> None:
    """WP11's review: Newton's deterministic record arrays are sized for 32 / 64, so those rows must not grow."""
    class Base:
        def __init__(self, model, deterministic: bool) -> None:
            self.model, self.particle_enable_self_contact = model, True
            self._particle_contact_use_gather, self.trimesh_collision_info = deterministic, _info()

    cls, model = R.with_rows(Base), SimpleNamespace(device=wp.get_device("cpu"))
    assert cls(model, False).alloy_rows is not None
    with caplog.at_level(logging.WARNING, logger=R.log.name):
        assert cls(model, True).alloy_rows is None
    assert "deterministic" in caplog.text


def _late_state(monkeypatch: pytest.MonkeyPatch) -> R.RowState:
    monkeypatch.setattr(R, "LATE_ON_CPU", True)                      # the CUDA reads, checked on the CPU
    state = _state()
    assert state.reader.late
    return state


def test_late_reads_never_wait_while_no_row_is_half_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """0.1.1 item 1 (XB4-10, rc1): a blocking read every period cost a calm drape 5.7 % on an L4."""
    state = _late_state(monkeypatch)
    for _ in range(30):
        _detect(state, [16] * VERTS, [32] * EDGES)                   # exactly half full: not over ARM
        assert state.grow() is None
    assert state.reader.blocking == 0 and state.reader.late
    assert state.need is None and not state.take_rebind()           # nothing per row allocated (5 % on an L4)


def test_per_row_need_comes_in_with_the_first_row_past_half_full() -> None:
    """Allocating the per-row need arrays at build alone cost a calm drape 5 % of GPU time (L4, vmprobe3): they come
    when a row first passes ARM, and the captured graphs (whose folds hold the stand-in) are dropped once."""
    state, calls = _state(), []
    state.solver.alloy_rows = state
    sim = _sim(state, calls)
    _detect(state, [16] * VERTS, [32] * EDGES)
    assert not G.regrow_rows(sim, (4, 5)) and state.need is None and calls == []
    _detect(state, [20] + [0] * (VERTS - 1), [0] * EDGES)            # over half full, under the fill mark
    assert not G.regrow_rows(sim, (4, 5)) and calls == ["graphs"]    # no growth; the graphs go once
    assert state.need["vertex"].numpy().tolist() == [20] + [0] * (VERTS - 1)
    _detect(state, [22] + [0] * (VERTS - 1), [0] * EDGES)
    assert not G.regrow_rows(sim, (4, 5)) and calls == ["graphs"]
    assert state.need["vertex"].numpy()[0] == 22                     # tracked from now on


def test_past_half_full_the_reads_wait_until_the_rows_are_calm_again(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _late_state(monkeypatch)
    _detect(state, [20] + [0] * (VERTS - 1), [0] * EDGES)            # over half full, under the fill mark
    assert state.grow() is None and state.reader.blocking == 0       # the late read sees the period before
    _detect(state, [25] + [0] * (VERTS - 1), [0] * EDGES)            # now over the fill mark too
    g = state.grow()                                                 # the late read sees 20: waits for this period
    assert g is not None and not g.dropped and state.reader.blocking == 1 and not state.reader.late
    for k in range(R.CALM_PERIODS):
        _detect(state, [0] * VERTS, [0] * EDGES)
        assert state.grow() is None and state.reader.blocking == 2 + k
    assert state.reader.late                                         # calm again: late reads
    _detect(state, [0] * VERTS, [0] * EDGES)
    assert state.grow() is None and state.reader.blocking == 1 + R.CALM_PERIODS


def test_a_waiting_read_reruns_only_its_own_period(monkeypatch: pytest.MonkeyPatch,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    """A row that fills past its size within one calm period drops pairs before a late read can see it: that period
    stands (said once; health keeps the flag), the rows are tracked from then on, and the next drop re-runs."""
    state = _late_state(monkeypatch)
    _detect(state, [0] * VERTS, [0] * EDGES)
    assert state.grow() is None
    _detect(state, [40] + [0] * (VERTS - 1), [0] * EDGES)            # under half full to over full in one period
    assert state.grow() is None                                      # not seen yet: that period stands
    _detect(state, [10] + [0] * (VERTS - 1), [0] * EDGES)
    with caplog.at_level(logging.WARNING, logger="alloy.solvers.vbd"):
        g = state.grow()                                             # sees the drop a period late: no re-run
    assert g is None and state.need is not None and not state.reader.late
    assert sum("before the late read saw it" in r.getMessage() for r in caplog.records) == 1
    _detect(state, [0] * VERTS, [200] + [0] * (EDGES - 1))           # the reads now wait: this period's drop re-runs
    g = state.grow()
    assert g is not None and g.dropped


def test_a_simulation_without_such_rows_has_no_guard() -> None:
    sim = SimpleNamespace(solvers=(SimpleNamespace(),))
    assert G.row_guard(sim) is None and not G.regrow_rows(sim, (0, 0))


def test_rows_mode_comes_from_the_kwarg_then_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    on = {"particle_enable_self_contact": True}
    monkeypatch.delenv(R.ROWS_ENV, raising=False)
    assert R.ROWS_DEFAULT == R.ROWS_SCENE                            # the default (ruling 2026-11-10 04:34Z)
    assert R.pop_rows(dict(on)) and not R.pop_rows({**on, R.ROWS_KEY: "fixed"})
    monkeypatch.setenv(R.ROWS_ENV, "fixed")
    assert not R.pop_rows(dict(on)) and R.pop_rows({**on, R.ROWS_KEY: "scene"})
    assert not R.pop_rows({})                                        # no self contact: nothing to size
    with pytest.raises(ValueError, match=R.ROWS_ENV):
        R.pop_rows({**on, R.ROWS_KEY: "big"})
