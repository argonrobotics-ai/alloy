# SPDX-License-Identifier: Apache-2.0
"""patch-drywall (CPU): Gym-Anything's wallboard start, its five criteria at their edges and progress-v1 on synthetic
boards, the scripted strokes' geometry, and the scripted patch on a 10 mm grid: the holes filled, the patch flush,
the mesh covered, nothing stray, the books closed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import concrete_drywall as dw
from alloy.tasks import concrete_drywall_checks as ck
from alloy.tasks import concrete_drywall_start as ds
from alloy.tasks.concrete_materials import masco_factor


def test_the_wallboard_is_gym_anythings() -> None:
    """layout.py:529-562: four 6 mm holes through 5/8 in board, the 100 mm mesh at 0.15 mm, the mound 90 mm round,
    4 min old at 21 °C; their seeds draw different noise."""
    wb = ds.wallboard(0)
    assert wb.base.shape == (840, 600)
    assert wb.holes.sum() == 4 * 32 and np.all(wb.base[wb.holes] == -0.0159)
    assert wb.mesh.sum() == 100 * 100 and np.all(wb.base[wb.mesh] == pytest.approx(0.15e-3, rel=1e-12))
    assert not wb.film[wb.holes].any()
    r = np.hypot(wb.x - 1.80, wb.z - 1.25)
    assert not wb.film[r >= 0.09].any() and 2.0e-3 < wb.film.max() < 4.0e-3
    assert float(wb.film.sum()) * 1e-6 == pytest.approx(23.0e-6, rel=0.05)              # about 23 mL of compound
    assert np.all(wb.top[wb.mesh] >= wb.base[wb.mesh])
    assert wb.age == pytest.approx(240.0 * masco_factor(21.1), rel=1e-12)
    assert not np.array_equal(ds.wallboard(1).film, wb.film) and np.array_equal(ds.wallboard(0).film, wb.film)
    assert np.all(wb.support()[wb.mesh] == ds.MESH_T) and np.all(wb.support()[~wb.mesh] == wb.base[~wb.mesh])


@pytest.mark.parametrize("cell", [0.002, 0.005, 0.01])
def test_a_coarser_grid_keeps_the_compound_and_the_masks(cell: float) -> None:
    wb, g = ds.wallboard(0), ds.on_grid(ds.wallboard(0), cell)
    k = int(round(cell / 1e-3))
    assert g.base.shape == (840 // k, 600 // k)
    assert float(g.film.sum()) * cell ** 2 == pytest.approx(float(wb.film.sum()) * 1e-6, rel=1e-12)
    assert float(g.base.sum()) * cell ** 2 == pytest.approx(float(wb.base.sum()) * 1e-6, rel=1e-12)
    assert g.mesh.sum() * cell ** 2 == pytest.approx(0.01, rel=1e-12)            # the mesh's edges are cell edges
    assert len({tuple(m.nonzero()[0]) for m in ck.hole_masks(g)}) == 4 and all(m.any() for m in ck.hole_masks(g))
    for bad in (0.003, 0.0025):
        with pytest.raises(ValueError, match="1, 2, 5 or 10 mm"):
            ds.on_grid(wb, bad)


def _done(board: ds.Wallboard, mesh_h: float = 0.55e-3) -> ck.DrywallGrid:
    """A finished board: holes filled flush, the mesh at ``mesh_h``, the rest bare."""
    film = np.where(board.holes, -board.base, np.where(board.mesh, mesh_h - board.base, 0.0))
    return ck.DrywallGrid(h=board.base + film, t=film, board=board)


def _start(board: ds.Wallboard) -> ck.DrywallGrid:
    return ck.DrywallGrid(h=board.top, t=board.film, board=board)


def _failed(v) -> list[str]:  # noqa: ANN001
    return [c.name for c in v.criteria if not c.passed]


def test_the_criteria_at_their_edges() -> None:
    """task.py:683-705: each criterion just inside and just outside its threshold."""
    board = ds.on_grid(ds.wallboard(0), 0.005)
    start = _start(board)
    assert _failed(ck.patch_drywall(start))[:3] == ["holes_filled", "flush", "over_mesh"]
    good = _done(board)
    v = ck.patch_drywall(good, start)
    assert v.success and v.score == 1.0 and not _failed(v)
    hole = ck.hole_masks(board)[2]
    for dz, failed in ((-0.29e-3, []), (-0.31e-3, ["holes_filled", "flush"])):    # -0.3 mm; flush reads |h|
        g = _done(board)
        g.h[hole] += dz * board.frac[hole]                                    # the hole part dz short, the board flush
        g.t[hole] += dz * board.frac[hole]
        assert _failed(ck.patch_drywall(g)) == failed
    beside = board.near() & ~board.mesh & ~board.holes
    cell = tuple(np.argwhere(beside)[0])
    for t, ok in ((0.29e-3, True), (0.31e-3, False), (0.04e-3, True)):     # flush: 0.3 mm; under 0.05 mm: no compound
        g = _done(board)
        g.h[cell], g.t[cell] = t, t
        assert ("flush" not in _failed(ck.patch_drywall(g))) is ok
    for mh, ok in ((0.79e-3, True), (0.81e-3, False)):                       # over_mesh: 0.5 +- 0.3 mm
        assert ("over_mesh" not in _failed(ck.patch_drywall(_done(board, mh)))) is ok
    rows = np.argwhere(board.mesh)
    for n, ok in ((int(0.019 * len(rows)), True), (int(0.021 * len(rows)) + 1, False)):   # mesh_covered: under 2 %
        g = _done(board, 0.5e-3)
        for i, j in rows[:n]:
            g.h[i, j] -= 0.011e-3
        assert ("mesh_covered" not in _failed(ck.patch_drywall(g))) is ok
    far = tuple(np.argwhere(~board.near())[0])
    g = _done(board)
    g.h[far], g.t[far] = 0.06e-3, 0.06e-3                                     # one cell of compound beyond 150 mm
    assert _failed(ck.patch_drywall(g)) == ["no_stray_compound"]
    g = _done(board)
    g.off_m3 = 2.1e-8
    assert _failed(ck.patch_drywall(g)) == ["no_stray_compound"]
    d = np.min([np.hypot(board.x - x, board.z - z) for x, z in (ds.PATCH, *ds.HOLES)], axis=0)
    assert np.array_equal(board.near(), d < 0.15)                            # 150 mm of the patch or any hole
    for lo, hi, ok in ((0.14, 0.15, True), (0.15, 0.16, False)):
        g = _done(board)
        cell = tuple(np.argwhere((d > lo) & (d < hi) & ~board.mesh & ~board.holes)[0])
        g.h[cell], g.t[cell] = 0.06e-3, 0.06e-3
        assert ("no_stray_compound" not in _failed(ck.patch_drywall(g))) is ok


def test_the_graded_quantities() -> None:
    """task_set_progress.py:203-223 term by term: a hole's lowest cell; the flush area graded to 3 mm past 0.3 mm;
    the mesh's keep, the share of it under the envelope; stray compound kept to 10 cm²."""
    board = ds.on_grid(ds.wallboard(0), 0.005)
    g = _done(board)
    hole = ck.hole_masks(board)[1]
    i, j = np.argwhere(hole)[0]
    g.h[i, j] -= 0.2e-3 * board.frac[i, j]                                   # its hole part 0.2 mm short
    g.t[i, j] -= 0.2e-3 * board.frac[i, j]
    off = tuple(np.argwhere(board.near() & ~board.mesh & ~board.holes)[0])
    g.h[off], g.t[off] = 1.8e-3, 1.8e-3                                       # 1.5 mm past 0.3 mm: half credit
    rows = np.argwhere(board.mesh)
    for r, c in rows[: len(rows) // 10]:                                      # a tenth of the mesh under its fibres
        g.h[r, c] = 0.45e-3
    far = np.argwhere(~board.near())[:5]
    for r, c in far:                                                          # 5 cells: 0.05 mm over the visible
        g.t[r, c] = g.h[r, c] = 0.1e-3
    q = ck.quantities(g, ck.criteria(g))
    assert q["hole_1"].value == pytest.approx(-0.2, rel=1e-9)
    comp_off = (g.t > ck.COMP_T) & ~board.mesh
    n = int(comp_off.sum())
    assert q["flush_area"].value == pytest.approx((n - 1 + 0.5) / n, rel=1e-12)
    assert q["mesh_covered"].keep == pytest.approx(1.0 - (len(rows) // 10) / len(rows), rel=1e-12)
    area = 5 * board.cell ** 2 * 1e4                                          # cm², as 0.05 mm over each cell
    assert q["no_stray_compound"].value == pytest.approx(-area, rel=1e-9)
    assert q["no_stray_compound"].keep == pytest.approx(1.0 - area / 10.0, rel=1e-9)


def test_progress_v1_on_a_part_done_patch() -> None:
    """task_set_progress.py:197-223, 251-281: what passes at the start (no stray compound) is a gate; a half-filled
    hole earns 0.9 of its way from the start; stray compound past 10 cm² keeps nothing."""
    board = ds.on_grid(ds.wallboard(0), 0.005)
    start = _start(board)
    q0 = ck.quantities(start, ck.criteria(start))
    gates = {n for n, q in q0.items() if q.passed}
    assert "no_stray_compound" in gates and not gates & {"hole_0", "flush", "over_mesh"}
    g = _done(board)
    hole = ck.hole_masks(board)[0]
    g.h[hole] = board.base[hole] / 2                                         # hole 0 half full
    g.t[hole] = -board.base[hole] / 2
    v = ck.patch_drywall(g, start)
    lo = -ds.BOARD_T * 1e3                                                    # read at 1 mm depth: the board's back
    raw = (lo / 2 - lo) / (ck.HOLE_MM - lo)
    assert v.progress.quantities["hole_0"] == pytest.approx(0.9 * raw, rel=1e-12)
    assert v.progress.terms["holes"] == pytest.approx(0.9 * (raw + 3.0) / 4, rel=1e-12)    # the others: raw 1
    assert v.score == pytest.approx(0.95 * v.progress.value, rel=1e-12) and not v.success
    g = _done(board)
    far = ~board.near()
    g.t = np.where(far, 0.2e-3, g.t)                                          # compound all over beyond 150 mm
    g.h = g.h + np.where(far, 0.2e-3, 0.0)
    v = ck.patch_drywall(g, start)
    assert v.progress.gates["no_stray_compound"] == 0.0 and v.progress.value == 0.0


def test_the_strokes_never_span_the_mesh_edge() -> None:
    """A blade whose edge band straddles the mesh's edge is always in the air; one over the mesh rides its fibre
    tops; the joint knife clears the board first (below, above and beside the mesh), then the putty knife rides the
    mesh in a lane on each half, filling every hole between them; no move outruns the tools' speed."""
    task = dw.PatchDrywall()
    putty, joint = dw.strokes(task)
    assert len(putty) == len(joint)
    (cx, cz), h, b, eps = ds.PATCH, ds.MESH_HALF, task.band, 1e-9
    for knife, path in ((dw.PUTTY, putty), (dw.JOINT, joint)):
        for x, gap, z in path:
            if abs(x - cx) >= h + knife.half - eps:
                continue                                                      # wholly beside the mesh
            over, clear = abs(z - cz) <= h - b + eps, abs(z - cz) >= h + b - eps
            assert clear or (over and gap >= task.mesh_gap) or gap >= 0.5 * dw.PARK_GAP, (knife.name, x, gap, z)
        step = np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)
        assert step.max() <= dw.TOOL_SPEED * dw.CONTROL_DT + 1e-12
    working = lambda path: [k for k, p in enumerate(path) if p[1] < 1e-3]     # noqa: E731
    assert max(working(joint)) < min(working(putty))                          # the joint knife works first
    assert sorted({round(x, 9) for x, gap, _ in joint if gap < 1e-3}) == pytest.approx(
        [cx - h - dw.JOINT.half, cx, cx + h + dw.JOINT.half], abs=1e-12)
    lanes = [(x, gap) for x, gap, _ in putty if gap < 1e-3 and abs(x - cx) < h]
    assert sorted({round(x, 9) for x, _ in lanes}) == pytest.approx([cx - h + dw.PUTTY.half, cx + h - dw.PUTTY.half],
                                                                    abs=1e-12)
    assert all(gap == task.mesh_gap == ds.MESH_T for _, gap in lanes)        # on the fibre tops (GA's support)
    on = working(putty)
    for hx, hz in ds.HOLES:                                                   # between the two lanes
        k = next(k for k in on if abs(putty[k][0] - hx) < 1e-9 and abs(putty[k][2] - hz) < b)
        assert putty[k][1] == -task.press and putty[on[0]][0] < cx < putty[on[-1]][0]


def test_compound_off_the_board_counts() -> None:
    """A lifted knife's gob, drips and releases, spills and drains are off the board (GA: the floor's catch tiles,
    task.py:159-166); past 2e-8 m³ they fail no_stray_compound."""
    read = {"spilled_volume": np.array([1e-9, 0.0]), "drained_total_volume": np.array([2e-9, 0.0]),
            "tool_dropped": np.array([[3e-9, 4e-9], [0.0, 0.0]]), "tool_dripped": np.array([[5e-9, 0.0], [0.0, 0.0]]),
            "tool_released": np.array([[0.0, 6e-9], [0.0, 0.0]])}
    assert dw.off_board(read) == pytest.approx([21e-9, 0.0], rel=1e-12, abs=0.0)
    board = ds.on_grid(ds.wallboard(0), 0.005)
    g = _done(board)
    g.off_m3 = float(dw.off_board(read)[0])
    assert _failed(ck.patch_drywall(g)) == ["no_stray_compound"]


def test_open_books_void_the_verdict() -> None:
    """Each world's books gate its verdict at 1e-9: just past it no success, no score, no progress (unverifiable)."""
    board = ds.on_grid(ds.wallboard(0), 0.005)
    v = ck.patch_drywall(_done(board), _start(board))
    assert v.success and v.score == 1.0
    shut = dw._verdict_dict(v, 0.5e-9)
    assert shut["success"] and shut["verifiable"] and shut["score"] == 1.0 and shut["progress"] == v.progress.value
    for books in (2e-9, float("nan")):
        out = dw._verdict_dict(v, books)
        assert not out["success"] and not out["verifiable"], books
        assert out["score"] is None and out["progress"] is None and out["score_legacy"] is None
    assert dw.BOOKS_REL == 1e-9


def test_the_defaults_are_the_verdicts() -> None:
    """Their 1 mm grid and no uptake (GA has none) by default, in the task and on the command line; a coarser grid is
    labelled a smoke."""
    task, args = dw.PatchDrywall(), dw.parser().parse_args([])
    assert task.cell == args.cell == ds.CELL == 0.001 and task.uptake is False and args.uptake is False
    assert dw.grid_label(task.cell) == "theirs (1 mm): the verdict"
    assert all(dw.grid_label(c).startswith("smoke") for c in (0.002, 0.005, 0.01))
    assert dw.make_flow(task).config.process is None and dw.make_flow(dw.PatchDrywall(uptake=True)).config.process


def test_the_flow_reads_gas_support() -> None:
    """The flow's ``support`` input is GA's support height (paste.py:128-135): the fibre tops on the mesh, the base
    elsewhere, per world. It decides where a blade within 0.05 mm bears (its contact, and its fill's cap)."""
    boards = [ds.on_grid(ds.wallboard(seed), 0.01) for seed in (0, 1)]
    got = dw.inputs(boards)
    assert set(got) == {"base", "support"} and all(v.dtype == np.float32 for v in got.values())
    for w, b in enumerate(boards):
        assert np.array_equal(got["base"][w], b.base.astype(np.float32).ravel())
        want = np.where(b.mesh, ds.MESH_T, b.base).astype(np.float32).ravel()
        assert np.array_equal(got["support"][w], want) and (got["support"][w] == np.float32(ds.MESH_T)).sum() == 100


def test_the_mesh_holds_a_knife_at_its_fibre_tops() -> None:
    """GA's support height (paste.py:128-135): a pose whose edge band overlaps the mesh is lifted to 0.5 mm; one
    beside it, or the script's, is left as it is."""
    task = dw.PatchDrywall()
    (cx, cz), half = ds.PATCH, ds.MESH_HALF
    assert dw.supported(dw.PUTTY, (cx, -1e-5, cz), task.band) == (cx, ds.MESH_T, cz)
    assert dw.supported(dw.JOINT, (cx + half + dw.JOINT.half - 1e-3, 0.0, cz), task.band)[1] == ds.MESH_T
    edge = (cx, -1e-5, cz - half - task.band)                                # the band's edge on the mesh's
    assert dw.supported(dw.PUTTY, edge, task.band) == edge
    for knife, path in zip((dw.PUTTY, dw.JOINT), dw.strokes(task), strict=True):
        assert all(dw.supported(knife, p, task.band) == p for p in path)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_bare_mesh_openings_fill_to_the_fibre_tops(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry mesh cell (the fibres' volume, 0.15 mm, bare) under the putty knife riding the fibre tops fills to them,
    0.5 mm, as GA's knife does (coupling.py:454-458, 581): it guards the stores' fill rule (bare openings fill up to
    the knife), which the script needs; the ``support`` input is pinned by :func:`test_the_flow_reads_gas_support`.
    Their 1 mm grid starts with 56-140 such cells; a 10 mm grid with none, so two are made in each lane."""
    (cx, cz), real = ds.PATCH, ds.on_grid
    dry = [(cx + sx * 0.025, cz + sz * 0.02) for sx in (-1, 1) for sz in (-1, 1)]

    def bared(wb: ds.Wallboard, cell: float) -> ds.Wallboard:
        g = real(wb, cell)
        film = g.film.copy()
        for x, z in dry:
            film[np.unravel_index(np.argmin(np.hypot(g.x - x, g.z - z)), g.x.shape)] = 0.0
        return ds.Wallboard(**{**g.__dict__, "film": film})

    monkeypatch.setattr(ds, "on_grid", bared)
    task = dw.PatchDrywall(cell=0.01)
    board = task.board
    cells = [np.unravel_index(np.argmin(np.hypot(board.x - x, board.z - z)), board.x.shape) for x, z in dry]
    assert all(board.mesh[c] and board.film[c] == 0.0 for c in cells)
    assert ck.criteria(_start(board))[3].value >= 0.04                          # bare at the start: 4 of 100
    ends: list[ck.DrywallGrid] = []
    real_verdict = dw.patch_drywall
    monkeypatch.setattr(dw, "patch_drywall", lambda g, start=None: ends.append(g) or real_verdict(g, start))
    res = dw.run(task)
    v = res.verdicts[0]
    assert v["success"] and v["criteria"]["mesh_covered"]["value"] == 0.0, v["criteria"]
    assert [ends[0].h[c] for c in cells] == pytest.approx([ds.MESH_T] * len(cells), rel=0.0, abs=1e-6)
    assert res.books_rel[0] < 1e-12 and res.off_ml == [0.0]


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
@pytest.mark.parametrize("uptake", [False, True])
def test_scripted_patch_on_a_10mm_grid(uptake: bool) -> None:
    """A smoke on a 10 mm grid (theirs is 1 mm): every criterion passes, each world's books close, nothing leaves the
    board; with the paper's uptake on, it takes compound and the mesh still reads covered."""
    task = dw.PatchDrywall(cell=0.01, uptake=uptake)
    res = dw.run(task)
    v = res.verdicts[0]
    assert res.grid.startswith("smoke") and v["verifiable"] and v["contract"] == "progress-v1"
    assert v["success"] and v["score"] == 1.0 and v["score_legacy"] == 1.0, v["criteria"]
    c = v["criteria"]
    board, root = task.board, math.sqrt(res.sim_s) * task.sorptivity * uptake            # the most a cell can lose
    shallow = float(board.frac[board.holes].min())                       # read at 1 mm: the cell's level over its share
    tol = 2.0e-7 / shallow * 1e3                                          # float32 poses at 1.8 m: ~1.2e-7 m
    assert -(task.press + root) / shallow * 1e3 - tol <= c["holes_filled"]["value"] <= -task.press / shallow * 1e3 + tol
    assert c["over_mesh"]["value"] <= root * 1e3 + 1e-3 and c["mesh_covered"]["value"] == 0.0     # on the fibre tops
    assert res.books_rel[0] < 1e-12 and res.off_ml == [0.0]
    if uptake:
        wet = math.pi * 0.09 ** 2 + 4 * math.pi * 0.003 ** 2                            # the mound and the holes
        assert 0.0 < res.absorbed_ml[0] <= 1e6 * root * wet
    else:
        assert res.absorbed_ml == [0.0]
    assert 0.0 < res.carried_ml[0][0] < 1e6 * dw.PUTTY.carry and 0.0 < res.carried_ml[0][1] < 1e6 * dw.JOINT.carry
