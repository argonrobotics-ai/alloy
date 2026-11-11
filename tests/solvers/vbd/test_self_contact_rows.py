# SPDX-License-Identifier: Apache-2.0
"""GAP-13B laws: VBD self-contact rows sized to the scene (``docs/known_gaps/vbd-self-contact-buffer-overflow.md``).

Newton keeps each vertex's colliding triangles and each edge's colliding edges in rows of 32 and 64 pairs and drops
the rest. The fix (``alloy.solvers.vbd.self_contact_rows``, opt-in ``alloy_self_contact_rows="scene"``, the module's
``ROWS_DEFAULT``) gives each row its own size, grows rows past 75 % full before a drop, and re-runs a period that
dropped pairs.

- R1 no drop: ten fine sheets stacked 1 mm apart (public API, defaults). With "fixed" rows it overflows (peak counts
  over 32 / 64, SELF_CONTACT_OVERFLOW); with "scene" rows after EVERY step every row's count <= its size and the flag is
  absent. (Bound: a pair beyond a row is never resolved.)
- R2 growth rule: ``grown_sizes`` never shrinks, holds the need, stays under the cap, multiples of 8 when grown; the
  brute-force cross-layer count of N coincident flat grids (the least a row must hold; in-layer pairs inside the reach
  are the one-ring neighbours the filter drops) exceeds 32 / 64, and one scene-rows step sizes the rows above it.
- R3 same physics: a scene that never passes 75 % fill steps bit for bit as with "fixed" rows; no array replaced.
- R4 memory: ``RowState.bytes()`` is 8 x the sum of the buffer sizes and the nbytes of the two data arrays; a tiny
  ``alloy_self_contact_row_bytes`` exhausts the state, logs one warning, and pairs drop as before (health flag source).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
import warp as wp

import alloy
from alloy._vendor import newton
from alloy.integration.stepper import entry_solvers
from alloy.precision import self_contact
from alloy.solvers.vbd import make_vbd
from alloy.solvers.vbd import self_contact_rows as rows

NEWTON_ROWS = (32, 64)                     # Newton's vertex / edge row defaults
H = 1.0e-3                                 # [m] contact margin of the analytic rigs
LONG = pytest.mark.timeout(1800)           # cold VBD kernels take minutes on the CPU; the host is loaded


def _vbd(sim: Any) -> newton.solvers.SolverVBD:
    [v] = [s for s in entry_solvers(sim.engine.stepper).values() if isinstance(s, newton.solvers.SolverVBD)]
    return v


def _over(v: newton.solvers.SolverVBD) -> tuple[int, int, bool]:
    """(peak vertex count, peak edge count, whether any row's count exceeds its own size), host-read."""
    ci = v.trimesh_collision_info
    cv, ce = ci.vertex_colliding_triangles_count.numpy(), ci.edge_colliding_edges_count.numpy()
    sv, se = ci.vertex_colliding_triangles_buffer_sizes.numpy(), ci.edge_colliding_edges_buffer_sizes.numpy()
    return int(cv.max()), int(ce.max()), bool((cv > sv).any() or (ce > se).any())


def probe(sim: Any, steps: int) -> dict[str, Any]:
    """Step ``steps`` times; the peak count per family, whether any row exceeded its size after any step, the health
    flags and the number of period re-runs."""
    v, pv, pe, over = _vbd(sim), 0, 0, False
    for _ in range(steps):
        sim.step(1)
        cv, ce, o = _over(v)
        pv, pe, over = max(pv, cv), max(pe, ce), over or o
    flags = sorted({f for w in sim.health()["worlds"] for f in w["flags"]})
    return {"peak_vertex": pv, "peak_edge": pe, "over": over, "flags": flags,
            "regrows": getattr(sim.engine, "row_regrows", 0)}


def build_with_mode(scene: alloy.Scene, mode: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv(rows.ROWS_ENV, raising=False)       # the environment wins over the default (AUDIT-STACK)
    monkeypatch.setattr(rows, "ROWS_DEFAULT", mode)
    return alloy.build(scene, device="cpu")


def stacked_sheets(layers: int, resolution: int, gap: float = 0.004) -> alloy.Scene:
    """Public API, defaults: ``layers`` fine 0.3 m sheets a few mm apart on a table."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.5, 0.5, 0.4), pos=(0, 0, 0.2), static=True)
    for i in range(layers):
        scene.add_cloth(f"c{i}", size=(0.3, 0.3), pos=(0, 0, 0.404 + gap * i), resolution=resolution, mass=0.05)
    return scene


# ---------------------------------------------------------------- R1
R1_LAYERS, R1_RES, R1_GAP, R1_STEPS = 10, 10, 0.001, 2     # 1 mm apart, inside the 2 mm margin: a pile


@LONG
@pytest.mark.slow
def test_r1_fixed_rows_overflow_the_pile(monkeypatch) -> None:
    out = probe(build_with_mode(stacked_sheets(R1_LAYERS, R1_RES, R1_GAP), "fixed", monkeypatch), R1_STEPS)
    assert out["peak_vertex"] > NEWTON_ROWS[0] and out["peak_edge"] > NEWTON_ROWS[1], out
    assert out["over"] and "SELF_CONTACT_OVERFLOW" in out["flags"], out


@LONG
@pytest.mark.slow
def test_r1_scene_rows_never_drop_a_pair(monkeypatch) -> None:
    sim = build_with_mode(stacked_sheets(R1_LAYERS, R1_RES, R1_GAP), "scene", monkeypatch)
    assert _vbd(sim).alloy_rows is not None
    out = probe(sim, R1_STEPS)
    assert not out["over"] and "SELF_CONTACT_OVERFLOW" not in out["flags"], out
    assert out["peak_vertex"] > NEWTON_ROWS[0], out                  # the rows really held more than Newton's 32


# ---------------------------------------------------------------- R2
@pytest.mark.parametrize("need, size", [([0, 10, 30, 31, 200, 250], [32] * 6), ([5, 49, 70, 300], [32, 64, 64, 256])])
def test_r2_grown_sizes_never_shrink_and_hold_the_need(need, size) -> None:
    need, size = np.asarray(need), np.asarray(size)
    out = rows.grown_sizes(need, size)
    assert (out >= size).all() and (out <= np.maximum(size, rows.ROW_CAP)).all()
    grown = out > size
    assert (out[grown] % 8 == 0).all() and (out[grown] >= np.minimum(need[grown], rows.ROW_CAP)).all()
    kept_need, kept_size = need[~grown], size[~grown]
    assert (out[~grown] == kept_size).all()
    assert ((kept_need <= rows.GROW_FILL * kept_size) | (kept_size >= rows.ROW_CAP)).all()


def _layer_model(layers: int, dim: int, cell: float, spacing: float) -> newton.Model:
    b = newton.ModelBuilder()
    for i in range(layers):
        b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, spacing * i), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=dim,
                         dim_y=dim, cell_x=cell, cell_y=cell, mass=1.0e-3, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0e-2,
                         particle_radius=H / 2)
    b.color()
    return b.finalize(device="cpu")


def _dist_to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D distance of points p [P,2] to segments a-b [S,2] -> [P,S]."""
    ab = (b - a)[None]
    t = np.clip(((p[:, None] - a[None]) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
    return np.linalg.norm(p[:, None] - (a[None] + t[..., None] * ab), axis=-1)


def _cross(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    return u[..., 0] * w[..., 1] - u[..., 1] * w[..., 0]


def _point_tri_2d(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """2D distance of points [P,2] to triangles [T,3,2] -> [P,T] (0 inside)."""
    d = np.min([_dist_to_segments(p, tri[:, i], tri[:, (i + 1) % 3]) for i in range(3)], axis=0)
    s = np.stack([_cross(tri[None, :, (i + 1) % 3] - tri[None, :, i], p[:, None] - tri[None, :, i]) for i in range(3)])
    inside = (s >= 0).all(0) | (s <= 0).all(0)
    return np.where(inside, 0.0, d)


def _seg_seg_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D distance between segments a [E,2,2] and b [E,2,2] -> [E,E] (0 when they cross)."""
    d = np.min([_dist_to_segments(a[:, i], b[:, 0], b[:, 1]) for i in (0, 1)]
               + [_dist_to_segments(b[:, i], a[:, 0], a[:, 1]).T for i in (0, 1)], axis=0)
    a0, a1, b0, b1 = a[:, None, 0], a[:, None, 1], b[None, :, 0], b[None, :, 1]
    s1, s2 = np.sign(_cross(a1 - a0, b0 - a0)) * np.sign(_cross(a1 - a0, b1 - a0)), \
        np.sign(_cross(b1 - b0, a0 - b0)) * np.sign(_cross(b1 - b0, a1 - b0))
    return np.where((s1 < 0) & (s2 < 0), 0.0, d)


def brute_force_peaks(model: newton.Model, layer_of: np.ndarray, reach: float) -> tuple[int, int]:
    """Most triangles within ``reach`` of one vertex, most edges within ``reach`` of one edge, over other layers.
    Layers are flat and parallel (z only differs), so a 3D distance is sqrt(dz^2 + planar distance^2)."""
    q, tri, edge = model.particle_q.numpy().astype(np.float64), model.tri_indices.numpy(), model.edge_indices.numpy()
    ev = edge[:, 2:4]
    dzv = np.abs(q[:, 2][:, None] - q[tri[:, 0], 2][None])                       # [P,T]
    dv = np.hypot(dzv, _point_tri_2d(q[:, :2], q[tri][..., :2]))
    other_v = layer_of[:, None] != layer_of[tri[:, 0]][None]
    dze = np.abs(q[ev[:, 0], 2][:, None] - q[ev[:, 0], 2][None])
    de = np.hypot(dze, _seg_seg_2d(q[ev][..., :2], q[ev][..., :2]))
    other_e = layer_of[ev[:, 0]][:, None] != layer_of[ev[:, 0]][None]
    return int(((dv <= reach) & other_v).sum(1).max()), int(((de <= reach) & other_e).sum(1).max())


R2_CASES = [(6, 8, 0.01, 0.002, 0.0075), (8, 8, 0.01, 0.001, 0.0075)]    # layers, dim, cell, spacing, reach


@LONG
@pytest.mark.parametrize("layers, dim, cell, spacing, reach", R2_CASES)
def test_r2_layers_need_more_than_newtons_rows_and_scene_rows_hold_them(layers, dim, cell, spacing, reach) -> None:
    m = _layer_model(layers, dim, cell, spacing)
    layer_of = np.repeat(np.arange(layers), m.particle_count // layers)
    want_v, want_e = brute_force_peaks(m, layer_of, reach)
    assert want_v > NEWTON_ROWS[0] and want_e > NEWTON_ROWS[1], (want_v, want_e)
    s = make_vbd(m, {"iterations": 2, "particle_enable_self_contact": True, "particle_self_contact_margin": H,
                     "particle_self_contact_gap": reach - H, rows.ROWS_KEY: rows.ROWS_SCENE})
    s0, s1 = m.state(), m.state()
    s.step(s0, s1, m.control(), None, 1.25e-3)
    ci = s.trimesh_collision_info
    assert int(ci.vertex_colliding_triangles_buffer_sizes.numpy().max()) >= want_v
    assert int(ci.edge_colliding_edges_buffer_sizes.numpy().max()) >= want_e
    assert not _over(s)[2]


# ---------------------------------------------------------------- R3
def _two_sheet_scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("a", size=(0.2, 0.2), pos=(0, 0, 0.003), resolution=10, mass=0.02)
    scene.add_cloth("b", size=(0.2, 0.2), pos=(0, 0, 0.0065), resolution=10, mass=0.02)
    return scene


def _run_mode(mode: str, steps: int, monkeypatch) -> tuple[np.ndarray, Any, Any, dict[str, Any]]:
    sim = build_with_mode(_two_sheet_scene(), mode, monkeypatch)
    ci = _vbd(sim).trimesh_collision_info
    ids = (ci.vertex_colliding_triangles, ci.edge_colliding_edges)
    out = probe(sim, steps)
    after = (ci.vertex_colliding_triangles, ci.edge_colliding_edges)
    same_arrays = all(a is b for a, b in zip(ids, after))
    pts = np.concatenate([sim.state(world=0)[f"{n}.points"].ravel() for n in ("a", "b")])
    return pts, _vbd(sim), same_arrays, out


@LONG
@pytest.mark.slow
def test_r3_a_scene_under_the_fill_mark_steps_as_with_fixed_rows(monkeypatch) -> None:
    fixed, _, _, out_f = _run_mode("fixed", 5, monkeypatch)
    scene, v, same_arrays, out_s = _run_mode("scene", 5, monkeypatch)
    assert out_f["peak_vertex"] > 0 and out_f["peak_vertex"] <= rows.FILL * NEWTON_ROWS[0], out_f
    assert out_f["peak_edge"] <= rows.FILL * NEWTON_ROWS[1], out_f
    assert v.alloy_rows is not None and v.alloy_rows.grown == [] and same_arrays and out_s["regrows"] == 0
    np.testing.assert_array_equal(fixed, scene)


# ---------------------------------------------------------------- R4
def _layer_solver(extra: dict[str, Any]) -> tuple[newton.Model, newton.solvers.SolverVBD]:
    m = _layer_model(8, 8, 0.01, 0.001)
    kw = {"iterations": 2, "particle_enable_self_contact": True, "particle_self_contact_margin": H,
          "particle_self_contact_gap": 0.0065, rows.ROWS_KEY: rows.ROWS_SCENE, **extra}
    return m, make_vbd(m, kw)


def _step(m: newton.Model, s: Any) -> None:
    s.step(m.state(), m.state(), m.control(), None, 1.25e-3)


def test_r4_bytes_match_the_allocated_rows() -> None:
    m, s = _layer_solver({})
    _step(m, s)
    ci, st = s.trimesh_collision_info, s.alloy_rows
    data = 4 * (ci.vertex_colliding_triangles.size + ci.edge_colliding_edges.size)   # int32
    sizes = (ci.vertex_colliding_triangles_buffer_sizes.numpy().sum()
             + ci.edge_colliding_edges_buffer_sizes.numpy().sum())
    assert st.grown and st.bytes() == 8 * int(sizes) == data


def test_r4_a_tiny_byte_cap_exhausts_warns_once_and_pairs_still_drop(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="alloy.solvers.vbd"):
        m, s = _layer_solver({rows.BYTES_KEY: 1024})
        watch = self_contact.watch(s, 1)                              # health's source for SELF_CONTACT_OVERFLOW
        _step(m, s)
        _step(m, s)
    st = s.alloy_rows
    assert st.exhausted and st.grown == [] and st.bytes() > 1024
    warns = [r for r in caplog.records if "self-contact rows" in r.getMessage() and r.levelno == logging.WARNING]
    assert len(warns) == 1 and rows.BYTES_KEY in warns[0].getMessage()
    assert watch is not None and _over(s)[2] and int(watch.dropped.numpy().max()) > 0
