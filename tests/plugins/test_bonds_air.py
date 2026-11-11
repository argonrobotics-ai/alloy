# SPDX-License-Identifier: Apache-2.0
"""Trapped air under a Bonds film (alloy/plugins/bonds/air.py, SCR-WP8-11 Part B2): the gaps, the relabel books
through splits, merges, vanishing pockets and overflow (WP13 must-fixes 3 and 5, law T-3), the formation guard. The
pocket laws on the coupled rig (T-1, T-2) are in laws_air.py; the slow tests below run them. CPU."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.bonds import air_kernels as K
from alloy.plugins.bonds.air import AIR_ROUGHNESS, READINGS, SLOT_STATE, AirField, books_error, ga_trapped_air
from alloy.plugins.bonds.layout import box_distance, vertex_areas, vertex_spacing, world_ranges
from alloy.plugins.bonds.rig import BOPP_60
from alloy.solvers.film.config import register_custom_attributes
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.rigs import grid_mesh

DX, N, GAP = 2.0e-3, 8, 1.0e-5          # 9 x 9 vertices, 10 µm above a box
H = GAP + AIR_ROUGHNESS                  # the gas gap
PA = float(K.P_ATM)


@wp.kernel
def _sdf(kind: int, h: wp.vec3, p: wp.array[wp.vec3], out: wp.array[wp.vec4]):
    i = wp.tid()
    out[i] = K.local_sdf(kind, h, p[i])


def _local(kind: int, h: tuple, pts: np.ndarray) -> np.ndarray:
    out = wp.zeros(len(pts), dtype=wp.vec4, device="cpu")
    wp.launch(_sdf, dim=len(pts), inputs=[kind, wp.vec3(*h), wp.array(pts.astype(np.float32), dtype=wp.vec3,
                                                                      device="cpu")], outputs=[out], device="cpu")
    return out.numpy().astype(np.float64)


def test_the_gap_functions_match_their_shapes():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.03, 0.03, (400, 3))
    r, hh = 0.01, 0.015
    cap = _local(int(K.CAPSULE), (r, hh, 0.0), pts)
    seg = np.stack([np.zeros(400), np.zeros(400), np.clip(pts[:, 2], -hh, hh)], 1)
    np.testing.assert_allclose(cap[:, 3], np.linalg.norm(pts - seg, axis=1) - r, atol=1e-7)
    cyl = _local(int(K.CYLINDER), (r, hh, 0.0), pts)
    rr, dz = np.hypot(pts[:, 0], pts[:, 1]) - r, np.abs(pts[:, 2]) - hh
    want = np.where((rr <= 0) & (dz <= 0), np.maximum(rr, dz), np.hypot(np.maximum(rr, 0), np.maximum(dz, 0)))
    np.testing.assert_allclose(cyl[:, 3], want, atol=1e-7)
    for got in (cap, cyl):
        np.testing.assert_allclose(np.linalg.norm(got[:, :3], axis=1), 1.0, atol=1e-5)


def _model(worlds: int = 1, gap: float = GAP) -> newton.Model:
    tmpl = newton.ModelBuilder()
    register_custom_attributes(tmpl)
    tmpl.add_shape_box(-1, xform=wp.transform(wp.vec3(0.5 * N * DX, 0.5 * N * DX, -0.005), wp.quat_identity()),
                       hx=0.02, hy=0.02, hz=0.005, label="plate")
    v, t = grid_mesh(N, N, DX)
    add_film_mesh(tmpl, v, t, BOPP_60, pos=(0.0, 0.0, BOPP_60.extras.contact_radius + gap), label="label")
    root = newton.ModelBuilder()
    register_custom_attributes(root)
    root.replicate(tmpl, worlds)
    root.color()
    return root.finalize(device="cpu")


def test_the_box_gap_matches_the_layout_distance():
    m = _model()
    x = m.particle_q.numpy().astype(np.float64)
    want, _ = box_distance(m, np.zeros((0, 7)), 0, x)
    got = _local(int(K.BOX), tuple(m.shape_scale.numpy()[0]), x - m.shape_transform.numpy()[0, :3])
    np.testing.assert_allclose(got[:, 3], want, atol=1e-7)


class Rig:
    """An AirField on a free-standing film grid 10 µm above a box (no solver): set bond states, relabel, read."""

    def __init__(self, worlds: int = 1, gap: float = GAP) -> None:
        self.m = m = _model(worlds, gap)
        n, W = int(m.particle_count), worlds
        x, tris = m.particle_q.numpy().astype(np.float64), m.tri_indices.numpy()
        self.arrays = {"bond_state": wp.zeros(n, dtype=wp.int32, device="cpu"),
                       "air_slot": wp.full(n, -1, dtype=wp.int32, device="cpu")}
        for name, t in SLOT_STATE:
            self.arrays[name] = wp.zeros((W, K.SLOTS), dtype=t, device="cpu")
        self.arrays["air_root"].fill_(-1)
        for name in READINGS:
            self.arrays[name] = wp.zeros(W, dtype=wp.float64, device="cpu")
        init = {"start": world_ranges(m.particle_world.numpy(), W), "area": vertex_areas(x, tris),
                "h": vertex_spacing(x, tris), "surface": {"margin": np.zeros(int(m.shape_count))}}
        subs = np.asarray(m.shape_world.numpy() >= 0).nonzero()[0].reshape(W, 1)
        self.field = AirField(m, self.arrays, init, subs)
        self.x0 = m.particle_q.numpy()
        self.state = SimpleNamespace(particle_q=wp.array(self.x0, dtype=wp.vec3, device="cpu"))
        self.area, self.gap = init["area"], gap

    def step(self, bonded: np.ndarray) -> dict[str, np.ndarray]:
        """Bond ``bonded`` (moved down into contact, as a bond forms) and relabel."""
        q = self.x0.copy()
        q[bonded == 1, 2] -= self.gap
        self.state.particle_q.assign(q)
        self.arrays["bond_state"].assign(bonded.astype(np.int32))
        self.field.after(self.state, wp.zeros(1, dtype=wp.transform, device="cpu"), 1.0e-3)
        return {k: v.numpy().copy() for k, v in self.arrays.items()}

    def volume(self, bonded: np.ndarray) -> np.ndarray:
        """Each vertex's gas volume with ``bonded`` in contact."""
        return self.area * (np.where(bonded == 1, 0.0, self.gap) + AIR_ROUGHNESS)


def _grid(cells: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((N + 1) * (N + 1), np.int32)
    for i, j in cells:
        out[i * (N + 1) + j] = 1
    return out


def _ring(r: int) -> list[tuple[int, int]]:
    c = N // 2
    return [(i, j) for i in range(N + 1) for j in range(N + 1) if max(abs(i - c), abs(j - c)) == r]


def _slots(d: dict) -> list[tuple[float, float, int]]:
    """(amount, area, enclosed) of world 0's active slots, by area."""
    act = d["air_root"][0] >= 0
    rows = zip(d["air_amount"][0][act], d["air_area"][0][act], d["air_enclosed"][0][act], strict=True)
    return sorted(rows, key=lambda r: -r[1])


def _vanished_is_roughness_air(rig: Rig, before: dict, after: dict, bonded: np.ndarray, p: float) -> None:
    """The air booked vanished in a step is the roughness layer's of the cells that bonded (WP13 should-fix 1)."""
    gone = after["air_vanished"][0] - before["air_vanished"][0]
    assert gone <= p * (AIR_ROUGHNESS + 0.05e-6) * rig.area[bonded].sum() * (1.0 + 1e-6)


def test_books_hold_through_split_merge_and_vanish():
    rig = Rig()
    c = N // 2
    inside = np.array([i * (N + 1) + j for i in range(c - 1, c + 2) for j in range(c - 1, c + 2)])
    row = np.array([c * (N + 1) + j for j in range(c - 1, c + 2)])
    ring = np.array([i * (N + 1) + j for i, j in _ring(2)])
    a9, a3 = rig.area[inside].sum(), rig.area[row].sum()
    d0 = rig.step(_grid([]))                                 # one open pocket, filled from the air
    assert d0["pockets"][0] == 1 and _slots(d0)[0][2] == 0
    np.testing.assert_allclose(d0["air_formed"][0], PA * H * rig.area.sum(), rtol=1e-4)     # float32 gaps
    a_all = _slots(d0)[0][0]
    b1 = _grid(_ring(2))
    d1 = rig.step(b1)                                        # a ring bonds (in contact): 9 cells enclosed
    v1 = rig.volume(b1)
    big, inner = _slots(d1)[:2]
    assert d1["pockets"][0] == 2 and inner[2] == 1 and big[2] == 0
    np.testing.assert_allclose(inner[1], a9, rtol=1e-6)
    np.testing.assert_allclose(inner[0], a_all * v1[inside].sum() / v1.sum(), rtol=1e-4)
    np.testing.assert_allclose(d1["air_vanished"][0], a_all * v1[ring].sum() / v1.sum(), rtol=1e-4)
    _vanished_is_roughness_air(rig, d0, d1, ring, a_all / v1.sum())
    np.testing.assert_allclose(d1["trapped_area"][0], a9, rtol=1e-6)
    a_in = inner[0]
    b2 = _grid(_ring(2) + [(c, c - 1), (c, c), (c, c + 1)])
    d2 = rig.step(b2)                                        # a bonded row splits it, shared by volume
    halves = [s for s in _slots(d2) if s[2] == 1]
    assert len(halves) == 2 and d2["pockets"][0] == 3
    v2 = rig.volume(b2)
    shares = v2[inside].reshape(3, 3)[[0, 2]].sum(axis=1) / v2[inside].sum() * a_in
    np.testing.assert_allclose(sorted(h[0] for h in halves), sorted(shares), rtol=1e-4)
    _vanished_is_roughness_air(rig, d1, d2, row, a_in / v2[inside].sum())
    a_in = sum(h[0] for h in halves)
    d3 = rig.step(b1)                                        # the row lets go: the halves merge, the row draws air
    merged = [s for s in _slots(d3) if s[2] == 1]
    np.testing.assert_allclose(merged[0][0], a_in + PA * H * a3, rtol=1e-4)
    b4 = _grid(_ring(2) + [(i, j) for i in range(c - 1, c + 2) for j in range(c - 1, c + 2)])
    d4 = rig.step(b4)                                        # the inner pocket bonds away (forced): booked
    assert d4["pockets"][0] == 1 and d4["air_vanished"][0] > d3["air_vanished"][0]
    for _ in range(3):
        d4 = rig.step(b1)
    assert abs(books_error(d4)[0]) < 1e-13


def test_more_pockets_than_slots_are_counted_and_their_air_booked():
    rig = Rig()
    free = {(i, j) for i in range(1, N, 2) for j in range(1, N, 2)} | {(1, 2)}     # 15 pockets, one of 3 cells
    bonded = [(i, j) for i in range(N + 1) for j in range(N + 1) if (i, j) not in free]
    d = rig.step(_grid(bonded))
    assert d["pockets"][0] == 15 and d["air_overflow"][0] == 15 - K.SLOTS
    assert d["air_untracked"][0] > 0.0 and abs(books_error(d)[0]) < 1e-13
    np.testing.assert_allclose(d["trapped_area"][0], rig.area[[N + 2, N + 3, N + 4]].sum(), rtol=1e-6)
    assert (d["air_root"][0] >= 0).sum() == K.SLOTS and bool(ga_trapped_air(d["trapped_area"])[0])


def test_an_enclosed_pocket_holding_air_keeps_its_cells_free():
    rig = Rig()
    d = rig.step(_grid(_ring(2)))
    key = wp.full(rig.m.particle_count, 7, dtype=wp.int64, device="cpu")
    rig.field.block(key)
    kept = key.numpy() > 0
    inside = d["air_slot"] == next(k for k in range(K.SLOTS) if d["air_enclosed"][0, k] == 1)
    assert not kept[inside].any() and kept[~inside].all()


def test_roughness_air_is_not_trapped_and_does_not_block():
    """A label in contact (gap 0) holds only its roughness layer's air: enclosed, it is not trapped air, its cells may
    bond, and the film feels no push from it (air_kernels.held)."""
    rig = Rig(gap=0.0)
    d = rig.step(_grid(_ring(2)))
    inner = [s for s in _slots(d) if s[2] == 1]
    assert len(inner) == 1 and d["trapped_area"][0] == 0.0
    key = wp.full(rig.m.particle_count, 7, dtype=wp.int64, device="cpu")
    rig.field.block(key)
    assert np.all(key.numpy() == 7)


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("together", [False, True])
def test_an_edges_first_application_traps_air_and_fails_ga(together):
    """T-4 (WP13 review of B2): the frame pad presses the 3 mm edge band first; it bonds and encloses the centre's
    air (197 mm²), which the centre pad then only compresses: GA's trapped-air criterion fails. Both pads together
    bond the whole label (0 mm²). The air booked vanished per period is the bonded cells' roughness layer's."""
    from alloy.plugins.bonds.laws_air import edge_application

    out = edge_application(together)
    rows = out["periods"]
    v, t = grid_mesh(10, 10, DX)
    area = vertex_areas(v, t)
    if together:
        assert out["trapped"][0] == 0.0 and bool(out["ga_pass"][0]) and np.all(rows[-1]["state"][0] == 1)
    else:
        assert out["trapped"][0] > 150e-6 and not bool(out["ga_pass"][0])
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        new = (b["state"][0] == 1) & (a["state"][0] != 1)
        gone = b["vanished"][0] - a["vanished"][0]
        assert gone <= max(a["p_max"][0], b["p_max"][0], PA) * (AIR_ROUGHNESS + 0.05e-6) * area[new].sum() + 1e-15
        assert abs(b["books"][0]) < 1e-12


@pytest.mark.parametrize("worlds", [2])
def test_worlds_keep_their_own_pockets(worlds):
    rig = Rig(worlds)
    one = _grid(_ring(2))
    d = rig.step(np.concatenate([one, np.zeros_like(one)]))
    assert d["pockets"].tolist() == [2.0, 1.0] and np.all(np.abs(books_error(d)) < 1e-13)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["T-1_boyle", "T-1_free_dt0.5", "T-1_free_dt1", "T-2_vent"])
def test_the_air_laws_on_the_pocket_rig(name):
    """Rows (CPU, 1 world, 30-60 s each):
    - T-1_boyle: P·(depth + roughness - rise) within 5.4e-5 while P doubles (107.9 -> 202.2 kPa) at the 0.5 ms step;
    - T-1_free: the free label's volume swing 6e-8 (0.5 ms) and 1.1e-7 (1 ms) of V after a 10 % squeeze;
    - T-2_vent: the excess pressure follows the slot ODE within 0.08 % while it decays to 0.32."""
    from alloy.plugins.bonds.laws_air import air_laws
    from alloy.plugins.bonds.plugin import Bonds
    from alloy.plugins.harness.run import run_scene

    law = next(x for x in air_laws() if x.name == name)
    run = run_scene(law.scene, [Bonds(**dict(law.config))], "cpu")
    assert run.error is None, run.error
    res = law.check(run)
    assert res.passed, res


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_harness_checks_with_trapped_air():
    """Contract, crash, hooks, determinism, identical and isolated worlds and graph replay with pockets on: the
    venting pocket rig, two worlds (the pressure force, relabels and the core's vent step all run)."""
    import alloy.plugins as ap
    from alloy.plugins.bonds.laws import scene
    from alloy.plugins.bonds.laws_air import VENT, air_config
    from alloy.plugins.bonds.plugin import Bonds

    class AirBonds(Bonds):
        """Bonds with trapped air (units as Bonds), checked on the venting pocket rig. Example: AirBonds(**cfg)."""

        def test_scene(self):
            return scene(VENT, "pull180", 6, worlds=2, speed=0.0, lift=1.0e-6, t_lift=0.05)

    report = ap.check(AirBonds(**air_config(VENT)), laws=False, speed=False)
    assert report.passed, report.summary()
