# SPDX-License-Identifier: Apache-2.0
"""Herschel-Bulkley on the MPM paste route (docs/notes/viscoplastic-design.md §2-§3): the card fields, the viscosity
kernel, the solver swap, that the written viscosity is the array the solve reads and that it acts, and the film laws
L4 (Bingham plug speed) and L5 (power-law index) on CPU."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp
from alloy._vendor.newton.solvers import SolverImplicitMPM

from alloy.cards.library import CardLibrary
from alloy.solvers.mpm_rheology import laws as L
from alloy.solvers.mpm_rheology.kernels import (gather_strain_rates, herschel_bulkley_viscosity,
                                                herschel_bulkley_viscosity_smoothed)
from alloy.solvers.mpm_rheology.solver import SolverRheologyMPM

HB = L.Paste(1100.0, 0.0, 2.51, 10.0, 0.4)          # power-law paste, K 10 Pa s^n, n 0.4 (2.51 Pa s at 10 1/s)


def _material_issues(tmp_path, paste: L.Paste, drop: str = "") -> list:
    """Issues of a law card pair for ``paste``; ``drop`` removes the material line holding that key."""
    L.write_cards(tmp_path, "law_hb", paste, L.block_shape((0.02, 0.02, 0.02)), 0.005)
    p = tmp_path / "materials" / "law_hb_mat.yaml"
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    p.write_text("".join(x for x in lines if not (drop and x.strip().startswith(drop + ":"))), encoding="utf-8")
    return CardLibrary(roots=[str(tmp_path)], assets=None).object_issues("law_hb")


def test_kernel_sets_eta_from_the_step_start_shear_rate() -> None:
    dev = "cpu"
    sheared = np.zeros((3, 3), np.float32)
    sheared[0, 2] = 4.0                                     # u_x = 4 z: simple shear, gdot = 4 1/s
    still = np.zeros((3, 3), np.float32)
    grads = wp.array(np.stack([sheared, still, sheared, still]), dtype=wp.mat33, device=dev)
    k = wp.array([10.0] * 4, dtype=float, device=dev)
    n = wp.array([0.5, 0.5, 1.2, 1.2], dtype=float, device=dev)            # thinning, then thickening
    eta = wp.zeros(4, dtype=float, device=dev)
    wp.launch(herschel_bulkley_viscosity, dim=4, inputs=[wp.array([0, 1, 2, 3], dtype=wp.int32, device=dev), grads, k,
                                                         n, 1e-2, 1e-3], outputs=[eta], device=dev)
    assert eta.numpy() == pytest.approx([10.0 * 4.0 ** -0.5, 10.0 * 1e-2 ** -0.5,              # still: thinning's cap
                                         10.0 * 4.0 ** 0.2, 10.0 * 1e-2 ** 0.2], rel=1e-5)     # still: thickening's floor


def _noisy_shear_worlds(rates: tuple[float, ...], noise: float, seed: int = 0):
    """Particles on a 2 mm lattice in a 40 mm cube, one copy per world at the same positions, each in simple shear
    u_x = rate z plus Gaussian noise of ``noise`` 1/s in every gradient component."""
    rng = np.random.default_rng(seed)
    g = np.arange(0.001, 0.040, 0.002)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    m = len(pts)
    q, grads, world = [], [], []
    for w, r in enumerate(rates):
        grad = rng.normal(0.0, noise, (m, 3, 3))
        grad[:, 0, 2] += r
        q.append(pts)
        grads.append(grad)
        world.append(np.full(m, w))
    return (np.concatenate(q).astype(np.float32), np.concatenate(grads).astype(np.float32),
            np.concatenate(world).astype(np.int32), pts)


def test_smoothed_eta_cancels_gradient_noise_within_each_world() -> None:
    """Two worlds overlapping in space, in simple shear at 0.5 and 2 1/s, each with 1 1/s of noise in every gradient
    component (SOFT-2's probe: 0.5-2 1/s in the films). Each particle's own strain rate reads ~3 1/s whatever the shear,
    so eta falls to ~K 3^(n-1) in both; averaged over 7.5 mm in its own world, eta lands within 15 % of K rate^(n-1)
    in the interior, and the worlds do not mix: world 0's eta is bitwise the same with world 1 redrawn."""
    rates, k, n = (0.5, 2.0), 10.0, 0.4
    q, grad, world, pts = _noisy_shear_worlds(rates, 1.0)
    N = len(q)
    dev = "cpu"
    idx = wp.array(np.arange(N, dtype=np.int32), device=dev)
    K, Nn = wp.full(N, k, dtype=float, device=dev), wp.full(N, n, dtype=float, device=dev)
    wq, wg, ww = (wp.array(q, dtype=wp.vec3, device=dev), wp.array(grad, dtype=wp.mat33, device=dev),
                  wp.array(world, dtype=wp.int32, device=dev))
    raw = wp.zeros(N, dtype=float, device=dev)
    wp.launch(herschel_bulkley_viscosity, dim=N, inputs=[idx, wg, K, Nn, 1e-2, 1e-3], outputs=[raw], device=dev)
    points, dv, groups = (wp.zeros(N, dtype=wp.vec3, device=dev), wp.zeros(N, dtype=wp.mat33, device=dev),
                          wp.zeros(N, dtype=wp.int32, device=dev))
    wp.launch(gather_strain_rates, dim=N, inputs=[idx, wq, wg, ww], outputs=[points, dv, groups], device=dev)
    grid = wp.HashGrid(64, 64, 64, device=dev)
    grid.build(points, 0.0075, groups=groups)
    sm = wp.zeros(N, dtype=float, device=dev)
    wp.launch(herschel_bulkley_viscosity_smoothed, dim=N,
              inputs=[grid.id, idx, points, dv, groups, 0.0075, K, Nn, 1e-2, 1e-3], outputs=[sm], device=dev)
    inner = np.all((pts > 0.0105) & (pts < 0.0295), axis=1)
    m = len(pts)
    for w, r in enumerate(rates):
        want = k * r ** (n - 1.0)
        got, own = sm.numpy()[w * m:(w + 1) * m][inner], raw.numpy()[w * m:(w + 1) * m][inner]
        assert np.median(got) == pytest.approx(want, rel=0.15), (r, np.median(got), want)
        if r < 1.0:                                                         # the noise thins the slow world
            assert np.median(own) < 0.8 * want, (r, np.median(own), want)
    q2, grad2, _, _ = _noisy_shear_worlds((0.5, 3.0), 1.0, seed=0)          # world 1 redrawn at 3 1/s
    grad2[:m] = grad[:m]                                                    # world 0 exactly as before
    wp.launch(gather_strain_rates, dim=N, inputs=[idx, wq, wp.array(grad2, dtype=wp.mat33, device=dev), ww],
              outputs=[points, dv, groups], device=dev)
    grid.build(points, 0.0075, groups=groups)
    sm2 = wp.zeros(N, dtype=float, device=dev)
    wp.launch(herschel_bulkley_viscosity_smoothed, dim=N,
              inputs=[grid.id, idx, points, dv, groups, 0.0075, K, Nn, 1e-2, 1e-3], outputs=[sm2], device=dev)
    assert np.array_equal(sm2.numpy()[:m], sm.numpy()[:m])                  # world 0 bitwise: worlds independent


def test_card_needs_both_fields_and_a_sane_flow_index(tmp_path, caplog) -> None:
    def errs(issues):                                   # material errors come back under "material", path in the text
        return [f"{i.field}: {i.message}" for i in issues if i.level == "error"]

    assert errs(_material_issues(tmp_path / "ok", HB)) == []
    too_high = L.Paste(1100.0, 0.0, 2.51, 10.0, 2.0)
    assert any("fluid.flow_index" in e for e in errs(_material_issues(tmp_path / "n", too_high)))
    assert any("fluid.flow_index" in e for e in errs(_material_issues(tmp_path / "k", HB, drop="flow_index")))

    def sph_warned(paste, root):                        # a thin power-law fluid routes to SPH, which ignores K and n
        caplog.clear()
        _material_issues(tmp_path / root, paste)        # material warnings are logged when the card loads
        return any("routes to SPH" in r.getMessage() for r in caplog.records)

    assert sph_warned(L.Paste(1040.0, 0.0, 0.9, 2.5, 0.55), "thin")
    assert not sph_warned(L.Paste(1040.0, 11.7, 0.9, 2.5, 0.55), "yields")  # a yield stress routes it to MPM anyway


def _solver(sim):
    mpm = [s for s in sim.engine.solvers if isinstance(s, SolverImplicitMPM)]
    assert len(mpm) == 1
    return mpm[0]


@pytest.mark.compile_heavy
def test_solver_swaps_in_only_for_herschel_bulkley_particles_and_its_viscosity_acts(tmp_path) -> None:
    import alloy  # noqa: PLC0415 - heavy

    def build(root, paste):
        L.write_cards(root, "law_hb", paste, L.block_shape((0.04, 0.04, 0.03)), 0.005)
        scene = alloy.Scene(roots=[str(root)])
        scene.add_ground(friction=L.ROUGH)
        scene.add_object("law_hb", "law_hb")
        return alloy.build(scene, device="cpu")

    bingham = build(tmp_path / "b", L.Paste(1100.0, 20.0, 2.0))
    assert type(_solver(bingham)) is SolverImplicitMPM                       # no HB particle: the stock solver
    tops = {}
    for k in (10.0, 40.0):
        sim = build(tmp_path / f"k{k:g}", L.Paste(1100.0, 0.0, 2.51, k, 0.4))
        s = _solver(sim)
        assert isinstance(s, SolverRheologyMPM)
        assert s._visc is s._mpm_model.material_parameters.viscosity          # the array the rheology solve reads
        v0 = s._visc.numpy().copy()
        pts = sim.run(0.1, record=["law_hb"], every=10).world(0)["law_hb.points"]
        assert not np.allclose(s._visc.numpy(), v0)                           # rewritten from the shear rate
        tops[k] = float(np.percentile(pts[-1][:, 2], 99))
    assert tops[40.0] > tops[10.0] + 1e-4                                     # a stiffer K slumps less: it acts


@pytest.mark.compile_heavy
@pytest.mark.timeout(420)                               # ~4 min of CPU: 11,520 particles for 0.8 s
def test_l4_bingham_film_speed_on_cpu(tmp_path) -> None:
    """tau_y 50 Pa, 50 Pa s, 30 deg, 30 mm: depth-averaged speed 1.055 of analytic at its own thickness on the dense
    grid (0.995 on the sparse grid, whose floor offset slows the base; 5 mm voxel, 200 x 30 mm layer, 0.8 s)."""
    row = L.film(tmp_path, L.Paste(1100.0, 50.0, 50.0), thickness=0.03, theta_deg=30.0, length=0.2, width=0.03,
                 seconds=0.8, window=(0.4, 0.8), device="cpu")
    assert row["check_ok"] and 0.9 <= row["ratio"] <= 1.1, row


def _inv_n(rows: list[dict], thetas: tuple[float, float], n: float) -> float:
    """Slope exponent of the speed between two films, each brought to their mean thickness (speed ~ h^(1 + 1/n))."""
    h = 0.5 * (rows[0]["h_window_mm"] + rows[1]["h_window_mm"])
    u = [r["speed_mm_s"] * (h / r["h_window_mm"]) ** (1.0 + 1.0 / n) for r in rows]
    return math.log(u[1] / u[0]) / math.log(math.sin(math.radians(thetas[1])) / math.sin(math.radians(thetas[0])))


@pytest.mark.gpu
def test_l5_power_law_index_from_two_slopes_on_gpu(tmp_path) -> None:
    """K 10 Pa s^n, n 0.4 films on 3 and 4.5 deg slopes: the slope exponent of the speed is 1/n within 15 %. The film
    is 600 mm long: a 200 mm released film stretches through its middle third (0.02-0.15 1/s), which is real flow
    and thins an n 0.4 paste's plug, so it does not measure the steady shear law. CPU at this voxel: 0.989 / 0.878 of
    analytic, 1/n 2.21 (5 mm: 0.993 / 0.864, 2.16); the per-particle eta before the smoothing gave 1.60."""
    thetas = (3.0, 4.5)
    rows = [L.film(tmp_path / f"t{t}", HB, thickness=0.02, theta_deg=t, length=0.6, voxel=0.0025, device="cuda:0")
            for t in thetas]
    assert all(r["check_ok"] for r in rows)
    assert _inv_n(rows, thetas, HB.flow_index) == pytest.approx(1.0 / HB.flow_index, rel=0.15), rows


def test_no_library_material_both_shear_thins_and_melts() -> None:
    """The thermal plug-in writes a melting material's MPM viscosity before each step and the Herschel-Bulkley kernel
    rewrites it inside the step, so a material with both would silently lose its melt law: none may have both yet."""
    from alloy.plugins.thermal.phase_materials import MATERIALS  # noqa: PLC0415 - plug-in table

    lib = CardLibrary(assets=None)
    names = set(lib.names("materials"))
    both = [m for m, t in MATERIALS.items()
            if t.flows and m in names and lib.material(m).fluid is not None and lib.material(m).fluid.consistency]
    assert both == []
