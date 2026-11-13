# SPDX-License-Identifier: Apache-2.0
"""Vendored patch 0015 (alloy/_vendor/patches/0015-mpm-delassus-regularization.patch, GAP #6 slice 2b): Newton's
absolute Delassus regularization ``_DELASSUS_PROXIMAL_REG`` (1e-6) becomes ``SolverImplicitMPM.Config``'s
``delassus_regularization``, a kernel argument of ``compute_delassus_diagonal``. At the default every MPM entry runs
bit for bit as before (the kernel below is upstream's, verbatim: Newton release-1.8 db915e1c, Apache-2.0); the j2
route sets 1e-12 for its entry (``mpm_solid.DELASSUS_REGULARIZATION``): with quasi-static mass scaling a metal's
diagonal is 1e-9 to 1e-7, where 1e-6 stalls the solve ([internal note] notes/wp7-dent/slice2/b8)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
import warp.sparse as sp  # noqa: F401  (read by the upstream kernel)

import alloy
from alloy._vendor.newton._src.solvers.implicit_mpm import rheology_solver_kernels as RK
from alloy._vendor.newton._src.solvers.implicit_mpm import solve_rheology as SR
from alloy._vendor.newton._src.solvers.implicit_mpm.rheology_solver_kernels import (  # noqa: F401 (upstream kernel)
    _DELASSUS_PROXIMAL_REG, _ISOTROPIC_LOCAL_LHS, _symmetric_part_op, mat13, mat55, mat66, symmetric_eigenvalues_qr,
    vec6)
from alloy._vendor.newton._src.solvers.implicit_mpm.solver_implicit_mpm import SolverImplicitMPM
from alloy.solvers.plastic import mpm_solid


# --- upstream's kernel before the patch, verbatim (only its name changed) ------------------------------------------
@wp.kernel
def _upstream_compute_delassus_diagonal(
    strain_mat_offsets: wp.array[int],
    strain_mat_columns: wp.array[int],
    strain_mat_values: wp.array[mat13],
    inv_volume: wp.array[float],
    compliance_mat_offsets: wp.array[int],
    compliance_mat_columns: wp.array[int],
    compliance_mat_values: wp.array[mat66],
    strain_batch: wp.array[int],
    mass_multiplicity: wp.array2d[float],
    majorize: bool,
    delassus_rotation: wp.array[mat55],
    delassus_diagonal: wp.array[vec6],
):
    """Factor local Delassus blocks for iteration and strain reconstruction.

    Drop spherical/deviatoric coupling before QR. If ``majorize`` is true,
    bound that coupling in the iteration metric.

    Empty ``mass_multiplicity`` uses unit weights (GS); otherwise use the
    supplied per-batch weights (Jacobi or batched GS).
    """
    tau_i = wp.tid()
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    compliance_diag_index = sp.bsr_block_index(tau_i, tau_i, compliance_mat_offsets, compliance_mat_columns)
    if compliance_diag_index == -1:
        diag_block = mat66(0.0)
    else:
        diag_block = compliance_mat_values[compliance_diag_index]

    has_multiplicity = mass_multiplicity.shape[0] > 0
    bi = int(0)
    if has_multiplicity:
        bi = strain_batch[tau_i]

    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]

        mass_ratio = float(1.0)
        if has_multiplicity:
            mass_ratio = mass_multiplicity[bi, u_i]

        b_val = strain_mat_values[b]
        inv_frac = inv_volume[u_i] * mass_ratio

        b_v0 = _symmetric_part_op(b_val, wp.vec3(1.0, 0.0, 0.0))
        diag_block += inv_frac * wp.outer(b_v0, b_v0)
        b_v1 = _symmetric_part_op(b_val, wp.vec3(0.0, 1.0, 0.0))
        diag_block += inv_frac * wp.outer(b_v1, b_v1)
        b_v2 = _symmetric_part_op(b_val, wp.vec3(0.0, 0.0, 1.0))
        diag_block += inv_frac * wp.outer(b_v2, b_v2)

    diag_block += _DELASSUS_PROXIMAL_REG * wp.identity(n=6, dtype=float)

    coupling = vec6(0.0)
    for k in range(1, 6):
        coupling[k] = diag_block[k, 0]
        diag_block[0, k] = 0.0
        diag_block[k, 0] = 0.0

    diag, ev = symmetric_eigenvalues_qr(diag_block, _DELASSUS_PROXIMAL_REG * 0.1)

    if not (wp.ddot(ev, ev) < 1.0e16 and wp.length_sq(diag) < 1.0e16):
        diag = wp.get_diag(diag_block)
        ev = wp.identity(n=6, dtype=float)

    if majorize:
        # W = [[a, g^T], [g, D]] <= (1 + sqrt(g^T D^-1 g / a)) diag(a, D).
        coupling = ev @ coupling
        rho_sq = float(0.0)
        for k in range(1, 6):
            rho_sq += coupling[k] * coupling[k] / diag[k]
        diag *= 1.0 + wp.sqrt(rho_sq / diag[0])

    if wp.static(_ISOTROPIC_LOCAL_LHS):
        diag = vec6(wp.max(diag))
        ev = wp.identity(n=6, dtype=float)

    delassus_diagonal[tau_i] = diag
    delassus_rotation[tau_i] = wp.transpose(ev[1:6, 1:6])


def _dough(monkeypatch, periods: int = 2, device: str = "cpu") -> list[dict]:
    """A dough block pressed by a kinematic plate; every compute_delassus_diagonal launch's inputs and outputs, cloned
    right after the launch (they are temporaries). On a GPU without graphs (profile "debug"), so each launch is seen."""
    seen: list[dict] = []
    launch = wp.launch

    def spy(kernel, *a, **k):
        out = launch(kernel, *a, **k)
        if kernel is RK.compute_delassus_diagonal:
            seen.append({"dim": k["dim"], "inputs": [wp.clone(x) if isinstance(x, wp.array) else x
                                                     for x in k["inputs"]],
                         "outputs": [wp.clone(x) for x in k["outputs"]]})
        return out

    monkeypatch.setattr(wp, "launch", spy)
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("dough", size=(0.04, 0.04, 0.02), pos=(0, 0, 0.01), material="dough_wheat")
    s.add_box("press", size=(0.06, 0.06, 0.01), pos=(0, 0, 0.026), kinematic=True, material="steel_carbon")
    sim = alloy.build(s, device=device, profile="debug" if device != "cpu" else "standard")
    for _ in range(periods):
        sim.drive("press", velocity=(0, 0, -0.05), duration=sim.dt)
        sim.step(1)
    assert sim.check()["ok"]
    return seen


def test_the_default_is_newtons_constant() -> None:
    import inspect

    assert SolverImplicitMPM.Config().delassus_regularization == 1.0e-6 == float(_DELASSUS_PROXIMAL_REG)
    assert inspect.signature(SR.solve_rheology).parameters["delassus_regularization"].default == 1.0e-6
    assert mpm_solid.DELASSUS_REGULARIZATION == 1.0e-12


def _same_as_upstream(seen: list[dict]) -> None:
    assert seen and {c["inputs"][10] for c in seen} == {1.0e-6}
    for c in seen:
        rot, diag = (wp.empty_like(x) for x in c["outputs"])
        wp.launch(_upstream_compute_delassus_diagonal, dim=c["dim"], inputs=c["inputs"][:10], outputs=[rot, diag],
                  device=rot.device)                    # the arrays' device, not the default (cuda:0 on a GPU host)
        assert rot.numpy().tobytes() == c["outputs"][0].numpy().tobytes()
        assert diag.numpy().tobytes() == c["outputs"][1].numpy().tobytes()


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_the_kernel_at_the_default_is_upstreams_bit_for_bit(monkeypatch) -> None:
    """Every launch of a dough step at the default (1e-6) gives the same bytes as upstream's kernel on the same
    inputs, majorized or not."""
    _same_as_upstream(_dough(monkeypatch))


@pytest.mark.gpu
def test_on_a_gpu_the_kernel_at_the_default_is_upstreams_bit_for_bit(monkeypatch) -> None:
    _same_as_upstream(_dough(monkeypatch, device="cuda:0"))


J2_CARD = """schema: alloy.material/1
name: j2_reg_solid
density: {value: 1076.0, source: model, note: "patch 0015 fixture"}
surface:
  mu_static: {value: 1.0e-4, source: model, note: "fixture"}
  mu_dynamic: {value: 1.0e-4, source: model, note: "fixture"}
  restitution: {value: 0.0, source: model, note: "fixture"}
elastic:
  youngs: {value: 3.0e6, source: model, note: "fixture"}
  poisson: {value: 0.3, source: model, note: "fixture"}
plastic:
  yield_stress: {value: 1.0e4, source: model, note: "fixture"}
  model: j2
"""


def _j2_scene(tmp_path) -> alloy.Scene:
    (tmp_path / "materials").mkdir(exist_ok=True)
    (tmp_path / "materials" / "j2_reg_solid.yaml").write_text(J2_CARD)
    s = alloy.Scene(dt=0.01, roots=[str(tmp_path)])
    s.add_ground()
    s.add_box("blk", size=(0.012, 0.012, 0.012), pos=(0, 0, 0.006), material="j2_reg_solid",
              card_extra={"route": {"route": "paste_mpm", "reason": "patch 0015 fixture"}})
    return s


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_a_j2_entry_passes_its_regularization_to_the_kernel(tmp_path, monkeypatch) -> None:
    """The j2 route's entry is built with mpm_solid.DELASSUS_REGULARIZATION, and that value reaches every launch."""
    seen: list[float] = []
    launch = wp.launch

    def spy(kernel, *a, **k):
        if kernel is RK.compute_delassus_diagonal:
            seen.append(k["inputs"][10])
        return launch(kernel, *a, **k)

    monkeypatch.setattr(wp, "launch", spy)
    sim = alloy.build(_j2_scene(tmp_path), device="cpu")
    assert sim.engine.solvers[0].delassus_regularization == mpm_solid.DELASSUS_REGULARIZATION == 1.0e-12
    sim.step(2)
    assert seen and set(seen) == {1.0e-12}
    assert "Delassus regularization 1e-12" in sim.routes()["blk"]["mpm_j2_solid"]


@pytest.mark.parametrize("bad", [0.0, -1.0e-6, float("nan")])
def test_a_value_that_is_not_finite_and_positive_is_refused(tmp_path, monkeypatch, bad) -> None:
    monkeypatch.setattr(mpm_solid, "DELASSUS_REGULARIZATION", bad)
    with pytest.raises(Exception, match="delassus_regularization must be finite and positive"):
        alloy.build(_j2_scene(tmp_path), device="cpu")
