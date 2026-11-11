# SPDX-License-Identifier: Apache-2.0
"""A sleeping scene's compact solve finds the guard's scratch (SCR-WP5-18; CPU).

WP11 found the CPL-8 sleeping worlds NaN on rc-1109 (<gpu-vm>, 8.3 % of 1,024 worlds on an L4). With
sleeping on and ``nvmax`` set, MuJoCo Warp's ``solve_compact`` runs the Newton solve on
``dataclasses.replace(d, M=d.cM, ...)`` at ``nvmax_pad``, a new Data every step. Keyed on id(Data), the guard missed
its scratch on every such copy. Eager steps prepared a fresh [W, nv + 1, nv] float64 slab per call, and a captured
step ran MuJoCo Warp's Newton Cholesky unguarded. Keyed on the shared ``solver_niter``, the copy finds the slab
allocated at ``put_data``, sized for ``nvmax_pad``.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverMuJoCo

from alloy.precision import mjw_chol
from alloy.testing.coupling import resting_boxes_model

DT = 1.25e-3


@pytest.fixture
def sleeping():
    """A 2-world CPL-8 scene (4 resting boxes per world) with sleeping and compaction on (``nvmax`` = nv)."""
    assert mjw_chol.install()
    model = resting_boxes_model(2, 4, device="cpu")
    nv = model.joint_dof_count // model.world_count
    solver = SolverMuJoCo(model, use_mujoco_contacts=True, njmax=256, nconmax=64, enable_sleeping=True,
                          sleep_tolerance=1.0e-3, nvmax=nv)
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    return solver, model, s0, s1


def _steps(solver, model, s0, s1, n: int) -> None:
    control = model.control()
    for _ in range(n):
        solver.step(s0, s1, control, None, DT)
        s0, s1 = s1, s0


def test_eager_compact_steps_reuse_the_put_data_scratch(sleeping, monkeypatch) -> None:
    solver, model, s0, s1 = sleeping
    d = solver.mjw_data
    assert d.cM.size > 0 and int(d.nvmax_pad) > 0                      # the compact path is on
    seen, prepared = [], []
    scratch_for, prepare = mjw_chol._scratch_for, mjw_chol.prepare
    monkeypatch.setattr(mjw_chol, "_scratch_for", lambda m, dd: seen.append((int(m.nv), id(dd), scratch_for(m, dd)))
                        or seen[-1][2])
    monkeypatch.setattr(mjw_chol, "prepare", lambda m, dd: prepared.append(id(dd)) or prepare(m, dd))
    _steps(solver, model, s0, s1, 4)
    compact = [s for nv, i, s in seen if nv == int(d.nvmax_pad) and i != id(d)]
    assert compact, "no Newton call ran on a compact Data copy"
    assert prepared == []                                              # was: a new slab on every compact call
    assert all(s is mjw_chol._scratch_for(solver.mjw_model, d) for s in compact)
    assert mjw_chol.clamp_counts(d) is compact[0][0]                   # health reads the copies' fallbacks too


def test_a_captured_compact_copy_finds_its_scratch(sleeping, monkeypatch) -> None:
    solver, model, s0, s1 = sleeping
    m, d = solver.mjw_model, solver.mjw_data
    _steps(solver, model, s0, s1, 1)
    nvp = int(d.nvmax_pad)
    m2 = dataclasses.replace(m, nv=nvp, nv_pad=nvp, is_sparse=False)  # solve_compact's copies (solver.py ~4444)
    d2 = dataclasses.replace(d, M=d.cM, qacc=d.cqacc, efc=dataclasses.replace(d.efc, J=d.cJ, Ma=d.cMa))
    monkeypatch.setattr(mjw_chol.wp, "get_device", lambda *a: SimpleNamespace(is_capturing=True))
    s = mjw_chol._scratch_for(m2, d2)
    assert s is not None and s[1].shape[2] >= nvp                       # was None: the stock factor, unguarded
    assert mjw_chol.clamp_counts(d2) is mjw_chol.clamp_counts(d)


def test_a_wider_slab_keeps_the_fallback_count() -> None:
    """A Data whose slab grows (a call at a larger nv) keeps its counter: health holds the counter it read at build."""
    d = SimpleNamespace(solver_niter=wp.zeros(2, dtype=wp.int32, device="cpu"), nworld=2,
                        qpos=wp.zeros((2, 3), dtype=wp.float32, device="cpu"), cM=None)
    clamps, f64 = mjw_chol._allocate(3, d)
    wider = mjw_chol._allocate(5, d)
    assert wider[0] is clamps and wider[1].shape == (2, 6, 5) and f64.shape == (2, 4, 3)
    assert mjw_chol._allocate(4, SimpleNamespace(**vars(d))) is wider            # a copy sharing solver_niter
