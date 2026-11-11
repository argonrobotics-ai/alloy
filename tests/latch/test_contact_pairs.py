# SPDX-License-Identifier: Apache-2.0
"""Kept intra-assembly contacts (``LatchSpec.contacts``, DC3's detent ask): an explicit MuJoCo pair that the CON-47
exclude does not filter, the same in the MJCF piece, Newton's MJCF import (card route) and ``add_latch``."""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest

from alloy._vendor import newton

from alloy.latch import Box, ContactPair, LatchSpec, LatchSpecError, Member, Spring
from alloy.latch.builder import add_latch
from alloy.latch.mjcf import latch_mjcf

GAP = 0.5e-3          # slider bump to body bump at rest
PUSH_N = 3.0          # would carry the slider to its 5 mm stop without the contact


def _spec(contacts=(), **pair) -> LatchSpec:
    h = (0.5e-3, 2.0e-3, 0.5e-3)
    root = (Box("dt_base", (3.0e-3, 3.0e-3, 0.5e-3), (0.0, 0.0, -1.5e-3)),
            Box("dt_body_bump", h, (1.0e-3 + GAP, 0.0, 0.0), role="stop"))
    slider = Member("slider", "slide", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 5.0e-3, Spring(100.0, 0.5, 1.0e-2),
                    (Box("dt_slider_bump", h, (0.0, 0.0, 0.0), role="stop"),), 1.0e-2)
    if contacts is None:
        contacts = (ContactPair("dt_slider_bump", "dt_body_bump", 0.3, **pair),)
    return LatchSpec("dt", root, 0.1, (slider,), static=True, design_dt=1.0e-3, contacts=contacts)


def _settle_q(spec: LatchSpec) -> float:
    m = mujoco.MjModel.from_xml_string(latch_mjcf(spec))
    m.opt.timestep = spec.design_dt
    d = mujoco.MjData(m)
    d.qfrc_applied[0] = PUSH_N
    for _ in range(1500):
        mujoco.mj_step(m, d)
    return float(d.qpos[0])


def test_kept_contact_blocks_the_slider_the_exclude_lets_through():
    assert _settle_q(_spec()) > 4.5e-3                                           # excluded: runs to the stop
    q = _settle_q(_spec(contacts=None))
    assert q == pytest.approx(GAP, abs=5e-5)                                     # kept: rests on the body bump


def test_mjcf_pair_row_carries_the_precision_contact():
    spec = _spec(contacts=None)
    m = mujoco.MjModel.from_xml_string(latch_mjcf(spec))
    assert m.npair == 1 and m.nexclude == 1
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (m.pair_geom1[0], m.pair_geom2[0])}
    assert names == {"dt_slider_bump", "dt_body_bump"}
    assert np.allclose(m.pair_solref[0], (2.0e-3, 1.0)) and np.allclose(m.pair_friction[0][:2], 0.3)
    assert np.allclose(m.pair_solimp[0], (0.99, 0.99, 0.001, 0.5, 2.0)) and m.pair_margin[0] == 0.0
    assert "<pair" not in latch_mjcf(_spec())


def _solver_pair(b: newton.ModelBuilder) -> tuple[set[str], np.ndarray, np.ndarray]:
    model = b.finalize(device="cpu")
    mj = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True).mj_model
    assert mj.npair == 1
    names = {re.sub(r"_\d+$", "", (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "").split("/")[-1])
             for g in (mj.pair_geom1[0], mj.pair_geom2[0])}           # the solver suffixes geom names with an index
    return names, np.asarray(mj.pair_solref[0]), np.asarray(mj.pair_friction[0])


def test_card_import_and_builder_keep_the_same_pair(tmp_path):
    spec = _spec(contacts=None, timeconst=4.0e-3)
    path = tmp_path / "dt.xml"
    path.write_text(latch_mjcf(spec))
    bm = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(bm)
    bm.add_mjcf(str(path), floating=False)
    bb = newton.ModelBuilder()
    add_latch(bb, spec)
    for names, solref, friction in (_solver_pair(bm), _solver_pair(bb)):
        assert names == {"dt_slider_bump", "dt_body_bump"}
        assert np.allclose(solref, (4.0e-3, 1.0)) and np.allclose(friction[:2], 0.3)


@pytest.mark.parametrize("pair, match", [(("dt_base", "dt_body_bump"), "within part"),
                                         (("dt_slider_bump", "nope"), "unknown")])
def test_contact_spec_errors(pair, match):
    with pytest.raises(LatchSpecError, match=match):
        _spec(contacts=(ContactPair(*pair, 0.3),))
    with pytest.raises(LatchSpecError, match="condim"):
        ContactPair("a", "b", 0.3, condim=2)
