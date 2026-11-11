# SPDX-License-Identifier: Apache-2.0
"""SchedulerADMM cross-entry contacts (WP12, engine item 2, point 1).

Newton's ``SolverCoupledADMM.Config.contact_pairs`` defaults to (), which turns every ADMM-managed cross-entry contact
off, and a particle-only entry in a scene without static shapes kept its partner's body shapes as static colliders
(Newton's shape-visibility rule returns early for an entry with nothing visible). Box-on-cloth under SchedulerADMM:
the box fell through the cloth; with both fixed it rests where the lagged proxy rests it (1.035 m).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.newton_compat import SolverCoupledADMM, SolverCoupledProxy
from alloy.coupling.scheduler import SchedulerADMM, admm_contact_pairs, admm_entry_views
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
from alloy.coupling.static_walls import static_wall_shapes
from alloy.testing.coupling import CLOTH_Z, ClothSceneSpec, _world_builder, cloth_entries, coupling_box_on_cloth

pytestmark = pytest.mark.compile_heavy

DT = 1.0 / 480.0
STEPS = 144                      # 0.3 s: free fall from the start reaches the cloth plane in 0.1 s
COLLIDE_PARTICLES = int(newton.ShapeFlags.COLLIDE_PARTICLES)
PROXY_REST_Z = 1.035             # box centre at rest under the lagged proxy Scheduler (cloth_scheduler, 1.5 s)


def _entry(name: str, capability: str | None = None, **kw) -> AlloyEntry:
    return AlloyEntry(SolverCoupledProxy.Entry(name=name, solver=lambda v: None, **kw), adaptive=False,
                      capability=capability)


def test_contact_pairs_follow_the_two_way_pairs() -> None:
    entries = [_entry("mjc", "mujoco"), _entry("vbd", "vbd"), _entry("cloth2", "vbd"), _entry("sph", "sph")]
    pairs = [PairPolicy("mjc", "vbd", mode="admm"), PairPolicy("vbd", "mjc", mode="admm"),
             PairPolicy("mjc", "cloth2", mode="lagged", one_way=True), PairPolicy("mjc", "sph", mode="staggered"),
             PairPolicy("sph", "vbd", mode="admm")]               # a contact-free source end (D5 orders by name)
    got = admm_contact_pairs(entries, pairs)
    assert got == [SolverCoupledADMM.ContactPair(source="mjc", destination="vbd")]


def test_only_particle_only_entries_get_the_view_hook() -> None:
    model = coupling_box_on_cloth(1, device="cpu")
    e_m, e_v = cloth_entries(model)
    out = admm_entry_views(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)])
    assert out[0].entry.configure_view is None and out[1].entry.configure_view is not None


def test_static_walls_stay_visible_to_the_particle_entry() -> None:
    """A world-welded wall (SCR-JOB2-5, a robot's bench) keeps its particle bit in the cloth's view; the box loses it."""
    b = _world_builder(ClothSceneSpec())
    at = wp.transform((0.0, 0.4, 0.9), wp.quat_identity())
    bench = b.add_link(xform=at, label="bench")
    wall = b.add_shape_box(bench, hx=0.3, hy=0.02, hz=0.05)
    b.add_articulation([b.add_joint_fixed(-1, bench, parent_xform=at)])
    b.color()
    model = b.finalize(device="cpu")
    box = int(np.flatnonzero(model.shape_body.numpy() == 0)[0])
    assert static_wall_shapes(model) == [wall]
    e_m, e_v = cloth_entries(model)
    sched = SchedulerADMM(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                          [PairPolicy("mjc", "vbd", mode="admm")], SchedulerPolicy())
    flags = sched._entries["vbd"].view.shape_flags.numpy()
    assert flags[wall] & COLLIDE_PARTICLES and not flags[box] & COLLIDE_PARTICLES


def _box_after(admm_config) -> tuple[float, float]:
    model = coupling_box_on_cloth(1, device="cpu")
    e_m, e_v = cloth_entries(model)
    sched = SchedulerADMM(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                          [PairPolicy("mjc", "vbd", mode="admm")], SchedulerPolicy(), admm_config=admm_config)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    for _ in range(STEPS):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        sched.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
    return float(s0.body_q.numpy()[0, 2]), float(s0.body_qd.numpy()[0, 2])


def test_admm_box_rests_on_the_cloth() -> None:
    z, vz = _box_after(None)
    assert abs(z - PROXY_REST_Z) < 0.02, z                       # on the sagged cloth, where the lagged proxy rests it
    assert abs(vz) < 0.1, vz
    z_off, _ = _box_after(SolverCoupledADMM.Config())           # Newton's default: no contact rows
    assert z_off < CLOTH_Z - 0.2, z_off                          # falls through (0.62 m at 0.3 s)
    assert np.isfinite([z, vz, z_off]).all()
