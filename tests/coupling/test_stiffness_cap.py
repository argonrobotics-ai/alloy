# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-4d exchange stiffness cap (alloy/coupling/stiffness_cap.py), CPU.

The kernels on hand-made records (exact sums), the seeding hook on a real SolverVBD contact refresh, and the
installer's routes. C16 / C7 GPU acceptance is in docs/notes/wp10-sc4-4d.md.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

from alloy.core.errors import CouplingError
from alloy.coupling import stiffness_cap as sc
from alloy.coupling.types import SchedulerPolicy

DEV = "cpu"
R = 1.0e-3            # particle radius [m]
DT_X = 6.25e-4        # C16's exchange interval [s]


def _records(gaps: list[float], bodies: list[int], ke: float = 8.05e4, kd: float = 400.0):
    """Vertex records 1 m apart along x, each over a body point on z = 0 (normal +z) at the given gap."""
    n = len(gaps)
    q = np.array([[float(i), 0.0, R + g] for i, g in enumerate(gaps)], np.float32)
    return {
        "count": wp.array([n], dtype=wp.int32, device=DEV),
        "shape": wp.array(bodies, dtype=wp.int32, device=DEV),           # shape i on body bodies[i]
        "corners": wp.array([wp.vec3i(i, -1, -1) for i in range(n)], dtype=wp.vec3i, device=DEV),
        "bary": wp.array([wp.vec3(1.0, 0.0, 0.0)] * n, dtype=wp.vec3, device=DEV),
        "q": wp.array(q, dtype=wp.vec3, device=DEV),
        "radius": wp.full(n, R, dtype=float, device=DEV),
        "body_pos": wp.array(np.c_[q[:, :2], np.zeros(n)].astype(np.float32), dtype=wp.vec3, device=DEV),
        "normal": wp.array([wp.vec3(0.0, 0.0, 1.0)] * n, dtype=wp.vec3, device=DEV),
        "ke": wp.full(n, ke, dtype=float, device=DEV),
        "penalty": wp.full(n, 0.5 * ke, dtype=float, device=DEV),
        "kd": wp.full(n, kd, dtype=float, device=DEV),
    }


def _cap(rec: dict, shape_body: list[int], capped: list[int], mass: list[float], c: float, reach: float,
         gate=None, c_d: float = 0.0):
    """Both kernels; returns (near ke sums, near kd sums, peak ke / cap, peak kd / cap) per body."""
    gate = sc._SoftContactFilterData() if gate is None else gate      # no mesh features: every record has force
    nb = len(capped)
    total = wp.zeros((nb, 2), dtype=wp.int64, device=DEV)
    peak = wp.zeros((nb, 2), dtype=float, device=DEV)
    sb = wp.array(shape_body, dtype=wp.int32, device=DEV)
    body_q = wp.array([wp.transform_identity()] * nb, dtype=wp.transform, device=DEV)
    n = rec["shape"].shape[0]
    wp.launch(sc.sum_near_record_ke, dim=n, device=DEV,
              inputs=[rec["count"], rec["shape"], sb, wp.zeros(0, dtype=float, device=DEV),
                      wp.array(capped, dtype=wp.int32, device=DEV), rec["corners"], rec["bary"], rec["q"],
                      rec["radius"], body_q, rec["body_pos"], rec["normal"], rec["ke"], rec["kd"], reach, gate,
                      total])
    wp.launch(sc.scale_body_records, dim=n, device=DEV,
              inputs=[rec["count"], rec["shape"], sb, total, wp.array(mass, dtype=float, device=DEV),
                      c / DT_X**2, c_d / DT_X, rec["ke"], rec["penalty"], rec["kd"], peak])
    t, p = total.numpy() / sc.QUANT, peak.numpy()
    return t[:, 0], t[:, 1], p[:, 0], p[:, 1]


@wp.kernel
def _gaps(c: wp.array(dtype=wp.vec3i), b: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec3),
          r: wp.array(dtype=float), p: wp.array(dtype=wp.vec3), n: wp.array(dtype=wp.vec3),
          out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = sc.record_gap(c[i], b[i], q, r, wp.transform_identity(), p[i], n[i], 1.0e-4)


def test_record_gap_is_newtons_distance_less_radius_and_margin() -> None:
    rec = _records([-2.0e-4, 0.0, 3.0e-4], [0, 0, 0])
    out = wp.zeros(3, dtype=float, device=DEV)
    wp.launch(_gaps, dim=3, device=DEV, inputs=[rec["corners"], rec["bary"], rec["q"], rec["radius"],
                                                 rec["body_pos"], rec["normal"], out])
    np.testing.assert_allclose(out.numpy(), [-3.0e-4, -1.0e-4, 2.0e-4], atol=1e-7)


def test_a_body_over_its_cap_has_its_near_records_scaled_to_the_cap() -> None:
    m, c, reach = 0.55, 0.1, 0.5 * DT_X                              # reach 0.31 mm
    gaps = [-1e-4] * 30 + [2e-4] * 10 + [1e-3] * 5                    # 40 near, 5 far, all on body 0
    rec = _records(gaps, [0] * len(gaps))
    total, _, peak, _ = _cap(rec, [0], [1], [m], c, reach)
    cap = c * m / DT_X**2
    assert total[0] == pytest.approx(40 * 8.05e4)                     # far records do not count
    ke = rec["ke"].numpy()
    assert ke[:40].sum() == pytest.approx(cap, rel=1e-5)              # the near records now sum to the cap
    np.testing.assert_allclose(ke, ke[0])                             # one factor for every record of the body
    np.testing.assert_allclose(rec["penalty"].numpy(), 0.5 * ke, rtol=1e-6)
    assert peak[0] == pytest.approx(40 * 8.05e4 / cap, rel=1e-5)


def test_bodies_under_the_cap_uncapped_or_massless_keep_their_records() -> None:
    gaps = [-1e-4] * 801
    bodies = [0] + [1] * 400 + [2] * 400                              # record shape i sits on body shape_body[i]
    rec = _records(gaps, bodies)
    _, _, peak, _ = _cap(rec, [0, 1, 2], [1, 0, 1], [0.55, 0.55, 0.0], 0.1, 3e-4)
    np.testing.assert_array_equal(rec["ke"].numpy(), np.float32(8.05e4))   # 8.05e4 < cap 1.4e5; mask 0; mass 0
    np.testing.assert_array_equal(peak, 0.0)


def test_records_vbd_gives_no_force_do_not_count() -> None:
    """Newton's gate: a row with an unresolved sign (SC4-6's muted flat edges) or no ACTIVE particle has no force."""
    from alloy._vendor.newton import ParticleFlags
    from alloy._vendor.newton._src.geometry.soft_contacts_mesh import _MESH_FEATURE_SIGN_UNKNOWN

    rec = _records([-1e-4] * 4, [0] * 4)
    flags = np.full(4, int(ParticleFlags.ACTIVE), np.int32)
    flags[3] = 0                                                       # record 3: its particle is not ACTIVE
    gate = sc._SoftContactFilterData()
    gate.features = wp.array([wp.vec3i(0, 0, 0), wp.vec3i(0, 0, 0), wp.vec3i(_MESH_FEATURE_SIGN_UNKNOWN, 0, 0),
                              wp.vec3i(0, 0, 0)], dtype=wp.vec3i, device=DEV)
    gate.contact_count, gate.contact_max = rec["count"], 4
    gate.source_overflow = wp.array([1], dtype=wp.int32, device=DEV)   # overflow: no geometric re-check
    gate.soft_contact_indices, gate.particle_flags = rec["corners"], wp.array(flags, dtype=wp.int32, device=DEV)
    total = _cap(rec, [0], [1], [1.0e9], 0.1, 3e-4, gate)[0]
    assert total[0] == pytest.approx(2 * 8.05e4)                       # records 0 and 1 only


def test_the_sum_is_exact_in_any_record_order() -> None:
    rng = np.random.default_rng(3)
    ke = rng.uniform(1.0e3, 1.0e5, 500).astype(np.float32)
    sums = []
    for perm in (np.arange(500), rng.permutation(500)):
        rec = _records([-1e-4] * 500, [0] * 500)
        rec["ke"].assign(ke[perm])
        sums.append(_cap(rec, [0], [1], [1.0e9], 0.1, 3e-4)[0][0])    # huge mass: no scaling, just the sum
    assert sums[0] == sums[1]


def test_the_damping_cap_scales_kd_only_and_the_stiffness_cap_ke_only() -> None:
    """C16: MJCF solref (0.0025, 1) gives the YAM tips kd 800 N s/m, so ~400 per record. 40 near records sum to
    16,000 N s/m against c_d m / dt_x = 0.5 x 0.55 / 6.25e-4 = 440."""
    m = 0.55
    rec = _records([-1e-4] * 40, [0] * 40)
    _, kd_sum, pk, pd = _cap(rec, [0], [1], [m], 1.0e3, 3e-4, c_d=0.5)    # c huge: no ke cap
    assert kd_sum[0] == pytest.approx(40 * 400.0)
    assert rec["kd"].numpy().sum() == pytest.approx(0.5 * m / DT_X, rel=1e-5)
    np.testing.assert_array_equal(rec["ke"].numpy(), np.float32(8.05e4))
    assert pk[0] == 0.0 and pd[0] == pytest.approx(40 * 400.0 / (0.5 * m / DT_X), rel=1e-5)
    rec = _records([-1e-4] * 40, [0] * 40)
    _cap(rec, [0], [1], [m], 0.1, 3e-4)                                  # c_d off: kd untouched
    np.testing.assert_array_equal(rec["kd"].numpy(), np.float32(400.0))
    assert rec["ke"].numpy().sum() == pytest.approx(0.1 * m / DT_X**2, rel=1e-5)


def test_c_must_be_positive_and_none_installs_no_cap() -> None:
    assert sc.install_stiffness_caps(object(), None, None) == []
    assert (SchedulerPolicy().exchange_stiffness_c, SchedulerPolicy().exchange_damping_c) == (None, None)  # opt-in
    with pytest.raises(CouplingError, match="must be > 0 or None"):
        sc.ExchangeStiffnessCap(_FakeVBD(), np.zeros(0, np.int64), 0.0)
    with pytest.raises(CouplingError, match="not both None"):
        sc.ExchangeStiffnessCap(_FakeVBD(), np.zeros(0, np.int64), None, None)
    with pytest.raises(CouplingError, match="lacks Newton private"):
        sc.ExchangeStiffnessCap(object(), np.zeros(0, np.int64), 0.1)
    assert dataclasses.replace(SchedulerPolicy(), exchange_damping_c=0.5).exchange_damping_c == 0.5


class _FakeVBD:
    """The private surface the cap needs, with a seeding call that records what it was asked."""

    def __init__(self, bodies: int = 2) -> None:
        from types import SimpleNamespace

        self.model = SimpleNamespace(body_count=bodies, body_mass=wp.full(bodies, 0.5, dtype=float, device=DEV),
                                     particle_count=4, device=DEV)
        self.body_particle_contact_material_ke = wp.zeros(8, dtype=float, device=DEV)
        self.body_particle_contact_penalty_k = wp.zeros(8, dtype=float, device=DEV)
        self.body_particle_contact_material_kd = wp.zeros(8, dtype=float, device=DEV)
        self._particle_contact_adjacency_initialized = True
        self.seeded: list[bool] = []
        self.stepped: list[object] = []

    def _refresh_body_particle_contact_state(self, contacts, refresh: bool) -> None:
        self.seeded.append(refresh)

    def step(self, state_in, state_out, control, contacts, dt: float) -> None:
        self.stepped.append(state_in)


def test_the_hook_keeps_the_step_state_and_reads_vbds_refresh_condition() -> None:
    from types import SimpleNamespace

    vbd = _FakeVBD()
    cap = sc.ExchangeStiffnessCap(vbd, np.array([1]), 0.1)
    np.testing.assert_array_equal(cap.capped.numpy(), [0, 1])
    applied = []
    cap.apply = applied.append                                        # observe, do not launch
    contacts = SimpleNamespace(soft_contact_max=8)
    state = object()
    vbd.step(state, None, None, contacts, 1e-3)
    assert vbd.stepped == [state] and cap._state is state
    vbd._refresh_body_particle_contact_state(contacts, False)          # VBD keeps last seeds: no re-scale
    assert applied == []
    vbd._refresh_body_particle_contact_state(contacts, True)
    assert vbd.seeded == [False, True] and applied == [contacts]
    vbd._particle_contact_adjacency_initialized = False                # VBD forces a seed: so does the cap
    vbd._refresh_body_particle_contact_state(contacts, False)
    assert len(applied) == 2
    contacts.soft_contact_max = 16                                     # buffers grow: VBD re-seeds
    vbd._particle_contact_adjacency_initialized = True
    vbd._refresh_body_particle_contact_state(contacts, False)
    assert len(applied) == 3


ON = SchedulerPolicy(exchange_stiffness_c=4.0, exchange_damping_c=0.5)


def _pinch_scheduler(monkeypatch, policy: SchedulerPolicy = ON, **pair_kw):
    from alloy.coupling import pinch_rigs as pr

    if pair_kw:
        real = pr.PairPolicy
        monkeypatch.setattr(pr, "PairPolicy", lambda *a, **kw: real(*a, **{**kw, **pair_kw}))
    return pr._cloth_scheduler(pr.gripper_model(), full_surface=True, policy=policy)


def test_the_default_scheduler_installs_no_cap(monkeypatch) -> None:
    sched = _pinch_scheduler(monkeypatch, SchedulerPolicy())
    assert sched.stiffness_caps == ()
    vbd = sched.solver("vbd")
    assert type(vbd)._refresh_body_particle_contact_state is vbd._refresh_body_particle_contact_state.__func__


def test_the_scheduler_caps_the_dynamic_proxies_of_its_coupled_vbd_destination(monkeypatch) -> None:
    sched = _pinch_scheduler(monkeypatch)
    (cap,) = sched.stiffness_caps
    vbd = sched.solver("vbd")
    assert cap.solver is vbd and (cap.c, cap.c_d) == (4.0, 0.5)
    mass = vbd.model.body_mass.numpy()
    ids = np.concatenate([m.proxy_ids_local.numpy() for m in sched._proxy_mappings])
    np.testing.assert_array_equal(np.flatnonzero(cap.capped.numpy()), np.sort(ids[mass[ids] > 0.0]))
    assert type(vbd)._refresh_body_particle_contact_state is not vbd._refresh_body_particle_contact_state


def test_a_one_way_pair_gets_no_cap(monkeypatch) -> None:
    assert _pinch_scheduler(monkeypatch, one_way=True).stiffness_caps == ()
