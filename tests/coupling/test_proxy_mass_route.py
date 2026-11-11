# SPDX-License-Identifier: Apache-2.0
"""A MuJoCo body that cannot translate at its centre of mass gets its own mass as its proxy on the Warp route too
(``alloy.coupling.axis_mass.ProxyMassMixin``, WP9's wrist 2026-11-08).

MuJoCo Warp's set_const_0 puts such a body's rotational invweight0 in the translational slot, and Newton reads that as
1 / mass. WP9's wrist (0.3 kg, centre of mass on its hinge, I = 1e-4 kg m²) got a 1.6 g proxy, 3·I read as kg.
MuJoCo C's invweight0 has 0 there, and Newton keeps the body's mass, as on the C route. The pads on their slides keep
Newton's value. The suite runs the CPU on the Warp route (conftest)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.newton_compat import SolverCoupledProxy
from alloy.coupling.axis_mass import untranslatable


def _wrist():
    from alloy.coupling.pinch_rigs import _cloth_scheduler
    from tests.coupling.test_shapeless_proxy import _model

    model = _model()
    sched = _cloth_scheduler(model, False)
    labels = [str(x).split("/")[-1] for x in model.body_label]
    return model, sched, labels


def test_a_body_on_a_hinge_through_its_centre_of_mass_takes_its_own_mass():
    model, sched, labels = _wrist()
    entry = sched._entries["mjc"]
    proxy = sched._proxy_mappings[0]
    ids = [labels[int(g)] for g in proxy.src_ids.numpy()]
    newton_m, _ = SolverCoupledProxy._eval_effective_body_inertial_properties(sched, entry, proxy.src_ids)
    ours, _ = sched._eval_effective_body_inertial_properties(entry, proxy.src_ids)
    newton_m, ours = dict(zip(ids, newton_m)), dict(zip(ids, ours))
    assert newton_m["wrist"] < 0.005                                     # MuJoCo Warp's fallback, read as 1/mass
    mass = dict(zip(labels, model.body_mass.numpy()))
    assert ours["wrist"] == pytest.approx(mass["wrist"], rel=1e-6) == pytest.approx(0.3, rel=1e-6)
    for b in ("pad_a", "pad_b", "ghost"):                                # bodies that translate: Newton's value
        assert ours[b] == newton_m[b]
    mj = entry.solver.mj_model
    w = np.asarray(mj.body_invweight0).reshape(-1, 2)
    flag = untranslatable(entry.solver)
    assert [labels[i] for i in np.flatnonzero(flag)] == ["wrist"]
    table = entry.solver.mjc_body_to_newton.numpy()[0]                 # MuJoCo body -> Newton body (world 0)
    wid = int(np.flatnonzero(table == labels.index("wrist"))[0])
    assert w[wid, 0] == 0.0 and w[wid, 1] > 0.0                          # the C model's invweight0 the flag reads


def test_the_installed_proxy_is_the_wrists_mass():
    """What the Scheduler installs in the VBD entry (Newton's _apply_proxy_body_effective_masses, through the mixin)."""
    model, sched, labels = _wrist()
    sched._apply_proxy_body_effective_masses()
    proxy = sched._proxy_mappings[0]
    dst = sched._entries["vbd"].solver.model
    installed = dst.body_mass.numpy()[proxy.proxy_ids_local.numpy()]
    got = dict(zip((labels[int(g)] for g in proxy.src_ids.numpy()), installed))
    assert got["wrist"] == pytest.approx(0.3 * float(proxy.mass_scale), rel=1e-6)
