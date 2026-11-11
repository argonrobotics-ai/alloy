# SPDX-License-Identifier: Apache-2.0
"""SCR-SC5-3: the cut-front predicate the health monitor uses to exempt Blade x phase-1 twin contacts from PEN.

- During a stroke, ``cut_front_blade`` equals a host recomputation from the bond phases at every step: the Blade
  body for each twin of a phase-1 TET_CUT bond, -1 elsewhere.
- On the stroke rig, the soft contacts deeper than the thin-part limit (the ones health flags today) are all
  cut-front contacts, and they stay within the ruling's limit (blade half-thickness + particle radius). The rest stay
  within the thin-part limit: WP5's "cut stroke reads 0 PEN_FAIL" case.
- A twin whose bond is already broken (phase 2) and a plain particle, both inside the blade, are not cut-front
  contacts: they still flag.
- ``blade_half_thickness``: the blade face box's thin half-extent, whichever knife shape ``Blade.shape`` names.
- Scenes without a cut keep -1 everywhere; the refresh replays from a CUDA graph.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.testing.topology import (STROKE_EDGE, STROKE_SPEED, STROKE_TOP, blade_pose, clamp_particles,
                                    cut_stroke_rig, presplit_block, run_cut_stroke)
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.cut import PHASE_CUTTING
from alloy.topology.cut_front import blade_half_thickness
from alloy.topology.types import Blade, BondKind, TopologyBudget

DT = 1.0e-3


def _descent(n: int) -> np.ndarray:
    return STROKE_TOP + 2.0e-3 - STROKE_SPEED * DT * np.arange(n)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:, :3], q[:, 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _soft_depths(m: newton.Model, contacts, state: newton.State) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(particle, body, depth [m]) of every soft contact on a body shape, measured as the health monitor does:
    contact point moved with the body to ``state``, depth = radius - normal . (x - point)."""
    n = int(contacts.soft_contact_count.numpy()[0])
    p, s = contacts.soft_contact_particle.numpy()[:n], contacts.soft_contact_shape.numpy()[:n]
    body = m.shape_body.numpy()[s]
    keep = body >= 0
    p, body = p[keep], body[keep]
    bq = state.body_q.numpy()[body]
    point = bq[:, :3] + _rotate(bq[:, 3:7], contacts.soft_contact_body_pos.numpy()[:n][keep])
    nrm = contacts.soft_contact_normal.numpy()[:n][keep]
    depth = m.particle_radius.numpy()[p] - np.einsum("ij,ij->i", nrm, state.particle_q.numpy()[p] - point)
    return p, body, depth


def _expected(ctl: TopologyController, blade_of_world: dict[int, int]) -> np.ndarray:
    m, b = ctl.model, ctl.bonds
    out = np.full(m.particle_count, -1)
    world = m.particle_world.numpy()
    a, bb, kind, phase = b.a.numpy(), b.b.numpy(), b.kind.numpy(), ctl.phase.numpy()
    for i in np.flatnonzero((phase == PHASE_CUTTING) & (kind == int(BondKind.TET_CUT))):
        body = blade_of_world[int(world[a[i]])]
        out[a[i]], out[bb[i]] = max(out[a[i]], body), max(out[bb[i]], body)
    return out


def test_cut_front_follows_phase_one_twins_and_holds_the_flagged_contacts(device):
    rig = cut_stroke_rig(device, worlds=2)
    m, ctl = rig.sc.model, rig.ctl
    blade_of_world = {int(m.body_world.numpy()[b]): int(b) for b in rig.sc.blade_bodies}
    half = ctl.blade_half_thickness.numpy()
    radius = float(m.particle_radius.numpy().max())
    thin = 0.5 * min(2.0 * radius, 2.0 * float(half[rig.sc.blade_bodies[0]]))     # health's thin-part limit
    seen = {"marked": 0, "front": 0.0, "front_n": 0, "other": 0.0}

    def observe(k, s0, s1, contacts):
        front = ctl.cut_front_blade.numpy()
        np.testing.assert_array_equal(front, _expected(ctl, blade_of_world), err_msg=f"step {k}")
        seen["marked"] += int((front >= 0).any())
        p, body, depth = _soft_depths(m, contacts, s1)
        cf = front[p] == body
        seen["front"] = max(seen["front"], float(depth[cf].max(initial=0.0)))
        seen["front_n"] += int((depth[cf] > thin).sum())
        seen["other"] = max(seen["other"], float(depth[~cf].max(initial=0.0)))

    run_cut_stroke(rig, _descent(300), dt=DT, observer=observe)
    print(f"cut-front: {seen['marked']} steps, {seen['front_n']} contacts over {thin * 1e3:.2f} mm, deepest "
          f"{seen['front'] * 1e3:.3f} mm (limit {(half.max() + radius) * 1e3:.2f}); other contacts "
          f"{seen['other'] * 1e3:.3f} mm; cut {int(ctl.cut_done.numpy().sum())}")
    assert seen["marked"] > 50 and int(ctl.cut_done.numpy().sum()) > 0
    # item 9 step 1 (coordinator 2026-11-06): BladeContact ramps a twin onto its face as the edge nears, so the
    # cut front stays within the thin-part limit (it used to cross 0.5-1.5 mm deep)
    assert seen["front"] <= 0.5e-3
    assert seen["other"] <= thin


def _deep_contacts(rig, z: float, x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pin the particles on the plane x, teleport the blade over them for one step: (particle, body,
    cut_front_blade) of the contacts deeper than 0.5 mm (the stroke rig's thin-part limit)."""
    m = rig.sc.model
    clamp_particles(m, np.flatnonzero(np.abs(m.particle_q.numpy()[:, 0] - x) < 1.0e-6))
    out = {}

    def observe(k, s0, s1, contacts):
        p, body, depth = _soft_depths(rig.sc.model, contacts, s1)
        deep = depth > 0.5e-3
        out["r"] = (p[deep], body[deep], rig.ctl.cut_front_blade.numpy()[p[deep]])

    run_cut_stroke(rig, np.array([z]), x=x, dt=DT, observer=observe)
    return out["r"]


def test_broken_twins_and_plain_particles_in_the_blade_are_not_cut_front(device):
    rig = cut_stroke_rig(device)
    kind = rig.ctl.bonds.kind.numpy()
    rig.ctl.preset_broken(np.flatnonzero(kind == int(BondKind.TET_CUT)))
    p, body, front = _deep_contacts(rig, 2.0e-3, 0.01)                  # seam twins, bonds broken (phase 2)
    assert p.size and np.all(front == -1)
    rig = cut_stroke_rig(device)
    a, b = rig.ctl.bonds.a.numpy(), rig.ctl.bonds.b.numpy()
    p, body, front = _deep_contacts(rig, 2.0e-3, 0.005)                 # x = 5 mm: no seam plane
    plain = ~np.isin(p, np.concatenate([a, b]))
    assert plain.any() and np.all(front[plain] == -1)
    assert np.all((front == -1) | (front == body))


def _knife(device: str) -> tuple[newton.Model, list[Blade], int]:
    """C9's knife (massless, as a kinematic blade): edge capsule (Blade.shape), 1.2 mm blade box, handle box; a
    box-less pin blade; and a plain body."""
    b = newton.ModelBuilder()
    knife = b.add_body(mass=0.0)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    edge = b.add_shape_capsule(knife, radius=1.0e-4, half_height=0.075, cfg=cfg)
    b.add_shape_box(knife, hx=6.0e-4, hy=0.075, hz=0.0175, cfg=cfg)
    b.add_shape_box(knife, hx=8.0e-3, hy=0.055, hz=0.011, cfg=cfg)                       # handle
    pin = b.add_body(mass=0.0)
    b.add_shape_capsule(pin, radius=2.0e-4, half_height=0.01, cfg=cfg)
    other = b.add_body()
    b.add_shape_box(other, hx=1.0e-3, hy=1.0e-3, hz=1.0e-3)
    m = b.finalize(device=device)
    seg = ((0.0, -0.05, 0.0), (0.0, 0.05, 0.0))
    return m, [Blade(knife, edge, *seg), Blade(pin, -1, *seg, edge_radius=3.0e-4)], other


@pytest.mark.filterwarnings("ignore:Inertia validation corrected")     # massless kinematic blade bodies
def test_blade_half_thickness_is_the_face_box_whatever_shape_the_blade_names(device):
    m, blades, other = _knife(device)
    half = blade_half_thickness(m, blades)
    assert half.shape == (m.body_count,) and half.dtype == np.float32
    np.testing.assert_allclose(half[[blades[0].body, blades[1].body, other]], [6.0e-4, 3.0e-4, 0.0])
    sc = presplit_block(device, cells=(2, 2, 2))
    ctl = TopologyController(sc.model, bonds_from_model(sc.model), TopologyBudget(), ())     # TET_CUT, no blades
    ctl.pre_step(sc.model.state(), None, DT)
    assert ctl.cut_front_blade.shape == (sc.model.particle_count,)
    assert np.all(ctl.cut_front_blade.numpy() == -1) and not ctl.blade_half_thickness.numpy().any()


@pytest.mark.gpu
def test_cut_front_replays_from_a_graph(device):
    """The captured pre_step (cut kernel, then the refresh) gives the phase-1 twins at every step of a stroke."""
    if not device.startswith("cuda"):
        pytest.skip("needs CUDA (ALLOY_DEVICE=cuda:0)")
    rig = cut_stroke_rig(device, worlds=2)
    m, ctl = rig.sc.model, rig.ctl
    blade_of_world = {int(m.body_world.numpy()[b]): int(b) for b in rig.sc.blade_bodies}
    s0, s1 = rig.states
    with wp.ScopedCapture(device=device) as cap:
        ctl.pre_step(s0, None, DT)
    marked = 0
    edge, down = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    for k, z in enumerate(_descent(200)):                  # fixed buffers: state 0 in, state 1 out, copied back
        bq = s0.body_q.numpy()
        bq[rig.sc.blade_bodies] = blade_pose(np.array([0.01, 0.0075, z + STROKE_EDGE]), edge, down)
        s0.body_q.assign(bq)
        wp.capture_launch(cap.graph)
        front = ctl.cut_front_blade.numpy()
        np.testing.assert_array_equal(front, _expected(ctl, blade_of_world), err_msg=f"step {k}")
        marked += int((front >= 0).any())
        rig.pipe.collide(s0, rig.contacts)
        rig.blade_contact.adjust_contacts(s0, rig.contacts)
        rig.vbd.step(s0, s1, m.control(), rig.contacts, DT)
        s1.body_q.assign(bq)
        ctl.apply(s1)
        wp.copy(s0.particle_q, s1.particle_q)
        wp.copy(s0.particle_qd, s1.particle_qd)
    assert marked > 50
