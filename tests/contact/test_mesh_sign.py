# SPDX-License-Identifier: Apache-2.0
"""Inside soft-contact rows on parity-signed meshes are re-checked (``alloy.contact.mesh_sign``, WP2's trace).

Warp 1.18.0's ray-parity sign says INSIDE at a degenerate cavity point of ``open_cylinder(0.03, 0.06, 0.004)`` (WP2's
float32 point, 4.3 mm from the inner wall; one ulp away it agrees with the pseudo-normal and the winding number).
Newton's per-particle soft contact there points into the cup wall; after the re-check (a perturbed-parity vote) it
points out, and particles really inside the wall keep their inside rows, including three of WP2's mid-wall points where
Warp's pseudo-normal sign says outside."""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton
from alloy.api import meshes
from alloy.contact.mesh_sign import ParityRecheck, has_parity_meshes
from alloy.coupling.proxy_contacts import ProxyContactPipeline

Q = np.frombuffer(bytes.fromhex("3564423c679dbabc19a56b3c"), np.float32).copy()     # WP2's collide-time point
# inside the 4 mm wall: one point, and three of WP2's mid-wall points where the pseudo-normal sign errs
MID_WALL = ("853bdfbbc969ff3c451f1b3c", "b5fce33c09f380bc78fa273c", "bc8f02bd38aef1ba6d38d73c")
IN_WALL = [np.array([0.032, 0.0, 0.03], np.float32)] + [np.frombuffer(bytes.fromhex(h), np.float32).copy()
                                                         for h in MID_WALL]
R = 0.002


def _cup_and_particles() -> newton.Model:
    v, f = meshes.open_cylinder(inner_radius=0.03, depth=0.06, wall=0.004)
    b = newton.ModelBuilder()
    b.add_shape_mesh(-1, mesh=newton.Mesh(np.asarray(v, np.float32), np.asarray(f, np.int32).ravel()))
    for p in (Q, *IN_WALL):
        b.add_particle(tuple(float(c) for c in p), (0.0, 0.0, 0.0), 1e-3, radius=R)
    return b.finalize(device="cpu")


def _gaps(contacts, state) -> dict[int, float]:
    """{particle: n.(x - p)} of each soft row (the shape is static: body_pos is in world)."""
    n = int(contacts.soft_contact_count.numpy()[0])
    x = state.particle_q.numpy()
    part, nrm, bp = (contacts.soft_contact_particle.numpy()[:n], contacts.soft_contact_normal.numpy()[:n],
                     contacts.soft_contact_body_pos.numpy()[:n])
    return {int(p): float(np.dot(nrm[k], x[p] - bp[k])) for k, p in enumerate(part)}


def test_parity_inside_row_is_flipped_and_a_real_one_kept():
    model = _cup_and_particles()
    assert has_parity_meshes(model)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.003)
    contacts, state = pipeline.contacts(), model.state()
    pipeline.collide(state, contacts)
    before = _gaps(contacts, state)
    wall = range(1, 1 + len(IN_WALL))
    assert before[0] < -0.004 and all(before[k] < -0.0015 for k in wall)   # Newton: all inside (the first falsely)
    recheck = ParityRecheck(model)
    recheck.apply(model, state, contacts)
    after = _gaps(contacts, state)
    assert after[0] == -before[0]                                         # flipped bit for bit
    assert all(after[k] == before[k] for k in wall)                       # the real ones untouched
    assert int(recheck.flips.numpy()[0]) == 1


def test_alloy_pipelines_carry_the_recheck():
    model = _cup_and_particles()
    pipeline = ProxyContactPipeline(rigid_shapes=())(model)
    contacts, state = pipeline.contacts(), model.state()
    pipeline.collide(state, contacts)
    gaps = _gaps(contacts, state)
    assert gaps[0] > 0.004 and all(gaps[k] < -0.0015 for k in range(1, 1 + len(IN_WALL)))


def test_no_parity_meshes_no_recheck():
    b = newton.ModelBuilder()
    b.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
    b.add_particle((0.0, 0.0, 0.101), (0.0, 0.0, 0.0), 1e-3, radius=R)
    model = b.finalize(device="cpu")
    assert not has_parity_meshes(model) and not ParityRecheck(model).enabled


def test_plain_scenes_run_the_recheck(monkeypatch):
    """WP2's re-review: a scene without coupling collides through AlloyContactLayer's bulk pass, not a proxy pipeline;
    a VBD sheet over a static mesh cup gets its rows re-signed there too (CPU, 2 periods)."""
    import alloy
    from alloy.contact import mesh_sign

    calls = {"apply": 0, "enabled": 0}
    apply = mesh_sign.ParityRecheck.apply

    def counting(self, model, state, contacts):
        calls["apply"] += 1
        calls["enabled"] += int(self.enabled)
        return apply(self, model, state, contacts)

    monkeypatch.setattr(mesh_sign.ParityRecheck, "apply", counting)
    v, f = meshes.open_cylinder(inner_radius=0.03, depth=0.06, wall=0.004)
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_mesh("cup", vertices=v, faces=f, collision="sdf", static=True)
    scene.add_cloth("sheet", size=(0.1, 0.1), pos=(0.0, 0.0, 0.07), resolution=8, mass=0.01)
    sim = alloy.build(scene, device="cpu")
    sim.step(2)
    assert calls["apply"] > 0 and calls["enabled"] == calls["apply"]
