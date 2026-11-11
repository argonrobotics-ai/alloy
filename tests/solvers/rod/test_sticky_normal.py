# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD F3: sticky contact matching keeps the fresh normal (vendored patch
alloy/_vendor/patches/*-contact-match.patch).

Newton's sticky mode replays a matched contact's body-frame points and offsets, and it also replayed the previous
frame's world-frame normal, which does not turn with the bodies. Two boxes in contact turn together 0.3 rad about x
in small steps (each frame matched): the replayed contact must report the turned normal. Before the patch it kept
the first one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton


DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


def _two_boxes(device: str) -> newton.Model:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for z in (0.0, 0.0995):                                   # 0.5 mm overlap along z
        body = b.add_body(xform=wp.transform((0.0, 0.0, z), wp.quat_identity()), mass=1.0)
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    return b.finalize(device=device)


def _normals(pipeline, model, state) -> np.ndarray:
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    n = int(contacts.rigid_contact_count.numpy()[0])
    assert n > 0
    return contacts.rigid_contact_normal.numpy()[:n].copy()


@pytest.mark.parametrize("device", DEVICES)
def test_a_sticky_contact_turns_its_normal_with_the_bodies(device):
    """Both boxes turn 0.01 rad per collide about x through their contact, 30 times, so every frame matches."""
    model = _two_boxes(device)
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    state = model.state()
    n0 = _normals(pipeline, model, state)
    assert np.allclose(np.abs(n0[:, 2]), 1.0, rtol=0, atol=1e-5)          # the first contact: along z
    q0 = state.body_q.numpy().copy()
    pivot = np.array([0.0, 0.0, 0.04975])                                 # the overlap's middle
    steps, da = 30, 0.01
    for k in range(1, steps + 1):
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), k * da)
        q = q0.copy()
        for i in range(2):
            p = np.asarray(wp.quat_rotate(rot, wp.vec3(*(q0[i, :3] - pivot)))) + pivot
            q[i, :3], q[i, 3:] = p, np.asarray([rot[0], rot[1], rot[2], rot[3]])
        state.body_q.assign(q)
        n1 = _normals(pipeline, model, state)
    a = steps * da
    turned = np.array([0.0, -math.sin(a), math.cos(a)]) * np.sign(n0[0, 2])
    assert np.allclose(n1, turned, rtol=0, atol=1e-4), n1
