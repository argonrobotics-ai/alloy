# SPDX-License-Identifier: Apache-2.0
"""TactileSensor end to end (OSS-21): worlds, body attachment, markers, stick-slip, augmentation, errors, bench."""
from __future__ import annotations

import json

import numpy as np
import pytest
import warp as wp

from alloy.camera.errors import CameraError
from alloy.camera.tactile import Augment, TactileRenderer, TactileSensor
from alloy.camera.tactile.bench import main as bench_main

GEL = ((0.0, 0.0, 0.002), (0.0, 0.0, 0.0, 1.0))
QUIET = Augment(noise_std=0.0)


def ball_model(worlds: int = 1, depth: float = 0.0005, radius: float = 0.004):
    from alloy._vendor import newton
    w = newton.ModelBuilder()
    pad = w.add_body(label="pad")
    w.add_shape_box(pad, hx=0.012, hy=0.01, hz=0.002)
    ball = w.add_body(xform=wp.transform((0.0, 0.0, 0.002 + radius - depth), wp.quat_identity()), label="ball")
    w.add_shape_sphere(ball, radius=radius)
    if worlds == 1:
        return w.finalize(device="cpu")
    b = newton.ModelBuilder()
    b.replicate(w, copy_count=worlds)
    return b.finalize(device="cpu")


def test_sensor_renders_every_world_and_a_free_gel_is_the_background():
    model = ball_model(worlds=2)
    state = model.state()
    q = state.body_q.numpy().copy()
    q[3, 2] += 0.01  # lift world 1's ball off the gel
    state.body_q.assign(q)
    s = TactileSensor("tip", attach_to="pad", pose=GEL, width=160, augment=QUIET, markers=False)
    out = s.render(model, state, output="all")
    img = out.rgb.numpy()
    assert img.shape == (2, 120, 160, 3) and img.dtype == np.uint8 and s.size == (120, 160)
    bg = s._bound[3]._bg.numpy()
    assert np.abs(img[1] - bg).max() <= 0.5 + 1e-3  # only uint8 rounding
    assert np.abs(img[0] - bg).max() > 20
    assert out.press.numpy()[0].max() == pytest.approx(0.5, abs=1e-3) and out.press.numpy()[1].max() == 0.0


def test_attach_by_label_follows_each_world_body():
    model = ball_model(worlds=3)
    s = TactileSensor("tip", attach_to="pad", pose=GEL, width=80, augment=QUIET)
    img = s.render(model, model.state()).numpy()
    assert np.array_equal(img[0], img[1]) and np.array_equal(img[0], img[2])
    with pytest.raises(CameraError, match="no body 'paw'"):
        TactileSensor("x", attach_to="paw", width=80).render(model, model.state())


def test_stick_then_slip_and_reset():
    model = ball_model()
    state = model.state()
    base = state.body_q.numpy().copy()
    s = TactileSensor("tip", attach_to="pad", pose=GEL, width=160, augment=QUIET, slip_mm=0.6)
    got = []
    for dx in (0.0, 0.0002, 0.0004, 0.0008, 0.0016):
        q = base.copy()
        q[1, 0] = dx
        state.body_q.assign(q)
        got.append(s.render(model, state, output="all").motion.numpy()[0].copy())
    assert np.allclose([g[0] for g in got], [0.0, 0.2, 0.4, 0.6, 0.6], atol=2e-3)  # mm: sticks, then slips at 0.6
    assert all(g[3] == 1.0 for g in got)
    s.reset()
    assert s._bound[2].motion.numpy()[0, 3] == 0.0


def test_markers_dilate_outward_and_follow_given_displacements():
    tr = TactileRenderer("gelsight", width=160, num_worlds=1, device="cpu", augment=QUIET)
    yy, xx = np.mgrid[0:120, 0:160].astype(np.float64)
    x, y = (xx + 0.5 - 80) * tr.mm_per_px, (yy + 0.5 - 60) * tr.mm_per_px
    press = np.clip(np.sqrt(np.clip(4.0 - x * x - y * y, 0, None)) - 1.0, 0, None)[None].astype(np.float32)
    tr.render(press)
    rest = tr.grid.rest_mm
    disp = tr.marker_pos.numpy()[0] - rest
    near = np.argsort(np.linalg.norm(rest, axis=1))[:6]
    radial = (disp[near] * rest[near]).sum(1) / np.linalg.norm(rest[near], axis=1)
    assert (radial > 0.05).all() and (radial < 0.5).all()  # mm, outward
    given = np.zeros((1, tr.grid.count, 2), np.float32)
    given[0, :, 0] = 0.3
    tr.render(press, marker_disp=given)
    assert np.allclose(tr.marker_pos.numpy()[0], rest + [0.3, 0.0], atol=1e-6)
    with pytest.raises(CameraError, match="marker_disp"):
        tr.render(press, marker_disp=given[:, :5])


def test_augmentation_is_seeded_per_world():
    press = np.zeros((2, 60, 80), np.float32)
    a = TactileRenderer("digit", width=80, num_worlds=2, device="cpu", augment=Augment.strong(seed=1))
    b = TactileRenderer("digit", width=80, num_worlds=2, device="cpu", augment=Augment.strong(seed=1))
    ia, ib = a.render(press, seed=5).numpy().copy(), b.render(press, seed=5).numpy().copy()
    assert np.array_equal(ia, ib) and not np.array_equal(ia[0], ia[1])
    b.resample(2)
    assert not np.array_equal(ia, b.render(press, seed=5).numpy())
    with pytest.raises(CameraError, match="between 0 and 1"):
        Augment(wear=2.0).check()


def test_bad_requests_say_what_to_change():
    model = ball_model()
    s = TactileSensor("tip", attach_to="pad", pose=GEL, width=80)
    with pytest.raises(CameraError, match="output"):
        s.render(model, model.state(), output="rgb")
    with pytest.raises(CameraError, match="cannot sense"):
        s.render(object())
    tr = TactileRenderer("gelsight", width=80, num_worlds=1, device="cpu")
    with pytest.raises(CameraError, match="press map has shape"):
        tr.render(np.zeros((1, 10, 10), np.float32))
    try:
        TactileSensor("x", attach_to="paw").render(model, model.state())
    except CameraError as err:
        assert set(err.to_json()) == {"error", "fix"}


def test_bench_cli_writes_json(tmp_path, capsys):
    out = tmp_path / "speed.jsonl"
    assert bench_main(["--worlds", "2", "--sizes", "80", "--frames", "1", "--device", "cpu", "--out", str(out)]) == 0
    row = json.loads(out.read_text().splitlines()[0])
    assert row["worlds"] == 2 and row["width"] == 80 and row["images_per_s"] > 0 and row["frac_worlds_rendered"] == 1.0


def test_reset_only_the_given_worlds():
    """Audit T2-006: a masked reset restarts the stick-slip of those worlds only; the others keep their shear."""
    model = ball_model(worlds=2)
    state = model.state()
    base = state.body_q.numpy().copy()
    s = TactileSensor("tip", attach_to="pad", pose=GEL, width=160, augment=QUIET)

    def shift(dx: float) -> np.ndarray:
        q = base.copy()
        q[[1, 3], 0] += dx  # each world's ball
        state.body_q.assign(q)
        return s.render(model, state, output="all").motion.numpy().copy()

    shift(0.0)
    assert np.allclose(shift(0.0002)[:, 0], 0.2, atol=2e-3)
    s.reset(worlds=[0])
    m = s._bound[2].motion.numpy()
    assert m[0, 3] == 0.0 and m[1, 3] == 1.0 and m[1, 0] == pytest.approx(0.2, abs=2e-3)
    after = shift(0.0003)
    assert after[0, 0] == pytest.approx(0.0, abs=2e-3) and after[1, 0] == pytest.approx(0.3, abs=2e-3)
    s.reset(worlds=np.array([False, True]), seed=4)
    assert s._bound[2].motion.numpy()[0, 3] == 1.0 and s._bound[2].motion.numpy()[1, 3] == 0.0
    with pytest.raises(CameraError, match="outside 0..1"):
        s.reset(worlds=[2])
