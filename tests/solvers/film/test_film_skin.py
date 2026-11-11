# SPDX-License-Identifier: Apache-2.0
"""A thin film draped over a static box's edge stays out of the box (skin.py; docs/known_gaps/film-thin-sheet-box-edge.md).

Newton's VBD gives a particle-shape contact no Hessian outside the particle's skin, so a film vertex just outside
overshot deep into the shape in one iteration; the strain projection then pulled the taut sheet further over the edge.
``FilmConfig.contact_skin_entry`` limits both. Laws (CPU):
- the engine rig at the API's contact gains (ke 2e6): every static contact record of every step ends at most 1 um into
  its skin after VBD and at most the guard's 10 % after the projection; without the limit it went 0.2 mm deep;
- a sheet dropped across the seam of two coplanar static boxes rises no faster than on one box (a vertex on the seam
  has two coplanar records; each pushing it would throw it out: 4.1 vs 0.8 mm/s);
- the library's 50 um sheet over a 0.1 m cube (the API, the gap's scene) stays valid with no penetration flag.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.film import FilmConfig
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.materials import FilmExtras, FilmSpec
from alloy.solvers.film.rigs import FilmRig, grid_mesh
from alloy.solvers.film.skin import GUARD_DEPTH

pytestmark = pytest.mark.compile_heavy
ENTRY = 0.002                                    # the measured entry (opt-in for the release: FilmConfig default None)
LDPE_50UM = FilmSpec(youngs=3.4e8, poisson=0.4, density=931.0, thickness=5.0e-5, yield_stress=16e6)   # film_ldpe card
H = 0.05                                         # box half size: a 0.1 m cube on z = 0
R = FilmExtras().contact_radius


def _drape(entry: float | None) -> FilmRig:
    v, t = grid_mesh(21, 21, 0.15 / 21)                      # 0.15 m sheet, 7 mm edges, centred over the cube
    v[:, :2] -= v[:, :2].mean(axis=0)

    def build(b):                                          # the API build's gains for a static box and a film
        b.default_shape_cfg.ke, b.default_shape_cfg.kd, b.default_shape_cfg.mu = 2.0e6, 0.0, 0.4
        b.soft_contact_ke, b.soft_contact_kd, b.soft_contact_mu = 2.0e6, 10.0, 0.7
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, H), wp.quat_identity()), hx=H, hy=H, hz=H)
        add_film_mesh(b, v, t, LDPE_50UM, pos=(0.0, 0.0, 2 * H + R + 1e-3))
        return {}

    return FilmRig(build, dt=1.25e-3, ground=True, film=FilmConfig(contact_skin_entry=entry))


def _skin_depth(contacts, x: np.ndarray, static: np.ndarray) -> float:
    """The deepest static particle record: radius minus the distance along the record's normal (health's measure)."""
    n = int(contacts.soft_contact_count.numpy()[0])
    ids = contacts.soft_contact_indices.numpy()[:n]
    keep = (ids[:, 1] < 0) & static[contacts.soft_contact_shape.numpy()[:n]]
    p = ids[keep, 0]
    d = (contacts.soft_contact_normal.numpy()[:n][keep] * (x[p] - contacts.soft_contact_body_pos.numpy()[:n][keep])).sum(1)
    return float((R - d).max()) if len(p) else 0.0


def _depths(entry: float | None, seconds: float = 0.1) -> tuple[float, float]:
    """The deepest record over the run, after VBD and after the projection."""
    rig = _drape(entry)
    s = rig.solver
    static = rig.model.shape_body.numpy() < 0
    worst = {"vbd": 0.0, "out": 0.0}
    project = s._project

    def watched(state_out, dt, contacts=None):
        worst["vbd"] = max(worst["vbd"], _skin_depth(contacts, state_out.particle_q.numpy(), static))
        project(state_out, dt, contacts)
        worst["out"] = max(worst["out"], _skin_depth(contacts, state_out.particle_q.numpy(), static))

    s._project = watched
    rig.run(seconds)
    return worst["vbd"], worst["out"]


def test_drape_over_a_box_edge_stays_out_of_the_skin() -> None:
    entry = ENTRY
    vbd, out = _depths(entry)
    assert vbd <= entry * R + 1.0e-6, vbd                      # 1 um: float32 positions at 0.1 m, VBD's last iterate
    assert out <= GUARD_DEPTH * R + 1.0e-6, out                # the projection may press a light vertex that far
    vbd0, out0 = _depths(None)
    assert max(vbd0, out0) > 1.0e-4, (vbd0, out0)                # without the limit: 0.2 mm (the law catches it)


def _seam_drop(seam: bool, seconds: float = 0.1) -> tuple[float, float]:
    """A 0.08 m sheet (a vertex row on x = 0) dropped 1 mm onto a 0.2 m top, one box or two meeting at x = 0: the
    deepest skin depth and the fastest upward particle speed over the run."""
    v, t = grid_mesh(14, 14, 0.08 / 14)
    v[:, :2] -= v[:, :2].mean(axis=0)

    def build(b):
        b.default_shape_cfg.ke, b.default_shape_cfg.kd, b.default_shape_cfg.mu = 2.0e6, 0.0, 0.4
        b.soft_contact_ke, b.soft_contact_kd, b.soft_contact_mu = 2.0e6, 10.0, 0.7
        hx = 0.05 if seam else 0.1
        for sx in ((-0.05, 0.05) if seam else (0.0,)):
            b.add_shape_box(-1, xform=wp.transform(wp.vec3(sx, 0.0, H), wp.quat_identity()), hx=hx, hy=0.1, hz=H)
        add_film_mesh(b, v, t, LDPE_50UM, pos=(0.0, 0.0, 2 * H + R + 1e-3))
        return {}

    rig = FilmRig(build, dt=1.25e-3, ground=True, film=FilmConfig(contact_skin_entry=ENTRY))
    deep = up = 0.0
    for _ in range(int(round(seconds / rig.dt))):
        rig.step()
        deep = max(deep, float((R - (rig.s0.particle_q.numpy()[:, 2] - 2 * H)).max()))
        up = max(up, float(rig.s0.particle_qd.numpy()[:, 2].max()))
    return deep, up


def test_sheet_on_a_seam_of_two_coplanar_boxes_is_pushed_out_once() -> None:
    deep, up = _seam_drop(True)
    deep1, up1 = _seam_drop(False)
    limit = GUARD_DEPTH * R + 1.0e-6
    assert deep <= limit and deep1 <= limit, (deep, deep1)
    assert up <= 1.25 * up1 + 1.0e-4, (up, up1)               # m/s; a double push read 4.1 vs 0.8 mm/s


def test_library_thin_sheet_over_a_cube_stays_valid(monkeypatch) -> None:
    import functools

    import alloy
    from alloy.solvers.film import plugin

    monkeypatch.setattr(plugin, "FilmConfig", functools.partial(FilmConfig, contact_skin_entry=ENTRY))   # the opt-in

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05), static=True)
    scene.add_object("sheet", "film_ldpe_sheet", pos=(0.0, 0.0, 0.102))   # 50 um LDPE, 2 mm over the top
    sim = alloy.build(scene, device="cpu")
    r = sim.routes()
    assert "contact_skin" in r["sheet"] and "contact_skin" not in r["cube"], r
    worst = 0.0
    for _ in range(30):                                        # 0.3 s: was INVALID (PEN_FAIL) at 0.11 s
        sim.run(0.01)
        h = sim.health()["worlds"][0]
        worst = max(worst, h.get("max_penetration_m") or 0.0)
        assert h["severity"] != "INVALID", h.get("problem")
        assert not {"PEN_WARN", "PEN_FAIL"} & set(h["flags"]), h["flags"]
    assert worst <= GUARD_DEPTH * R + 1.0e-6, worst            # 0.05 mm (health's limit here: 0.5 mm)


def test_film_config_checks_the_skin_entry() -> None:
    assert FilmConfig().contact_skin_entry is None                            # opt-in for the release
    assert FilmConfig(contact_skin_entry=ENTRY).contact_skin_entry == ENTRY
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="contact_skin_entry"):
            FilmConfig(contact_skin_entry=bad)
