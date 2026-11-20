# SPDX-License-Identifier: Apache-2.0
"""Each randomization row is drawn for its own object and lands on that object, also in a tree-weld scene, and a
randomized start coordinate replaces that coordinate of a per-world ``pos`` row.

A chain welded at its root to another object becomes a tree joint, and the chain is emitted right after that object
(``alloy.scene.tree_weld.emission_order``), so the assembly plan lists the objects in another order than the scene
card. The draws (``apply_randomization`` at assembly and on a redraw, ``_all_draws`` on a reset) paired the plan's
names with the card's placements by position: with a cube between a plug and its cable in the card, the cube's rows
went to the cable (its link masses and joint coordinates) and the cable's friction to the cube. In the Scene API, a
free body given one ``pos`` row per world and ``randomize(name, "x", ...)`` started at the row's x: the build placed
the rows after the assembly had drawn x, and overwrote it.
"""

from __future__ import annotations

import numpy as np
import pytest

PLUG, CUBE, CABLE = "usb_a_plug", "cube_beech_30", "usb_cable_1m"
POSES = {PLUG: (0.3, -0.31, 0.00415, 0.5, 0.5, 0.5, 0.5),                  # the plug and cable as in task T09
         CUBE: (0.6, 0.3, 0.0155, 0.0, 0.0, 0.0, 1.0),
         CABLE: (0.28, -0.31, 0.00225, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476)}
ANCHOR = (0.0, -0.0019, -0.02, 0.0, 0.0, 0.0, 1.0)
WORLDS, SEED = 4, 7


def _rows():
    from alloy.cards.types import Randomization as R

    return {PLUG: (R("material.density", 7000.0, 8500.0),),             # steel, 7,850 kg/m3 on its card
            CUBE: (R("pose.0", 0.55, 0.65), R("material.density", 600.0, 800.0)),      # beech, 720 kg/m3
            CABLE: (R("material.surface.mu_dynamic", 0.2, 0.45),)}         # under its mu_static 0.5


@pytest.fixture
def built():
    """Plug, cube, cable in the card (the cube between the plug and the cable it is welded to), four worlds; built
    for each test, as the draws write the model."""
    from alloy.cards.library import CardLibrary
    from alloy.cards.types import Attachment, Placement, SceneCard
    from alloy.scene.assembler import SceneAssembler
    from alloy.scene.builtin_plugins import register_builtin_solvers

    register_builtin_solvers()
    rows = _rows()
    card = SceneCard(robots=(), objects=tuple(Placement(n, POSES[n], rows[n]) for n in (PLUG, CUBE, CABLE)),
                     attachments=(Attachment(PLUG, CABLE, "weld", anchor=ANCHOR),), control_dt=0.02)
    return SceneAssembler(CardLibrary(assets=None)).assemble(card, "cpu", world_count=WORLDS)


def _expected(built, seed: int) -> dict[tuple[str, str], np.ndarray]:
    """Every row's draws in every world, keyed by the object whose placement holds the row."""
    from alloy.scene.randomize import draw

    gids = [built.global_world_id(w) for w in range(built.world_count)]
    return {(n, r.field): np.asarray([draw(seed, g, f"{n}.{r.field}", r.lo, r.hi) for g in gids])
            for n, rows in _rows().items() for r in rows}


def _index(built, name: str, kind: str) -> np.ndarray:
    """(worlds, n) model rows of object ``name``'s ``kind`` ("body", "joint", "shape") in every world."""
    h = built.handles[name]
    rng = np.asarray(list(getattr(h.template, kind)), np.int64)
    return np.arange(built.world_count)[:, None] * len(getattr(h.world_stride, kind)) + rng[None, :]


def _coords(built, name: str) -> np.ndarray:
    """(worlds, n) joint coordinates of object ``name`` in every world."""
    qs = built.model.joint_q_start.numpy()
    rows = _index(built, name, "joint")
    cols = np.stack([np.concatenate([np.arange(qs[j], qs[j + 1]) for j in row]) for row in rows])
    return built.model.joint_q.numpy()[cols]


def _same_in_every_world(a: np.ndarray) -> bool:
    return bool(np.array_equal(a, np.broadcast_to(a[:1], a.shape)))


def _check(built, seed: int) -> None:
    exp = _expected(built, seed)
    held = built.randomized
    assert set(held) == set(exp), sorted(held)
    for k, v in exp.items():
        np.testing.assert_array_equal(held[k], v, err_msg=str(k))
    m = built.model
    mass, mu = m.body_mass.numpy().astype(np.float64), m.shape_material_mu.numpy().astype(np.float64)
    for name in (PLUG, CUBE):                                    # mass scales with each world's density draw
        d, b = exp[(name, "material.density")], mass[_index(built, name, "body")]
        np.testing.assert_allclose(b / b[:1], np.broadcast_to((d / d[0])[:, None], b.shape), rtol=1e-5, err_msg=name)
    qs, jq = m.joint_q_start.numpy(), m.joint_q.numpy().astype(np.float64)
    x = jq[qs[_index(built, CUBE, "joint")[:, 0]]]               # the cube's free joint: x first
    np.testing.assert_allclose(x, exp[(CUBE, "pose.0")], atol=1e-6)
    cable_mu = mu[_index(built, CABLE, "shape")]
    np.testing.assert_allclose(cable_mu, np.broadcast_to(exp[(CABLE, "material.surface.mu_dynamic")][:, None],
                                                         cable_mu.shape), rtol=1e-6)
    assert _same_in_every_world(mass[_index(built, CABLE, "body")])     # the cable has no density row
    assert _same_in_every_world(_coords(built, CABLE))                  # nor a start coordinate
    for name in (PLUG, CUBE):                                    # neither has a friction row
        assert _same_in_every_world(mu[_index(built, name, "shape")]), name


def test_tree_weld_scene_draws_each_object_its_own_rows(built):
    """At assembly (seed 0) and redrawn at another seed, every row is drawn under its own object's name and written
    to that object: the cube's x and mass, the plug's mass, the cable's friction; the cable's masses and joint
    coordinates stay the same in every world."""
    from alloy.scene.randomize import apply_randomization

    line = next(x for x in built.report.entry_options if x.startswith("mujoco weld "))
    assert "tree joint" in line, line
    assert built._plan is not None                               # the plan the draws read (not a `plan` attribute)
    assert [t.name for t in built._plan.objects] == [PLUG, CABLE, CUBE]  # the chain follows its parent
    assert [p.object for p in built._plan.scene.objects] == [PLUG, CUBE, CABLE]
    _check(built, 0)
    apply_randomization(built, SEED)
    _check(built, SEED)


def test_reset_draws_are_each_object_own_rows(built):
    """``Simulation.reset``'s draws (``_all_draws``) are each object's own rows, so after a draw at a seed they equal
    what the model holds and a reset at that seed writes nothing."""
    from alloy.scene.randomize import apply_randomization
    from alloy.sim import _all_draws, _differing

    apply_randomization(built, SEED)
    got, exp = _all_draws(built, built._plan, SEED), _expected(built, SEED)
    assert set(got) == set(exp), sorted(got)
    for k, v in exp.items():
        np.testing.assert_array_equal(got[k], v, err_msg=str(k))
    assert not any(d.any() for d in _differing(built, got).values())


def test_per_world_pos_row_keeps_the_randomized_coordinate():
    """A free box with one ``pos`` row per world and a randomized x starts at its x draw and the row's y and z, and a
    reset keeps it there; a box with rows only starts at its rows."""
    import alloy
    from alloy.scene.randomize import draw

    seed, lo, hi = 3, -0.25, -0.15
    rows = np.asarray([[0.0, 0.1 * w, 0.3 + 0.01 * w] for w in range(WORLDS)])
    s = alloy.Scene(worlds=WORLDS, dt=0.01)
    s.add_ground()
    s.add_box("drawn", size=(0.05, 0.05, 0.05), pos=rows, mass=0.1)
    s.add_box("rows", size=(0.05, 0.05, 0.05), pos=rows + (0.5, 0.0, 0.0), mass=0.1)
    s.randomize("drawn", "x", lo, hi)
    sim = alloy.build(s, device="cpu", seed=seed)
    x = np.asarray([draw(seed, w, "drawn.pose.0", lo, hi) for w in range(WORLDS)])
    for _ in range(2):                                           # as built, then after a reset
        p = sim.state(["drawn", "rows"])
        np.testing.assert_allclose(p["drawn.pos"][:, 0], x, atol=1e-6)
        np.testing.assert_allclose(p["drawn.pos"][:, 1:], rows[:, 1:], atol=1e-6)
        np.testing.assert_allclose(p["rows.pos"], rows + (0.5, 0.0, 0.0), atol=1e-6)
        sim.reset()
