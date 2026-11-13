# SPDX-License-Identifier: Apache-2.0
"""Per-object materials take their base material's pair cards by name (``MaterialCard.pairs_from``,
``alloy.scene.pairs.pair_card``), AUDIT S2-010 (Gemini 0.1.2): ``mass=`` on each of N boxes made N materials and
N (N + 1) / 2 copied pair cards, written, read and validated at every build (120 boxes: ``check --quick`` 52 s on a
CPU, 500 boxes over 300 s). The laws: the saved cards grow with the objects, not with their pairs, and the contact
rows are the base card's values (an object with its own friction= takes none)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.cards.library import CardLibrary, default_root
from alloy.cards.loader import load_material
from alloy.scene.pairs import pair_card


def _boxes(n: int, **extra) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    for i in range(n):
        s.add_box(f"b{i}", size=(0.03, 0.03, 0.03), pos=(0.06 * (i % 10), 0.06 * (i // 10), 0.015),
                  mass=0.05 + 0.001 * i, **extra)
    return s


def test_saved_cards_grow_with_the_objects_not_their_pairs(tmp_path):
    _boxes(40).save(tmp_path)
    assert not list((tmp_path / "pairs").glob("*.yaml"))                 # hundreds of copied cards before the fix
    mats = sorted((tmp_path / "materials").glob("abs_for_*.yaml"))
    assert len(mats) == 40
    assert {load_material(p).pairs_from for p in mats} == {"abs"}


def test_a_per_object_material_takes_the_base_card_keyed_onto_it(tmp_path):
    s = _boxes(2)
    s.add_box("own", size=(0.03, 0.03, 0.03), pos=(0.5, 0.0, 0.015), friction=0.9)
    s.save(tmp_path)
    lib = CardLibrary(roots=[tmp_path, default_root()])
    base = lib.pair("abs", "abs")
    card = pair_card(lib, (), "abs_for_b0", "abs_for_b1")
    assert (card.a, card.b) == ("abs_for_b0", "abs_for_b1") and card.mu_dynamic == base.mu_dynamic
    assert pair_card(lib, (), "abs_for_b0", "abs").mu_static == base.mu_static
    assert pair_card(lib, (), "abs_for_b0", "fr4").mu_dynamic == lib.pair("abs", "fr4").mu_dynamic
    assert load_material(tmp_path / "materials" / "abs_for_own.yaml").pairs_from is None
    assert pair_card(lib, (), "abs_for_b0", "abs_for_own") is None                 # its own friction: no card


def test_the_contact_rows_carry_the_base_card():
    """Two boxes given a mass (abs, abs/abs card mu 0.575; the material's own mu 0.4) get one MuJoCo row at the
    card's mu; a box with its own friction 0.9 meets them at the harmonic mean, as before the fix. ~5 s."""
    s = _boxes(2)
    s.add_box("own", size=(0.03, 0.03, 0.03), pos=(0.5, 0.0, 0.015), friction=0.9)
    sim = alloy.build(s, device="cpu")
    m = sim.engine.model
    body = m.shape_body.numpy()
    label = {s_: (m.body_label[b] if b >= 0 else "ground") for s_, b in enumerate(body)}
    rows = {frozenset((label[int(g1)], label[int(g2)])): float(f[0]) for g1, g2, f in
            zip(m.mujoco.pair_geom1.numpy(), m.mujoco.pair_geom2.numpy(), m.mujoco.pair_friction.numpy())}
    mu_card = float(CardLibrary().pair("abs", "abs").mu_dynamic.value)
    assert rows[frozenset(("b0", "b1"))] == pytest.approx(mu_card, rel=1e-6)
    harmonic = 2.0 * 0.4 * 0.9 / (0.4 + 0.9)
    assert rows[frozenset(("b0", "own"))] == pytest.approx(harmonic, rel=1e-6)
    assert np.isfinite(list(rows.values())).all()
