# SPDX-License-Identifier: Apache-2.0
"""alloy.fit.params: leaf grammar, unit box, priors, start values, card leaf access."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.cards import types as ct
from alloy.core.quantity import Q
from alloy.fit.errors import FitError
from alloy.fit.params import (INIT_SIGMA, START_SIGMA, ParamSpace, get_leaf, new_pair_card, parse_leaf,
                              set_leaf)
from alloy.fit.types import FitParam


def _material(name: str, mu: float, e: float) -> ct.MaterialCard:
    note = "test"
    surf = ct.Surface(Q(mu, "measured", note=note), Q(0.8 * mu, "measured", note=note), Q(e, "measured", note=note))
    return ct.MaterialCard(name, Q(1000.0, "measured", note=note), surf)


def test_parse_material_and_pair_leaves():
    m = parse_leaf("material:pp.surface.mu_static")
    assert (m.kind, m.name, m.path, m.field, m.materials) == ("material", "pp", "surface.mu_static", "mu_static",
                                                             ("pp",))
    p = parse_leaf("pair:pp__steel.mu_dynamic")
    assert (p.kind, p.name, p.field, p.materials) == ("pair", "pp__steel", "mu_dynamic", ("pp", "steel"))


@pytest.mark.parametrize("leaf", [
    "pp.surface.mu_static",                 # no kind
    "material:pp",                          # no path
    "object:box.loads.grip_force",          # unknown kind
    "material:pp.surface.no_such_field",    # not a MaterialCard leaf
    "pair:steel__pp.mu_static",             # names not sorted
    "pair:pp.mu_static",                    # not a pair name
    "pair:pp__steel.density",               # not a PairCard numeric field
])
def test_parse_rejects_bad_leaves(leaf):
    with pytest.raises(FitError):
        parse_leaf(leaf)


def test_unit_box_linear_and_log_round_trip():
    space = ParamSpace((FitParam("pair:a__b.mu_static", 0.1, 1.1),
                        FitParam("material:a.surface.contact_stiffness", 1e4, 1e8, scale="log")))
    x = np.array([[0.6, 1e6], [0.1, 1e4], [1.1, 1e8]])
    u = space.to_unit(x)
    # linear: (0.6 - 0.1) / 1.0 = 0.5; log: (ln 1e6 - ln 1e4) / (ln 1e8 - ln 1e4) = 2/4 = 0.5
    np.testing.assert_allclose(u, [[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(space.from_unit(u), x, rtol=1e-12)
    np.testing.assert_allclose(space.from_unit([[-0.5, 1.7]]), [[0.1, 1e8]])  # clipped to the box


def test_prior_penalty_in_scale_space():
    space = ParamSpace((FitParam("pair:a__b.mu_static", 0.0, 1.0, prior=(0.5, 0.1)),
                        FitParam("material:a.surface.contact_stiffness", 1e4, 1e8, scale="log",
                                 prior=(math.log(1e6), 1.0)),
                        FitParam("pair:a__b.mu_dynamic", 0.0, 1.0)))
    pen = space.prior_penalty(np.array([[0.7, 1e6 * math.e, 0.3]]))
    # ((0.7 - 0.5)/0.1)^2 + ((ln(1e6 e) - ln 1e6)/1)^2 = 4 + 1 = 5; the third has no prior
    assert pen[0] == pytest.approx(5.0)


def test_initial_prior_start_and_centre():
    plain = ParamSpace((FitParam("pair:a__b.mu_static", 0.0, 2.0),))
    mean, sigma = plain.initial()
    assert mean.tolist() == [0.5] and sigma == INIT_SIGMA
    mean, sigma = plain.initial({"pair:a__b.mu_static": 0.5})
    assert mean.tolist() == [0.25] and sigma == START_SIGMA
    mean, sigma = plain.initial({"pair:a__b.mu_static": 9.0})   # clipped into the box
    assert mean.tolist() == [1.0]
    prior = ParamSpace((FitParam("pair:a__b.mu_static", 0.0, 2.0, prior=(1.5, 0.2)),))
    mean, sigma = prior.initial({"pair:a__b.mu_static": 0.5})   # the prior wins over a start value
    assert mean.tolist() == [0.75] and sigma == pytest.approx(0.1)


@pytest.mark.parametrize("bad", [
    FitParam("pair:a__b.mu_static", 1.0, 1.0),
    FitParam("pair:a__b.mu_static", 0.0, 1.0, scale="log"),
    FitParam("pair:a__b.mu_static", 0.0, math.inf),
    FitParam("pair:a__b.mu_static", 0.0, 1.0, prior=(0.5, 0.0)),
])
def test_bad_params_raise(bad):
    with pytest.raises(FitError):
        ParamSpace((bad,))


def test_duplicate_and_empty_params_raise():
    p = FitParam("pair:a__b.mu_static", 0.0, 1.0)
    with pytest.raises(FitError):
        ParamSpace((p, p))
    with pytest.raises(FitError):
        ParamSpace(())


def test_get_and_set_leaf_keep_other_fields():
    card = _material("pp", 0.4, 0.6)
    assert get_leaf(card, "surface.mu_static").value == 0.4
    assert get_leaf(card, "surface.contact_stiffness") is None      # unset optional leaf
    fitted = Q(0.45, "fit", (0.44, 0.46), note="fit-x")
    out = set_leaf(card, "surface.mu_static", fitted)
    assert out.surface.mu_static == fitted and out.surface.mu_dynamic == card.surface.mu_dynamic
    out = set_leaf(card, "surface.contact_stiffness", Q(3e6, "fit", note="fit-x"))   # created
    assert out.surface.contact_stiffness.value == 3e6
    with pytest.raises(FitError):
        set_leaf(card, "surface.nope", fitted)
    with pytest.raises(FitError):
        set_leaf(card, "elastic.youngs", fitted)                    # below an unset (None) node


def test_new_pair_card_harmonic():
    card = new_pair_card(_material("steel", 0.6, 0.8), _material("pp", 0.3, 0.4))
    assert (card.a, card.b) == ("pp", "steel")
    assert card.mu_static.value == pytest.approx(2 * 0.6 * 0.3 / 0.9)    # Drake 2ab/(a+b), D13
    assert card.restitution.value == pytest.approx(2 * 0.8 * 0.4 / 1.2)
    assert card.mu_static.source == "model" and "harmonic" in card.mu_static.note
