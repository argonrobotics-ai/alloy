# SPDX-License-Identifier: Apache-2.0
"""OSS-22: soft phrases parse into soft items: liquids filled or poured, grain fills and piles, cloth forms, cables
with length, gauge and attached ends, the default robot arm; bad requests raise errors that say what to change."""

from __future__ import annotations

import pytest

from alloy.compose import SceneSpec
from alloy.compose.parse import parse
from alloy.compose.soft.parse import normalise
from alloy.compose.soft.spec import SoftSpec
from alloy.compose.spec import ComposeError, ItemSpec


def _items(sentence: str) -> dict[str, ItemSpec]:
    return {i.query: i for i in parse(sentence).items}


def test_fill_by_volume_and_container_of_phrases():
    it = _items("a cup with 200 ml of water")
    assert (it["water"].relation, it["water"].target) == ("in", "cup")
    assert it["water"].soft == SoftSpec("liquid", "fill", volume_ml=200.0)
    it = _items("a bowl of rice and a glass half full of milk")
    assert it["rice"].soft.form == "fill" and it["rice"].target == "bowl"
    assert it["milk"].soft.fill == 0.5 and it["milk"].target == "glass"
    assert _items("a bin with 0.5 kg of sand")["sand"].soft.mass_g == 500.0
    assert _items("a cup filled with water")["water"].soft.fill == 0.85


def test_piles_pours_and_names_that_only_look_soft():
    assert _items("a pile of 300 g of sand on a table")["sand"].soft == SoftSpec("grain", "pile", mass_g=300.0)
    pour = _items("pour water from a bottle into a cup")["water"]
    assert (pour.soft.form, pour.soft.source, pour.target) == ("pour", "bottle", "cup")
    assert _items("pour 100 ml of honey into a jar")["honey"].soft.source is None
    bottle = _items("a water bottle and a mug")["bottle"]
    assert bottle.soft is None and bottle.words == ("water",)
    for sentence, noun, word in (("a paper bag on a table", "bag", "paper"), ("a wire basket", "basket", "wire")):
        item = _items(sentence)[noun]                           # a cloth / cable word before an object noun
        assert item.soft is None and item.words == (word,), sentence
    cup = _items("a paper cup with water")
    assert cup["cup"].soft is None and cup["water"].target == "cup"
    assert _items("a cable 2 m long")["cable"].soft.kind == "cable"        # a size word after it: still a cable


def test_cloth_forms():
    assert _items("a towel lying flat on a table")["towel"].soft.form == "flat"
    drape = _items("a towel draped over a bowl")["towel"]
    assert (drape.relation, drape.target, drape.soft.form) == ("over", "bowl", "draped")
    assert _items("a towel hanging over the edge of the table")["towel"].soft.form == "hanging"
    assert _items("a towel folded in half")["towel"].soft.form == "folded"
    sized = _items("a 50 x 30 cm towel")["towel"].soft
    assert (sized.length_m, sized.width_m) == (0.5, 0.3)


def test_cables_lengths_gauges_and_ends():
    c = _items("a 2 m usb cable plugged into a laptop")
    assert c["cable"].soft.length_m == 2.0 and c["cable"].soft.attach == ("laptop",) and "laptop" in c
    assert c["cable"].words == ("usb",)
    c = _items("a 3 mm cable from the laptop to the monitor")["cable"].soft
    assert c.diameter_mm == 3.0 and c.attach == ("laptop", "monitor")


def test_robot_arm_mixed_scene():
    spec = parse("a robot arm next to a bowl of rice and a towel")
    assert [r.path for r in spec.robots] == ["robots/emb_e2_panda/scene.xml"]
    assert {i.query for i in spec.items} == {"bowl", "rice", "towel"}


def test_soft_targets_default_and_specs_round_trip():
    spec = parse("a mug on a towel")
    towel = next(i for i in spec.items if i.query == "towel")
    assert towel.soft.form == "flat" and spec.items[0].query == "towel"            # the cloth goes down first
    again = SceneSpec.from_dict(parse("a cup with 200 ml of water").to_dict())
    assert again.items[1].soft.volume_ml == 200.0
    assert normalise([ItemSpec("rope")])[0].soft.kind == "cable"


@pytest.mark.parametrize("sentence, word", [("water on the table", "container"),
                                            ("pour water", "pours into nothing")])
def test_requests_that_cannot_be_built_say_what_to_change(sentence, word):
    with pytest.raises(ComposeError) as e:
        parse(sentence)
    assert word in e.value.message and e.value.fix


@pytest.mark.parametrize("data, word", [({"kind": "liquid", "fill": 0.95}, "no room"),
                                        ({"kind": "cloth", "volume_ml": 5}, "no volume"),
                                        ({"kind": "liquid", "attach": ["x"]}, "attach"),
                                        ({"kind": "grain", "form": "draped"}, "cannot be"),
                                        ({"kind": "liquid", "volume_ml": 1, "mass_g": 1}, "one amount")])
def test_soft_spec_validation(data, word):
    with pytest.raises(ComposeError) as e:
        SoftSpec.from_dict(data)
    assert word in e.value.message


def test_tilting_pours():
    w = _items("pour 180 ml of water from a container tilted 35 degrees per second to 100 degrees into a cup")["water"]
    assert (w.soft.source, w.soft.tilt_deg_s, w.soft.tilt_to_deg, w.target) == ("container", 35.0, 100.0, "cup")
    assert _items("tilt 150 ml of water from a jar into a mug")["water"].soft.tilt_deg_s == 40.0
    with pytest.raises(ComposeError):
        SoftSpec("liquid", "fill", tilt_deg_s=40.0)
