# SPDX-License-Identifier: Apache-2.0
"""The YCB Object and Model Set: object list, measured masses and surface materials.

Sources (short keys used in card notes):

- ``ICAR15``: Calli, Walsman, Singh, Srinivasa, Abbeel, Dollar, "The YCB object and model set: towards common
  benchmarks for manipulation research", ICAR 2015, Table I (masses, digital scale). Its IDs 1-52 equal the dataset
  IDs; from 53 on the dataset inserts 055_baseball, so ICAR IDs 55-58 are dataset 056-059 and ICAR 60-61 are 061-062.
- ``IJRR17``: Calli et al., "Yale-CMU-Berkeley dataset for robotic manipulation research", IJRR 36(3):261-268, 2017,
  Table 1 (dataset IDs). Its third column from 56 on repeats values of the second column (typesetting error), so only
  055_baseball (148 g, the regulation 142-149 g range) is taken from it; the rest come from ICAR15.
- ``RAM15v1``: Calli et al., "Benchmarking in manipulation research: the YCB object and model set and benchmarking
  protocols", arXiv:1502.03143v1 (2015), Table II: the per-size marble masses that the later tables list as N/A.
- Meshes and licence: http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/ ("License for the data set: Creative
  Commons Attribution 4.0 International (CC BY 4.0)"); ``google_16k`` scans (metres, +Z up).

``mass_g`` None means no published mass for the scanned object: the card gets an ESTIMATE density by ``estimate``
and a row in the sourcing queue. ``estimate`` is ``"material"`` (solid base material), ``"packed:<f>"`` (base material
density x packing fraction f, for a scanned pile or jar of parts) or ``"like:<card>"`` (the fitted effective density
of another YCB card of the same make). ``material`` names the library material whose surface (friction, restitution) and
elastic values the object's effective material copies; ``material_note`` says why.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
LICENCE = "CC-BY-4.0"
ATTRIBUTION = ("YCB Object and Model Set (Calli et al. 2015, 2017), ycbbenchmarks.com, licensed CC BY 4.0; "
               "google_16k scan, processed by Alloy (watertight remesh, CoACD hulls)")
REF_ICAR = "Calli et al., ICAR 2015, Table I"
REF_IJRR = "Calli et al., IJRR 36(3) 2017, Table 1"
REF_RAM = "Calli et al., arXiv:1502.03143v1 2015, Table II"


@dataclass(frozen=True)
class YcbObject:
    """One YCB dataset entry with a ``google_16k`` scan (076_timer is left out: every google_* archive of it is a
    truncated 45-byte file upstream, checked 2026-11-06)."""

    dataset_id: str                   # e.g. "025_mug" (the archive and folder name)
    mass_g: float | None
    ref: str                          # citation of mass_g ("" when None)
    material: str                     # library material for surface and elastic values
    material_note: str
    estimate: str = ""                # rule for the ESTIMATE density when mass_g is None (see module docstring)

    @property
    def card_name(self) -> str:
        """Alloy card name (``-`` becomes ``_``).

        Example::

            by_card_name()["ycb_065_a_cups"].dataset_id   # "065-a_cups"
        """
        return "ycb_" + self.dataset_id.replace("-", "_")

    @property
    def url(self) -> str:
        """Archive of the ``google_16k`` scan.

        Example::

            by_card_name()["ycb_025_mug"].url   # ".../data/google/025_mug_google_16k.tgz"
        """
        return f"{BASE_URL}google/{self.dataset_id}_google_16k.tgz"

    @property
    def mesh_member(self) -> str:
        """Path of the textured mesh inside the archive.

        Example::

            by_card_name()["ycb_025_mug"].mesh_member   # "025_mug/google_16k/textured.obj"
        """
        return f"{self.dataset_id}/google_16k/textured.obj"


_CARTON = "printed paperboard carton; nearest library surface"
_CAN = "tinplate steel can (label or print on the steel)"
_TOYFRUIT = "hollow plastic toy fruit; soft PVC surface assumed"
_KITCHEN = "rigid plastic kitchenware (assumed PP or melamine); PP surface"
_CLAMP = "plastic spring clamp (glass-filled nylon body assumed)"
_BALL = "sports ball cover (leather, felt or vinyl); nearest library surface"
_TOY = "moulded ABS toy part"


def _o(i: str, g: float | None, ref: str, mat: str, note: str, estimate: str = "") -> YcbObject:
    return YcbObject(i, g, ref if g is not None else "", mat, note, estimate)


I, J, R = REF_ICAR, REF_IJRR, REF_RAM
_AIRPLANE = "like:ycb_072_a_toy_airplane"
OBJECTS: tuple[YcbObject, ...] = (
    _o("002_master_chef_can", 414, I, "steel_tinplate", _CAN),
    _o("003_cracker_box", 411, I, "cardboard_corrugated", _CARTON),
    _o("004_sugar_box", 514, I, "cardboard_corrugated", _CARTON),
    _o("005_tomato_soup_can", 349, I, "steel_tinplate", _CAN),
    _o("006_mustard_bottle", 603, I, "hdpe", "squeeze bottle, HDPE assumed"),
    _o("007_tuna_fish_can", 171, I, "steel_tinplate", _CAN),
    _o("008_pudding_box", 187, I, "cardboard_corrugated", _CARTON),
    _o("009_gelatin_box", 97, I, "cardboard_corrugated", _CARTON),
    _o("010_potted_meat_can", 370, I, "steel_tinplate", _CAN),
    _o("011_banana", 66, I, "vinyl_soft", _TOYFRUIT),
    _o("012_strawberry", 18, I, "vinyl_soft", _TOYFRUIT),
    _o("013_apple", 68, I, "vinyl_soft", _TOYFRUIT),
    _o("014_lemon", 29, I, "vinyl_soft", _TOYFRUIT),
    _o("015_peach", 33, I, "vinyl_soft", _TOYFRUIT),
    _o("016_pear", 49, I, "vinyl_soft", _TOYFRUIT),
    _o("017_orange", 47, I, "vinyl_soft", _TOYFRUIT),
    _o("018_plum", 25, I, "vinyl_soft", _TOYFRUIT),
    _o("019_pitcher_base", 178, I, "pp_homopolymer", _KITCHEN),
    _o("021_bleach_cleanser", 1131, I, "hdpe", "household chemical bottle, HDPE assumed"),
    _o("022_windex_bottle", 1022, I, "pet", "spray bottle, PET assumed"),
    _o("024_bowl", 147, I, "pp_homopolymer", _KITCHEN),
    _o("025_mug", 118, I, "pp_homopolymer", _KITCHEN),
    _o("026_sponge", 6.2, I, "pu_foam_sponge", "kitchen sponge, PU foam"),
    _o("028_skillet_lid", 652, I, "glass_soda_lime", "glass lid with a steel rim; glass surface"),
    _o("029_plate", 279, I, "pp_homopolymer", _KITCHEN),
    _o("030_fork", 34, I, "pp_homopolymer", _KITCHEN),
    _o("031_spoon", 30, I, "pp_homopolymer", _KITCHEN),
    _o("032_knife", 31, I, "pp_homopolymer", _KITCHEN),
    _o("033_spatula", 51.5, I, "nylon_pa6", "nylon kitchen spatula assumed"),
    _o("035_power_drill", 895, I, "abs", "power tool housing, ABS assumed"),
    _o("036_wood_block", 729, I, "wood_pine", "softwood block (species not published)"),
    _o("037_scissors", 82, I, "stainless_304", "stainless blades; plastic handles not modelled separately"),
    _o("038_padlock", 304, I, "steel_carbon", "steel padlock body assumed"),
    _o("040_large_marker", 15.8, I, "pp_homopolymer", "marker barrel, PP assumed"),
    _o("042_adjustable_wrench", 252, I, "steel_crv", "chrome-vanadium tool steel"),
    _o("043_phillips_screwdriver", 97, I, "abs", "plastic handle (contact surface); steel shaft not separate"),
    _o("044_flat_screwdriver", 98.4, I, "abs", "plastic handle (contact surface); steel shaft not separate"),
    _o("048_hammer", 665, I, "steel_carbon", "steel head; handle not modelled separately"),
    _o("050_medium_clamp", 59, I, "nylon_pa6", _CLAMP),
    _o("051_large_clamp", 125, I, "nylon_pa6", _CLAMP),
    _o("052_extra_large_clamp", 202, I, "nylon_pa6", _CLAMP),
    _o("053_mini_soccer_ball", 123, I, "vinyl_soft", _BALL),
    _o("054_softball", 191, I, "vinyl_soft", _BALL),
    _o("055_baseball", 148, J, "vinyl_soft", _BALL),
    _o("056_tennis_ball", 58, I, "cotton_woven", _BALL),
    _o("057_racquetball", 41, I, "nitrile_70a", "rubber ball; nitrile 70A surface"),
    _o("058_golf_ball", 46, I, "hdpe", "ionomer cover; HDPE surface as nearest"),
    _o("059_chain", 98, I, "steel_zinc_plated", "zinc-plated steel chain"),
    _o("061_foam_brick", 28, I, "pu_foam_sponge", "foam brick"),
    _o("062_dice", 5.2, I, "pmma_cast", "solid plastic die (1.2 g/cm^3 from mass and size)"),
    _o("063-a_marbles", None, "", "glass_soda_lime", "mesh bag of marbles scanned as one object", "packed:0.6"),
    _o("063-b_marbles", 59, R, "glass_soda_lime", "solid glass marble (XL, 35.2 mm)"),
    *(_o(f"065-{c}_cups", g, I, "pp_homopolymer", "stacking cup, plastic (ICAR mass list in size order)")
      for c, g in zip("abcdefghij", (13, 14, 17, 19, 21, 26, 28, 31, 35, 38))),
    _o("070-a_colored_wood_blocks", None, "", "wood_beech", "jar of painted wood cubes (the whole set) as one object",
       "packed:0.5"),
    _o("070-b_colored_wood_blocks", 10.8, I, "wood_beech", "painted hardwood cube (species not published)"),
    _o("071_nine_hole_peg_test", None, "", "wood_pine", "wooden peg board; the published 1435 g (ICAR15) is the "
       "whole kit and would give 2.3 g/cm^3 for this board", "material"),
    _o("072-a_toy_airplane", 570, I, "abs", "assembled toy airplane (the published mass is the whole toy)"),
    *(_o(f"072-{c}_toy_airplane", None, "", "abs", _TOY, _AIRPLANE) for c in "bcde"),
    *(_o(f"073-{c}_lego_duplo", None, "", "abs", "hollow ABS building brick or assembly (set mass 523 g, ICAR15)",
         _AIRPLANE) for c in "abcdefg"),
    _o("077_rubiks_cube", None, "", "abs", "puzzle cube, hollow ABS", _AIRPLANE),
)


def by_card_name() -> dict[str, YcbObject]:
    """Card name -> entry.

    Example::

        by_card_name()["ycb_025_mug"].mass_g   # 118
    """
    return {o.card_name: o for o in OBJECTS}
