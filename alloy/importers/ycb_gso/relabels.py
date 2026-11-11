# SPDX-License-Identifier: Apache-2.0
"""Google Scanned Objects whose upstream title names a different object than the mesh shows.

Found by a CLIP sweep of every rendered card against its own title, then confirmed by eye on our thumbnails and the
upstream ``thumbnails/0.jpg`` (2026-11-06). The card is named for what the mesh shows, keeps the upstream source and
says "upstream title ... is a mislabel" in its header, and its mass is estimated for the real object (the title's own
product weight would be wrong). Keys are the upstream model names (Gazebo Fuel, GoogleResearch).
"""

from __future__ import annotations

from dataclasses import dataclass

from alloy.importers.ycb_gso.estimates import CATEGORY, CategoryRule


@dataclass(frozen=True)
class Relabel:
    """The card name, what the mesh is, and how to estimate its mass."""

    card: str
    what: str
    rule: CategoryRule


_PLUSH = CATEGORY["Stuffed Toys"]
_CAN_OPENER = CategoryRule("abs", mass=0.18, unc=(0.12, 0.25),
                           note="manual can opener with soft-grip handles; typical mass, scan volume not used")
RELABELS: dict[str, Relabel] = {
    "Great_Dinos_Triceratops_Toy": Relabel(
        "gso_android_figure_green", "green Android collectible figure",
        CategoryRule("vinyl_soft", mass=0.08, unc=(0.05, 0.12),
                     note="vinyl collectible figure, same make and size as gso_android_figure_orange (99 cm^3)")),
    "Granimals_20_Wooden_ABC_Blocks_Wagon": Relabel("gso_can_opener_black", "black manual can opener", _CAN_OPENER),
    "Granimals_20_Wooden_ABC_Blocks_Wagon_g2TinmUGGHI": Relabel(
        "gso_can_opener_black_2", "black manual can opener (second scan)", _CAN_OPENER),
    "ProSport_Harness_to_Booster_Seat": Relabel(
        "gso_salad_plate", "dinner plate (0.23 m) with a salad",
        CategoryRule("stoneware_glazed", mass=0.6, unc=(0.4, 0.8),
                     note="ceramic dinner plate 0.4-0.6 kg plus about 0.15 kg of salad; engineering estimate")),
    "Neat_Solutions_Character_Bib_2_pack": Relabel(
        "gso_square_bowl_white", "white square bowl (0.2 m)",
        CategoryRule("stoneware_glazed", mass=0.5, unc=(0.2, 0.9),
                     note="0.2 m square serving bowl, ceramic or melamine; engineering estimate")),
    "Calphalon_Kitchen_Essentials_12_Cast_Iron_Fry_Pan_Black": Relabel(
        "gso_serving_bowl_white", "white serving bowl (0.32 m)",
        CategoryRule("stoneware_glazed", mass=1.5, unc=(0.8, 2.2),
                     note="0.32 m ceramic serving bowl; engineering estimate")),
    "Avengers_Thor_PLlrpYniaeB": Relabel(
        "gso_storage_bin_beige", "beige plastic stacking bin (0.28 m)",
        CategoryRule("pp_homopolymer", mass=0.35, unc=(0.2, 0.6),
                     note="0.28 x 0.28 x 0.13 m polypropylene stacking bin; engineering estimate")),
    "Ortho_Forward_Facing": Relabel("gso_plush_giraffe", "plush giraffe", _PLUSH),
    "Ortho_Forward_Facing_3Q6J2oKJD92": Relabel("gso_plush_hippo_blue", "blue plush hippo", _PLUSH),
    "Ortho_Forward_Facing_QCaor9ImJ2G": Relabel("gso_plush_koala", "grey plush koala", _PLUSH),
    "Ortho_Forward_Facing_CkAW6rL25xH": Relabel(
        "gso_helmet_black", "black open-face helmet (0.28 m)",
        CategoryRule("abs", mass=0.9, unc=(0.4, 1.5), note="open-face motorcycle or sports helmet; engineering estimate")),
}
