# SPDX-License-Identifier: Apache-2.0
"""P3-2b embodiments: the robots of the Gym-Anything for Robotics comparison, as Alloy scenes (one MJCF each).

Each :class:`Embodiment` names its pinned model source and licence, the world it stands in, and its start pose. The
scenes are built by :mod:`alloy.tasks.emb_build` and written by :mod:`alloy.tasks.emb_scene` to
``library/robots/emb_<key>/scene.xml``. The bully targets in :mod:`alloy.tasks.emb_targets` run them, and WP8's
``bench/xbench/ours/scene_pack.py`` turns each into a rival-engine pack (brief [internal note], P3-2b).

Model sources live outside the repo, at the pinned commits below (``ALLOY_MENAGERIE``, ``ALLOY_GYM_ANYTHING``). The
converted scene and its meshes in ``library/robots`` are the vendored copy, with each source file's sha256 in
``source.json``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from alloy.core.errors import TraceError

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
GYM_ANYTHING_URL = "https://github.com/gym-anything/robotics"
GYM_ANYTHING_COMMIT = "febb260b0d34d9d3641bc2b3545acabeacf402bb"
UNITREE_ROS_URL = "https://github.com/unitreerobotics/unitree_ros"
UNITREE_ROS_COMMIT = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"   # gym-anything's h1_2 assets.lock.json pin
_REFS = Path("[internal note]").expanduser()


def menagerie_root() -> Path:
    return Path(os.environ.get("ALLOY_MENAGERIE", _REFS / "menagerie-98a70c5")).expanduser()


def gym_anything_root() -> Path:
    return Path(os.environ.get("ALLOY_GYM_ANYTHING", _REFS / "gym-anything-robotics")).expanduser()


def gar_assets_root() -> Path:
    """gym-anything's installed robot assets (``python -m robots.models.<id>.install --dest <root>/<id>``)."""
    return Path(os.environ.get("ALLOY_GAR_ASSETS", _REFS / "gar-assets")).expanduser()


def franka_description_root() -> Path:
    return Path(os.environ.get("ALLOY_FRANKA_DESCRIPTION", _REFS / "franka_description-7aeeddc")).expanduser()


def playground_root() -> Path:
    """MuJoCo Playground's pinned LEAP file and licence (``alloy.tasks.emb_robots.PLAYGROUND_LEAP_SOURCE``)."""
    return Path(os.environ.get("ALLOY_PLAYGROUND", _REFS / "mujoco_playground-124a73f")).expanduser()


FRANKA_DESCRIPTION_URL = "https://github.com/frankarobotics/franka_description"
FRANKA_DESCRIPTION_COMMIT = "7aeeddc449edf8d62b594f9e36a81da53e7796f9"  # gym-anything's mobile-fr3-duo lock pin
_ROOTS = {"menagerie": menagerie_root, "gym_anything": gym_anything_root,
          "h1_2": lambda: gar_assets_root() / "h1_2", "fr3_duo": lambda: gar_assets_root() / "mobile-fr3-duo"}
_REFS_OF = {"menagerie": (MENAGERIE_URL, MENAGERIE_COMMIT), "gym_anything": (GYM_ANYTHING_URL, GYM_ANYTHING_COMMIT),
            "h1_2": (UNITREE_ROS_URL, UNITREE_ROS_COMMIT),
            "fr3_duo": (FRANKA_DESCRIPTION_URL, FRANKA_DESCRIPTION_COMMIT)}  # fmt: skip


@dataclass(frozen=True)
class Embodiment:
    key: str                     # e1_fr3: scene emb_e1_fr3, bully target emb_e1_fr3
    title: str
    world: str                   # tabletop | legged | hand
    files: tuple[str, ...]       # source model files, relative to the source root (the first is the robot)
    licences: tuple[str, ...]    # licence files, relative to the source root
    licence: str                 # SPDX-style summary recorded in the pack
    source: str = "menagerie"    # menagerie | gym_anything | h1_2 | fr3_duo (assets under ALLOY_GAR_ASSETS)
    keyframe: str = ""           # the source keyframe that is the start state ("" = the builder sets it)
    profile: str = ""            # gym-anything robot.json whose actuation profile gives the gains (URDF robots)
    base: str = ""               # a variant's verdict embodiment (its robot builder and sources)
    contact: str = ""            # "" = contacts as shipped (the verdict scene); "stiff" / "route" = a contact variant

    @property
    def name(self) -> str:
        return f"emb_{self.key}"

    @property
    def builder(self) -> str:
        return self.base or self.key

    def source_root(self) -> Path:
        return _ROOTS[self.source]()

    def source_ref(self) -> tuple[str, str]:
        return _REFS_OF[self.source]


EMBODIMENTS: dict[str, Embodiment] = {e.key: e for e in (
    Embodiment("e1_fr3", "Franka FR3 + Franka Hand", "tabletop",
               ("franka_fr3/fr3.xml", "franka_emika_panda/hand.xml"),
               ("franka_fr3/LICENSE", "franka_emika_panda/LICENSE"), "Apache-2.0 (mujoco_menagerie franka_fr3, "
               "franka_emika_panda hand)"),
    Embodiment("e2_panda", "Franka Panda + Franka Hand", "tabletop", ("franka_emika_panda/panda.xml",),
               ("franka_emika_panda/LICENSE",), "Apache-2.0 (mujoco_menagerie franka_emika_panda)"),
    Embodiment("e3_ur10e_2f85", "Universal Robots UR10e + Robotiq 2F-85", "tabletop",
               ("universal_robots_ur10e/ur10e.xml", "robotiq_2f85/2f85.xml"),
               ("universal_robots_ur10e/LICENSE", "robotiq_2f85/LICENSE"),
               "BSD-3-Clause (mujoco_menagerie universal_robots_ur10e, robotiq_2f85)"),
    Embodiment("e4_fr3_bimanual", "Two Franka FR3 + Franka Hand (bimanual)", "tabletop",
               ("franka_fr3/fr3.xml", "franka_emika_panda/hand.xml"),
               ("franka_fr3/LICENSE", "franka_emika_panda/LICENSE"), "Apache-2.0 (mujoco_menagerie franka_fr3, "
               "franka_emika_panda hand)"),
    Embodiment("e5_fr3_duo", "Franka Mobile FR3 Duo (TMR base, spine, two FR3 + hands)", "mobile",
               ("mobile_fr3_duo.urdf", "GENERATED.json"), ("LICENSE",),
               "Apache-2.0 (frankarobotics/franka_description; gains and pose: gym-anything/robotics "
               "mobile-fr3-duo, MIT)", source="fr3_duo", profile="src/robots/models/mobile-fr3-duo/robot.json"),
    Embodiment("e6_g1", "Unitree G1 with Dex3 hands (43 actuators)", "legged",
               ("unitree_g1/g1_with_hands.xml",), ("unitree_g1/LICENSE",),
               "BSD-3-Clause (mujoco_menagerie unitree_g1, Unitree Robotics)", keyframe="stand"),
    Embodiment("e7_h1_2", "Unitree H1-2 with Inspire hands (39 actuators)", "legged",
               ("robots/h1_2_description/h1_2.urdf",), ("LICENSE",),
               "BSD-3-Clause (unitreerobotics/unitree_ros h1_2_description; gains: gym-anything/robotics h1_2, MIT)",
               source="h1_2", profile="src/robots/models/h1_2/robot.json"),
    Embodiment("e8_go2", "Unitree Go2 quadruped (go2.xml + go2_mjx.xml's position actuators)", "legged",
               ("unitree_go2/go2.xml", "unitree_go2/go2_mjx.xml"), ("unitree_go2/LICENSE",),
               "BSD-3-Clause (mujoco_menagerie unitree_go2, Unitree Robotics)", keyframe="home"),
    Embodiment("e9_leap", "LEAP Hand (right)", "hand", ("leap_hand/right_hand.xml",), ("leap_hand/LICENSE",),
               "MIT (mujoco_menagerie leap_hand)"),
)}  # fmt: skip


STIFF_SOLREF = (0.005, 1.0)          # [s, -] the contact variant's solref on every geom (>= 2 x the 2 ms step)
DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)   # MuJoCo's default; the variant's floor for softer shipped solimp


def stiff_variant(e: Embodiment) -> Embodiment:
    """``<key>_stiff``: the verdict scene with every geom's contact solref set to ``STIFF_SOLREF``, and any solimp
    softer than MuJoCo's default raised to it (Go2's feet), nothing else changed (WP8, 2026-11-03: a model-variant row beside the verdict, for robots whose shipped contacts rest deeper
    than 5 mm or trip S1 at rest on Alloy's route; read only by engines that use solref)."""
    return replace(e, key=f"{e.key}_stiff", title=f"{e.title}, contact variant solref {STIFF_SOLREF[0]} s",
                   base=e.key, contact="stiff")  # fmt: skip


STIFF_KEYS = ("e3_ur10e_2f85", "e4_fr3_bimanual", "e5_fr3_duo", "e6_g1", "e7_h1_2", "e8_go2")  # S1 from contact softness
EMBODIMENTS |= {f"{k}_stiff": stiff_variant(EMBODIMENTS[k]) for k in STIFF_KEYS}

ROUTE_DT = 0.000625                  # [s] Alloy's route step for every emb bully target (32 steps per 20 ms tick)
ROUTE_SOLREF = (2 * ROUTE_DT, 1.0)   # [s, -] the route variant's solref on every geom: MuJoCo's floor, 2 dt


def route_variant(e: Embodiment) -> Embodiment:
    """``<key>_route``: the verdict scene with every geom's contact solref set to ``ROUTE_SOLREF`` and, as in
    ``_stiff``, a solimp softer than MuJoCo's default raised to it (Go2's feet); nothing else changed (coordinator
    2026-11-05 21:04Z: the SB-7 embodiment classes, so SB-7 measures Alloy rather than the sources' soft contacts; the
    verdict scenes stay as shipped, for cross-engine rows). Valid at Alloy's route step only: an engine at the
    sources' 2 ms step clamps the time constant to 4 ms."""
    return replace(e, key=f"{e.key}_route", base=e.key, contact="route",
                   title=f"{e.title}, route contact variant solref {ROUTE_SOLREF[0]} s (Alloy's route step only)")


ROUTE_KEYS = ("e1_fr3", "e2_panda", *STIFF_KEYS)   # every embodiment whose SB-7 S1 come from its source contacts
EMBODIMENTS |= {f"{k}_route": route_variant(EMBODIMENTS[k]) for k in ROUTE_KEYS}
CONTACT_SOLREF = {"stiff": STIFF_SOLREF, "route": ROUTE_SOLREF}   # Embodiment.contact -> every geom's solref


def embodiment(key: str) -> Embodiment:
    """The embodiment ``key`` (``e1_fr3``) or its scene / target name (``emb_e1_fr3``)."""
    k = key.removeprefix("emb_")
    if k not in EMBODIMENTS:
        raise TraceError(f"no embodiment {key!r}; known: {sorted(EMBODIMENTS)}")
    return EMBODIMENTS[k]
