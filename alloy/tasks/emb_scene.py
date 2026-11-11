# SPDX-License-Identifier: Apache-2.0
"""Write a P3-2b embodiment scene: ``library/robots/emb_<key>/`` and its scene card (offline, CPU).

The scene from :func:`alloy.tasks.emb_build.build_spec` is compiled by MuJoCo and written by the same converter as the
recorded task scenes (:func:`alloy.tasks.mjb_convert.write_scene`): explicit values everywhere, meshes in the shared
content-addressed store, and every collision hull above 64 vertices split into error-bounded convex pieces
(:mod:`alloy.tasks.split_scene`, within 0.2 mm, never outside the hull). The ``init`` keyframe is written with it.
Next to ``scene.xml``:
- ``source.json``: pinned source (repository, commit, sha256 of every source file used), licence, changes;
- the source licence files (``LICENSE.<n>``);
- ``hull_split.json`` (the split report).

The scene card ``library/scenes/task_emb_<key>.yaml`` is the replay-tier card (the whole MJCF as one robot), as for
the recorded task scenes.
"""

from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from alloy.tasks import emb_build
from alloy.tasks.emb_firm import FIRM_CONTACTS, FIRM_SOLIMP, FIRM_SOLMIX, FIRM_SOLREF
from alloy.tasks.emb_robots import PLAYGROUND_LEAP, PLAYGROUND_LEAP_SOURCE
from alloy.tasks.emb_specs import EMBODIMENTS, ROUTE_DT, ROUTE_SOLREF, STIFF_SOLREF, Embodiment, playground_root
from alloy.tasks.hull_split import SplitConfig
from alloy.tasks.msh import sha256_file, write_manifest
from alloy.tasks.split_scene import REPORT_NAME, SceneSplit
from alloy.cards.library import default_root

log = logging.getLogger("alloy.tasks")

REPO = Path(__file__).resolve().parents[2]
ROBOTS = default_root() / "robots"
SCENES = default_root() / "scenes"
SOURCE_JSON = "source.json"
SPLIT = SceneSplit(".", SplitConfig(max_verts=64, tol_m=2e-4, max_pieces=64))
CHANGES = (
    "visual-only geoms and their meshes, materials and textures dropped",
    "source keyframes dropped; keyframe 'init' = start state (robot start pose, ctrl = its joint targets, objects at "
    "rest), both clipped into the joint ranges; free-floating robots lifted so the deepest floor contact just "
    "touches",
    "collision hulls above 64 vertices split into convex pieces within 0.2 mm (hull_split.json)",
    "robot self-pairs overlapping > 0.5 mm at the start pose, or at qpos0 (where alloy.build starts a robot), "
    "excluded: H1-2's wrist yaw | thumb proximal, each hand; the Franka arms' link5 | hand (Panda) or link7 (FR3), "
    "whose hulls overlap 5.6-43 mm at qpos0 and 15-33 mm at joint6's lower limit; none elsewhere",
    "options: jacobian sparse; <flag multiccd=\"disable\"/> written out (Alloy's MJCF route, as for T12: Newton "
    "reads an absent flag as off, MuJoCo as on; explicit, both read off)",
)
HAND_CHANGE = ("Franka Hand: tendon 'split' + tendon actuator -> joint position actuator 'hand_finger' on "
               "finger_joint1 (kp 100 N/m, kv 10 N s/m, force ±100 N, ctrl 0-0.04 m; the j1 = j2 equality kept)")
GO2_CHANGE = ("Go2 joint angles are measured from Menagerie's stand: add thigh 0.9, calf -1.8 rad to get Menagerie's "
              "angles. Each thigh and calf body is turned by its stand angle and its joint range, ctrl range and the "
              "'init' keyframe shifted by it, so qpos0 (where alloy.build starts a robot) is the stand, feet on the "
              "floor, held by ctrl 0; Menagerie's calf range [-2.72, -0.84] excludes 0. The robot and its motion are "
              "unchanged (alloy.tasks.emb_robots.zero_at_start)")
GO2_FEET_CHANGE = ("Go2's four foot balls (FL, FR, RL, RR): solimp 0.015 1 0.022 -> MuJoCo's default 0.9 0.95 0.001 "
                   "and solref 0.02 -> 0.005 s. Menagerie's values carry 1.5 % of the contact force at the surface and "
                   "all of it only 22 mm deep, so the robot stood 13.7 mm inside a rigid floor; it now rests 0.17 mm "
                   "deep. Shapes, friction, condim and priority unchanged (alloy.tasks.emb_robots.firm_feet)")
DUO_WHEELS_CHANGE = ("Duo wheels (two casters, two drive wheels): solref 0.02 -> 0.005 s and solimp 0.9 0.95 -> 0.95 "
                     "0.99. The URDF has no contact parameters; at MuJoCo's defaults the 0.2 kg casters rested 10.0 mm "
                     "inside a rigid floor under the 328 kg robot; they now rest 0.43 mm deep. Shapes and friction "
                     "unchanged (alloy.tasks.emb_robots.firm_wheels)")
H1_2_STAND_CHANGE = ("H1-2 leg and torso servos (hips, knees, ankles, torso: 13): kp and kd x 16 (ankle pitch 60 -> "
                     "960 N m/rad, knee 300 -> 4800). The forklift profile's gains hold a seated operator; standing, "
                     "two ankles at kp 60 give 120 N m/rad against the 590 that keep it from tipping, so it lay flat "
                     "by 2 s and landed 20-33 mm deep in the floor. It now stands (deepest contact 0.33 mm). Force "
                     "limits and the arm, hand and head servos are the profile's (alloy.tasks.emb_robots.STAND_GAIN)")
LEAP_CHANGE = (f"LEAP joint and servo dynamics from MuJoCo Playground (dynamics_source; LICENSE.1): every hand joint "
               f"damping 0.03 -> {PLAYGROUND_LEAP['damping']:g} N m s/rad, armature 0 -> "
               f"{PLAYGROUND_LEAP['armature']:g} kg m^2, frictionloss 0.001 -> "
               f"{PLAYGROUND_LEAP['frictionloss']:g} N m, "
               f"every position servo's torque capped at +-{PLAYGROUND_LEAP['torque_nm']:g} N m (none before). As "
               "shipped, random finger targets flung a 7 cm cube in 66 % of worlds (TA-4 F3). Kp 3 and kv 0.01 are "
               "menagerie's (alloy.tasks.emb_robots.playground_dynamics)")
FIRM_CHANGE = (f"robot geoms (every link, the hand or gripper base, fingers and pads): solref -> {list(FIRM_SOLREF)} "
               "(from MuJoCo's default 0.02 s; the 2F-85's pads from 0.004 s; 2 x the 1.25 ms robot step) and solmix "
               f"1 -> {FIRM_SOLMIX:g}, so the robot's solref governs a pair "
               "with the 0.02 s table, floor or an object. At MuJoCo's default 0.02 s the 15 g Franka finger rested "
               "5.5 mm deep at 91 N and an arm pressing on its table went INVALID (TRAIN-FIX 1b; WP3 root cause, "
               "TA-1 F1). "
               "The world's table, floor, box and cylinder: solref 0.02 -> 0.0025 too (solmix 1): with the table at "
               "0.02 s the hand pressed a 4 cm cube 11.8 mm into it. Both, where solimp was MuJoCo's default 0.9 0.95 "
               f"0.001: solimp -> {list(FIRM_SOLIMP)[:3]} (the 2F-85 pads' and the mug's), half the steady press "
               "depth. Friction and condim unchanged (alloy.tasks.emb_firm)")
WORLD_NOTE = {
    "tabletop": "table 1.2 x 1.2 m, top z = 0 (robot base on it), floor z = -0.75; box 50 mm 0.1 kg, cylinder "
                "r 30 x 100 mm 0.15 kg, T14's recorded mug (library/robots/task_t14, convex pieces, 0.404 kg)",
    "legged": "floor z = 0; box obstacle 0.3 x 0.3 x 0.1 m at x = 1 m",
    "mobile": "floor z = 0, the robot on it; table 1.2 x 1.2 m, top 0.75 m, x 0.7-1.9 m in front; box, cylinder "
              "and T14's mug on it as for tabletop",
    "hand": "hand as shipped (palm up, base welded); floor z = 0; a 40 mm cube of 0.05 kg lowered onto the palm "
            "until it just touches",
}


ROUTE_VALID = (f"valid at Alloy's route step ({ROUTE_DT} s) only: an engine at the sources' 2 ms step clamps the "
               "time constant to 4 ms (MuJoCo's 2 dt floor)")


def card_name(e: Embodiment) -> str:
    return f"task_{e.name}"


def scene_path(e: Embodiment) -> Path:
    return ROBOTS / e.name / "scene.xml"


def _source_record(e: Embodiment) -> dict:
    url, commit = e.source_ref()
    root = e.source_root()
    files = {f: sha256_file(root / f) for f in (*e.files, *e.licences)}
    changes = list(CHANGES) + ([HAND_CHANGE] if any("franka" in f for f in e.files) else [])
    changes += [GO2_CHANGE, GO2_FEET_CHANGE] if e.key.startswith("e8_go2") else []
    changes += [DUO_WHEELS_CHANGE] if e.key.startswith("e5_fr3_duo") else []
    changes += [H1_2_STAND_CHANGE] if e.key.startswith("e7_h1_2") else []
    changes += [LEAP_CHANGE] if e.key == "e9_leap" else []
    changes += [FIRM_CHANGE] if e.key in FIRM_CONTACTS else []
    if e.contact and e.base in FIRM_CONTACTS:
        changes.append(f"the verdict scene here is emb_{e.base} without TRAIN-FIX 1b's firm contacts: this variant "
                       "keeps the source's solimp and solmix (alloy.tasks.emb_firm applies to the verdict scene only)")
    if e.contact == "stiff":
        changes.append(f"contact variant of emb_{e.base}: every geom's solref set to {list(STIFF_SOLREF)} (time "
                       "constant, damping ratio); a solimp softer than MuJoCo's default (dmin < 0.9; Go2's feet) set "
                       "to the default; friction, margins, stiffer solimp and everything else as the verdict scene")
    if e.contact == "route":
        changes.append(f"route contact variant of emb_{e.base}: every geom's solref set to {list(ROUTE_SOLREF)} (time "
                       f"constant = 2 x Alloy's route step {ROUTE_DT} s, MuJoCo's floor; damping ratio); a solimp "
                       "softer than MuJoCo's default (dmin < 0.9; Go2's feet) set to the default, as in _stiff; "
                       "friction, margins, stiffer solimp and everything else as the verdict scene")
    rec = {"embodiment": e.key, "title": e.title, "source": {"url": url, "commit": commit, "files": files},
           "licence": e.licence, "world": WORLD_NOTE[e.world], "changes": changes}  # fmt: skip
    if e.contact == "route":
        rec["contact"] = {"source_solref": source_solrefs(e), "variant_solref": list(ROUTE_SOLREF),
                          "route_dt_s": ROUTE_DT, "valid": ROUTE_VALID}  # fmt: skip
    if e.key == "e9_leap":
        rec["dynamics_source"] = _playground_record()
    if e.world == "tabletop":
        rec["mug"] = {"scene": str(emb_build.MUG_SCENE.relative_to(REPO)), "sha256": sha256_file(emb_build.MUG_SCENE)}
    return rec


def _playground_record() -> dict:
    """The LEAP dynamics' source, its pinned file checked against the sha256 the values were read from."""
    rec = dict(PLAYGROUND_LEAP_SOURCE)
    got = sha256_file(playground_root() / rec["file"])
    if got != rec["sha256"]:
        raise ValueError(f"MuJoCo Playground file {rec['file']}: sha256 {got}, expected {rec['sha256']}")
    return rec | {"licence_sha256": sha256_file(playground_root() / "LICENSE")}


def source_solrefs(e: Embodiment) -> list[dict]:
    """A variant's verdict scene, as vendored: each distinct geom solref and how many geoms carry it."""
    m = mujoco.MjModel.from_xml_path(str(scene_path(EMBODIMENTS[e.base])))
    vals, counts = np.unique(np.round(m.geom_solref, 9), axis=0, return_counts=True)
    return [{"solref": [float(x) for x in v], "geoms": int(n)} for v, n in zip(vals, counts)]


def _card_header(e: Embodiment, rec: dict) -> str:
    head = f"P3-2b embodiment {e.key}: {e.title}. Built by alloy.tasks.emb_scene from {rec['source']['url']}@" \
           f"{rec['source']['commit'][:12]} ({e.licence}). Replay tier: the whole MJCF as one robot."
    if e.contact != "route":
        return head
    src = ", ".join(f"{c['solref'][0]:g} s x {c['geoms']} geoms" for c in rec["contact"]["source_solref"])
    return f"{head} Contact: solref {ROUTE_SOLREF[0]:g} s on every geom (source emb_{e.base}: {src}); {ROUTE_VALID}."


def _copy_licences(e: Embodiment, out: Path) -> None:
    for i, f in enumerate(e.licences):
        shutil.copyfile(e.source_root() / f, out / f"LICENSE.{i}")
    if e.key == "e9_leap":
        shutil.copyfile(playground_root() / "LICENSE", out / f"LICENSE.{len(e.licences)}")


def check_init(xml: Path) -> dict:
    """The written scene compiles, has keyframe ``init``, and starts with no contact deeper than 0.5 mm."""
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("init").id)
    mujoco.mj_forward(m, d)
    deepest = min((float(c.dist) for c in d.contact[: d.ncon]), default=0.0)
    return {"nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu), "nbody": int(m.nbody), "ngeom": int(m.ngeom),
            "init_contacts": int(d.ncon), "init_deepest_mm": round(1e3 * deepest, 4),
            "ok": deepest > -5e-4 and bool(np.all(np.isfinite(d.qacc)))}  # fmt: skip


def explicit_multiccd(xml: Path) -> None:
    """Write ``<flag multiccd="disable"/>``: multi-CCD off, as Alloy's route runs every MJCF scene (Newton's importer
    reads an absent flag as off; T12 runs so). MuJoCo C reads an absent flag as on, so the scene says it.
    Multi-CCD on gave a resting cylinder four contacts (0.2 mm deep instead of 1.1 mm on its rim) but reports false
    deep contacts between convex mesh pieces: BUL-0013, −45.9 mm for a true −0.034 mm on 53- and 56-vertex FR3
    pieces (the BUL-0002 / 0006 / 0007 mechanism). WP13 decision 2026-11-03."""
    tree = ET.parse(xml)
    option = tree.getroot().find("option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("multiccd", "disable")
    ET.indent(tree, space=" ")
    tree.write(xml, encoding="unicode", xml_declaration=False)


def _referenced_meshes() -> set[str]:
    """Mesh file names any vendored scene uses."""
    out: set[str] = set()
    for xml in ROBOTS.glob("*/scene.xml"):
        out |= {Path(m.get("file", "")).name for m in ET.parse(xml).getroot().iter("mesh")}
    return out


def _prune_new_meshes(before: set[str]) -> int:
    """Delete mesh-store files this write added that no scene uses: the full hulls the split replaced (rebuilt from
    the pinned source on demand). Returns how many were deleted."""
    store = ROBOTS / "meshes"
    used = _referenced_meshes()
    stale = [f for f in store.iterdir() if f.name not in before and f.name not in used]
    for f in stale:
        f.unlink()
    return len(stale)


def write(e: Embodiment) -> Path:
    """Build, compile and write the scene, its card and source record; returns ``scene.xml``."""
    from alloy.tasks.mjb_convert import write_scene
    from alloy.tasks.scenes import converted_scene_card, write_scene_card

    spec = emb_build.build_spec(e)
    m = spec.compile()
    before = {f.name for f in (ROBOTS / "meshes").iterdir()}
    xml = write_scene(m, ROBOTS, e.name, SPLIT, keyframes=True)
    explicit_multiccd(xml)
    log.info("%s: %d replaced full-hull meshes not kept in the store", e.name, _prune_new_meshes(before))
    out = xml.parent
    _copy_licences(e, out)
    rec = _source_record(e) | {"hull_split": json.loads((out / REPORT_NAME).read_text())["max_bound_m"],
                                "init": check_init(xml)}  # fmt: skip
    (out / SOURCE_JSON).write_text(json.dumps(rec, indent=1) + "\n")
    write_scene_card(converted_scene_card(e.name, ROBOTS), SCENES, card_name(e), _card_header(e, rec))
    write_manifest(ROBOTS)
    if not rec["init"]["ok"]:
        log.warning("%s: start state is not at rest: %s", e.name, rec["init"])
    return xml


def main(argv: list[str] | None = None) -> int:
    """``python -m alloy.tasks.emb_scene <key>...`` (``all`` = every registered embodiment): write the scenes."""
    import argparse

    from alloy.tasks.emb_specs import EMBODIMENTS, embodiment

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("keys", nargs="+")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    keys = list(EMBODIMENTS) if args.keys == ["all"] else args.keys
    for key in keys:
        xml = write(embodiment(key))
        log.info("%s: %s", key, json.loads((xml.parent / SOURCE_JSON).read_text())["init"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
