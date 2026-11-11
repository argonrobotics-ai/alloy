# SPDX-License-Identifier: Apache-2.0
"""The concrete-finishing ports' own lists for ``concrete_verdict`` (audit AB-gym-7, 2026-11-09): which Gym-Anything
criteria each port does not judge, judges in part or judges by another measure, and how its scene differs.

Every line states what the port's code does (the module named in it) against Gym-Anything's code (file:line under
``benchmarks/roboworld/environments/concrete_finishing``, read-only in [internal note]). Common to all: the actor and the
site (their bleed variation applies to the pad's concrete only). Gym-Anything's success for these tasks reads
material fields only, plus the forms, the tools and the arms on the pad tasks (``task.py:393-395``).
"""

from __future__ import annotations

from alloy.tasks.concrete_verdict import Port

ROBOT_ONLY = ("Gym-Anything's task interface is robot-only (env.py:1-8: nothing moves a tool or the paste except "
              "through the robot's contacts); its carriage is an ASSISTED probe fixture, not reachable from the "
              "policy interface (probes/common.py:1-4)")
CARRIAGE = ("actor: a privileged tool carriage (concrete_tools.ToolCarriage) moves the tools on a keyframed program "
            "and there is no robot, so no grip, reach or arm can fail; " + ROBOT_ONLY)
KNIVES = ("actor: two kinematic knives are posed each control period by a script (concrete_drywall._place) and there "
          "is no robot; " + ROBOT_ONLY + "; its hawk and the knives' flex are not modelled")
SITE = ("site: every world starts at the task's one temperature (21.1 C by default); Gym-Anything varies it by +-2 C "
        "for every seed but 0 (variants.py:26-33)")
SITE_PAD = ("site: every world starts at the task's one temperature (21.1 C by default) and the mean bleed; "
            "Gym-Anything varies the temperature by +-2 C and the bleed by +-20 % for every seed but 0 "
            "(variants.py:26-33)")
FORMS = ("not judged: the forms are chart cells, not bodies, so no force on them is measured (the value is a "
         "default 0 N); Gym-Anything judges the peak horizontal contact force on a form or kicker, <= 150 N "
         "(task.py:707-710)")
ARMS = ("not judged: there is no robot (the value is a default 0); Gym-Anything judges that no arm link, hand or "
        "finger is over the pad below 0.40 m (task.py:839-844)")
FORMS_CELLS = "forms: the forms are chart cells, not bodies: nothing can push them"


def _tools(mine: str, within: str, theirs: str, rule: str = "inside the cart's footprint (task.py:712-733)") -> str:
    return (f"judged in part: {mine}, by a point test (the body within {within} of its rest), put there by the "
            f"privileged carriage, not released by a gripper; Gym-Anything judges {theirs} {rule}. A tool that is "
            f"not back fails; a pass is not counted")


PAD_EDGE = Port(
    "pad-edge",
    not_judged={"forms_undisturbed": FORMS, "arms_clear": ARMS},
    partly={"tools_back": _tools("the edger alone", "2 cm (pad_run.ON_CART)",
                                "each of its 3 tools (edger, margin trowel, penetrometer; layout.py:153-154)")},
    scene_differs=(CARRIAGE, FORMS_CELLS,
                   "tools: the scene has the edger alone, not Gym-Anything's cart with three tools", SITE_PAD))

# pad-trowel keeps WP15's ruling (pad_trowel_lifts._proxied): forms by a clearance proxy, arms not applicable, the
# tools a judged pass; the lists here only say so
PAD_TROWEL = Port(
    "pad-trowel",
    not_judged={"forms_undisturbed": FORMS, "arms_clear": ARMS},
    notes={"tools_back": "the port's own tools (the steel trowel, and the margin trowel when it lifts) within 2 cm "
                         "of their rests, put there by the privileged carriage; Gym-Anything judges its 3 tools "
                         "(trowel, edger, penetrometer; layout.py:157-158) inside the cart's footprint "
                         "(task.py:712-733)"},
    scene_differs=(CARRIAGE, FORMS_CELLS, SITE_PAD))

PAD_DARBY = Port(
    "pad-darby",
    not_judged={"forms_undisturbed": FORMS, "arms_clear": ARMS},
    partly={"tools_back": _tools("the darby and the margin trowel", "1 cm",
                                "each of its 3 tools (darby, magnesium float, margin trowel; layout.py:151)")},
    scene_differs=(CARRIAGE, FORMS_CELLS, SITE_PAD))

PAD_FINISH = Port(
    "pad-finish",
    not_judged={"forms_undisturbed": FORMS,
                "robot_off_pad": "not judged: there is no robot (the value is a default 0); Gym-Anything judges the "
                                 "robot's links and fingertips against the pad (task.py:810-838)"},
    partly={"tools_back": _tools("the stages' four tools (darby, margin trowel, edger, steel trowel)",
                                 "1-2 cm", "each of its 5 tools (those and the magnesium float; layout.py:159-161)",
                                 "on a shelf of the cart, let go and resting (task.py:712-733, shelves)")},
    scene_differs=(CARRIAGE, FORMS_CELLS, SITE_PAD))

PATCH_FLUSH = Port(
    "patch-flush",
    notes={"no_feather_edge": "the off-region part is the port's own tally (concrete_patch.off_region): paste let "
                              "go into Alloy's bucket is not counted"},
    scene_differs=(CARRIAGE,
                   "bucket: Alloy adds a bucket beside the slab for the trowel's carry (Gym-Anything's patch-flush "
                   "has no bucket, tub, hawk or rag, layout.py:162-163): paste let go into it is not off-region, "
                   "which eases no_feather_edge", SITE))

TOOL_JOINTS = Port(
    "tool-joints",
    partly={"smear": "judged in part: the face part as Gym-Anything (mortar over 0.2 mm on the block faces, "
                     "< 2 cm2/m); the off-wall part reads the film's books (concrete_joints_books.off_wall: gobs, drips "
                     "and releases of the trowel, overflow and spill past the chart) and fails over 2e-8 m3, but "
                     "its pass is not counted: with the wall laid flat the lifted trowel's stick limit "
                     "(stores_lift.stick_volume at the film's one yield) is over what it can carry, so the gob "
                     "that falls from Gym-Anything's blade on its vertical wall (coupling.py:922-938) cannot fall "
                     "here, and mortar in the drains is not counted"},
    scene_differs=(CARRIAGE,
                   "wall: the infill is laid flat (gravity into its face); Gym-Anything's stands (layout.py:432), "
                   "where a lifted blade keeps half as much",
                   "mortar board: free drains at each end of every bed joint stand for Gym-Anything's mortar "
                   "board (concrete_joints.boards; layout.py:587-596): mortar pushed into them leaves the wall "
                   "without counting as smear or off-wall", SITE))

TOOL_SEALANT = Port(
    "tool-sealant",
    notes={"smear": "the off-region part is the gobs the lifted nose dropped (the film's tool_dropped books); the "
                    "chart's edges are walls, so no sealant leaves over them"},
    scene_differs=(CARRIAGE,
                   "wall: the chart lies flat with gravity turned along the joint (concrete_sealant.GRAVITY), "
                   "standing for Gym-Anything's vertical wall (layout.py:497)",
                   "rag: a wipe is a scripted release of the nose's load on Gym-Anything's wipe law "
                   "(world.py:259-285) beside the joint, booked out of the film; there is no rag body to touch, and "
                   "the script wipes after every stroke of at most max_stroke (30 mm)", SITE))

PATCH_DRYWALL = Port(
    "patch-drywall",
    notes={"no_stray_compound": "the off-board part is the port's own tally from the film's books "
                                "(concrete_drywall.off_board: the knives' gobs, drips and releases)"},
    scene_differs=(KNIVES, SITE))

BOOKS_GATE = ("Alloy's own gate, stricter than Gym-Anything (which has none): the film's and the knives' books "
              "close to 1e-9 of the start, else the world is unverifiable (no success, no score)")
FINISH_GATES = ("Alloy's own gate, stricter than Gym-Anything: the ride rule of the margin and the trowel, the "
                "paste and water books closed, the material clock within 5 h (advisory in Gym-Anything, "
                "task.py:852-859) and every stage handed off at rest")

PORTS = {p.task: p for p in (PAD_EDGE, PAD_TROWEL, PAD_DARBY, PAD_FINISH, PATCH_FLUSH, TOOL_JOINTS, TOOL_SEALANT,
                             PATCH_DRYWALL)}
# a pad task without a list of its own
PAD = Port("pad", not_judged={"forms_undisturbed": FORMS, "arms_clear": ARMS},
           scene_differs=(CARRIAGE, FORMS_CELLS, SITE_PAD))

__all__ = ["BOOKS_GATE", "FINISH_GATES", "PAD", "PAD_DARBY", "PAD_EDGE", "PAD_FINISH", "PAD_TROWEL", "PATCH_DRYWALL",
           "PATCH_FLUSH", "PORTS", "TOOL_JOINTS", "TOOL_SEALANT"]
