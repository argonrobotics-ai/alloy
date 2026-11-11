# SPDX-License-Identifier: Apache-2.0
"""Open a taped supply carton (the reference task set's open-carton; docs/notes/paper-pack.md): peel the seam tape off
the top, then open the four flaps.

- **Carton, contents, hook** as in open-flaps (gift_carton.py, gift_carton_flaps.py). The +y major's crease is set
  12 deg open as there, but the tape holds it shut (``CartonSpec.held_deg``: its elastic hinge starts loaded), so it
  springs up once the tape is off. The Bonds plug-in's bonds act on the film only (their reaction never reaches the
  flap), so the tape's hold is a torque on that hinge equal to its load (``hold_torque``), switched off once the tape
  is peeled; without it the flap sprang up under the bonded tape at the start. Every world starts alike: the
  Bonds plug-in holds the tab from the build, so the hook's tine starts on it, and the gantry starts in one place for
  all worlds.
- **Tape** (library ``gift_carton_tape``): 48 x 370 mm of 3M 371 along the seam between the majors, bonded where it
  lies on them (Bonds pre-bond), with a 10 mm tab past the -x edge. Adhesive card ``3m_371/kraft_liner``
  (alloy.plugins.bonds.adhesives): the liner's internal bond, G = 200 J/m² (ESTIMATE), so a 180 deg peel pulls
  G·b/2 = 4.8 N.
- **Peel.** The hook's tine (52 mm wide, as the tape) starts 0.9 mm over the tab's outer row and holds it (Bonds
  grip; ``tine_gap``: at 0.1 mm the arm, folding back under the tine, was pinched against the board, and 2-9 of 16
  GPU worlds launched the hook the moment it went taut; 0.9 mm: 0 of 16 then, ~1 in 8 later in the peel:
  docs/known_gaps/paper-open-carton-peel.md). It lifts
  it 1 mm and pulls it over the tape along +x (no moment about the majors' hinges, which run along x): the arm folds
  back slack and goes taut a tab's length past the front, at ~174 deg; the grip travels twice the peeled length. Bonds
  reads the peel angle from the middle of the held vertices to the front and puts the pull on the flap there: held
  over 40 mm and lifted, a flat tab read as a ~10 deg peel at Kendall's ~180 N; held over 10 mm and folded over the
  front, the arm went taut near 90 deg and pulled ~5 N up on a ~10 g flap edge (2 of 8 GPU worlds launched).
  Then the hook carries the tape clear of the carton (to the gantry's x end, 0.15 m up) and lets it go there: let go
  at the end of the pull, it hung down the +x side 3 cm out, by the +x minor. Then it opens the flaps as in
  open-flaps.

Checks, per world (the reference's): at most 10 % of the tape still joined; every flap at least 90 deg open through
the 3 s hands-off window; the carton moved at most 30 mm and tilted at most 10 deg; every content item inside and moved
at most 10 mm. That is 8 of the reference's 9 open-carton criteria, and its hands-off window: GA's reaction force on every robot
link over the 3 s is judged by distance (no hook box within 5 mm of the carton, a flap, a content item or the counter at
any control period; gift_hands_off.py; the released tape strip, lying at the gantry's far end, is not a target). That is
9 of 9: ``ga_criteria``, ``ga_judged`` ("9 of 9"), ``ga_notes`` (the differences), no port check.

``python -m alloy.tasks.gift_carton_tape --device cuda:0 --worlds 4`` prints one JSON row.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from alloy.solvers.film.materials import FilmExtras

from .gift_carton import CartonSpec, _hinges, hold_torque
from .gift_carton_flaps import FLAP_BODIES, ITEMS, FlapsSpec, contents, drive, evaluate, ga_report
from .gift_gantry import FINGER_LEN, HOOK_T, HOOK_TINE, gantry_mjcf

TAPE_CARD = "gift_carton_tape"          # 0.37 x 0.048 m (the card's box)
ADHESIVE = "3m_371/kraft_liner"
_PRE = "carton/worldbody/carton/"
SUBSTRATES = tuple(f"{_PRE}{f}_slider/{f}/{f}_board" for f in ("major_p", "major_m"))
GRIP = "gift_gantry/worldbody/carriage_x/carriage_y/palm/wrist/hook_tine"
TAPE_NOTES = ("the released tape strip is not a target of the hands-off test (it lies at the gantry's far end)",)
JOINTS = ("gantry_x", "gantry_y", "gantry_z", "wrist_yaw")
CLEAR = (0.88, 0.15)                    # the grip's x offset (gantry_x range +-0.9 m) and rise when it lets go [m]


def _taped_flaps() -> FlapsSpec:
    return FlapsSpec(carton=CartonSpec(held_deg=12.0), jitter_xy=0.0, jitter_yaw_deg=0.0)


@dataclass(frozen=True)
class TapeSpec:
    length: float = 0.37                # m, the card's
    tab: float = 0.010                  # past the majors' -x edge [m]
    lift: float = 1.0e-4                # start gap under the tape's particles [m]
    tine_gap: float = 9.0e-4            # start gap between the held row's contact skin and the tine [m]: room
                                        # for the folded arm under it (0.1 mm: 2-5 of 16 GPU worlds launched
                                        # the hook as the arm went taut; 0.9 mm: 0 of 16); within grip_reach
    backing_et: float = 2.0e9 * 2.8e-5  # N/m, the peel arm's E·t: BOPP ~2 GPa (ESTIMATE) x 371's 28 um backing
    tine_width: float = 0.052           # m: the tine covers the tape's width (a narrower hold reads as lateral stretch)
    grip_reach: float = 1.5e-3          # m: tab vertices this close to the tine are held
    overlap: float = 0.002              # the tine's tip over the tab's outer end [m]: it holds that row only
    pull_height: float = 0.001          # the held row over the tape while pulling [m]
    wrist_armature: float = 0.05        # kg·m² on the wrist: Bonds' lagged pull acts ~35 mm off its axis, where the
                                        # bare wrist (~3e-4 kg·m²) is a 0.2 kg mass: 8 of 8 GPU worlds launched the hook
    speed: float = 0.2                  # grip speed [m/s] (the peel has no rate dependence)
    clear_s: float = 0.6                # the carry clear of the carton, held [s]
    joined_max: float = 0.10            # pass: at most this share of the tape still bonded


@dataclass(frozen=True)
class OpenSpec:
    flaps: FlapsSpec = field(default_factory=_taped_flaps)
    tape: TapeSpec = field(default_factory=TapeSpec)


def layout(s: OpenSpec) -> dict:
    """The tape's centre and mid-surface height, the majors' x edges, and the hook's palm start: its tine lies on the
    tab's outer row (``overlap``), the shaft and the rest of the tine past the tab's end."""
    c, r = s.flaps.carton, FilmExtras().contact_radius
    top = _hinges(c)["major_p"]["p"][2] + 0.5 * c.caliper           # the closed majors' top face
    z = top + r + s.tape.lift
    x1 = 0.5 * (c.length + c.caliper)
    if abs((2 * x1 + s.tape.tab) - s.tape.length) > 1e-9:
        raise ValueError(f"TapeSpec: the card's {s.tape.length} m strip must cover the majors ({2 * x1:.3f} m) and "
                         f"the {s.tape.tab} m tab")
    tine_bottom = z + r + s.tape.tine_gap                           # the tine lies on the tab
    palm = (-x1 - s.tape.tab + s.tape.overlap - HOOK_TINE, 0.0, tine_bottom + FINGER_LEN)
    return {"top": top, "z": z, "x0": -x1, "x1": x1, "centre": (-0.5 * s.tape.tab, 0.0, z), "palm": palm,
            "tine": (palm[0], palm[0] + HOOK_TINE)}


def bonds_plugin(s: OpenSpec):
    from alloy.plugins.bonds import Bonds  # noqa: PLC0415

    return Bonds(substrates=SUBSTRATES, adhesives=(ADHESIVE,) * len(SUBSTRATES), grip=GRIP,
                 grip_reach=s.tape.grip_reach, prebond=True, prebond_reach=1.0e-3, face_et=s.tape.backing_et,
                 form=False)


def build(s: OpenSpec = OpenSpec(), worlds: int = 1, seed: int = 0, device: str | None = None):
    """(sim, bonds, layout): the taped carton with its contents, the tape bonded, the tab held by the tine."""
    import alloy  # noqa: PLC0415

    from .gift_carton import carton_mjcf  # noqa: PLC0415

    lay = layout(s)
    scene = alloy.Scene(worlds=worlds, dt=s.flaps.dt)
    scene.add_ground()
    scene.add_robot("carton", mjcf=carton_mjcf(s.flaps.carton))
    for name, kind, size, mass, c in contents(s.flaps.carton):
        if kind == "box":
            scene.add_box(name, size=size, pos=tuple(c), mass=mass, friction=0.5)
        else:
            scene.add_cylinder(name, size[0], size[1], pos=tuple(c), mass=mass, friction=0.5)
    scene.add_object("tape", TAPE_CARD, pos=lay["centre"])
    scene.add_robot("hook", mjcf=gantry_mjcf(lay["palm"], tool="hook", tine_width=s.tape.tine_width,
                                             wrist_armature=s.tape.wrist_armature))
    bonds = bonds_plugin(s)
    with exchange_caps(*EXCHANGE_CAPS):
        sim = alloy.build(scene, device=device, seed=seed, plugins=[bonds])
    hold(sim, s, True)
    return sim, bonds, lay


HOLD = "carton/major_p_hold"
EXCHANGE_CAPS = (4.0, 0.5)              # the coupler's (c, c_d) on the bodies coupled to the tape (SCR-SC4-4d)


@contextlib.contextmanager
def exchange_caps(c: float | None, c_d: float | None):
    """Build under the "standard" profile with the coupler's exchange caps set, then put the profile back. The caps
    stop most of the peel's lagged-contact launches (1 in 40 worlds vs 1 in 8), and ``alloy.build`` cannot set them
    yet (docs/known_gaps/paper-open-carton-peel.md; coordinator 2026-11-08 15:16Z, route b)."""
    from alloy.coupling.types import SchedulerPolicy  # noqa: PLC0415
    from alloy.integration import profiles  # noqa: PLC0415

    base = profiles.PROFILES["standard"]
    pol = dataclasses.replace(base.scheduler_policy or SchedulerPolicy(), exchange_stiffness_c=c, exchange_damping_c=c_d)
    profiles.PROFILES["standard"] = dataclasses.replace(base, scheduler_policy=pol)
    try:
        yield
    finally:
        profiles.PROFILES["standard"] = base


def hold(sim, s: OpenSpec, on: bool) -> None:
    """The tape's hold on the sprung major (``hold_torque``): on while the tape is bonded, off once it is released."""
    if s.flaps.carton.held_deg:
        sim.set_controls({HOLD: hold_torque(s.flaps.carton) if on else 0.0})


def peel_plan(s: OpenSpec, lay: dict) -> list[tuple[str, float, dict]]:
    """(phase, seconds, joint targets as palm offsets): lift the held row ``pull_height`` and pull it over the tape
    along +x to past the +x edge (the grip runs twice the peeled length), hold. The arm folds back slack and goes taut
    a tab's length past the front, at ~174 deg."""
    h = s.tape.pull_height
    dx = 2.0 * (lay["x1"] - (lay["x0"] - s.tape.tab)) + 0.03
    zero = {"gantry_y": 0.0, "wrist_yaw": 0.0}
    return [("lift", 0.1, dict(zero, gantry_x=0.0, gantry_z=h)),
            ("pull", dx / s.tape.speed, dict(zero, gantry_x=dx, gantry_z=h)),
            ("hold", 0.2, dict(zero, gantry_x=dx, gantry_z=h))]


def clear_plan(s: OpenSpec, lay: dict) -> list[tuple[str, float, dict]]:
    """(phase, seconds, targets): carry the peeled tape, still held, to the gantry's x end and up, clear of the carton
    (its trailing end ``clear_gap`` past the +x edge), so it falls there when the grip lets go."""
    h = s.tape.pull_height
    return [("clear", s.tape.clear_s, {"gantry_x": CLEAR[0], "gantry_y": 0.0, "gantry_z": h + CLEAR[1],
                                       "wrist_yaw": 0.0})]


def clear_gap(s: OpenSpec, lay: dict) -> float:
    """How far past the majors' +x edge the peeled tape's trailing end lies when it is carried clear, flat [m]."""
    held = lay["x0"] - s.tape.tab + CLEAR[0]                         # the held row
    return held - s.tape.length - lay["x1"]


def _run_plan(sim, plan, cur: dict, on_phase=None) -> dict:
    for phase, sec, tgt in plan:
        new = {k: np.full(sim.worlds, float(v)) for k, v in tgt.items()}
        sim.run(sec, targets={k: (cur[k], new[k]) for k in JOINTS})
        cur = new
        if on_phase is not None:
            on_phase(phase)
    return cur


def joined(bonds, start: np.ndarray) -> np.ndarray:
    """Per world: the share of the start's bonded area still bonded."""
    return np.asarray(bonds.read()["bonded"], float) / np.maximum(start, 1e-12)


def run(s: OpenSpec = OpenSpec(), device: str | None = None, worlds: int = 1, seed: int = 0, on_phase=None) -> dict:
    """The task, one JSON-able row; ``on_phase(name)`` is called after each phase (lift, pull, hold, clear, release,
    each flap)."""
    t0 = time.perf_counter()
    sim, bonds, lay = build(s, worlds, seed, device)
    build_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    sim.run(0.1)                                                    # the readings start with the first step
    a0 = np.asarray(bonds.read()["bonded"], float).copy()
    names = FLAP_BODIES + ITEMS
    start = sim.state(names)
    zero = {k: np.zeros(sim.worlds) for k in JOINTS}
    cur = _run_plan(sim, peel_plan(s, lay), zero, on_phase)
    left = joined(bonds, a0)
    peel_s = round(float(sim.t), 2)
    hold(sim, s, False)                                             # peeled: the sprung major is free
    cur = _run_plan(sim, clear_plan(s, lay), cur, on_phase)
    bonds.set_input("release_grip", np.ones(sim.worlds, np.int32))
    sim.run(0.2)                                                    # the tape falls off the tine
    if on_phase is not None:
        on_phase("release")
    log = drive(sim, s.flaps, on_phase=on_phase, p0=lay["palm"], cur=cur)
    tr = sim.run(s.flaps.window, record=names + ["hook/wrist"], every=1)
    per = []
    for w, v in enumerate(evaluate(s.flaps, start, tr, s.tape.tine_width)):
        v["tape_joined"] = round(float(left[w]), 4)
        v["tape_off"] = bool(left[w] <= s.tape.joined_max)
        per.append(ga_report(v, {"seam_open": v["tape_off"]}, TAPE_NOTES))
    check = sim.check()
    return {"spec": asdict(s), "worlds": per, "ga_judged": per[0]["ga_judged"],
            "success_rate": float(np.mean([v["success"] for v in per])),
            "success": all(v["success"] for v in per), "check_ok": bool(check["ok"]),
            "problems": [p["what"] for p in check["problems"][:5]], "peel_end_s": peel_s,
            "build_s": round(build_s, 1), "wall_s": round(time.perf_counter() - t1, 1), "sim_s": round(float(sim.t), 2),
            "flaps": log}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = run(OpenSpec(), device=a.device, worlds=a.worlds)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("success", "success_rate", "ga_judged", "check_ok", "peel_end_s", "wall_s", "sim_s")}))
    print(json.dumps(out["worlds"][0]))


if __name__ == "__main__":
    main()
