# SPDX-License-Identifier: Apache-2.0
"""Test scenes and laws of the drilling plug-in (plan law R-5, notes/wp13-engine10-plans.md section 2).

Every scene puts a static aluminium block (the workpiece) under a 3 kg spindle body whose drill tip sits 40 mm below
its housing, so the solver never touches the block: only the drill does. A ``FeedHead`` companion feeds the spindle
down at a per-world rate and spins it at a per-world rate (3000 rpm by default); each world starts spinning.

- ``shaw_feed`` (R-5): worlds at feeds of 0.05, 0.1 and 0.2 mm/rev, plus 0.1 mm/rev at half the spin. At steady state
  the lip thrust (thrust minus the chisel term) scales as f^0.8 (log-log slope within 0.05), thrust and torque at a
  given feed do not depend on the spin (2 %), and the hole bottom advances at the commanded feed rate (1 %).
- ``shaw_diameter`` (R-5): 4, 6 and 8 mm drills at 0.1 mm/rev; torque scales as d^1.8 and lip thrust as d^0.8
  (slopes within 0.05), each normalized to the same realized feed.
- ``through_hole``: a 6 mm drill through a 3 mm plate removes exactly density·π·r²·t (1e-9), every gram goes to the
  chips sink or the part, and thrust and torque are zero after breakthrough.
- ``no_cut_when_lifted``: a spinning drill held 5 mm above the face cuts nothing and pushes nothing.

Holes as geometry (drilling v2, laws D-1 to D-4 and D-8):

- ``hole_axis`` (D-1): drills held at 0, 0.5, 2, 5 and 10 deg from the face normal cut holes along their own axis
  (bound 1 deg from the bit's mean axis over the cut). A square-to-the-face check at 1 deg (Gym-Anything's) passes
  at 0 and 0.5 deg and fails at 2, 5 and 10.
- ``blind_depth`` (D-2): feeds stopped at 3, 6 and 9 mm. Each hole stays blind; its full-diameter depth is the
  bottom minus the point height (0.30 d at 118 deg). The bottom follows the tip less the front's compression (2 µm),
  never passes where the drill stopped less the chisel edge's hold T_c (1/k + 1/k_head) (2 µm), and creeps toward it
  (within 250 µm after the dwell).
- ``stack`` (D-3, D-8): a 3 mm aluminium plate on a steel block. The thrust follows aluminium's law while the point
  is in the plate and steel's once it has left it (1 %), and rises without a dip in between (0.5 %). Both layers cut
  at one front: while the point spans the interface, the block's bottom is the plate's less the plate's span (1 µm).
  The two holes share one axis and one entry line (1 µm), and each layer books its own density times its hole's
  volume (1e-9).
- ``pilot`` (D-4, D-8): a 6 mm drill enlarging a 3 mm through pilot in a 3 mm plate. The chisel edge cuts nothing,
  the lips cut the annulus only (thrust and torque on the annulus fractions, 1 %), nothing pushes while the tip is
  less than r_p / tan(kappa) into the pilot (the lips have not reached the face), and the plate loses exactly
  density·pi·(R² - r_p²)·t (1e-9).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.holes.geometry import union_volume_np
from alloy.plugins.npgeom import quat_rotate

HEAD_HALF = 0.03                     # m: cubic housing half extent
HEAD_MASS = 3.0                      # kg
BIT_LEN = 0.04                       # m: drill tip below the housing
BLOCK_HALF = (0.03, 0.03, 0.015)     # m: a 30 mm block (blind holes)
PLATE_HALF = (0.03, 0.03, 0.0015)    # m: a 3 mm plate (through holes)
SPIN = 100.0 * math.pi               # rad/s (3000 rpm)
GAP = 2.0e-4                         # m: tip above the face at the start
STEP = 6.25e-4                       # s: exchange step of every scene (control_dt / 32)
FEEDS = (0.05e-3, 0.1e-3, 0.2e-3)    # m/rev
DIAMETERS = (4.0e-3, 6.0e-3, 8.0e-3)
F_REF = 0.1e-3                       # m/rev
PERIODS, WINDOW = 125, 20            # 2.5 s runs, the last 0.4 s is the steady window
TIP = (0.0, 0.0, -(HEAD_HALF + BIT_LEN))


def rate_of(feed: float, spin: float = SPIN) -> float:
    """Feed rate [m/s] for a feed per revolution [m] at ``spin`` [rad/s]."""
    return feed * spin / (2.0 * math.pi)


def spindle_and_block(c: Any, half: tuple[float, float, float] = BLOCK_HALF, gap: float = GAP):
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, half[2]), hx=half[0], hy=half[1], hz=half[2],
                        label=c.workpiece)
        z = 2.0 * half[2] + gap - float(c.tip[2])
        head = b.add_body(xform=body_xform(0.0, 0.0, z), label=c.tool)
        density = HEAD_MASS / (2.0 * HEAD_HALF) ** 3
        b.add_shape_box(head, hx=HEAD_HALF, hy=HEAD_HALF, hz=HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=density))
    return build


def feeding(rates: tuple[float, ...], spins: tuple[float, ...], diameters: tuple[float, ...] | None = None):
    """init: per-world feed rates and spins on the head (worlds beyond the tuples repeat the last), initial spin."""
    def init(sim: Any) -> None:
        W = sim.model.world_count
        pick = lambda xs: np.asarray([xs[min(w, len(xs) - 1)] for w in range(W)], np.float64)  # noqa: E731
        head = sim.plugins["feed_head"]
        head.set_input("feed_rate", pick(rates))
        head.set_input("spin", pick(spins))
        if diameters is not None:
            sim.plugins["drill"].set_diameter(pick(diameters))
        qd = sim.state.joint_qd.numpy().reshape(-1, 6)       # one free joint per world: (vx, vy, vz, wx, wy, wz)
        qd[:, :5] = 0.0
        qd[:, 5] = pick(spins)
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    return init


def _head(c: Any):
    def companions():
        from alloy.plugins.drilling.head import FeedHead   # noqa: PLC0415 (the head imports nothing from here)

        return [FeedHead(head=c.tool)]
    return companions


def scene(c: Any, init, worlds: int, periods: int = PERIODS, half=BLOCK_HALF, gap: float = GAP,
          gravity=(0.0, 0.0, -9.81)) -> TestScene:
    return TestScene(build=spindle_and_block(c, half, gap), init=init, worlds=worlds, periods=periods,
                     dt=STEP, ground=False, gravity=gravity, companions=_head(c))


def drilling_scene(c: Any) -> TestScene:
    """Three identical worlds drill 0.8 s into the block at 0.1 mm/rev (entry, then a steady cut)."""
    return scene(c, feeding((rate_of(F_REF),), (SPIN,)), worlds=3, periods=40)


def drill_laws(c: Any) -> list[Law]:
    spins = (SPIN, SPIN, SPIN, 0.5 * SPIN)
    rates = tuple(rate_of(f) for f in FEEDS) + (rate_of(F_REF, 0.5 * SPIN),)
    through = dataclasses.replace(c, thickness=None)
    return [
        Law("shaw_feed", check_feed, scene(c, feeding(rates, spins), worlds=4),
            "R-5: lip thrust ~ f^0.8, no spin dependence, hole advances at the feed rate"),
        Law("shaw_diameter", check_diameter,
            scene(c, feeding((rate_of(F_REF),), (SPIN,), DIAMETERS), worlds=3),
            "R-5: torque ~ d^1.8 and lip thrust ~ d^0.8 at a fixed feed"),
        Law("through_hole", check_through,
            scene(through, feeding((rate_of(F_REF),), (SPIN,)), worlds=2, periods=80, half=PLATE_HALF),
            "a through hole removes density pi r^2 t exactly; no thrust after breakthrough"),
        Law("no_cut_when_lifted", check_lifted,
            scene(c, feeding((0.0,), (SPIN,)), worlds=2, periods=10, gap=0.005, gravity=(0.0, 0.0, 0.0)),
            "a spinning drill 5 mm above the face cuts and pushes nothing"),
        Law("hole_axis", check_axis, scene(c, tilted(TILTS), worlds=len(TILTS), periods=40),
            "D-1: a hole runs along the drill's real axis; a square check fails a tilted one"),
        Law("blind_depth", check_blind, scene(c, stopping(STOPS), worlds=len(STOPS), periods=110),
            "D-2: a stopped feed leaves a blind hole whose depth is where the drill stopped"),
        Law("stack", check_stack, stack_scene(c), "D-3: each layer of a stack yields at its own Shaw law",
            config={"layers": STACK, "stiffness": STACK_STIFFNESS}),
        Law("pilot", check_pilot, scene(through, pilot_feeding(), worlds=2, periods=70, half=PLATE_HALF),
            "D-4: enlarging a through pilot cuts the annulus only, with no chisel thrust"),
        *_pin_laws(c),
        *_concrete_laws(c),
    ]


def _pin_laws(c: Any) -> list[Law]:
    from alloy.plugins.holes.pin_laws import pin_laws  # noqa: PLC0415 (the plug-in imports this module)

    return pin_laws(c)


def _concrete_laws(c: Any) -> list[Law]:
    from alloy.plugins.holes.concrete_laws import concrete_laws  # noqa: PLC0415

    return concrete_laws(c)


# --- checks ------------------------------------------------------------------------------------------------------

def _series(run: Any, name: str) -> np.ndarray:
    return np.asarray(run.trace.plugin(run.plugin.label, name), np.float64)


def _steady(run: Any, name: str) -> np.ndarray:
    return _series(run, name)[-WINDOW:].mean(axis=0)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def check_feed(run: Any) -> LawResult:
    p = run.plugin
    d = p.config.diameter
    thrust, torque, feed = (_steady(run, k) for k in ("thrust", "torque", "feed"))
    if (feed <= 0.0).any() or (thrust <= 0.0).any():
        return LawResult(False, math.nan, 1.0, f"no steady cut: feed {feed}, thrust {thrust}",
                         "the drill must reach the block and cut: check the tip, the gap and the feed head")
    lips = thrust[:3] - p.chisel_thrust(d)
    slope = _slope(feed[:3], lips)
    ratio = lambda w: (thrust[w] / p.thrust_law(feed[w]), torque[w] / p.torque_law(feed[w]))   # noqa: E731
    speed = max(abs(a / b - 1.0) for a, b in zip(ratio(3), ratio(1)))
    depth = _series(run, "depth")
    elapsed = WINDOW * float(run.trace.t[-1] - run.trace.t[-2])
    advance = (depth[-1] - depth[-WINDOW - 1]) / elapsed
    want = np.asarray([rate_of(f) for f in FEEDS] + [rate_of(F_REF, 0.5 * SPIN)])
    z = run.trace.body_q[:, run.bodies(p.config.tool), 2].astype(np.float64)       # [K, W]: tip moves with it
    tip = (z[-WINDOW - 1] - z[-1]) / elapsed
    follow = float(max(np.abs(advance / want - 1.0).max(), np.abs(advance / tip - 1.0).max()))
    worst = max(abs(slope - 0.8) / 0.05, speed / 0.02, follow / 0.01)
    return LawResult.within(worst, 1.0,
                            f"lip thrust slope {slope:.4f} (0.8 +- 0.05); spin change {speed:.2%} (2 %); hole "
                            f"advance vs feed rate and tip {follow:.3%} (1 %); thrust {np.round(thrust, 1)} N at feeds "
                            f"{np.round(feed * 1e3, 4)} mm/rev, torque {np.round(torque, 3)} N m",
                            "cut at the feed where the engaged edges' Shaw-Oxford thrust equals the front force")


def check_diameter(run: Any) -> LawResult:
    p = run.plugin
    d = np.asarray(DIAMETERS)
    thrust, torque, feed = (_steady(run, k) for k in ("thrust", "torque", "feed"))
    if (feed <= 0.0).any():
        return LawResult(False, math.nan, 1.0, f"no steady cut: feed {feed}", "the drills must cut")
    norm = (F_REF / feed) ** 0.8
    m_slope = _slope(d, torque * norm)
    t_slope = _slope(d, (thrust - p.chisel_thrust(d)) * norm)
    worst = max(abs(m_slope - 1.8), abs(t_slope - 0.8)) / 0.05
    return LawResult.within(worst, 1.0, f"torque slope {m_slope:.4f} (1.8 +- 0.05), lip thrust slope {t_slope:.4f} "
                            f"(0.8 +- 0.05); torque {np.round(torque, 3)} N m for d {d * 1e3} mm",
                            "torque = phi_m C_M f^0.8 d^1.8 and lip thrust = phi_l C_T f^0.8 d^0.8, per world d")


def check_through(run: Any) -> LawResult:
    p = run.plugin
    radius = 0.5 * float(np.float32(p.config.diameter))
    want = p.config.density * math.pi * radius**2 * float(p.work.thickness)
    removed, sink, loose = (_series(run, k)[-1] for k in ("removed", "sink", "loose"))
    after = float(np.abs(_series(run, "thrust")[-10:]).max() + np.abs(_series(run, "torque")[-10:]).max())
    volume = float(np.abs(removed / want - 1.0).max())
    ledger = float(np.abs(removed - sink - loose).max() / want)
    if float(_series(run, "thrust").max()) <= 0.0:
        return LawResult(False, math.nan, 1e-9, "the drill never cut", "the drill must reach the plate")
    worst = max(volume, ledger) if after == 0.0 else math.inf
    return LawResult.within(worst, 1.0e-9, f"removed {removed[0] * 1e6:.6g} mg vs density pi r^2 t "
                            f"{want * 1e6:.6g} mg ({volume:.2e}), ledger {ledger:.2e}, thrust + torque after "
                            f"breakthrough {after:.3g}",
                            "removed volume V(depth) - V(depth - t) of the drill-point solid, in float64")


def check_lifted(run: Any) -> LawResult:
    moved = sum(float(np.abs(_series(run, k)).max()) for k in ("removed", "thrust", "torque"))
    return LawResult.within(moved, 0.0, "removed mass + thrust + torque of a drill 5 mm above the face",
                            "cut and push only when the tip is past the hole bottom inside the plate")


# --- holes as geometry (drilling v2) --------------------------------------------------------------------------------

TILTS = (0.0, 0.5, 2.0, 5.0, 10.0)        # deg from the face normal (D-1)
SQUARE_DEG = 1.0                          # Gym-Anything's "axis within 1 deg of the face normal"
STOPS = (3.0e-3, 6.0e-3, 9.0e-3)          # m: tip depth below the face where the feed stops (D-2)
STACK_PLATE, STACK_BLOCK = 3.0e-3, 0.01   # m: aluminium plate on a steel block (D-3)
STACK = ({"workpiece": "plate", "work": "aluminium", "density": 2700.0},
         {"workpiece": "block", "work": "steel_200bhn", "density": 7850.0})
STACK_STIFFNESS = 4.0e6                   # N/m: the front compresses 0.5 mm, not 2 mm, at steel's 1.9 kN
PILOT = 3.0e-3                            # m: the through pilot the 6 mm drill enlarges (D-4)


def _pose_heads(sim: Any, tilts: tuple[float, ...], face: float, gap: float) -> None:
    """Tilt each world's head about x by its angle, its tip ``gap`` above the face mark, spinning; aim its feed."""
    W = sim.model.world_count
    head, drill = sim.plugins["feed_head"], sim.plugins["drill"]
    reach = -float(drill.config.tip[2])
    q = sim.state.joint_q.numpy().reshape(W, 7)                # one free joint per world: the head
    qd = sim.state.joint_qd.numpy().reshape(W, 6)
    dirs = np.zeros((W, 3))
    for w in range(W):
        th = math.radians(tilts[min(w, len(tilts) - 1)])
        dirs[w] = (0.0, math.sin(th), -math.cos(th))
        com = np.array([0.0, 0.0, face]) - dirs[w] * (gap / math.cos(th) + reach)
        q[w] = (*com, math.sin(0.5 * th), 0.0, 0.0, math.cos(0.5 * th))
        qd[w] = (0.0, 0.0, 0.0, *(-SPIN * dirs[w]))
    sim.state.joint_q.assign(q.reshape(-1))
    sim.state.joint_qd.assign(qd.reshape(-1))
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    head.hold.assign(q[:, :3].astype(np.float32))
    head.set_direction(dirs)
    head.set_input("feed_rate", rate_of(F_REF))
    head.set_input("spin", SPIN)


def tilted(tilts: tuple[float, ...], gap: float = GAP):
    """init: each world's head tilted by its angle [deg] about x, feeding along its own axis at 0.1 mm/rev."""
    return lambda sim: _pose_heads(sim, tilts, 2.0 * BLOCK_HALF[2], gap)


def stopping(stops: tuple[float, ...]):
    """init: feed at 0.1 mm/rev and stop each world's feed with its tip ``stops[w]`` below the face."""
    def init(sim: Any) -> None:
        feeding((rate_of(F_REF),), (SPIN,))(sim)
        W = sim.model.world_count
        sim.plugins["feed_head"].set_input("travel", [GAP + stops[min(w, len(stops) - 1)] for w in range(W)])
    return init


def stack_scene(c: Any) -> TestScene:
    """A 3 mm aluminium plate (label ``plate``) resting on a 10 mm steel block (``block``) under the spindle."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.5 * STACK_BLOCK), hx=0.03, hy=0.03, hz=0.5 * STACK_BLOCK,
                        label="block")
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, STACK_BLOCK + 0.5 * STACK_PLATE), hx=0.03, hy=0.03,
                        hz=0.5 * STACK_PLATE, label="plate")
        z = STACK_BLOCK + STACK_PLATE + GAP - float(c.tip[2])
        head = b.add_body(xform=body_xform(0.0, 0.0, z), label=c.tool)
        b.add_shape_box(head, hx=HEAD_HALF, hy=HEAD_HALF, hz=HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=HEAD_MASS / (2.0 * HEAD_HALF) ** 3))
    return TestScene(build=build, init=feeding((rate_of(F_REF),), (SPIN,)), worlds=2, periods=80, dt=STEP,
                     ground=False, companions=_head(c))


def pilot_feeding():
    """init: a 3 mm through pilot already in the plate at the mark; a 6 mm drill feeds at 0.1 mm/rev."""
    def init(sim: Any) -> None:
        feeding((rate_of(F_REF),), (SPIN,))(sim)
        drill = sim.plugins["drill"]
        S = int(drill.config.max_stages)
        drill.hole_n.assign(np.ones(drill.hole_n.shape, np.int32))
        axis = np.zeros(drill.hole_axis.shape, np.float32)
        axis[:, 0] = (0.0, 0.0, -1.0)
        drill.hole_axis.assign(axis)
        n = np.zeros(drill.stage_n.shape, np.int32)
        n[:, 0] = 1
        drill.stage_n.assign(n)
        sd, sb = np.zeros(drill.stage_d.shape, np.float32), np.zeros(drill.stage_b.shape, np.float64)
        sd[:, 0], sb[:, 0] = PILOT, 2.0 * PLATE_HALF[2] + 0.01                      # through, with its point out
        drill.stage_d.assign(sd)
        drill.stage_b.assign(sb)
        assert S >= 2
    return init


def _deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(min(max(c, -1.0), 1.0)))


def check_axis(run: Any) -> LawResult:
    p = run.plugin
    idx = run.bodies(p.config.tool)
    q = run.trace.body_q[len(run.trace.body_q) // 2:, idx].astype(np.float64)        # [K/2, W, 7] while cutting
    errs, tilts = [], []
    for w in range(len(idx)):
        bit = np.mean([quat_rotate(qk[3:], np.asarray(p.config.axis, np.float64)) for qk in q[:, w]], axis=0)
        holes = p.holes(world=w)
        if len(holes) != 1:
            return LawResult(False, math.inf, 1.0, f"world {w}: {len(holes)} holes, want 1", "the drill must cut one")
        errs.append(_deg(holes[0].in_world(p.face(w, 0, run.trace.body_q[-1]))[1], bit))
        tilts.append(holes[0].tilt_deg)
    want = np.asarray([TILTS[min(w, len(TILTS) - 1)] for w in range(len(idx))])
    square = np.asarray(tilts) <= SQUARE_DEG
    ok = bool((square == (want <= SQUARE_DEG)).all()) and float(np.abs(np.asarray(tilts) - want).max()) <= 0.1
    worst = max(errs) if ok else math.inf
    return LawResult.within(worst, 1.0, f"hole axis vs the bit's axis {np.round(errs, 4)} deg (1 deg); hole tilt "
                            f"{np.round(tilts, 3)} deg for drills at {want} deg; square to 1 deg: {square.tolist()}",
                            "a hole starts along the bit's axis and follows it until the full diameter is in")


def check_blind(run: Any) -> LawResult:
    p = run.plugin
    head = run.sim.plugins["feed_head"]
    hold = float(p.chisel_thrust(float(np.float32(p.config.diameter)))) * (1.0 / p.config.stiffness +
                                                                            1.0 / head.config.kp)
    stage = _series(run, "stage_b")[:, :, 0]                                     # [K, W]: layer 0, hole 0, stage 0
    stops = np.asarray([STOPS[min(w, len(STOPS) - 1)] for w in range(stage.shape[1])]) - hold
    over = float((stage.max(axis=0) - stops).max())
    short = float((stops - stage[-1]).max())
    body_q = run.trace.body_q[-1]
    idx = run.bodies(p.config.tool)
    tip_off = np.asarray(p.config.tip, np.float64)
    cons, through = [], []
    for w in range(stage.shape[1]):
        h = p.holes(world=w)[0]
        entry, axis = h.in_world(p.face(w, 0, body_q))
        tip = body_q[idx[w], :3] + quat_rotate(body_q[idx[w], 3:], tip_off)
        s = float(np.dot(tip - entry, axis))
        cons.append(abs(h.stages[0].depth - (s - float(_series(run, "thrust")[-1, w]) / p.config.stiffness)))
        through.append(h.through)
    worst = max(over / 2.0e-6, max(cons) / 2.0e-6, short / 2.5e-4) if not any(through) else math.inf
    return LawResult.within(worst, 1.0, f"bottom past the stop by {over * 1e6:.2f} um (2 um), short of it by "
                            f"{short * 1e6:.1f} um after the dwell (250 um); record vs tip {max(cons) * 1e6:.2f} um "
                            f"(2 um); through {through}; full depths "
                            f"{[round(p.holes(world=w)[0].full_depth() * 1e3, 4) for w in range(stage.shape[1])]} mm",
                            "the bottom follows the tip less the front's compression and never passes the stop")


def _law_at(p: Any, layer: int, feed: np.ndarray) -> np.ndarray:
    return p.thrust_law(feed, layer=layer)


def check_stack(run: Any) -> LawResult:
    p = run.plugin
    d = float(np.float32(p.config.diameter))
    R, T = 0.5 * d, float(np.float32(p.bit.tan_half))
    H, S = int(p.config.max_holes), int(p.config.max_stages)
    b = _series(run, "stage_b")
    top, low = b[:, :, 0], b[:, :, H * S]                                       # layer 0 and layer 1, hole 0, stage 0
    thrust, feed = _series(run, "thrust"), _series(run, "feed")
    in_top = (top > R / T + 2.0e-4) & (top < STACK_PLATE - 3.0e-4)
    in_low = (low > R / T + 2.0e-4) & (top > STACK_PLATE + R / T)
    if not (in_top.any() and in_low.any()):
        return LawResult(False, math.nan, 1.0, "no steady window in the plate or in the block", "feed further")
    e_top = float(np.abs(thrust[in_top] / _law_at(p, 0, feed[in_top]) - 1.0).max())
    e_low = float(np.abs(thrust[in_low] / _law_at(p, 1, feed[in_low]) - 1.0).max())
    span = (low != 0.0) & (top < STACK_PLATE + R / T)                        # the point spans the interface
    front = float(np.abs(low[span] - (top[span] - STACK_PLATE)).max()) if span.any() else math.inf
    dip = 0.0
    for w in range(top.shape[1]):                                            # plate window to block window
        k0, k1 = np.flatnonzero(in_top[:, w])[-1], np.flatnonzero(in_low[:, w])[0]
        run_t = thrust[k0:k1 + 1, w]
        dip = max(dip, float((np.maximum.accumulate(run_t) - run_t).max() / run_t.max()))
    align, ledger = [], []
    body_q = run.trace.body_q[-1]
    for w in range(top.shape[1]):
        h0, h1 = p.holes(world=w, layer=0)[0], p.holes(world=w, layer=1)[0]
        e0, a0 = h0.in_world(p.face(w, 0, body_q))
        e1, a1 = h1.in_world(p.face(w, 1, body_q))
        off = (e1 - e0) - float(np.dot(e1 - e0, a0)) * a0
        align.append(max(float(np.linalg.norm(off)), float(np.linalg.norm(a1 - a0)) * 0.01))
        lr = _series(run, "layer_removed")[-1, w]
        want = (2700.0 * union_volume_np([R], [h0.stages[0].depth], h0.span, T),
                7850.0 * union_volume_np([R], [h1.stages[0].depth], h1.span, T))
        ledger.append(max(abs(lr[i] / want[i] - 1.0) for i in range(2)))
    worst = max(e_top / 0.01, e_low / 0.01, dip / 0.005, front / 1.0e-6, max(align) / 1.0e-6,
                max(ledger) / 1.0e-9)
    return LawResult.within(worst, 1.0, f"thrust vs aluminium's law in the plate {e_top:.3%}, vs steel's in the block "
                            f"{e_low:.3%} (1 %); dip across the interface {dip:.3%} (0.5 %); one front: block bottom "
                            f"vs plate bottom - span {front * 1e6:.3f} um (1 um); holes off one line "
                            f"{max(align) * 1e6:.3f} um (1 um); layer ledgers {max(ledger):.1e} (1e-9)",
                            "each layer yields at its own Shaw-Oxford law at one feed, at one cut front")


def check_pilot(run: Any) -> LawResult:
    p = run.plugin
    d = float(np.float32(p.config.diameter))
    R, T, rp = 0.5 * d, float(p.bit.tan_half), 0.5 * float(np.float32(PILOT))
    rc = float(np.float32(R * p.config.chisel_ratio))
    sh = p.coefficients
    b = _series(run, "stage_b")[:, :, 1]                                        # the enlarging stage
    thrust, torque, feed = _series(run, "thrust"), _series(run, "torque"), _series(run, "feed")
    steady = (b > R / T + 2.0e-4) & (b < 2.0 * PLATE_HALF[2] + rp / T - 3.0e-4)
    idx = run.bodies(p.config.tool)
    q = run.trace.body_q[:, idx].astype(np.float64)                             # [K, W, 7]
    tip_z = np.stack([[(qw[:3] + quat_rotate(qw[3:], np.asarray(p.config.tip, np.float64)))[2] for qw in qk]
                      for qk in q])
    s = 2.0 * PLATE_HALF[2] - tip_z                                             # the tip's depth below the face
    early = (s > 1.0e-4) & (s < rp / T - 1.0e-4)
    if not steady.any():
        return LawResult(False, math.nan, 1.0, "no steady window", "feed further")
    f = feed[steady]
    lips = (R - rp) / (R - rc) * float(sh.lips) * f**0.8 * d**0.8                # no chisel term
    moment = (R * R - rp * rp) / (R * R - rc * rc) * float(sh.torque) * f**0.8 * d**1.8
    e_t = float(np.abs(thrust[steady] / lips - 1.0).max())
    e_m = float(np.abs(torque[steady] / moment - 1.0).max())
    pushed = float(np.abs(thrust[early]).max()) if early.any() else 0.0
    want = 2700.0 * math.pi * (R * R - rp * rp) * float(np.float32(p.work.thickness))
    removed = float(np.abs(_series(run, "removed")[-1] / want - 1.0).max())
    worst = max(e_t / 0.01, e_m / 0.01, removed / 1.0e-9) if pushed == 0.0 and early.any() else math.inf
    return LawResult.within(worst, 1.0, f"thrust vs the annulus lips' law {e_t:.3%}, torque {e_m:.3%} (1 %); push over "
                            f"{int(early.sum())} periods with the tip short of the lips' reach {pushed:.3g} N; "
                            f"removed vs density pi (R^2 - r^2) t {removed:.1e} (1e-9)",
                            "the edges inside a pilot's radius cut nothing")


__all__ = ["drill_laws", "drilling_scene", "rate_of", "spindle_and_block", "tilted"]
