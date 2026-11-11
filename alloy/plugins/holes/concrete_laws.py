# SPDX-License-Identifier: Apache-2.0
"""The rotary-hammer and rebar laws of the drill plug-in (drilling v2, phase C).

Both drill a static 60 mm concrete block (``concrete_medium_hard``, 2400 kg/m^3) with the v1 spindle (3 kg housing,
tip 40 mm below it) on a ``FeedHead`` turning it at 1160 rpm (the Hilti TE 6-S's no-load speed) and fed 10 % faster
than the hammer can cut, so the front stays pressed.

``hammer_rate`` (D-6): a hole deepens at P / (e_v A), P = blow energy x blow rate, A the section the point cuts (pi
d^2 / 4, less a pilot's), at the drill's default 1 mm/rev flute limit (which blows alone do not obey).

| world | bit | blow | predicted |
|---|---|---|---|
| 0-3 | 8, 10, 12, 16 mm | 1.8 J at 85 /s (153 W) | 9.28, 7.32, 6.63, 4.06 mm/s (e_v 0.328, 0.266, 0.204, 0.187) |
| 4 | 12 mm | 3.6 J at 85 /s | twice world 2 |
| 5 | 12 mm | 1.8 J at 170 /s | twice world 2 |
| 6 | 12 mm | hammer off, fed 1 mm onto the face | not cut: held, pressed |
| 7 | 12 mm over a 6 mm through pilot | 1.8 J at 85 /s | P / (e_v pi (d^2 - d_p^2) / 4): 8.84 mm/s |
| 8 | 12 mm at 280 rpm (1.42 mm/rev, past the flutes' 1 mm/rev) | 1.8 J at 85 /s | as world 2: 6.63 mm/s |

The check: the bottom's advance over the last 0.4 s against the prediction (2 %) in every world but 6, worlds 4 and
5 against twice world
2 (2 %), world 6 uncut and holding the tool: the head short of its target by thrust / kp and the bit pressed
thrust / k into the work (2 %); the layer's ledger against density x the union volume (1e-9).

``rebar_stop`` (D-7): a #4 bar (12.7 mm, along y) under 25 mm of cover; a 1/2 in (12.7 mm) bit at 7.2 J, 85 /s (a
heavier hammer, to reach the bar sooner), 3 s, fed 1.5 mm past where it first touches the bar (about 0.75 kN).
World 0 is over the bar's axis, world 1 8 mm off it (the point's cone meets the bar's side), world 2
16 mm off (misses: beyond r + r_bar = 12.7 mm); world 3 is over the axis fed 4 mm past the first touch (about 1.6 kN,
past the 1.04 kN the chisel edge needs in steel_200bhn). The check: worlds 0 and 1 stop where the point solid first
touches the bar (``bars.clearance_np``, bisected on the host) within 5 um, jammed, with no cut into the bar (1 um),
the front holding the tool (the head short of its target by thrust / kp, the bit pressed thrust / k: 2 %);
world 2 passes the bar's depth unjammed; world 3 cuts into the bar at the Shaw-Oxford rate of steel_200bhn at its
thrust (the ESTIMATE in ``bars``) within 2 % over the last 0.4 s, and past GA's 0.5 mm within the run.

``rebar_through``: the hole's bottom is the bar's memory of its cut (``bars.bar_through``). The same bit and a #4 bar
under 12 mm of cover (within the spindle's 40 mm reach), holes seeded down the axis; worlds 0 and 1 (over the bar's
axis, 8 mm off it) 0.2 mm short of the bottom past which the point has cut all of the bar within its radius, fed 4
mm past the seed (about 1.8 kN); world 2 over the axis 0.2 mm past it, world 3 16 mm off (no bar) as deep as world
2, both fed 4 mm on; world 4 as world 0 under a light push (1.5 mm, 650 N, below the chisel's 1.04 kN). The check:
worlds 0 and 1 cut the rest of the bar at the steel estimate's rate at their thrust (3 %, the advance while in the
bar against the rate integrated over the trace), then drill on in the concrete until the bit rests on the bottom
(past the bar by 2 mm; the bottom at the tip within 20 um), with the bar cut min(r_bar, R + r_bar - offset) (2 um);
world 2 drills on unjammed as its bar-free twin (1 um); world 4 stays jammed at its seed (1 um); the layer's ledger
against density x the union volume less the seeds' (1e-9).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.plugins.holes.bars import Bar, clearance_np, through_np
from alloy.plugins.holes.geometry import union_volume_np
from alloy.plugins.holes.work import PERCUSSIVE, energy_per_volume

ROW = "concrete_medium_hard"
BLOCK = (0.04, 0.04, 0.03)              # m: half extents (60 mm deep)
SPIN = 1160.0 * 2.0 * math.pi / 60.0    # rad/s
BLOW_E, BLOW_F = 1.8, 5100.0 / 60.0     # J, 1/s
LEAD = 1.10                             # the head's feed over the predicted hammer rate
PERIODS, WINDOW = 75, 20                # 1.5 s, the last 0.4 s is steady
DIAMETERS = (0.008, 0.010, 0.012, 0.016, 0.012, 0.012, 0.012, 0.012, 0.012)
ENERGIES = (BLOW_E, BLOW_E, BLOW_E, BLOW_E, 2.0 * BLOW_E, BLOW_E, 0.0, BLOW_E, BLOW_E)
RATES = (BLOW_F, BLOW_F, BLOW_F, BLOW_F, BLOW_F, 2.0 * BLOW_F, BLOW_F, BLOW_F, BLOW_F)
PILOTS = (0.0,) * 7 + (0.006, 0.0)      # m: a through pilot under the bit
SPINS = (SPIN,) * 8 + (280.0 * 2.0 * math.pi / 60.0,)
BASE, DOUBLED, OFF = 2, (4, 5), 6        # hammer_rate's 12 mm world, its doubled worlds, the hammer-off world
COVER, BAR_R, BIT = 0.025, 0.00635, 0.0127
OFFSETS = (0.0, 0.008, 0.016, 0.0)
PRESS = (0.0015, 0.0015, 0.0, 0.004)    # m: rebar_stop feeds this far past the first touch (world 2: to 40 mm)
REBAR_PERIODS = 150
THROUGH_X = (0.0, 0.008, 0.0, 0.016, 0.0)                     # rebar_through: the bit's offset from the bar
THROUGH_SEED = (-2.0e-4, -2.0e-4, 2.0e-4, 2.0e-4, -2.0e-4)    # m: the seed less the through bottom (world 3: world 2's)
THROUGH_PRESS = (0.004, 0.004, 0.004, 0.004, 0.0015)          # m: fed this far past the seed
THROUGH_COVER = 0.012                                         # m: rebar_through's bar, within the bit's 40 mm reach
LAYER = {"workpiece": "workpiece", "work": ROW, "density": 2400.0}
BAR = Bar(point=(0.0, 0.0, -(COVER + BAR_R)), direction=(0.0, 1.0, 0.0), radius=BAR_R)
BAR_T = Bar(point=(0.0, 0.0, -(THROUGH_COVER + BAR_R)), direction=(0.0, 1.0, 0.0), radius=BAR_R)


def predicted_rate(d: float, energy: float, rate: float, pilot: float = 0.0) -> float:
    """P / (e_v A) [m/s], A the full diameter's section less a through pilot's."""
    return energy * rate / (energy_per_volume(PERCUSSIVE[ROW], d) * 0.25 * math.pi * (d * d - pilot * pilot))


def _seed_pilots(drill: Any, pilots: np.ndarray) -> None:
    """A through pilot of ``pilots[w]`` (0: none) under the bit, at the face's centre."""
    n, sn = drill.hole_n.numpy(), drill.stage_n.numpy()
    axis, sd, sb = drill.hole_axis.numpy(), drill.stage_d.numpy(), drill.stage_b.numpy()
    for w in np.flatnonzero(pilots > 0.0):
        n[w, 0], sn[w, 0], axis[w, 0] = 1, 1, (0.0, 0.0, -1.0)
        sd[w, 0], sb[w, 0] = pilots[w], 2.0 * BLOCK[2] + 0.01
    for name, v in (("hole_n", n), ("stage_n", sn), ("hole_axis", axis), ("stage_d", sd), ("stage_b", sb)):
        getattr(drill, name).assign(v)


def _init(diameters, energies, rates, offsets=None, travel=None, pilots=(0.0,), spins=(SPIN,)):
    def init(sim: Any) -> None:
        from alloy.plugins.drilling import laws as dl  # noqa: PLC0415 (the drill imports this module)

        W = sim.model.world_count
        pick = lambda xs: np.asarray([xs[min(w, len(xs) - 1)] for w in range(W)], np.float64)  # noqa: E731
        lead = [LEAD * predicted_rate(d, e or BLOW_E, f, pilots[min(k, len(pilots) - 1)])
                for k, (d, e, f) in enumerate(zip(diameters, energies, rates, strict=True))]
        dl.feeding(tuple(lead), spins, diameters)(sim)
        drill, head = sim.plugins["drill"], sim.plugins["feed_head"]
        drill.set_hammer(pick(energies), pick(rates))
        _seed_pilots(drill, pick(pilots))
        if travel is not None:
            head.set_input("travel", pick(travel))
        if offsets is not None:
            q = sim.state.joint_q.numpy().reshape(W, 7)
            q[:, 0] = pick(offsets)
            sim.state.joint_q.assign(q.reshape(-1))
            newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
            hold = head.hold.numpy()
            hold[:, 0] = pick(offsets)
            head.hold.assign(hold)
    return init


def _scene(c: Any, init, worlds: int, periods: int = PERIODS) -> TestScene:
    from alloy.plugins.drilling import laws as dl  # noqa: PLC0415

    return dl.scene(c, init, worlds=worlds, periods=periods, half=BLOCK)


def _layer_ledger(run: Any, pilots: tuple[float, ...] = (0.0,)) -> float:
    """The worst relative error of the layer's removed mass against density x the union volume of its holes, less
    the seeded pilots'."""
    p, r = run.plugin, run.plugin.read()
    worst = 0.0
    for w in range(run.model.world_count):
        vol = sum(union_volume_np([0.5 * s.diameter for s in h.stages], [s.depth for s in h.stages], h.span,
                                  h.tan_half) - _seeded(h, pilots[min(w, len(pilots) - 1)]) for h in p.holes(w, 0))
        worst = max(worst, abs(float(r["layer_removed"][w, 0]) - 2400.0 * vol) / max(2400.0 * vol, 1e-12))
    return worst


def _seeded(h: Any, pilot: float) -> float:
    """The seeded through pilot's own volume in hole ``h`` (0 without one)."""
    if pilot <= 0.0:
        return 0.0
    return union_volume_np([0.5 * float(np.float32(pilot))], [2.0 * BLOCK[2] + 0.01], h.span, h.tan_half)


def check_hammer(run: Any) -> LawResult:
    from alloy.plugins.drilling.laws import _series  # noqa: PLC0415

    depth = _series(run, "depth")
    thrust = _series(run, "thrust")[-1]
    elapsed = WINDOW * float(run.trace.t[-1] - run.trace.t[-2])
    advance = (depth[-1] - depth[-WINDOW - 1]) / elapsed
    want = np.asarray([predicted_rate(d, e, f, p) if e > 0.0 else 0.0
                       for d, e, f, p in zip(DIAMETERS, ENERGIES, RATES, PILOTS, strict=True)])
    cut = [k for k in range(len(DIAMETERS)) if k != OFF]
    rate_err = float(np.abs(advance[cut] / want[cut] - 1.0).max())
    double = float(max(abs(advance[k] / (2.0 * advance[BASE]) - 1.0) for k in DOUBLED))
    off = float(depth[-1, OFF]) == 0.0 and float(thrust[OFF]) > 0.0
    hold = holding(run, (OFF,))
    ledger = _layer_ledger(run, PILOTS)
    worst = max(rate_err / 0.02, double / 0.02, 0.0 if off else math.inf, hold / 0.02, ledger / 1e-9)
    return LawResult.within(worst, 1.0,
                            f"advance {np.round(advance * 1e3, 3)} mm/s vs P/(e_v A) {np.round(want * 1e3, 3)} "
                            f"({rate_err:.3%}, 2 %); doubled energy / rate "
                            f"{advance[DOUBLED[0]] / advance[BASE]:.4f} / {advance[DOUBLED[1]] / advance[BASE]:.4f} "
                            f"(2 %); hammer off: depth {depth[-1, OFF]:.3g} m "
                            f"at {thrust[OFF]:.1f} N, the tool held to {hold:.3%} (2 %); ledger {ledger:.2g}",
                            "a rotary hammer deepens a hole at its blow power over the energy per removed volume")


def holding(run: Any, worlds: tuple[int, ...]) -> float:
    """The largest relative error of the tool's hold against the reported thrust T in ``worlds`` (at rest): the head
    short of its feed target by T / kp, and the bit's tip pressed T / k past the hole's bottom."""
    p, head = run.plugin, run.host["feed_head"]
    r = p.read()
    z = run.trace.body_q[-1][run.bodies(p.config.tool), 2].astype(np.float64)
    target = head.hold.numpy()[:, 2].astype(np.float64) - head.travel.numpy().astype(np.float64)
    tip_depth = 2.0 * BLOCK[2] - (z + float(p.config.tip[2]))
    worst = 0.0
    for w in worlds:
        t = float(r["thrust"][w])
        lag = (z[w] - target[w]) * float(head.config.kp) / t
        press = (tip_depth[w] - float(r["depth"][w])) * float(p.config.stiffness) / t
        worst = max(worst, abs(lag - 1.0), abs(press - 1.0))
    return worst


def bar_stop_depth(x: float) -> float:
    """Host bisection: the tip depth (below the face) at which a vertical 12.7 mm bit at x first touches BAR."""
    t = math.tan(math.radians(59.0))
    lo, hi = 0.0, COVER + 2.0 * BAR_R
    if clearance_np((x, 0.0, -hi), (0.0, 0.0, -1.0), 0.5 * BIT, t, BAR) > 0.0:
        return math.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if clearance_np((x, 0.0, -mid), (0.0, 0.0, -1.0), 0.5 * BIT, t, BAR) > 0.0 else (lo, mid)
    return lo


def bar_rate(run: Any, thrust: float) -> float:
    """The ESTIMATE's cut rate into steel_200bhn [m/s] at ``thrust`` (host side)."""
    from alloy.plugins.drilling.plugin import Layer, shaw_coefficients  # noqa: PLC0415

    sh = shaw_coefficients(run.plugin.config, Layer("bar", work="steel_200bhn"))
    net = thrust - float(sh.chisel) * BIT**2
    feed = (net / (float(sh.lips) * BIT**0.8)) ** 1.25 if net > 0.0 else 0.0
    return min(feed, float(run.plugin.config.feed_max)) * SPIN / (2.0 * math.pi)


def check_rebar(run: Any) -> LawResult:
    from alloy.plugins.drilling.laws import _series  # noqa: PLC0415

    r = run.plugin.read()
    depth, jammed, cut = r["depth"], r["jammed"], r["bar_cut"]
    stops = [bar_stop_depth(x) for x in OFFSETS]
    errs = [abs(float(depth[w]) - stops[w]) / 5e-6 for w in (0, 1)]
    errs += [0.0 if jammed[w] == 1 else math.inf for w in (0, 1, 3)]
    errs += [float(cut[:2].max()) / 1e-6]
    errs += [0.0 if (jammed[2] == 0 and float(depth[2]) > COVER + 2.0 * BAR_R) else math.inf]
    series, thrust = _series(run, "bar_cut")[:, 3], _series(run, "thrust")[:, 3]
    elapsed = WINDOW * float(run.trace.t[-1] - run.trace.t[-2])
    rate = (series[-1] - series[-WINDOW - 1]) / elapsed
    want = bar_rate(run, float(thrust[-WINDOW:].mean()))
    errs += [abs(rate / want - 1.0) / 0.02 if want > 0.0 else math.inf, 0.0 if cut[3] > 5e-4 else math.inf]
    hold = holding(run, (0, 1))
    errs += [hold / 0.02]
    return LawResult.within(max(errs), 1.0,
                            f"stopped at {np.round(depth[:2] * 1e3, 4)} mm vs first touch {np.round(stops[:2], 7)} m "
                            f"(5 um); jammed {jammed.tolist()}; light push: bar cut {float(cut[:2].max()) * 1e6:.3f} "
                            f"um (1 um); 16 mm off: {float(depth[2]) * 1e3:.2f} mm deep; hard push "
                            f"{float(thrust[-WINDOW:].mean()):.0f} N: cut {rate * 1e3:.4f} mm/s vs the steel estimate "
                            f"{want * 1e3:.4f} (2 %), {float(cut[3]) * 1e3:.3f} mm in all (0.5 mm); the jammed "
                            f"tools held to {hold:.3%} of thrust / kp and thrust / k (2 %)",
                            "a carbide bit stops on a bar and cuts it only as fast as steel's Shaw feed at its push")


def bar_through_depth(x: float) -> float:
    """The bottom (below the face) past which a vertical 12.7 mm hole at x has cut all of BAR_T it reaches (-inf:
    it never meets the bar)."""
    return through_np((x, 0.0, 0.0), (0.0, 0.0, -1.0), 0.5 * BIT, math.tan(math.radians(59.0)), BAR_T)


def through_seeds() -> list[float]:
    """rebar_through's seeded bottoms: each world's through bottom plus its THROUGH_SEED (world 3: world 2's)."""
    ref = bar_through_depth(0.0)
    return [(bar_through_depth(x) if math.isfinite(bar_through_depth(x)) else ref) + s
            for x, s in zip(THROUGH_X, THROUGH_SEED, strict=True)]


def _seed_holes(init, xs: tuple[float, ...], bottoms: list[float]):
    """init, then a BIT-wide hole down the axis at (xs[w], 0) to bottoms[w] in each world (its layer's first)."""
    def seeded(sim: Any) -> None:
        init(sim)
        drill = sim.plugins["drill"]
        n, sn, entry = drill.hole_n.numpy(), drill.stage_n.numpy(), drill.hole_entry.numpy()
        axis, sd, sb = drill.hole_axis.numpy(), drill.stage_d.numpy(), drill.stage_b.numpy()
        for w in range(n.shape[0]):
            k = min(w, len(xs) - 1)
            n[w, 0], sn[w, 0], entry[w, 0], axis[w, 0] = 1, 1, (xs[k], 0.0), (0.0, 0.0, -1.0)
            sd[w, 0], sb[w, 0] = BIT, bottoms[k]
        for name, v in (("hole_n", n), ("stage_n", sn), ("hole_entry", entry), ("hole_axis", axis), ("stage_d", sd),
                        ("stage_b", sb)):
            getattr(drill, name).assign(v)
    return seeded


def _bar_phase(run: Any, w: int, seed: float, end: float) -> tuple[float, float]:
    """World ``w``'s advance while its bottom was in the bar (from ``seed`` to short of ``end``), and the steel
    estimate's: its rate at the traced thrust, integrated (trapezoid) over the same periods."""
    from alloy.plugins.drilling.laws import _series  # noqa: PLC0415

    d, t = _series(run, "depth")[:, w], _series(run, "thrust")[:, w]
    dt = float(run.trace.t[-1] - run.trace.t[-2])
    inside = (d[:-1] >= seed) & (d[1:] < end)
    rates = np.asarray([bar_rate(run, float(x)) for x in t])
    return float(np.diff(d)[inside].sum()), float((0.5 * (rates[:-1] + rates[1:]) * dt)[inside].sum())


def check_through(run: Any) -> LawResult:
    p, r = run.plugin, run.plugin.read()
    depth, jammed, cut = r["depth"].astype(np.float64), r["jammed"], r["bar_cut"]
    seeds, ends = through_seeds(), [bar_through_depth(x) for x in THROUGH_X]
    z = run.trace.body_q[-1][run.bodies(p.config.tool), 2].astype(np.float64)
    tip = 2.0 * BLOCK[2] - (z + float(p.config.tip[2]))
    errs, rates = [], []
    for w in (0, 1):
        got, want = _bar_phase(run, w, seeds[w], ends[w])
        rates.append((got, want))
        errs += [abs(got / want - 1.0) / 0.03 if want > 0.0 else math.inf]
        errs += [0.0 if depth[w] > ends[w] + 0.002 else math.inf, abs(depth[w] - tip[w]) / 2e-5]
        errs += [abs(float(cut[w]) - min(BAR_R, 0.5 * BIT + BAR_R - THROUGH_X[w])) / 2e-6]
    errs += [0.0 if jammed[2] == 0 and jammed[3] == 0 and depth[2] > seeds[2] + 0.002 else math.inf]
    errs += [abs(depth[2] - depth[3]) / 1e-6, 0.0 if jammed[4] == 1 else math.inf, abs(depth[4] - seeds[4]) / 1e-6]
    ledger = 0.0
    for w in range(run.model.world_count):
        h = p.holes(w, 0)[0]
        r_bit = 0.5 * float(np.float32(BIT))
        vol = (union_volume_np([0.5 * s.diameter for s in h.stages], [s.depth for s in h.stages], h.span, h.tan_half)
               - union_volume_np([r_bit], [seeds[min(w, len(seeds) - 1)]], h.span, h.tan_half))
        ledger = max(ledger, abs(float(r["layer_removed"][w, 0]) - 2400.0 * vol) / max(2400.0 * vol, 1e-12))
    errs += [ledger / 1e-9]
    return LawResult.within(max(errs), 1.0,
                            f"in the bar: advance {np.round(np.array(rates) * 1e6, 2).tolist()} um vs the steel "
                            f"estimate (got, want; 3 %); through at {np.round(np.array(ends[:2]) * 1e3, 4)} mm, "
                            f"drilled "
                            f"on to {np.round(depth[:2] * 1e3, 3)} mm (tip {np.round(tip[:2] * 1e3, 3)}); bar cut "
                            f"{np.round(cut[:2] * 1e3, 4)} mm; past the bar {depth[2] * 1e3:.4f} vs bar-free "
                            f"{depth[3] * 1e3:.4f} mm; light push held at {depth[4] * 1e3:.4f} mm (seed "
                            f"{seeds[4] * 1e3:.4f}); jammed {jammed.tolist()}; ledger {ledger:.1e}",
                            "a hole's bottom past a bar means the bar is cut there: the bit drills on")


def concrete_laws(c: Any) -> list[Law]:
    hammer = _init(DIAMETERS, ENERGIES, RATES, travel=(1.0e9,) * OFF + (0.0012, 1.0e9, 1.0e9), pilots=PILOTS,
                   spins=SPINS)
    feeds = tuple(bar_stop_depth(x) + p if math.isfinite(bar_stop_depth(x)) else 0.04
                  for x, p in zip(OFFSETS, PRESS, strict=True))
    n = len(OFFSETS)
    rebar = _init((BIT,) * n, (4.0 * BLOW_E,) * n, (BLOW_F,) * n, offsets=OFFSETS, travel=feeds)
    cfg = {"layers": (LAYER,), "feed_max": 2.0e-3}
    seeds = through_seeds()
    m = len(THROUGH_X)
    feed_to = tuple(s + q for s, q in zip(seeds, THROUGH_PRESS, strict=True))
    through = _seed_holes(_init((BIT,) * m, (4.0 * BLOW_E,) * m, (BLOW_F,) * m, offsets=THROUGH_X, travel=feed_to),
                          THROUGH_X, seeds)
    return [Law("hammer_rate", check_hammer, _scene(c, hammer, len(DIAMETERS)),
                "D-6: a rotary hammer deepens a hole at blow power / (energy per volume x the cut section)",
                config={"layers": (LAYER,)}),
            Law("rebar_stop", check_rebar, _scene(c, rebar, len(OFFSETS), periods=REBAR_PERIODS),
                "D-7: a carbide bit stops on rebar where its point first touches the bar", config={
                    **cfg, "layers": ({**LAYER, "bars": (BAR,)},)}),
            Law("rebar_through", check_through, _scene(c, through, m, periods=REBAR_PERIODS),
                "D-7: a bit that has cut through a bar drills on, and a hole past a bar is re-entered freely",
                config={**cfg, "layers": ({**LAYER, "bars": (BAR_T,)},)})]


__all__ = ["BAR", "BAR_T", "bar_rate", "bar_stop_depth", "bar_through_depth", "check_hammer", "check_rebar",
           "check_through", "concrete_laws", "predicted_rate", "through_seeds"]
