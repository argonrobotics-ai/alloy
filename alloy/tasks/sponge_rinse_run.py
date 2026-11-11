# SPDX-License-Identifier: Apache-2.0
"""rinse-sponge, run: the scripted rinse at the pot sink (:mod:`alloy.tasks.sponge_rinse_scene`), the measured state
(``sponge_rinse_score.SpongeReadout``) and each world's verdict.

Stage 1 (WP15's order: the scene, the gates and the score first). A privileged hand (WP8's ``PoseHand``, no robot)
holds the sponge, as in GA's assisted demo A5 (refills and 60 N squeezes, then a 120 N wring):
1. ``Wring.cycles`` times: dunk the sponge ``dunk`` under the rinse bowl's level and swish it for ``soak`` s (its
   pores fill and swap with the clean water), then press it on the ledge's left end with ``squeeze_n`` N for
   ``press`` s (the excess goes onto the ledge's film and down its drain: GA's ledge keeps it).
2. A last press with ``wring_n`` N, then carry the sponge to the rest spot (``Wring.rest_at``; by default the
   ledge's right end), set it down and let go.
3. The score's pore concentration and fill are read then (GA's evaluator reads them live), and the sink settles
   hands-off on GA's schedule: 2 s, then 1 s at a time until quiet (the faucet off, the sponge still, the film's books
   and the pores' water steady within 0.2 mL or 2 % per second: stricter than GA's quiet, which has no pore term), at
   most ``settle`` s. The gates are read then. The readout's ``held`` is the script's own grip, not a state read: it
   can fail hands-free, never pass it; let go, the hands part is not judged (``sponge_rinse_score``, AB-gym-3).
A press is pose-controlled (WP9): the hand (stiffer than the pad, at the 1 ms step) sets the pad's depth for the
force asked, F / K_PAD, and the foam spring gives the strain; the target goes F / KP under the touch, KP being the hand
and the pad in series. The books: ``books`` (film, pores, bowls, network, whole system), each closing against the
start's.

Every period ORs MuJoCo Warp's capacity overflow bits into the fault gate, over every substep: WP8's
``OverflowWatch`` plug-in keeps them (Newton clears the word before each MuJoCo substep; DC5's review) and
``period_overflow`` reads them. Alloy's CPU route (MuJoCo C, ``alloy.route.mjc_core``, the default without a GPU)
writes the same bits for what each call dropped beyond ``CAPACITY`` (SCR-WP8-8): the starve test passes on both.

Example::

    from alloy.tasks.sponge_rinse_run import Wring, run
    print(run(wring=Wring(cycles=2))["worlds"][0]["failed"])
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.tasks.can_wash_books import GONE, closed, dropped_back, in_flight, net_book, tank_book, with_pending
from alloy.tasks.can_wash_run import finite, quiet_now, settle_steps
from alloy.tasks.can_wash_score import row
from alloy.tasks.paint_supply import supply_drift
from alloy.tasks.rinse_flush_checks import quiet_books
from alloy.tasks.rinse_flush_policy import CONTROL_DT, bodies
from alloy.tasks.rinse_hand import OverflowWatch, PoseHand, mjw_solver, overflow_watch, period_overflow
from alloy.tasks.rinse_plates import DROP_Z
from alloy.tasks.rinse_sink import BOWL, BOWLS, FILL, GPM
from alloy.tasks.rinse_wash_station import SPECIES
from alloy.tasks.sponge_rinse_scene import A_EFF, BLOCK, CELLULOSE_H, MODULUS, SPONGE, SpongeSink
from alloy.tasks.sponge_rinse_score import SpongeReadout, pore_conc, verdict

GAINS = {"kp": 25000.0, "kd": 50.0, "kr": 6.0, "dr": 0.012}  # PoseHand's PD on the 67 g sponge at the 1 ms step
                                      # (within its limits: kp dt²/m, kr dt²/I <= 0.5; kd dt/m, dr dt/I <= 1)
K_PAD = MODULUS / CELLULOSE_H * A_EFF  # N/m, the pad's spring (12.5 kN/m)
KP = K_PAD * GAINS["kp"] / (K_PAD + GAINS["kp"])   # the hand and the pad in series: a press of F newtons sets its
                                      # target F/KP under the touch, so the pad settles F/K_PAD deep (WP9: pose-
                                      # controlled depth for v1; the force follows from the foam spring)
SOAP = SPECIES.index("soap")


@dataclass(frozen=True)
class Wring:
    """The script (module doc); distances [m], speeds [m/s], forces [N], times [s]."""

    cycles: int = 3
    dunk: float = 0.10
    swish: float = 0.05
    swish_speed: float = 0.20
    soak: float = 1.0
    squeeze_n: float = 60.0
    wring_n: float = 120.0
    press: float = 0.5
    travel: float = 0.5
    clear: float = 0.08                   # over the rim while carried
    end_inset: float = 0.10               # the squeeze and rest spots, in from the ledge's ends
    rest_at: float | None = None          # the rest spot, from the ledge's left end (None: end_inset from its right)
    settle: float = 30.0                  # s, the settle's cap (GA's)


class Hand:
    """Steps the simulation while the hand moves the sponge; every world in lockstep."""

    def __init__(self, sim: Any, hand: Any, worlds: int) -> None:
        self.sim, self.hand = sim, hand
        q = sim.state.body_q.numpy()[hand.held.numpy()]
        self.pos, self.quat = q[..., :3].astype(np.float64), q[..., 3:].astype(np.float64)
        self.vel = np.zeros_like(self.pos)
        self.grip = np.zeros(self.pos.shape[:2], np.float32)
        self.solver, self.overflow = mjw_solver(sim), np.zeros(worlds, bool)
        self.watch = overflow_watch(sim)          # every substep's bits (the sim lists an OverflowWatch)

    def step(self) -> None:
        for name, value in (("target_pos", self.pos), ("target_quat", self.quat), ("target_vel", self.vel),
                            ("grip", self.grip)):
            self.hand.set_input(name, value.astype(np.float32))
        self.sim.step_period()
        bits = period_overflow(self.watch, self.solver)
        if bits is not None:
            self.overflow |= bits

    def hold(self, seconds: float) -> None:
        self.vel[:] = 0.0
        for _ in range(max(1, int(round(seconds / CONTROL_DT)))):
            self.step()

    def move(self, to: np.ndarray, speed: float) -> None:
        """A straight line from the current target to ``to`` at ``speed``."""
        start = self.pos[0, 0].copy()
        n = max(1, int(np.ceil(np.linalg.norm(to - start) / (speed * CONTROL_DT))))
        v = (to - start) / (n * CONTROL_DT)
        for k in range(1, n + 1):
            self.pos[:, 0], self.vel[:, 0] = start + (to - start) * k / n, v
            self.step()
        self.vel[:] = 0.0


def script(task: SpongeSink, wr: Wring, hand: Hand, probe: Any = None) -> None:
    """The rinse (module doc, steps 1-2)."""
    sink, lo, hi = task.sink, *(np.asarray(v) for v in (task.ledge().lo, task.ledge().hi))
    y, top = 0.5 * (lo[1] + hi[1]), task.rest_z
    carry = sink.rim_z + 0.5 * BLOCK[2] + wr.clear
    level = sink.rim_z - BOWL[2] + FILL
    bowl = np.array([sink.bowl_x(1), 0.5 * BOWL[1], level - wr.dunk])
    squeeze = np.array([lo[0] + wr.end_inset, y, top])
    rest = np.array([hi[0] - wr.end_inset if wr.rest_at is None else lo[0] + wr.rest_at, y, top])

    def press(force: float) -> None:
        hand.move(np.array([*squeeze[:2], carry]), wr.travel)
        hand.move(squeeze + [0.0, 0.0, 0.005], wr.travel)
        hand.move(squeeze - [0.0, 0.0, force / KP], 0.1)
        hand.hold(wr.press)
        hand.move(np.array([*squeeze[:2], carry]), 0.2)

    hand.grip[:] = 1.0
    hand.move(hand.pos[0, 0] + [0.0, 0.0, wr.clear], 0.2)
    for _ in range(wr.cycles):
        hand.move(np.array([*bowl[:2], carry]), wr.travel)
        hand.move(bowl, wr.travel)
        for k in range(int(round(wr.soak * wr.swish_speed / (2.0 * wr.swish)))):
            hand.move(bowl + [wr.swish * (1 - 2 * (k % 2)), 0.0, 0.0], wr.swish_speed)
        hand.move(bowl, wr.swish_speed)
        hand.move(np.array([*bowl[:2], carry]), wr.travel)
        press(wr.squeeze_n)
    press(wr.wring_n)
    hand.move(np.array([*rest[:2], carry]), wr.travel)
    hand.move(rest + [0.0, 0.0, 0.002], 0.1)
    if probe is not None:
        probe()                           # a readout while the hand still grips (it fails hands-free)
    hand.grip[:] = 0.0
    hand.hold(CONTROL_DT)


def system_book(out: dict, lumps: dict, net_species: np.ndarray) -> np.ndarray:
    """``[W, K]`` kg of each species anywhere or gone: the film, the pores, the bowls, what left the film other than
    into tanks, what left the network into nothing (``can_wash_books.GONE``), and the pores' drips: waiting, and
    those that left as drops less what the drops took back (``can_wash_books.dropped_back``: onto a chart, into a
    bowl), so in flight or below the floor."""
    lump = lumps["lump_species"] + lumps.get("lump_species_drip_pending", 0.0) + lumps.get("lump_species_dropped", 0.0)
    total = out["species_mass"] + lump.sum(axis=1) + net_species.sum(axis=1) + out.get("film_species_pending", 0.0)
    total = total + in_flight(out)
    return total + out["species_out"].sum(axis=1) - out["supply_species_tanks"] + sum(
        out[f"supply_species_{c}"] for c in GONE) - dropped_back(out)


def pairs(out: dict, lumps: dict) -> np.ndarray:
    """``[W, 3]`` m³ the water hand-offs' two sides' difference, 0 while they agree: the pores and the bowls
    (``lump_pooled`` + pending against the supply's ``lump_pool``, less the drops and drips it took in:
    ``drops_to_tanks``, ``lump_drip_tanks``), the film and the pores (``to_lumps`` against ``lump_uptake``), and the
    pores and the film (what the film was supplied against what the pores squeezed or shed onto it,
    ``drops_lump_wet``, and what landed as drops)."""
    took = out.get("drops_to_tanks", 0.0) + lumps.get("lump_drip_tanks", np.zeros((1, 1))).sum(1)
    pool = lumps["lump_pooled"].sum(1) + lumps["lump_pool_pending"].sum(1) + out["supply_lump_pool"] - took
    up = out["to_lumps_volume"] - lumps["lump_uptake"].sum(1)
    shed = out.get("drops_lump_wet", 0.0) + out.get("drops_landed", 0.0)
    give = out["supplied_volume"] + out["pending_volume"] - lumps["lump_squeezed"].sum(1) - shed
    return np.stack([pool, up, give], 1)


def books(out: dict, lumps: dict, net_species: np.ndarray, tanks: Sequence[int], edges: Sequence[int]) -> dict:
    """Every book's value now; later readouts close against the start's (the start writes the pores' water and soap):
    the film's water and the pores' (their ledgers), the bowls' water (``can_wash_books.tank_book`` over the three
    tanks and their edges), the film's, the pores', the network's and the whole system's species."""
    return {"water": out["ledger"] + lumps["lump_ledger"].sum(1), "tank": tank_book(out, tanks, edges),
            "pairs": pairs(out, lumps),
            "species": out["species_ledger"] + lumps["lump_species_ledger"].sum(1),
            "net": net_book(out, net_species), "system": system_book(out, lumps, net_species)}


def ledgers(out: dict, lumps: dict, net_species: np.ndarray, ref: dict, tanks: Sequence[int],
            edges: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[W]`` water drift (the film's and the pores' books and their hand-offs, ``pairs``, relative to what moved:
    supplied, squeezed, pooled; the bowls' relative to what they hold), species drift (relative to each species'
    start, kg; ``can_wash_books.closed`` ORs per species, so a species that starts at 0 must close within 1e-12 kg),
    since ``ref``."""
    W = out["h"].shape[0]
    now = books(out, lumps, net_species, tanks, edges)
    moved = out["supplied_volume"] + out["pending_volume"] + np.abs(lumps["lump_squeezed"].sum(1))
    moved = np.maximum(moved + np.abs(lumps["lump_pooled"].sum(1)), 1e-9)
    held = out["supply_amount"][:, list(tanks)].sum(axis=1)      # the bowls' 212 L: their book rounds at its scale
    water = np.maximum(np.abs(now["water"] - ref["water"]) / moved, np.abs(now["tank"] - ref["tank"]) / held)
    water = np.maximum(water, np.abs(now["pairs"] - ref["pairs"]).max(axis=1) / moved)
    ran = np.asarray(out["supply_sent"]) > 0.0
    water = np.maximum(water, [supply_drift(out, w) if ran[w] else 0.0 for w in range(W)])
    drift = np.maximum.reduce([np.abs(now[k] - ref[k]) for k in ("species", "net", "system")])
    rel, kg = closed(drift, np.maximum(ref["start_kg"], 1e-30))
    return water, rel, kg


def scour_face(q: np.ndarray) -> np.ndarray:
    """``[W, 3]`` the sponge's scour face's centre (GA's origin for the ledge gate), from its body poses [W, 7]."""
    q = np.asarray(q, float)
    u, w = q[:, 3:6], q[:, 6:7]
    v = np.array([0.0, 0.0, -0.5 * BLOCK[2]])
    t = 2.0 * np.cross(u, np.broadcast_to(v, u.shape))
    return q[:, :3] + v + w * t + np.cross(u, t)


def intact_of(pos: np.ndarray) -> np.ndarray:
    """``[W]`` not dropped: the sponge's origin (its scour face) at or over GA's 0.25 m line (``task.py:689-692``;
    the sponge is not a floor item)."""
    return np.asarray(pos)[:, 2] >= DROP_Z


def quiet_state(out: dict, r: SpongeReadout) -> tuple[np.ndarray, np.ndarray]:
    """What the settle's quiet check compares (``can_wash_run.quiet_now``): the film's books and the pores' water
    [mL]. GA's quiet has no pore term (the pores' water counts in its 1 g/s rule); this is stricter."""
    return quiet_books(out), np.asarray(r.pore_m3) * 1e6


@dataclass
class Where:
    """The network's pieces the readout needs: the bowls' tanks, the edges into them, the faucet's edge."""

    tanks: tuple[int, ...]
    edges: tuple[int, ...]
    faucet: int

    @classmethod
    def of(cls, flow: Any) -> Where:
        net = flow.supply.net
        tanks = tuple(net.node_index(b) for b in BOWLS)
        edges = tuple(i for i, e in enumerate(net.edges) if net.node_index(e.a) in tanks)
        return cls(tanks, edges, net.edge_index("faucet"))


def readout(flow: Any, hand: Hand, body: np.ndarray, where: Where, ref: dict | None) -> tuple[SpongeReadout, dict]:
    """The measured state now (``SpongeReadout``) and the books (with ``start_kg``); ``ref``: the start's (None:
    now)."""
    out, lumps = with_pending(flow.read(), flow.tables.area_np), flow.lumped.read()
    net_sp = flow.supply.fluid.species.numpy()
    now = {**books(out, lumps, net_sp, where.tanks, where.edges), "start_kg": lumps["lump_species"][:, 0, :].copy()}
    ref = ref or now
    water, rel, kg = ledgers(out, lumps, net_sp, ref, where.tanks, where.edges)
    W = len(body)
    q = hand.sim.state.body_q.numpy()[body]
    r = SpongeReadout(pore_soap_kg=lumps["lump_species"][:, 0, SOAP], pore_m3=lumps["lump_water"][:, 0],
                      pos=scour_face(q), faucet_gpm=np.abs(out["supply_discharge"][:, where.faucet]) / GPM,
                      held=hand.grip[:, 0] > 0.0, intact=intact_of(scour_face(q)), water_rel=water, species_rel=rel,
                      species_kg=kg, finite=np.ones(W, bool))
    r.finite = finite(r, hand.overflow)
    return r, now


def settle(flow: Any, hand: Hand, body: np.ndarray, where: Where, ref: dict,
           cap: float) -> tuple[list[SpongeReadout], list[float]]:
    """GA's settle schedule, as rinse-garbage-can's (``can_wash_run.settle_steps``, ``quiet_now``): hands off, 2 s,
    then 1 s at a time until quiet (the faucet off, the sponge still, ``quiet_state`` steady: stricter than GA's, which
    has no pore term), at most ``cap`` s; each world's readout then, and when."""
    W = len(body)
    rows: list[SpongeReadout | None] = [None] * W
    at = [0.0] * W
    t, prev = 0.0, None
    for step in settle_steps(cap):
        hand.hold(step)
        t += step
        r, _ = readout(flow, hand, body, where, ref)
        now = quiet_state(flow.read(), r)
        done = np.full(W, t >= cap - 1e-9)
        if prev is not None:
            speed = np.linalg.norm(hand.sim.state.body_qd.numpy()[body][:, :3], axis=1)
            done |= quiet_now(prev, now, r.faucet_gpm, speed, step)
        for w in np.flatnonzero(done):
            if rows[w] is None:
                rows[w], at[w] = r, round(t, 3)
        prev = now
        if all(x is not None for x in rows):
            break
    return rows, at


def run(task: SpongeSink | None = None, wring: Wring | None = None, worlds: int = 1, seed: int = 0,
        device: str = "cpu") -> dict:
    """The scripted rinse (module doc) in ``worlds`` worlds; each world's verdict and its start, live and settled
    readouts."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run

    task, wr = task or SpongeSink(), wring or Wring()
    flow, made = task.make_flow(), {}
    pose_hand = PoseHand(bodies=(SPONGE,), f_max=1.25 * wr.wring_n, **GAINS)

    def init(sim: Any) -> None:
        made["strength"] = task.start(flow, worlds, seed)

    sim = build_simulation(task.scene(init, worlds), [flow, pose_hand, OverflowWatch()], device)
    hand, body, where = Hand(sim, pose_hand, worlds), bodies(sim.model, SPONGE), Where.of(flow)
    r0, ref = readout(flow, hand, body, where, None)
    gripped: dict = {}
    script(task, wr, hand, lambda: gripped.update(r=readout(flow, hand, body, where, ref)[0]))
    live, _ = readout(flow, hand, body, where, ref)
    ends, at = settle(flow, hand, body, where, ref, wr.settle)
    ledge, rows = task.ledge(), []
    for w in range(worlds):
        v = verdict(r0, ends[w], w, ledge, live=live)
        rows.append(row(v, at[w]))
    end = ends[0]
    return {"worlds": rows, "seed": seed, "strength": np.asarray(made["strength"]).tolist(),
            "overflow": hand.overflow.tolist(), "start": _row(r0), "live": _row(live), "end": [_row(e) for e in ends],
            "pore_ml": [float(e.pore_m3[w]) * 1e6 for w, e in enumerate(ends)],
            "pore_g_l": [float(pore_conc(e, w)) for w, e in enumerate(ends)], "first_end": _row(end),
            "gripped": _row(gripped["r"]), "route": type(hand.solver).__name__ if hand.solver else "MuJoCo C"}


def _row(r: SpongeReadout) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(r).items()}


__all__ = ["GAINS", "Hand", "KP", "K_PAD", "Where", "Wring", "books", "intact_of", "ledgers", "pairs", "quiet_state",
           "readout", "run", "scour_face", "script", "settle", "system_book"]
