# SPDX-License-Identifier: Apache-2.0
"""RodCard derivation and the CHAIN cantilever self-test (D7, D8, SPEC §3.8 Rods; CARD-7, ROD-1, ROD-2).

Self-test, on CPU with SolverMuJoCo at the routed links, integrator and grid dt:
1. static: clamp link 0, damp every hinge with τ = 1.4/ω1 (first mode ζ ≈ 0.7), settle 6 periods, tip drop vs the
   elastica (rodref.elastica_tip_drop) → `sag_err`; also vs the exact discrete chain → `sag_err_discrete`.
2. release ("stiff release"): the same chain at γ ≤ 0.5 (EI raised if the card is limper, so linear theory holds),
   undamped, released from straight; first peak and period vs rodref.release_reference → `drop_err`,
   `period_err` (and `period_err_continuum`).
Results are cached in-process and in `$ALLOY_CACHE_DIR/rodcheck.json`, keyed by what the test depends on: the RodCard's
physics (length, radius, density, EI, GJ, EA, damping ratio, γ at 12 significant digits), links, integrator, dt and
REF_VERSION (SCR-OSS22-3). A card hash also carries the name and curve points, so every scene-builder cable re-ran it
(a 0.6 m cotton rope: 210-240 s per route on CPU). An entry under the old card-hash key is still found and copied to
the new key.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from alloy.cards.types import ResolvedObject
from alloy.route import rodref
from alloy.route.types import RodCard, RouteDecision

log = logging.getLogger("alloy.route")

SAG_BOUND = 0.05            # CARD-7 / D8: |sag_err| ≤ 5 % at the routed N
DROP_BOUND = 0.05           # ROD-2: release tip drop ≤ 5 %
PERIOD_BOUND = 0.02         # ROD-2: first-mode period ≤ 2 %
RELEASE_GAMMA = 0.5
SETTLE_PERIODS = 6.0
RECORD_EVERY = 4
_MEMO: dict[str, dict[str, float]] = {}
# Stock VBD rod on newton ac7f7830 (solvers §3.2, SV:194-202; bench proto/solvers/vbd_rod_recheck.py): static sag
# 406.0 mm vs elastica 297.5 mm at 10 iterations (+36.5 %), 374.9 mm at 50 (+26 %). The record of why ROD_VBD runs
# Alloy's rod solve (SolverVBDRod, SCR-WP9-ROD), not stock VBD.
VBD_ROD_MEASURED = {"sag_err_it10": 0.365, "sag_err_it50": 0.26}


def rod_card(L: float, radius: float, area: float, inertia: float, mat, links: int, *,
             integrator: str = "discrete", self_contact: bool = False, features=None) -> RodCard:
    """RodCard from a rod section and its material card (E, ν, ρ; ζ from Rayleigh β if given, else 0.02).
    `Features.bend_stiffness` / `twist_stiffness` (measured composite EI / GJ, SCR-WP1-7) override the isotropic
    E·I and G·J."""
    E, nu = float(mat.elastic.youngs.value), float(mat.elastic.poisson.value)
    rho = float(mat.density.value)
    EI, GJ, EA = E * inertia, E / (2.0 * (1.0 + nu)) * 2.0 * inertia, E * area
    if features is not None and getattr(features, "bend_stiffness", None) is not None:
        EI = float(features.bend_stiffness.value)
    if features is not None and getattr(features, "twist_stiffness", None) is not None:
        GJ = float(features.twist_stiffness.value)
    gamma = rho * area * rodref.G * L**3 / EI
    rod = RodCard(length=L, radius=radius, density=rho * area / (math.pi * radius**2), EI=EI, GJ=GJ, EA=EA,
                  links=links, gamma=gamma, self_contact=self_contact, integrator=integrator)
    beta = mat.elastic.rayleigh_beta
    if beta is not None:
        w1 = rodref.CANTILEVER_BETA1**2 * math.sqrt(EI / (rho * area * L**4))
        rod = replace(rod, damping_ratio=0.5 * float(beta.value) * w1)
    return rod


ROD_NUMBER_KEYS = ("rod_length", "rod_radius", "rod_density", "EI", "GJ", "EA", "links", "gamma", "damping_ratio")


def rod_numbers(rod: RodCard) -> dict[str, float]:
    """RodCard fields as RouteDecision.numbers entries (the emitter rebuilds the card from them)."""
    vals = (rod.length, rod.radius, rod.density, rod.EI, rod.GJ, rod.EA, rod.links, rod.gamma, rod.damping_ratio)
    return {k: float(v) for k, v in zip(ROD_NUMBER_KEYS, vals)}


def rod_card_from_decision(obj: ResolvedObject, d: RouteDecision) -> RodCard:
    """Rebuild the router's RodCard from a CHAIN decision (integrator from scene effects)."""
    n = d.numbers
    integ = "implicitfast" if "mujoco:integrator=implicitfast" in d.scene_effects else "discrete"
    return RodCard(length=n["rod_length"], radius=n["rod_radius"], density=n["rod_density"], EI=n["EI"],
                   GJ=n["GJ"], EA=n["EA"], links=int(n["links"]), damping_ratio=n["damping_ratio"],
                   gamma=n["gamma"], self_contact=obj.card.features.self_contact, integrator=integ)


def _cache_path() -> Path:
    root = Path(os.environ.get("ALLOY_CACHE_DIR", Path.home() / ".cache" / "alloy"))
    return root / "rodcheck.json"


# v2: the static reference is solved by damped Newton for any γ (v1's fixed point diverged above γ ≈ 10 and gave the
# 1 m USB cable, γ = 1944, a negative elastica reference: sag_err −3009 %)
# v3: link inertias as authored (Newton's finalize floor had raised stubby links' to 1e-6 kg·m²; harness_lead)
# v4: CHAIN armature floor 1e-6 kg·m² on every rotational dof (emitters/chain.ARMATURE_FLOOR; A10)
REF_VERSION = 4


KEY_FIELDS = ("length", "radius", "density", "EI", "GJ", "EA", "damping_ratio", "gamma")


def _cache_key(rod: RodCard, dt: float) -> str:
    phys = "|".join(f"{float(getattr(rod, f)):.12g}" for f in KEY_FIELDS)
    return f"v{REF_VERSION}|rod|{phys}|{rod.links}|{rod.integrator}|{dt:.9e}"


def _legacy_key(card_hash: str, rod: RodCard, dt: float) -> str:
    return f"v{REF_VERSION}|{card_hash}|{rod.links}|{rod.integrator}|{dt:.9e}"


def _recall(key: str, legacy: str) -> dict[str, float] | None:
    """The cached result under `key`, else under the legacy card-hash key (then stored under `key` too)."""
    hit = _MEMO.get(key)
    if hit is None:
        data = _cache_load()
        hit = data.get(key)
        if hit is None and data.get(legacy) is not None:
            hit = data[legacy]
            _cache_store(key, hit)
    if hit is not None:
        _MEMO[key] = hit
    return hit


def _cache_load() -> dict[str, dict[str, float]]:
    try:
        return json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return {}


def _cache_store(key: str, value: dict[str, float]) -> None:
    path = _cache_path()
    data = _cache_load()
    data[key] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, sort_keys=True, indent=1))
        tmp.replace(path)
    except OSError as exc:
        log.warning("rodcheck cache not written (%s): %s", path, exc)


def self_test(rod: RodCard, card_hash: str, dt: float = 1.25e-3, *, use_cache: bool = True) -> dict[str, float]:
    """Run (or recall) the cantilever self-test of `rod` at its links and integrator; returns the error numbers."""
    key = _cache_key(rod, dt)
    if use_cache:
        hit = _recall(key, _legacy_key(card_hash, rod, dt))
        if hit is not None:
            return dict(hit)
    out = _static_test(rod, dt)
    out.update(_release_test(rod, dt))
    out["passed"] = float(passes(out))
    _MEMO[key] = out
    if use_cache:
        _cache_store(key, out)
    log.info("rod self-test %s links=%d %s: sag %+.3f %%, drop %+.3f %%, period %+.3f %%", card_hash[:8],
             rod.links, rod.integrator, 100 * out["sag_err"], 100 * out["drop_err"], 100 * out["period_err"])
    return out


def warm(device: str = "cpu") -> dict[str, int]:
    """Run every self-test that a library scene, or the test helpers' cable, can ask for, so a cold CI runner starts
    with `rodcheck.json` filled (SCR-OSS1-9: WP0's `python -m alloy.core.warm` rodcheck job; one test timed out at
    900 s in a cold self-test).

    Each library cable (a dynamic card whose shape is a curve: the USB cable of T09 and the weld tests, the harness
    lead, the scene builder's shoelace fallback) and `alloy.testing.scene.cable()` go through the default Router in
    both scene contexts: discrete, and implicitfast for a scene with a gap-ladder (L1) precision pair, falling back to
    discrete as the router does. The keys are therefore the ones real routes look up, with the router's own link
    counts. Solid primitives of rod shape (pencils, a chopstick) are left out: they route rigid in their tasks, and the
    untasked pencil's CHAIN fails its self-test at every link count, a ladder of 9 cold self-tests no scene or test
    needs. `device` is the warm job's common argument: the self-test runs on CPU wherever its scene runs, and its key
    has no device. Returns the cables routed, the self-tests run (cache misses) and the routes that failed (logged)."""
    from alloy.cards.library import CardLibrary
    from alloy.cards.types import CurveSource
    from alloy.core.errors import AlloyError
    from alloy.route.router import Router
    from alloy.route.types import Route, SceneContext
    from alloy.testing import scene as test_scene

    lib = CardLibrary(assets=None)
    rods = [lib.resolve(card) for _, card in lib.iter_cards("objects")
            if card.motion == "dynamic" and isinstance(card.shape, CurveSource)]
    rods.append(test_scene.cable())
    dt = Router().chain_dt
    counts = {"cables": len(rods), "self_tests_run": 0, "failed": 0}

    def counted(rod: RodCard, card_hash: str) -> dict[str, float]:
        if _recall(_cache_key(rod, dt), _legacy_key(card_hash, rod, dt)) is None:
            counts["self_tests_run"] += 1
        return self_test(rod, card_hash, dt)

    router = Router(chain_dt=dt, rod_self_test=counted)
    routes = frozenset(str(r) for r in Route)
    for obj in rods:
        for l1 in (False, True):
            ctx = SceneContext(rigid_partners={}, free_bodies_per_world=1, registered_routes=routes, needs_l1=l1)
            try:
                router.route(obj, ctx)
            except AlloyError as exc:
                counts["failed"] += 1
                log.warning("rodcheck warm: %s (L1 %s) did not route: %s", obj.card.name, l1, exc)
    log.info("rodcheck warm (%s): %d cables, %d self-tests run, %d failed routes", device, counts["cables"],
             counts["self_tests_run"], counts["failed"])
    return counts


def sag_links(rod: RodCard, max_links: int) -> int:
    """SCR-DC3-7: the smallest link count from `rod.links` up whose exact discrete chain sags within SAG_BOUND of the
    elastica, at most `max_links`. The self-test's sag_err is this discretization: the simulated chain matched the
    discrete reference to < 0.01 % on all 12 library cables ([internal note]), and at the bare sag rule
    (⌈L/c⌉ + 1) a stiff short cord sags 13.9 % short at 3 links."""
    lin = _lin_mass(rod)
    ref = rodref.elastica_tip_drop(rod.length, lin, rod.EI)
    n = rod.links
    while n < max_links:
        disc = rodref.static_tip_drop(rodref.chain_layout(n, rod.length, lin, rod.EI, rod.radius))
        if abs(disc / ref - 1.0) <= SAG_BOUND:
            return n
        n += 1
    return max(n, rod.links)


def passes(r: dict[str, float]) -> bool:
    return (abs(r["sag_err"]) <= SAG_BOUND and abs(r["drop_err"]) <= DROP_BOUND
            and abs(r["period_err"]) <= PERIOD_BOUND and r.get("finite", 1.0) > 0.0)


def _lin_mass(rod: RodCard) -> float:
    return rod.density * math.pi * rod.radius**2


def _static_test(rod: RodCard, dt: float) -> dict[str, float]:
    from alloy.route.rodsim import run_clamped

    lin = _lin_mass(rod)
    w1 = rodref.CANTILEVER_BETA1**2 * math.sqrt(rod.EI / (lin * rod.length**4))
    t_end = SETTLE_PERIODS * 2.0 * math.pi / w1
    t, drop = run_clamped(rod, dt, t_end, damping_tau=1.4 / w1)
    ref = rodref.elastica_tip_drop(rod.length, lin, rod.EI)
    disc = rodref.static_tip_drop(rodref.chain_layout(rod.links, rod.length, lin, rod.EI, rod.radius))
    tail = drop[t >= t[-1] - 2.0 * math.pi / w1]
    final = float(drop[-1])
    return {"sag_m": final, "elastica_m": ref, "sag_err": final / ref - 1.0, "sag_err_discrete": final / disc - 1.0,
            "settle_ptp_m": float(np.ptp(tail)) if tail.size else math.nan, "finite": float(np.isfinite(drop).all())}


def _release_test(rod: RodCard, dt: float) -> dict[str, float]:
    from alloy.route.rodsim import period_from_signal, run_clamped

    stiff = rod if rod.gamma <= RELEASE_GAMMA else replace(rod, EI=rod.EI * rod.gamma / RELEASE_GAMMA,
                                                         GJ=rod.GJ * rod.gamma / RELEASE_GAMMA, gamma=RELEASE_GAMMA)
    lin = _lin_mass(stiff)
    lay = rodref.chain_layout(stiff.links, stiff.length, lin, stiff.EI, stiff.radius)
    probe = rodref.release_reference(lay, stiff.length, lin, stiff.EI, np.asarray([0.0]))
    t, drop = run_clamped(stiff, dt, 3.2 * probe.period, damping_tau=0.0)
    ref = rodref.release_reference(lay, stiff.length, lin, stiff.EI, t)
    first = t <= ref.period
    peak = float(np.max(drop[first]))
    period = period_from_signal(t, drop - ref.static_drop)
    return {"release_gamma": stiff.gamma, "peak_m": peak, "peak_ref_m": ref.peak_drop,
            "drop_err": peak / ref.peak_drop - 1.0, "period_s": period, "period_ref_s": ref.period,
            "period_err": period / ref.period - 1.0, "period_err_continuum": period / ref.period_continuum - 1.0}
