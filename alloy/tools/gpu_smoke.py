# SPDX-License-Identifier: Apache-2.0
"""GPU smoke scenes: one short graph-mode run per solver family (WP5; coordinator ruling 2026-11-06 11:20Z).

    python -m alloy.tools.gpu_smoke --scene NAME --out DIR      # one scene; DIR/smoke.json; exit 0 pass, 1 fail
    python -m alloy.tools.gpu_smoke --list

The post-merge runner (WP1) runs these every 3 hours on main's tip through ``python -m alloy.tools.nightly_gpu run
--smoke``, one step per scene beside the GPU smoke-marked tests. They cover what CPU tests cannot: graph capture and
replay, memory at 64 worlds, order-sensitive reductions and device transfers. A scene runs through
``coupled_det_probe`` (the build the standard profile makes) and fails on:
- an error (build, capture or step);
- a world latched INVALID by health (NaN, blow-up, penetration);
- eager periods where a graph was expected;
- two builds of a RUN_TO_RUN scene (the ordered route) that differ in any world after any period;
- a mempool high-water above ``MEM_CAP_GIB``.
The film family has no card scene yet: its scene is the film hang law (``alloy.solvers.film.laws.hang``) cut to
0.3 s, which must stay finite and keep its step gain below 1e-2. Cables: the router builds a cable as a MuJoCo
chain, so ``cable`` runs that; WP2's ``SolverVBDRod`` is in no scene yet (its tests are CPU only), so the smoke does
not cover it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("alloy.tools.gpu_smoke")

MPM_WORLDS = 16
MEM_CAP_GIB = 16.0          # an L4 has 22 GiB; a 64-world smoke scene above this has a leak or a capacity bug
DEVICE = "cuda:0"


@dataclass(frozen=True)
class Scene:
    case: str            # coupled_det_probe.CASES key
    worlds: int
    periods: int
    repeat: int = 1      # 2: the two builds must be bitwise equal when the claim is RUN_TO_RUN
    graph: bool = True   # the standard profile captures this scene


SCENES: dict[str, Scene] = {
    "t12_route": Scene("t12_route", 64, 10, repeat=2),           # robot arm, MuJoCo Warp on the ordered route
    "vbd_cloth": Scene("vbd_cloth", 64, 4),                      # cloth (VBD) on a table, 64 worlds
    "sph_pour": Scene("demo_pour", 16, 3),                       # SPH water into a tumbler (library card)
    # MPM sand heap: 16 worlds ran out of memory instantiating the period graph on an L4 until WP12's df857077d
    # (the persistent grid reserved 8 cells per particle; smoke trial 2026-11-06 ~12:00Z)
    "mpm_sand": Scene("mixed_sand_heap", MPM_WORLDS, 3),
    "cable": Scene("mixed_cable_on_table", 16, 3),               # cable on a table, as the router builds it
}
FILM = "film_hang"


def judge(meta: dict, digests: np.ndarray, scene: Scene, mem_cap_gib: float = MEM_CAP_GIB) -> list[str]:
    """Why the run fails (empty: it passes). ``digests``: [repeat, periods + 1, worlds, 2] from the probe."""
    if meta.get("error"):
        return [f"error: {meta['error']}"]
    why = []
    if any(int(n) for n in meta.get("invalid_worlds", [])):
        why.append(f"worlds latched INVALID per build: {meta['invalid_worlds']}")
    if meta.get("periods_done", 0) < scene.periods:
        why.append(f"{meta.get('periods_done', 0)} of {scene.periods} periods ran")
    if scene.graph and not meta.get("use_graph"):
        why.append("ran eagerly: the standard profile captures this scene")
    if scene.repeat > 1 and meta.get("claim") == "RUN_TO_RUN" and len(digests) > 1:
        eq = (digests[0] == digests[1]).all(axis=-1)                   # [periods + 1, worlds]
        if not eq.all():
            p = int(np.flatnonzero(~eq.all(axis=1))[0])
            why.append(f"RUN_TO_RUN builds differ from period {p} in {int((~eq.all(axis=0)).sum())} worlds")
    mem = meta.get("mem_high_bytes", 0) / 2**30
    if mem > mem_cap_gib:
        why.append(f"mempool high-water {mem:.2f} GiB > {mem_cap_gib} GiB")
    return why


def run_scene(name: str, out: Path) -> dict:
    from alloy.tools import coupled_det_probe as P

    s = SCENES[name]
    npz = out / "probe.npz"
    meta = asdict(P.run(P.ProbeArgs(s.case, s.worlds, s.periods, str(npz), repeat=s.repeat, device=DEVICE)))
    digests = np.load(npz, allow_pickle=False)["digests"]
    why = judge(meta, digests, s)
    return {"scene": name, **asdict(s), "verdict": "fail" if why else "pass", "why": why, "claim": meta["claim"],
            "use_graph": meta["use_graph"], "build_s": round(meta["build_s"], 1),
            "mem_high_gib": round(meta["mem_high_bytes"] / 2**30, 3), "invalid_worlds": meta["invalid_worlds"],
            "solvers": meta["solvers"]}


def run_film(out: Path) -> dict:
    from alloy.solvers.film import laws
    from alloy.solvers.film.materials import HDPE_15UM

    r = laws.hang(HDPE_15UM, device=DEVICE, seconds=0.3)
    vals = {k: float(v) for k, v in r.items() if isinstance(v, (int, float))}
    why = [f"{k} not finite" for k, v in vals.items() if not math.isfinite(v)]
    if vals.get("step_gain_max_rel", 0.0) >= 1e-2:
        why.append(f"step_gain_max_rel {vals['step_gain_max_rel']:.3g} >= 1e-2")
    return {"scene": FILM, "verdict": "fail" if why else "pass", "why": why, "values": vals}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="python -m alloy.tools.gpu_smoke")
    p.add_argument("--scene", choices=[*SCENES, FILM])
    p.add_argument("--out")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    if a.list or not a.scene:
        for n, s in SCENES.items():
            print(f"{n:16s} {s.case:24s} {s.worlds:3d} worlds {s.periods:3d} periods x{s.repeat}")
        print(f"{FILM:16s} film hang law, 0.3 s")
        return 0
    out = Path(a.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    try:
        row = run_film(out) if a.scene == FILM else run_scene(a.scene, out)
    except Exception as e:  # noqa: BLE001 - any failure of the scene is the smoke's finding, written as a row
        row = {"scene": a.scene, "verdict": "fail", "why": [f"{type(e).__name__}: {e}"[:400]]}
    (out / "smoke.json").write_text(json.dumps(row, indent=1) + "\n")
    print(json.dumps(row))
    return 0 if row["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
