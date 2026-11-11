# SPDX-License-Identifier: Apache-2.0
"""task pad_finish: Gym-Anything's pad-finish, a surface-flow concrete proof.

"The pad was just struck off. Finish it with tooled edges and a smooth, hard steel-trowel finish." (``task.py:26``).
Their start is the raw strike-off at 5 min (MASCO-scaled; ``concrete_pad_start.pad_fields("pad-finish")``, the same as
pad-darby's). Success is their 14 pad-finish criteria and progress-v3 (``pad_finish_metrics``, T:398-596) on the whole
pad at their 2 mm cells: flat (3 mm plane and straightedge), stone down and smooth-hard on the top, the 9.5 mm arris
and the lip round every form, no blister, delamination risk, trowel burn, worked bleed water or dusting, the tools
back.

The scripted finisher (a privileged carriage, no robot, as their fixtures), stage by stage (``pad_finish_stages``),
each its owner's code on its own simulation, chained at rest points (``pad_finish_stage``, ``pad_finish_handoff``):
1. **darby** at 5 min (WP6): tub fill of the low, the darby's passes, the end strokes; the pad left as a plane
   ``ref`` (1.5 mm under the form tops);
2. their **wait** until the edger's time (their pad-edge starts at 60 min, ``AGES_21C``) and no bleed sheen stands in
   the edge band;
3. **edge** (WP4): the edger round the four forms, on the plane;
4. **margin** (``pad_finish_margin``): the margin trowel cuts the edger's bead (its spill, 2-4 mm high 79-98 mm in,
   corner piles to 12.5 mm) down to the plane while it is soft, emptying its carry in the tub; left there, the steel
   trowel's ride rule fails on it at 165 min (187 N against their 20 N hold) and its roll carries it into the rim;
5. their **wait** until the steel's time (their pad-trowel starts at 165 min), the bleed has ended and no sheen stands
   (the trowel's sealing then traps nothing; initial set is at about 4 h, so no burn);
6. **steel trowel** (WP4): the lane and cross passes at levels from the plane, under their ride rule;
7. every tool back on its cart.
Each wait is a stage of its own (``pad_finish_stages.wait``): the trowel's chart, the tool parked, the rest watch.
No mag float: no criterion needs it (disclosed; a float stage is the fallback if D or E fall short).

Records the layer cannot be started with are folded on the host across stages (late and wet_worked ORed, worked
summed); the books are summed per world: the paste (``paste_books``: start + supplied - the film's outs - off the pad =
end) and the bleed water (``water_books``: start + bled = end + evaporated + worked in).

Alloy's choices, disclosed:
- the stages are separate simulations, the chart re-split at each hand-off by the next stage's rule (the self-chain
  test, ``test_pad_finish``, is the evidence that a hand-off at rest is a pause);
- the edger's and the trowel's heights are from the darby's plane, not the form-top plane; the trowel works at the
  plane without planing it (``pad_finish_stages.FLOOR`` 0) and its rim is 10 mm, not WP4's 30 mm (``RIM``);
- the ride rule's 20 N hold is WP4's assumed push (``pad_trowel.HOLD``), not GA's (their hold is the actual push);
- the margin pass is Alloy's policy with their tools (GA scripts none); its plan lays the most a loaded knife can;
- each wait runs at the end of the stage before it, on its chart, in 60 s commands (``WAIT_CHUNK``), each followed by
  0.2 s of full ticks; GA asks for 0.5 s watched at rest (REST_WINDOW_S), Alloy's settle ends at the first still
  period, then the 0.2 s watch;
- a film counts as at rest under the rest bound or when its own last step held every face
  (``pad_finish_stage.film_rest``); their relax during a wait is deferred to the watch after the command, which steps
  a moving film on until it rests (the row counts these, ``film.stepped_on_checks``, and the film they moved,
  ``film.stepped_on_moved_m3``);
- the layer's blisters lag theirs by under 1e-5 m (their skip, ``paste.py:208``);
- the darby's hand-off depends on its film's speed cap (WP6's darby sets none: surface flow's default 1.5 m/s; faces
  on the paste still sit at the cap at 3 m/s late in its program): its straightedge is 0.52 / 0.79 mm at u_cap 1.5 /
  3.0 (seed 0; seeds 0-7 0.17-0.52 mm at 1.5), while its plane, flatness, stone_down and books do not move, and the
  finals are unchanged (straightedge 0.715 mm from the 3.0 darby against 0.70-0.72 mm over the seeds; SCR-WP6-5);
- no robot: a privileged carriage holds the tools (their task interface is robot-only; their carriage is an
  ASSISTED probe fixture, ``probes/common.py:1-4``). So the verdict judges 11 of their 14 criteria (audit AB-gym-7;
  ``concrete_ports.PAD_FINISH``): ``forms_undisturbed`` (the forms are cells: no force is measured) and
  ``robot_off_pad`` are not judged (``passed`` None, never a pass), and ``tools_back`` is judged in part (the
  stages' four tools by a point test at their rests; theirs: five tools on a shelf, let go and resting). Every world
  starts at one temperature and bleed (theirs vary by seed).

Success per world is every judged criterion of their verdict and Alloy's own gates (not their criteria, stricter;
``alloy_gates`` in the verdict): the ride rule of the margin and the trowel, the books closed (paste within
``PASTE_TOL``, bleed water within ``WATER_TOL``, relative, atol 0), the material clock within their 5 h (advisory in
theirs), and every stage's hand-off at rest (``gates``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_verdict as CV
from alloy.tasks import pad_finish_handoff as PH
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_finish_metrics as PFM
from alloy.tasks import pad_finish_stages as SG
from alloy.tasks import pad_run as PR
from alloy.tasks.concrete_metrics import PadScalars, PasteGrid
from alloy.tasks.concrete_pad_start import AGES_21C, FORM_H, pad_fields
from alloy.tasks.concrete_ports import FINISH_GATES, PAD_FINISH

log = logging.getLogger("alloy.tasks.pad_finish")
TASK = "pad-finish"
LIMIT_S = 5 * 3600.0                       # s: their time limit, robot time plus waits (T:49)
EDGE_BAND = 0.080                          # m: the edger's band (pad_edge.edge_wet)
TOOLS = 4                                  # darby, margin trowel, edger, steel trowel
PASTE_TOL = 1e-9                           # the paste books' residual, relative to the start's paste (DC5 4)
WATER_TOL = 1e-12                          # the water books' residual, relative to what bled (L4 runs: 1.3e-14)


@dataclass
class FinishResult:
    verdicts: list[dict]
    success: int
    worlds: int
    plane_mm: float                        # the darby's end plane (median core surface, relative to the form tops)
    material_s: list[float]                # per world: material clock used, end age - start age (their limit 5 h)
    stages: dict[str, dict] = field(default_factory=dict)   # per stage: seconds stepped / waited, rest, tools off, diag
    books: dict[str, list[float]] = field(default_factory=dict)
    rest_errors: dict[str, str] = field(default_factory=dict)
    within_limit: list[bool] = field(default_factory=list)   # per world: material_s within limit_s (GA T:852-859)
    limit_s: float = LIMIT_S
    flags: list[str] = field(default_factory=list)         # the film's over-depth cells and capped faces, per stage
    commit: str = ""
    not_measured: tuple[str, ...] = PR.NOT_MEASURED


def next_fields(fs: list[dict[str, Any]], end: SG.StageEnd) -> list[dict[str, Any]]:
    """The next stage's start per world from a stage's end (``after_stage``: the surface, the states, ``extra``)."""
    return [PH.after_stage(f, end.surface[w], {k: v[w] for k, v in end.states.items()},
                           {k: v[w] for k, v in end.extra.items()}) for w, f in enumerate(fs)]


def end_grid(f: dict[str, Any], end: SG.StageEnd, w: int) -> PasteGrid:
    """The verdict's grid for world ``w`` at the chain's end: the surface from the form-top plane with the blisters'
    lift (theirs b + tb + ts + bl, ``paste.py:108``; the layer keeps bl off the film), the layer's states, the chain's
    records and the lip's mark as C."""
    s, st, rec = end.surface[w], end.states, end.records.fields
    return PasteGrid(h=s - FORM_H + st["bl"][w], t=np.maximum(s - np.asarray(f["b"]), 0.0), cell=float(f["cell"]),
                     paste_mask=f["paste_mask"], d_edge=f["d_edge"], D=st["D"][w], a=st["a"][w], E=st["E"][w],
                     S=st["S"][w], w=st["w"][w], q=st["q"][w], bl=st["bl"][w], dwc=st["dwc"][w],
                     C=np.asarray(f["compact"], float), late=rec["late"][w], wet_worked=rec["wet_worked"][w],
                     worked=rec["worked"][w])


OUTS = ("spilled", "evaporated", "absorbed", "drained_total", "clipped", "dripped", "pooled_total", "to_tools")
# ^ what leaves the film, as surface flow's own ledger counts it (host_api.read "ledger"); plus off_pad (the darby's)


def paste_books(f0: list[dict[str, Any]], end: SG.StageEnd) -> dict[str, np.ndarray]:
    """The whole chain's paste per world [m³] (DC5's books): the pad's paste at the start (over the subgrade and the
    form tops, theirs b + tb + ts) plus what the film was supplied (the tub's gobs), less what left it (``OUTS``) and
    what the darby left off the pad, against the end's; ``residual`` the difference, relative to the start's."""
    b = end.books
    start = np.array([float(((f["h"] - f["b"]) * float(f["cell"]) ** 2).sum()) for f in f0])
    final = np.array([float(((end.surface[w] - f["b"]) * float(f["cell"]) ** 2).sum()) for w, f in enumerate(f0)])
    gone = sum(np.asarray(b.get(k, 0.0), np.float64) for k in OUTS) + np.asarray(b.get("off_pad", 0.0), np.float64)
    supplied = np.asarray(b.get("supplied", 0.0), np.float64)
    return {"start": start, "supplied": supplied * np.ones_like(start), "gone": gone * np.ones_like(start),
            "end": final, "residual": (final - start - supplied + gone) / start}


def water_books(f0: list[dict[str, Any]], end: SG.StageEnd) -> dict[str, np.ndarray]:
    """The whole chain's bleed water per world [m³] (DC5's books): the start's film and trapped water plus what bled,
    against the end's plus what evaporated and what was worked in (the layer's books, ``water_*``: what the states
    took); ``residual`` relative to what bled."""
    b, area = end.books, [float(f["cell"]) ** 2 for f in f0]
    on = [np.asarray(f["paste_mask"], bool) for f in f0]
    start = np.array([float((np.asarray(f["w"], np.float64) + np.asarray(f["q"], np.float64))[m].sum()) * a
                      for f, m, a in zip(f0, on, area, strict=True)])
    final = np.array([float((end.states["w"][w].astype(np.float64) + end.states["q"][w])[m].sum()) * a
                      for w, (m, a) in enumerate(zip(on, area, strict=True))])
    bled, evap, inn = (np.asarray(b.get(f"water_{k}", 0.0), np.float64) for k in ("bled", "evaporated", "worked_in"))
    return {"start": start, "bled": bled * np.ones_like(start), "evaporated": evap * np.ones_like(start),
            "worked_in": inn * np.ones_like(start), "end": final,
            "residual": (final + evap + inn - start - bled) / np.maximum(bled, 1e-30)}


def _stage_info(end: SG.StageEnd) -> dict:
    return {"stepped_s": end.stepped_s, "waited_s": end.waited_s, "rest": end.rest,
            "tools_off": end.tools_off.tolist(), "rest_error": end.rest_error,
            "diag": {k: v for k, v in end.diag.items() if k != "ledger"}}


def _plain(x: Any) -> Any:
    return np.asarray(x).tolist()


def _save(path: str | None, name: str, end: SG.StageEnd) -> None:
    """A stage's end under ``path`` (``<name>.npz``): what the next stage starts from, what came before its wait, and
    its rest summary and diagnostics (``info``, JSON: DC5 N5)."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)
    info = {"rest": end.rest, "diag": {k: v for k, v in end.diag.items() if k not in ("ledger", "ride_log")}}
    np.savez_compressed(os.path.join(path, f"{name}.npz"), surface=end.surface, tools_off=end.tools_off,
                        seconds=np.array([end.stepped_s, end.waited_s]), rest_error=np.array(end.rest_error or ""),
                        info=np.array(json.dumps(info, default=_plain)),
                        **{f"state_{k}": v for k, v in end.states.items()},
                        **{f"record_{k}": v for k, v in end.records.fields.items()},
                        **{f"book_{k}": np.asarray(v, np.float64) for k, v in end.books.items()},
                        **{f"extra_{k}": v for k, v in end.extra.items()},
                        **{f"pre_{k}": v for k, v in end.pre_wait.items()},
                        **{f"startw_{k}": v for k, v in end.start_wait.items()},
                        **({"ride_ok": np.asarray(end.diag["ride_ok"], bool)} if "ride_ok" in end.diag else {}))


def load_stage(path: str, name: str) -> SG.StageEnd:
    """A stage's end saved by ``_save`` (to rerun the stages after it)."""
    z = np.load(os.path.join(path, f"{name}.npz"))

    def part(prefix: str) -> dict[str, np.ndarray]:
        return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
    info = json.loads(str(z["info"])) if "info" in z.files else {"rest": {}, "diag": {}}
    return SG.StageEnd(surface=z["surface"], states=part("state_"), records=PH.Records(part("record_")),
                       books=part("book_"), stepped_s=float(z["seconds"][0]), waited_s=float(z["seconds"][1]),
                       tools_off=z["tools_off"], rest=info["rest"], extra=part("extra_"),
                       rest_error=str(z["rest_error"]) or None, pre_wait=part("pre_"), start_wait=part("startw_"),
                       diag={**info["diag"], "loaded": os.path.join(path, f"{name}.npz"),
                             "tools_total": {"darby": 2, "edge": 1, "margin": 1, "trowel": 1}.get(name, 0),
                             **({"ride_ok": z["ride_ok"].tolist()} if "ride_ok" in z.files else {})})


def _stage(name: str, out: str | None, resume: str | None, go: Any) -> SG.StageEnd:
    """The stage ``name``: loaded from ``resume`` when saved there, else run by ``go()``; saved under ``out``."""
    if resume and os.path.exists(os.path.join(resume, f"{name}.npz")):
        log.info("%s: loaded from %s", name, resume)
        end = load_stage(resume, name)
    else:
        end = go()
    _save(out, name, end)
    return end


def budget(f0: list[dict[str, Any]], fs: list[dict[str, Any]]) -> float:
    """Their 5 h left at a stage's start ``fs``: ``LIMIT_S`` less the material clock used so far, the most of any
    world (their limit is robot time plus waits, T:49; DC5 S1)."""
    return LIMIT_S - max(float(f["age"]) - float(g["age"]) for f, g in zip(fs, f0, strict=True))


def gates(f0: list[dict[str, Any]], ends: dict[str, SG.StageEnd]) -> dict[str, np.ndarray]:
    """Per world, what success needs beside the verdict (DC5): the ride rule of the margin and the trowel; the books
    closed (``|paste residual| <= PASTE_TOL`` and ``|water residual| <= WATER_TOL``, atol 0); the material clock
    (end age - start age, GA's material_clock_s) within their 5 h (``within_limit``, T:852-859)."""
    t, n = ends["trowel"], len(f0)
    ride = np.logical_and.reduce([np.asarray(ends[k].diag.get("ride_ok", [True] * n), bool)
                                  for k in ("margin", "trowel")])
    paste, water = paste_books(f0, t)["residual"], water_books(f0, t)["residual"]
    used = np.array([PH.age_of(t.states["age"][w], f["paste_mask"]) - float(f["age"]) for w, f in enumerate(f0)])
    return {"ride_ok": ride, "books_ok": (np.abs(paste) <= PASTE_TOL) & (np.abs(water) <= WATER_TOL),
            "within_limit": used <= LIMIT_S, "material_s": used}


def world_success(passed: bool, gate: dict[str, bool], errors: dict[str, str]) -> bool:
    """A world succeeds when its verdict's judged criteria pass (``passed``), all its ``gates`` hold and no stage's
    hand-off was refused."""
    return bool(passed and all(gate.values()) and not errors)


def flags(ends: dict[str, SG.StageEnd]) -> list[str]:
    """The film over its step bound's depth (``cells_over_depth``) or at its face cap (``faces_at_cap``), per stage,
    sampled every 25 periods (DC5 N4)."""
    return [f"{k}: {n} {s.diag[n]}" for k, s in ends.items() for n in ("cells_over_depth", "faces_at_cap")
            if s.diag.get(n, 0) > 0]


def _chain(f0: list[dict[str, Any]], temperature: float, device: str, out: str | None, resume: str | None,
           edge_age: float | None) -> tuple[dict[str, SG.StageEnd], list[dict[str, Any]], float]:
    """The stages in order, each waiting within the budget left; their ends, the fields after the last, the plane."""
    band = f0[0]["paste_mask"] & (f0[0]["d_edge"] < EDGE_BAND)
    masco, paste = CM.masco_factor(temperature), f0[0]["paste_mask"]
    edge_ready = SG.all_of(SG.age_at_least(paste, (AGES_21C["pad-edge"] if edge_age is None else edge_age) * masco),
                           SG.sheen_gone(band))
    trowel_ready = SG.all_of(SG.age_at_least(paste, AGES_21C["pad-trowel"] * masco),
                             SG.bleed_over(paste, temperature))
    d = _stage("darby", out, resume, lambda: SG.darby(f0, device=device))
    ref = SG.plane(d.surface, f0)
    log.info("darby done: plane %.3f mm", ref * 1e3)
    f1 = next_fields(f0, d)
    w1 = _stage("wait_edge", out, resume, lambda: SG.wait(f1, ref, d.records, d.books, edge_ready, budget(f0, f1),
                                                          device))
    f1 = next_fields(f1, w1)
    e = _stage("edge", out, resume, lambda: SG.edge(f1, ref, w1.records, w1.books, device=device,
                                                    ready=SG.sheen_gone(MG.band(f0[0])), limit=budget(f0, f1)))
    f2 = next_fields(f1, e)
    m = _stage("margin", out, resume, lambda: SG.margin(f2, ref, e.records, e.books, device=device))
    f2 = next_fields(f2, m)
    w2 = _stage("wait_trowel", out, resume, lambda: SG.wait(f2, ref, m.records, m.books, trowel_ready,
                                                            budget(f0, f2), device))
    f2 = next_fields(f2, w2)
    t = _stage("trowel", out, resume, lambda: SG.trowel(f2, ref, w2.records, w2.books, device=device))
    ends = {"darby": d, "wait_edge": w1, "edge": e, "margin": m, "wait_trowel": w2, "trowel": t}
    return ends, next_fields(f2, t), ref


def run(worlds: int = 1, seed: int = 0, device: str = "cpu", temperature: float = 21.1, air: str = "still",
        out: str | None = None, resume: str | None = None, edge_age: float | None = None) -> FinishResult:
    """The chain on ``worlds`` worlds (world w: seed + w); each stage's end saved under ``out`` when given, and taken
    from ``resume`` when saved there (the stages after it rerun). ``edge_age`` [s at 21.1 C]: when the edger may
    start (their pad-edge's age by default); the trowel starts at their pad-trowel's, the bleed over and no sheen."""
    from alloy.bench.provenance import code_versions  # noqa: PLC0415

    f0 = [pad_fields(TASK, seed + w, temperature, air) for w in range(worlds)]
    ends, f3, ref = _chain(f0, temperature, device, out, resume, edge_age)
    t = ends["trowel"]
    off = sum(s.tools_off for s in ends.values())
    errors = {k: s.rest_error for k, s in ends.items() if s.rest_error}
    g = gates(f0, ends)
    verdicts, success = [], 0
    for w, f in enumerate(f0):
        v = PFM.pad_finish(end_grid(f3[w], t, w), PR.start_grid(f, C=np.asarray(f["compact"], float)), f["x"],
                           f["y"], t.states["age"][w], temperature,
                           PadScalars(tools_off=int(off[w]), tools_total=TOOLS), start_age=float(f["age"]))
        ok = {k: bool(g[k][w]) for k in ("ride_ok", "books_ok", "within_limit")}
        d = PR.verdict_dict(v, PAD_FINISH, gates=CV.alloy_gates(FINISH_GATES, **ok, stages_at_rest=not errors))
        verdicts.append({**d, **ok, "side_radius": v.side_radius, "side_arc_lip": v.side_arc_lip})
        success += int(world_success(CV.judged_pass(d), ok, errors))          # = d["success"]
    books = {**t.books, **{f"paste_{k}": v for k, v in paste_books(f0, t).items()},
             **{f"water_book_{k}": v for k, v in water_books(f0, t).items()}}
    return FinishResult(verdicts=verdicts, success=success, worlds=worlds, plane_mm=ref * 1e3,
                        material_s=g["material_s"].tolist(), stages={k: _stage_info(s) for k, s in ends.items()},
                        books={k: np.asarray(v, float).tolist() for k, v in books.items()}, rest_errors=errors,
                        within_limit=g["within_limit"].tolist(), flags=flags(ends),
                        commit=code_versions()["alloy"])


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    from dataclasses import asdict  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None, help="directory for each stage's end fields (npz)")
    ap.add_argument("--resume", default=None, help="a directory of saved stage ends to start after")
    ap.add_argument("--edge-min", type=float, default=None, help="minutes of age before the edger (their pad-edge's)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = run(args.worlds, args.seed, args.device, out=args.out, resume=args.resume,
              edge_age=None if args.edge_min is None else args.edge_min * 60.0)
    text = json.dumps(asdict(res), indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    print(text)
    return 0 if res.success == res.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
