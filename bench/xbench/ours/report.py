# SPDX-License-Identifier: Apache-2.0
"""P3-2 report: metric x engine rows from the results tree (numpy-free; runs anywhere).

    python bench/xbench/ours/report.py [internal note] \\
        --rows bench/xbench/our_tests_rows.json --doc bench/xbench/OUR_TESTS.md

Reads, per engine directory (layout: [internal note] section 5): ``laws/laws_scored.json``,
``laws/fluid_scored.json``, ``laws/<law>.scored.json`` (A15, A8, A9), ``laws/not_supported.json`` ({law: reason}),
``laws/not_run.json`` ({law: reason}), ``laws_alloy/``, ``repeat/``, ``repeat_alloy/``, ``sb7/<row>/summary.json``,
``sb7_alloy/summary.json``, ``settings.json``, ``versions.json``. A follow-up package ``<engine>_followup/`` in the same
layout overlays the engine's: its scored laws, repeat settings and SB-7 rows replace the same ones (its ``not_run``
entries only fill gaps).
Every row: test, engine (``alloy@<vm engine>`` for Alloy beside), row, value, unit, verdict, n, notes, source.
SB-12 is the pass share over the laws the engine supports, with the two totals the brief asks for (not supported
counted as fail; excluded); a law missing from every file is ``not_run`` and never a pass, and a port error / port
mismatch (the engine's run is not the pack's scene) counts as not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score_repeat import shares  # noqa: E402

SB12_LAWS = ("A1.freefall", "A2.pendulum", "A3.pendulum_energy", "A3.contact_energy", "A4.momentum", "A5.onset",
             "A5.slide", "A5.isotropy", "A6.restitution", "D1.rerun", "A10.capstan", "A11.hanging_chain",
             "A11.catenary", "A11.cantilever", "A12.buckling", "A15.neohookean", "A13.hydrostatic",
             "A13.archimedes", "A14.dam_break", "A8.push", "A9.thread_rundown")
SB7_TARGET, SB11_TARGET = 1e-5, 1.0
# XB4-4: T12's scene.xml clamps the arm actuators (joint actuatorfrcrange); t12.json said "no force limit" until
# e3b8b92d3. A package's settings.json "actuator_clamp" decides; else Alloy (its route applies the clamp) and Genesis
# (its MJCF loader composes actuatorfrcrange, utils/mjcf.py:414-479) are "yes", every other port "no".
CLAMPED_PORTS = ("genesis",)


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _row(test, engine, row, value, unit, verdict, n=None, notes="", source="") -> dict:
    return {"test": test, "engine": engine, "row": row, "value": value, "unit": unit, "verdict": verdict, "n": n,
            "notes": notes, "source": str(source)}


def _law_status_one(d: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    per_law = [_json(p) for p in sorted(d.glob("*.scored.json"))]
    for r in (_json(d / "laws_scored.json") or []) + (_json(d / "fluid_scored.json") or []) + [x for x in per_law if x]:
        err = "; ".join(f"{m['name']} {m['error']:.3g} (bound {m['bound']:.3g})" for m in r.get("metrics", []))
        if r.get("guard"):
            err += f"; {r['guard']['name']}: sink {r['guard']['sink_m']:.3g} m (bound {r['guard']['bound_m']:.3g})"
        out[r["law"]] = {"status": r["status"], "detail": err or r.get("error", ""), "source": str(d)}
    for law, why in (_json(d / "not_supported.json") or {}).items():
        if out.get(law, {}).get("status", "not_run") == "not_run":       # an info-only partial run stays unsupported
            out[law] = {"status": "not_supported", "detail": why, "source": str(d)}
    for law, why in (_json(d / "not_run.json") or {}).items():
        out.setdefault(law, {"status": "not_run", "detail": why, "source": str(d)})
    return out


def law_status(*dirs: Path) -> dict[str, dict]:
    """law -> {status, detail, source} from laws_scored.json, fluid_scored.json (A13 / A14), the per-law
    ``<law>.scored.json`` files (A15, A8, A9), not_supported.json and not_run.json. A later directory (a follow-up
    package) replaces the laws it has a verdict for; its not_run entries only fill gaps."""
    out: dict[str, dict] = {}
    for d in dirs:
        for law, v in _law_status_one(d).items():
            if v["status"] != "not_run" or law not in out:
                out[law] = v
    return out


def sb12_rows(engine: str, dirs: list[Path], row: str) -> list[dict]:
    st = law_status(*dirs)
    rows = [_row(f"SB-12:{law}", engine, row, None, "", st.get(law, {}).get("status", "not_run"), None,
                 st.get(law, {}).get("detail", ""), st.get(law, {}).get("source", dirs[0])) for law in SB12_LAWS]
    ok = sum(r["verdict"] == "pass" for r in rows)
    ran = sum(r["verdict"] in ("pass", "fail", "error") for r in rows)
    ns = sum(r["verdict"] == "not_supported" for r in rows)
    nr = sum(r["verdict"] in ("not_run", "port_error", "port_mismatch") for r in rows)   # no engine verdict
    excl = ok / ran if ran else None
    note = f"{ok}/{ran} supported laws pass; {ns} not supported; {nr} not run; counting not supported as fail " \
           f"{ok}/{ran + ns}"
    head = _row("SB-12", engine, row, excl, "share (supported laws)",
                "pass" if ran and ok == ran and not nr else "fail" if ran else "not_run", ran, note,
                ", ".join(map(str, dirs)))
    head.update(ok=ok, not_supported=ns, not_run=nr)
    return [head, *rows]


def sb11_rows(engine: str, dirs: list[Path]) -> list[dict]:
    groups: dict[tuple[str, str], list[Path]] = {}
    for d in dirs:
        mine: dict[tuple[str, str], list[Path]] = {}
        for p in sorted(d.glob("*_p[0-9].json")):
            scene, _, setting = p.stem.rsplit("_p", 1)[0].rpartition("_")
            mine.setdefault((scene, setting), []).append(p)
        groups.update(mine)                                   # a later directory's setting replaces the earlier one
    out = []
    for (scene, setting), files in sorted(groups.items()):
        s = shares([json.loads(f.read_text()) for f in files])
        v = s["sb11_share"]
        out.append(_row(f"SB-11:{scene}", engine, setting, v, "share", "pass" if v == SB11_TARGET else "fail",
                        s["worlds"], f"R1 {s['r1_share']}, R2 {s['r2_share']} ({s['processes']} processes), "
                        f"R4 {s['r4_equal']}, step {s['step_ms']} ms", files[0].parent))
    return out


def sb7_rows(engine: str, dirs: list[Path]) -> list[dict]:
    """One row per SB-7 setting; a later directory's summary replaces the same row (e.g. a completed tuned run)."""
    found: dict[str, Path] = {}
    for d in dirs:
        for summ in sorted(d.glob("*/summary.json")) + ([d / "summary.json"] if (d / "summary.json").exists() else []):
            found[summ.parent.name if summ.parent != d else "alloy"] = summ
    out = []
    for row, summ in found.items():
        s = json.loads(summ.read_text())
        out.append(_row("SB-7:t12", engine, row, s["upper95"], "S1 per episode (95 % upper bound)",
                        "pass" if s["passes"] else "fail", s["episodes"],
                        f"{s['s1']} S1 confirmed ({s['causes']}), {s['s1_raw']} raw ({s['causes_raw']}), rate "
                        f"{s['rate']:.3g}", summ))
        out[-1]["rate"] = s["rate"]
    return out


def clamp_tag(pkg: Path, engine: str) -> str:
    """``actuator_clamp`` of a package's T12 rows: its settings.json says, else the defaults above."""
    said = (_json(pkg / "settings.json") or {}).get("actuator_clamp")
    if said is not None:
        return said
    return "yes" if engine.startswith("alloy@") or engine in CLAMPED_PORTS else "no"


def _t12_rows(engine: str, pkgs: list[Path], beside: bool) -> list[dict]:
    """SB-11 and SB-7 T12 rows of one or more packages (a later one replaces the same row), each tagged."""
    name, rows = f"alloy@{engine}" if beside else engine, []
    for kind, fn in (("repeat", sb11_rows), ("sb7", sb7_rows)):
        for r in fn(name, [d / (f"{kind}_alloy" if beside else kind) for d in pkgs]):
            if r["test"] in ("SB-11:t12", "SB-7:t12"):        # law scenes have no actuators: untagged
                pkg = next(d for d in pkgs if str(r["source"]).startswith(f"{d}/"))
                r["actuator_clamp"] = clamp_tag(pkg, name)
            rows.append(r)
    return rows


def engine_rows(root: Path, engine: str) -> list[dict]:
    """The engine package ``root/<engine>``, overlaid by ``root/<engine>_followup`` when that exists; the clamped
    T12 rerun ``root/<engine>_clamp`` (XB4-4) adds its rows beside the unclamped ones, which stay, labelled."""
    ds = [root / engine] + [d for d in [root / f"{engine}_followup"] if d.is_dir()]
    rows = sb12_rows(engine, [d / "laws" for d in ds], "recommended")
    rows += sb12_rows(f"alloy@{engine}", [d / "laws_alloy" for d in ds], "alloy")
    rows += _t12_rows(engine, ds, False) + _t12_rows(engine, ds, True)
    if (root / f"{engine}_clamp").is_dir():
        rows += _t12_rows(engine, [root / f"{engine}_clamp"], False)
    return rows


def markdown(rows: list[dict]) -> str:
    heads = [r for r in rows if ":" not in r["test"] or r["test"].startswith(("SB-7", "SB-11"))]
    lines = ["| metric | engine | row | actuator_clamp | value | verdict | N | notes |", "|" + "---|" * 8]
    for r in heads:
        v = r["value"] if not isinstance(r["value"], float) else f"{r['value']:.4g}"
        lines.append(f"| {r['test']} | {r['engine']} | {r['row']} | {r.get('actuator_clamp', '-')} | {v} | "
                     f"{r['verdict']} | {r['n']} | {r['notes']} |")
    return "\n".join(lines) + "\n"


SHORT = {"pass": "pass", "fail": "FAIL", "not_supported": "n.s.", "not_run": "not run", "error": "error",
         "port_error": "port err", "port_mismatch": "port err"}


def _cell(rows: list[dict], engine: str, test: str) -> str:
    """Headline cell: every row of one test for one engine, compact. T12 rows without the actuator clamp are not
    compared (XB4-4): the cell names them only as pending a clamped rerun."""
    sel = [r for r in rows if r["engine"] == engine and r["test"] == test]
    if any(r.get("actuator_clamp") == "no" for r in sel):
        sel = [r for r in sel if r.get("actuator_clamp") != "no"]
        if not sel:
            return "unclamped port, not compared (clamped rerun pending)"
    if test == "SB-12":
        return "; ".join(f"{r['ok']}/{r['n']} ({r['not_supported']} n.s., {r['not_run']} not run)" for r in sel) or "-"
    if test.startswith("SB-11"):
        return "; ".join(f"{r['row']} {r['value']:.3g}" for r in sel) or "-"
    return "; ".join(f"{r['row']} {r['rate']:.3g} (up95 {r['value']:.2g}, N {r['n']:,})" for r in sel) or "-"


def headline(rows: list[dict], engines: list[str]) -> str:
    """metric x engine; Alloy beside per VM in the engine's column (``alloy@<engine>``)."""
    tests = (("SB-12", "laws passed / run (supported)", "all pass"), ("SB-11:t12", "T12 repeat identity share", "1.0"),
             ("SB-7:t12", "T12 S1 rate per episode", "up95 <= 1e-5"))
    lines = ["| metric | target | " + " | ".join(f"{e} | alloy@{e}" for e in engines) + " |",
             "|---|---|" + "---|---|" * len(engines)]
    for test, name, target in tests:
        cells = " | ".join(f"{_cell(rows, e, test)} | {_cell(rows, f'alloy@{e}', test)}" for e in engines)
        lines.append(f"| {test} {name} | {target} | {cells} |")
    return "\n".join(lines) + "\n"


def law_matrix(rows: list[dict], engines: list[str]) -> str:
    """law x engine verdicts (engine, then Alloy beside on its VM)."""
    cols = [c for e in engines for c in (e, f"alloy@{e}")]
    got = {(r["test"][6:], r["engine"]): r["verdict"] for r in rows if r["test"].startswith("SB-12:")}
    lines = ["| law | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for law in SB12_LAWS:
        lines.append(f"| {law} | " + " | ".join(SHORT.get(got.get((law, c), "not_run"), "?") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def population_note() -> str:
    """The SB-7 counted-population sentence (audit AB-laws-7), printed with every block that shows a glitch rate."""
    from alloy.bench.sb7 import population_line

    return ("**SB-7 counted population.** " + population_line() + ". The changes are listed in `POPULATION` in "
            "`alloy/bench/sb7.py`.\n")


def fill_doc(doc: Path, blocks: dict[str, str]) -> None:
    """Replace each ``<!-- report.py:<name> -->`` ... ``<!-- /report.py:<name> -->`` block of ``doc`` in place.

    A doc may carry any subset of the blocks, but at least one."""
    text, found = doc.read_text(), 0
    for name, body in blocks.items():
        head, tail = f"<!-- report.py:{name} -->", f"<!-- /report.py:{name} -->"
        i, j = text.find(head), text.find(tail)
        if i < 0 and j < 0:
            continue
        if i < 0 or j < i:
            raise ValueError(f"{doc}: broken {head} ... {tail} block")
        text, found = text[:i + len(head)] + "\n" + body + text[j:], found + 1
    if not found:
        raise ValueError(f"{doc}: no report.py block")
    doc.write_text(text)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--engines", default="isaaclab,genesis,maniskill")
    p.add_argument("--rows", type=Path)
    p.add_argument("--md", type=Path)
    p.add_argument("--doc", type=Path, action="append", default=[],
                   help="fill the report.py:{headline,laws,rows} blocks this markdown file has (repeatable)")
    a = p.parse_args()
    rows = [r for e in a.engines.split(",") for r in engine_rows(a.root, e)]
    if a.rows:
        a.rows.write_text(json.dumps(rows, indent=1))
    engines = a.engines.split(",")
    note = population_note()
    blocks = {"headline": headline(rows, engines) + "\n" + note, "laws": law_matrix(rows, engines),
              "rows": note + "\n" + markdown(rows)}
    text = "\n".join(blocks.values())
    for doc in a.doc:
        fill_doc(doc, blocks)
    if a.md:
        a.md.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
