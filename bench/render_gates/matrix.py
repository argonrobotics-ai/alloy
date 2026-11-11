# SPDX-License-Identifier: Apache-2.0
"""The render-gate matrix: one verdict per scene x size x GPU, from per-GPU score reports and speed rows.

The spec (``matrix.json`` next to this file) names the scenes and frames, sizes, GPUs, the raw engine (parity and
equal-spp), the end-to-end engine with the setting it claims per size (G1), and the speed bars.

A cell PASSES when parity, equal-spp and G1 pass and every gated speed bar for it passes; it FAILS when any of them
fails or the engine could not render the scene at that size; otherwise it is PENDING (missing images, rows or
references).

Example:
    >>> spec = load_spec()
    >>> sorted(spec["gpus"])
    ['l4', 'rtx_pro_6000']
    >>> build_matrix(spec, {}, [])["cells"][0]["status"]
    'PENDING'
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from .checks import check_speed

SPEC_PATH = Path(__file__).with_name("matrix.json")
STATES = ("PASS", "FAIL", "PENDING")


def load_spec(path: Path | None = None) -> dict:
    """Read and validate the matrix spec; errors name the missing key.

    Example:
        >>> load_spec()["sizes"]
        ['128x128', '640x480', '1280x720']
    """
    spec = json.loads((path or SPEC_PATH).read_text())
    for key in ("scenes", "sizes", "gpus", "engines", "claims", "speed"):
        if key not in spec:
            raise ValueError(f"matrix spec {path or SPEC_PATH} lacks '{key}'; see bench/render_gates/README.md")
    return spec


def _status(check: Mapping | None) -> str:
    if not check:
        return "PENDING"
    s = check.get("status")
    return s if s in ("PASS", "FAIL") else "PENDING"


def claim_for(spec: Mapping, scene: str, res: str) -> Mapping | None:
    """The setting an engine claims for one scene and size: ``scenes.<scene>.claims.<res>`` first (the fastest setting
    that passes every gate, per scene), else the size default ``claims.<res>``.

    Example:
        >>> claim_for(load_spec(), "t12", "128x128")["spp"]
        64
        >>> claim_for(load_spec(), "c3", "128x128")["spp"]
        5
    """
    return (spec["scenes"].get(scene, {}).get("claims") or {}).get(res) or spec["claims"].get(res)


def _g1(cell: Mapping, engine: str, claim: Mapping | None) -> dict:
    if not claim:
        return {"status": "PENDING", "reason": "no claimed setting for this size in matrix.json"}
    key = f"{claim['spp']}spp_{claim.get('den', 'on')}"
    rec = cell.get(engine, {}).get("g1", {}).get(key)
    if not rec:
        return {"status": "PENDING", "reason": f"no {engine} images at {key}"}
    return {"status": rec["status"], "setting": key, "pass": rec["pass"], "total": rec["total"],
            "worst_ratio": rec["worst_ratio"], "per_frame": rec["per_frame"]}


def _speed(spec: Mapping, rows: list[Mapping], scene: str, res: str, gpu: str) -> list[dict]:
    """Speed bars for one cell; rows count only at the cell's claimed spp (a row from an older claim never does)."""
    claim = claim_for(spec, scene, res)
    out = []
    for bar in spec["speed"]:
        if bar.get("gpu", gpu) != gpu or bar.get("res", res) != res or scene not in bar.get("scenes", [scene]):
            continue
        match = {"gpu": gpu, "scene": scene, "res": res, **bar.get("match", {}), **({"spp": claim["spp"]} if claim else {})}
        out.append({"bar": bar.get("name", ""), "gated": "min_images_per_s" in bar,
                    **check_speed(rows, {**bar, "match": match})})
    return out


def render_check(report: Mapping | None, scene: str, res: str) -> dict:
    """FAIL when the engine's render command for this scene and size failed (``report["meta"]["renders"]`` rows with
    ``rc != 0``): a scene the camera cannot render fails the gate, with the last error line as the reason.

    Example:
        >>> render_check({"meta": {"renders": [{"scene": "c2", "res": "128x128", "rc": 1, "error": "no glass"}]}},
        ...              "c2", "128x128")["status"]
        'FAIL'
    """
    rows = [r for r in ((report or {}).get("meta") or {}).get("renders", []) if r.get("scene") == scene
            and r.get("res") in (res, None, "*")]
    bad = [r for r in rows if r.get("rc")]
    if bad:
        err = bad[0].get("error") or (bad[0].get("stderr_tail") or "").strip().splitlines()[-1:] or ["render failed"]
        return {"status": "FAIL", "reason": err if isinstance(err, str) else err[0]}
    return {"status": "PASS" if rows else "n/a"}


def cell_verdict(spec: Mapping, report: Mapping | None, rows: list[Mapping], scene: str, res: str, gpu: str) -> dict:
    """Verdict for one scene, size and GPU (see module docstring)."""
    render = render_check(report, scene, res)
    cell = ((report or {}).get("cells") or {}).get(f"{scene}/{res}", {})
    raw, e2e = spec["engines"]["raw"], spec["engines"]["e2e"]
    parity = cell.get(raw, {}).get("parity")
    eq = cell.get(raw, {}).get("equal_spp")
    g1 = _g1(cell, e2e, claim_for(spec, scene, res))
    speed = _speed(spec, rows, scene, res, gpu)
    gated = [_status(parity), _status(eq), g1["status"]] + [s["status"] for s in speed if s["gated"]]
    if render["status"] == "FAIL":
        gated.append("FAIL")
    status = "FAIL" if "FAIL" in gated else "PASS" if all(s == "PASS" for s in gated) else "PENDING"
    return {"scene": scene, "res": res, "gpu": gpu, "status": status, "render": render,
            "parity": parity or {"status": "PENDING"},
            "equal_spp": eq or {"status": "PENDING"}, "g1": g1, "speed": speed}


def build_matrix(spec: Mapping, reports: Mapping[str, Mapping], rows: Iterable[Mapping]) -> dict:
    """All cells: ``reports`` maps GPU label to a score report; ``rows`` are speed rows (each with gpu, scene, res).

    Example:
        >>> m = build_matrix(load_spec(), {}, [])
        >>> len(m["cells"]) == len(m["scenes"]) * len(m["sizes"]) * len(m["gpus"])
        True
    """
    rows = list(rows)
    cells = [cell_verdict(spec, reports.get(gpu), rows, scene, res, gpu)
             for scene in spec["scenes"] for res in spec["sizes"] for gpu in spec["gpus"]]
    counts = {s: sum(c["status"] == s for c in cells) for s in STATES}
    return {"schema": "alloy.render_gates.matrix/1", "scenes": list(spec["scenes"]), "sizes": list(spec["sizes"]),
            "gpus": list(spec["gpus"]), "engines": dict(spec["engines"]), "counts": counts, "cells": cells}


def _cell_text(c: Mapping) -> str:
    if c["status"] == "PASS":
        return "PASS"
    bits = [f"render: {c['render']['reason'][:60]}"] if c.get("render", {}).get("status") == "FAIL" else []
    for name in ("parity", "equal_spp", "g1"):
        chk = c[name]
        if chk.get("status") == "FAIL":
            worst = chk.get("worst_ratio")
            bits.append(f"{name} {worst:.2f}" if isinstance(worst, (int, float)) else name)
    bits += [f"speed {s.get('value', 0):.0f}/s" for s in c["speed"] if s["status"] == "FAIL"]
    return f"{c['status']}" + (f" ({', '.join(bits)})" if bits else "")


def to_markdown(matrix: Mapping, spec: Mapping) -> str:
    """Scene x size rows, one column per GPU; failing checks show their worst ratio to the limit.

    Example:
        >>> print(to_markdown(build_matrix(load_spec(), {}, []), load_spec()).splitlines()[0])
        | Scene | Size | NVIDIA L4 | NVIDIA RTX PRO 6000 |
    """
    gpus = matrix["gpus"]
    lines = ["| Scene | Size | " + " | ".join(spec["gpus"][g] for g in gpus) + " |",
             "|---|---|" + "---|" * len(gpus)]
    by = {(c["scene"], c["res"], c["gpu"]): c for c in matrix["cells"]}
    for scene in matrix["scenes"]:
        for res in matrix["sizes"]:
            cols = [_cell_text(by[(scene, res, g)]) for g in gpus]
            lines.append(f"| {spec['scenes'][scene]['title']} | {res} | " + " | ".join(cols) + " |")
    return "\n".join(lines) + "\n"
