# SPDX-License-Identifier: Apache-2.0
"""The report of a fit: a text table, the randomization lines to paste, and overlay plots of sim against real.

``text(result)`` works from a ``SysidResult`` or from a saved ``*.sysid.json`` (``from_file``); ``plots`` needs the
result of the ``fit`` call itself (the curves are not kept in the file) and ``pip install matplotlib``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from alloy.sysid.errors import SysidError, need
from alloy.sysid.result import SCHEMA, SysidResult

MIN_SPREAD, MAX_SPREAD = 0.05, 0.5      # suggested randomization: +- this share of the fitted value, at least / at most
SCALES = {"kp": "kp_scale", "kd": "kd_scale", "vel": "kd_scale", "armature": "armature_scale"}


def from_file(path: str | Path) -> dict[str, Any]:
    """The dict of a saved sysid file with its ``fit`` block. Example: ``from_file("my_robot.sysid.json")``."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SysidError(f"cannot read sysid file {str(path)!r} ({exc})", "pass the file written by `python -m "
                         "alloy.sysid fit`", field="sysid") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise SysidError(f"{str(path)!r} is not a sysid file", f"it needs \"schema\": \"{SCHEMA}\"", field="sysid")
    return data


def _fmt(x: Any, width: int = 9) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-".rjust(width)
    return f"{float(x):.4g}".rjust(width)


def text(result: SysidResult | dict[str, Any], robot_name: str = "robot") -> str:
    """The report as text: error before / after per joint, the fitted values against the robot file's, what the
    recording does not determine, and the ``scene.randomize`` lines. Example: ``print(text(result))``."""
    data = result.to_dict() if isinstance(result, SysidResult) else result
    fit, joints = data.get("fit", {}), data["joints"]
    lines = [f"System identification of {data.get('robot') or 'robot'}: {len(joints)} joints, control period "
             f"{1e3 * float(data.get('control_dt', 0.0)):.3g} ms"]
    rec = fit.get("recording")
    if rec:
        lines.append(f"recording: {rec.get('seconds')} s in {rec.get('episodes')} episode(s), sampled every "
                     f"{1e3 * float(rec.get('sample_dt', 0.0)):.3g} ms")
    before, after = fit.get("rms_before_deg", {}), fit.get("rms_after_deg", {})
    fields = [f for f in ("kp", "kd", "damping", "friction", "armature", "torque_limit", "delay")
              if any(f in v for v in joints.values())]
    lines += ["", "Tracking error (RMS; degrees for hinges, metres for slides) and fitted values (file's value in "
              "brackets):", "joint".ljust(18) + "before".rjust(11) + "after".rjust(11) +
              "".join((f if f != "delay" else "delay ms").rjust(20) for f in fields)]
    nominal = fit.get("nominal", {})
    for name, vals in joints.items():
        row = name.ljust(18) + _fmt(before.get(name), 11) + _fmt(after.get(name), 11)
        for f in fields:
            k = 1e3 if f == "delay" else 1.0
            v = None if vals.get(f) is None else k * vals[f]
            nom = nominal.get(name, {}).get(f)
            row += (_fmt(v) + (f" [{_fmt(nom, 0).strip()}]" if f != "delay" else "")).rjust(20)
        lines.append(row)
    lines += ["", *_weak(fit), "", *randomize_lines(data, robot_name)]
    if fit.get("notes"):
        lines += ["", "Notes:"] + [f"- {n}" for n in fit["notes"]]
    return "\n".join(lines)


def _weak(fit: dict[str, Any]) -> list[str]:
    weak = fit.get("poorly_determined", {})
    if not weak:
        return ["Every fitted parameter is determined by this recording (1-sigma under 8 % of its search range)."]
    out = ["POORLY DETERMINED by this recording (the value is a guess inside its bounds; do not rely on it):"]
    for joint, names in weak.items():
        moved = fit.get("moved", {}).get(joint, True)
        why = "the joint never moved" if not moved else "the motion does not separate it from the others"
        out.append(f"- {joint}: {', '.join(_label(n) for n in names)} ({why})")
    out.append("Record more motion of these joints (faster and slower moves, both directions) and fit again.")
    return out


def _label(variable: str) -> str:
    return "kd + damping" if variable == "vel" else variable


def randomize_lines(data: dict[str, Any], robot_name: str = "robot") -> list[str]:
    """``scene.randomize`` lines around the fit, for the fields FEAT-DR's randomization has (servo gains and
    armature, as factors on the fitted values). The spread is 3 sigma of the least certain joint among those the
    recording determines, held within
    +-5 % and +-50 %. Example: ``randomize_lines(result.to_dict(), "panda")``."""
    sigma = data.get("fit", {}).get("sigma", {})
    weak = data.get("fit", {}).get("poorly_determined", {})
    out = ["Suggested randomization around the fit (paste after add_robot; factors on the fitted values):"]
    seen: dict[str, float] = {}
    for variable, per_joint in sigma.items():
        field = SCALES.get(variable)
        if field is None:
            continue
        finite = [s for j, s in per_joint.items()
                  if s is not None and math.isfinite(s) and variable not in weak.get(j, ())]
        spread = float(np.clip(3.0 * max(finite), MIN_SPREAD, MAX_SPREAD)) if finite else MAX_SPREAD
        seen[field] = max(seen.get(field, 0.0), spread)
    if not seen:
        seen = {"kp_scale": 0.1, "kd_scale": 0.1}
    for field, spread in seen.items():
        out.append(f'scene.randomize("{robot_name}", "{field}", {1.0 - spread:.2f}, {1.0 + spread:.2f})')
    out.append("(joint damping, dry friction and delay cannot be randomized per world yet: "
               "docs/guide/randomize_robots.md)")
    return out


def plots(result: SysidResult, folder: str | Path) -> list[Path]:
    """One PNG per joint in ``folder``: measured position, simulated before the fit and after it, over time.

    Example: ``plots(result, "sysid_report")``."""
    if not result.curves:
        raise SysidError("this result has no curves (it was loaded from a file)", "plot from the result of "
                         "alloy.sysid.fit(...), or run `python -m alloy.sysid fit ... --report DIR`", field="result")
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise need("matplotlib", "pip install matplotlib", "overlay plots") from exc
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    c = result.curves
    t = np.arange(len(c["measured"])) * float(c["dt"])
    out = []
    for j, name in enumerate(result.joints):
        k = 180.0 / math.pi if result.hinge is None or result.hinge[j] else 1.0
        unit = "deg" if k != 1.0 else "m"
        fig, (top, low) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, height_ratios=(2, 1))
        top.plot(t, k * c["command"][:, j], color="0.75", lw=1.0, label="commanded")
        top.plot(t, k * c["measured"][:, j], color="k", lw=1.6, label="real (recording)")
        top.plot(t, k * c["before"][:, j], color="tab:red", lw=1.0, ls="--", label="sim, robot file")
        top.plot(t, k * c["after"][:, j], color="tab:blue", lw=1.0, label="sim, fitted")
        top.set_ylabel(f"{name} [{unit}]")
        top.legend(loc="upper right", fontsize=8, ncol=4)
        low.plot(t, k * (c["before"][:, j] - c["measured"][:, j]), color="tab:red", lw=1.0, ls="--")
        low.plot(t, k * (c["after"][:, j] - c["measured"][:, j]), color="tab:blue", lw=1.0)
        low.set_ylabel(f"sim - real [{unit}]")
        low.set_xlabel("time [s]")
        fig.tight_layout()
        path = folder / f"{name.replace('/', '_')}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        out.append(path)
    return out


__all__ = ["from_file", "plots", "randomize_lines", "text"]
