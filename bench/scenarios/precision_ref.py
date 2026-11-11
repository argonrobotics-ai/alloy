# SPDX-License-Identifier: Apache-2.0
"""Offline CPU MuJoCo 3.16 (float64) reference rows for the precision family (never in the simulation loop).

    python -m bench.scenarios.precision_ref p04      # -> bench/results/precision/P04_hertz_sphere.ref.jsonl
    python -m bench.scenarios.precision_ref p05      # -> bench/results/precision/P05_thin_wall_tunnel.ref.jsonl

Each reference rebuilds the scenario as MJCF with the contact parameters Newton's SolverMuJoCo writes for the same
shapes (convert_solref(ke, kd, 1, 1) -> solref (2 / kd, kd / 2 sqrt(1 / ke)), default solimp, newton solver,
elliptic cone, implicitfast, 15 / 100 iterations) and runs one CPU MuJoCo model per condition. Contacts come from
MuJoCo's own collision (margin 0), so the rows separate float32 MuJoCo Warp on Newton contacts from the
float64 constraint law itself. Laws and metrics are the scenario's own (HertzSphere / ThinWallTunnel).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from bench.harness import REPO, SCHEMA, versions
from bench.scenarios.precision import HertzSphere, ThinWallTunnel

OPTION = ('<option timestep="{dt}" integrator="implicitfast" cone="elliptic" solver="Newton" iterations="15" '
          'ls_iterations="100" impratio="1" gravity="{g}"/>')


def solref(ke: float, kd: float) -> str:
    """Newton's legacy ke/kd -> MuJoCo solref mapping (newton@ac7f7830 solvers/mujoco/kernels.py:190-204)."""
    return f"{2.0 / kd:.10g} {kd / 2.0 * math.sqrt(1.0 / ke):.10g}"


def _ease(x: float) -> float:
    y = min(max(x, 0.0), 1.0)
    return y * y * (3.0 - 2.0 * y)


def _row(scenario: str, nworld: int, dt: float, horizon: float, steps: int, wall: float, acc, unit: str,
         metrics: dict, cmd: str) -> dict:
    """One record in the bench schema. steps = steps per condition; wall = total CPU seconds over all
    nworld conditions (run one after another, single thread)."""
    import mujoco

    return {"schema": SCHEMA, "scenario": scenario, "family": "precision", "config": "ref_mujoco_cpu",
            "solver": "ref:mujoco_cpu", "params": {"engine": f"mujoco {mujoco.__version__} CPU float64"},
            "device": "cpu", "device_name": "cpu", "nworld": nworld, "dt": dt, "substeps": 1, "horizon_s": horizon,
            "steps": steps, "cuda_graph": False, "status": "ok", "error": None,
            "ms_per_step": wall / (nworld * steps) * 1e3 if steps else None,  # per world-step (dt), one CPU thread
            "world_steps_per_s": nworld * steps / wall if wall else None,
            "single_world_realtime_factor": None, "stable": True, "complete": True,
            "penetration_p99_mm": None, "accuracy_error": acc, "accuracy_unit": unit,
            "metrics": metrics, "notes": "offline reference (CPU MuJoCo, one model per condition)",
            "versions": versions(), "command": cmd, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# ---- P04 Hertz sphere ---------------------------------------------------------------------------------------
def p04_xml(hs: HertzSphere, z0: float) -> str:
    sr = solref(hs.KE, hs.KD)
    c = f'friction="0.3 0 0" condim="3" solref="{sr}" solimp="0.9 0.95 0.001 0.5 2"'
    return (f'<mujoco>{OPTION.format(dt=5e-5, g="0 0 -9.81")}<worldbody>'
            f'<geom type="box" size="0.03 0.03 0.005" pos="0 0 -0.005" {c}/>'
            f'<body name="ball" pos="0 0 {z0!r}"><freejoint/>'
            f'<geom type="sphere" size="{hs.R!r}" density="{hs.RHO!r}" {c}/></body></worldbody></mujoco>')


def p04_condition(hs: HertzSphere, cond: int) -> dict:
    """Run one condition; returns trace rows (t, delta, force, vz) every 10 steps plus first-contact events."""
    import mujoco

    gap = 0.0 if cond == 0 else hs.GAP0
    m = mujoco.MjModel.from_xml_string(p04_xml(hs, hs.R + gap))
    d = mujoco.MjData(m)
    if cond:
        d.qvel[2] = -hs.SPEEDS[cond - 1]
    h, mass, g = m.opt.timestep, float(m.body_mass[1]), 9.81
    steps = int(round(hs.horizon_s / h))
    rows, t_on, t_off, v_off, d_max = [], None, None, None, 0.0
    t0 = time.perf_counter()
    for i in range(steps):
        f = hs.F_MAX * _ease(i * h / hs.T_RAMP) if cond == 0 else 0.0
        d.xfrc_applied[1, 2] = -f
        mujoco.mj_step(m, d)
        t, delta = (i + 1) * h, hs.R - d.qpos[2]
        if i % 10 == 0:
            rows.append((t, delta, f + mass * g, d.qvel[2]))
        if t_off is None:
            if delta > 0:
                t_on = t if t_on is None else t_on
                d_max = max(d_max, delta)
            elif t_on is not None:
                t_off, v_off = t, d.qvel[2]
    return {"rows": np.array(rows), "t_on": t_on, "t_off": t_off, "v_off": v_off, "d_max": d_max, "mass": mass,
            "steps": steps, "wall": time.perf_counter() - t0}


def p04_reference() -> dict:
    hs = HertzSphere()
    runs = [p04_condition(hs, c) for c in range(1 + len(hs.SPEEDS))]
    r0, mass = runs[0]["rows"], runs[0]["mass"]
    delta, force, vz = r0[:, 1], r0[:, 2], np.abs(r0[:, 3])
    sel = (delta > 0.005 * hs.R) & (delta < 0.05 * hs.R) & (vz < 0.01)
    n, logk = np.polyfit(np.log(delta[sel]), np.log(force[sel]), 1)
    law_err = float(np.max(np.abs(force[sel] - hs._mujoco_force(mass, delta[sel])) / hs._mujoco_force(mass, delta[sel])) * 100)
    per, errs = [], [law_err]
    for c, v in enumerate(hs.SPEEDS, start=1):
        r = runs[c]
        tc = None if r["t_off"] is None else r["t_off"] - r["t_on"]
        tc_law, e_law = hs._mujoco_impact(r["d_max"])
        row = {"v": v, "tc_ms": None if tc is None else tc * 1e3, "tc_law_ms": tc_law * 1e3,
               "delta_max_mm": r["d_max"] * 1e3, "e_meas": None if r["v_off"] is None else r["v_off"] / v,
               "e_law": e_law}
        if tc is not None:
            row["tc_err_pct"] = abs(tc - tc_law) / tc_law * 100
            errs.append(row["tc_err_pct"])
        per.append(row)
    met = {"law_kind": "mujoco_soft_penalty", "exponent": float(n), "prefactor": float(math.exp(logk)),
           "penalty_law_err_pct": law_err, "fit_points": int(sel.sum()), "delta_max_mm": float(delta.max() * 1e3),
           "force_end_n": float(force[-1]), "hertz_exponent_pass": bool(abs(n - 1.5) <= 0.05),
           "per_condition": per, "conditions_covered": True, "law_pass": bool(max(errs) <= 10.0)}
    wall = sum(r["wall"] for r in runs)
    return _row(hs.id, len(runs), 5e-5, hs.horizon_s, runs[0]["steps"], wall, float(max(errs)), "%",
                met, "python -m bench.scenarios.precision_ref p04")


# ---- P05 thin-wall tunnelling --------------------------------------------------------------------------------
def p05_xml(tw: ThinWallTunnel, thick: float, h: float) -> str:
    sr = solref(1.0e7, 1.0e4)
    c = f'friction="0.3 0 0" condim="3" solref="{sr}" solimp="0.9 0.95 0.001 0.5 2" density="7800"'
    x0 = -(thick / 2 + tw.GAP0 + tw.R + tw.HH)
    return (f'<mujoco>{OPTION.format(dt=h, g="0 0 0")}<worldbody>'
            f'<body name="wall" pos="0 0 0"><freejoint/><geom type="box" size="{thick / 2!r} 0.05 0.05" {c}/></body>'
            f'<body name="capsule" pos="{x0!r} 0 0"><freejoint/>'
            f'<geom type="capsule" size="{tw.R!r} {tw.HH!r}" euler="0 90 0" {c}/></body>'
            f'</worldbody></mujoco>')


def p05_condition(tw: ThinWallTunnel, thick: float, v: float, h: float) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_string(p05_xml(tw, thick, h))
    d = mujoco.MjData(m)
    d.qvel[6] = v  # capsule free joint: qvel[6:9] linear
    steps = int(round(tw.horizon_s / h))
    tunnel, max_pen, reach = False, 0.0, tw.R + tw.HH
    t0 = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(m, d)
        xc, xw = d.qpos[7], d.qpos[0]
        if xc > xw:
            tunnel = True
        elif not tunnel:
            max_pen = max(max_pen, xc + reach - (xw - thick / 2))
    m1, m2 = float(m.body_mass[2]), float(m.body_mass[1])
    v1, v2 = float(d.qvel[6]), float(d.qvel[0])
    sep = v2 - v1
    v2_law = (1 + tw.E) * m1 * v / (m1 + m2)
    return {"thick_mm": thick * 1e3, "v": v, "tunnel": tunnel, "hit": bool(v2 > 1e-6 or tunnel),
            "e_meas": sep / v, "rebound_err_pct": abs(sep - tw.E * v) / (tw.E * v) * 100,
            "momentum_err_pct": abs(m1 * v1 + m2 * v2 - m1 * v) / (m1 * v) * 100,
            "wall_vel_err_pct": abs(v2 - v2_law) / v2_law * 100, "max_pen_mm": max_pen * 1e3,
            "steps": steps, "wall": time.perf_counter() - t0}


def p05_reference() -> dict:
    tw = ThinWallTunnel()
    h = (1 / 60) / 20
    per = [p05_condition(tw, t, v, h) for t in tw.THICK for v in tw.SPEEDS]
    ok = [p for p in per if p["hit"] and not p["tunnel"]]
    worst = max(p["rebound_err_pct"] for p in ok) if ok else None
    wall = sum(p.pop("wall") for p in per)
    steps = per[0].pop("steps")
    for p in per[1:]:
        p.pop("steps")
    met = {"tunnel_rate": float(np.mean([p["tunnel"] for p in per])), "max_pen_mm": max(p["max_pen_mm"] for p in per),
           "rebound_err_pct": worst, "momentum_err_pct": max(p["momentum_err_pct"] for p in ok) if ok else None,
           "wall_vel_err_pct": max(p["wall_vel_err_pct"] for p in ok) if ok else None, "e_configured": tw.E,
           "conditions_covered": True, "per_condition": per,
           "law_pass": bool(not any(p["tunnel"] for p in per) and worst is not None and worst <= 5.0)}
    return _row(tw.id, len(per), h, tw.horizon_s, steps, wall, worst, "%", met,
                "python -m bench.scenarios.precision_ref p05")


def write(row: dict) -> Path:
    path = REPO / "bench" / "results" / "precision" / f"{row['scenario']}.ref.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return path


if __name__ == "__main__":
    jobs = {"p04": p04_reference, "p05": p05_reference}
    keys = sys.argv[1:] or list(jobs)
    if any(k not in jobs for k in keys):
        sys.exit("usage: python -m bench.scenarios.precision_ref [p04] [p05]")
    for k in keys:
        r = jobs[k]()
        print(write(r), json.dumps({"acc": r["accuracy_error"], **{x: r["metrics"].get(x) for x in
                                    ("exponent", "penalty_law_err_pct", "delta_max_mm", "tunnel_rate",
                                     "max_pen_mm", "rebound_err_pct", "law_pass")}}))
