# SPDX-License-Identifier: Apache-2.0
"""Speed-ladder bench family (SPEC §3.9, parallel §4.9): rungs L0-L4 for the baseline harness.

Each rung is a harness ``Scenario`` whose configs sweep its parameter (objects, links); the harness sweeps worlds.
One harness step is one solver substep (``Config.dt`` = the grid dt 1.25 ms, so harness speed fields keep their
meaning); the scene, drive and gate come from ``alloy.parallel.rungs.RawRung`` (raw Newton + SolverMuJoCo, the PAR-1
baseline). Extra fields per record go in ``metrics``: ``t_floor_s, c_w_s, n_knee`` (running fit over the rungs run so
far in this config, per control period), ``kernels_per_substep`` (CUDA), ``iters_p50, iters_p99``,
``bytes_per_world``, ``usd_per_million_sim_s`` (``ALLOY_VM_USD_PER_H`` env, the VM's list spot price), and
``quality_gate`` (no gate, no result). L3 (task traces, WP13) and the SceneCard ``alloy_*`` configs (WP1/WP2) are
added when their inputs land.

    python -m bench.run --family ladder --device cuda:0 --ladder 1,16,256,1024,4096
"""

from __future__ import annotations

import os

import numpy as np

from alloy.parallel.costmodel import fit_cost_model
from alloy.parallel.counters import ProbeSample, kernels_per_call
from alloy.parallel.rungs import RawConfig, RawRung
from bench.harness import Built, Config, Scenario

GRID_DT = RawConfig().dt  # 1.25 ms = control_dt / 16 (D4)
LADDER_WORLDS = (1, 4, 16, 64, 256, 1024, 4096)


def _configs(scene: str, key: str, values: tuple[int, ...], caps: dict[int, int] | None = None) -> tuple[Config, ...]:
    caps = caps or {}
    return tuple(Config(name=f"raw_mujoco_{key[0]}{v}", solver="mujoco", dt=GRID_DT, substeps=1,
                        params={"scene": scene, key: v}, max_nworld=caps.get(v)) for v in values)


class _LadderScenario(Scenario):
    family = "ladder"
    horizon_s = 2.0
    ladder = LADDER_WORLDS
    accuracy_unit = "gate"
    sample_every = 40  # substeps between counter samples (2 control periods)
    warmup_steps = 32
    scene = ""
    param_key = ""

    def __init__(self) -> None:
        self._points: dict[str, list[tuple[int, float]]] = {}

    def build(self, nworld: int, device: str, config: Config) -> Built:
        rung = RawRung(self.scene, int(config.params.get(self.param_key, 0)), nworld, device, RawConfig())
        return Built(model=rung.model, solver=rung.solver, state_0=rung.s0, state_1=rung.s1, control=rung.control,
                     nworld=nworld, extra={"rung": rung, "samples": []})

    def step(self, b: Built, config: Config) -> None:
        b.extra["rung"].substep()

    def state_arrays(self, b: Built) -> dict[str, np.ndarray]:
        rung = b.extra["rung"]
        return {"body_q": rung.s0.body_q.numpy(), "body_qd": rung.s0.body_qd.numpy()}

    def measure(self, b: Built, t: float, rec: dict) -> None:
        b.extra["samples"].append(b.extra["rung"].counters())

    def finish(self, b: Built, rec: dict) -> None:
        rung: RawRung = b.extra["rung"]
        samples: list[ProbeSample] = b.extra["samples"]
        gate = rung.gate()
        m = rec["metrics"]
        m.update(quality_gate=gate, iters_p50=_p(samples, "iters", 50), iters_p99=_p(samples, "iters", 99),
                 contacts_p99=_p(samples, "contacts", 99), rows_p99=_p(samples, "rows", 99),
                 overflow=max((s.overflow for s in samples), default=0),
                 kernels_per_substep=kernels_per_call(rung.substep, rung.device),
                 bytes_per_world=None if rec["gpu_mem_mb_per_world"] is None else rec["gpu_mem_mb_per_world"] * 2**20)
        rec["accuracy_error"] = 0.0 if gate["pass"] else 1.0
        usd = os.environ.get("ALLOY_VM_USD_PER_H")
        if usd and rec["sim_seconds_per_gpu_second"]:
            m["usd_per_million_sim_s"] = float(usd) / (3600.0 * rec["sim_seconds_per_gpu_second"]) * 1e6
        self._running_fit(rec, gate["pass"])

    def _running_fit(self, rec: dict, passed: bool) -> None:
        if not passed or rec["ms_per_step"] is None:
            return
        pts = self._points.setdefault(rec["config"], [])
        pts.append((rec["nworld"], rec["ms_per_step"] * 1e-3 * RawConfig().substeps))
        if len({n for n, _ in pts}) >= 2:
            t_floor, c_w = fit_cost_model([n for n, _ in pts], [t for _, t in pts])
            rec["metrics"].update(t_floor_s=t_floor, c_w_s=c_w, n_knee=(t_floor / c_w) if c_w > 0 else None,
                                  fit_points=[n for n, _ in pts], fit_unit="s per control period")


def _p(samples: list[ProbeSample], name: str, q: float) -> float | None:
    vals = [np.ravel(getattr(s, name)) for s in samples if getattr(s, name) is not None]
    return float(np.percentile(np.concatenate(vals), q)) if vals else None


class L0Arm(_LadderScenario):
    id = "L0_arm"
    title = "Arm only (Panda stand-in for the T12 arm until WP13), sine joint targets"
    accuracy_ref = "max joint tracking error <= 0.1 rad"
    scene = "arm"
    configs = (Config(name="raw_mujoco", solver="mujoco", dt=GRID_DT, substeps=1, params={"scene": "arm"}),)


class L1BoxPile(_LadderScenario):
    id = "L1_box_pile"
    title = "Free 5 cm boxes, pre-settled half-offset pyramid"
    accuracy_ref = "settled penetration <= 1 mm, no box below ground"
    scene, param_key = "box_pile", "objects"
    configs = _configs("box_pile", "objects", (1, 4, 16, 64, 100), caps={64: 1024, 100: 1024})


class L2HullPile(_LadderScenario):
    id = "L2_hull_pile"
    title = "32-vertex convex hulls (8 shapes), loose pyramid dropped onto a plane"
    accuracy_ref = "settled penetration <= 1 mm, no hull below ground"
    scene, param_key = "hull_pile", "objects"
    configs = _configs("hull_pile", "objects", (1, 4, 16, 64, 100), caps={64: 1024, 100: 1024})


class L4CableChain(_LadderScenario):
    id = "L4_cable_chain"
    title = "Cantilever cable, capsule links on 3-axis angular joints (D7), discrete integrator"
    accuracy_ref = "time-averaged tip drop within 5 % of the small-deflection elastica wL^4/(8EI)"
    scene, param_key = "cable_chain", "links"
    configs = _configs("cable_chain", "links", (10, 20, 40, 80, 160))


SCENARIOS = [L0Arm(), L1BoxPile(), L2HullPile(), L4CableChain()]

__all__ = ["SCENARIOS", "L0Arm", "L1BoxPile", "L2HullPile", "L4CableChain"]
