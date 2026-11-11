# SPDX-License-Identifier: Apache-2.0
"""Speed-ladder rungs (SPEC §3.9 ``LadderRung``, parallel §4.9) and raw-Newton step targets for them.

``LADDER`` lists every rung and its sweep. ``RawRung`` builds one pack of a rung directly with Newton (template
from ``rung_scenes`` / ``rung_arm``, ``ModelBuilder.replicate`` with zero spacing, SolverMuJoCo) and exposes it as a
probe ``StepTarget`` whose ``step()`` is one control period of ``control_dt / dt`` substeps (an even number, so the
ping-pong swap is graph-safe). Arm targets come from a device kernel driven by a device substep counter, so the period
captures into a CUDA graph. ``gate()`` is the rung's quality gate (PRF-06: no gate, no result).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.defaults import CONTROL_DT, GRAVITY, PEN_LIMIT_M
from alloy.parallel import rung_arm, rung_scenes
from alloy.parallel.counters import ProbeSample, mujoco_counters
from alloy.parallel.probe import StepTarget
from alloy.parallel.types import LadderRung

log = logging.getLogger("alloy.parallel")

LADDER: tuple[LadderRung, ...] = (
    LadderRung(id="L0", scene="arm", sweep={}),
    LadderRung(id="L1", scene="box_pile", sweep={"objects": (1, 4, 16, 64, 100)}),
    LadderRung(id="L2", scene="hull_pile", sweep={"objects": (1, 4, 16, 64, 100)}),
    LadderRung(id="L3", scene="task_T12", sweep={}),
    LadderRung(id="L3", scene="task_T09", sweep={}),
    LadderRung(id="L4", scene="cable_chain", sweep={"links": (10, 20, 40, 80, 160)}),
    LadderRung(id="L5", scene="vbd_cloth", sweep={"resolution": (32, 64)}),
    LadderRung(id="L6", scene="mpm_sand", sweep={"particles": (100_000, 1_000_000)}, worlds=(1, 4, 16)),
    LadderRung(id="L7", scene="coupled_arm_cable", sweep={}),
)
RAW_SCENES = ("arm", "box_pile", "hull_pile", "cable_chain")
CABLE_GATE = 0.05  # tip drop within 5 % of the small-deflection elastica (ROD-1 bound)
ARM_GATE = 0.1  # [rad / m] max tracking error over the run


@dataclass(frozen=True)
class RawConfig:
    """Solver setup of a raw rung. dt is on the D4 grid (control_dt / 2^j)."""

    name: str = "raw_mujoco"
    control_dt: float = CONTROL_DT
    dt: float = 1.25e-3
    solver_kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def substeps(self) -> int:
        k = round(self.control_dt / self.dt)
        if not math.isclose(k * self.dt, self.control_dt, rel_tol=1e-9) or k % 2:
            raise ValueError(f"control_dt/dt must be an even integer, got {self.control_dt}/{self.dt}")
        return k


def solver_kwargs_for(scene: str, param: int) -> dict[str, Any]:
    """SolverMuJoCo options per raw scene (baseline-suite settings: R05 pile, D01 cable, R06 arm)."""
    if scene in ("box_pile", "hull_pile"):
        # box-box pairs make up to 8 MuJoCo contacts; a 64-box pyramid peaks at ~740 (11.5 per box), so 16 per box;
        # elliptic condim-3 contacts take 3 rows each
        ncon = 16 * param + 16
        return {"integrator": "implicitfast", "cone": "elliptic", "iterations": 50, "ls_iterations": 50,
                "nconmax": ncon, "njmax": 3 * ncon}
    if scene == "cable_chain":
        return {"integrator": "discrete", "iterations": 50, "disable_contacts": True}
    if scene == "arm":
        return {"iterations": 20, "ls_iterations": 20, "nconmax": 8, "njmax": 64}
    raise ValueError(f"no raw template for scene {scene!r} (raw scenes: {RAW_SCENES})")


@wp.kernel
def _arm_targets(step: wp.array[wp.int32], home: wp.array[wp.float32], swing: wp.array[wp.float32],
                 dt: float, omega: float, first: int, stride: int,
                 q_target: wp.array[wp.float32], qd_target: wp.array[wp.float32]):
    w, i = wp.tid()
    t = float(step[0]) * dt
    d = w * stride + first + i
    q_target[d] = home[i] + swing[i] * wp.sin(omega * t)
    qd_target[d] = swing[i] * omega * wp.cos(omega * t)


@wp.kernel
def _tick(step: wp.array[wp.int32]):
    step[0] = step[0] + 1


class RawRung:
    """One pack of a raw-Newton rung: model, SolverMuJoCo, states, drive, counters and gate."""

    def __init__(self, scene: str, param: int, n: int, device: str, cfg: RawConfig = RawConfig()) -> None:
        self.scene, self.param, self.n, self.device, self.cfg = scene, param, n, device, cfg
        template, self.facts, self.arm_first = _template(scene, param)
        main = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
        main.replicate(template, copy_count=n)
        self.model = main.finalize(device=device)
        kwargs = {**solver_kwargs_for(scene, param), **cfg.solver_kwargs}
        self.solver = newton.solvers.SolverMuJoCo(self.model, **kwargs)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.step_count = wp.zeros(1, dtype=wp.int32, device=device)  # device substep counter (graph-safe)
        self._arm_arrays()
        self._read = mujoco_counters(self.solver, self.model)
        self.tip_z: list[float] = []
        self.track_err = 0.0
        self.min_dist: list[float] = []
        self.overflow = 0  # worlds with an MJW capacity bit, max over samples (CONTACT/ROW_OVERFLOW are S1)
        self.iter_limit = 0  # worlds at the iteration cap (information, not in the gate)

    def _arm_arrays(self) -> None:
        if self.arm_first < 0:
            return
        self.home = wp.array(rung_arm.ARM_HOME, dtype=wp.float32, device=self.device)
        self.swing = wp.array(rung_arm.ARM_SWING, dtype=wp.float32, device=self.device)

    periods_per_call = 1  # one timed call = one control period

    @property
    def dt(self) -> float:
        return self.cfg.dt

    @property
    def control_dt(self) -> float:
        return self.cfg.control_dt

    @property
    def steps_per_period(self) -> int:
        return self.cfg.substeps

    def substep(self) -> None:
        """One solver step of ``cfg.dt``: arm targets at t = k dt (device counter), step, swap, tick."""
        if self.arm_first >= 0:
            stride = self.model.joint_dof_count // self.n
            wp.launch(_arm_targets, dim=(self.n, rung_arm.ARM_DOFS),
                      inputs=[self.step_count, self.home, self.swing, self.cfg.dt, 2.0 * math.pi * rung_arm.ARM_FREQ,
                              self.arm_first, stride, self.control.joint_target_q, self.control.joint_target_qd],
                      device=self.device)
        self.solver.step(self.s0, self.s1, self.control, None, self.cfg.dt)
        self.s0, self.s1 = self.s1, self.s0
        wp.launch(_tick, dim=1, inputs=[self.step_count], device=self.device)

    def step(self) -> None:
        """One control period (``cfg.substeps`` substeps, an even number of swaps)."""
        for _ in range(self.cfg.substeps):
            self.substep()

    def counters(self) -> ProbeSample:
        """Solver counters plus the gate's running measurements (host, between timed chunks)."""
        sample = self._read()
        self.overflow = max(self.overflow, sample.overflow)
        self.iter_limit = max(self.iter_limit, sample.iter_limit)
        self._gate_sample()
        return sample

    def target(self) -> StepTarget:
        return StepTarget(step=self.step, nworld=self.n, sim_dt=self.cfg.control_dt, device=self.device,
                          counters=self.counters, buffers=lambda: (self.s0.body_q, self.s1.body_q), model=self.model)

    # ---- quality gate ---------------------------------------------------------------------------------------
    def _gate_sample(self) -> None:
        if self.facts.kind == "cable":
            self.tip_z.append(float(np.mean(_cable_tip_z(self))))
        elif self.facts.kind == "arm":
            q = self.s0.joint_q.numpy().reshape(self.n, -1)[:, self.arm_first:self.arm_first + rung_arm.ARM_DOFS]
            tgt = self.control.joint_target_q.numpy().reshape(self.n, -1)
            tgt = tgt[:, self.arm_first:self.arm_first + rung_arm.ARM_DOFS]
            self.track_err = max(self.track_err, float(np.max(np.abs(q - tgt))))
        else:
            self.min_dist.append(_mjw_min_dist(self.solver))

    def gate(self) -> dict[str, Any]:
        """Quality gate of the run so far: {"pass": bool, metric: value, ...}."""
        finite = all(bool(np.all(np.isfinite(a.numpy()))) for a in (self.s0.body_q, self.s0.body_qd))
        out: dict[str, Any] = {"finite": finite, "overflow_worlds": self.overflow, "iter_limit_worlds": self.iter_limit}
        if self.facts.kind == "cable":
            out.update(_cable_gate(self))
        elif self.facts.kind == "arm":
            out.update(track_err=self.track_err, ok=self.track_err <= ARM_GATE)
        else:
            z = self.s0.body_q.numpy()[:, 2]
            late = self.min_dist[len(self.min_dist) // 2:]  # settled half of the run
            pen = max([0.0] + [-d for d in late if math.isfinite(d)])
            out.update(penetration_max_m=pen, min_z_m=float(z.min()), ok=pen <= PEN_LIMIT_M and float(z.min()) > 0.0)
        ok = out.pop("ok")
        out["pass"] = bool(finite and self.overflow == 0 and ok)
        return out


def _template(scene: str, param: int) -> tuple[newton.ModelBuilder, rung_scenes.SceneFacts, int]:
    if scene == "box_pile":
        return (*rung_scenes.box_pile(param), -1)
    if scene == "hull_pile":
        return (*rung_scenes.hull_pile(param), -1)
    if scene == "cable_chain":
        return (*rung_scenes.cable_chain(param), -1)
    if scene == "arm":
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
        first = rung_arm.panda_arm(b)
        return b, rung_scenes.SceneFacts(kind="arm", n_dynamic=b.body_count, arm_dofs=rung_arm.ARM_DOFS), first
    raise ValueError(f"no raw template for scene {scene!r} (raw scenes: {RAW_SCENES})")


def _cable_tip_z(r: RawRung) -> np.ndarray:
    """World z of each world's cable tip (end of the last link), [W]."""
    q = r.s0.body_q.numpy().reshape(r.n, -1, 7)[:, r.facts.tip_body]
    half = 0.5 * rung_scenes.CABLE_L / r.param
    x, y, z, w = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    # z-component of rotate(quat, (half, 0, 0)) = 2 (xz - wy) half
    return q[:, 2] + 2.0 * (x * z - w * y) * half


def _cable_gate(r: RawRung) -> dict[str, Any]:
    if len(r.tip_z) < 2:
        return {"ok": False, "tip_drop_err": None, "note": "fewer than 2 samples"}
    late = r.tip_z[len(r.tip_z) // 2:]  # second half: past the settling transient
    drop = rung_scenes.CABLE_Z - float(np.mean(late))
    err = drop / r.facts.tip_drop_ref - 1.0
    return {"ok": abs(err) <= CABLE_GATE, "tip_drop_m": drop, "tip_drop_ref_m": r.facts.tip_drop_ref,
            "tip_drop_err": err}


def _mjw_min_dist(solver: Any) -> float:
    data = solver.mjw_data
    nacon = int(data.nacon.numpy()[0])
    if nacon == 0:
        return math.inf
    return float(np.min(data.contact.dist.numpy()[:nacon]))
