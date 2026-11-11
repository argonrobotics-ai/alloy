# SPDX-License-Identifier: Apache-2.0
"""Newton or CG for a MuJoCo entry that steps a chain attachment (SCR-DC1-2, coordinator ruling at the end of
[internal note]).

A weld or joint on a CHAIN object marks its MuJoCo entry for the CG solver (`attachments.CG_EFFECT`). MuJoCo Warp's
Newton solver factorises H = M + JᵀDJ in float32, and on light chain links that went non-finite (T09, A10). This rule
keeps Newton where the float32 Hessian stays positive definite, by the criterion of the #1801 analysis:

    eps32 · |JᵀDJ| < λ_min(M)

- Both sides are taken on the Jacobi-scaled matrices S·M·S and S·JᵀDJ·S, with S = diag(M)^-1/2. A Cholesky pivot
  relative to its diagonal does not change under that scaling, so the comparison has no units; the unscaled form
  mixes kg and kg·m² dofs. M includes joint armature, so the CHAIN floor (`emitters.chain.ARMATURE_FLOOR`) counts.
- JᵀDJ sums every constraint row as if it were active: an upper bound on the Newton curvature.
- The probe runs the scene as it would run on Newton: the chain's CG effect is dropped, so the impratio is the one
  `entries.mujoco_overrides` gives a chain on Newton. It runs 1 world (world 0's draw) on MuJoCo C (float64) on the
  CPU for PROBE_PERIODS control periods (1 s at a 20 ms period). The ratio is the largest at the start and after
  every period. The probe sees the scene settle, not its task: contacts the task makes later (an insertion, a pull)
  are not in it. A scene that needs Newton through such contacts brings evidence instead.
- Newton when the ratio is below RULE_BOUND, else CG as before. A scene that fails may still run Newton with evidence
  (`alloy.scene.newton_evidence`).

Measured with this probe ([internal note], rows1/rows2.jsonl):
- T09's fast profile at chain armature 1e-5 (0 INVALID in ~6.8e7 seated world-steps): 0.59.
- The same profile at 1e-6 (1 non-finite world in ~35,300 world-s): 1.9.
- T09 on Newton at its routed impratio 300 (non-finite on float32): 290.

Results are cached in-process and in `$ALLOY_CACHE_DIR/newtonrule.json`. The key is the template model's digest, the
candidate entry's options and the rule version; the route report prints it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from alloy.route.types import Route
from alloy.scene.attachments import CG_EFFECT
from alloy.scene.template import TemplateObject

log = logging.getLogger("alloy.scene")

EPS32 = float(np.finfo(np.float32).eps)
RULE_BOUND = 1.0            # Newton when eps32·|S·JᵀDJ·S| / λ_min(S·M·S) is below this
PROBE_PERIODS = 50          # control periods the probe steps: 1 s at 20 ms, for objects placed above a surface
RULE_VERSION = 1
_UNKEYED = ("njmax", "nconmax", "use_mujoco_cpu")      # capacity options scale with worlds; the probe sets the last
_MEMO: dict[str, dict] = {}


@dataclass(frozen=True)
class Margin:
    ratio: float            # eps32·|S·K·S|₂ / λ_min(S·M·S)
    lam: float              # λ_min(S·M·S)
    curvature: float        # |S·K·S|₂
    dof: int                # the dof with the largest weight in λ_min's eigenvector


@dataclass(frozen=True)
class Verdict:
    newton: bool
    key: str
    probe: dict             # ratio, lam, curvature, dof, finite, periods, impratio
    evidence: str | None = None

    def line(self) -> str:
        p = self.probe
        if "error" in p:
            return (f"mujoco solver: CG kept for the chain attachment, the rule's probe failed ({p['error']}); probe "
                    f"key {self.key} (SCR-DC1-2)")
        at = (f"margin eps32·|S·JᵀDJ·S| / λ_min(S·M·S) = {p['ratio']:.3g} (bound {RULE_BOUND:g}; worst dof "
              f"{p['dof']}; 1 world on MuJoCo C, {p['periods']} periods at impratio {p['impratio']:g}"
              f"{'' if p['finite'] else ', non-finite'})")
        if self.evidence is not None:
            head = f"Newton for the chain attachment by evidence ({self.evidence}); rule {at}"
        elif self.newton:
            head = f"Newton for the chain attachment, float32 Hessian stays positive definite: {at}"
        else:
            head = f"CG kept for the chain attachment, float32 Newton Hessian not shown positive definite: {at}"
        return f"mujoco solver: {head}; probe key {self.key} (SCR-DC1-2)"


def margin(M: np.ndarray, K: np.ndarray) -> Margin:
    """The rule's ratio for mass matrix M and constraint curvature K (both dense, nv x nv)."""
    s = 1.0 / np.sqrt(np.diag(M))
    lam, vec = np.linalg.eigh(s[:, None] * M * s[None, :])
    k = float(np.linalg.eigvalsh(s[:, None] * K * s[None, :])[-1]) if K.size and np.any(K) else 0.0
    ratio = EPS32 * k / lam[0] if lam[0] > 0.0 else math.inf
    return Margin(float(ratio), float(lam[0]), k, int(np.argmax(np.abs(vec[:, 0]))))


def mj_matrices(m, d) -> tuple[np.ndarray, np.ndarray]:
    """Dense M (armature included) and K = Σ D_r J_rᵀJ_r over every efc row, from MuJoCo C model/data."""
    import mujoco

    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    n = int(d.nefc)
    if mujoco.mj_isSparse(m):
        J = np.zeros((n, m.nv))
        for r in range(n):
            a, k = d.efc_J_rowadr[r], d.efc_J_rownnz[r]
            J[r, d.efc_J_colind[a:a + k]] = d.efc_J[a:a + k]
    else:
        J = np.asarray(d.efc_J[: n * m.nv], np.float64).reshape(n, m.nv)
    D = np.asarray(d.efc_D[:n], np.float64)
    return M, J.T @ (D[:, None] * J)


def dof_name(m, i: int) -> str:
    import mujoco

    jnt = int(m.dof_jntid[i])
    body = int(m.jnt_bodyid[jnt])
    return f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or f'body{body}'}/dof{i - int(m.jnt_dofadr[jnt])}"


# --- which entries the rule decides -----------------------------------------------------------------------------

def candidates(objs: Sequence[TemplateObject], fast_newton: bool) -> list[TemplateObject] | None:
    """`objs` with the chain attachments' CG effect dropped (the scene as Newton would run it), or None when the rule
    has nothing to decide: no chain carries the effect, or the fast profile already chose Newton (SCR-WP3-8)."""
    if fast_newton or not any(_marked(t) for t in objs):
        return None
    return [replace(t, decision=_without_cg(t.decision)) if _marked(t) else t for t in objs]


def _marked(t: TemplateObject) -> bool:
    return t.decision.route == Route.CHAIN and CG_EFFECT in t.decision.scene_effects


def _without_cg(d):
    """The decision on Newton: CG effect dropped, the attachment kept as a mark (`chain_grip.ATTACHED_EFFECT`, so the
    chain-grip rule leaves a chain attachment's entry as it was)."""
    from alloy.contact.chain_grip import ATTACHED_EFFECT

    return replace(d, scene_effects=tuple(e for e in d.scene_effects if e != CG_EFFECT) + (ATTACHED_EFFECT,))


def apply_newton(objs: Sequence[TemplateObject], report) -> None:
    """Drop the CG effect from the chains' decisions and from the report's scene effects (in place)."""
    for t in objs:
        if _marked(t):
            t.decision = _without_cg(t.decision)
    report.scene_effects[:] = [e for e in report.scene_effects if e != CG_EFFECT]


def probe_key(builder, kwargs: dict, target_dt: float, control_dt: float) -> str:
    """Digest of what the probe depends on: the template builder, the candidate MuJoCo options and the rule. The
    same on every device (`builder_digest`), so CPU and CUDA assemblies share the probe and the evidence."""
    opts = {k: kwargs[k] for k in sorted(kwargs) if k not in _UNKEYED}
    h = hashlib.sha256(f"v{RULE_VERSION}|{PROBE_PERIODS}|{RULE_BOUND}|{target_dt:.9e}|{control_dt:.9e}".encode())
    h.update(builder_digest(builder).encode())
    h.update(json.dumps(opts, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def builder_digest(builder) -> str:
    """sha256 over the builder's numeric per-element lists (name, values as float64), its custom attributes and the
    shapes' mesh content. Precision feature SDFs are left out: they are baked on CUDA only (`_feature_sdfs`), and
    finalizing a template with them on the CPU raises."""
    from alloy.scene.assembler import _digest_meshes

    h = hashlib.sha256()
    for name in sorted(vars(builder)):
        if "sdf" not in name.lower():
            _digest_list(h, name, getattr(builder, name))
    for name, attr in sorted(builder.custom_attributes.items()):
        vals = attr.values
        _digest_list(h, f"custom.{name}", [vals[k] for k in sorted(vals)] if isinstance(vals, dict) else vals)
    _digest_meshes(h, builder.shape_source)
    return h.hexdigest()


def _digest_list(h, name: str, val) -> None:
    if not isinstance(val, list) or not val:
        return
    try:
        arr = np.asarray(val, dtype=np.float64)
    except (TypeError, ValueError):          # labels, sources and other objects: not numbers
        return
    h.update(f"{name}|{arr.shape}|".encode())
    h.update(np.ascontiguousarray(arr).tobytes())


# --- the probe -----------------------------------------------------------------------------------------------

def run_probe(assembler, scene) -> dict:
    """Step the Newton candidate of `scene` (1 world, MuJoCo C, CPU) and return the worst margin seen."""
    import mujoco

    from alloy.scene.assembler import SceneAssembler
    from alloy.scene.noslip_route import NOSLIP_KEY
    from alloy.sim import Simulation

    sub = SceneAssembler(assembler.library, assembler.router, route_profile=assembler.route_profile,
                         newton_candidate=True)
    built = sub.assemble(scene, "cpu", world_count=1)
    mj = next(e for e in built.entries if e.name == "mujoco")
    mj.solver_kwargs["use_mujoco_cpu"] = True
    mj.solver_kwargs.pop(NOSLIP_KEY, None)         # scoped noslip runs on MuJoCo Warp only; it is after the solve
    sim = Simulation(built)
    solver = next(s for s in sim.stepper.solvers() if hasattr(s, "mj_model"))
    m, d = solver.mj_model, solver.mj_data
    mujoco.mj_forward(m, d)
    worst, finite = margin(*mj_matrices(m, d)), True
    for _ in range(PROBE_PERIODS):
        sim.step_period()
        if not np.isfinite(sim.state.body_q.numpy()).all():
            finite = False
            break
        g = margin(*mj_matrices(m, d))
        worst = g if g.ratio > worst.ratio else worst
    return {"ratio": worst.ratio if finite else math.inf, "lam": worst.lam, "curvature": worst.curvature,
            "dof": dof_name(m, worst.dof), "finite": finite, "periods": PROBE_PERIODS,
            "impratio": float(m.opt.impratio)}


def decide(assembler, scene, key: str) -> Verdict:
    """The rule's verdict for the candidate with probe key `key` (cached), with evidence when it fails."""
    from alloy.scene import newton_evidence as EV

    p = _MEMO.get(key) or _cache_load().get(key)
    if p is not None and p.get("ratio") is None:            # JSON keeps a non-finite ratio as null
        p = {**p, "ratio": math.inf}
    if p is None:
        try:
            p = run_probe(assembler, scene)
        except Exception as exc:  # noqa: BLE001 - the probe must not stop an assembly; CG stays (not cached)
            log.warning("SCR-DC1-2 probe %s failed, CG kept: %s: %s", key, type(exc).__name__, exc)
            p = {"ratio": math.inf, "lam": math.nan, "curvature": math.nan, "dof": "-", "finite": False,
                 "periods": 0, "impratio": math.nan, "error": f"{type(exc).__name__}: {exc}"[:200]}
        else:
            _cache_store(key, p)
            log.info("SCR-DC1-2 probe %s: margin %.3g (worst dof %s)", key, p["ratio"], p["dof"])
    _MEMO[key] = p
    if p["finite"] and p["ratio"] < RULE_BOUND:
        return Verdict(True, key, p)
    ev = EV.find(key)
    return Verdict(ev is not None, key, p, None if ev is None else ev.describe())


def _cache_path() -> Path:
    return Path(os.environ.get("ALLOY_CACHE_DIR", Path.home() / ".cache" / "alloy")) / "newtonrule.json"


def _cache_load() -> dict[str, dict]:
    try:
        return json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return {}


def _cache_store(key: str, value: dict) -> None:
    path = _cache_path()
    data = _cache_load()
    data[key] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in value.items()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, sort_keys=True, indent=1))
        tmp.replace(path)
    except OSError as exc:
        log.warning("newton rule cache not written (%s): %s", path, exc)


__all__ = ["EPS32", "PROBE_PERIODS", "RULE_BOUND", "Margin", "Verdict", "apply_newton", "candidates", "decide",
           "margin", "mj_matrices", "probe_key", "run_probe"]
