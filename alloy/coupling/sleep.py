# SPDX-License-Identifier: Apache-2.0
"""Sleeping policy (SPEC §3.6 "Sleeping is delegated", D10; coupling-time §4.4.7).

Sleeping is MuJoCo Warp's (``SolverMuJoCo(enable_sleeping=True, sleep_tolerance, nvmax)``; GPU backend, MuJoCo contact
handling and islands only: newton@ac7f7830:newton/_src/solvers/mujoco/solver_mujoco.py:4350-4352, 4460-4485, 4661).
This module turns a ``SleepPolicy`` into those kwargs, sets the per-body ``mujoco:sleep_policy`` custom attribute
(INIT for layout objects that start asleep, NEVER for the ``never`` cases), and provides the coupler-level interface
freeze test (every source endpoint of a world asleep in MuJoCo and the destination proxies quiet).
Sleeping is forced off for ``native_contacts="newton"`` (D10), on CPU, and in drift/creep tests (policy.enabled=False).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy.coupling.types import SleepPolicy

log = logging.getLogger("alloy.coupling")

SLEEP_ATTRIBUTE = "mujoco:sleep_policy"
# newton SolverMuJoCo.SleepPolicy (solver_mujoco.py:813-823)
POLICY_AUTO, POLICY_NEVER, POLICY_ALLOWED, POLICY_INIT = 0, 3, 4, 5


def sleep_off_reason(policy: SleepPolicy, native_contacts: str, device: Any) -> str | None:
    """Why sleeping must be off for this entry, or None if it may be on."""
    if not policy.enabled:
        return "SleepPolicy.enabled is False"
    if native_contacts == "newton":
        return 'native_contacts="newton": MuJoCo sleeping needs MuJoCo contact handling (D10)'
    if not wp.get_device(device).is_cuda:
        return "MuJoCo Warp sleeping needs the GPU backend"
    return None


def mujoco_sleep_kwargs(policy: SleepPolicy, *, nv: int, native_contacts: str = "mujoco", precision: bool = False,
                        device: Any = None) -> dict[str, Any]:
    """SolverMuJoCo kwargs for this policy; ``{}`` (sleeping off) when ``sleep_off_reason`` says so.

    ``sleep_tolerance`` = ``lin_tol`` (``lin_tol_precision`` in precision scenes); ``nvmax = nv`` when
    ``nv <= mujoco_nvmax_full_below`` (compaction pays only for large nv, findings/timestep.md §4.4)."""
    reason = sleep_off_reason(policy, native_contacts, device)
    if reason is not None:
        log.info("MuJoCo sleeping off: %s", reason)
        return {}
    kwargs: dict[str, Any] = {
        "enable_sleeping": True,
        "sleep_tolerance": float(policy.lin_tol_precision if precision else policy.lin_tol),
    }
    if nv <= policy.mujoco_nvmax_full_below:
        kwargs["nvmax"] = int(nv)
    return kwargs


def body_sleep_policy(*, actuated: bool = False, precision_pair: bool = False, applied_wrench: bool = False,
                      layout: bool = False, policy: SleepPolicy = SleepPolicy()) -> int:
    """mujoco:sleep_policy value for one body: NEVER for the ``policy.never`` cases, INIT for layout objects when
    ``init_asleep``, else AUTO. (Precision pairs never sleep while touching; a static NEVER is the safe v1 rule.)"""
    never = set(policy.never)
    if (actuated and "actuated" in never) or (applied_wrench and "applied_wrench" in never):
        return POLICY_NEVER
    if precision_pair and ({"precision_pair_in_contact", "precision_pair_within_gap"} & never):
        return POLICY_NEVER
    if layout and policy.init_asleep:
        return POLICY_INIT
    return POLICY_AUTO


def apply_body_sleep_policies(builder: Any, policies: Mapping[int, int]) -> None:
    """Write ``mujoco:sleep_policy`` overrides (body index -> value) on a builder that registered SolverMuJoCo's
    custom attributes."""
    attr = builder.custom_attributes.get(SLEEP_ATTRIBUTE)
    if attr is None:
        raise KeyError(f"{SLEEP_ATTRIBUTE!r} is not registered: call SolverMuJoCo.register_custom_attributes first")
    if attr.values is None:
        attr.values = {}
    for body, value in policies.items():
        attr.values[int(body)] = int(value)


@wp.kernel(enable_backward=False)
def interface_asleep_kernel(
    csr_start: wp.array[wp.int32],
    csr_index: wp.array[wp.int32],
    endpoint_mjc_body: wp.array[wp.int32],
    proxy_local: wp.array[int],
    body_awake: wp.array2d[wp.int32],
    dst_qd: wp.array[wp.spatial_vector],
    lin_tol: float,
    start_frozen: wp.array[wp.int32],
):
    """start_frozen[w] = 1 iff the world has endpoints, all asleep in MuJoCo, and every proxy is below lin_tol."""
    w = wp.tid()
    n = csr_start[w + 1] - csr_start[w]
    frozen = int(n > 0)
    for j in range(csr_start[w], csr_start[w + 1]):
        i = csr_index[j]
        b = endpoint_mjc_body[i]
        if b < 0 or body_awake[w, b] != 0:
            frozen = 0
        if wp.length(wp.spatial_top(dst_qd[proxy_local[i]])) > lin_tol:
            frozen = 0
    start_frozen[w] = frozen


def endpoint_mjc_bodies(solver: Any, src_local: Iterable[int], world_count: int) -> np.ndarray:
    """MuJoCo body index of each source endpoint (entry-view body ids), -1 where unmapped."""
    table = getattr(solver, "mjc_body_to_newton", None)
    src = np.asarray(list(src_local), dtype=np.int64)
    out = np.full(src.shape[0], -1, dtype=np.int32)
    if table is None:
        return out
    t = table.numpy()
    inverse: dict[int, int] = {}
    for w in range(t.shape[0]):
        for b in range(t.shape[1]):
            if t[w, b] >= 0:
                inverse[int(t[w, b])] = b
    for k, body in enumerate(src):
        out[k] = inverse.get(int(body), -1)
    return out


def solver_sleeping(solver: Any) -> bool:
    return bool(getattr(solver, "enable_sleeping", False)) and getattr(solver, "mjw_data", None) is not None
