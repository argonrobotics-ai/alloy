# SPDX-License-Identifier: Apache-2.0
"""ctypes mirror of ``alloy_mjc.h`` (ABI v2, agreed with WP8: [internal note] notes/wp8-results/cpu-item11/ABI.md).

:class:`MjcConfig` and :class:`MjcOut` hold the C structs and every numpy array they point to, so the arrays stay
alive as long as the struct. Field order and types must match ``alloy_mjc.h`` exactly (ctypes lays them out with the C
alignment rules); ``tests/route/test_mjc_step.py`` checks the sizes against the library.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_double, c_int32, c_int64, c_uint8, c_void_p
from typing import Any

import numpy as np

ABI = 1
OK, E_ABI, E_INTEGRATOR, E_NOSLIP_TWICE, E_CONE, E_SLEEP, E_ARGS, E_NOMEM = 0, -1, -2, -3, -4, -5, -6, -7
ERRORS = {E_ABI: "ABI mismatch", E_INTEGRATOR: "RK4 / discrete integrator (use mj_step)",
          E_NOSLIP_TWICE: "the model's own noslip is on", E_CONE: "scoped noslip needs the elliptic cone",
          E_SLEEP: "sleeping is not supported", E_ARGS: "bad arguments (sizes, NULL buffers, MuJoCo thread pool)",
          E_NOMEM: "scratch allocation failed"}

_i32p, _f64p = POINTER(c_int32), POINTER(c_double)


class Config(ctypes.Structure):
    _fields_ = [("abi", c_int32), ("noslip_iterations", c_int32), ("noslip_tolerance", c_double),
                ("noslip_all", c_int32), ("noslip_restart", c_int32), ("noslip_warm_qcqp", c_int32),
                ("noslip_max_slots", c_int32), ("geom_scope", _i32p), ("dof_scope", _i32p), ("npair", c_int32),
                ("pairs", _i32p), ("line", c_int32), ("export_contacts", c_int32), ("pull_state", c_int32),
                ("nbody_out", c_int32), ("body_map", _i32p)]


class Out(ctypes.Structure):
    _fields_ = [("ncon_max", c_int32), ("ncon", c_int32), ("geom", _i32p), ("pos", _f64p), ("frame", _f64p),
                ("dist", _f64p), ("force", _f64p), ("body_q", _f64p), ("body_qd", _f64p), ("traj_cap", c_int32),
                ("traj_body_q", _f64p), ("traj_body_qd", _f64p), ("steps", c_int64), ("nonfinite", c_int64),
                ("contact_overflow", c_int64), ("pen_max", c_double), ("qvel_max", c_double),
                ("noslip_slots", c_int64), ("noslip_sweeps", c_int64), ("noslip_overflow", c_int64),
                ("line_fired", c_int64)]


def declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Set the prototypes of the library's functions."""
    cfgp, outp = POINTER(Config), POINTER(Out)
    protos = {
        "alloy_mjc_abi": ([], c_int32),
        "alloy_mjc_mujoco_version": ([], c_int32),
        "alloy_mjc_tick": ([cfgp, c_void_p, c_void_p, c_int32, outp], c_int32),
        "alloy_mjc_tick_batch": ([c_int32, POINTER(cfgp), POINTER(c_void_p), POINTER(c_void_p), POINTER(c_uint8),
                                  c_int32, outp, _i32p], c_int32),
        "alloy_mjc_noslip": ([cfgp, c_void_p, c_void_p, outp], c_int32),
        "alloy_mjc_boxbox": ([c_void_p, c_void_p, c_int32, c_int32, c_double, c_int32, _f64p, _f64p, _f64p], c_int32),
        "alloy_mjc_unhook": ([], None),
        "alloy_mjc_struct_sizes": ([_i32p], None),
    }
    for name, (args, res) in protos.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    sizes = (c_int32 * 2)()
    lib.alloy_mjc_struct_sizes(sizes)
    if (sizes[0], sizes[1]) != (ctypes.sizeof(Config), ctypes.sizeof(Out)):
        raise OSError(f"struct layout differs from alloy_mjc.h: C {tuple(sizes)}, ctypes "
                      f"{(ctypes.sizeof(Config), ctypes.sizeof(Out))}")
    return lib


def slots(spec: Any) -> int:
    """Slots per world, as alloy.contact.noslip._slots: rows_per_world if given, 64 for scope "robot" without
    surfaces, else every row (C allocates per call, so no njmax bound)."""
    if spec.rows_per_world is not None:
        return int(spec.rows_per_world)
    if spec.scope != "all" and not tuple(getattr(spec, "surfaces", ())):
        return 64
    return 2**31 - 1


def _i32(a: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.int32))


def _ptr(a: np.ndarray | None, kind: Any) -> Any:
    return a.ctypes.data_as(kind) if a is not None and a.size else kind()


class MjcConfig:
    """One model's config: the noslip spec and scope (alloy.contact.noslip ``NoslipSpec`` / ``NoslipScope``, or
    None for no noslip), the line switch, and what a tick exports. ``body_map``: MuJoCo body id per Newton body row
    (-1 = skip)."""

    def __init__(self, *, noslip: Any = None, scope: Any = None, line: bool = True, export_contacts: bool = False,
                 pull_state: bool = False, body_map: Any = None) -> None:
        self.struct = Config(abi=ABI, line=int(line), export_contacts=int(export_contacts),
                             pull_state=int(pull_state))
        self._keep: list[np.ndarray] = []
        if noslip is not None:
            self._set_noslip(noslip, scope)
        if body_map is not None:
            bm = self._hold(_i32(body_map))
            self.struct.nbody_out, self.struct.body_map = int(bm.size), _ptr(bm, _i32p)

    def _hold(self, a: np.ndarray) -> np.ndarray:
        self._keep.append(a)
        return a

    def _set_noslip(self, spec: Any, scope: Any) -> None:
        s = self.struct
        s.noslip_iterations, s.noslip_tolerance = int(spec.iterations), float(spec.tolerance)
        s.noslip_all, s.noslip_restart = int(spec.scope == "all"), int(spec.restart)
        s.noslip_warm_qcqp = int(spec.warm_qcqp)
        s.noslip_max_slots = slots(spec)
        s.geom_scope = _ptr(self._hold(_i32(scope.geom)), _i32p)
        s.dof_scope = _ptr(self._hold(_i32(scope.dof)), _i32p)
        pairs = self._hold(_i32(scope.pairs).reshape(-1, 2))
        s.npair, s.pairs = int(pairs.shape[0]), _ptr(pairs, _i32p)


class MjcOut:
    """One world's outputs: numpy buffers (``ncon_max`` contacts, ``nbody`` body rows, ``traj_cap`` trajectory
    rows) and the counters."""

    def __init__(self, *, ncon_max: int = 0, nbody: int = 0, traj_cap: int = 0) -> None:
        self.geom = np.zeros((ncon_max, 2), np.int32)
        self.pos, self.frame = np.zeros((ncon_max, 3)), np.zeros((ncon_max, 9))
        self.dist, self.force = np.zeros(ncon_max), np.zeros((ncon_max, 6))
        self.body_q, self.body_qd = np.zeros((nbody, 7)), np.zeros((nbody, 6))
        self.traj_body_q, self.traj_body_qd = np.zeros((traj_cap, nbody, 7)), np.zeros((traj_cap, nbody, 6))
        self.struct = Out(ncon_max=ncon_max, traj_cap=traj_cap)
        for name in ("geom", "pos", "frame", "dist", "force", "body_q", "body_qd", "traj_body_q", "traj_body_qd"):
            arr = getattr(self, name)
            setattr(self.struct, name, _ptr(arr, _i32p if arr.dtype == np.int32 else _f64p))

    def counters(self) -> dict[str, float]:
        names = ("steps", "nonfinite", "contact_overflow", "pen_max", "qvel_max", "noslip_slots", "noslip_sweeps",
                 "noslip_overflow", "line_fired")
        return {n: getattr(self.struct, n) for n in names}


__all__ = ["ABI", "ERRORS", "Config", "MjcConfig", "MjcOut", "Out", "declare"]
