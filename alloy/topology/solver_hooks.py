# SPDX-License-Identifier: Apache-2.0
"""Solver-specific access for the topology controller, kept in one place.

Private names used here are the ones in SCR-WP10-1:
- ``SolverMuJoCo.mjw_data.efc`` / ``nefc`` and ``mjc_eq_to_newton_eq``, for weld forces.
- A coupled stepper's ``_entries[name]`` (Newton's ``SolverEntry``: ``solver``, ``view``,
  ``attribute_projections``), for a MuJoCo entry that runs on a coupled ``ModelView`` (SCR-DC5-21, below).
- ``SolverImplicitMPM._mpm_model.{particle_flags, material_particle_flags, material_particle_volume, particle_volume}``,
  for activating spares in place.

Everything else is public Newton API. Absent arrays are replaced by 1-element dummies, so every kernel has one
signature.

Coupled MuJoCo entries (SCR-DC5-21). Under a coupled stepper (Scheduler, OneWayFollower, stock proxy), the MuJoCo
entry's solver runs on the entry's ``ModelView``. The view holds its own copies of ``mujoco:equality_constraint_*``,
rows renumbered by the entry's projection, so a weld disabled in the parent Model stayed active in MuJoCo. Given the
stepper (``coupled``), the hooks find that entry, map MuJoCo equality ids to parent rows through its
``local_to_global``, and copy the topology-managed rows (enabled, relpose) into the view before every solve and
before every solver notify (``sync_views``; kernels only). Without the stepper they log a warning.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology.bonds import NS
from alloy.topology.breaking import Efc
from alloy.topology.crumble_runtime import FlagTargets, MpmTargets
from alloy.topology.tables import BondTables
from alloy.topology.types import BondSet

log = logging.getLogger("alloy.topology")


def _dummy(dtype, device, shape=1):
    return wp.zeros(shape, dtype=dtype, device=device)


class SolverHooks:
    def __init__(self, model: newton.Model, entries: Mapping[str, newton.solvers.SolverBase], tables: BondTables,
                 bonds: BondSet, *, coupled: object | None = None, managed_eq: np.ndarray | None = None) -> None:
        self.model, self.tables = model, tables
        dev = model.device
        self.mujoco = [s for s in entries.values() if isinstance(s, newton.solvers.SolverMuJoCo)]
        self.mpm = [s for s in entries.values() if isinstance(s, newton.solvers.SolverImplicitMPM)]
        has = lambda n: getattr(model, n, None) is not None and getattr(model, n).shape[0] > 0  # noqa: E731
        self.tri_indices = model.tri_indices if has("tri_indices") else _dummy(wp.int32, dev, (1, 3))
        self.tri_poses = model.tri_poses if has("tri_poses") else _dummy(wp.mat22, dev)
        self.tet_indices = model.tet_indices if has("tet_indices") else _dummy(wp.int32, dev, (1, 4))
        self.tet_poses = model.tet_poses if has("tet_poses") else _dummy(wp.mat33, dev)
        self.spring_stiffness = model.spring_stiffness if model.spring_count else _dummy(wp.float32, dev)
        self.spring_damping = model.spring_damping if model.spring_count else _dummy(wp.float32, dev)
        mj = getattr(model, "mujoco", None)
        has_eq = mj is not None and getattr(mj, "equality_constraint_count", 0) > 0
        self.eq_enabled = mj.equality_constraint_enabled if has_eq else _dummy(wp.bool, dev)
        self.eq_relpose = mj.equality_constraint_relpose if has_eq else _dummy(wp.transform, dev)
        ns = getattr(model, NS, None)
        cl = getattr(ns, "crumble_local", None) if ns is not None else None
        self.crumble_local = cl if cl is not None and model.particle_count else _dummy(wp.vec3, dev)
        self.cut_strain = bonds.cut_strain if getattr(bonds, "cut_strain", None) is not None else tables.cut_strain
        self.cut_force = bonds.cut_force if getattr(bonds, "cut_force", None) is not None else tables.cut_force
        self._entry = _coupled_entry(coupled, self.mujoco[0]) if self.mujoco else None
        self._check_view(has_eq and tables.has_eq)
        self.mjc_eq = self._mjc_eq_map(bonds)
        self._mirror = EqViewMirror.build(model, self._entry, managed_eq) if has_eq else None
        self._needs_notify = bool(self.mujoco) and has_eq
        if has_eq and not self.mujoco and tables.has_eq:
            log.warning("equality bonds exist but no SolverMuJoCo entry was given: breaks change the Model only")

    # -- MuJoCo ------------------------------------------------------------------------------------------------
    def _check_view(self, welds: bool) -> None:
        """A MuJoCo entry on another model (a coupled view) needs its stepper, or weld breaks never reach it.

        A warning, not an error, until wiring passes the stepper: DC5's scenes still build (SCR-DC5-21)."""
        if not welds or not self.mujoco or self._entry is not None or self.mujoco[0].model is self.model:
            return
        log.warning("TopologyController: the SolverMuJoCo entry runs on a coupled entry view and no stepper was "
                    "given, so weld breaks reach the Model but not MuJoCo (the welds keep holding); pass "
                    "coupled=<the stepper> (SCR-DC5-21)")

    def _mjc_eq_map(self, bonds: BondSet) -> wp.array:
        """Newton equality row -> MuJoCo equality index (per its world), −1 if unknown."""
        dev = self.model.device
        out = np.full(max(bonds.count, 1), -1, np.int32)
        if not self.mujoco or not self.tables.has_eq:
            return wp.array(out, dtype=wp.int32, device=dev)
        table = np.asarray(self.mujoco[0].mjc_eq_to_newton_eq.numpy())       # [W, neq_mjc] -> Newton row (solver's)
        parent = _eq_local_to_global(self._entry)                            # view row -> Model row
        to_model = (lambda r: r) if parent is None else (lambda r: parent[r])  # noqa: E731
        inverse = {int(to_model(r)): j for row in table for j, r in enumerate(row) if r >= 0}
        eq = self.tables.eq.numpy()
        for i, r in enumerate(eq[:bonds.count]):
            out[i] = inverse.get(int(r), -1) if r >= 0 else -1
        return wp.array(out, dtype=wp.int32, device=dev)

    def efc(self) -> Efc:
        e = Efc()
        dev = self.model.device
        s = self.mujoco[0] if self.mujoco else None
        if s is not None and not s.use_mujoco_cpu:
            d = s.mjw_data
            e.use, e.nefc, e.type, e.id, e.force = 1, d.nefc, d.efc.type, d.efc.id, d.efc.force
        else:
            e.use = 0
            e.nefc = _dummy(wp.int32, dev)
            e.type, e.id = _dummy(wp.int32, dev, (1, 1)), _dummy(wp.int32, dev, (1, 1))
            e.force = _dummy(wp.float32, dev, (1, 1))
        return e

    def sync_views(self) -> None:
        """Copy the topology-managed equality rows into a coupled MuJoCo entry's view (no-op otherwise)."""
        if self._mirror is not None:
            self._mirror.launch()

    def notify(self) -> None:
        if self._needs_notify:
            self.sync_views()
            for s in self.mujoco:
                s.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)

    # -- particles / MPM ---------------------------------------------------------------------------------------
    def flag_targets(self) -> FlagTargets:
        m = self.model
        arrays = [m.particle_flags]
        for s in self.mpm:
            view_flags = s.model.particle_flags
            if view_flags.ptr != m.particle_flags.ptr:
                arrays.append(view_flags)
            arrays += [s._mpm_model.particle_flags, s._mpm_model.material_particle_flags]
        if len(arrays) > 4:
            raise AssemblyError(f"crumble: {len(arrays)} particle flag arrays to update (max 4: one MPM entry)")
        f = FlagTargets()
        f.n = len(arrays)
        f.f0, f.f1, f.f2, f.f3 = (arrays + [arrays[0]] * 4)[:4]
        return f

    def mpm_targets(self, state: newton.State) -> MpmTargets:
        t = MpmTargets()
        dev = self.model.device
        st = getattr(state, "mpm", None)
        if not self.mpm or st is None:
            t.use = 0
            m33, f = _dummy(wp.mat33, dev), _dummy(float, dev)
            t.qd_grad = t.elastic_strain = t.transform = t.stress = m33
            t.jp = t.jp0 = t.volume = t.material_volume = f
            return t
        mm = self.mpm[0]._mpm_model
        t.use = 1
        t.qd_grad, t.elastic_strain = st.particle_qd_grad, st.particle_elastic_strain
        t.transform, t.stress, t.jp = st.particle_transform, st.particle_stress, st.particle_Jp
        t.jp0, t.volume, t.material_volume = self.model.mpm.particle_Jp, mm.particle_volume, mm.material_particle_volume
        return t

    # -- snapshots ---------------------------------------------------------------------------------------------
    def mutated_model_arrays(self) -> Mapping[str, wp.array]:
        m = self.model
        out: dict[str, wp.array] = {}
        if m.spring_count:
            out["topology.model.spring_stiffness"] = m.spring_stiffness
            out["topology.model.spring_damping"] = m.spring_damping
        if m.edge_count and self.tables.edge_ids.shape[0]:
            out["topology.model.edge_bending_properties"] = m.edge_bending_properties
        if self.eq_enabled.shape[0] > 1 or getattr(getattr(m, "mujoco", None), "equality_constraint_count", 0):
            out["topology.model.eq_enabled"] = self.eq_enabled
            out["topology.model.eq_relpose"] = self.eq_relpose
        if m.particle_count:
            out["topology.model.particle_flags"] = m.particle_flags
        for k, s in enumerate(self.mpm):
            out[f"topology.mpm{k}.particle_flags"] = s._mpm_model.particle_flags
            out[f"topology.mpm{k}.material_particle_flags"] = s._mpm_model.material_particle_flags
            out[f"topology.mpm{k}.material_particle_volume"] = s._mpm_model.material_particle_volume
        return out

    def mutated_model_worlds(self) -> Mapping[str, wp.array]:
        m, dev = self.model, self.model.device
        out: dict[str, wp.array] = {}
        w = max(int(m.world_count), 1)
        fix = lambda x: np.where(np.asarray(x) < 0, 0, x).astype(np.int32) if w <= 1 else np.asarray(x, np.int32)  # noqa: E731
        if m.particle_count:
            pw = fix(m.particle_world.numpy())
            out["topology.model.particle_flags"] = wp.array(pw, dtype=wp.int32, device=dev)
            for k in range(len(self.mpm)):
                for n in ("particle_flags", "material_particle_flags", "material_particle_volume"):
                    out[f"topology.mpm{k}.{n}"] = out["topology.model.particle_flags"]
            if m.spring_count:
                sw = wp.array(pw[m.spring_indices.numpy().reshape(-1, 2)[:, 0]], dtype=wp.int32, device=dev)
                out["topology.model.spring_stiffness"] = out["topology.model.spring_damping"] = sw
            if "topology.model.edge_bending_properties" in self.mutated_model_arrays():
                ew = pw[m.edge_indices.numpy()[:, 2]]
                out["topology.model.edge_bending_properties"] = wp.array(ew, dtype=wp.int32, device=dev)
        mj = getattr(m, "mujoco", None)
        if mj is not None and getattr(mj, "equality_constraint_count", 0):
            bw = fix(m.body_world.numpy()) if m.body_count else np.zeros(1, np.int32)
            b1, b2 = mj.equality_constraint_body1.numpy(), mj.equality_constraint_body2.numpy()
            ew = np.maximum(np.where(b1 >= 0, bw[np.maximum(b1, 0)], -1), np.where(b2 >= 0, bw[np.maximum(b2, 0)], -1))
            eqw = wp.array(np.maximum(ew, 0).astype(np.int32), dtype=wp.int32, device=dev)
            out["topology.model.eq_enabled"] = out["topology.model.eq_relpose"] = eqw
        bw_all = fix(m.body_world.numpy()) if m.body_count else np.zeros(1, np.int32)   # placeholder -> world 0
        out["topology.body_active"] = wp.array(bw_all, dtype=wp.int32, device=dev)
        return out


# -- coupled MuJoCo entry views (SCR-DC5-21) ---------------------------------------------------------------------
EQ_FREQUENCY = "mujoco:equality_constraint"


def _coupled_entry(coupled: object | None, solver: object) -> object | None:
    """The stepper's entry whose solver is ``solver``: Scheduler / SolverCoupled ``_entries``, OneWayFollower (its
    ``_entries`` come from ``base``), or a stock-proxy wrapper's ``solver``. None when not found."""
    for owner in (coupled, getattr(coupled, "solver", None)):
        entries = getattr(owner, "_entries", None) if owner is not None else None
        if isinstance(entries, Mapping):
            for entry in entries.values():
                if getattr(entry, "solver", None) is solver:
                    return entry
    return None


def _eq_local_to_global(entry: object | None) -> list[int] | None:
    proj = getattr(entry, "attribute_projections", {}).get(EQ_FREQUENCY) if entry is not None else None
    return None if proj is None else list(proj.local_to_global)


@wp.kernel
def copy_eq_rows(src: wp.array[wp.int32], dst: wp.array[wp.int32], en: wp.array[wp.bool],
                 rel: wp.array[wp.transform], view_en: wp.array[wp.bool], view_rel: wp.array[wp.transform]):
    k = wp.tid()
    view_en[dst[k]] = en[src[k]]
    view_rel[dst[k]] = rel[src[k]]


@dataclass
class EqViewMirror:
    """Parent rows ``src`` -> view rows ``dst`` of the topology-managed equalities, with both array pairs."""

    src: wp.array
    dst: wp.array
    arrays: tuple[wp.array, wp.array, wp.array, wp.array]     # parent enabled, relpose; view enabled, relpose

    @classmethod
    def build(cls, model: newton.Model, entry: object | None, managed: np.ndarray | None) -> EqViewMirror | None:
        if entry is None or managed is None:
            return None
        mj, vmj = model.mujoco, entry.view.mujoco
        arrays = (mj.equality_constraint_enabled, mj.equality_constraint_relpose,
                  vmj.equality_constraint_enabled, vmj.equality_constraint_relpose)
        if arrays[0].ptr == arrays[2].ptr and arrays[1].ptr == arrays[3].ptr:
            return None                                        # the view shares the parent's arrays
        g2l = entry.attribute_projections[EQ_FREQUENCY].global_to_local
        rows = np.unique(np.asarray(managed, np.int64))
        rows = rows[(rows >= 0) & (rows < len(g2l))]
        local = np.asarray([g2l[r] for r in rows], np.int64)
        keep = local >= 0
        if not keep.any():
            return None
        dev = model.device
        log.info("topology: mirroring %d equality rows into the coupled MuJoCo entry view (SCR-DC5-21)", keep.sum())
        return cls(wp.array(rows[keep].astype(np.int32), dtype=wp.int32, device=dev),
                   wp.array(local[keep].astype(np.int32), dtype=wp.int32, device=dev), arrays)

    def launch(self) -> None:
        wp.launch(copy_eq_rows, dim=self.src.shape[0], device=self.src.device,
                  inputs=[self.src, self.dst, *self.arrays])
