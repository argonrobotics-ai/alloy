# SPDX-License-Identifier: Apache-2.0
"""Newton pin and the list of Newton private names Alloy touches (SPEC §3.1, §6.6, D10, D21).

Every module that touches a Newton private name lists it in ``PRIVATE_HOOKS``; ``tests/core/test_newton_contract.py``
fails on a Newton bump that changes one. Coupled-solver classes are re-exported from here so packages import them
from one place.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import inspect
import json
import logging
import re
from typing import Final

from alloy._vendor.newton.solvers.experimental.coupled import (  # re-exported
    CouplingInterface,
    ModelView,
    SolverCoupled,
    SolverCoupledADMM,
    SolverCoupledProxy,
)

from alloy.core.errors import NewtonPinError, SolverOptionError

log = logging.getLogger("alloy.core")

# Private pin (NEWTON-BUMP 2026-11-09): public release-1.8 db915e1c + our 12 commits (alloy/_vendor/VENDORED.md),
# also as vendor/newton/pin.bundle for scripts/newton_env.sh; Alloy imports the vendored copy. Keep the two pin lines
# bare: scripts (newton_env.sh, gpu_run.sh, gate_remote.py) read them with an end-anchored regex.
NEWTON_PIN: Final[str] = "25b4603cfb37dfa250d8b9e1727b39514ab6e8a0"
# MuJoCo Warp release v3.17.0: Newton release-1.8 needs MuJoCo 3.17 (eq_data is 12 wide). It keeps the noslip solver.
MJWARP_PIN: Final[str] = "f34c6dd52813852d0f1a2ab6d1f4d21ef1f0829f"

_SC = "alloy._vendor.newton._src.solvers.coupled.solver_coupled"
_SCP = "alloy._vendor.newton._src.solvers.coupled.solver_coupled_proxy"

# SCR-WP5-9: MuJoCo Warp kernel factories Alloy's ordered SolverMuJoCo route (alloy/precision/mjw_route.py) replaces
# at launch or lists in its SAFE table; every one carries MuJoCo Warp's ``deterministic`` switch, which the route's
# coverage scan keys on.
_MJW_ROUTE_FACTORIES: Final[dict[str, tuple[str, ...]]] = {
    "alloy._vendor.mujoco_warp.solver": ("_JTDACJ_sparse", "_update_gradient_grad", "_update_constraint_init_qfrc_constraint_sparse",
                           "_solve_init_jaref_kernel", "_linesearch_jv_fused_kernel", "_update_constraint_efc"),
    "alloy._vendor.mujoco_warp.smooth": ("_subtree_com_acc", "_crb_accumulate", "_cfrc_backward", "_transmission", "_tendon_armature",
                           "_tendon_bias_coef", "_tendon_bias_qfrc", "_qLD_acc"),  # _qLD_acc: SCR-WP5-11 (CG/CHAIN)
    "alloy._vendor.mujoco_warp.constraint": ("_efc_contact_jac_sparse",),
    "alloy._vendor.mujoco_warp.set_const": ("_accumulate_subtreemass",),
    "alloy._vendor.mujoco_warp.passive": ("_gravity_force", "_flex_bending", "_flex_elasticity", "_flex_passive_bend_interp",
                            "_spring_damper_flexedge_passive"),
    "alloy._vendor.mujoco_warp.forward": ("_qfrc_actuator",),
    "alloy._vendor.mujoco_warp.derivative": ("_qderiv_actuator_passive_actuation_sparse",),
    "alloy._vendor.mujoco_warp.sensor": ("_sensor_touch", "_tendon_actuator_force"),
}
# SCR-WP5-9: stage functions ``stage(m, d, ...)`` the route wraps so its substitutions act only inside them
_MJW_ROUTE_STAGES: Final[dict[str, tuple[str, ...]]] = {
    "alloy._vendor.mujoco_warp.solver": ("solve",),
    "alloy._vendor.mujoco_warp.smooth": ("com_pos", "crb", "_rne_cfrc_backward", "_factor_i_sparse",  # last two: SCR-WP5-11
                           "_solve_LD_sparse"),
    "alloy._vendor.mujoco_warp.constraint": ("make_constraint",),
    "alloy._vendor.mujoco_warp.set_const": ("set_const_fixed", "set_const_0"),
    "alloy._vendor.mujoco_warp.passive": ("passive",),  # SCR-WP5-13: scopes _gravity_force to the ordered gravcomp sum
}

PRIVATE_HOOKS: Final[tuple[tuple[str, str], ...]] = (
    # SPEC §3.1 (D10 append, D21 runtime substeps)
    ("SolverCoupledProxy", "_step_coupled"), ("SolverCoupledProxy", "_blend_proxy_feedback"),
    ("SolverCoupledProxy", "_step_proxy"), ("SolverCoupledADMM", "_step_coupled"),
    ("alloy._vendor.mujoco_warp.Callback", "contactfilter"),
    ("SolverCoupled", "_entries"), ("SolverCoupled", "_step_entry"),
    # SCR-WP3-1: L1 speculative rows on appended contacts set their targets in callback.passive
    ("alloy._vendor.mujoco_warp.Callback", "passive"),
    # SCR-WP4-2: Scheduler internals (proxy restarts, endpoint tables, entry records, ADMM loop, state adapters)
    ("SolverCoupled", "_distribute_state"),
    # SCR-JOB2-5 (WP12 / WP4 SchedulerMixin): static shapes reach particle entries as walls; both are overridden
    ("SolverCoupled", "_entry_visible_shapes"), ("SolverCoupled", "_customize_compact_view"),
    # WP4 proxy mass (alloy/coupling/axis_mass.ProxyMassMixin): a body that cannot translate at its centre of mass
    # takes its own mass on the MuJoCo Warp route too (overridden; reads SolverEntry.body_global_to_local)
    ("SolverCoupled", "_eval_effective_body_inertial_properties"),
    # SCR-SC4-4d (WP10 alloy/coupling/stiffness_cap.py): caps a lagged proxy body's contact stiffness by wrapping
    # VBD's body-particle record seeding per instance; the last three are instance attributes
    ("SolverVBD", "_refresh_body_particle_contact_state"), ("SolverVBD", "_particle_contact_adjacency_initialized"),
    ("SolverVBD", "body_particle_contact_material_ke"), ("SolverVBD", "body_particle_contact_penalty_k"),
    ("SolverVBD", "body_particle_contact_material_kd"),                  # the cap scales damping too
    # ... and VBD's own per-record soft-contact force gate, which the cap calls (WP4 review)
    *(("soft_contacts_mesh", n) for n in ("_soft_contact_force_enabled", "_make_soft_contact_filter_data",
                                          "_SoftContactFilterData")),
    # gaps-now #10a (alloy.coupling.record_momentum, WP10; share 2026-11-08 20:16Z): VBD's record force law and the
    # attributes its harvest reads
    ("rigid_vbd_kernels", "_eval_soft_ef_contact"),
    *(("SolverVBD", n) for n in ("_integrates_rigid_bodies", "body_particle_contact_material_mu", "friction_epsilon",
                                 "rigid_soft_contact_use_log_barrier")),          # the two snapshots are listed above
    ("SolverCoupledProxy", "_proxy_mappings"), ("SolverCoupledProxy", "_proxy_particle_mappings"),
    ("_ProxyEntityMapping", "coupling_forces"), ("_ProxyEntityMapping", "proxy_ids_global"),
    ("_ProxyEntityMapping", "proxy_ids_local"), ("_ProxyEntityMapping", "src_ids"), ("_ProxyEntityMapping", "mode"),
    ("_ProxyEntityMapping", "src_name"), ("_ProxyEntityMapping", "dst_name"),
    ("SolverEntry", "substeps"), ("SolverEntry", "state_tmp"), ("SolverEntry", "state_0"), ("SolverEntry", "state_1"),
    ("SolverEntry", "in_place"), ("SolverEntry", "view"), ("SolverEntry", "body_global_to_local"),
    ("SolverEntry", "particle_global_to_local"),
    ("SolverCoupledADMM", "_prepare_admm_iteration_state"), ("SolverCoupledADMM", "_accumulate_admm_forces"),
    ("SolverCoupledADMM", "_apply_admm_force_inputs"), ("SolverCoupledADMM", "_update_admm_dual"),
    ("SolverCoupledADMM", "_admm_begin_step"), ("SolverCoupledADMM", "_refresh_collision_contact_groups"),
    ("SolverVBD", "_coupling_body_q_prev_snapshot"), ("SolverVBD", "_coupling_particle_q_prev_snapshot"),
    ("SolverImplicitMPM", "_last_step_data"),
    # SCR-WP5-5: determinism and overflow reads
    ("SolverMuJoCo", "_deterministic"), ("Contacts", "_reduction_overflow"), ("SolverBase", "_module_options"),
    ("SolverMuJoCo", "_use_mujoco_contacts"),                       # SCR-WP5-7: Newton-contact overflow check
    # SCR-WP9-1: SPH boundary density from Newton's shape SDFs. SCR-WP9-9: its volume branch (mesh / convex) reads
    # texture_sdf_table[shape_sdf_index] unchecked, and a shape without a texture SDF has index -1 (every one on CPU),
    # so callers never pass a -1 index (WP9's boundary queries the mesh BVH instead; test_newton_contract pins both)
    ("soft_contacts_sdf", "eval_shape_sdf"), ("Model", "_shape_sdf_index"), ("Model", "_texture_sdf_data"),
    ("sdf_texture", "TextureSDFData"),
    # SCR-WP10-1: in-place MPM activation (crumble) and MJWarp equality forces (weld breaking)
    ("SolverImplicitMPM", "_mpm_model"),
    ("ImplicitMPMModel", "particle_flags"), ("ImplicitMPMModel", "material_particle_flags"),
    ("ImplicitMPMModel", "material_particle_volume"), ("ImplicitMPMModel", "particle_volume"),
    ("alloy._vendor.mujoco_warp.Data", "efc"), ("alloy._vendor.mujoco_warp.Data", "nefc"),
    ("alloy._vendor.mujoco_warp.Constraint", "force"), ("alloy._vendor.mujoco_warp.Constraint", "type"), ("alloy._vendor.mujoco_warp.Constraint", "id"),
    ("SolverMuJoCo", "mjc_eq_to_newton_eq"),
    # SCR-WP3-4: scoped noslip (alloy.contact.noslip) replaces mujoco_warp's module-level noslip and reads its rows
    ("alloy._vendor.mujoco_warp.solver", "noslip"), ("alloy._vendor.mujoco_warp.solver", "_qcqp"), ("alloy._vendor.mujoco_warp.solver", "_qcqp2"),
    ("alloy._vendor.mujoco_warp.solver", "_qcqp3"), ("alloy._vendor.mujoco_warp.types", "ConstraintType"), ("alloy._vendor.mujoco_warp.types", "MJ_MINVAL"),
    ("alloy._vendor.mujoco_warp.types", "vec5"), ("alloy._vendor.mujoco_warp.types", "mat55"),
    *(("alloy._vendor.mujoco_warp.Constraint", n) for n in ("J", "J_rownnz", "J_rowadr", "J_colind", "D", "aref", "frictionloss",
                                             "Ma")),
    *(("alloy._vendor.mujoco_warp.Contact", n) for n in ("dim", "efc_address", "geom", "friction")),
    *(("alloy._vendor.mujoco_warp.Data", n) for n in ("contact", "qacc_smooth", "qfrc_smooth", "qfrc_constraint", "M",
                                        "solver_niter", "naconmax")),
    *(("alloy._vendor.mujoco_warp.Model", n) for n in ("tree_dofadr", "tree_dofnum", "dof_treeid", "body_treeid", "geom_bodyid",
                                         "M_rownnz", "M_rowadr", "M_colind", "is_sparse")),
    ("alloy._vendor.mujoco_warp.Option", "noslip_iterations"), ("alloy._vendor.mujoco_warp.Option", "noslip_tolerance"),
    # SCR-WP5-8: P3-1 MuJoCo Warp determinism (alloy/precision/mjw_det.py sets Option.deterministic and checks the
    # contact/constraint sort functions, still TODO stubs at the pin)
    ("alloy._vendor.mujoco_warp.collision_driver", "_sort_contacts"), ("alloy._vendor.mujoco_warp.constraint", "_sort_constraints"),
    # SCR-WP8-5 (alloy.core.mjw_fastpath): the Python callers of math.upper_trid_index get a plain-Python twin
    ("alloy._vendor.mujoco_warp.math", "upper_trid_index"), ("alloy._vendor.mujoco_warp.collision_convex", "upper_trid_index"),
    ("alloy._vendor.mujoco_warp.collision_primitive", "upper_trid_index"),
    ("alloy._vendor.mujoco_warp.Option", "deterministic"),
    # SCR-WP5-9: P3-1 round 5 ordered SolverMuJoCo route (mjw_route, mjw_launch, mjw_hessian, mjw_sums): the factories
    # and stages above, plus the helpers and types its replacement kernels copy or call
    *((owner, n) for table in (_MJW_ROUTE_FACTORIES, _MJW_ROUTE_STAGES) for owner, names in table.items()
      for n in names),
    ("alloy._vendor.mujoco_warp.solver", "_elliptic_hessian_entry_from_projections"),
    ("alloy._vendor.mujoco_warp.solver", "_JTDAJ_THREADS_PER_GROUP"), ("alloy._vendor.mujoco_warp.math", "safe_div"),
    ("alloy._vendor.mujoco_warp.solver", "_update_gradient_grad_tiled"),             # SCR-WP5-10: launched as is for grad_dot
    # SCR-WP3-9 (BUL-0011): alloy.contact.mjw_boxbox counts box-box pairs and sets NATIVECCD in the Warp model's
    # disable flags, which collision_driver._narrowphase reads to put box-box pairs on the primitive collider
    ("alloy._vendor.mujoco_warp.Model", "geom_pair_type_count"), ("alloy._vendor.mujoco_warp.Option", "disableflags"),
    # SCR-WP5-11: per-level (i, k, Madr_ki) lists of the sparse L'DL; alloy/precision/mjw_qld.prepare_qld checks layout
    ("alloy._vendor.mujoco_warp.Model", "qLD_updates"),
    # SCR-WP5-11 addendum: the fused sparse L'DL solve, substituted by factory args (no ``deterministic`` switch, so
    # not in _MJW_ROUTE_FACTORIES), the funcs its replacement calls, and the level tables it reads
    ("alloy._vendor.mujoco_warp.smooth", "_solve_LD_sparse_fused"), ("alloy._vendor.mujoco_warp.smooth", "_solve_LD_sparse_copy_fn"),
    ("alloy._vendor.mujoco_warp.smooth", "_solve_LD_sparse_diag_fn"),
    ("alloy._vendor.mujoco_warp.Model", "qLD_all_updates"), ("alloy._vendor.mujoco_warp.Model", "qLD_level_offsets"),
    # SCR-WP5-13: alloy/precision/mjw_gravcomp's ordered per-dof gravcomp sum calls this func
    ("alloy._vendor.mujoco_warp.support", "jac_dof"),
    # Newton pair-row order (WP3 alloy.contact.mjw_pairs; ruling in docs/notes/wp0-scr-log.md): the hook maps MuJoCo
    # pairs back to Newton rows through the geom map and relies on _update_pair_properties copying row
    # world * ppw + j onto pair j (tests/contact/test_mjw_pairs.py pins it); the device pair arrays are read in tests
    ("SolverMuJoCo", "mjc_geom_to_newton_shape"), ("SolverMuJoCo", "_update_pair_properties"),
    *(("alloy._vendor.mujoco_warp.Model", n) for n in ("pair_friction", "pair_solref", "pair_solreffriction", "pair_solimp",
                                         "pair_margin", "pair_gap")),
    *(("alloy._vendor.mujoco_warp.types", n) for n in ("ConstraintState", "ConeType", "SolverType", "IntegratorType", "EnableBit",
                                         "vec6i", "vec10", "vec16")),
    # SCR-WP5-12 (BUL-0016): alloy/precision/mjw_chol.py clamps Newton Cholesky pivots like mju_cholFactor. It
    # replaces the module-level solve that _update_gradient(_incremental) call by global name, copies the tile kernel
    # body, runs the blocked variants unchanged, and reads the context; its own tripwire is mjw_pin.GUARD
    *(("alloy._vendor.mujoco_warp.solver", n) for n in ("_cholesky_factorize_solve", "_update_gradient_cholesky",
                                          "_update_gradient_cholesky_blocked",
                                          "_update_gradient_cholesky_blocked_skip_unchanged")),
    ("alloy._vendor.mujoco_warp.block_cholesky", "solve_search_sums"), ("alloy._vendor.mujoco_warp.Model", "block_dim"),
    ("alloy._vendor.mujoco_warp.BlockDim", "update_gradient_cholesky"),
    *(("alloy._vendor.mujoco_warp.SolverContext", n) for n in ("grad", "h", "hfactor", "search", "search_dot", "newton_decrement",
                                                 "done", "state_changed_count")),
    # SCR-SC10-3 (WP3 coordinate carry, default-on): the carry runs between SolverMuJoCo._update_newton_state's
    # coordinate conversion and its forward kinematics by replacing the module-level eval_fk it calls by global name
    ("solver_mujoco", "eval_fk"), ("SolverMuJoCo", "_update_newton_state"),
    # DC5 sf19 (vessel water as body mass; WP15 review): alloy/plugins/host.py pushes per-world body mass, COM and
    # inertia into MuJoCo Warp each exchange through this two-kernel update, graph-safe. set_const does not rerun, so
    # invweight0 / subtreemass keep their build values. On use_mujoco_cpu it does not reach mj_model (notify does).
    ("SolverMuJoCo", "_update_model_inertial_properties"),
    # ... and on the MuJoCo C paths, MuJoCo's midphase boxes (bvh_aabb) live in each body's inertial frame: a moved
    # centre needs them re-expressed from the compiled boxes and frame, as Newton's notify does for its template
    # (_refit_body_bvh_to_inertial_frame). Instance attributes, copied at the C model's build.
    *(("SolverMuJoCo", n) for n in ("_canonical_bvh_aabb", "_canonical_body_ipos", "_canonical_body_iquat")),
)

# Where each owner name in PRIVATE_HOOKS lives: (module, attribute); attribute None = the module itself.
_HOOK_OWNERS: Final[dict[str, tuple[str, str | None]]] = {
    "SolverCoupledProxy": ("alloy._vendor.newton.solvers.experimental.coupled", "SolverCoupledProxy"),
    "SolverCoupledADMM": ("alloy._vendor.newton.solvers.experimental.coupled", "SolverCoupledADMM"),
    "SolverCoupled": ("alloy._vendor.newton.solvers.experimental.coupled", "SolverCoupled"),
    "SolverEntry": (_SC, "SolverEntry"),
    "_ProxyEntityMapping": (_SCP, "_ProxyEntityMapping"),
    "SolverBase": ("alloy._vendor.newton.solvers", "SolverBase"),
    "SolverMuJoCo": ("alloy._vendor.newton.solvers", "SolverMuJoCo"),
    "solver_mujoco": ("alloy._vendor.newton._src.solvers.mujoco.solver_mujoco", None),
    "SolverVBD": ("alloy._vendor.newton.solvers", "SolverVBD"),
    "SolverImplicitMPM": ("alloy._vendor.newton.solvers", "SolverImplicitMPM"),
    "ImplicitMPMModel": ("alloy._vendor.newton._src.solvers.implicit_mpm.implicit_mpm_model", "ImplicitMPMModel"),
    "Model": ("alloy._vendor.newton", "Model"),
    "Contacts": ("alloy._vendor.newton", "Contacts"),
    "soft_contacts_sdf": ("alloy._vendor.newton._src.geometry.soft_contacts_sdf", None),
    "soft_contacts_mesh": ("alloy._vendor.newton._src.geometry.soft_contacts_mesh", None),
    "rigid_vbd_kernels": ("alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels", None),
    "sdf_texture": ("alloy._vendor.newton._src.geometry.sdf_texture", None),
    "alloy._vendor.mujoco_warp.Callback": ("alloy._vendor.mujoco_warp", "Callback"),
    "alloy._vendor.mujoco_warp.Data": ("alloy._vendor.mujoco_warp", "Data"),
    "alloy._vendor.mujoco_warp.Constraint": ("alloy._vendor.mujoco_warp._src.types", "Constraint"),
    "alloy._vendor.mujoco_warp.collision_driver": ("alloy._vendor.mujoco_warp._src.collision_driver", None),
    "alloy._vendor.mujoco_warp.collision_convex": ("alloy._vendor.mujoco_warp._src.collision_convex", None),
    "alloy._vendor.mujoco_warp.collision_primitive": ("alloy._vendor.mujoco_warp._src.collision_primitive", None),
    "alloy._vendor.mujoco_warp.constraint": ("alloy._vendor.mujoco_warp._src.constraint", None),
    "alloy._vendor.mujoco_warp.Contact": ("alloy._vendor.mujoco_warp._src.types", "Contact"),
    "alloy._vendor.mujoco_warp.Model": ("alloy._vendor.mujoco_warp", "Model"),
    "alloy._vendor.mujoco_warp.Option": ("alloy._vendor.mujoco_warp._src.types", "Option"),
    "alloy._vendor.mujoco_warp.solver": ("alloy._vendor.mujoco_warp._src.solver", None),
    "alloy._vendor.mujoco_warp.types": ("alloy._vendor.mujoco_warp._src.types", None),
    "alloy._vendor.mujoco_warp.smooth": ("alloy._vendor.mujoco_warp._src.smooth", None),
    "alloy._vendor.mujoco_warp.set_const": ("alloy._vendor.mujoco_warp._src.set_const", None),
    "alloy._vendor.mujoco_warp.forward": ("alloy._vendor.mujoco_warp._src.forward", None),
    "alloy._vendor.mujoco_warp.derivative": ("alloy._vendor.mujoco_warp._src.derivative", None),
    "alloy._vendor.mujoco_warp.passive": ("alloy._vendor.mujoco_warp._src.passive", None),
    "alloy._vendor.mujoco_warp.support": ("alloy._vendor.mujoco_warp._src.support", None),
    "alloy._vendor.mujoco_warp.sensor": ("alloy._vendor.mujoco_warp._src.sensor", None),
    "alloy._vendor.mujoco_warp.math": ("alloy._vendor.mujoco_warp._src.math", None),
    "alloy._vendor.mujoco_warp.block_cholesky": ("alloy._vendor.mujoco_warp._src.block_cholesky", None),
    "alloy._vendor.mujoco_warp.SolverContext": ("alloy._vendor.mujoco_warp._src.types", "SolverContext"),
    "alloy._vendor.mujoco_warp.BlockDim": ("alloy._vendor.mujoco_warp._src.types", "BlockDim"),
}

__all__ = [
    "MJWARP_PIN", "NEWTON_PIN", "PRIVATE_HOOKS", "CouplingInterface", "ModelView", "SolverCoupled",
    "SolverCoupledADMM", "SolverCoupledProxy", "check_newton_pin", "enable_mujoco_noslip", "installed_commit",
    "installed_mujoco_warp_commit", "installed_newton_commit", "missing_private_hooks", "mujoco_warp_has_noslip",
    "solver_entry_class",
]


def installed_newton_commit() -> str:
    """The commit Alloy's Newton runs at: the vendored copy's pin (alloy/_vendor/VENDORED.md)."""
    return installed_commit("newton")


def installed_mujoco_warp_commit() -> str:
    """The commit Alloy's MuJoCo Warp runs at: the vendored copy's pin."""
    return installed_commit("mujoco-warp")


# Newton and MuJoCo Warp are vendored (alloy/_vendor): their commits are the pins, whatever a user has installed.
_VENDORED_COMMITS: Final[dict[str, str]] = {"newton": NEWTON_PIN, "mujoco-warp": MJWARP_PIN, "mujoco_warp": MJWARP_PIN}


def installed_commit(dist: str) -> str:
    """The commit ``dist`` runs at: the pin for the vendored engines, else the git commit pip installed it from
    (``direct_url.json``), or "unknown"."""
    if dist in _VENDORED_COMMITS:
        return _VENDORED_COMMITS[dist]
    try:
        text = importlib.metadata.distribution(dist).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    if not text:
        return "unknown"
    try:
        commit = json.loads(text).get("vcs_info", {}).get("commit_id", "")
    except json.JSONDecodeError:
        return "unknown"
    return commit or "unknown"


def check_newton_pin(strict: bool = False) -> str:
    """Return the installed Newton commit. A mismatch with ``NEWTON_PIN`` (or an unknown commit) logs a warning,
    or raises ``NewtonPinError`` when ``strict``."""
    commit = installed_newton_commit()
    if commit != NEWTON_PIN:
        msg = f"installed newton commit {commit} != pinned {NEWTON_PIN} (SPEC header; a bump is a coordinator task)"
        if strict:
            raise NewtonPinError(msg)
        log.warning("%s", msg)
    return commit


def _resolve_owner(owner: str) -> object:
    """The class (or module) named by a PRIVATE_HOOKS owner."""
    if owner not in _HOOK_OWNERS:
        raise NewtonPinError(f"PRIVATE_HOOKS owner {owner!r} has no known location")
    module_name, attr = _HOOK_OWNERS[owner]
    module = importlib.import_module(module_name)
    return module if attr is None else getattr(module, attr)


def _has_member(cls: object, name: str) -> bool:
    """Class/module attribute, dataclass field, or an instance attribute assigned as ``self.<name> =`` in the class
    (or a base) source."""
    if hasattr(cls, name):
        return True
    if not isinstance(cls, type):
        return False
    if dataclasses.is_dataclass(cls) and name in {f.name for f in dataclasses.fields(cls)}:
        return True
    pattern = re.compile(rf"\bself\.{re.escape(name)}\s*(?::[^=\n]+)?=")
    for klass in cls.__mro__:
        try:
            source = inspect.getsource(klass)
        except (OSError, TypeError):
            continue
        if pattern.search(source):
            return True
    return False


def missing_private_hooks() -> list[tuple[str, str]]:
    """Every ``PRIVATE_HOOKS`` entry the installed Newton / MuJoCo Warp no longer provides (empty when all exist)."""
    missing = []
    for owner, name in PRIVATE_HOOKS:
        try:
            cls = _resolve_owner(owner)
        except (ImportError, AttributeError):
            missing.append((owner, name))
            continue
        if not _has_member(cls, name):
            missing.append((owner, name))
    return missing


def solver_entry_class() -> type:
    """Newton's private per-entry record whose ``substeps`` the Scheduler sets at runtime (D21)."""
    module = importlib.import_module("alloy._vendor.newton._src.solvers.coupled.solver_coupled")
    cls = getattr(module, "SolverEntry", None)
    if cls is None:
        raise NewtonPinError("newton._src.solvers.coupled.solver_coupled.SolverEntry is gone (D21 hook)")
    return cls


def mujoco_warp_has_noslip() -> bool:
    """True when the installed mujoco-warp has the noslip solver (``Option.noslip_iterations``; not in 3.16.0)."""
    try:
        from alloy._vendor import mujoco_warp
    except ImportError:
        return False
    return "noslip_iterations" in getattr(mujoco_warp.Option, "__annotations__", {})


def enable_mujoco_noslip(solver: object, iterations: int, tolerance: float = 1e-6) -> None:
    """Set MuJoCo's noslip post-solve on a built Newton ``SolverMuJoCo``; ``iterations=0`` turns it off.

    Newton ac7f7830 has no noslip option: it builds its MjSpec without one
    (newton@ac7f7830:newton/_src/solvers/mujoco/solver_mujoco.py:6885-6921). This sets ``opt.noslip_iterations`` and
    ``opt.noslip_tolerance`` on ``solver.mj_model`` and, on the Warp path, on ``solver.mjw_model``, which MuJoCo Warp
    reads on every call (mujoco_warp@86e2fae0:mujoco_warp/_src/solver.py:4037-4050, 5508). Call it after
    construction and before the first step or CUDA graph capture. A recorded scene's own values are in its MJCF
    ``<option noslip_iterations noslip_tolerance>``.
    """
    import mujoco

    if iterations < 0 or not tolerance > 0.0:
        raise SolverOptionError(f"noslip needs iterations >= 0 and tolerance > 0, got {iterations}, {tolerance}")
    opt = solver.mj_model.opt  # type: ignore[attr-defined]
    if iterations > 0 and int(opt.integrator) == int(mujoco.mjtIntegrator.mjINT_DISCRETE):
        raise SolverOptionError("MuJoCo Warp has no noslip for the discrete integrator (SPEC D7 integrator table)")
    if not getattr(solver, "use_mujoco_cpu", False) and not mujoco_warp_has_noslip():
        raise NewtonPinError(f"installed mujoco-warp has no noslip solver; pin is mujoco-warp@{MJWARP_PIN}")
    opt.noslip_iterations = int(iterations)
    opt.noslip_tolerance = float(tolerance)
    if getattr(solver, "use_mujoco_cpu", False):
        return
    mjw_opt = solver.mjw_model.opt  # type: ignore[attr-defined]
    mjw_opt.noslip_iterations = int(iterations)
    mjw_opt.noslip_tolerance.fill_(float(tolerance))
    log.debug("mujoco noslip: %d iterations, tolerance %g", iterations, tolerance)
