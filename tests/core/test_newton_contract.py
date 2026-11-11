# SPDX-License-Identifier: Apache-2.0
"""Newton contract (SPEC §6.6): the pin, the signatures Alloy calls, PRIVATE_HOOKS, friction-as-scale, D21 substeps.
A Newton bump must pass this file first."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re

import pytest

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase, SolverMuJoCo

from alloy.core import newton_compat as nc
from alloy.testing.fakes import FakeSolver
from alloy.testing.scenes import box_on_plane


def _params(fn) -> set[str]:
    return set(inspect.signature(fn).parameters)


def test_pinned_commit(newton_pin: str) -> None:
    assert newton_pin == nc.NEWTON_PIN
    assert newton.__version__.startswith("1.8")


def test_private_hooks_present() -> None:
    assert nc.missing_private_hooks() == []
    assert "substeps" in {f.name for f in dataclasses.fields(nc.solver_entry_class())}
    assert len(set(nc.PRIVATE_HOOKS)) == len(nc.PRIVATE_HOOKS)


def test_eval_shape_sdf_signature() -> None:
    """SCR-WP9-1: eval_shape_sdf(geo, scale, x_local, shape_sdf_index, texture_sdf_table) -> (phi_lower, phi, grad)
    (newton@ac7f7830:newton/_src/geometry/soft_contacts_sdf.py:248)."""
    fn = importlib.import_module("alloy._vendor.newton._src.geometry.soft_contacts_sdf").eval_shape_sdf
    py = getattr(fn, "func", fn)
    assert list(inspect.signature(py).parameters) == ["geo", "scale", "x_local", "shape_sdf_index",
                                                      "texture_sdf_table"]


def test_eval_shape_sdf_needs_a_texture_sdf_for_meshes() -> None:
    """SCR-WP9-9: the volume branch indexes the texture table with no -1 check, and a CPU mesh has index -1, so a
    caller (SPH boundary.py) must not pass a ``_shape_sdf_index`` < 0 (WP9 queries the mesh BVH for those shapes).
    If Newton adds the check, this fails and the caller's guard can go."""
    fn = importlib.import_module("alloy._vendor.newton._src.geometry.soft_contacts_sdf").eval_shape_sdf
    src = inspect.getsource(getattr(fn, "func", fn))
    assert "tex = texture_sdf_table[shape_sdf_index]" in src and "shape_sdf_index < 0" not in src
    b = newton.ModelBuilder()
    body = b.add_body()
    verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    b.add_shape_mesh(body, mesh=newton.Mesh(verts, [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]))
    model = b.finalize(device="cpu")
    assert int(model.shape_type.numpy()[0]) == int(newton.GeoType.MESH)
    assert int(model._shape_sdf_index.numpy()[0]) == -1


def test_coupled_view_hooks_for_static_walls() -> None:
    """SCR-JOB2-5: WP4's SchedulerMixin overrides ``_entry_visible_shapes(cfg, visible_bodies) -> set[int]`` and
    ``_customize_compact_view(view)``; Newton's ``_build_entries`` calls the latter after compaction (or the full-layout
    fallback) and before ``cfg.configure_view``, and the shape-visibility passes call the former."""
    sc = nc.SolverCoupled
    assert list(inspect.signature(sc._entry_visible_shapes).parameters) == ["self", "cfg", "visible_bodies"]
    assert list(inspect.signature(sc._customize_compact_view).parameters) == ["self", "view"]
    build = inspect.getsource(sc._build_entries)
    assert 0 <= build.find("self._customize_compact_view(") < build.find("cfg.configure_view(")
    users = [n for n, f in inspect.getmembers(sc, inspect.isfunction) if n != "_entry_visible_shapes"
             and "self._entry_visible_shapes(cfg, visible_bodies)" in inspect.getsource(f)]
    assert users, "nothing calls _entry_visible_shapes(cfg, visible_bodies) any more"

def test_vbd_body_particle_seeding_hook() -> None:
    """SCR-SC4-4d: WP10's stiffness cap wraps ``SolverVBD._refresh_body_particle_contact_state(contacts, refresh)``
    per instance and reads the instance attributes it fills (PRIVATE_HOOKS checks they are still assigned)."""
    from alloy._vendor.newton.solvers import SolverVBD

    params = list(inspect.signature(SolverVBD._refresh_body_particle_contact_state).parameters)
    assert params == ["self", "contacts", "refresh"]
    mesh = importlib.import_module("alloy._vendor.newton._src.geometry.soft_contacts_mesh")
    gate = mesh._soft_contact_force_enabled
    assert list(inspect.signature(getattr(gate, "func", gate)).parameters) == ["contact_index", "data"]
    assert list(inspect.signature(mesh._make_soft_contact_filter_data).parameters) == [
        "model", "contacts", "particle_q", "body_q"]

@pytest.mark.parametrize(("fn", "required"), [
    (newton.CollisionPipeline.__init__, {"model", "broad_phase", "narrow_phase", "deterministic", "contact_matching",
                                         "rigid_contact_max", "shape_pairs_max", "publish_capacity",
                                         "speculative_contact_gap_max"}),
    (newton.CollisionPipeline.collide, {"state", "contacts", "dt"}),
    (newton.Contacts.__init__, {"rigid_contact_max", "soft_contact_max", "per_contact_shape_properties",
                                "requested_attributes", "contact_matching", "device"}),
    (SolverBase.step, {"state_in", "state_out", "control", "contacts", "dt", "observables"}),
    (SolverBase.validate_observables, {"observables", "contacts"}),
    (SolverBase.reset, {"state", "world_mask", "flags"}),
    (SolverBase.notify_model_changed, {"flags"}),
    (nc.SolverCoupledProxy.__init__, {"model", "entries", "coupling"}),
    (nc.SolverCoupledADMM.__init__, {"model", "entries", "coupling"}),
    (nc.SolverCoupled._step_entry, {"entry", "control", "contacts", "dt"}),
    (nc.SolverCoupledProxy._step_coupled, {"state_in", "state_out", "control", "contacts", "dt"}),
    (SolverMuJoCo.__init__, {"enable_sleeping", "use_speculative_contacts", "use_mujoco_contacts", "njmax",
                             "nconmax", "integrator", "cone", "impratio"}),
    (newton.ModelBuilder.replicate, {"builder", "copy_count", "spacing"}),
    (newton.ModelBuilder.add_custom_attribute, {"attribute"}),
])
def test_signatures(fn, required: set[str]) -> None:
    missing = required - _params(fn)
    assert not missing, f"{fn.__qualname__} lost {missing}"


def test_entry_and_proxy_fields() -> None:
    entry = {f.name for f in dataclasses.fields(nc.SolverCoupled.Entry)}
    assert {"name", "solver", "bodies", "particles", "joints", "shapes", "substeps", "in_place"} <= entry
    proxy = {f.name for f in dataclasses.fields(nc.SolverCoupledProxy.Proxy)}
    assert {"source", "destination", "bodies", "particles", "mode", "mass_scale", "collision_pipeline"} <= proxy
    attr = {f.name for f in dataclasses.fields(newton.ModelBuilder.CustomAttribute)}
    assert {"name", "dtype", "frequency", "namespace", "default"} <= attr


def test_rigid_contact_friction_is_a_scale() -> None:
    """D13: the pipeline writes a friction *scale*; MuJoCo multiplies its mixed geom friction by it."""
    collide = inspect.getsource(importlib.import_module("alloy._vendor.newton._src.sim.collide"))
    assert re.search(r"out_friction\[index\]\s*=\s*contact_data\.contact_friction_scale", collide)
    kernels = inspect.getsource(importlib.import_module("alloy._vendor.newton._src.solvers.mujoco.kernels"))
    assert re.search(r"friction_scale\s*=\s*rigid_contact_friction\[tid\]", kernels)
    assert re.search(r"friction\[0\]\s*\*\s*friction_scale", kernels)


class _Counting(FakeSolver):
    calls = 0

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):  # noqa: D401
        type(self).calls += 1
        super().step(state_in, state_out, control, contacts, dt, observables=observables)


def test_runtime_substeps_take_effect() -> None:
    """D21: an entry built at substeps = max rung keeps its scratch state; changing the private
    SolverEntry.substeps between steps changes the number of sub-solver steps and never raises."""
    model = box_on_plane(2, gap=0.2).model
    entry = nc.SolverCoupled.Entry(name="fake", solver=lambda view: _Counting(view),
                                   bodies=list(range(model.body_count)), joints=list(range(model.joint_count)),
                                   substeps=4)
    coupled = nc.SolverCoupled(model, [entry])
    s0, s1, control = model.state(), model.state(), model.control()
    _Counting.calls = 0
    coupled.step(s0, s1, control, None, 1.25e-3)
    assert _Counting.calls == 4
    coupled._entries["fake"].substeps = 2
    coupled.step(s1, s0, control, None, 1.25e-3)
    assert _Counting.calls == 6
    coupled._entries["fake"].substeps = 1
    coupled.step(s0, s1, control, None, 1.25e-3)
    assert _Counting.calls == 7


def test_observables_container_check() -> None:
    """solver.py:909: validate_observables rejects a container of another class (why PrecisionObservables is for
    Alloy plug-ins only)."""
    model = box_on_plane(1).model
    fake = FakeSolver(model)
    with pytest.raises(TypeError):
        fake.validate_observables(SolverBase.Observables())


def test_mujoco_warp_noslip_dispatch() -> None:
    """SCR-WP3-4: ``solve`` calls the module-level ``noslip(m, d)`` by global name (mujoco_warp@86e2fae0:
    mujoco_warp/_src/solver.py:4037-4050), so rebinding ``solver.noslip`` reroutes it; the cone projections keep the
    argument lists WP3's kernels call them with."""
    solver = importlib.import_module("alloy._vendor.mujoco_warp._src.solver")
    assert "noslip" in inspect.unwrap(solver.solve).__code__.co_names
    assert list(inspect.signature(solver.noslip).parameters) == ["m", "d"]
    for name, params in (("_qcqp", ["Ain", "bin", "d", "r", "n"]), ("_qcqp2", ["Ain", "bin", "d", "r"]),
                         ("_qcqp3", ["Ain", "bin", "d", "r"])):
        fn = getattr(solver, name)
        assert list(inspect.signature(getattr(fn, "func", fn)).parameters) == params, name


def test_mujoco_warp_cholesky_guard_dispatch() -> None:
    """SCR-WP5-12 (BUL-0016): ``_update_gradient`` and ``_update_gradient_incremental`` call the module-level
    ``_cholesky_factorize_solve(m, d, ctx, skip_unchanged, skip_noflip)`` by global name, so alloy/precision/mjw_chol.py
    can rebind it; the tile kernel factory keeps the arguments the guard's copy takes."""
    solver = importlib.import_module("alloy._vendor.mujoco_warp._src.solver")
    for caller in ("_update_gradient", "_update_gradient_incremental"):
        assert "_cholesky_factorize_solve" in inspect.unwrap(getattr(solver, caller)).__code__.co_names, caller
    assert list(inspect.signature(solver._cholesky_factorize_solve).parameters) == [
        "m", "d", "ctx", "skip_unchanged", "skip_noflip"]
    assert list(inspect.signature(solver._update_gradient_cholesky).parameters) == ["tile_size", "skip_noflip"]

def test_mujoco_warp_route_shapes() -> None:
    """SCR-WP5-9: the ordered SolverMuJoCo route finds race-prone kernels by the ``deterministic`` parameter on their
    factory and turns its substitutions on per model inside ``stage(m, d, ...)``; a pin bump that drops either breaks
    the route's coverage scan or its scoping."""
    for owner, names in nc._MJW_ROUTE_FACTORIES.items():
        module = nc._resolve_owner(owner)
        for name in names:
            assert "deterministic" in inspect.signature(getattr(module, name)).parameters, f"{owner}.{name}"
    for owner, names in nc._MJW_ROUTE_STAGES.items():
        module = nc._resolve_owner(owner)
        for name in names:
            assert list(inspect.signature(getattr(module, name)).parameters)[:2] == ["m", "d"], f"{owner}.{name}"


def test_mjw_jac_dof_signature() -> None:
    """SCR-WP5-13: WP5's ordered gravcomp kernel calls ``support.jac_dof`` with the pin's arguments."""
    func = nc._resolve_owner("alloy._vendor.mujoco_warp.support").jac_dof
    assert list(inspect.signature(getattr(func, "func", func)).parameters) == [
        "body_parentid", "body_rootid", "dof_bodyid", "body_isdofancestor", "subtree_com_in", "cdof_in", "point",
        "bodyid", "dofid", "worldid"]
