# SPDX-License-Identifier: Apache-2.0
"""SCR-OSS22-4: a cloth card's features.self_contact turns on VBD particle self-contact (``entries.vbd_self_contact``),
interaction distance = the sheet thickness h (mid-surfaces of two touching sheets), detection one more h. Law (CPU): a
towel folded in half, dropped 1 mm onto a table, rests as two layers. The layers' gap stays above 0.5 h, and the late
drift and end speed are within alloy.check's rest limits. Without self-contact the top layer falls into the bottom one
(the control). Default off: a cloth without the feature leaves the VBD entry's kwargs unchanged."""

from __future__ import annotations

import dataclasses
import hashlib
import logging

import numpy as np
import pytest

import alloy.scene.entries as entries
from alloy.cards.types import Budget, Failure, MeshSource, ObjectCard, ResolvedObject
from alloy.check.motion import DRIFT_WARN_M, SPEED_WARN_MPS
from alloy.core.ids import digest_of
from alloy.integration.wiring import build_simulation
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.scene.vbd_contact import vbd_self_contact
from alloy.solvers.vbd.reach import SELF_CONTACT_SPEED, SPEED_KEY
from alloy.sim import Simulation
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.topology.types import BondKind, TopologyBudget

H = 2.0e-3            # towel thickness [m]
L, N = 0.16, 8        # folded towel: L x L, N x N cells per layer (CPU: ~1.2 s per period)
TOP = 0.02            # S.table() top [m]
PERIODS = 50          # 1 s at the 20 ms control period (the drift is judged over the second 0.5 s)


def _folded_obj_text() -> str:
    """Layer A at z = 0 and layer B at z = H over [0, L]^2, joined at x = L by one strip of height H (the fold)."""
    xs = np.linspace(0.0, L, N + 1)
    verts = [(x, y, z) for z in (0.0, H) for x in xs for y in xs]
    idx = lambda layer, i, j: layer * (N + 1) ** 2 + i * (N + 1) + j          # noqa: E731
    faces = []
    for layer in (0, 1):
        for i in range(N):
            for j in range(N):
                a, b, c, d = idx(layer, i, j), idx(layer, i + 1, j), idx(layer, i + 1, j + 1), idx(layer, i, j + 1)
                faces += [(a, b, c), (a, c, d)]
    for j in range(N):                                                     # the fold strip at x = L
        a, b, c, d = idx(0, N, j), idx(1, N, j), idx(1, N, j + 1), idx(0, N, j + 1)
        faces += [(a, b, c), (a, c, d)]
    lines = [f"v {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in verts]
    return "\n".join(lines + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces])


def _towel(tmp_path, self_contact: bool) -> ResolvedObject:
    path = tmp_path / "towel.obj"
    path.write_text(_folded_obj_text() + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    mat = T.material("cotton", **T.COTTON, beta=1e-3)
    card = ObjectCard("towel", MeshSource(str(path), sha), mat.name,
                      features=T.features(thickness=H, self_contact=self_contact))
    return ResolvedObject(card, mat, {}, digest_of((card, mat)), (L, L, H), {}, ())


class _Lib(S.TestLibrary):
    __test__ = False

    def resolve(self, card, placement=None):
        return self.towel if card.name == "towel" else super().resolve(card, placement)


def _run(tmp_path, self_contact: bool):
    towel, table = _towel(tmp_path, self_contact), S.table()
    card, _ = S.scene([table, towel], [S.place("table"), S.place("towel", -0.5 * L, -0.5 * L, TOP + 0.5 * H + 1e-3)],
                      worlds=1)
    lib = _Lib(materials={o.material.name: o.material for o in (table, towel)},
               objects={o.card.name: o.card for o in (table, towel)}, pairs={})
    lib.towel = towel
    built = SceneAssembler(lib).assemble(card, "cpu")
    assert built.handles["towel"].decision.route == Route.CLOTH_VBD
    (vbd,) = [e for e in built.entries if e.capability == "vbd"]
    sim = Simulation(built)
    half = None
    for k in range(PERIODS):
        sim.step_period()
        if k == PERIODS // 2:
            half = sim.state.particle_q.numpy().copy()
    return dict(vbd.solver_kwargs), half, sim.state.particle_q.numpy(), sim.state.particle_qd.numpy()


def _layer_gap(q: np.ndarray) -> float:
    """Smallest vertical gap from a top-layer vertex to the bottom-layer vertices within one cell (fold excluded)."""
    n = (N + 1) ** 2
    bottom, top = q[:n].reshape(N + 1, N + 1, 3), q[n:2 * n].reshape(N + 1, N + 1, 3)
    cell = L / N
    gaps = []
    for i in range(N - 1):                                   # the last columns are the fold
        for j in range(N + 1):
            p = top[i, j]
            near = bottom[np.linalg.norm(bottom[..., :2] - p[:2], axis=-1) <= cell]
            gaps.append(p[2] - near[:, 2].max() if len(near) else np.inf)
    return float(np.min(gaps))


def test_default_off_leaves_the_entry_unchanged(tmp_path):
    towel = _towel(tmp_path, False)
    t = _entry_obj(towel, Route.CLOTH_VBD)
    assert vbd_self_contact([t]) == {}
    t.obj = _towel(tmp_path, True)
    assert vbd_self_contact([t]) == {"particle_enable_self_contact": True, "particle_self_contact_margin": H,
                                     SPEED_KEY: SELF_CONTACT_SPEED}         # no pre-split object: nothing excluded


def _entry_obj(obj, route):
    return type("T", (), {"obj": obj, "name": obj.card.name, "decision": type("D", (), {"route": route})()})()


def test_entry_takes_the_thinnest_object_and_names_thicker_ones(tmp_path, caplog):
    towel = _towel(tmp_path, True)
    thin = dataclasses.replace(towel, card=dataclasses.replace(towel.card, name="tissue",
                                                                features=T.features(thickness=0.4 * H)))
    sponge = dataclasses.replace(towel, card=dataclasses.replace(towel.card, name="sponge"))
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        vbd_self_contact([_entry_obj(towel, Route.CLOTH_VBD)])
        assert not caplog.records
        kw = vbd_self_contact([_entry_obj(towel, Route.CLOTH_VBD), _entry_obj(thin, Route.CLOTH_VBD),
                               _entry_obj(sponge, Route.SOFT_VBD)], radii={"sponge": 2.0 * H})
    assert kw["particle_self_contact_margin"] == pytest.approx(0.4 * H)     # the thinnest object's thickness
    (rec,) = caplog.records
    assert "sponge, towel" in rec.getMessage()                              # more than 2x thicker


def test_a_mesh_edge_under_the_contact_distance_warns(tmp_path, caplog):
    """WP2 2026-11-08: a cloth mesh with edges under the sheet thickness starts in self contact and blows apart (a
    jittered-hex disc, 1.1 mm edges at h 2 mm: 89 % strain in 0.5 s). The build names the sheet and the edge. The
    folded towel's fold strip (edges exactly h) does not warn."""
    path = tmp_path / "sliver.obj"
    path.write_text("v 0 0 0\nv 0.001 0 0\nv 0.02 0.02 0\nv 0 0.02 0\nf 1 2 3\nf 1 3 4\n")      # edge 1-2: 1 mm
    towel = _towel(tmp_path, True)
    sliver = dataclasses.replace(towel, card=dataclasses.replace(
        towel.card, name="sliver", shape=MeshSource(str(path), hashlib.sha256(path.read_bytes()).hexdigest())))
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        vbd_self_contact([_entry_obj(towel, Route.CLOTH_VBD)])
        assert not [r for r in caplog.records if r.name == "alloy.api"]
        vbd_self_contact([_entry_obj(sliver, Route.CLOTH_VBD)])
    (rec,) = [r for r in caplog.records if r.name == "alloy.api"]
    assert "sliver: a mesh edge is 1 mm, under the self-contact distance 2 mm" in rec.getMessage()


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)       # cloth's default 40 iterations (cloth_accuracy, 2026-11-08): 281 s on a loaded VM
def test_folded_towel_rests_as_two_layers(tmp_path):
    kw, half, q, qd = _run(tmp_path, True)
    assert kw["particle_enable_self_contact"] and kw["particle_self_contact_margin"] == H
    gap = _layer_gap(q)
    drift = float(np.linalg.norm(q - half, axis=1).max())
    speed = float(np.linalg.norm(qd, axis=1).max())
    S.record("SCR-OSS22-4.folded_towel", gap / H, "pass" if gap > 0.5 * H and drift < DRIFT_WARN_M else "fail",
             thickness=H, min_gap_m=gap, late_drift_m=drift, end_speed=speed, min_z=float(q[:, 2].min()))
    assert gap > 0.5 * H, f"layers {gap * 1e3:.3f} mm apart (h {H * 1e3} mm)"
    assert drift < DRIFT_WARN_M and speed < SPEED_WARN_MPS
    assert q[:, 2].min() > TOP                                            # nothing through the table


@pytest.mark.compile_heavy
def test_without_self_contact_the_layers_cross(tmp_path):
    kw, _, q, _ = _run(tmp_path, False)
    assert "particle_enable_self_contact" not in kw
    assert _layer_gap(q) < 0.25 * H


def _tearable_towel(self_contact: bool) -> ResolvedObject:
    mat = dataclasses.replace(T.material("cotton", **T.COTTON, beta=1e-3),
                              failure=Failure(fracture_strain=T.q(0.3), tear_energy=T.q(500.0)))
    return T.box("towel", mat, 0.05, 0.05, 0.5 * H, budget=Budget(particles=121),
                 topology=TopologyBudget(spare_vertices=400),
                 features=T.features(thickness=H, self_contact=self_contact))


def _presplit_rest(self_contact: bool, periods: int = 25) -> tuple[float, int, int]:
    """A pre-split (tearable) 10 x 10 cm towel dropped 1 mm onto a table, 0.5 s with the topology controller: max
    seam twin gap, intact bonds, all bonds."""
    card, lib = S.scene([S.table(), _tearable_towel(self_contact)],
                        [S.place("table"), S.place("towel", z=TOP + 0.5 * H + 1e-3)], worlds=1)
    built = SceneAssembler(lib, Router()).assemble(card, "cpu")
    sim = build_simulation(built, "standard", seed=0)
    for _ in range(periods):
        sim.step_period()
    bonds, q = built.bonds, sim.state.particle_q.numpy()
    twin = bonds.kind.numpy() == int(BondKind.CLOTH_EDGE)
    gap = np.linalg.norm(q[bonds.a.numpy()[twin]] - q[bonds.b.numpy()[twin]], axis=1).max()
    return float(gap), int(bonds.intact.numpy().sum()), bonds.count


@pytest.mark.compile_heavy
def test_presplit_seams_stay_closed_with_self_contact():
    """WP11 review: seam twins start coincident, joined only by bond springs. The rest-shape exclusion (0.5 h) keeps
    contact off them, so the seams stay as closed as without self-contact and nothing tears at rest."""
    gap_off, intact_off, n = _presplit_rest(False)
    gap, intact, _ = _presplit_rest(True)
    S.record("SCR-OSS22-4.presplit_seams", gap / H, "pass" if gap <= gap_off + 0.1 * H and intact == n else "fail",
             thickness=H, twin_gap_m=gap, twin_gap_off_m=gap_off, intact=intact, intact_off=intact_off, bonds=n)
    assert intact_off == n and intact == n, f"{n - intact} of {n} seam bonds broke at rest"
    assert gap <= gap_off + 0.1 * H, f"seams open {gap * 1e3:.3f} mm (off: {gap_off * 1e3:.3f} mm)"


@pytest.mark.compile_heavy
def test_without_the_rest_exclusion_presplit_seams_tear(monkeypatch):
    """The control: the same contact without the exclusion pushes every coincident twin pair apart and tears the
    towel at rest at the route's cloth settings before 2026-11-08 (Newton's inertial start, 10 iterations): 17 of 354
    bonds intact, twins about 1 m apart. The control pins those settings. With a better-converged step (the adaptive
    start, or 20 iterations) the coincident twins stay exactly together at rest (0.000 mm, all 354 intact): the push
    comes from the unconverged step, and any motion that breaks the twins' symmetry meets the same contact, which the
    exclusion keeps off."""
    rule, kwargs = entries.vbd_self_contact, entries.solver_kwargs
    monkeypatch.setattr(entries, "vbd_self_contact",
                        lambda objs, radii=None: {k: v for k, v in rule(objs, radii).items() if "exclusion" not in k})
    monkeypatch.setattr(entries, "solver_kwargs", lambda entry, *a, **k: {
        **kwargs(entry, *a, **k), **({"alloy_initial_guess": "inertial", "alloy_chebyshev_rho": None, "iterations": 10}
           if entry == "vbd" else {})})
    gap, intact, n = _presplit_rest(True)
    assert intact < n and gap > H
