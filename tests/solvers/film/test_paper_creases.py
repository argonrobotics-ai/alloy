# SPDX-License-Identifier: Apache-2.0
"""Paper fold lines (creases) on the film route: crease hinges take their own stiffness and elastic range
(``FilmExtras.crease_ratio`` / ``crease_yield``, card ``film.crease_*``), a card mesh's OBJ ``l`` lines are its
creases, a fold made on a crease springs back by the crease's elastic range (FOLD-1), and a bend localises on a weak
crease (FOLD-2). Paper values: ``library/materials/paper_kraft.yaml``; notes: ``docs/notes/paper-pack.md``."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.cards.library import CardLibrary
from alloy.cards.types import MeshSource
from alloy.solvers.film.emit import add_film_mesh, crease_mask, obj_creases, spec_from_card
from alloy.solvers.film.rigs import FilmRig, grid_mesh, pin
from alloy.solvers.film.topology import bending_ke

LIB = CardLibrary(assets=None)
KRAFT = spec_from_card(LIB.resolve(LIB.object("paper_kraft_sheet")))      # 100 g/m2 kraft, 135 um


def _spec(crease_ratio: float, crease_yield: float):
    return replace(KRAFT, extras=replace(KRAFT.extras, crease_ratio=crease_ratio, crease_yield=crease_yield))


def _kappa(spec) -> float:
    return spec.plate_rigidity / (spec.extras.crease_ratio * spec.thickness)


def _strip(n_a: int, n_b: int, ny: int, dx: float):
    """A flat strip, crease line across it at x = 0 (n_a cells before, n_b after): vertices, triangles, crease pairs."""
    v, t = grid_mesh(n_a + n_b, ny, dx)
    v[:, 0] -= n_a * dx
    col = [n_a * (ny + 1) + j for j in range(ny + 1)]
    return v, t, np.asarray(list(zip(col[:-1], col[1:])), np.int64)


def test_crease_hinges_take_the_crease_values() -> None:
    v, t, creases = _strip(5, 5, 4, 2e-3)
    spec = _spec(200.0, 0.4)
    b = newton.ModelBuilder()
    add_film_mesh(b, v, t, spec, creases=creases)
    m = b.finalize(device="cpu")
    edges = m.edge_indices.numpy()
    on = crease_mask(edges, creases)
    ke, ya = m.edge_bending_properties.numpy()[:, 0], m.film.yield_angle.numpy()
    assert on.sum() == len(creases)                               # every crease segment is an interior hinge
    assert np.allclose(ke[on], _kappa(spec), rtol=1e-5) and np.allclose(ya[on], 0.4)
    assert np.allclose(ke[~on], bending_ke(v, edges, spec.plate_rigidity)[~on], rtol=1e-5)
    assert np.allclose(ya[~on], spec.extras.yield_angle)


def test_obj_lines_are_the_creases(tmp_path) -> None:
    path = tmp_path / "sheet.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\nl 1 3\nl 2 3 4\n")
    obj = SimpleNamespace(card=SimpleNamespace(shape=MeshSource(path=str(path), sha256="")))
    assert obj_creases(obj).tolist() == [[0, 2], [1, 2], [2, 3]]
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n")
    assert obj_creases(obj) is None


def _dihedral_deg(q: np.ndarray, edge) -> float:
    """Fold angle of hinge (o0, o1, v0, v1): 0 flat, 180 folded flat (the angle between its triangles' normals)."""
    o0, o1, a, c = edge
    n1, n2 = np.cross(q[c] - q[a], q[o0] - q[a]), np.cross(q[o1] - q[a], q[c] - q[a])
    return math.degrees(math.acos(float(np.clip(n1 @ n2 / np.linalg.norm(n1) / np.linalg.norm(n2), -1.0, 1.0))))


def _opening(r) -> float:
    q = r.q()[r.names[0]["tipB"]].mean(0) - np.array([0.0, 0.0, 0.5])
    return math.degrees(math.atan2(q[2], q[0]))                   # 0 = flat open, 180 - gap = folded


def fold_release(spec, creased: bool, *, leg_cells: int = 4, ny: int = 2, dx: float = 1e-2, gap_deg: float = 2.0,
                 dt: float = 1e-3, seconds: float = 1.0) -> dict:
    """FOLD-1: the strip folded flat on its crease line (leg B rotated π − gap about x = 0, no gravity), leg A pinned,
    then released. Returns the fold hinges' angle and set rest angle after the first step (deg; the flat-rest strip
    springs open ~5 deg at the fold in that step, before the plastic update) and the angle the fold opened by
    (180 − final fold angle at the leg's tip, deg). 1 cm cells, as library sheets: at 2 mm cells a kraft hinge is ~1e5
    times stiffer than its vertices' inertia term and VBD's local solves barely move the sheet."""
    v, t, creases = _strip(leg_cells, leg_cells, ny, dx)
    ang = math.pi - math.radians(gap_deg)
    folded = v.copy()
    B = v[:, 0] > 1e-9
    folded[B, 0], folded[B, 2] = v[B, 0] * math.cos(ang), v[B, 0] * math.sin(ang)
    tip = np.flatnonzero(np.abs(v[:, 0] - leg_cells * dx) < 1e-9)

    def build(b):
        p0, _ = add_film_mesh(b, v, t, spec, pos=(0.0, 0.0, 0.5), creases=creases if creased else None)
        for n in range(len(v)):
            b.particle_q[p0 + n] = (float(folded[n, 0]), float(folded[n, 1]), 0.5 + float(folded[n, 2]))
        pin(b, p0 + np.flatnonzero(v[:, 0] < -1e-9))
        return {"tipB": p0 + tip}

    rig = FilmRig(build, dt=dt, iterations=10, gravity=0.0)
    rig.step(1)                          # the plastic update follows the step: the set is taken at this fold angle
    edges = rig.model.edge_indices.numpy()
    fold = crease_mask(edges, creases)
    rest = float(np.degrees(np.abs(rig.model.edge_rest_angle.numpy()[fold])).mean())
    first = float(np.mean([_dihedral_deg(rig.q(), e) for e in edges[fold]]))
    series = rig.run(seconds, every=0.05, probe=_opening)
    return {"fold_deg_step1": first, "rest_deg": rest, "opened_deg": 180.0 - float(np.mean(series[-4:]))}


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_fold1_a_crease_sets_and_springs_back_by_its_elastic_range() -> None:
    """Sheet hinges keep 36 deg of fold memory, the crease 23 deg (0.4 rad). Folded flat on the crease, the crease
    hinges set to the fold angle minus 23 deg; released freely, the fold opens by 23-46 deg (the leg's elastic swing
    can add up to one more elastic range of plastic set), and by less than an uncreased fold."""
    spec = _spec(200.0, 0.4)
    creased, plain = fold_release(spec, creased=True), fold_release(spec, creased=False)
    yc, ys = math.degrees(0.4), math.degrees(spec.extras.yield_angle)
    assert abs(creased["rest_deg"] - (creased["fold_deg_step1"] - yc)) <= 1.0, creased
    assert abs(plain["rest_deg"] - (plain["fold_deg_step1"] - ys)) <= 1.0, plain
    assert yc - 2.0 <= creased["opened_deg"] <= 2.0 * yc + 2.0, creased
    assert ys - 2.0 <= plain["opened_deg"] <= 2.0 * ys + 2.0, plain
    assert plain["opened_deg"] - creased["opened_deg"] >= 10.0, (creased, plain)


def cantilever_kinks(spec, creased: bool, *, n_a: int = 4, n_b: int = 8, ny: int = 2, dx: float = 1e-2,
                     dt: float = 1e-3, seconds: float = 1.5) -> np.ndarray:
    """FOLD-2: a strip clamped over its first two columns, crease n_a cells out, sagging under gravity: the change
    of slope (deg) at each interior column of the centre line, root first, averaged over the run (the strip bounces:
    little damping)."""
    v, t, creases = _strip(n_a, n_b, ny, dx)
    mid = ny // 2
    line = np.asarray([i * (ny + 1) + mid for i in range(n_a + n_b + 1)])

    def build(b):
        p0, _ = add_film_mesh(b, v, t, spec, pos=(0.0, 0.0, 0.5), creases=creases if creased else None)
        pin(b, p0 + np.flatnonzero(v[:, 0] <= -(n_a - 1) * dx + 1e-9))    # the first two columns: a clamp
        return {"line": p0 + line}

    def kinks(r):
        q = r.q()[r.names[0]["line"]]
        return np.abs(np.diff(np.degrees(np.arctan2(np.diff(q[:, 2]), np.diff(q[:, 0])))))

    rig = FilmRig(build, dt=dt, iterations=10)
    return np.mean(rig.run(seconds, every=0.05, probe=kinks), axis=0)     # kink at columns 1 .. n-1


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_fold2_a_bend_localises_on_a_weak_crease() -> None:
    """Without a crease the clamped strip bends most at its root; a crease at L*/h = 200 (0.037 N·m/rad per m at
    135 um, about 4 times weaker than the sheet's hinges at 1 cm cells) takes the largest kink where it lies."""
    n_a = 4
    spec = _spec(200.0, 0.0)
    plain = cantilever_kinks(spec, creased=False, n_a=n_a)
    creased = cantilever_kinks(spec, creased=True, n_a=n_a)
    assert int(np.argmax(plain)) == 0, plain.round(2)
    assert int(np.argmax(creased)) + 1 == n_a, creased.round(2)
    assert creased[n_a - 1] > 2.0 * plain[n_a - 1], (creased.round(2), plain.round(2))
