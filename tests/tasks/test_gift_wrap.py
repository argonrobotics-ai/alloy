# SPDX-License-Identifier: Apache-2.0
"""The wrapped present (alloy/tasks/gift_wrap_mesh.py, gift_present.py; card gift_wrapped_present). CPU.

Mesh: the library OBJ is the generator's output and the card's hash matches it; the paper covers the box at one contact
radius and stacks its layers a pitch apart (no two vertices closer than 0.8 mm unless they share an edge); the surface
is one sheet (every edge in at most two triangles); every fold line runs along mesh edges; the vt lines are the flat
sheet, and folding never shortens the paper. Build: the present on its box builds on the film route at the mesh's
vertices; a copy of the card with tear lines pre-splits them along the flat sheet's 44 mm grid (the library card has
none: with them the present does not rest, docs/known_gaps/paper-unwrap-gift.md).
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
import pytest

from alloy.tasks.gift_wrap_mesh import OBJ_PATH, WrapSpec, obj_text, wrap_mesh

ROOT = Path(__file__).resolve().parents[2]
S = WrapSpec()
M = wrap_mesh(S)


def _edges(t: np.ndarray) -> collections.Counter:
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    return collections.Counter(map(tuple, e.tolist()))


def test_library_mesh_is_the_generators_and_the_card_hash_matches():
    text = (ROOT / OBJ_PATH).read_text()
    assert text == obj_text(M, S)
    card = (ROOT / "library/objects/gift_wrapped_present.yaml").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() in card


def test_paper_covers_the_box_with_layers_apart():
    v = M.vertices
    L, W, H = S.box
    off = np.maximum(np.abs(v - [0.0, 0.0, 0.5 * H]) - [0.5 * L, 0.5 * W, 0.5 * H], 0.0)
    gap = np.linalg.norm(off, axis=1)
    assert gap.min() > S.r - 1e-9 and gap.max() < S.r + 6 * S.pitch     # at most 7 layers (an end under the seam)
    assert np.isclose(v[:, 2].min(), -(S.r + S.pitch))                  # the seam's outer layer, lowest
    d = np.linalg.norm(v[:, None] - v[None], axis=2)
    close = {(int(a), int(b)) for a, b in zip(*np.nonzero(d < 8e-4)) if a < b}
    assert close <= set(_edges(M.triangles))                          # layers never touch


def test_one_sheet_with_folds_on_edges_and_a_flat_layout():
    t = M.triangles
    edges = _edges(t)
    assert max(edges.values()) == 2
    assert len(M.folds) == 14                   # 4 girth corners, 2 end edges, 8 tuck diagonals
    assert all((min(a, b), max(a, b)) in edges for line in M.folds for a, b in zip(line[:-1], line[1:]))
    i, j = np.divmod(np.arange(len(M.vertices)), S.nv + 1)
    assert np.allclose(M.layout, np.stack([i, j], 1) * S.c)
    e = np.array(list(edges))
    ratio = (np.linalg.norm(np.diff(M.vertices[e], axis=1)[:, 0], axis=1)
             / np.linalg.norm(np.diff(M.layout[e], axis=1)[:, 0], axis=1))
    assert ratio.min() > 1.0 - 1e-9 and ratio.max() < 1.52           # folding only adds slack


def test_the_wrapped_present_card_has_its_flat_sheet():
    from types import SimpleNamespace

    from alloy.cards.library import CardLibrary
    from alloy.cards.types import MeshSource
    from alloy.solvers.film.layout import layout_tear_edges, obj_layout

    lib = CardLibrary(assets=None)
    shape = lib.object("gift_wrapped_present").shape                 # the path as the assembler resolves it
    shape = MeshSource(path=str(lib.asset_path(shape.path)), sha256=shape.sha256)
    layout = obj_layout(SimpleNamespace(card=SimpleNamespace(shape=shape)))
    assert np.allclose(layout, M.layout)
    assert len(layout_tear_edges(M.triangles, layout, 0.044)) == 390      # no line within 22 mm of the edge


def _tear_card(root: Path) -> str:
    """A copy of the library card with tear lines on a 44 mm grid of the flat sheet, in a scene root ``root``."""
    (root / "objects").mkdir(parents=True)
    (root / "meshes").mkdir()
    (root / "meshes" / OBJ_PATH.name).write_bytes((ROOT / OBJ_PATH).read_bytes())
    card = (ROOT / "library/objects/gift_wrapped_present.yaml").read_text()
    card = card.replace("name: gift_wrapped_present", "name: gift_wrapped_present_tear")
    card = card.replace("budget:\n  particles: 1014\n",
                        "budget:\n  particles: 1614\ntopology:\n  spare_vertices: 600\n  split_spacing: 0.044\n")
    (root / "objects" / "gift_wrapped_present_tear.yaml").write_text(card)
    return "gift_wrapped_present_tear"


def _build(card: str | None = None, roots=()):
    import alloy
    from alloy.tasks import gift_present as GP

    scene = alloy.Scene(dt=0.01, **({"roots": [str(r) for r in roots]} if roots else {}))
    scene.add_ground()
    old, GP.CARD = GP.CARD, card or GP.CARD
    try:
        wrap, box = GP.add_present(scene)
    finally:
        GP.CARD = old
    sim = alloy.build(scene, device="cpu")
    st = sim.state(world=0)
    return sim, st[f"{wrap}.points"], st[f"{box}.pos"][2] - 0.5 * S.box[2], wrap


@pytest.mark.timeout(240)
def test_present_builds_on_the_film_route_at_the_mesh():
    sim, p, z0, wrap = _build()
    assert sim.routes()[wrap]["route"] == "cloth_film"
    assert p.shape == M.vertices.shape                                              # no tear lines: no copies
    assert np.abs(p - (M.vertices + [0.0, 0.0, z0])).max() < 1e-6


@pytest.mark.timeout(240)
def test_tear_lines_follow_the_flat_sheet(tmp_path):
    from alloy.solvers.film.layout import layout_tear_edges

    sim, p, z0, _ = _build(_tear_card(tmp_path), roots=[tmp_path])
    n = len(M.vertices)
    assert np.abs(p[:n] - (M.vertices + [0.0, 0.0, z0])).max() < 1e-6             # originals unmoved
    on = np.unique(layout_tear_edges(M.triangles, M.layout, 0.044))
    dup = p[n:] - [0.0, 0.0, z0]
    assert 0 < len(dup) <= 600                                                     # within the card's spare budget
    nearest = np.linalg.norm(dup[:, None] - M.vertices[None], axis=2).argmin(1)
    assert np.isin(nearest, on).all()                                              # every copy on a tear line
