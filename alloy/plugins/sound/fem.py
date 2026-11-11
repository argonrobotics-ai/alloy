# SPDX-License-Identifier: Apache-2.0
"""Offline FEM modes for ``ModalSound``: tetrahedra (P2 by default, or P1), scipy ``eigsh``, as a ``modes.npz``.

The model is linear isotropic elasticity on a tet mesh in the body frame. It assembles stiffness K and consistent
mass M with scipy sparse, then solves K φ = ω² M φ by shift-invert ``eigsh`` near 0. For a free body the 6 rigid
modes (ω ≈ 0) are dropped; ``fixed`` vertices clamp instead. Mass-normalised shapes (φᵀMφ = 1) become the
plug-in's convention, unit mass-weighted mean square:
- gains g = sqrt(m)·φ at the boundary vertices;
- modal mass m (the body's mass).
So modal force = g·F and the surface motion is Σ g_k q_k, as for the analytic presets.

- Radiating area: half the boundary area (one side, ESTIMATE).
- Damping: the material's tan δ / 2 at each frequency.
- Element order:
  - P1 (4-node) tets lock in bending. A 300 x 20 x 5 mm steel bar reads +54 % at 2 cells through the thickness and
    +3 % at 8.
  - P2 (10-node, straight-sided; the default) tets converge in a few cells.
  - Element matrices are exact: ∫ L^α dV = 6V·α! / (|α| + 3)! over barycentric monomials.
  - ``tests/plugins/test_sound_fem.py`` checks convergence against Euler-Bernoulli on a square bar.

Use it from Python (``box_modes``, ``mesh_modes``) or the CLI, which reads an OBJ surface and tetrahedralises it
with pytetwild:
    python -m alloy.plugins.sound.fem part.obj --material steel --modes 16 --edge 0.004 --out part_modes.npz
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import sys
import tempfile

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from alloy.plugins.sound.modes import Material, ModeSet


def box_mesh(dims: tuple[float, float, float], cells: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A box (L, w, t along body x, y, z, centred on the origin): ``cells`` hexes per axis, each split into 6 tets
    around its main diagonal (Freudenthal, conforming across cells). Returns (vertices [N, 3], tets [M, 4])."""
    n = [c + 1 for c in cells]
    axes = [np.linspace(-0.5 * d, 0.5 * d, k) for d, k in zip(dims, n)]
    verts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    vid = lambda i, j, k: (i * n[1] + j) * n[2] + k  # noqa: E731
    tets = []
    for i, j, k in itertools.product(*(range(c) for c in cells)):
        for perm in itertools.permutations(range(3)):
            p = [0, 0, 0]
            path = [vid(i, j, k)]
            for a in perm:
                p[a] += 1
                path.append(vid(i + p[0], j + p[1], k + p[2]))
            tets.append(path)
    return verts, np.asarray(tets, np.int64)


def _elements(verts: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape-function gradients [M, 4, 3] and volumes [M]; flips inverted tets."""
    x = verts[tets]                                               # [M, 4, 3]
    jac = x[:, 1:] - x[:, :1]
    det = np.linalg.det(jac)
    if np.any(np.abs(det) < 1e-30):
        raise ValueError("fem: degenerate tetrahedron (zero volume)")
    a = np.concatenate([np.ones((len(tets), 4, 1)), x], axis=2)  # rows [1, x, y, z]
    grads = np.linalg.inv(a)[:, 1:, :].transpose(0, 2, 1)        # ∇N_i
    return grads, np.abs(det) / 6.0


EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))          # P2 mid-edge nodes 4..9 of a tet


def _poly_mul(a: dict, b: dict) -> dict:
    out: dict = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            key = tuple(x + y for x, y in zip(ea, eb))
            out[key] = out.get(key, 0.0) + ca * cb
    return out


def _integral(poly: dict) -> float:
    """∫ poly(L1..L4) dV / V over a tetrahedron: Σ c · 6 Π a! / (Σ a + 3)!."""
    return sum(c * 6.0 * math.prod(math.factorial(x) for x in e) / math.factorial(sum(e) + 3)
               for e, c in poly.items())


def _p2_mass_reference() -> np.ndarray:
    """∫ N_k N_l dV / V for the 10 P2 shape functions (corners 2L² - L, edges 4 L_i L_j)."""
    unit = lambda *ix: tuple(sum(1 for i in ix if i == a) for a in range(4))  # noqa: E731
    shapes = [{unit(k, k): 2.0, unit(k): -1.0} for k in range(4)] + [{unit(i, j): 4.0} for i, j in EDGES]
    return np.array([[_integral(_poly_mul(a, b)) for b in shapes] for a in shapes])


P2_MASS = _p2_mass_reference()


def _b_matrix(grads: np.ndarray) -> np.ndarray:
    """Strain-displacement B [..., 6, 3n] from nodal gradient vectors [..., n, 3] (Voigt, engineering shear)."""
    n = grads.shape[-2]
    b = np.zeros(grads.shape[:-2] + (6, 3 * n))
    for i in range(n):
        gx, gy, gz = grads[..., i, 0], grads[..., i, 1], grads[..., i, 2]
        c = 3 * i
        b[..., 0, c], b[..., 1, c + 1], b[..., 2, c + 2] = gx, gy, gz
        b[..., 3, c + 1], b[..., 3, c + 2] = gz, gy                   # γ_yz
        b[..., 4, c], b[..., 4, c + 2] = gz, gx                       # γ_xz
        b[..., 5, c], b[..., 5, c + 1] = gy, gx                       # γ_xy
    return b


def _elasticity(mat: Material) -> np.ndarray:
    e, nu = mat.youngs, mat.poisson
    lam, mu = e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[np.arange(3), np.arange(3)] += 2 * mu
    d[np.arange(3, 6), np.arange(3, 6)] = mu
    return d


def p2_nodes(tets: np.ndarray, n_verts: int) -> np.ndarray:
    """10-node connectivity [M, 10]: the 4 corners, then mid-edge nodes numbered from ``n_verts`` (shared)."""
    pairs = np.sort(tets[:, EDGES], axis=2).reshape(-1, 2)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    return np.concatenate([tets, n_verts + inv.reshape(len(tets), 6)], axis=1)


def assemble(verts: np.ndarray, tets: np.ndarray, mat: Material, order: int = 2
             ) -> tuple[sp.csc_matrix, sp.csc_matrix, float, int]:
    """Stiffness K, consistent mass M, total mass and node count (P1: the vertices; P2: + mid-edge nodes)."""
    grads, vol = _elements(verts, tets)
    d = _elasticity(mat)
    if order == 1:
        b = _b_matrix(grads)
        ke = np.einsum("mji,jk,mkl->mil", b, d, b) * vol[:, None, None]
        mref, nodes = (np.ones((4, 4)) + np.eye(4)) / 20.0, tets
    elif order == 2:
        g = np.zeros((len(tets), 4, 10, 3))                   # ∇N_k = Σ_a g[a, k] L_a
        for k in range(4):
            for a in range(4):
                g[:, a, k] = (4.0 * (a == k) - 1.0) * grads[:, k]
        for e, (i, j) in enumerate(EDGES):
            g[:, j, 4 + e] += 4.0 * grads[:, i]
            g[:, i, 4 + e] += 4.0 * grads[:, j]
        b = _b_matrix(g)                                       # [M, 4, 6, 30]
        iab = (np.ones((4, 4)) + np.eye(4)) / 20.0             # ∫ L_a L_b dV / V
        ke = np.einsum("maji,jk,mbkl,ab->mil", b, d, b, iab, optimize=True) * vol[:, None, None]
        mref, nodes = P2_MASS, p2_nodes(tets, len(verts))
    else:
        raise ValueError("fem: order is 1 or 2")
    nn = nodes.shape[1]
    me = np.kron(mref, np.eye(3))[None] * (mat.density * vol)[:, None, None]
    dofs = (3 * nodes[:, :, None] + np.arange(3)).reshape(len(tets), 3 * nn)
    rows = np.repeat(dofs, 3 * nn, axis=1).ravel()
    cols = np.tile(dofs, (1, 3 * nn)).ravel()
    n_nodes = int(nodes.max()) + 1
    n = 3 * n_nodes
    k = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    m = sp.coo_matrix((me.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    return k, m, float(mat.density * vol.sum()), n_nodes


def boundary(tets: np.ndarray) -> np.ndarray:
    """Boundary triangles [F, 3]: tet faces that belong to one tet only."""
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    key = np.sort(faces, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def _solve(k: sp.csc_matrix, m: sp.csc_matrix, count: int) -> tuple[np.ndarray, np.ndarray]:
    """``count`` lowest eigenpairs of K φ = λ M φ. Shift-invert at σ = -1 factors K + M ≈ K, singular to cond ~λ_max
    next to the 6 rigid modes, and left elastic pairs unconverged (residual ~1 on a P2 bar). So the first pass
    estimates the lowest elastic λ at a well-conditioned σ = -1e-3·median(diag K / diag M) (WP13), and the second
    shifts to half of it, which separates rigid and elastic modes. Raises if an elastic pair's residual exceeds
    1e-6."""
    dk, dm = k.diagonal(), m.diagonal()
    scale = float(np.median(dk[dm > 0] / dm[dm > 0]))
    lam, _ = eigsh(k, k=count, M=m, sigma=-1e-3 * scale, which="LM")
    pos = lam[lam > 1e-8 * lam.max()]
    lam, vec = eigsh(k, k=count, M=m, sigma=0.5 * float(pos.min()) if pos.size else -1.0, which="LM")
    for i in np.flatnonzero(lam > 1e-8 * lam.max()):
        kv = k @ vec[:, i]
        res = np.linalg.norm(kv - lam[i] * (m @ vec[:, i])) / max(float(np.linalg.norm(kv)), 1e-300)
        if res > 1e-6:
            raise ValueError(f"fem: eigenpair {i} did not converge (residual {res:.1e})")
    return lam, vec


def mesh_modes(verts: np.ndarray, tets: np.ndarray, mat: Material, n: int = 16, fixed: np.ndarray | None = None,
               name: str = "mesh", order: int = 2) -> ModeSet:
    """The ``n`` lowest elastic modes of a tet mesh (body frame), as a ``ModeSet`` sampled at its boundary
    vertices."""
    verts, tets = np.asarray(verts, float), np.asarray(tets, np.int64)
    k, m, mass, n_nodes = assemble(verts, tets, mat, order)
    keep = np.ones(3 * n_nodes, bool)
    if fixed is not None and len(fixed):
        keep[(3 * np.asarray(fixed)[:, None] + np.arange(3)).ravel()] = False
    rigid = 0 if (fixed is not None and len(fixed)) else 6
    kk, mm = k[keep][:, keep], m[keep][:, keep]
    lam, vec = _solve(kk, mm, n + rigid + 4)
    idx = np.argsort(lam)
    lam, vec = lam[idx], vec[:, idx]
    elastic = lam > 1e-8 * lam.max()          # ARPACK can return a rigid mode twice (6-fold zero eigenvalue)
    if rigid and np.count_nonzero(~elastic) < rigid:
        raise ValueError("fem: fewer than 6 rigid modes found; refine the eigen-solve")
    if rigid and np.any(lam[elastic][0] < 1e-6 * lam.max()):
        raise ValueError("fem: a free body has more than 6 zero modes (disconnected mesh?)")
    lam, vec = lam[elastic][:n], vec[:, elastic][:, :n]
    phi = np.zeros((3 * n_nodes, len(lam)))
    phi[keep] = vec
    tri = boundary(tets)
    pts = np.unique(tri)                                          # boundary corner vertices
    gains = math.sqrt(mass) * phi.reshape(n_nodes, 3, -1)[pts].transpose(2, 0, 1)
    e1, e2 = verts[tri[:, 1]] - verts[tri[:, 0]], verts[tri[:, 2]] - verts[tri[:, 0]]
    area = 0.25 * float(np.linalg.norm(np.cross(e1, e2), axis=1).sum())     # half the boundary
    freq = np.sqrt(lam) / (2.0 * math.pi)
    return ModeSet(freq, mat.zeta(freq), np.full(len(freq), mass), np.ones(len(freq)), verts[pts], gains, area, name)


def box_modes(dims: tuple[float, float, float], mat: Material, n: int = 16, cells: tuple[int, int, int] | None = None,
              name: str = "fem_box", order: int = 2) -> ModeSet:
    """FEM modes of a free box (L, w, t). Default ``cells``: 2 through the thinnest side, cubic-ish elsewhere (at
    most 60 per axis)."""
    if cells is None:
        h = min(dims) / 2.0
        cells = tuple(int(min(max(round(d / h), 1), 60)) for d in dims)
    return mesh_modes(*box_mesh(dims, cells), mat, n, name=name, order=order)


def read_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of an OBJ (polygons fanned into triangles)."""
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                v.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":                                # 1-based, or negative = relative to the end
                idx = [i - 1 if i > 0 else len(v) + i for i in (int(p.split("/")[0]) for p in parts[1:])]
                f.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    return np.asarray(v, float), np.asarray(f, np.int32)


def main(argv: list[str] | None = None) -> int:
    from alloy.plugins.sound.presets import material  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="FEM modes of a closed OBJ surface -> modes.npz (ModalSound npz:<path>)")
    ap.add_argument("obj")
    ap.add_argument("--material", default="steel")
    ap.add_argument("--loss", type=float, default=0.0, help="constant tan delta (overrides the material)")
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--edge", type=float, default=None, help="target tet edge length, m (default: bbox / 20)")
    ap.add_argument("--scale", type=float, default=1.0, help="OBJ units to m")
    ap.add_argument("--order", type=int, default=2, choices=(1, 2), help="tet order (P2 default: P1 locks in bending)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import pytetwild  # noqa: PLC0415

    v, f = read_obj(a.obj)
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):   # fTetWild writes __tracked_surface.stl
        tv, tt = pytetwild.tetrahedralize(v * a.scale, f, edge_length_abs=a.edge, quiet=True)
    ms = mesh_modes(tv, tt, material(a.material, a.loss or None), a.modes, name=a.obj.rsplit("/", 1)[-1],
                    order=a.order)
    ms.save(a.out)
    print(f"{len(tt)} tets, {ms.count} modes {ms.freq_hz[0]:.1f}..{ms.freq_hz[-1]:.1f} Hz -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
