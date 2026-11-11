# SPDX-License-Identifier: Apache-2.0
"""R6 Cusick drape twin (bench §4.4 R6): drape coefficient DC (± 0.05) and node count (W₁ ≤ 1).

R6 is registered with ``alloy.scene.twins.drape_route.DrapeRouteTwin`` (Petrak 2021's measured fabrics on the
production route, SCR-WP2-14). This raw-Newton twin and its ``synthetic_dataset`` stay as a CPU regression fixture
with no bar (ruling 2026-11-08 10:56Z (3)); ``LAW`` is shared by both.

Rig: a 30 cm disc of fabric clamped between 18 cm discs, hanging under gravity; DC = (A_shadow − A_d)/(A_D − A_d)
(Cusick 1965; ISO 9073-9) from the silhouette seen from above; nodes = folds of the hanging skirt.

Twin: one VBD cloth disc per world (an unstructured, jittered-hex Delaunay mesh, one random mesh per specimen so the
fold pattern is not imposed by mesh symmetry); vertices inside the 18 cm clamp are pinned (zero mass, which VBD keeps
at their step-start position: newton@ac7f7830:newton/_src/solvers/vbd/particle_vbd_kernels.py:1426). The skirt
falls from flat under gravity and settles under Rayleigh damping (edge_kd = β·edge_ke, tri_kd = β·tri_ke as
`emit_cloth`). No self-contact (folds of these fabrics do not touch at DC >= 0.3; it would multiply the CPU cost).
DC is the area of the projected triangles rasterised at 1 mm; nodes are the peaks of the rim's radial extent.

Calibration (calib trials only): bending length c and areal density w (bench §4.3: "for drape, bending length and
areal density calibrate; DC validates"). Rigidity per width B = w·g·c³ (Peirce; ASTM D1388). VBD's hinge energy is
½·edge_ke·ℓ·θ² per edge (edge_ke [N], ℓ rest length, θ dihedral angle; particle_vbd_kernels.py:752-753), not the
plate's D [N·m] that the USD-parity mapping writes (`derive.cloth_stiffness`, SCR-WP2-7). Matching ½·B·κ² for
cylindrical bending: an edge whose triangle centroids are h_e apart sees θ = κ·h_e·cos²φ (φ: edge to bending axis);
averaging cos⁴φ over orientations (3/8) and giving each edge a third of its two triangles' area gives
edge_ke = 4·B/(h1 + h2), h1, h2 the two triangle heights over the edge (equilateral: 4B/(√3ℓ) ≈ 2.31 B/ℓ). It is
written per edge after finalize (`model.edge_bending_properties`).

Synthetic truth (`dataset`): no closed form exists for Cusick drape with folds. DC comes from the one analytic
idealisation at hand, inextensible radial strips each hanging off the support as a Peirce heavy elastica
(`strip_drape_coefficient`); it ignores hoop coupling, so a folding fabric drapes less (higher DC) than it. The
trials carry no node count (no closed form), so node-count rows are `not_run`. Real drape data replace both.

Measured on this VM (CPU quick: 12 mm mesh, 2 specimens per fabric, 1.5 s, VBD 10 iterations, dt 2.5 ms, 49 s):
DC cotton 0.357 / knit 0.280 / felt 0.454 against strip 0.493 / 0.151 / 0.947, 9-11 nodes, and the skirts still
move at 0.3-0.8 m/s at the end (20 iterations with β = 0.5 s: 0.32 / 0.26 / 0.25, still 0.3-1.0 m/s). The cause is
the calibration itself: an ASTM D1388 strip (L = 2c, which must hold the chord at 41.5°) on VBD with the hinge
mapping above hangs at 76-98° (8-12 mm meshes, 10-50 iterations, 3 s), and a 3 cm small-deflection strip (4 mm mesh) sags
7.8 mm at 2000 iterations against w·L⁴/(8B) = 0.81 mm: VBD's per-vertex Gauss-Seidel does not converge hinge
bending once edge_ke·ℓ/h² >> m/dt². Style3D (Bergou quadratic bending, newton@ac7f7830:newton/_src/solvers/style3d/
kernels.py:71-110, edge_ke = B) settles but is stiff: 0.376 mm on the same small-deflection cantilever (2.15x) and
7-23° on the ASTM strips, still creeping at 3 s. The twin therefore fails R6 on CPU; see tests/scene/
test_twins_soft.py.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp
from scipy.signal import find_peaks
from scipy.spatial import Delaunay

from alloy.bench.analytic.common import G, BaseTwin
from alloy.bench.solvers import SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, RigCard, RigDataset, RigTrial, Tolerance
from alloy.core.errors import LawError
from alloy.testing.fakes import synthetic_rig

log = logging.getLogger("alloy.scene")

R_SPECIMEN, R_SUPPORT = 0.15, 0.09   # [m] 30 cm disc on an 18 cm support
EDGE = {True: 0.012, False: 0.008}   # [m] mesh edge, quick / full
SEEDS = {True: 2, False: 3}          # random meshes (specimens) per fabric
SETTLE = {True: 1.5, False: 3.0}     # [s]
DT = 2.5e-3
BETA = 0.05                          # [s] Rayleigh damping of bending and membrane
MEMBRANE_KE = 1.0e3                  # [N/m] tri_ke; drape is insensitive once tri_ke >> w·g·L (strain ~1e-4)
RASTER = 1.0e-3                      # [m] silhouette pixel (bench §4.4 R6)
NODE_PROMINENCE = 0.15               # peak prominence / (max − min) of the rim profile
_SRC = "bench §4.4 R6"

LAW = LawSpec(
    id="R6.cloth_drape", suite="rig",
    metrics={"drape_coefficient": Tolerance(abs=0.05, source=f"{_SRC}: DC +/- 0.05"),
             "node_count": Tolerance(abs=1.0, source=f"{_SRC}: node count W1 <= 1")},
    stat={"drape_coefficient": "mean", "node_count": "distribution"}, dt=DT, substeps=1,
    duration_s=SETTLE[False], worlds_per_condition=SEEDS[False], solvers=("vbd",), producer="WP2",
    quick={"worlds": 3 * SEEDS[True], "duration_s": SETTLE[True]},
)


# --- mesh --------------------------------------------------------------------------------------------------------

def _ring(r: float, edge: float, phase: float) -> np.ndarray:
    n = max(8, int(round(2.0 * math.pi * r / edge)))
    a = phase + 2.0 * math.pi * np.arange(n) / n
    return r * np.column_stack([np.cos(a), np.sin(a)])


def disc_mesh(edge: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices [V, 2] and triangles [T, 3] of the specimen disc: rings on the rim and the clamp edge, a randomly
    rotated, jittered hex lattice elsewhere, Delaunay-triangulated."""
    rng = np.random.default_rng(seed)
    n = int(math.ceil(R_SPECIMEN / edge)) + 2
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    pts = np.column_stack([(i + 0.5 * j).ravel(), (j * math.sqrt(3.0) / 2.0).ravel()]) * edge
    rot = rng.uniform(0.0, math.pi / 3.0)
    c, s = math.cos(rot), math.sin(rot)
    pts = pts @ np.array([[c, s], [-s, c]]) + rng.normal(0.0, 0.15 * edge, pts.shape)
    r = np.hypot(pts[:, 0], pts[:, 1])
    keep = (r < R_SPECIMEN - 0.6 * edge) & (np.abs(r - R_SUPPORT) > 0.6 * edge)
    verts = np.concatenate([pts[keep], _ring(R_SPECIMEN, edge, rng.uniform(0, 1)), _ring(R_SUPPORT, edge, 0.0)])
    tris = Delaunay(verts).simplices.astype(np.int32)
    a, b, d = (verts[tris[:, k]] for k in range(3))
    area2 = (b[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0])
    tris[area2 < 0] = tris[area2 < 0][:, [0, 2, 1]]  # counter-clockwise: normals +z
    return verts, tris[np.abs(area2) > 1e-3 * edge * edge]


def hinge_stiffness(rest: np.ndarray, edges: np.ndarray, rigidity: float) -> np.ndarray:
    """edge_ke = 4·B/(h1 + h2) per hinge [N] (module docstring); 0 for boundary edges (no opposite vertex)."""
    out = np.zeros(len(edges))
    ok = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
    e0, e1 = rest[edges[ok, 2]], rest[edges[ok, 3]]
    t = (e1 - e0) / np.linalg.norm(e1 - e0, axis=1, keepdims=True)

    def height(o: np.ndarray) -> np.ndarray:
        d = rest[o] - e0
        return np.linalg.norm(d - np.sum(d * t, axis=1, keepdims=True) * t, axis=1)

    out[ok] = 4.0 * rigidity / (height(edges[ok, 0]) + height(edges[ok, 1]))
    return out


# --- measurements ------------------------------------------------------------------------------------------------

def silhouette_area(xy: np.ndarray, tris: np.ndarray, pixel: float = RASTER) -> float:
    """Area [m²] of the union of the projected triangles on a `pixel` grid (pixel centres inside any triangle)."""
    lo = xy.min(axis=0) - pixel
    shape = np.ceil((xy.max(axis=0) + pixel - lo) / pixel).astype(int)
    covered = np.zeros(shape, bool)
    for tri in xy[tris]:
        i0, j0 = np.floor((tri.min(axis=0) - lo) / pixel).astype(int)
        i1, j1 = np.ceil((tri.max(axis=0) - lo) / pixel).astype(int)
        gx, gy = np.meshgrid(lo[0] + (np.arange(i0, i1) + 0.5) * pixel, lo[1] + (np.arange(j0, j1) + 0.5) * pixel,
                             indexing="ij")
        covered[i0:i1, j0:j1] |= _inside(tri, gx, gy)
    return float(covered.sum()) * pixel * pixel


def _inside(tri: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    (ax, ay), (bx, by), (cx, cy) = tri
    d1 = (gx - bx) * (ay - by) - (ax - bx) * (gy - by)
    d2 = (gx - cx) * (by - cy) - (bx - cx) * (gy - cy)
    d3 = (gx - ax) * (cy - ay) - (cx - ax) * (gy - ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def drape_coefficient(area: float) -> float:
    a_d, a_s = math.pi * R_SUPPORT**2, math.pi * R_SPECIMEN**2
    return (area - a_d) / (a_s - a_d)


def node_count(rim_angle: np.ndarray, rim_xy: np.ndarray) -> int:
    """Peaks of the rim's projected radius vs its rest angle (periodic), prominence >= NODE_PROMINENCE of range."""
    order = np.argsort(rim_angle)
    rho = np.hypot(rim_xy[order, 0], rim_xy[order, 1])
    span = float(rho.max() - rho.min())
    if span < 1e-4:
        return 0
    n = rho.size
    tiled = np.concatenate([rho, rho, rho])
    peaks, _ = find_peaks(tiled, prominence=NODE_PROMINENCE * span)
    return int(np.sum((peaks >= n) & (peaks < 2 * n)))


# --- twin --------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Specimen:
    cond: str
    start: int
    end: int
    edges: tuple[int, int]
    tris: np.ndarray        # [T, 3] global particle indices
    rim: np.ndarray         # global indices of the rim vertices
    rim_angle: np.ndarray   # their rest polar angle
    clamp: np.ndarray       # global indices of the pinned vertices


@wp.kernel
def _record_vmax(qd: wp.array(dtype=wp.vec3), start: wp.array(dtype=wp.int32), end: wp.array(dtype=wp.int32),
                 col: int, out: wp.array2d(dtype=float)):
    w = wp.tid()
    m = float(0.0)
    for i in range(start[w], end[w]):
        m = wp.max(m, wp.length(qd[i]))
    out[w, col] = m


class DrapeTwin(BaseTwin):
    """R6 twin: VBD cloth discs, bending from the calib bending length and areal density."""

    law_id = LAW.id
    units = {"drape_coefficient": "1", "node_count": "1"}
    needs_contacts = False

    def __init__(self, iterations: int = 10, edge: float | None = None, beta: float = BETA) -> None:
        super().__init__()
        self.beta = float(beta)
        self.settings = {"vbd": SolverSettings("vbd", {"iterations": int(iterations)})}
        self.edge_override = edge
        self.quick = True
        self.calib: dict[str, tuple[float, float]] = {}   # cond -> (rigidity B [N·m], areal density [kg/m²])
        self._sp: list[_Specimen] = []
        self._final_q: np.ndarray | None = None

    @property
    def edge(self) -> float:
        return self.edge_override or EDGE[self.quick]

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        for cid in dict.fromkeys(t.condition_id for t in calib):
            rows = [t for t in calib if t.condition_id == cid]
            c = float(np.mean([t.scalars["bending_length"] for t in rows]))
            w = float(np.mean([t.scalars["areal_density"] for t in rows]))
            self.calib[cid] = (w * G * c**3, w)
        self.monitors["calib"] = {k: {"rigidity_Nm": b, "areal_density": w} for k, (b, w) in self.calib.items()}

    def expand(self, cond: Condition, quick: bool, per: int) -> list[dict[str, float]]:
        self.quick = quick
        return [{"specimen": float(k)} for k in range(SEEDS[quick])]

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        if cond.id not in self.calib:
            raise LawError(f"R6: no calib bending length / areal density for {cond.id!r}")
        seed = int(sample["specimen"]) + 1000 * int(cond.params["mesh_seed"])
        verts, tris = disc_mesh(self.edge, seed)
        p0, e0 = b.particle_count, b.edge_count
        b.add_cloth_mesh(pos=wp.vec3(), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(),
                         vertices=np.column_stack([verts, np.zeros(len(verts))]).astype(np.float32),
                         indices=tris.reshape(-1).tolist(), density=self.calib[cond.id][1], tri_ke=MEMBRANE_KE,
                         tri_ka=0.0, tri_kd=self.beta * MEMBRANE_KE, edge_ke=0.0, edge_kd=0.0,
                         particle_radius=0.25 * self.edge, label=f"drape_{cond.id}")
        r = np.hypot(verts[:, 0], verts[:, 1])
        rim = np.flatnonzero(np.isclose(r, R_SPECIMEN, atol=1e-6))
        self._sp.append(_Specimen(cond.id, p0, b.particle_count, (e0, b.edge_count), p0 + tris, p0 + rim,
                                  np.arctan2(verts[rim, 1], verts[rim, 0]),
                                  p0 + np.flatnonzero(r <= R_SUPPORT + 1e-6)))

    def on_model(self, model: newton.Model) -> None:
        mass = model.particle_mass.numpy()
        clamp = np.concatenate([s.clamp for s in self._sp])
        mass[clamp] = 0.0
        model.particle_mass.assign(mass)
        model.particle_inv_mass.assign(np.where(mass > 0, 1.0 / np.maximum(mass, 1e-30), 0.0).astype(np.float32))
        rest = model.particle_q.numpy().astype(np.float64)
        edges = model.edge_indices.numpy().reshape(-1, 4)
        props = model.edge_bending_properties.numpy()
        for s in self._sp:
            lo, hi = s.edges
            ke = hinge_stiffness(rest, edges[lo:hi], self.calib[s.cond][0])
            props[lo:hi, 0], props[lo:hi, 1] = ke, self.beta * ke
        model.edge_bending_properties.assign(props)
        dev = model.device
        self._start = wp.array(np.array([s.start for s in self._sp], np.int32), dtype=wp.int32, device=dev)
        self._end = wp.array(np.array([s.end for s in self._sp], np.int32), dtype=wp.int32, device=dev)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: DeviceTwinLog) -> None:
        col = log_.column(step)
        if col >= 0:
            wp.launch(_record_vmax, dim=len(self._sp), inputs=[state.particle_qd, self._start, self._end, col],
                      outputs=[log_.channel("vmax")], device=self.model.device)
        if step == log_.n_steps:
            self._final_q = state.particle_q.numpy()

    def _measure(self, s: _Specimen) -> tuple[float, int]:
        q = self._final_q
        dc = drape_coefficient(silhouette_area(q[:, :2], s.tris))
        return dc, node_count(s.rim_angle, q[s.rim, :2])

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        if self._final_q is None:
            raise LawError("R6: the run ended without a final state")
        rows = [self._measure(s) for s in self._sp]
        vmax = log_.series("vmax")
        self.monitors.update({"dc": [r[0] for r in rows], "nodes": [r[1] for r in rows],
                              "vmax_end": vmax[:, -1].tolist(), "edge_m": self.edge})
        log.info("R6 twin: DC %s nodes %s", self.monitors["dc"], self.monitors["nodes"])
        return {"drape_coefficient": np.array([r[0] for r in rows]),
                "node_count": np.array([float(r[1]) for r in rows])}


# --- synthetic truth: inextensible radial strips (Peirce heavy elastica) --------------------------------------------

def strip_reach(overhang_over_c: float) -> float:
    """Horizontal reach x(L)/L of a horizontally clamped inextensible heavy strip: B·θ'' + w·g·(L − s)·cos θ = 0,
    θ(0) = 0, θ'(L) = 0 (Peirce 1930), in σ = s/L with λ = (L/c)³: θ'' = −λ(1 − σ)cos θ."""
    from scipy.integrate import solve_bvp

    lam = overhang_over_c**3
    sigma = np.linspace(0.0, 1.0, 201)

    def rhs(s, y):
        return np.vstack([y[1], -lam * (1.0 - s) * np.cos(y[0]), np.cos(y[0])])

    def bc(ya, yb):
        return np.array([ya[0], yb[1], ya[2]])

    guess = np.vstack([0.5 * sigma, np.zeros_like(sigma), sigma])
    sol = solve_bvp(rhs, bc, sigma, guess, tol=1e-8, max_nodes=20000)
    if not sol.success:
        raise LawError(f"R6: heavy-elastica solve failed for L/c = {overhang_over_c}: {sol.message}")
    return float(sol.sol(1.0)[2])


def strip_drape_coefficient(bending_length: float) -> float:
    """DC of the radial-strip idealisation: every radius hangs as an independent heavy elastica off the support."""
    overhang = R_SPECIMEN - R_SUPPORT
    reach = R_SUPPORT + overhang * strip_reach(overhang / bending_length)
    return drape_coefficient(math.pi * reach * reach)


CONDITIONS = (Condition("cotton", {"mesh_seed": 1.0}), Condition("knit", {"mesh_seed": 2.0}),
              Condition("felt", {"mesh_seed": 3.0}))
TRUTH = {"cotton": {"bending_length": 0.025, "areal_density": 0.120},
         "knit": {"bending_length": 0.014, "areal_density": 0.180},
         "felt": {"bending_length": 0.045, "areal_density": 0.350}}
NOISE = {"bending_length": 5e-4, "areal_density": 2e-3, "drape_coefficient": 0.01}
INSTRUMENTS = {"bending_length": {"u95": 5e-4, "what": "ASTM D1388 cantilever, 1 mm scale"},
               "areal_density": {"u95": 1e-3, "what": "100 cm² die, 0.01 g balance"},
               "drape_coefficient": {"u95": 0.01, "what": "photo at 1.5 m, segmentation"}}
TRIALS = 12  # 3 specimens x 2 faces x 2 drapings; half calib


def _truth(params: Mapping[str, float], cond: Condition) -> dict[str, float]:
    c, w = params[f"{cond.id}.bending_length"], params[f"{cond.id}.areal_density"]
    return {"bending_length": c, "areal_density": w, "drape_coefficient": strip_drape_coefficient(c)}


def synthetic_dataset(seed: int = 0) -> RigDataset:
    """Synthetic drape trials: a CPU regression fixture for ``DrapeTwin`` with no bar (R6 itself is judged on
    Petrak 2021's fabrics, ``alloy.scene.twins.drape_route``; ruling 2026-11-08 10:56Z (3)). DC truth is the radial-strip heavy elastica (no hoop coupling, no folds), the only
    closed form at hand: a fabric that folds is stiffer than independent strips, so it bounds DC from below and the
    trials carry no node count (the node-count rows are `not_run`)."""
    params = {f"{c}.{k}": v for c, m in TRUTH.items() for k, v in m.items()}
    data = synthetic_rig(LAW.id, params, NOISE, conditions=CONDITIONS, trials_per_condition=TRIALS,
                         calib_per_condition=TRIALS // 2, seed=seed, model=_truth)
    card = dataclasses.replace(data.card, dataset_id="synthetic-v1", protocol_version="synthetic/1",
                               instruments=dict(INSTRUMENTS),
                               objects={"specimen": {"radius_m": R_SPECIMEN, "support_radius_m": R_SUPPORT}})
    return dataclasses.replace(data, card=card)


__all__ = ["LAW", "DrapeTwin", "synthetic_dataset", "disc_mesh", "hinge_stiffness", "silhouette_area", "drape_coefficient",
           "node_count", "strip_reach", "strip_drape_coefficient"]
