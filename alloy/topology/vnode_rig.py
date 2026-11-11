# SPDX-License-Identifier: Apache-2.0
"""Virtual-node cut block rig (WP10 item 9 step 3, prototype): ``cut_block_stroke``'s block with no seams, every tet
cuttable (two copies per node, two fragment tets per tet), and the same blade. ``VNodeSim`` is a ``DirectSim`` whose substep is::

    pre_substep hooks (pose writer) -> VNodeCutter.pre_step -> collide -> VNodeCutter.filter_contacts -> VBD
    -> keep the blades kinematic -> observers

``reset`` restores the start, for all worlds or a world mask, from a snapshot taken at construction: the state with
the blade poses, the model arrays a split rewrites (tet materials, particle mass and flags, weld stiffness) and the
cutter's split state (``VNodeCutter.state_arrays``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.bully.direct import DirectSim
from alloy.topology.blade_contact import quat_rotate
from alloy.topology.bully_target import FramedActions, keep_blades
from alloy.topology.cut_sweep import SweptEdges
from alloy.topology.meshes import grid_tet_mesh
from alloy.topology.tet_copies import TetCopies, add_tet_copies, recolor_copies
from alloy.topology.types import Blade
from alloy.topology.vnode_cut import VNodeCutter

LEAD = 1.6e-3                  # [m] split lead: edge radius + reach + particle radius of the seam rig


@dataclass
class VNodeScene:
    model: newton.Model
    copies: TetCopies                 # model indices, all worlds
    blades: list[Blade]
    blade_bodies: np.ndarray          # [W]
    clamp: np.ndarray                 # clamped hub particles (x = 0)
    size: tuple[float, float, float]
    hubs: int                         # mesh particles per world


def _tile(c: TetCopies, worlds: int, particles: int, tets: int, springs: int) -> TetCopies:
    """The template's copies for every world of ``replicate`` (per-world index offsets)."""
    def tile(a: np.ndarray, step: int) -> np.ndarray:
        return np.concatenate([a + w * step for w in range(worlds)])

    rep = lambda a: np.concatenate([a] * worlds)  # noqa: E731
    return replace(c, tet=tile(c.tet, tets), hub=tile(c.hub, particles), hub_row=tile(c.hub_row, len(c.node_hub)),
                   copy_tet=tile(c.copy_tet, tets), material=rep(c.material), mass=rep(c.mass),
                   node_hub=tile(c.node_hub, particles), node=tile(c.node, particles), weld=tile(c.weld, springs),
                   weld_ke=rep(c.weld_ke), edge=tile(c.edge, len(c.node_hub)), tet_edge=tile(c.tet_edge, len(c.edge)),
                   tet_edge_flip=rep(c.tet_edge_flip))


def vnode_block(device: str, *, worlds: int = 1, cells: tuple[int, int, int] = (4, 3, 3), cell: float = 5.0e-3,
                k_mu: float = 5.0e4, k_lambda: float = 5.0e4, density: float = 1000.0, gravity: float = 0.0,
                copies: bool = True) -> VNodeScene:
    """The block clamped at x = 0 with ``presplit_block``'s blade per world; ``copies=False`` builds the plain
    block (the TOP-4 reference)."""
    from alloy.testing.topology import clamp_particles, soft_kwargs

    t = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    v, tets = grid_tet_mesh(*cells, cell)
    t.add_soft_mesh(vertices=[wp.vec3(*p) for p in v], indices=tets.ravel().tolist(),
                    **soft_kwargs(k_mu=k_mu, k_lambda=k_lambda, density=density))
    hubs = t.particle_count
    rec = add_tet_copies(t, range(len(t.tet_indices)), density) if copies else None
    blade = t.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
    t.add_shape_box(blade, hx=0.0005, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    counts = (t.particle_count, len(t.tet_indices), len(t.spring_stiffness))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    b.replicate(t, worlds)
    b.color()
    tiled = _tile(rec, worlds, *counts) if rec is not None else None
    if tiled is not None:
        recolor_copies(b, tiled)
    m = b.finalize(device=device)
    x0 = m.particle_q.numpy()
    is_hub = (np.arange(m.particle_count) % counts[0]) < hubs
    clamp = np.flatnonzero(np.isclose(x0[:, 0], 0.0) & is_hub)
    clamp_particles(m, clamp)
    bodies = np.flatnonzero(m.body_world.numpy() >= 0)
    blade_bodies = bodies[np.argsort(m.body_world.numpy()[bodies], kind="stable")]
    blades = [Blade(body=int(bb), shape=-1, edge_a=(0.0, -0.05, -0.02), edge_b=(0.0, 0.05, -0.02))
              for bb in blade_bodies]
    return VNodeScene(m, tiled, blades, blade_bodies, clamp, tuple(float(n * cell) for n in cells), hubs)


def cut_face_points(rest: np.ndarray, copies: TetCopies, edge_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rest-frame corners of the cut faces: ([E, 3] each mesh edge's crossing point, [E] crossed). A crossing's edge
    parameter survives a tet's affine map, so the rest point is exact."""
    hub = copies.node_hub[copies.edge]
    crossed = edge_t >= 0.0
    t = np.where(crossed, edge_t, 0.0)[:, None]
    return rest[hub[:, 0]] + t * (rest[hub[:, 1]] - rest[hub[:, 0]]), crossed


def behind_edges(points: np.ndarray, world: np.ndarray, blades, body_q: np.ndarray) -> np.ndarray:
    """[N] bool: world ``points`` [N, 3] level with or behind their world's blade edge (on its spine side), as
    ``vnode_contacts.behind`` in the world frame. A cutter's ``CutterBlades`` (no spine: always behind; no blade:
    never)."""
    j = blades.of_world.numpy()[world]
    ok = j >= 0
    js = np.where(ok, j, 0)
    q = np.asarray(body_q, np.float64)[blades.body.numpy()[js]]
    ea = q[:, :3] + quat_rotate(q[:, 3:7], blades.edge_a.numpy()[js].astype(np.float64))
    eb = q[:, :3] + quat_rotate(q[:, 3:7], blades.edge_b.numpy()[js].astype(np.float64))
    spine = quat_rotate(q[:, 3:7], blades.spine.numpy()[js].astype(np.float64))
    ab = eb - ea
    t = np.clip(np.einsum("ij,ij->i", points - ea, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30), 0.0, 1.0)
    side = np.einsum("ij,ij->i", points - (ea + t[:, None] * ab), spine)
    return ok & ((side >= 0.0) | (np.einsum("ij,ij->i", spine, spine) == 0.0))


class VNodeSim(FramedActions, DirectSim):
    """DirectSim with the virtual-node cutter and kinematic blades (and an optional blade-frame action frame)."""

    def __init__(self, scene: VNodeScene, solver: newton.solvers.SolverVBD, lead: float = LEAD, **kw) -> None:
        super().__init__(scene.model, solver, **kw)
        self.scene = scene
        self.cutter = VNodeCutter(scene.model, scene.copies, scene.blades, lead) if scene.copies else None
        self.blades = wp.array(np.asarray(scene.blade_bodies, np.int32), dtype=wp.int32, device=scene.model.device)
        from alloy.core.snapshot import WorldSnapshot

        cut = self.cutter
        self.snap = WorldSnapshot(scene.model, self.state, extra=cut.state_arrays() if cut else None,
                                  extra_world=cut.state_worlds() if cut else None)
        self.snap.save(self.state)       # the start: blade poses as set before construction, split model arrays
        self._inv = np.zeros(int(scene.model.world_count))
        if self.health is not None:
            from alloy.topology.soft_strain import SoftStrain

            self.health.set_soft_strain(SoftStrain(scene.model))     # STRAIN_RANGE (ruling 2026-11-06T14:17Z)
        self.sweep = None
        if self.cutter is not None:
            world = scene.model.particle_world.numpy()[scene.copies.node_hub[scene.copies.edge[:, 0]]]
            self._edge_world = np.maximum(world, 0)
            self.sweep = SweptEdges(scene.blades, scene.model.body_world.numpy(), world, int(scene.model.world_count))
            self.sweep.reset(self.state.body_q.numpy())

    def cut_face_deviation(self) -> np.ndarray:
        """[W] the max distance of the cut faces' rest-frame corners (mesh-edge crossings) from the edge's swept
        surface in the episode (SB-7 row ``cut_face_deviation``; each call samples the edges). A crossing counts once
        the edge has passed it (``behind_edges`` on its current position): a split runs up to a cell ahead of the
        edge, and counted from its split a crossing read that look-ahead (6 mm mid-stroke on cut_block_free_copies,
        falling 0.8 mm a tick as the blade came down), not the cut's path."""
        edge_t = self.cutter.edge_t.numpy().astype(np.float64)
        copies, body_q = self.scene.copies, self.state.body_q.numpy()
        pts, crossed = cut_face_points(self.model.particle_q.numpy().astype(np.float64), copies, edge_t)
        now, _ = cut_face_points(self.state.particle_q.numpy().astype(np.float64), copies, edge_t)
        crossed &= behind_edges(now, self._edge_world, self.cutter.blades, body_q)
        return self.sweep.update(body_q, crossed, pts)

    def live_inverted(self) -> np.ndarray:
        """[W] live tets (nonzero material: whole tets and fragments) with det F <= 0 now."""
        m = self.model
        mat, tets = m.tet_materials.numpy(), m.tet_indices.numpy()
        live = np.flatnonzero((mat[:, 0] != 0.0) | (mat[:, 1] != 0.0))
        x = self.state.particle_q.numpy()[tets[live]]
        f = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2) @ m.tet_poses.numpy()[live]
        world = m.particle_world.numpy()[tets[live, 0]]
        return np.bincount(world[np.linalg.det(f) <= 0.0], minlength=int(m.world_count))

    def bully_monitors(self) -> dict[str, np.ndarray]:
        """SB-7 rows as ``CutSim``'s: no spare pool (0), the episode's peak live inverted tets, the cut-face
        deviation (each call samples the blade edges)."""
        self._inv = np.maximum(self._inv, self.live_inverted())
        w = int(self.model.world_count)
        return {"cut_pool_overflow": np.zeros(w), "inverted_tets": np.asarray(self._inv, np.float64).copy(),
                "cut_face_deviation": self.cut_face_deviation()}

    def substep(self) -> None:
        s, nxt, dev = self.state, self._next, self.model.device
        s.clear_forces()
        for hook in self.pre_substep:
            hook(s)
        if self.cutter is not None:
            self.cutter.pre_step(s)
        self.pipeline.collide(s, self.contacts)
        if self.cutter is not None:
            self.cutter.filter_contacts(s, self.contacts)
        self.solver.step(s, nxt, self.control, self.contacts, self.dt)
        wp.launch(keep_blades, dim=self.blades.shape[0], inputs=[self.blades, s.body_q, s.body_qd, nxt.body_q,
                                                                 nxt.body_qd], device=dev)
        for fn in self.observers:
            fn(s, nxt, self.control, self.contacts, self.dt)
        self.state, self._next = nxt, s
        self.clock.advance(1)

    def reset(self, world_mask: wp.array | None = None) -> None:
        """The start for all worlds, or those with ``world_mask[w] != 0``: state, the model arrays a split rewrites
        and the cutter's split state (``snap``, saved at construction), as ``CutSim.reset``. DirectSim's own initial
        state alone put the blades back at their build pose, 1 m up (``make_sim`` sets the poses after the build): the
        next tick's pose pulled them down at 60 m/s."""
        super().reset(world_mask)
        sel = np.ones(int(self.model.world_count), bool)
        mask = None
        if world_mask is not None:
            sel = np.asarray(world_mask.numpy() if hasattr(world_mask, "numpy") else world_mask).astype(bool)
            mask = wp.array(sel.astype(np.int32), dtype=wp.int32, device=self.model.device)
        self.snap.restore(self.state, mask)
        if self.sweep is not None:
            self.sweep.reset(self.state.body_q.numpy(), sel)
        self._inv[sel] = 0
