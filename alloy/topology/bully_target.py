# SPDX-License-Identifier: Apache-2.0
"""SB-7 bully target ``cut_block_stroke`` (WP10 for WP7's cutting class, 2026-11-05).

The SC5-3 cut-stroke rig as a B0 target: per world a 20 x 15 x 15 mm pre-split tet block (TET_CUT seams x = 10 mm and
y = 10 mm, 1 mm particles) clamped at x = 0, no gravity, and a 1 mm blade box on a kinematic body. It runs real soft
contacts, side-aware ``BladeContact``, and the health monitor's cut-front predicate (SCR-SC5-3). TOP-3's fuzz rig steps
VBD without contacts, so its blade never touches the material.

Cut faces collide (2026-11-08, gap #9; WP15's bisect of the S1 jump to WP5's SOFT-3-1 check, a10f7a04d): the VBD
runs self-contact at the SC5-2 press rig's settings (``SELF_MARGIN`` 0.5 mm, ``SELF_GAP`` 0.25 mm, rest exclusion 0)
with the controller's seam filter, so welded twins ignore each other and a broken seam's halves collide instead of
passing through. S1 on seed 100000, worlds 0-9 (CPU): 8 -> 6 of 10 on main, 4 -> 2 with gap #9's face plane; about
1.5x the time per world. ``cut_block_free`` and ``cut_block_free_copies`` still run without it: their broken faces
pass through each other, so read their PEN_FAIL S1s as that until they get the same (follow-up).

``CutSim`` is a ``DirectSim`` whose substep is::

    pre_substep hooks (WP7's body_pose writer) -> ctl.pre_step -> collide -> BladeContact.adjust_contacts -> VBD
    -> keep the blades kinematic -> observers -> ctl.apply

``reset`` also restores the controller's bond, damage and cut arrays (``WorldSnapshot``) and clears the S1 counts.

Actions (``action_kind="body_pose"``, WP7's ``HookedPoseActions``): the blade body origin (x, y, z), world frame,
rate-limited to 0.05 m/s (SC5-3's stroke speed). Every world starts as SC5-3's stroke does: blade face in the x = 10 mm
seam, edge along y, 2 mm above the block. The variety is the bully's paths.
- x stays within 1 mm of the seam. A pre-split block can only split on its seams, so pushing the blade into seamless
  material is a known model limit, not a finding.
- y saws along the edge (+-17.5 mm about the block centre).
- z goes from 2 mm above the block to 3 mm past its bottom.

S1 rows (``CutSim.bully_monitors``, rows registered by WP7): per world, spare-pool overflow events in the episode
(``cut_pool_overflow``) and the episode's peak count of inverted tets, det F <= 0 (``inverted_tets``). Diagnostic row
``cut_face_deviation``: the episode max distance (m) of an opened twin from the blade edge's swept surface
(``alloy.topology.cut_sweep``), sampled once per call, i.e. per control period in a bully episode.

The blade cuts only the seam it lies in (2026-11-09, gap #9 wave 2; ``Config.cut_alignment`` = ``CUT_ALIGNMENT``).
Its edge crosses the y = 10 mm seam, and the ungated cut damage softened and broke that seam's bonds along the
crossing line, though no blade face ever opens them. ``cut_block_free`` runs gated too: on its lattice the gate took
S1 from 1,954 to 1,324 of 2,048 (seed 100000, L4; S2 1 -> 16), with the same clean cut depth (15.5 -> 15.9 mm).

``cut_block_free`` (item 9 step 2, agreed with WP7 2026-11-06) frees the blade:
- Lattice pre-split: a seam on every 5 mm cell face (x, y and z), so a cut can stair-step along any path.
- Per world, a fixed blade yaw (about z, +-30 deg) and pitch (a lean about the edge: the face normal tilts +-15 deg
  from horizontal), drawn from ``FREE_SEED`` and the world's index (``free_angles``). Build a replay of batch world k
  alone with ``cut_block_free(first_world=k)``; the registered factory starts at world 0.
- Actions stay translation-only, in each world's blade frame (WP7's ``bully_action_frame``; coordinator ruling
  2026-11-06T14:17Z): face normal +-1 mm (``SEAM_BAND``, as ``cut_block_stroke``'s x band: pushing a flat blade
  into the soft block is outside a cutting task), the edge saws +-17.5 mm, and the cutting side runs from the start
  (edge 2 mm above the block at its x, y centre) to 3 mm past its bottom at the steepest lean (``FREE_DEPTH``).
  Bully traces that drove the face through the block ploughed it (tets at p90 85 % strain before a cut). The same
  band holds for ``cut_block_free_copies``; S1 still counts within the legal actions.

Known gap (model limit): large compression of soft FEM under a flat tool. Material that parts only where an edge
passes is squeezed without limit by a face pushed into it; ``soft_strain`` measures it for the health flag (60 %
principal strain). Large-deformation remeshing is post-launch.
"""

from __future__ import annotations

import functools

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.bully.direct import DirectSim
from alloy.bully.targets import register_target
from alloy.bully.types import ActionSpace, BullyTarget
from alloy.topology.blade_contact_kernels import SEAM_ALIGNMENT
from alloy.topology.cut_sweep import SweptEdges, twin_points

CELLS = (4, 3, 3)                 # [cells] the cut-stroke rig's block (5 mm cells)
CONTROL_DT, SUBSTEPS = 0.016, 16  # 1 ms substeps, as the cut-stroke rig
SPEED = 0.05                      # [m/s] blade rate limit: SC5-3's stroke speed
EDGE_OFFSET = 0.02                # [m] blade body origin above its edge (presplit_block's blade)
SEAM_X = 0.010                    # [m] the TET_CUT seam the blade face lies in
SEAM_BAND = 0.001                 # [m] x range about the seam
STROKE_KE = 3.0e3                 # [N/m] soft contact ke of the cut-stroke rig
SELF_MARGIN, SELF_GAP = 5.0e-4, 2.5e-4   # [m] VBD self-contact of the cut faces: the SC5-2 press rig (5 mm cells)
CUT_ALIGNMENT = SEAM_ALIGNMENT    # cut_block_stroke: the blade cuts only seams aligned with its face (the y seam
                                  # stays welded); the blade contact's min_alignment
SIZE = tuple(5.0e-3 * c for c in CELLS)
TOP = SIZE[2]
FREE_SEED = 20261106              # cut_block_free: per-world blade angles from (FREE_SEED, world index)
FREE_YAW = np.deg2rad(30.0)       # [rad] blade yaw about world z, uniform in +-FREE_YAW
FREE_PITCH = np.deg2rad(15.0)     # [rad] blade lean about its edge, uniform in +-FREE_PITCH
LATTICE = 5.0e-3                  # [m] cut_block_free seam spacing: every cell face
FREE_DEPTH = (2.0e-3 + 3 * 5.0e-3 + 3.0e-3) / np.cos(FREE_PITCH)  # [m] cut_block_free: travel along the cutting side


@wp.kernel
def keep_blades(body: wp.array(dtype=wp.int32), q_in: wp.array(dtype=wp.transform),
                qd_in: wp.array(dtype=wp.spatial_vector), q_out: wp.array(dtype=wp.transform),
                qd_out: wp.array(dtype=wp.spatial_vector)):
    """Kinematic blades: the solver's output keeps the commanded pose and velocity."""
    i = wp.tid()
    b = body[i]
    q_out[b] = q_in[b]
    qd_out[b] = qd_in[b]


def start_origin() -> np.ndarray:
    """The blade body origin at the start: SC5-3's stroke start (edge 2 mm above the block, in the x seam)."""
    return np.array([SEAM_X, 0.5 * SIZE[1], TOP + 2.0e-3 + EDGE_OFFSET])


def item_worlds(model: newton.Model, ctl, extra: dict[str, wp.array]) -> dict[str, wp.array]:
    """World of each item of the controller's state arrays (bonds, springs, particles, bodies), for per-world reset;
    arrays with one entry per world are left to ``WorldSnapshot``."""
    dev = model.device
    spring_world = model.particle_world.numpy()[model.spring_indices.numpy().reshape(-1, 2)[:, 0]]
    by_kind = {"spring": spring_world, "particle": model.particle_world.numpy(), "body": model.body_world.numpy()}
    out = {}
    for key, arr in extra.items():
        kind = next((k for k in by_kind if f".{k}_" in key or key.endswith(f".{k}_active")), None)
        if kind is not None and arr.shape[0] == by_kind[kind].shape[0]:
            out[key] = wp.array(by_kind[kind].astype(np.int32), dtype=wp.int32, device=dev)
        elif arr.shape[0] == ctl.bonds.count:
            out[key] = ctl.bonds.world
    return out


class FramedActions:
    """Mixin for a target sim: WP7's per-world body_pose action frame (``alloy.bully.actions.framed_pose_trace``)."""

    action_frame = None  # (origin [W, 3], axes [W, 3, 3]): blade-frame body_pose actions (cut_block_free), else None

    def bully_action_frame(self):
        """The per-world action frame; None: world coordinates."""
        return self.action_frame


class CutSim(FramedActions, DirectSim):
    """DirectSim with the topology controller, side-aware blade contact and kinematic blades."""

    def __init__(self, model: newton.Model, solver: newton.solvers.SolverVBD, ctl, blade_contact,
                 blade_bodies: np.ndarray, **kw) -> None:
        super().__init__(model, solver, **kw)
        from alloy.core.snapshot import WorldSnapshot

        self.ctl, self.blade_contact = ctl, blade_contact
        self.blades = wp.array(np.asarray(blade_bodies, np.int32), dtype=wp.int32, device=model.device)
        extra = ctl.state_arrays()
        self.snap = WorldSnapshot(model, self.state, extra=extra, extra_world=item_worlds(model, ctl, extra))
        self.snap.save(self.state)
        if self.health is not None:
            from alloy.topology.soft_strain import SoftStrain

            self.health.set_cut_front(ctl.cut_front_blade, ctl.blade_half_thickness)
            self.health.set_soft_strain(SoftStrain(model))      # STRAIN_RANGE (ruling 2026-11-06T14:17Z)
        w = int(model.world_count)
        self._ovf0 = np.zeros(w, np.int64)
        self._inv = np.zeros(w, np.int64)
        self._twins, twin_world, self._twin_points = twin_points(ctl.bonds, model.particle_q.numpy())
        self.sweep = SweptEdges(blade_contact.blades, model.body_world.numpy(), twin_world, w)
        self.sweep.reset(self.state.body_q.numpy())

    def substep(self) -> None:
        s, nxt, dev = self.state, self._next, self.model.device
        s.clear_forces()
        for hook in self.pre_substep:
            hook(s)
        self.ctl.pre_step(s, self.contacts, self.dt)
        self.pipeline.collide(s, self.contacts)
        self.blade_contact.adjust_contacts(s, self.contacts)
        self.solver.step(s, nxt, self.control, self.contacts, self.dt)
        wp.launch(keep_blades, dim=self.blades.shape[0], inputs=[self.blades, s.body_q, s.body_qd, nxt.body_q,
                                                                 nxt.body_qd], device=dev)
        for fn in self.observers:
            fn(s, nxt, self.control, self.contacts, self.dt)
        self.ctl.apply(nxt)
        self.state, self._next = nxt, s
        self.clock.advance(1)

    def reset(self, world_mask: wp.array | None = None) -> None:
        super().reset(world_mask)
        sel = np.ones(int(self.model.world_count), bool)
        mask = None
        if world_mask is not None:
            sel = np.asarray(world_mask.numpy() if hasattr(world_mask, "numpy") else world_mask).astype(bool)
            mask = wp.array(sel.astype(np.int32), dtype=wp.int32, device=self.model.device)
        self.snap.restore(self.state, mask)
        self._ovf0[sel] = self._overflow()[sel]
        self._inv[sel] = 0
        self.sweep.reset(self.state.body_q.numpy(), sel)

    def _overflow(self) -> np.ndarray:
        return np.asarray(self.ctl.ledger().overflow, np.int64).reshape(-1)

    def inverted_tets(self) -> np.ndarray:
        """Per world, tets with det F <= 0 now (F = Ds Dm^-1; Newton's tet_poses is Dm^-1)."""
        m = self.model
        tets = m.tet_indices.numpy()
        x = self.state.particle_q.numpy()[tets]
        ds = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
        det = np.linalg.det(ds @ m.tet_poses.numpy())
        world = m.particle_world.numpy()[tets[:, 0]]
        return np.bincount(world[det <= 0.0], minlength=int(m.world_count))

    def bully_monitors(self) -> dict[str, np.ndarray]:
        """WP7's rows for the episode so far: spare-pool overflow events and the peak inverted-tet count (S1), and
        the opened-face deviation (diagnostic; each call samples the blade edges)."""
        self._inv = np.maximum(self._inv, self.inverted_tets())
        return {"cut_pool_overflow": self._overflow() - self._ovf0, "inverted_tets": self._inv.copy(),
                "cut_face_deviation": self.sweep.update(self.state.body_q.numpy(),
                                                        self.ctl.bonds.intact.numpy()[self._twins] == 0,
                                                        self._twin_points)}


def cut_sim(sc, poses: np.ndarray, self_contact: bool = False, cut_alignment: float | None = CUT_ALIGNMENT,
            reentry: bool = True) -> CutSim:
    """A ``CutSim`` over a ``presplit_block`` scene with blade body poses [W, 7] (one free joint per world).
    ``self_contact``: the block's cut faces collide (VBD self-contact at the SC5-2 press rig's settings, the
    controller's seam filter ignoring twins until their bond breaks); without it broken halves pass through each other.
    ``cut_alignment``: the controller's cut damage gate (``TopologyController.Config``; None: any seam).
    ``reentry``: the blade contact's re-entry case (``BladeContactConfig``; False: the rule before)."""
    from alloy.topology.blade_contact import BladeContact, BladeContactConfig
    from alloy.topology.bonds import bonds_from_model
    from alloy.topology.controller import TopologyController
    from alloy.topology.types import TopologyBudget

    m = sc.model
    m.soft_contact_ke = STROKE_KE
    q, starts = m.joint_q.numpy(), m.joint_q_start.numpy()
    for w, pose in enumerate(poses):
        q[starts[w]:starts[w] + 7] = pose
    m.joint_q.assign(q)
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(max_events_per_step=64), sc.blades,
                             config=TopologyController.Config(cut_alignment=cut_alignment))
    bc = BladeContact(m, ctl.bonds, sc.blades, cut_alignment=cut_alignment, config=BladeContactConfig(reentry=reentry))
    contact = {}
    if self_contact:
        contact = {"particle_enable_self_contact": True, "particle_self_contact_margin": SELF_MARGIN,
                   "particle_self_contact_gap": SELF_GAP, "particle_rest_shape_contact_exclusion_radius": 0.0}
    solver = newton.solvers.SolverVBD(m, iterations=4, **contact)
    if self_contact:
        ctl.attach_self_contact(solver)
    return CutSim(m, solver, ctl, bc, sc.blade_bodies, control_dt=CONTROL_DT, substeps=SUBSTEPS)


def cut_block_stroke(self_contact: bool = True, cut_alignment: float | None = CUT_ALIGNMENT,
                     reentry: bool = True) -> BullyTarget:
    """The SB-7 cutting target (see the module docstring); ``self_contact=False`` is the target before its cut faces
    collided and ``cut_alignment=None`` the one whose blade also cut the y seam (for the laws that compare them)."""
    from alloy.testing.topology import blade_pose, presplit_block

    start = start_origin()

    def make_sim(worlds: int, device: str) -> CutSim:
        sc = presplit_block(device, worlds=worlds, cells=CELLS, gravity=0.0, k_mu=5.0e4, k_lambda=5.0e4)
        pose = blade_pose(start, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]))
        return cut_sim(sc, np.tile(pose, (worlds, 1)), self_contact=self_contact, cut_alignment=cut_alignment,
                       reentry=reentry)

    low = np.array([SEAM_X - SEAM_BAND, 0.5 * SIZE[1] - 0.0175, -3.0e-3 + EDGE_OFFSET])
    high = np.array([SEAM_X + SEAM_BAND, 0.5 * SIZE[1] + 0.0175, start[2]])
    actions = ActionSpace(low=low, high=high, rate_hz=1.0 / CONTROL_DT,
                          max_delta_per_tick=np.full(3, SPEED * CONTROL_DT), initial=start)
    name = "cut_block_stroke" if self_contact else "cut_block_stroke_noself"    # records never pass for the class's
    name += "" if cut_alignment is not None else "_anyseam"
    name += "" if reentry else "_noreentry"
    return BullyTarget(name=name, scene=None, make_sim=make_sim, actions=actions,
                       action_kind="body_pose", action_index=(0, 1, 2), robot_bodies=(0,))


def free_start() -> np.ndarray:
    """cut_block_free's blade body origin at the start: edge 2 mm above the block at its x, y centre."""
    return np.array([0.5 * SIZE[0], 0.5 * SIZE[1], TOP + 2.0e-3 + EDGE_OFFSET])


def free_angles(worlds: int, first_world: int = 0) -> np.ndarray:
    """[worlds, 2] (yaw, pitch) [rad] of worlds first_world, first_world + 1, ...: uniform in +-FREE_YAW and
    +-FREE_PITCH, from ``FREE_SEED`` and the world's index alone, so a world keeps its blade in any batch."""
    out = np.empty((worlds, 2))
    for i in range(worlds):
        out[i] = np.random.default_rng([FREE_SEED, first_world + i]).uniform(-1.0, 1.0, 2) * (FREE_YAW, FREE_PITCH)
    return out


def free_pose(origin: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """Blade body pose (x, y, z, qx, qy, qz, qw): edge along (-sin yaw, cos yaw, 0); the blade leans about its edge,
    its edge-side axis (body -z) tilted by ``pitch`` from straight down toward the face normal (cos yaw, sin yaw, 0)."""
    from alloy.testing.topology import blade_pose

    normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    edge = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    down = np.cos(pitch) * np.array([0.0, 0.0, -1.0]) + np.sin(pitch) * normal
    return blade_pose(np.asarray(origin, np.float64), edge, down)


def cut_block_free(first_world: int = 0, cut_alignment: float | None = CUT_ALIGNMENT) -> BullyTarget:
    """The SB-7 free cutting target (see the module docstring); ``first_world`` numbers the batch's first world.
    ``cut_alignment=None`` is the target under the rule before the gate (``cut_block_free_anyseam``)."""
    from alloy.bully.actions import frame_from_poses
    from alloy.testing.topology import presplit_block
    from alloy.topology.presplit import SplitPlan

    start = free_start()

    def make_sim(worlds: int, device: str) -> CutSim:
        plan = SplitPlan(spacing=LATTICE, axes=(0, 1, 2))
        sc = presplit_block(device, worlds=worlds, cells=CELLS, plan=plan, gravity=0.0, k_mu=5.0e4, k_lambda=5.0e4)
        poses = np.stack([free_pose(start, yaw, pitch) for yaw, pitch in free_angles(worlds, first_world)])
        sim = cut_sim(sc, poses, cut_alignment=cut_alignment)
        sim.action_frame = frame_from_poses(poses)  # blade frame: face normal, edge, cutting side
        return sim

    low = np.array([-SEAM_BAND, -0.0175, 0.0])
    high = np.array([SEAM_BAND, 0.0175, FREE_DEPTH])
    actions = ActionSpace(low=low, high=high, rate_hz=1.0 / CONTROL_DT,
                          max_delta_per_tick=np.full(3, SPEED * CONTROL_DT), initial=np.zeros(3))
    name = "cut_block_free" if cut_alignment is not None else "cut_block_free_anyseam"
    return BullyTarget(name=name, scene=None, make_sim=make_sim, actions=actions,
                       action_kind="body_pose", action_index=(0, 1, 2), robot_bodies=(0,))


def cut_block_free_copies(first_world: int = 0) -> BullyTarget:
    """``vnode_target.cut_block_free_copies`` (item 9 step 3), registered here so ``get_target`` finds it through
    WP7's provider list; imported on first use, since ``vnode_rig`` imports this module."""
    from alloy.topology.vnode_target import cut_block_free_copies as build

    return build(first_world)


register_target("cut_block_stroke", cut_block_stroke)
# the stroke under the rule before the alignment gate, so the board, the triage and corpus replays can still run it
# beside the counted class (WP6, AB-laws-7: a population change stays measurable)
register_target("cut_block_stroke_anyseam", functools.partial(cut_block_stroke, cut_alignment=None))
# the stroke under the blade contact's far-face rule before the re-entry case (gap #9 slice 3), for the same reason
register_target("cut_block_stroke_noreentry", functools.partial(cut_block_stroke, reentry=False))
register_target("cut_block_stroke_noself_anyseam",
                functools.partial(cut_block_stroke, self_contact=False, cut_alignment=None))
register_target("cut_block_free", cut_block_free)
register_target("cut_block_free_anyseam", functools.partial(cut_block_free, cut_alignment=None))
register_target("cut_block_free_copies", cut_block_free_copies)
