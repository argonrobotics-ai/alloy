# Alloy baseline benchmark suite

Written 2026-10-31 against Newton `ac7f7830` (1.8.0.dev0), warp-lang 1.18.0, mujoco / mujoco-warp 3.16.0.
This is the **baseline**: stock Newton solvers on 30 scenarios, before any Alloy layer exists. Every later Alloy
change is scored against these rows. The validation-bench design (rig laws, bully agent, scoreboard) is in
`docs/research/bench.md` and `docs/SPEC.md`. This suite is the speed ladder plus the analytic laws those documents
call A1–A15 and D1–D3. IDs here are bench-local and the A/D cross-references are given per row.

- **Harness:** `bench/harness.py` (Scenario, Config, Built, runner), `bench/metrics.py` (host-side measure
  helpers), `bench/run.py` (CLI). Smoke test: `bench/scenarios/_example.py`.
- **Ownership:** one builder agent per family. Each owns `bench/scenarios/<key>.py`, and may add helper modules
  `bench/scenarios/<key>_*.py` and assets under `bench/assets/<key>/`. Builders do not edit `harness.py`,
  `metrics.py` or `run.py`. Ask the suite owner for harness changes.
- **Rule:** Newton is the only engine in the loop. External engines (CPU MuJoCo, Genesis, IPC, PhysX) run
  **offline** as references. Their rows go to `bench/results/<family>/<scenario>.ref.jsonl`, with
  `solver = "ref:<engine>"` and the same schema.

```
python -m bench.run --family rigid --device cuda:0 --ladder 1,16,256,1024,4096
python -m bench.run --family precision --scenario P01_peg_hole_0p2 --config sdf_mujoco --device cuda:0
python -m bench.run --family _example --device cpu --ladder 1,16          # Mac CPU smoke test
python -m bench.run --summary bench/results/rigid
```

## 1. Families and scenarios

| Key | Family | # | Object types | Hard cases |
|---|---|---|---|---|
| `rigid` | Rigid bodies and articulations | 6 | primitives, boxes, robot arm | stack creep, friction-cone onset, dense piles |
| `precision` | Precision contact | 5 | SDF meshes, hydroelastic pairs | 0.2 mm clearance, threads, latches, tunnelling |
| `deform` | Deformables | 6 | rods/cables, cloth, volumetric soft | elastica sag, drape, pinch-and-hold, squeeze |
| `granular` | Granular and fluids (MPM) | 4 | sand, water, viscous paste | runout, dam-break front, volume loss, digging |
| `coupled` | Coupled multi-solver | 5 | robot + cable / sand / cloth / soft | interface forces, mass ratio, multi-rate |
| `scale` | Scale and robustness | 4 | ant, humanoid, piles, cloth, MPM | determinism, sleeping, graphs, memory, overflow |

Families follow the brief, with three changes:
- Hertz contact and tunnelling sit in `precision`, because they test the contact layer.
- Blade digging sits in `granular`. The blade is a one-way kinematic collider, so it is not coupling.
- CUDA graphs on/off is a config axis of the throughput scenario (S03), not a separate scenario. The same scenes
  give both numbers.

### Conventions every scenario follows

- **Conditions tile over worlds.** A scenario with a condition list (tilts, loads, materials) gives world `w`
  condition `w % ncond`. Speed rows are valid at every rung. The accuracy law is complete only when
  `nworld >= ncond`, which the scenario states in `metrics.conditions_covered`. Report per-condition values in
  `metrics.per_condition`.
- **No host sync inside `step()`.** Scripted motion (robot targets, plate or blade drives) runs from device arrays
  indexed by a device step counter, so CUDA graphs work. A scenario that cannot do this sets `host_control = True`,
  which turns graphs off. The record shows it as `cuda_graph: false`.
- **Loads are eased in** over ≥ 0.5 s (CON-51: a stepped gravity tilt slipped 4° early).
- **Contact capacity is set explicitly** (`rigid_contact_max`, `nconmax`, `njmax`), and the overflow counter is
  read in `finish()`. A dropped contact is recorded as `metrics.contact_overflow > 0`, never ignored. Default
  capacities have produced 17 m errors (`[internal reference]` §1).
- **Units.** Use SI internally. A printed field in another unit carries the suffix (`_mm`, `_deg`, `_pct`).
- **Ladder.** The default is `1,16,256,1024,4096`. A config caps it with `max_nworld` where memory or CPU time
  says so. On H100 runs, add `8192,16384` for the S03 scenes. Larger rungs after an error are recorded as
  `skipped`.
- **Production settings.** A law passed only at a special dt or iteration count is reported with
  `metrics.fit_dt_only = true` (docs/research/bench.md §4.1).

Paths below are relative to the Newton clone `[internal reference]` (ac7f7830), unless they
start with `alloy/` (this repo) or `pa/` (`[internal reference]`).

### 1.1 `rigid`: rigid bodies and articulations

| ID | Scenario | Configs to compare | Derives from | Law / reference | Scenario metrics |
|---|---|---|---|---|---|
| R01_free_fall_bounce | Sphere free fall and projectile (no ground), then a drop onto a plane with e ∈ {0, 0.5, 0.8} | mujoco (MJWarp), featherstone, xpbd, semi_implicit, kamino (dvi), steel; dt 1/600 | `alloy/bench/scenarios/_example.py`; `newton/tests/test_physics_verification.py:85,665-760` | z = z0 + v0z·t − ½gt², x = v0x·t, error ≤ 2·½g·h·t (A1). Bounce h₁ = e²h₀ within 2 %·h + 5·dt (A6) | traj_err_mm, law_pass, bounce_err_pct per e |
| R02_pendulum | Single pendulum at 5° and 90°, plus the double pendulum for energy | mujoco, featherstone, xpbd, kamino, steel, semi_implicit | `newton/examples/basic/example_basic_pendulum.py` | T₀ = 2π√(I_p/(mgd)); at 90°, T = 4√(I_p/(mgd))·K(sin 45°), with K = 1.85407. Period within 0.1 % at dt 1 ms (A2). Energy drift ≤ 0.5 % over 2 s (A3, `test_physics_verification.py:236`) | period_err_pct, energy_drift_pct |
| R03_incline_friction | Box on a plane with gravity tilted (eased); μ ∈ {0.3, 0.6}; 16 slide directions | mujoco (elliptic / pyramidal / noslip), kamino, xpbd, steel, featherstone, semi_implicit | `newton/tests/test_rigid_friction_ramp.py:68-88,391`; `alloy/docs/research/bench_probes/ramp_onset.py` | Coulomb (A5). Onset \|θ_s − atan μ\| ≤ 0.1°, bisected to 0.05°. a = g(sin θ − μ cos θ) within 1 %. Stop distance v₀²/(2μg) within 1 %. Creep at θ_c − 1° ≤ 0.01 mm per 10 s. Spread ≤ 1 % over the 16 directions. Prior: Kamino creeps 0.17 mm, XPBD 5.8 mm per 10 s at −1° | onset_err_deg, accel_err_pct, stop_err_pct, creep_mm_per_10s, isotropy_pct |
| R04_box_stack_10 | Tower of 10 cubes (5 cm), 10 s rest | mujoco (newton/cg, elliptic), kamino, xpbd, steel, featherstone, semi_implicit, vbd (rigid) | `newton/examples/contacts/example_pyramid.py`; `newton/examples/steel/example_steel_sleeping.py` (towers) | Static equilibrium. Top-cube drift ≤ 0.1 mm per 10 s (bench.md acceptance #8). No energy gain > 1 mJ (CON-30). Penetration ≤ 1 mm | top_drift_mm_per_10s, tilt_deg, energy_gain_mj, penetration_p99_mm |
| R05_box_pile | 64 and 256 mixed boxes, spheres and capsules dropped into a bin, 4 s (params n = 64, 256) | mujoco (MuJoCo contacts and Newton pipeline), kamino, xpbd, steel, featherstone | `pa/bench/tools/bin_scene.py` (MJCF), `pa/bench/tools/newton_bin.py`, `pa/bench/tools/fidelity.py` (bin_metrics) | No escapes. KE → 0 after settling. Rest penetration ≤ 1 mm. Creep over the 2nd half ≤ 0.1 mm per 10 s. Reference: CPU MuJoCo 3.16, offline, same MJCF (pile height, rest penetration). Prior (200 boxes): Kamino 0.001, CPU MuJoCo 0.018, XPBD 53–94 mm per 10 s; Steel lost 46–62 boxes (`pa/findings/bench.md`) | escaped, settle_time_s, rest_pen_mm, creep_mm, pile_height_mm, ke_end_j |
| R06_franka_pick_place | Franka picks a 5 cm cube, lifts 10 cm, carries it and places it. The joint-target trajectory is precomputed with Newton IK once and replayed from a device array | mujoco (Newton contacts / MuJoCo contacts), featherstone, kamino, xpbd | `newton/examples/ik/example_ik_cube_stacking.py`; `newton/examples/contacts/example_brick_stacking.py` | Coulomb hold: when 2μF_n ≥ mg, in-hand slip is 0 (target ≤ 0.5 mm over lift and carry). Place error ≤ 2 mm. Grasp success 100 % over worlds | success_rate, slip_mm, place_err_mm, joint_track_err_deg, penetration_p99_mm |

### 1.2 `precision`: precision contact

| ID | Scenario | Configs to compare | Derives from | Law / reference | Scenario metrics |
|---|---|---|---|---|---|
| P01_peg_hole_0p2 | Ø9.80 mm peg into a Ø10.00 mm hole, 20 mm deep (0.2 mm diametral clearance), chamfered 0.5 mm × 45° and unchamfered. Meshes built procedurally. Prismatic drive at 1 mm/s through a compliant holder (k_x, k_θ). Conditions: tilt {0, 0.5, 1, 2}° × offset {0, 0.05, 0.1} mm | sdf_mujoco, sdf_kamino, sdf_xpbd, sdf_steel, hydro_mujoco, primitive (cylinder in a box ring) baseline | `newton/examples/contacts/example_nut_bolt_sdf.py` (SDF mesh setup); `newton/examples/robot/example_robot_panda_hydro.py` (hydro setup) | Clearance geometry (A8): max tilt at engagement l from d·cos θ + l·sin θ = D. Centred gap preserved within 1 µm. A ramped 50 N push leaves < 0.05 mm (CON-46). Precision penetration p99 ≤ 0.02 mm (bench.md #9). Context: Factory uses 0.114 mm clearance (`IsaacLab factory_env_cfg.py:115`) and IndustReal drops episodes > 1 mm | success_rate, depth_reached_mm, peak_force_n, jam_rate, tilt_bound_err_deg, penetration_p99_mm |
| P02_nut_bolt | Nut run down a bolt by a velocity-driven revolute (π rad/s) under a 5 N axial load. Sizes: Factory m12 and m20 meshes (the example uses `m20_loose`; the other names are UNVERIFIED until downloaded), plus a procedural ISO 68-1 M10×1.5 profile to match rig law R2 | sdf × {mujoco, kamino, xpbd}; hydro × {mujoco, kamino} | `newton/examples/contacts/example_nut_bolt_sdf.py`, `.../example_nut_bolt_hydro.py` (`HydroelasticSDF`, kh 1e11) | Thread law (A9): advance per rev = p, within 0.5 %. Run-down torque T = F(d₂/2)(p + πμd₂ sec 30°)/(πd₂ − μp sec 30°), within 5 %. The stock examples check plausibility only (`example_nut_bolt_sdf.py:360-390`) | advance_err_pct, torque_err_pct, cross_thread (bool), penetration_p99_mm |
| P03_rj45_plug | RJ45 plug driven into the socket at 10 mm/s along the axis (the gizmo is replaced by a scripted drive), then pulled back with 5 N | vbd (as in the example), sdf_mujoco, sdf_kamino, sdf_xpbd | `newton/examples/contacts/example_contacts_rj45_plug.py` (`rj45_plug.usd`) | Latch as a cantilever: F = 3EIδ/L³ (Euler–Bernoulli), using the example's latch geometry and material. The plug stays locked under a pull below the latch force. Penetration p99 ≤ 0.05 mm. The example only checks drift < 1 m (`:513-519`) | peak_insert_force_n, latch_force_err_pct, locked (bool), lateral_drift_mm, penetration_p99_mm |
| P04_hertz_sphere | Sphere R = 10 mm on a plane: quasi-static load ramp 0 → 50 N, then impacts at v ∈ {0.1, 0.5, 1} m/s | hydro_mujoco, hydro_kamino, point_mujoco (solref), xpbd | `newton/examples/contacts/example_nut_bolt_hydro.py` (pressure callback, `:58`); `newton/examples/robot/example_robot_panda_hydro.py` | Hertz: F = (4/3)E*√R·δ^{3/2}; t_c = 2.87(m*²/(R*E*²v))^{1/5} (A7). Hydroelastic pairs use the closed form of their configured pressure field. Point contacts use their own penalty law. Fitted exponent 1.5 ± 0.05 for δ/R ∈ [0.005, 0.05]; prefactor and t_c within 10 % | exponent, prefactor_err_pct, tc_err_pct, penetration_p99_mm |
| P05_thin_wall_tunnel | Capsule fired into 2 mm and 5 mm walls at v ∈ {5, 10, 20, 50} m/s (one speed per world) | xpbd with adaptive collision scheduling on/off, mujoco, kamino, steel | `newton/examples/contacts/example_adaptive_collision.py` | No tunnelling at any speed. Rebound speed e·v within 5 %. Wall displacement matches momentum conservation | tunnel_rate, max_pen_mm, rebound_err_pct, penetration_p99_mm |

### 1.3 `deform`: deformables

| ID | Scenario | Configs to compare | Derives from | Law / reference | Scenario metrics |
|---|---|---|---|---|---|
| D01_cable_cantilever | 0.5 m cable clamped horizontally: static tip drop, then a pluck. Also a hanging chain pinned at the top. Segments: 20, 40, 80 | vbd_rod (iterations 10/50), joint_chain_mujoco (capsules + joint stiffness), joint_chain_featherstone, xpbd (particle springs + bending) | `newton/examples/vbd/example_cable_bend_stiffness.py`, `newton/examples/cable/example_cable_initial_rest_pose.py`, `alloy/bench/proto/solvers/vbd_rod_recheck.py` | Elastica tip drop (small deflection: δ = wL⁴/8EI; large deflection solved numerically). Hanging chain ω₁ = 1.2024√(g/L) (A11): sag 1 %, ω₁ 2 % at ≥ 40 segments. Prior: stock VBD sagged 557 mm vs 66 mm (`pa/findings/deform.md` §1.6). The recheck gave 375 mm vs 297.5 mm elastica, +26 % (`alloy/bench/proto/solvers/results/vbd_rod_recheck_cpu.jsonl`) | tip_drop_err_pct, period_err_pct, omega1_err_pct, energy_drift_pct |
| D02_cable_bend_twist | Exact discrete boundary-value bend/twist rows. Michell twisted ring across the stiffness-ratio sweep. Capstan wrap | vbd (iterations 10/50); mujoco tendon (capstan only) | `newton/examples/vbd/example_cable_bend_twist_analytic.py`, `.../example_cable_michell_threshold.py`, `.../example_cable_twist_buckling.py`, `newton/examples/tendon/example_tendon_capstan_friction.py` | Uniform bend θ_i = i/(N−1)·θ_tip. Michell critical twist = 2π√(3B/C), classified correctly on both sides. Capstan T_out/T_in = exp(μθ), within 2 % (A10) | bend_angle_err_deg, michell_class_acc, capstan_err_pct |
| D03_cloth_drape_cusick | Cusick drape: a 30 cm cotton-like disc on an 18 cm support disc, 3 s settle. Meshes at 3 refinements | vbd, style3d, xpbd, semi_implicit | `newton/examples/cloth/example_cloth_hanging.py` (four-solver switch) | DC = (A_s − A_d)/(A_D − A_d) (Cusick 1965; ISO 9073-9:2008), silhouette rasterised at 1 mm. No closed form, so the checks are: mesh convergence (ΔDC < 0.02 between the two finest meshes), the spread between solvers, max membrane strain ≤ 1 %, zero self-intersections. Rig law R6 provides real DC later | dc, node_count, dc_convergence, max_strain_pct, self_intersections |
| D04_cloth_stretch_bend | Uniaxial strip stretch to 5 %. ASTM D1388 fixed-angle cantilever (strip pushed over a 41.5° edge). Hanging strip under its own weight | style3d, vbd, xpbd, semi_implicit | `newton/examples/cloth/example_cloth_style3d.py`, `newton/examples/vbd/example_cloth_stiff_material_stretch.py`, `.../example_cloth_stiff_material_hanging.py` | Membrane: F = E_s·w·ε (E_s = E·t), Poisson contraction ν·ε. D1388: bending length c = l/2 at 41.5°, flexural rigidity G = W·c³. Hanging elongation δ = ρgL²/(2E_s/t). All within 5 % | stretch_k_err_pct, poisson_err, bend_length_err_pct, hang_err_pct |
| D05_cloth_gripper_hold | A parallel jaw pinches a 20 × 20 cm cloth (and the soft 1 × 1 grid edge case), then lifts and holds 2 s. Grip force sweeps across mg/(2μ), one value per world | vbd with full-surface contact on/off, vbd + featherstone (Franka), xpbd | `newton/examples/vbd/example_vbd_gripper_soft_grid.py`, `.../example_vbd_gripper_soft_triangle.py`, `newton/examples/cloth/example_cloth_franka.py`, `newton/examples/vbd/example_vbd_soft_rigid_contact.py` | Coulomb hold: held iff 2μF_n ≥ mg, so the onset grip force is within 5 % of mg/(2μ) and slip is 0 above it. Intersection-free (VBD guarantee) | hold_onset_err_pct, slip_mm, penetration_p99_mm, intersections |
| D06_soft_squeeze | 4 cm Neo-Hookean cube squeezed between kinematic plates to 30 % strain at 1 mm/s, eased, with lateral faces free. Also a cantilever block hanging under gravity | vbd_tet, semi_implicit_fem, xpbd_tet, implicit_mpm (soft) | `newton/examples/softbody/example_softbody_hanging.py`, `newton/examples/mpm/example_mpm_pinch_carry.py`, `newton/examples/multiphysics/example_rigid_soft_contact.py` | Uniaxial Neo-Hookean: P = μ(λ − λ⁻²) at λ = 0.8, within 5 % (A15; compressible form solved in 1D for the configured λ). Volume ratio 0.99–1.01 for ν ≈ 0.49 (`newton/tests/test_softbody_simulation.py:118-120`). Hanging sag vs linear elasticity at small strain | f10_err_pct, f20_err_pct, slope_err_pct, volume_ratio, penetration_p99_mm |

### 1.4 `granular`: granular and fluids (MPM)

| ID | Scenario | Configs to compare | Derives from | Law / reference | Scenario metrics |
|---|---|---|---|---|---|
| G01_sand_collapse_repose | Planar column collapse at aspect a = H/L₀ ∈ {0.5, 2, 6}. Lifting-cylinder heap for the angle of repose | implicit_mpm at voxel {10, 5, 2.5} mm × ppc {2, 4}; rigid spheres (mujoco) for coarse grains | `newton/examples/mpm/example_mpm_granular.py` (`mpm_sand.usda`) | Runout (L∞ − L₀)/L₀ = 1.2a for a < 2.3, 1.9a^{2/3} above (Lube et al., Phys. Rev. E 72, 041301, 2005): within 10 %. Repose angle within 2° of the configured friction angle. Prior: MPM 2.18–2.42 vs 2.4 published; rigid spheres 3.3–3.6 (`pa/findings/deform.md` §3.1, CON-75) | runout_err_pct, repose_err_deg, mass_loss_pct |
| G02_dam_break | Quasi-2D dam break (Martin & Moyce 1952 geometry, n² = 2) and a closed-box hydrostatic settle, 5 s | implicit_mpm at voxel {10, 5, 2.5} mm, critical_fraction {0, 0.5} | `newton/examples/mpm/example_mpm_water_dam_break.py`, `alloy/bench/proto/solvers/mpm_dambreak.py` | Front Z(T) vs Martin & Moyce, NRMSE ≤ 0.1. No front faster than Ritter, x_f ≤ a + 2t√(gH) (A14). Bottom pressure = ρgh within 3 %. Volume loss ≤ 1 % over 5 s (A13). Prior: the front stays inside the Ritter bound, but the settle kept h_end/h₀ = 0.658 at 1 cm voxels, a 34 % volume loss (`alloy/bench/proto/solvers/results/mpm_dambreak_cpu.jsonl`) | front_nrmse, ritter_violations, pressure_err_pct, volume_loss_pct |
| G03_viscous_spread | Axisymmetric constant-volume viscous gravity current on a plane, ν sweep (one per world). Funnel flow from the example as a robustness case | implicit_mpm viscous at voxel {5, 2.5} mm | `newton/examples/mpm/example_mpm_viscous.py` | Huppert 1982 (JFM 121:43-58): r(t) = 0.894(gV³t/(3ν))^{1/8}. Exponent 1/8 ± 0.01, prefactor within 10 %, volume loss ≤ 1 % | spread_exponent, prefactor_err_pct, volume_loss_pct |
| G04_blade_digging | 50 mm kinematic blade dragged through a sand bed at depth d ∈ {10, 20, 40} mm and 10 mm/s | implicit_mpm with a kinematic collider at voxel {5, 2.5} mm | `newton/examples/mpm/example_mpm_anymal.py`, `.../example_mpm_pinch_carry.py` (kinematic colliders) | Rankine passive pressure F = ½γd²w·K_p, with K_p = tan²(45° + φ/2), as an order-of-magnitude check. Exponent n in F ∝ dⁿ = 2 ± 0.2. Steady-force coefficient of variation ≤ 15 %. Rig law R8 provides real data later | force_n, exponent, force_cv, rankine_ratio |

### 1.5 `coupled`: coupled multi-solver

`SolverCoupled`, `SolverCoupledProxy` and `SolverCoupledADMM` are imported from
`newton.solvers.experimental.coupled` (`newton/examples/multiphysics/example_admm_contact_solver.py:23`).

| ID | Scenario | Configs to compare | Derives from | Law / reference | Scenario metrics |
|---|---|---|---|---|---|
| C01_franka_cable | Franka picks a deformable cable, carries it and places it | proxy (MuJoCo + VBD rod), admm (MuJoCo + VBD rod), admm (MuJoCo + XPBD rigid chain, the example default) | `newton/examples/multiphysics/example_franka_cable_ik_pick_place.py` (proxy), `.../example_mujoco_franka_vbd_cable_admm_solver.py` (ADMM, built to replicate across worlds) | Static hold force balance: gripper reaction = cable weight, within 2 %. Coulomb hold, so in-gripper slip ≤ 1 mm. No energy gain > 1 mJ per tick (CON-30) | success_rate, slip_mm, place_err_mm, force_balance_err_pct, energy_gain_events, penetration_p99_mm |
| C02_rigid_sand | Rigid boxes (density sweep) dropped into a granular MPM bed and left to rest. Multi-rate MPM:rigid dt ratio {1, 2, 4} | proxy mujoco+mpm, proxy kamino+mpm, one-way MPM colliders (baseline) | `newton/examples/multiphysics/example_mujoco_mpm_coupled_solver.py`, `newton/examples/mpm/example_mpm_twoway_coupling.py` | At rest, the harvested MPM wrench on each box = mg, within 2 %. Momentum ledger: Δ(p_rigid + p_sand) = gravity impulse + floor impulse, within 1 %. No energy gain | force_balance_err_pct, momentum_residual_pct, sink_depth_mm, energy_gain_events |
| C03_soft_on_cloth | Neo-Hookean tet block dropped onto a pinned cloth. Gift: four soft blocks with cloth straps | monolithic vbd; proxy (VBD soft + XPBD cloth) | `newton/examples/multiphysics/example_softbody_dropping_to_cloth.py`, `.../example_softbody_gift.py`, `.../example_xpbd_vbd_coupled_solver.py` | Static equilibrium: the sum of the pinned-particle reactions = total weight, within 2 %. Penetration-free: zero intersections. Sag consistent between monolithic and coupled runs (difference reported in mm) | support_err_pct, sag_mm, sag_diff_mm, intersections, penetration_p99_mm |
| C04_particles_into_mpm | XPBD particles, and separately a VBD tet cube, fall into an implicit-MPM granular bed | proxy xpbd+mpm, proxy vbd+mpm, monolithic MPM (the particles as MPM material) reference | `newton/examples/multiphysics/example_xpbd_mpm_coupled_solver.py`, `.../example_vbd_mpm_coupled_solver.py` | Momentum ledger closes within 1 %. At rest, the bed supports the weight within 2 %. No particle ends below the floor | momentum_residual_pct, support_err_pct, escaped, energy_gain_events |
| C05_rigid_deform_attach | Rigid boxes and a pendulum chain on cloth and soft bodies (proxy vs monolithic). A rigid ball attached to cloth (ADMM attachment). A closed four-bar split across Kamino and MuJoCo. A 1000:1 mass-ratio condition | proxy mujoco+vbd vs monolithic vbd; proxy mujoco+xpbd vs monolithic xpbd; admm attachments; kamino+mujoco admm four-bar vs single Kamino four-bar | `newton/examples/multiphysics/example_mujoco_vbd_coupled_solver.py`, `.../example_mujoco_xpbd_coupled_solver.py`, `.../example_mujoco_vbd_admm_solver.py`, `.../example_kamino_mujoco_admm_solver.py`, `newton/examples/kamino/example_kamino_basic_fourbar.py` | Attachment and loop-closure violation ≤ 1 mm (the S2 joint-separation limit). Coupled-vs-monolithic trajectory deviation reported in mm. No energy gain > 1 mJ per tick, including at a 1000:1 mass ratio | max_joint_sep_mm, traj_dev_mm, energy_gain_events, admm_residual |

### 1.6 `scale`: scale and robustness

The scenes are self-contained copies, so this family does not import other families. They are: the 64-body bin
(`pa/bench/tools/bin_scene.py`), `newton/examples/assets/nv_ant.xml`, `nv_humanoid.xml`, Franka + cube, a 32 × 32
VBD cloth, and a 10k-particle MPM block.

| ID | Scenario | Configs to compare | Derives from | Law / reference | Scenario metrics |
|---|---|---|---|---|---|
| S01_determinism | Each scene, 1000 steps, rerun 5 times. World 0 is compared at nworld 1, 64 and 4096. Neighbour worlds are perturbed | each solver × `CollisionPipeline(deterministic=True/False)` × `wp.DeterministicMode.RUN_TO_RUN` where it compiles | `newton/tests/determinism/test_solver_determinism.py:25-30`; `pa/findings/bench.md` | D1: rerun 0 ulp on CPU, and 0 ulp on GPU in deterministic mode. D2: batch invariance 0 mm in deterministic mode. D3: rerun spread W₁ over 10 runs (GPUSimBench, arXiv:2607.13059, Eq. 11). Outside deterministic mode these become measured noise floors | repeat_identical, max_ulp, batch_invariance_mm, rerun_w1_mm |
| S02_sleeping | Four ant stacks (MuJoCo sleeping). Four box towers plus a thrown block (Steel sleeping) | mujoco and steel × sleeping on/off × sleep_tolerance {default, ×0.1} | `newton/examples/mujoco/example_mujoco_sleeping.py`, `newton/examples/steel/example_steel_sleeping.py` | Sleeping must not change awake physics: deviation of awake bodies vs sleeping-off ≤ 0.1 mm. A woken tower responds within 1 step. No body sleeps with \|v\| above the tolerance | speedup, awake_dev_mm, wake_latency_steps, false_sleep_count |
| S03_throughput | Ant, humanoid, Franka + cube and 64-box bin across the full ladder (8192 and 16384 on H100). Heterogeneous worlds (different robots per world) | mujoco, featherstone, kamino, xpbd, steel × cuda_graph on/off | `newton/examples/selection/example_selection_articulations.py`, `newton/examples/kamino/example_kamino_basic_heterogeneous.py`, `pa/bench/tools/newton_bin.py` | Speed has no law. Graph on/off must agree bitwise where deterministic, otherwise within the S01 noise floor. Compare with the earlier MuJoCo Warp / Newton ladder (`pa/findings/speed.md`, `pa/bench/results/scaling.csv`, `newton_bin.csv`) | graph_speedup, launch_overhead_ms, scaling_efficiency |
| S04_memory_capacity | Ladder up to OOM per scene. Contact capacity at the default and at 0.25× the needed rows | mujoco, kamino, xpbd, steel, vbd, implicit_mpm | ASV `asv/benchmarks/benchmark_metrics.py:27-110` (SimulationMetrics); `pa/findings/bench.md` §1 (silent-capacity trap) | Memory is linear in worlds: fit mem = a + b·nworld with R² ≥ 0.99, and report b in MB/world and the max worlds on an L4 (23 GB) and an H100 (80 GB). Overflow must be detected: a silent drop is S1 (bench.md §4.7) | mb_per_world, mb_intercept, r2, max_nworld_l4, max_nworld_h100, overflow_detected |

## 2. Record schema (`alloy.bench.run/1`, one JSON line per scenario × config × nworld)

| Field | Meaning |
|---|---|
| `schema`, `scenario`, `family`, `config`, `solver`, `params` | identity; `params` = the config's solver kwargs |
| `device`, `device_name` | warp device string and its name (e.g. `cuda:0`, `NVIDIA L4`) |
| `nworld`, `dt`, `substeps`, `horizon_s`, `steps` | `dt` = outer step [s]; the solver sees `dt/substeps`; `steps` = outer steps run |
| `ms_per_step` | wall ms per outer step, all worlds, after warm-up; device-synchronised; host measure time excluded |
| `world_steps_per_s` | `nworld / step_s` (outer steps) |
| `sim_seconds_per_gpu_second` | `nworld · dt / step_s` (on CPU devices: per wall second) |
| `single_world_realtime_factor` | `dt / step_s`, set only at `nworld = 1` |
| `gpu_mem_mb`, `gpu_mem_mb_per_world` | warp mempool growth from before build to end of run (nvidia-smi fallback); `null` on CPU |
| `host_rss_mb` | process peak RSS |
| `stable` | no NaN/Inf in any state array at every sample over the horizon (scenarios may add blow-up checks and set it False) |
| `complete` | horizon reached before `--budget-s` ran out |
| `energy_drift_pct` | `100 · max_t |E(t) − E(0)| / |E(0)|` from `Scenario.energy()`; `null` if not defined |
| `penetration_p99_mm` | 99th percentile of rigid contact depth over all samples (default hook, `metrics.rigid_penetration_mm`) |
| `accuracy_error`, `accuracy_unit`, `accuracy_ref` | headline error vs the scenario law, its unit, and the law/source |
| `repeat_identical` | bitwise equality of the final state of two fresh runs (done for `nworld ≤ --repeat-max-nworld`, default 16) |
| `metrics` | scenario-specific values (see the tables above), incl. `law_pass`, `per_condition`, `conditions_covered`, `contact_overflow` |
| `status`, `error`, `notes` | `ok`, `error` (exception text, traceback tail in `notes`) or `skipped` (larger rung after an error) |
| `setup_s`, `first_steps_s`, `cuda_graph` | build time; warm-up time (includes kernel compile); whether a CUDA graph was replayed |
| `versions` | alloy git sha (`-dirty` if bench/ or alloy/ changed), newton version + commit, warp, mujoco, mujoco_warp, numpy, python, host, NVIDIA driver |
| `command`, `timestamp` | the CLI line that produced the row; local time |

## 3. Harness contract for builders

```python
from bench.harness import Built, Config, Scenario
from bench import metrics as M

class PegHole(Scenario):
    id = "P01_peg_hole_0p2"; family = "precision"; title = "..."
    horizon_s = 25.0; has_contacts = True
    accuracy_unit = "mm"; accuracy_ref = "A8 clearance geometry; CON-46"
    ladder = (1, 16, 256, 1024)
    configs = (Config("sdf_mujoco", "mujoco", dt=5e-4, substeps=1, params={...}, max_nworld=1024), ...)

    def build(self, nworld, device, config) -> Built:     # replicate the scene nworld times, finalize on device
        ...
        return Built(model, solver, s0, s1, control, contacts, pipeline, extra={...})
    def measure(self, b, t, rec): super().measure(b, t, rec); ...   # host side, between timed chunks
    def finish(self, b, rec): rec["accuracy_error"] = ...; rec["metrics"].update(...)
    def energy(self, b): return M.rigid_energy(b.model, b.state_0)

SCENARIOS = [PegHole()]
```

- **`build`.** Use `ModelBuilder.replicate(scene, copy_count=nworld)`; `world_count=` is deprecated in 1.8. Create a
  `newton.CollisionPipeline` when the solver uses Newton contacts; `Model.collide()` is deprecated.
- **Stepping.** The default `substep()` is Newton's ping-pong: clear forces, collide, `solver.step`, swap. Override
  `step()` or `substep()` for coupled solvers or for `collide_every` > 1. When the graph is on, the harness captures
  1 outer step for even substeps and 2 for odd substeps, so the Python state references stay valid.
- **Measuring.** `measure()` runs every `sample_every` outer steps (default about 20 samples per horizon) with a
  device sync before it. Keep it O(state) numpy work. Per-step quantities, such as peak force or creep onset,
  belong in device arrays that a Warp kernel inside `step()` writes, and `finish()` reads.
- **Failures.** Raise freely. The runner turns exceptions into `status: error` records and skips larger rungs of
  that config. A NaN stops the run with `stable: false`. Never catch-and-hide inside a scenario.
- **Solver factories.** Copy `make_rigid_solver` from `_example.py` as a starting point, and keep solver kwargs in
  `Config.params` so they appear in the record. Use one config per meaningful setting. Never change a solver
  setting silently between rungs.
- **GPU path status.** On 2026-10-31 the harness was verified on Mac CPU only (X01, 4 solvers, nworld 1 and 16).
  The CUDA graph and mempool paths use the Warp 1.18 APIs (`ScopedCapture`, `capture_launch`,
  `get_mempool_used_mem_current`) but have not run on a GPU yet. The first builder on a GPU VM runs
  `python -m bench.run --family _example --device cuda:0 --ladder 1,16,256` and reports any failure to the suite
  owner.
- **CPU first.** Every scenario must run at `--device cpu --ladder 1` on the Mac (maybe with `--horizon` shortened)
  before going to a GPU VM. GPU runs use the builder's own spot VM. Follow BOARD.md rule 3 and delete the VM after.

## 4. Baselines to reproduce first

These earlier measurements are each scenario's first sanity check. A large disagreement points to a harness or
setup bug before it points to physics.

| Scenario | Earlier number | Source |
|---|---|---|
| R03 | Kamino onset within 0.1°, creep 0.17 mm per 10 s; XPBD 5.8 mm per 10 s at −1° | `alloy/docs/research/bench.md` §3, `alloy/docs/research/bench_probes/ramp_onset_results.jsonl` |
| R05 | 200-box creep: Kamino 0.001, CPU MuJoCo 0.018, XPBD 53–94 mm per 10 s; Steel escapes | `pa/findings/bench.md`, `pa/bench/results/newton_bin.csv` |
| D01 | VBD 557 mm vs 66 mm (study); 375 mm vs 297.5 mm elastica at 50 iterations (alloy proto) | `pa/findings/deform.md` §1.6, `alloy/bench/proto/solvers/results/vbd_rod_recheck_cpu.jsonl` |
| G01 | MPM runout 2.18–2.42 vs 2.4 | `pa/findings/deform.md` §3.1 |
| G02 | front inside the Ritter bound; volume ratio 0.658 at 1 cm voxels | `alloy/bench/proto/solvers/results/mpm_dambreak_cpu.jsonl` |
| S03 | MuJoCo Warp / Newton throughput vs worlds on L4 and H100 | `pa/bench/results/scaling.csv`, `nv_sweep.csv`, `pa/findings/speed.md` |
| X01 | free fall error 4.089 mm = ½g·h·t at h = 1/600 s, t = 0.5 s (exactly half the A1 bound) on all four rigid solvers, Mac CPU | `bench/results/_example/X01_free_fall.jsonl` |

## 5. Offline references (never in the loop)

| Family | Reference | Use |
|---|---|---|
| rigid | CPU MuJoCo 3.16 on the same MJCF (`pa/bench/tools/cpu_replay.py`, `fidelity.py`) | pile height, rest penetration, creep reference for R05; R03 onset |
| precision | Drake hydroelastic (reading), IsaacLab Factory configs (reading) | pressure-field closed forms for P04; clearance context for P01 |
| deform | libuipc / IPC (offline runs if installed) | intersection-free drape and pinch reference for D03 and D05 |
| granular | Genesis SPH dam break (`Genesis@983bd0dd`, `tests/particles/test_sph.py:283`), on a builder VM only if it installs cleanly | second front curve for G02; hydrostatic column |
| scale | earlier MuJoCo Warp, Newton and PhysX ladders (`pa/bench/results`) | throughput comparison for S03 |
