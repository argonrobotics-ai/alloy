# Known gap: a MuJoCo body's coupling proxy carries an isotropic mass, heavier than the body along its moving axis

## What
- A MuJoCo body coupled into another solver (VBD cloth, SPH, MPM) is represented there by a proxy with one scalar
  mass and inertia: `1 / body_invweight0`.
- MuJoCo's `body_invweight0` is trace(W)/3, where W is the body's 3x3 translational (or rotational) block of
  J M^-1 J^T at qpos0.
- So in the other solver a body that moves along one axis responds as if it were 3x heavier along that axis (a slider,
  the tip of a hinged body; 1.5x with two free axes): a contact there pushes it back less than MuJoCo then moves it.
- What is sized from the proxy mass no longer reads it (2026-11-07): the exchange clock's interface bound
  sqrt(m / k_c), the strong-pair mass and the exchange stiffness cap c·m / dt_x^2 take the body's mass along its most
  mobile axis, 1 / lambda_max(W) (`alloy.coupling.axis_mass`, law `tests/coupling/test_axis_mass.py`). The gap left is
  the proxy's own dynamics in the destination.

## Number
- A 1.5 kg body on one prismatic joint gets a 4.5 kg proxy. A 1.2 g hinge tip 5 cm out gets 3.61 g, against 1.20 g
  along the direction it moves.
- In C7's two-pad gripper each pad is 0.56 kg on a jaw slider under a 1 kg vertical carriage: a 1.329 kg proxy, 0.56
  kg along the jaw, 2.12 kg along the carriage. Its clock bound was 1.15 ms on the 1.329 kg; it is 0.75 ms on the
  0.56 kg now. The CPU coupling suite (136 tests) passes unchanged.

- What it changes in a coupled contact (WP10, CPU Warp route, 2026-11-09, with the momentum-consistent feedback of
  `alloy.coupling.record_momentum`; measured with a development rig, not shipped): a 0.515 kg platen over C3's free
  16 g silicone pad, mid-air, as one MJCF body whose joint alone changes: one vertical slider, a hinge 5 cm from its
  centre of mass, and three slides (an exact proxy with the same contact, the slider's twin).
  - The proxy's answer to a force along the axis the body moves on is 0.333 of the body's own: 0.647 against
    1.941 per kg (slider), 0.571 against 1.712 per kg (hinge). Across that axis the proxy answers at the same 0.333
    where the body does not move at all.
  - In the first exchange of a contact the slider's proxy gives way 16.1 mm/s where its exact twin gives 65.0 mm/s
    (0.25x: steeper than the mass ratio).
  - Over a whole impact at 0.1 m/s the pad's momentum is 4.0 % (from touch) and 0.6 % (from a 1.7 mm flight) below
    the twin's. The same exact scene moved 0.3 m up changes by 1.4 % and 5.3 %, so this end-to-end number cannot
    tell the proxy's mass from float32 position noise. The pair's momentum closes in every run.
  - Withdrawn: this page said "+3.5 %" for the slider against a free box (2026-11-08). That compared an `add_box`
    body with an MJCF geom, which make different contact; it did not measure the proxy's mass.
  - Bodies light enough for the error to grow (a 1.2 g hinge tip or a 20 g box on that pad) are a strong pair and run
    the ADMM coupler, which does not use this proxy.

## Why
- trace(W)/3 is the least-squares isotropic fit of W (argmin_a ||aI - W||_F). One scalar cannot match a body whose
  mobility differs by direction.
- A proxy at the most mobile axis's mass was tried and withdrawn (WP11). It scales `body_invweight0` by
  3·lambda_max/trace, which makes the proxy too light in every other direction.
- In a cloth clamp test at the same 0.521 ms exchange step, that variant made the clamp bounce: gap spread 0.72 mm
  and an 18.2 N force peak, against a 1.2 mu N limit of 15.1 N. The isotropic fit gives 0.0013 mm and 12.5 N.
- The cause is the pads alone. Their proxies dropped from 1.33 to 0.56 kg, while a pad really carries 2.12 kg along
  the carriage that drags it.

## Opt-in fix for VBD (phase 1): `ALLOY_PROXY_TENSOR=1`
- With `ALLOY_PROXY_TENSOR=1` set before `alloy.build` (or `Scheduler(proxy_tensor=True)` at the engine layer), a
  MuJoCo body's proxy in VBD (cloth, soft bodies, rods, film) moves with a 3x3 inverse mass: the body's own W, kept
  in the body's frame and turned with it. Off by default; with it off the build steps the same kernels on the same
  inputs (three coupled scenes compared bit for bit against the tree without the change).
- The scalar 3 / trace(W) stays for every gate, clock, stiffness cap and solver seed.
- Set it through the environment variable when the build uses `rerun_worlds=` or `processes=`: their side
  simulations read the variable, not the engine-layer argument.
- Law `tests/coupling/test_proxy_tensor_law.py` (CPU, MuJoCo Warp route, 0.156 ms exchanges), each case on the
  scalar and on the tensor. A known force along the moving axis of a 0.515 kg platen on one slider, and of the same
  platen on a hinge 5 cm from its centre of mass: the scalar proxy answers 0.333 of its body's answer (0.647 against
  1.941 /kg, 0.571 against 1.712 /kg), the tensor within the law's 10 %. Across the axis the scalar answers 0.333
  where the body cannot move, the tensor within 10 % of zero. In the first contact exchange VBD pushes the slider's
  scalar proxy back 0.248 as far as its exact twin's (16.1 against 65.0 mm/s), the tensor's as far within 10 %.
  Free bodies stay exact. 29 pass; the 5 scalar cases are the strict xfails that keep this page.
- C7's clamp at the 0.52 ms exchange step, the case the most-mobile-axis scalar failed (limits: gap spread under
  100 um, peak force under 1.2 mu N = 15.12 N). Scalar, then tensor: 3 mm drag, gap spread 1.1 -> 1.3 um, peak 12.48
  -> 12.50 N; 3 mm the other way 2.2 -> 2.0 um, 12.56 -> 12.51 N; 0.5 mm drag 0.39 -> 0.32 um, 6.04 -> 6.54 N;
  0.5 mm the other way 0.47 -> 0.22 um, 12.53 -> 12.54 N. No pass-through. The same holds with the floor below at
  10 and at 1,000 (largest spread 2.6 um, largest peak 12.58 N).
  One row depends on the floor: the 0.5 mm drag's peak is 6.04 N scalar and 7.60 / 6.54 / 6.71 N at 10 / 100 /
  1,000. Which is closest to the bodies' own clamp force at that closure is not measured yet.
- A lone pad dragging a cotton sheet carries it 97 % of the pad's travel (scalar 99.8 %; the law asks 90 %). A film
  pinch carries 49.4 of 50 mm either way.
- A scene whose coupled bodies are all free (a box on cloth) keeps the scalar path with the switch on, bit for bit:
  a free body's W is already I / m.

### What phase 1 leaves open
- **Two pads dragging one flat sheet** is not a usable check of the tensor. The sheet's travel there is set by
  VBD's unconverged friction, not by the proxy: of the pads' 8.6 mm it follows +1.7, +0.8 and -1.7 mm at 10, 20 and
  40 iterations with the scalar, -0.0, -0.8 and +0.8 mm with the tensor (floor 100), and +0.6, -0.0 and +3.0 mm at
  floor 1,000. `tests/coupling/test_pad_drag_law.py` asserts no travel there for that reason.
- **The sewing ply pinch** (`tests/tasks/test_sew_ply_pinch.py`, two 0.2 kg pads on two 0.61 mm muslin plies with
  10 mm cells) runs the same with the switch on as off. Pressed at 30 N it passes its law (the plies rise at least
  40 % of the pads' 10 mm lift). Pressed at 100 N the plies stay on the table either way (-0.94 mm on, -0.93 mm
  off: `docs/known_gaps/sewing-pinch-heavy-press.md`). The rig's first version passed at 100 N with the switch off
  (96 %) only because its pads slid 0.3 mm off the edge of a vertex row and sank between the vertices to the table.
  The tensor holds the pads where MuJoCo holds them, so with it on that version failed. Its re-cut seats the pads on
  a vertex row.
- **Locked directions keep a floor.** W of a body on one slider has rank 1, and VBD's body solve needs its inverse.
  Each eigenvalue is kept at 1/100 of the largest or more (`ALLOY_PROXY_TENSOR_KAPPA`, default 100): across its
  axis a slider's proxy still gives way 1 % as much as along it.
- **W is the one at qpos0**, turned with the body. That is right when the mobile axis is the body's own (a hinge
  tip) or the body does not turn. It is wrong in direction when the axis is fixed in an ancestor and the body turns
  against it: a 0.2 kg pole on a 1 kg sliding cart answers a push along the cart's slide at 4.99 /kg upright, proxy
  and body alike; turned 90 degrees the body answers 0.83 /kg and the proxy 0.050 /kg (17x too heavy; the scalar is
  2x too light there). The floor bounds the error: at `ALLOY_PROXY_TENSOR_KAPPA=10` the proxy answers 0.50 /kg (1.7x
  too heavy). The same holds for a wristed finger on a gantry and for any arm far from its start pose. Pinned as an
  expected failure in `tests/coupling/test_proxy_tensor_host.py`.
- **A force off the centre of mass of a hinged body.** The translation-rotation block of the 6x6 W is dropped, and
  the proxy's inertia stays Newton's isotropic fit: such a contact is closer than before, not exact.
- **Liquid (SPH) and sand or paste (MPM)** still move the proxy with the scalar mass.
- **GPU:** not run yet (the build warns). A model with tendons keeps the scalar path. Set the flag before the first step; a captured CUDA graph keeps the kernels it was
  captured with.

## Workaround
- None needed for stability: the clock, the strong-pair class and the cap already use the axis mass.
- For a proxy whose response along its moving axis matters (a light slider pushed by cloth or liquid along its
  axis), expect the other solver to move it about trace(W)/(3·lambda_max(W)) as much as MuJoCo would (1/3 on one
  axis); check the source body's motion, not the proxy's.

## Planned fix
- Phase 1 (VBD) is in as the opt-in above. It becomes the default when its law passes on the GPU too, the two-pad
  drag is no worse than the scalar's, and the coordinator rules.
- Phase 2: the SPH body integrate and harvest. Phase 3: MPM's collider IJtm and rewind. Each reads the same tensor
  (`alloy.coupling.proxy_tensor`) and gets its own law first.
- Later: W per exchange instead of at qpos0, and the translation-rotation block.
