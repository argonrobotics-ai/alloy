# Keep in mind

What does not work yet, one line each. A linked line goes to a page with the measured size of the limit and what to
do instead today; a plain line carries its workaround. Read the lines for your area before you trust a result there.

## Compile, memory and speed

- An interrupted first compile can damage the kernel cache: later runs then fail with a `JSONDecodeError` or an
  nvJitLink error. Delete the folder `~/.cache/warp` and build again.
- `env.close()` does not free GPU memory, so a larger environment built later in the same process can step many
  times slower with no error. Use one environment size per process, or start a new process.
- A robot, a cable and cloth together in one scene can force a very small coupled step, and the scene then steps
  very slowly. Run `alloy.estimate(scene, worlds, seconds)` before you build.

## Rigid bodies and contact

- [A fast small body passes through a thin static part: a 2 cm steel ball lands on a 1 cm plate from 8.9 m/s and
  passes through from 9.9 m/s.](../known_gaps/rigid-tunnelling.md)
- [A box sliding at metres per second hops and rocks: a cube at about 2.5 m/s rises a few millimetres and pitches 1-2
  degrees (up to 7 for a small cube).](../known_gaps/rigid-sliding-box-hops.md)
- [A body on `collision="sdf"` or `"mesh"` never fully rests, and fails under about 40 g or from 40
  kg.](../known_gaps/rigid-sdf-mesh-contact-limits.md)
- [Light slender parts such as pens and pencils spin and roll slower about their thin axis, because their inertia is
  raised at build.](../known_gaps/rigid-slender-spin-floor.md)
- [With more than 8 free bodies on a GPU, a body spinning in free flight turns only to first order; the CPU's default
  route keeps rk4 at 3-6x the cost per step.](../known_gaps/rigid-clutter-rotation.md)
- [Stacks of free bodies step 2.4x slower by default than stock Newton MuJoCo, and the opt-in fast mode that closes
  the gap softens contacts.](../known_gaps/rigid-free-body-solve-cost.md)
- [The opt-in Newton 1.8 rigid solver lets a robot arm's contacts through 54 times more often than the copy before
  it.](../known_gaps/rigid-steel-pile-backend.md)
- [A lightly pressed box sliding at a 1.25 ms step can tip in scenes that set their own MuJoCo contact gains; the
  default path slides flat.](../known_gaps/contact-light-sliding-tip.md)
- [Mesh contacts can take the wrong inside or outside side at rare degenerate points, such as one point in a thin
  cup's cavity.](../known_gaps/contact-parity-full-surface.md)
- [A hydroelastic precision fit pushed sideways leans about 1.3 times as far as its clearance allows, with about 0.1
  mm of wall penetration.](../known_gaps/precision-hydroelastic-side-lean.md)
- [A precision fit pushed sideways leans 2 to 6 % past its clearance, by an amount that depends on the push
  direction.](../known_gaps/precision-sdf-side-lean-direction.md)
- [A peg that misses its bore and rolls across the block slowly gains speed instead of losing it, and health stays
  OK.](../known_gaps/precision-rolling-peg-gains-speed.md)
- [Rigid bodies in the cloth and soft-body solver turn by a first-order step, so an applied torque arrives short and
  can wrap, and a spin slows down.](../known_gaps/vbd-rigid-first-order-rotation.md)

## Robots and grasping

- [No call attaches or welds an object to a gripper: friction is the only
  grasp.](../known_gaps/grasp-no-attach-or-weld.md)
- [Robot scenes step physics at 1.25 ms (8 steps per 10 ms period) unless you ask for a coarser step, even when a
  larger one would be stable.](../known_gaps/robot-fixed-step.md)
- [A robot file that is a long chain of light bodies (from 6 nested bodies) can go non-finite at its first contact on
  MuJoCo Warp.](../known_gaps/robot-long-chain-first-contact.md)
- [Shipped robots left with no targets drift from their start angles: Franka arms start outside a joint range and
  servos sag under gravity.](../known_gaps/robots-at-rest-joint-drift.md)
- [The shipped Franka arms never collide link 5 with link 7 (the Panda: link 5 with the
  hand).](../known_gaps/fr3-link5-link7-exclude.md)
- [A weld or connect in a robot file is a stiff spring, not a rigid joint, and its stiffness falls with the lighter
  body.](../known_gaps/weld-stretch-under-load.md)
- [The shipped 200-capsule bin pile starts with 54 pairs of capsules overlapping by up to 7.97 mm, so health reads
  DEGRADED at first.](../known_gaps/task-bin200-starts-overlapped.md)
- [A body that touches cloth or soft parts is coupled through a mass that is the same in every direction, heavier than
  the body along its moving axis (4.5 kg for a 1.5 kg slider).](../known_gaps/coupling-proxy-isotropic-mass.md)

## Cloth and film

- [A free strip of cloth bends about 35 % too soft along a mesh axis.](../known_gaps/cloth-bending-free-strip-soft.md)
- [A stiff sheet (PET, nylon, polypropylene, cardboard, bending length over 5 cm) sags several times too deep between
  supports at the default iterations.](../known_gaps/cloth-stiff-sheet-sag.md)
- [A woven fabric card built from published bending data drapes differently from the same fabric on a drape tester,
  even when converged.](../known_gaps/cloth-woven-drape-isotropic.md)
- [A cloth card whose stretch stiffness is large for its mass, mesh and step stays nearly flat and creeps at about 2
  mm/s, while health reads OK.](../known_gaps/cloth-style3d-stretch-lock.md)
- [Sheets dropped onto each other can end with their layers crossed, and health does not flag
  it.](../known_gaps/cloth-falling-sheets-pass-through.md)
- [A rigid ball with a radius under about 0.7 of the sheet's cell size can go through a cloth or film sheet without
  touching it.](../known_gaps/cloth-small-body-passes-through.md)
- [A folded cloth keeps creeping under the default cloth accuracy instead of coming to rest within a fraction of a
  second.](../known_gaps/cloth-fold-unsettled-default.md)
- [Folded T-shirt sleeves spring part-way back at the default mesh and contact stiffness and do not stay
  folded.](../known_gaps/cloth-sleeve-fold-springback.md)
- [Cloth held between pads by friction alone creeps out below the Coulomb limit, at about 2.9 mm/s at half the limit
  with default settings.](../known_gaps/cloth-friction-creep.md)
- [A robot's friction pinch on cloth creeps and lets go before the Coulomb limit, so lifting, carrying or folding
  cloth can lose it.](../known_gaps/cloth-robot-grasp-carry.md)
- [Cloth gripped by friction on a moving rigid body can slip on it right after the body reverses, moving 0.35-0.65 of
  the body's travel.](../known_gaps/coupling-cloth-grip-slip-at-reversal.md)
- [Cloth resting under its own weight away from the world origin grips harder than its friction says: on a table at
  0.75 m it holds or creeps slowly.](../known_gaps/cloth-contact-depth-float32.md)
- [A cloth particle that starts a step inside a rigid contact is thrown out at its depth divided by the step, 18.7 m/s
  for 0.5 mm at robot-finger steps.](../known_gaps/cloth-contact-exit-kick.md)
- [Cloth and soft solids with self contact move at most about 3 m/s, so a sheet dropped from 1 m falls at 3 m/s
  instead of 4.4 m/s.](../known_gaps/vbd-self-contact-speed-cap.md)
- [Self contact keeps fixed per-element buffers and can drop pairs once a scene needs more than they
  hold.](../known_gaps/vbd-self-contact-buffer-overflow.md)
- [Paper and film keep only a flat fold made on one mesh line: a sheet bent 90 degrees over a plate edge does not keep
  the fold.](../known_gaps/film-fold-memory.md)
- [A very thin film (0.05 mm) draped over a sharp box edge goes INVALID, and the fix is opt-in in this
  release.](../known_gaps/film-thin-sheet-box-edge.md)
- [Metal sheets sag far more than real ones at real density, which can yield them, and they bend unlike a plate across
  the mesh diagonals.](../known_gaps/film-metal-sheets.md)
- [A film's strain projection carries no force to bodies: a 90 degree peel of a PET strip reads 42.9 N against 8.7
  N.](../known_gaps/film-projection-force.md)
- [The open-carton tape peel throws a light body past 20 m/s in some worlds, 1 in 40 with the task's caps and 3 of 24
  without them.](../known_gaps/paper-open-carton-peel.md)
- [The wrapped present cannot be unwrapped yet: the task is not run and 0 % of the paper comes
  off.](../known_gaps/paper-unwrap-gift.md)
- [A two-pad pinch does not lift plies it presses hard: at 100 N the pads rise and the plies stay on the table, with
  no health flag.](../known_gaps/sewing-pinch-heavy-press.md)

## Liquids and heat

- [Liquid needs finer particles than the default in openings under 32 mm (16 mm at super), and containers under 24 mm
  are unreliable at both.](../known_gaps/liquid-narrow-containers.md)
- [A pour alone can splash a few percent over its container's rim (3 % of a 6 ml pour), and health stays
  OK.](../known_gaps/liquid-pour-over-the-rim.md)
- [A stream poured on a fabric sheet with self contact on makes the world INVALID, and cloth takes up no
  water.](../known_gaps/liquid-on-cloth.md)
- [Cloth, liquid and free rigid bodies lighter than twice the cloth and soft mass do not build in one
  scene.](../known_gaps/liquid-cloth-light-rigid.md)
- [A floating container with a liquid-to-container mass ratio above 50 (such as a light cup) is refused at
  build.](../known_gaps/liquid-floating-container-ratio.md)
- [Liquid beside a paste or powder sees it as a solid wall: there is no wetting or
  percolation.](../known_gaps/sph-mpm-mixing.md)
- [Heated liquid does not convect: `buoyancy=True` (water only) gives the first push, then the flow stalls after about
  a millimetre.](../known_gaps/heat-no-buoyancy.md)
- [Boiled-off liquid particles stay in the solver and can block a heater: water at 99 degrees C on a 177 degrees C
  plate stops heating once 25 % of its mass has boiled off.](../known_gaps/heat-boiled-particles-stay.md)
- [A cylinder's force pressure lags when its chamber is joined to another small chamber by a wide edge, and the
  ringing grows.](../known_gaps/fluid-chamber-to-chamber-edges.md)
- [A syringe pushed by a constant force rebounds after its air leaves, because the plunger still carries the needle's
  liquid mass.](../known_gaps/syringe-force-push-air-phase.md)
- [A surface film feels only what its settings name; every other body and particle passes through
  it.](../known_gaps/surface-film-one-way.md)

## Cables and rods

- [A cable on the chain model creeps back in a held twisting grip on the CPU, at about 0.14 % per
  second.](../known_gaps/rod-chain-grip-slips.md)
- [A chain rod resting on an elastic rod sinks into it by up to the wire's radius, which can make the world
  INVALID.](../known_gaps/rod-chain-on-elastic-rod-sinks.md)
- [A thin elastic rod can fall through a soft body between its vertices when it is thinner than the
  gaps.](../known_gaps/rod-on-soft-body-falls-through.md)
- [A wire loaded close to its fully plastic moment settles slowly and less
  accurately.](../known_gaps/rod-plastic-near-full-yield.md)
- [A rod folded tighter than one segment's length (a hairpin or coil) passes through itself and the world fails its
  health check.](../known_gaps/rod-self-contact-segment-length.md)
- [A thin chain rod's twist vibration rings too slowly or not at all, though a steady end torque twists it
  correctly.](../known_gaps/rod-twist-dynamics.md)
- [An elastic rod beside a robot runs far slower than the chain model on the CPU: 94.9 s against 1.00 s per simulated
  second.](../known_gaps/rod-vbd-beside-robot-cost.md)

## Soft bodies, cutting and plasticity

- [A body dropped on a soft solid bounces back almost as fast as it arrived, far more than the materials' restitution
  says.](../known_gaps/soft-solid-bounce.md)
- [A flat tool pressed into a soft block compresses it without limit (up to 208 % strain) because only blade edges
  cut.](../known_gaps/soft-flat-tool-compression.md)
- [Cutting with copied fragments on a coarse mesh has five limits, such as a piece needing two layers of mesh
  nodes.](../known_gaps/cutting-copies-coarse-mesh.md)
- [The knife-force reference holds only until the first cut; after it the pre-split seam reads 0.65-0.95 N of contact
  artefact.](../known_gaps/cutting-seam-reference-plateau.md)
- [Plasticity covers soft volumes, joint springs, metal sheets, wires and coil springs; bulk metal parts stay
  rigid.](../known_gaps/plasticity-metal.md)
- [A soft body that yields can yield spuriously at rest when its yield strain is under about 10 %, then slump and
  creep.](../known_gaps/plasticity-soft-tets-low-yield.md)
- [A punch's force on a yielding MPM solid converges only for soft solids; metals need mass scaling, else the force
  reads a few % of the truth.](../known_gaps/plasticity-mpm-solid-punch-force.md)
- [A drill bit that lands on an exposed bar never cuts it, however hard it is
  pushed.](../known_gaps/drilling-bars-exposed.md)
- [Drilled holes are seen only by the parts listed as pins; every other body meets the undrilled
  shape.](../known_gaps/drilling-holes-pins-only.md)
- [A weld bead's start, its tack starts and a crater mid-joint are not right
  yet.](../known_gaps/weld-bead-starts-and-craters.md)
- [Trapped air under a label is scored at 20 mm2, not at 3 mm2 inside a
  barcode.](../known_gaps/label-barcode-trapped-air.md)

## Sand and pastes

- [Pastes and viscous flows along the ground run slow, or leak through thin floors on the dense grid, because floors
  sit between grid planes.](../known_gaps/mpm-sparse-floor-offset.md)
- [Pastes and viscous liquids flow slower than their card says along the ground: the bottom one or two particle layers
  move at about half speed.](../known_gaps/mpm-viscous-film.md)
- [The flank of a collapsed heap reads lower than the 26-27 degree repose angle, and a squat collapse's runout still
  grows with resolution.](../known_gaps/mpm-grain-slope-resolution.md)
- [Above 64 GPU worlds, sand and paste with a body in them are not validated (grains can sink into
  bodies).](../known_gaps/mpm-moving-container-many-worlds.md)
- [Sand and paste on a GPU use about 10x the memory with CUDA graphs; a scene too big for graphs runs without them,
  and one too big for both is refused.](../known_gaps/mpm-gpu-memory.md)
- [Sand, dough and paste at many worlds are limited by the material solve: 256 dough worlds take about 7.6 s per 10 ms
  period on an L4.](../known_gaps/mpm-solve-speed.md)
- [On the dense grid, a world that outgrows the grid's box couples the worlds by one rounding step (1.5e-8 to
  3.1e-8 m).](../known_gaps/mpm-dense-grid-world-coupling.md)
- [Shear-thinning pastes thin a little too little: fast flow runs up to about 15 %
  slow.](../known_gaps/paste-shear-thinning.md)
- [A material cannot both melt and shear-thin: the melt law is lost without a warning (no library material has
  both).](../known_gaps/paste-melt-and-shear-thinning.md)
- [A tool thinner than a paste's grid cell (whisk wires, skewers) drags it in
  pulses.](../known_gaps/paste-thin-tools.md)

## Health, determinism and coupling

- [On a GPU, rigid bodies and robots in contact do not repeat bit for bit once a body takes more than one contact
  pair.](../known_gaps/determinism-gpu-rigid.md)
- [Health's overlap flags (PEN_WARN, PEN_FAIL) for rigid bodies and robots do not yet catch every deep
  contact.](../known_gaps/health-mujoco-penetration.md)
- [Worlds of a coupled scene share one exchange clock, so one world's events change the steps of every
  world.](../known_gaps/coupling-shared-exchange-clock.md)
- [With the opt-in contact gate, cloth resting on fixed geometry while the robot is away settles at a coarser
  step.](../known_gaps/coupling-contact-gate-settling.md)
- [Contact between a soft or cloth particle and a MuJoCo body takes its gains from the exchange
  clock.](../known_gaps/coupled-record-damping.md)

## Cameras

- [Env camera images are bright and grainy: 4.2 to 4.8 grey levels of noise at 4 samples per pixel (16.8 on a textured
  cabinet), and a mean brightness of up to 212 of 255.](../known_gaps/camera-env-images.md)
- [Glass objects render opaque unless a look file makes them glass.](../known_gaps/camera-glass-opaque.md)
- [A robot file with one visual-only geom draws only its visual geoms, so the rest of the robot is invisible in
  images.](../known_gaps/camera-robot-collision-geoms.md)
- Each new look of the full camera (`quality="full"`: other lights, colours or a look file) compiles again, about
  100 s the first time on one L4. Keep one look for a whole run; a look used before starts in under a second.
- Lighting set with `camera_scene(sim, lighting=...)` carries into later default `sim.render()` calls on that
  simulation. Render the default pictures first, or build a second simulation for them.
