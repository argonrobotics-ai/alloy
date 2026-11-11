# Robot scenes take 1.25 ms physics steps unless you ask for a coarser one

## What is wrong
A scene with a robot steps its rigid bodies at the fixed step: 1/600 s snapped down to the grid `dt / 2**j`, which is
1.25 ms at `dt=0.01` and at `dt=0.005`. So one control period costs 8 physics steps at `dt=0.01` and 4 at
`dt=0.005`, also for a robot whose servos and contacts would be stable at a larger step. Scenes of free rigid bodies
only already pick the largest stable step by themselves (`docs/guide/worlds.md`); robot scenes do not.

The larger step is opt-in. Set it before building:

```python
scene.rigid_step = 0.0025      # pin a step in seconds (snapped down to the grid dt / 2**j)
scene.rigid_step = "auto"      # or: the largest step every cap below allows, never under the fixed step
```

`sim.routes()` and `alloy.estimate(scene, worlds, seconds)["rigid_step"]` name the step and what bound it. A scene
that never sets `rigid_step` builds and steps exactly as before.

The caps a robot adds (the free bodies beside it keep their own):
- servo: for every position servo, `omega dt <= max(1, zeta)`, with `omega = sqrt(kp / I)`, `zeta = (kv + joint
  damping) / (2 sqrt(kp I))` and `I` the joint's inertia at the start pose, armature included. The cap reads each joint alone at the start pose: a folded arm has a smaller inertia, and the servos together have a stiffer mode than any one joint (1.34x on the Panda, 1.16x on the Go2, 1.00x on FR3, UR10e, G1 and the LEAP hand, the worst of 200 poses). The cap sits a factor 2 or more below the stability limit, which covers both. Past it a step
  response is off by more than about a tenth of the step; the servo is unstable at `2 zeta + 2 sqrt(zeta^2 + 1)`.
- damping: joint damping and a servo's `kv` weaken a contact's push on the joint by `1 / (1 + dt b / I)`,
  `b = kv + joint damping`. `1 + dt b / I` may be at most 1.25 times its value at 1.25 ms, so a link's contact sinks
  at most 25 % deeper than at the fixed step. This is the cap that holds stiff, heavily damped arms.
- solref: `dt <= tau / 2` for the time constant of every contact shape, joint limit and equality row the robot file
  writes. MuJoCo raises a smaller `tau` to `2 dt`, so those rows get softer by `(tau / 2 dt)^2`.
- link_gap: `v dt <= ` the robot's contact gap (5 mm, or `add_robot(..., contact_gap=)`): a link travels at most
  its contact gap per step, so a contact starts at most that deep; `v` is the speed of the farthest collision point
  below one joint at 2 rad/s (0.5 m/s for a slide, or the file's joint velocity limit when that is lower). Several
  joints moving together give a faster tip.
- loose_body: a loose free body in the robot file (a box or a mug beside an arm) must not get a softer contact than
  at 1.25 ms. Alloy stiffens a light loose body's contact as far as the step allows, so a light one holds 1.25 ms.

"auto" takes the largest grid step that passes all of them. A pinned step is taken as asked; the build then warns
once and names each cap it breaks, the joint or shape, and the cost (for example "contacts 0.25x as stiff as at
1.25 ms").

## Number
- Physics steps per control period at the fixed step: 8 at `dt=0.01`, 4 at `dt=0.005`; with `rigid_step = 0.005` at
  `dt=0.005`: 1.
- The shipped Go2 (`emb_e8_go2`): "auto" gives 2.5 ms, half the physics steps. The damping cap binds: at 5 ms a
  calf's contact would sink 1.48x deeper than at 1.25 ms. link_gap allows 5 ms (its feet reach 0.32 m from the hip
  axis: 7.8 ms), its 20 ms default time constants 10 ms and its servos 18 ms.
- The shipped Panda (`emb_e2_panda`): "auto" stays at 1.25 ms. Three caps hold it: damping (its servos' `kv` of 450
  on a joint inertia of 0.22 kg m2: contacts on its links would sink 1.72x deeper at 2.5 ms, 3.15x at 5 ms), solref
  (its file's 2.5 ms contact time constant on the arm, its table and props: 0.25x as stiff at 2.5 ms, 0.06x at 5 ms;
  the FR3 and UR10e files write the same) and loose_body (the box, cylinder and mug in its file: 0.25x as stiff at
  2.5 ms, 0.06x at 5 ms). link_gap gives 3.2 ms (it reaches 0.79 m from joint 2), the finger coupling's 5 ms time
  constant 2.5 ms, its servos 5 ms.

Rows (`dt=0.005`; the step pinned with `scene.rigid_step`; speed against the same robot at 1.25 ms):

| scene | 2.5 ms (2 steps per period) | 5 ms (1 step per period) |
|---|---|---|
| speed, 1,024 worlds on one L4: Panda, FR3, UR10e, LEAP | 1.91-2.05x | 3.3-3.7x |
| speed, 1,024 worlds on one L4: Go2 | 1.66x | 2.45x |
| speed, 4 worlds on the CPU, six robots | 1.65-1.97x | 2.8-4.0x |
| arms moving freely: joints against the 1.25 ms run | within 0.009 rad | within 0.027 rad |
| loose box, cylinder, mug beside an arm: end position | within 0.05 mm | within 0.17 mm |
| LEAP hand: cube stays in the hand (1,024 worlds) | all, 0.04 mm off | all, 0.1 mm off |
| Go2 standing (1,024 worlds) | all stand, base 0.02 mm off | all stand |
| G1: joints against the 1.25 ms run, standing / moving | 0.04 / 0.26 rad | 0.08 / 0.44 rad |
| gripper holding a thin plate: slip (0.57 mm at 1.25 ms) | 1.28 mm | 4.22 mm |
| gripped chain: pad penetration (0.10 mm at 1.25 ms) | 0.34 mm | 0.72 mm |
| pick and place, 2,560 episodes: episodes with a penetration glitch (0 at 1.25 ms) | 12 (0.47 %) | 360 (14 %) |

No world raised a health flag in the free-motion and standing rows. In the LEAP hand's rows a solver warning
(UNCONVERGED) showed in 1 to 3 of 1,024 worlds at 1.25 ms (as shipped too) and at 2.5 ms, in none at 5 ms;
every world passed `sim.check()`. Reading: a robot that moves, reaches or stands
is as good at 2.5 ms and close at 5 ms; a robot that grips is not, and the build's warning says so (damping,
solref). Under "auto" only the Go2 leaves 1.25 ms (2.5 ms) among the shipped robots: the damping cap holds the
arms and both humanoids, and the LEAP hand is held by the cube in its file and its contacts' time constants
(loose_body, solref). G1 at 1,024 worlds and every robot at 4,096 worlds ran out of memory on a
24 GB L4 at every step, so they have no GPU rows.

A pinned step that breaks the damping or solref cap is not for a robot that grips: at 2.5 ms 0.47 % of pick-and-place
episodes had a penetration glitch and at 5 ms 14 % (0 at 1.25 ms).

## Why
The fixed step was chosen once for all robots: stiff position servos, thin fingers and contacts written for
1.25 ms. Whether a given robot is stable and as accurate at a larger step depends on its servo gains, its link
inertias and the contact time constants in its file, which the caps above now read. Making the larger step the
default would change every robot scene's motion, so it stays a choice.

## Workaround
- Set `scene.rigid_step = "auto"`, or pin a step, and read `sim.routes()` for the step and the cap that bound it.
  After a pinned step, read the build's warning before trusting grasps and contact depths.
- A robot file that writes 2.5 ms time constants on its own shapes, limits or equality rows stays at 1.25 ms
  under "auto" (the shipped FR3, Panda and UR10e files do). To step it coarser without softer contacts than the
  file intends, write time constants of at least twice the step you want.
- Light loose objects in the robot file hold "auto" at 1.25 ms. Add them with `scene.add_box` and the like instead:
  scene objects have their own caps, and a light one still holds the fixed step.
- Check the run as always: `sim.check()["ok"]` and `sim.health()`.

## Planned fix
Not fixed yet:
- the default: a robot scene that never sets `rigid_step` keeps 1.25 ms;
- arms held at 1.25 ms by their servos' damping, by 2.5 ms contact time constants in their files and by the light
  loose objects in them;
- a robot beside cloth, liquid, soft bodies, sand or paste, a kinematic object, a precision fit or an attachment
  keeps the fixed step, and so do the precision and debug profiles under "auto"; `rigid_step = "fast"` is not offered
  for robots;
- a servo on a tendon and joint limits written as stiffness and damping (URDF, USD) are not checked by the caps,
  and a loose body in the robot file is checked for its contact stiffness only, not for how far it moves in a step.
Making "auto" the default for robots is proposed (it changes only the Go2 among the shipped robots) and waits on a ruling.
