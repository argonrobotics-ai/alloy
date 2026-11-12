# Shipped robots left with no targets move away from their start angles

**What is wrong.** A shipped robot added with `scene.add_robot(...)` and stepped with no targets does not keep the
joint angles it starts at. Two causes, one per robot family:
- **The start is outside a joint range** (the Franka arms). `alloy.build` starts a robot at joint angles 0, and on these
  arms 0 is outside the range of joint 4 (and of joint 6 on the FR3). The joint limits push those joints into range
  in the first 0.2 s, and the servos, whose default target is also 0, then press them against the limit.
- **Position servos sag under gravity** (the legged robots, the humanoids' arms). A position servo holds a joint by
  its error: under a steady load the joint settles torque / kp away from its target. With the source files' gains
  that is up to a few degrees.
The health checks do not flag either: `sim.check()["ok"]` is True in every case below.

**The number.** Largest |q(2 s) - q(0)| per joint, one world, CPU, 10 ms control steps, no targets; the same to four
digits on the MuJoCo C and the MuJoCo Warp routes (and with or without the C extension).

| pack | joint | moves | cause |
|---|---|---|---|
| `emb_e1_fr3`, `emb_e4_fr3_bimanual` (each arm), `emb_e5_fr3_duo` (each arm) | joint 6 | +0.439 rad | start 0 below its range (0.44 to 4.62 rad): pushed to the limit, 0.43 rad of it in 0.1 s |
| same | joint 4 | -0.117 rad | start 0 above its range (-3.08 to -0.117 rad) |
| `emb_e2_panda` | joint 4 | -0.070 rad | start 0 above its range (-3.07 to -0.070 rad) |
| `emb_e8_go2` | rear / front calves | -0.147 / -0.119 rad | the 15 kg body on kp 50 servos (Menagerie's gains): the base settles from 0.288 to 0.264 m |
| `emb_e8_go2` | thighs | -0.033 rad | same |
| `emb_e7_h1_2` | elbows | 0.045 rad | the arms' weight on kp 80 servos (the source profile's): torque / kp = 0.046 rad |
| `emb_e7_h1_2` | shoulder pitch, wrist pitch | 0.028, 0.011 rad | same, kp 120 and 40 (torque / kp 0.032, 0.011 rad); the passive thumb joints follow their mimic parents (0.028 rad) |
| `emb_e6_g1` | ankle pitch, waist pitch, knees | 0.022, 0.020, 0.017 rad | kp 500 legs and waist under the body |
The FR3 and Panda arms' other joints move under 0.002 rad.

**Why.** The Franka packs keep the sources' joint zero, which is not inside every joint range; the start pose is
`alloy.build`'s rule (joint angles 0), not the files' `init` keyframe. The sag is the sources' servo gains: a
real robot's controller adds a gravity feed-forward that these position servos do not have.

**Workaround.**
- Franka arms: send targets inside the joint ranges, ramped from the start (for example to the pack's `init`
  keyframe, `0 0 0 -1.571 0 1.571 -0.785` on the FR3), before anything that depends on the arm's pose.
- Sag: read the joint state rather than assuming the target, or aim the target past the wanted angle by the sag (the
  torque the joint carries divided by its kp, in the pack's `scene.xml`).

**Planned fix.** After the release tag the FR3 and Panda packs get a start inside their joint ranges (the joint zero
moved to the home pose, as for the Go2: `docs/known_gaps/fr3-link5-link7-exclude.md`). The sag is the sources' gains
and stays; no gravity feed-forward is planned.
