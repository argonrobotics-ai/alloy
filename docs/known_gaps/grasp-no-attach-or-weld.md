# No call attaches or welds an object to a gripper: friction is the only grasp

**What is wrong.** A robot holds an object by contact and friction only. There is no call that attaches or welds an
object to a gripper while a simulation runs, and so none that releases one. A script or policy that closes the
fingers and then fixes the object to the hand has no equivalent here: if the grip does not hold by friction, the
object slips or stays behind.

**The number.** There is no error to measure: the call does not exist. What the two ways to hold an object did in
trials (`docs/guide/robots.md`, Grasping): by friction, the default 2 mm cloth slipped out of a fingertip pinch (0 of
64 worlds carried an edge); a cup built into the hand's body in the robot's MJCF poured 96-99 % into a second
container in all 64 worlds.

**Why.** Not built. The public API has no call that fixes an object to a robot's hand, or frees it, while a
simulation runs.

**Workaround.** Two ways, both in `docs/guide/robots.md`, Grasping:
- A friction grasp. Set the friction with the object's `material` or `friction=`, or with
  `scene.set_friction(a, b, mu)` for the finger-object pair. Give the fingers enough actuator force to squeeze (MJCF
  `kp` times the target error), close them onto the object, then ramp the lift. Judge the grasp by the object: record
  its `.pos` and check that it follows the gripper.
- A body of the hand. To carry something rigidly, put its geoms in a `<body>` under the hand body of the robot's MJCF,
  with no joint between them. It then moves exactly with the hand, and it pours from a held container. The cost: the
  object cannot be put down.

**Planned fix.** None planned.
