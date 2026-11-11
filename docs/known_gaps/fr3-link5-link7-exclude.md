# The shipped Franka arms do not collide link 5 with link 7 (Panda: link 5 with the hand)

**What is wrong.** The FR3 packs (`emb_e1_fr3`, `emb_e4_fr3_bimanual`, `emb_e5_fr3_duo` and their `_route` /
`_stiff` variants) carry an `<exclude>` for the pair link5 / link7, and `emb_e2_panda` one for link5 / hand: these
bodies never make a contact with each other. The exclude was added because the arms start at joint angles 0, which
is outside the range of joint 4 on every arm (FR3: -3.08 to -0.12 rad) and of joint 6 on the FR3 (0.44 to 4.62 rad),
and there the two links' collision hulls overlap. It also covers poses inside the joint ranges in which the hulls
overlap: there the arm passes through itself with no contact and no health flag.

**The number.** At joint angles 0 the hulls overlap by 5.6 mm (FR3 and each arm of the bimanual), 42.7 mm (Duo,
link 5 with link 7's hand piece) and 30.2 mm (Panda, link 5 with the hand); at the packs' `init` pose none overlaps.
Of 20,000 random FR3 poses inside the joint ranges, 198 (1 %) overlap by more than 0.5 mm, the deepest 16 mm, all
with joint 6 at or below 0.66 rad and joint 7 turned far; the Duo is similar, up to 19.7 mm. Not established:
whether the real links touch in those poses (the hulls are convex pieces of the collision meshes), and the Panda's
share of poses (only joint 6 was swept).

**Why.** `alloy.build` starts a robot at joint angles 0, not at the file's `init` pose, and for these arms 0 is
outside a joint range. Without the exclude every scene with one of these arms started with a self-overlap.

**Workaround.** Keep joint 6 above 0.66 rad while joint 7 is turned far, or check the clearance of link 5 and link 7
along a planned motion yourself (`mujoco.mj_geomDistance` on the pack's `scene.xml`).

**Planned fix.** After the release tag the FR3 and Panda packs get a start inside their joint ranges (the joint zero
moved to the home pose, as for the Go2), and the exclude is removed.
