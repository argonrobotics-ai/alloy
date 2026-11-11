# A robot file with one visual-only geom draws only its visual geoms

## What is wrong
- Images (`sim.render`, `alloy.Camera`) draw a robot from its visual geoms when its MJCF has any: geoms with
  `contype="0" conaffinity="0"`. Its collision geoms are then not drawn, though MuJoCo's own viewer draws geoms of
  groups 0 to 2.
- A hand-written robot with one such geom (a marker, a spout, a decoration) draws as that geom alone; the rest of
  the robot is invisible. Physics is not affected: the hidden geoms still collide.
- A robot file with no visual-only geom draws all its collision geoms (the arm in AGENTS.md section 5 is drawn).
  Robots from the shipped library draw their visual meshes as before.

## Number
- One capsule arm with a 1 cm visual-only sphere on it: the sphere is drawn, the capsule is not (1 of 2 geoms drawn).

## Why
- The importer splits a file's geoms into visual and collision geoms. When the file has visual geoms, it shows those
  and hides every collision geom, as robot files with separate display meshes expect.

## Workaround
- Give each part you want drawn a visual twin: the same geom with `contype="0" conaffinity="0" density="0"` (no
  contact, no mass), next to the collision geom.
- Or drop the visual-only geoms from the file: a file with none draws its collision geoms.

## Planned fix
- At import, keep collision geoms of groups 0 to 2 visible, as MuJoCo does (robot files with display meshes put
  their collision geoms in group 3, so they stay hidden).
