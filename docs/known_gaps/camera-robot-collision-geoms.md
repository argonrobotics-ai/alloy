---
owner: Alloy maintainers
---

# A robot file with one visual-only geom does not draw its welded parts' collision geoms

## What is wrong
- When a robot's MJCF has any visual-only geom (`contype="0" conaffinity="0"`), the importer hides all its collision
  geoms. Images (`sim.render`, `alloy.Camera`) draw them again on links that move: every collision geom of a link
  without a visual geom, and on a link with one, the collision geoms of groups 0 to 2 that stick out of its visual
  geoms (a marker on a hand no longer hides the hand).
- Parts welded in place keep only their visual geoms: a body without a joint (a robot's base, a rail, a stand) that
  has only collision geoms is not drawn. Physics is not affected: the hidden geoms still collide.
- A robot file with no visual-only geom draws all its collision geoms (the arm in AGENTS.md section 5 is drawn).
  Robots from the shipped library draw their visual meshes as before; collision geoms in group 3 stay hidden, as in
  MuJoCo's viewer.

## Number
- A slide and a hinge on a rail welded to the world, the hand a jug of 5 box geoms, and a 4 mm visual-only sphere on
  the hand: the marker, the carriage and the 5 jug boxes are drawn, the rail's box is not (7 of 8 geoms drawn, CPU).

## Why
- The importer splits a file's geoms into visual and collision geoms. When the file has visual geoms, it shows those
  and hides every collision geom, as robot files with separate display meshes expect. The camera draws the hidden
  geoms of moving links again; welded parts keep the importer's choice.

## Workaround
- Give each welded part you want drawn a visual twin: the same geom with `contype="0" conaffinity="0" density="0"` (no
  contact, no mass), next to the collision geom.
- Or drop the visual-only geoms from the file: a file with none draws its collision geoms.

## Planned fix
- Draw the collision geoms of groups 0 to 2 on welded parts too, as MuJoCo's viewer does.
