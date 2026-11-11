# Known gap: the wrapped present cannot be unwrapped yet

## What
- The reference task set's unwrap-gift task: tear the taped wrapping paper off a present box until the box is free of
  paper (paper on at most 5 % of the box's surface, the box closed and upright).
- Alloy has the start state: the library's `gift_wrapped_present`, a 187 x 132 x 55 mm box wrapped in 82 um kraft
  wrapping paper (girth wrap with the seam under the box, ends folded; its OBJ carries the flat sheet, so a card can
  put tear lines on it), placed on its box by `alloy.tasks.gift_present.add_present` (the wrapped-present
  measurements in the development notes, not shipped).
- No task unwraps it: there is no scripted motion that takes the paper off, and no test of one.

## Number
- The task is not run: 0 % of the paper comes off (the reference's gate: paper on at most 5 % of the box).
- The start state rests: on an NVIDIA L4, 64 worlds x 1 s, the wrap moves at most 1.7 mm and the box tilts at most 0.3
  deg, health ok (the wrapped-present measurements in the development notes, not shipped). With tear lines, 25 of 64
  worlds lose the rest within 0.2 s (reason 5).
- The reference task set's own demonstrations did not unwrap it either, even with hands that attach to the paper.

## Why
1. **Nothing to take hold of.** The paper lies flat on the box. The gantry's fingers and hook
   (`alloy/tasks/gift_gantry.py`) pinch or push edges that stand free; the wrap's free edges are the seam's edge,
   pressed under the box, and the end flaps' edges, lying flat on the stacked end layers. The film's grip
   (`alloy/solvers/film/grip.py`) holds a sheet between two jaws, which also have to get round an edge first.
2. **The layers stand 1 mm apart.** A film particle collides at 0.5 mm, so every layer stands a pitch of 1 mm off the
   one below it (12 times the paper's 82 um), and an end stacks up to six layers. No tine or finger fits between two
   layers to lift a flap, and a thinner contact radius would need a finer clock than the coupling already runs.
3. **Tape.** The reference tapes the seam and the ends. WP13's Bonds plug-in bonds film to rigid bodies only; tape on
   paper needs film-to-film bonds that break by peel, which are not modelled.
4. **Cost.** The box on the film couples at the lightest body's clock: about 0.16 ms (j = 6), so a task of tens of
   seconds is a GPU-only run.

5. **Tear lines do not hold on the folded wrap.** Tear lines on the flat sheet build (alloy/solvers/film/layout.py:
   390 edges at 44 mm, every vertex copy on the sheet's grid), but with them the present does not rest. On an NVIDIA
   L4, 64 identical worlds x 0.2 s: 25 lose the rest (the outer layer of an end stack peels out from 0.04 s, the box
   tips 17 deg by 0.2 s, the film's strain projection unconverged), 39 rest; without tear lines all 64 rest (wrap
   <= 2.1 mm, box tilt <= 0.4 deg). Which worlds fail follows the VBD colouring: Newton's colour balancing colours
   identical worlds differently (worlds 41-63 alike). One world on a CPU with world 41's colouring moved 20 mm by
   0.08 s and kept going; with world 0's it moved 13 mm at 0.04 s and settled back to 4 mm. So the library card has
   no tear lines.

## Workaround
None for the whole task. For tearing a wrap by hand, use `paper_wrap_tear_sheet` lying flat with an edge the gripper can
pinch (the tear-line measurements in the development notes, not shipped); the wrapped present serves as a resting start
state (for looks, perception and contact with the box).

## Planned fix
Not scheduled: unwrap-gift was parked on 2026-11-07 for the open-carton task. It needs:
- A grasp for a sheet lying on a surface: an attachment-style hold on film particles under a fingertip (as the
  reference's hands), or a finger that slides under a lifted corner.
- Film-to-film tape: breakable bonds between two film layers with a peel law (an extension of the film's seams).
- A paper contact radius nearer the sheet's thickness, with the clock it needs.
- Seams that hold a folded sheet in every colouring (or one colouring per world), so the wrap can carry tear lines.
