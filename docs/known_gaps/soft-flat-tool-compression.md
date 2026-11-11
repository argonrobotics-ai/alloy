# Known gap (model limit): a flat tool pressed into soft FEM compresses it without limit

## What
- Every cut model (pre-split seams, seam lattices, virtual-node copies) parts material only where a blade edge
  passes.
- A flat face pushed into a soft block cannot cut it, so the block is squeezed without limit. Tets invert, and any
  fragments cut from them later flip.
- An edge pressed where no seam plane lies under it does the same, for example a knife pressed onto cut slices that
  lie flat (the robot case below).

## Number
- Random action traces on `cut_block_free` (agar-like block, 5e4 Pa, 4 worlds, 0.3 s) took whole tets to 85 %
  principal strain at p90 and 208 % at most before a split.
- The fragments inherited that strain: p90 35 %, max 198 %.

## A robot case: a knife pressed onto cut slices (JOBRK finding 1, 2026-11-09)
- Setup: the T2 cucumber clear, restored from its hand-over checkpoint (4 worlds, t 105.9 s). The right arm pushes
  the knife straight down onto 1 mm cut slices that lie flat on the board, with their seam normals 81-89 deg from
  the blade normal. The blade comes down across the slices, so it lies on no seam plane that could open, and it
  crushes them.
- At 0.1 m/s, all 4 worlds go INVALID at t 106.8-107.9 s. In 3 worlds a slice particle is 1.6-1.7 mm into the
  blade; in the fourth the material is pressed 1.2 mm into itself.
- At the 0.6 m/s command (the hand reaches 0.23 m/s), 3 of 4 worlds go INVALID as the knife reaches the board
  (t 106.36-106.40 s). In two, a slice particle is 0.6 mm into the board; in the third, the material is 2.0 mm into
  itself. After that impact the knife turns in the grip by 0.6, 5.3 and 22 deg. At 0.1 m/s it never turned.
- A second effect, now limited to seams aligned with the blade face: in that push the cutter damaged every seam bond
  its edge reached (the edge's depth times its travel, in any direction), with no test of the blade's alignment.
  `BladeContact` does test it (`min_alignment` 0.5). Of the 134 bonds the 0.1 m/s push broke, 115-117 (86-87 %) had
  their seam normal at |n . blade normal| < 0.5, with a median of 0.002-0.05 in three worlds. `BladeContact` would
  not have opened them. The same holds for 23-24 of the 25 bonds damaged past 0.5 but still intact.
  - Since 2026-11-09 the cutter has the same test (`TopologyController.Config.cut_alignment`, default 0.5): a blade
    cuts only seams within 60 deg of its own plane, so those bonds stay welded.
  - The 17-19 aligned bonds of that push can still break, and the crush itself is unchanged.
  - The push has not been re-run with the gate. A unit law covers the case: a blade pressed through a flat seam
    touches none of its bonds (`tests/topology/test_cut_alignment.py`).
- Workaround: clear cut pieces with something other than the blade edge, as JOB1's scripted clear does (the pads
  push from behind the last cut). From the same restore it kept all 4 worlds valid over 9 s (max penetration
  0.21 mm).

## The cutting tasks after the cutter's alignment gate (2026-11-10)
Random legal blade paths, 2,048 episodes each, seed 100000, 2 s, one L4, both arms on one commit (f690c5e48). "Any
seam" is the rule before, kept as the targets `cut_block_stroke_anyseam` and `cut_block_free_anyseam`.

| Task | Rule | Failed (S1) | Flagged only (S2) | Clean |
|---|---|---|---|---|
| `cut_block_stroke` | any seam | 615 | 2 | 1,431 |
| `cut_block_stroke` | gated | 86 | 3 | 1,959 |
| `cut_block_stroke` | gated, with the blade contact's re-entry case (the default since 2026-11-10) | 12 | 0 | 2,036 |
| `cut_block_free` | any seam | 1,954 | 1 | 93 |
| `cut_block_free` | gated | 1,326 | 16 | 706 |

- A repeat of three of these runs read 82 / 4 (stroke, gated), 1,324 / 15 (free, gated) and 1,957 / 1 (free, any
  seam): about +-10 episodes run to run.
- The S2 episodes are all `STRAIN_RANGE`, this page's flag. In the repeat, all 15 of `cut_block_free`'s were failed
  (PEN_FAIL) episodes under any seam; none was clean before. They read S2 now because the failure that came first
  is gone.
- World by world in that repeat, any seam to gated: 628 failed episodes became clean, 15 became flagged only, 1,314
  stayed failed, and 10 clean ones failed. The 19 gated episodes whose cause reads `inverted_tets` were failed
  under any seam too (17 as PEN_FAIL).
- What still fails:
  - `cut_block_stroke`'s 86 were PEN_FAIL late in the episode (median tick 94 of 125). Of 4 traced, 3 were an edge
    coming down again into an opened cut 0.1-0.9 mm off its seam: the freed twin lay just across the blade's
    mid-plane and the blade's bottom pressed on it, 0.53-0.58 mm deep against the 0.5 mm limit. The blade contact
    now puts a twin that lies ahead of a descending edge on its own face (`BladeContactConfig.reentry`; the rule
    before is the target `cut_block_stroke_noreentry`), and 12 of 2,048 fail.
  - 10-12 fail with it in four runs (two of them an independent audit's), and some of these are new with the case:
    worlds 13, 1432 and 2021 failed with it and were clean without it in both audit runs, and only 6 of the 10-11
    left are shared with the rule before. CPU replays, both arms, of the 10 from one run:
    - 1077 and 1432: the case draws a parted twin from past the far face (0.8-1.3 mm across the mid-plane, the
      blade 1 mm thick) back into it, and the far face's stock record reaches 0.66 / 0.86 mm. Without the case 1432
      is clean and 1077 fails later. A follow-up that allows the case only for a twin whose centre is inside the
      blade's thickness was tried and not merged:
      - it removed this pull, but the pull also kept twins on their side in other worlds (1343, 557, 1911), which
        then failed or were flagged without it;
      - its count stayed within the run-to-run spread (9-14 against 10-15 in 2,048, two runs each by an independent
        audit).
      - In both mechanisms, two contradictory blade records take turns at a rule's boundary: the ramp's edge here,
        and the cut-front pair cap in 1343.
      - A rule that keeps a twin's face side fixed through a contact is the planned fix.
    - 360 and 497 (the same with and without the case): the seam's bottom corner twin is inside the neighbouring
      piece after the cut is through. 835: the same with the case, clean without it.
    - 1024, 1798, 1868 and 2021 are clean on the CPU in both arms, and 1288 is clean with the case (flagged only
      without it): those failures were seen only on the GPU.
  - No robot scene has fired the re-entry case yet: JOB1 (a first press, and two rounds with the knife sweep) and
    the C9 dough clip counted 0 records, so the default changes nothing there. Its evidence is `cut_block_stroke`
    (86 -> 12, in two runs) and the law in `tests/topology/test_blade_contact_zone.py`.
  - `cut_block_free`'s 1,326 are not diagnosed. Its cut faces do not collide (no self-contact on that task), so
    part of them is cut pieces passing through each other.

## Why
- The FEM is linear-elastic on a fixed mesh.
- No cut model splits a tet under a face, and nothing remeshes as the material deforms.
- The model is valid only well below these strains.

## Workaround
- The cutting tasks hold face-first blade motion to a +-1 mm band, because a cutting task drives the edge. The
  tasks are `cut_block_stroke`, `cut_block_free` and `cut_block_free_copies`.
- The health flag `STRAIN_RANGE` (DEGRADED) marks any world past 60 % principal strain, so results past the range
  are flagged, not silent. The strain is measured by `alloy.topology.soft_strain.SoftStrain`.

## Planned fix
- Large-deformation remeshing, after launch.
- The second effect: a re-run of JOBRK's push with the cutter's alignment gate, to count the bonds it still breaks
  (WP10).
