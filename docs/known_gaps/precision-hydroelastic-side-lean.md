# A hydroelastic precision fit leans past its clearance under a side push

## What is wrong
A peg seated in a bore whose precision features (an object card's `precision_features`) are `mode: hydroelastic`
leans about 1.3 times as far as the clearance allows when it is pushed sideways at its top, and the bore's wall takes
about 0.1 mm of penetration. A rigid peg would stop at the geometric limit, where its two wall contacts meet the bore.
The same pair in `mode: sdf` (the default) stops within 2 to 6 % of that limit, by push direction, with under 25 µm
of penetration (`precision-sdf-side-lean-direction.md`).

## Number
A 12 mm peg in a 12.2 mm bore, 20 mm deep (0.1 mm radial clearance; allowed lean 0.575 deg), 50 µm voxels,
kh 1e11. It is pushed 5 N sideways at its top for 0.5 s at a 0.3125 ms step, in 16 worlds with small random tilts and
offsets. GPU (L4), `tests/contact/test_lean_gpu.py` (bounds: 1.05 times the limit, 50 µm):

| pair mode, path | lean / allowed | wall penetration | commit |
|---|---|---|---|
| hydroelastic, MuJoCo Warp / Newton | 1.276 / 1.245 | 114 / 106 µm | 4b324c7 (2026-11-01) |
| hydroelastic, MuJoCo Warp, kh 1e12 | 1.338 | 130 µm | e7e697f (2026-11-01) |
| hydroelastic, MuJoCo Warp / Newton | 1.299 / 1.240 | - | rc2-1109 f94bcfc50 (2026-11-09) |
| sdf, MuJoCo Warp / Newton | 1.026 / 1.026 | 15 / 15 µm | 4b324c7 (2026-11-01) |
| hydroelastic, both paths, Newton 1.8 | 1.277-1.312 / 1.250-1.256 | 100-113 / 113-118 µm | bump (2026-11-10) |
| sdf, both paths, Newton 1.8 | 1.048-1.055 / 1.040-1.044 | 13-20 / 10-14 µm | bump (2026-11-10) |

- The same fit's insertion law (CT-1, `tests/contact/test_peg_hole_gpu.py`: a 50 N press and 5 N side pushes) reads, on
  one L4 at main f207ccaff, a deepest single sample of 100.5 µm against its 100 µm target in 1 of 3 runs; the others
  stay under it, and p99 is 19 µm against 50 µm. Its parent commit passed 3 of 3: at one failing run in six, the two are
  not shown to differ. MuJoCo Warp's GPU contact solve does not repeat run to run, so that worst sample moves by about
  0.5 %. The test marks that one bound as an expected failure that may pass; p99, tunnels, dropped contacts and a 110 µm
  guard stay hard.

## Why
Not resolved. A stiffer pressure field does not help (kh 1e12 leans further), so the field's modulus is not the
limit. The likely cause is the stiffness cap that the contact layer puts on precision rows at this step: their time
constant is at least two steps. The cap would hold the two wall patches of a leaning peg softer than the pressure law
asks. The sdf mode's per-patch weights reach the limit at the same step.

## Workaround
- Use `mode: sdf` (the default) for fits that carry side loads.
- Where a hydroelastic pair is needed (contact area, pressure), allow about 0.1 mm more wall penetration and 30 % more
  lean than the clearance gives.

## Planned fix
None scheduled. First check: the realised stiffness of the wall rows against the pressure law's (kh/2) A while the
peg leans, at this step and at half of it.
