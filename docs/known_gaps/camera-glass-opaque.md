# Glass objects render opaque unless a look file makes them glass

## What is wrong
- An object of a glass material (`material="glass_soda_lime"`) is drawn as an opaque coloured solid, on the default
  fast camera and with `quality="full"` alike: what stands behind a pane or inside a glass is hidden.
- Liquids are different: `quality="full"` draws a sim's liquids as clear liquid.

## Number
- A red cube 20 cm behind a 1 cm glass pane: hidden in both looks; with the look file below the cube is seen
  through the pane.

## Why
- By design the fast camera draws matte surfaces only. `quality="full"` gives every object a matte plastic of its
  colour: a physics material card does not choose how an object looks.

## Workaround
- Give the object a glass look in a look file, which also turns on the full-material camera:
  `look = {"materials": {"pane_glass": {"preset": "glass_clear"}}, "assign": [{"match": "pane", "material":
  "pane_glass"}]}`, then `sim.render(look=look)` or `alloy.camera.camera_scene(sim, look=look)`. `match` is a part
  of the object's name.
- Depth and segmentation images always take the glass surface (they do not see through it).

## Planned fix
- Default looks by material family (glass, metal, rubber) for `quality="full"` (after the launch).
