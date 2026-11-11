# Cutting soft objects with a knife

A kinematic knife with a cutting edge splits a soft (tetrahedral) object as it moves through it. Cutting is set up in
YAML cards: the object card says the object can be cut, and the knife's card marks its cutting edge. The scene API
(`add_box`, `add_mesh`) has no cutting option yet. Load the cards with `alloy.Scene.load`, move the knife with
`sim.drive`, and read the object's particles. Runnable example: `examples/topology/cut_anywhere.py`.

## 1. Quick start

```bash
python examples/topology/cut_anywhere.py            # writes its cards into ./cut_cards, then runs
```

It writes three cards, a 20 x 15 x 15 mm agar block, a 1.2 mm steel knife and a scene, then drives the knife down
across the block's long axis at 0.05 m/s. It prints the tets cut, how far the two pieces' mesh nodes have parted,
and the health check: `{"tets_cut": 24, "pieces_parted_mm": 0.09, "health_ok": true}`. The 1.2 mm knife parts the
pieces by only about 0.1 mm on this 5 mm mesh, because only mesh nodes meet the blade (a rendered run showed about
0.2 mm). It took 17-20 s in one process on an AMD EPYC 7B13 (a 4-core VM, other jobs running) once the
kernels were compiled; the first run compiles them for a few minutes.

## 2. A cuttable object

A soft object becomes cuttable through its card's `topology:` block. Two cut models exist:

| `cut_model` | Where it cuts | What the card needs |
|---|---|---|
| `copies` | along any path the blade takes | `topology: {cut_model: copies}` and a tet budget |
| `seams` (the default) | only along seams split into the mesh when the scene is built | the material's `failure` values (they set the seams' strength), and `split_spacing` (m) or `spare_vertices` in `topology:` |

```yaml
schema: alloy.object/1
name: agar_block
shape: {primitive: {kind: box, size: [0.01, 0.0075, 0.0075]}}   # half sizes
material: agar_cut                                              # a soft material: youngs, poisson, density
budget: {tets: 216}
topology: {cut_model: copies}
```

- With `copies` the knife meets no cutting resistance: it pushes only the material it meets face-first.
- With `copies` on a coarse mesh, a piece needs two layers of mesh nodes, `"<name>.points"` holds the copy vertices
  too, and a small cut piece creeps on the ground at the default accuracy
  (`docs/known_gaps/cutting-copies-coarse-mesh.md`).
- With `seams` a knife opens only seams within 60 degrees of its own plane. A seam it only crosses stays whole.
- Either way the material parts only where the edge passes. A flat face pushed into the block squeezes it instead;
  the health flag `STRAIN_RANGE` marks a world past 60 % strain (section 4).

## 3. A knife

The knife is a kinematic object (`motion: kinematic`) with a `blade:` block. `edge_a` and `edge_b` are the ends of its
cutting edge in the object's own frame, in metres. Optional: `edge_radius` (0.1 mm), `softness` (3.0, how fast the
edge's work weakens a seam) and `cut_spring_ke` (500 N/m, a seam's stiffness while it is cut).

```yaml
schema: alloy.object/1
name: knife
shape: {primitive: {kind: box, size: [0.06, 0.0125, 0.0006]}}   # a 1.2 mm plate (half sizes)
material: stainless_304
motion: kinematic
topology: {}
blade: {edge_a: [-0.06, -0.0125, 0.0], edge_b: [0.06, -0.0125, 0.0]}
```

Put both objects in a scene card (the example writes all three cards: copy `cut_cards/scenes/cut_block.yaml`), load
it, and move the knife:

- `scene = alloy.Scene.load("cut_cards/scenes/cut_block.yaml")`, then `sim = alloy.build(scene)`.
- `sim.drive("knife", velocity=(0, 0, -0.05), duration=0.33)` moves it down for 0.33 s.
- `sim.run(0.4, record=["agar_block"])` records `"agar_block.points"`, the block's particles (with `copies`, its
  copy vertices too), as the cut opens.
- `sim.check()` gives the health verdict, cut faces included.
- `alloy.export` frames carry the cut mesh, and `topology_version` counts the cut tets (`docs/guide/frame-export.md`).

The example passes no robot: the knife moves by `sim.drive`. No example of a robot holding a knife is published yet.

## 4. Limits

- A flat tool pressed into soft material compresses it without limit; a knife pressed onto pieces that lie flat
  crushes them (`docs/known_gaps/soft-flat-tool-compression.md`). How often a cut fails: on 2,048 random knife paths
  through a pre-split block (`cut_block_stroke`, one L4), 10 to 12 episodes failed a health check in four runs and
  none was flagged only (some of these failures come with the rule for an edge re-entering an opened cut: the
  known-gap page says which); with a knife free to tilt and turn through a block split every 5 mm
  (`cut_block_free`, measured before that rule), 1,326 failed and 16 were flagged only.
- A robot slicing task is not reliable. Our test task, two arms slicing a cucumber into 5 mm rounds and clearing them
  into a pan, was re-run on this release's cutting code with arguments and seed fixed in advance: 10 of its 16 worlds
  finished before a stop rule set in advance; 5 of the 10 were cut, 0 of the 10 cut and cleared, and 5 of the 10 ended
  INVALID by the health check for one soft piece pressed into another (a cut round pressed 10 % or more into its
  neighbour in a late press or in the finger clear; the health record does not name that rule yet). An earlier run of
  the task (cut 8 of 16, INVALID 4 of 16) was measured before that check existed, so the two INVALID rates cannot be
  compared: the earlier 4 of 16 is a lower bound under today's rules, since its valid worlds were never checked for
  this.
- The knife force a seam reports holds only until the first cut (`docs/known_gaps/cutting-seam-reference-plateau.md`).
- Cutting runs on the soft-body solver, so a scene with a cuttable object steps much slower than a rigid one,
  especially on a CPU (`docs/guide/install.md`).

Not covered:
- A knife cuts only soft (tetrahedral) objects, not rigid bodies, cloth or rods.
- The scene API has no cutting call: cutting is set up in cards.
- No example of a robot holding the knife is published.
- With `cut_model: copies` the knife feels no cutting force, so there is no cutting-force reading there.
- Cutting failure rates over many worlds are measured only for the two tasks above.
- With `cut_model: copies`, one knife per world: a second blade in a world raises an error at build.
- `alloy.Camera` and `sim.render` draw a cut object without its cuts. To see a cut, draw the mesh from `alloy.export`
  frames.
