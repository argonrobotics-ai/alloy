# alloy.compose: a sentence in, a checked scene out

`alloy.compose.compose(sentence)` picks objects from the card library, places them on a table (or the floor) with
support, clearance and clutter rules, runs `alloy.check` after every placement and a full check at the end, and
fixes what the checks flag. It returns the scene and N variations as scene cards.

```python
import alloy.api, alloy.compose

result = alloy.compose.compose("a mug, a spoon and three wooden cubes on a table", variations=4)
print(result.ok)                # the base scene passed the full check and has every object asked for
print(result.summary())         # one line per scene
result.save("my_scenes")        # my_scenes/scenes/scene.yaml, scene_v01.yaml ... + compose.json
sim = alloy.api.build(result.scene(0), worlds=64)     # result.scene(i) is an editable alloy.Scene
```

Command line (only JSON on stdout; exit code 0 = passed, 1 = built but not passing, 2 = bad input):

```bash
python -m alloy.compose "three nuts in a tray on a cluttered table" --variations 4 --out my_scenes
python -m alloy.compose parse "a tidy desk with two pencils and a mug"     # the spec a sentence gives
python -m alloy.compose build --spec spec.json --check load                # exact control as JSON
python -m alloy.compose search mug --hint ceramic -k 5                     # what exists, what can be placed
```

## What a sentence can say

| Say | Meaning |
|---|---|
| `a`, `two`, `3`, `a few`, `several`, plurals (`cubes` = 3) | how many |
| `wooden`, `steel`, `glass`, `plastic`, `ceramic`, ... | prefer that material |
| `small`, `large` | the smallest / largest of the equally good matches |
| `in a tray`, `on the plate`, `with two nuts in it` | on or in another item (it is placed first) |
| `on a table` / `desk` / `bench` (default), `on the floor` | the support |
| `cluttered`, `messy` / `tidy`, `neat` | gaps of 4 mm / 40 mm (normal 15 mm) and a smaller / larger work area |
| `in a row`, `in a grid` | arrangement (otherwise scattered, random yaw half the time, square to the table otherwise) |
| `in 64 worlds` | parallel worlds in the saved scene card |
| `randomized` / `jittered` (`in 64 randomized worlds`) | each world moves the free objects by up to 0.45 x the gap in x and y |

Words it does not use (colours, brands) are kept as hints and listed in `result.spec.notes`. For exact control,
write a `SceneSpec` as JSON:

```json
{"items": [{"query": "tray", "object": "tray_aluminium"},
           {"query": "nut", "count": 3, "relation": "in", "target": "tray"},
           {"query": "mug", "words": ["ceramic"]}],
 "support": "table", "clutter": "normal", "arrangement": "scatter", "worlds": 16}
```

## How it builds a scene

1. **Choose.** Each item's noun is searched in the catalog: Alloy's `library/`, the alloy-assets checkout and every
   import library under `<alloy-assets>/imports/<source>/` (new ones are picked up automatically; titles and
   categories in their `manifest.json` are searched too). Matches are ranked by noun, synonym and hint words, and
   the first placeable one wins. Rigid, static and soft solid objects (fruit, foam) with a primitive, mesh or MJCF
   shape are placeable; cloth, liquids, grains and cables are made by the soft builder (below). An imported object
   whose files are not on disk is fetched from `$ALLOY_ASSET_STORE` when it is chosen (sha256-checked).
   A soft solid (an object whose material would make it a soft body: an apple, a banana, a sponge) is placed as a
   **rigid copy**, `<card>_rigid`, written to `objects/` with the same shape, mass and friction, because a soft body
   steps far slower: three soft apples took about 90 s per 20 ms control step on a CPU, about 0.1 s rigid. Say so
   to get the soft body ("a soft apple"; also squishy, squashy, deformable, squeezable, spongy, compressible; in a
   spec, `"words": ["soft"]`). `result.scenes[i].log` says which way each such object went, with that cost.
   The copy keeps the object's mesh and material, so the object itself must be on the machine that runs the
   saved folder: `alloy.compose.load_scene()` fetches it there (as a build does for objects a scene names).
2. **Rest pose.** Each object keeps its authored orientation when that stands well, else lies on the convex-hull face
   that stands best (a pencil authored upright is laid down).
3. **Place.** Containers first, then the largest objects. Each object is dropped at a sampled (x, y, yaw): vertical
   rays find the highest surface under it, so items land on the table, a tray floor or a box top. It must stay
   inside the table (20 mm from the edge) or the outline of what it goes in, keep the clutter gap to its neighbours,
   and stand: a flat surface, or its centre of mass inside the patch it touches.
4. **Check after every placement.** `alloy.check.check_load` (cards, overlap in the engine's own collision shapes).
   A small overlap is fixed with the suggested lift; otherwise the object is tried elsewhere (3 times).
5. **Full check.** `alloy.check` with a short simulation (rest drift, penetration, health, energy). An object an
   error names is moved once. If it fails again in its new place, another object that matches the same words takes
   its place (the log says which, and why; the failed card is not offered again in that call); it is dropped, and
   listed in `missing`, only when no other object matches or the item named its card. The report is in
   `result.scenes[i].report`, and `result.summary()` prints a failed check's errors with the change for each.
6. **Variations.** New seeds: new poses and, unless `swap=False`, other equally good objects (any mug for "a mug").
   Only variations that pass are kept (up to `2 * variations + 1` tries). Within one scene, `jitter=True` gives each
   of its parallel worlds its own x/y offsets of the free rigid objects (per-world randomization in the card; less
   than half the gap, so no new overlaps; objects in or under others stay put).

`check="load"` skips the full check (fast, a few seconds); `check="none"` places by geometry only.

Each scene in `compose.json` carries `metrics`: `placed`, `missing`, and, after a full check, `stable` (share of
objects with no rest-drift finding) and `overlap_free` (share with no overlap finding at load).

## Liquids, grains, cloth and cables

Soft words make soft items. The builder makes a card for each from a library material or template card, puts it
where the sentence says, and the engine's router picks the solver (SPH or MPM liquids, MPM or rigid grains, VBD
cloth, chain cables).

```python
r = alloy.compose.compose("a mug with 200 ml of water and a towel draped over a jar", variations=8)
r.scenes[0].soft["items"]       # one row per soft item: amount, container, route, start shape, ...
r.scenes[0].soft["estimated"]   # what the builder assumed (stand-in materials, capped amounts, pour rate)
r.scenes[0].soft["settle_s"]    # how long the scene takes to settle (a drape falls, a pile slumps, a pour ends)
```

| Say | What you get |
|---|---|
| `a cup with 200 ml of water`, `a glass half full of milk`, `a jar of rice`, `a bin with 0.5 kg of sand`, `a jar with 100 ml of glycerol` | a fill: the liquid or grains start at rest on the container's inner floor, a cylinder or a box as wide as its cavity allows |
| `pour 150 ml of water into a mug`, `pour water from a bottle into a cup` | a stream (an emitter, 50 ml/s) into the container; a named vessel is held still and tilted over it |
| `pour 150 ml of water from a jar tilted at 30 deg/s into a bin` (`tilted to 100 degrees`) | the jar, filled and held still (kinematic) with its lip over the bin, and the tilt as data: `alloy.compose.tilt(sim, row)` drives it |
| `a pile of 300 g of sand`, `a pile of rice on a tray` | grains that slump to their angle of repose (the material's sourced friction angle) |
| `a towel`, `a 50 x 30 cm towel`, `a towel folded in half`, `a towel draped over a mug`, `a towel hanging over the edge of the table` | a cloth sheet: flat, folded in half (two layers), falling over an object, or with 40 % past a table edge |
| `a cotton cloth`, `a polyester cloth`, `a plastic sheet` (`film`, `plastic wrap`), `a napkin`, `a sheet of paper` (A4), `kraft paper` (`brown paper`), `wrapping paper` (`gift wrap`, `packing paper`), `paperboard` (`chipboard`), `scored paper` (folds on its crease) | the library sheet of that name (cloth, film, paper); `rag`, `blanket`, `tablecloth` use the closest one, listed in `estimated` |
| `a gift bag` (`paper bag`, `kraft bag`, `shopping bag`), `a bottom board in a gift bag` (`bag insert`), `a box in a gift bag`, `a plastic bag` | a library film object, placed like any object: the open kraft gift bag standing on its bottom, its chipboard bottom board centred on the bag's bottom, a flat plastic carrier bag |
| `a 1 m usb cable`, `a 3 mm cable`, `a rope plugged into a jar`, `a cable from the laptop to the monitor` | a cable laid in a loose curve (or `straight`), its ends welded to the objects named |
| `a robot arm next to a bin of rice and a mug` | the library Franka Panda (the robot alone) at the table edge |

A tilting pour, run (the scene is at rest until you drive it; `examples/compose/tilt_pour.py` runs it and reports
the share of the water inside the bin's opening, from the row's `opening_centre` and `opening_radius_m`;
`examples/api/06_tilt_pour.py` shows the same drive on hand-built boxes):

```python
r = alloy.compose.compose("pour 120 ml of water from a jar tilted at 25 deg/s into a bin")
plan = next(row for row in r.scenes[0].soft["items"] if "pivot" in row)   # vessel, pivot, angular_velocity, ...
sim = alloy.api.build(r.scene(0))
sim.run(plan["start_s"])
alloy.compose.tilt(sim, plan)                # sim.drive(vessel, angular_velocity=..., pivot=lip, duration=...)
traj = sim.run(plan["duration_s"] + 1.5, record=[plan["liquid"], plan["vessel"]], every=10)
```

Amounts: ml, l, cl, g, kg, `full`, `half full`, `a little`; with none, a fill takes 60 % of the container's depth and
a pile 100 ml. Unknown liquids with water's properties (milk, juice, coffee, tea, wine) are made as water and listed
in `estimated`; other unknown materials are an error listing what exists.

Never a block dropped into a container: a flat block striking a floor throws pressure jets out of the cup. A fill
starts resting inside its container and a pour is a stream.

Limits the builder states rather than hides:
- a fill must settle below 90 % of its container's depth (the error gives the capacity);
- grains made from a fill card with a grain size that routes to rigid grains are capped at the router's 2,000
  grains per world (listed in `estimated`); the soft pack's rice and sand fills route to MPM and are not capped;
- some solver pairs cannot share a scene yet (cloth beside grains or a paste, cloth beside a paper or plastic film
  sheet: `alloy.coupling.unsupported`). The builder plans each scene before any build and leaves one side out, a
  fill or pour first: "a bin of rice and a towel" gives the bin and the towel, with the rice in `missing`, the
  engine's reason and "ask for one of them per scene". Water with a towel works, and so does water with sand;
- a cable's laid curve is its rest shape (it keeps that set). The first time a machine routes a cable of a given
  material, gauge and length, the engine runs its cable self-test on one CPU core: about 11 min for the 1 m USB
  cable on a cloud GPU machine, 4 min for a 0.6 m cotton rope. The result is cached (`~/.cache/alloy`), and a
  cable with one free end measures exactly the length asked, so every later scene, variation and run with that
  cable finds it; a cable between two objects is as long as their distance needs, so it can pay again;
- `folded in half` cloth is one mesh of two layers joined at the fold, with self contact on; the fold is its rest
  shape (it does not spring open);
- a pour runs when you build the scene through the API: `alloy.load(path)` or `alloy.build(result.scene(i))`
  turns the card's stream into `Scene.add_pour`, named in the pour row's `liquid`. `alloy.check` does not run a
  card's pour yet, so the full check reports the pour as not run (info);
- in a scene with a tilting pour, every object is held still (placed as a static copy, listed in `estimated`):
  beside free rigid bodies Alloy's liquid does not yet feel a moving (kinematic) vessel, and the build says so. A
  robot arm in such a scene has the same problem;
- free objects resting on or in a film object (a box in the gift bag) touch the film at a fine time step set by
  the lightest of them (0.16 ms for a 420 g box): the scene runs several times slower than the bag alone, and the
  scene's `log` says so. A flat sheet card (the bag's bottom board, the paper sheets) is placed as an object only
  when named (`bottom board`, `bag insert`); otherwise ask for it as cloth (`a sheet of kraft paper`);
- grains (MPM) need GPU memory per world: on a 24 GB GPU, 16 worlds of a bin of rice with a towel ran out of
  memory; check such scenes with fewer worlds (`compose(..., worlds=4)`).

The full check adds the check `soft` to the report: the fill is inside its container at the end (98 %), the
particles stand for the amount asked (within 15 %), cloth stretch, cable length, everything inside the workspace and
settled. Its numbers are in `report["metrics"]["soft"]`. A spec item can be written directly:

```json
{"items": [{"query": "mug"},
           {"query": "water", "relation": "in", "target": "mug", "soft": {"kind": "liquid", "volume_ml": 200}},
           {"query": "cable", "words": ["usb"], "soft": {"kind": "cable", "length_m": 0.8, "attach": ["mug"]}}]}
```

## Robots and the work area

```python
result = alloy.compose.compose("three cubes and a mug on a table",
                               robots=[{"path": "my_arm.xml", "name": "arm", "reach": 0.7}])
```

A robot stands at the middle of the table's near edge (-x) facing +x; a second one at the far edge facing -x; a
`pose` (x, y, z, yaw) puts one anywhere. Nothing is placed within `clear` (default 0.15 m) of a base, and the work
area is the part of the table within `reach` (default 0.8 m) in front of the robots. The load checks after each
placement include the robot's links. Without robots, `area=(x0, y0, x1, y1)` limits where objects go. Both are also
spec fields (`"robots": [...]`, `"area": [...]`) and command-line flags (`--robot arm.xml`, `--area X0 Y0 X1 Y1`).

## Using a saved scene

```python
scene = alloy.compose.load_scene("my_scenes/scenes/scene.yaml")    # adds the import libraries' card roots
report = alloy.check("my_scenes/scenes/scene.yaml", library=alloy.compose.library(["my_scenes"]))
```

The floor support is a generated card (`objects/compose_floor.yaml`) saved next to the scenes, so the folder loads
on its own. Imported objects need their files on disk: `compose` fetches the ones it uses when `$ALLOY_ASSET_STORE`
is set; `search` never downloads and says when an object's files are missing. Fetched files go into the import root
when it is a writable checkout, else (cards installed with the package) into `$ALLOY_CACHE/imports/<source>` or
`~/.cache/alloy/imports/<source>`; `alloy.compose.library()` and `load_scene()` include those cache folders.
