# Scene variants: moved objects and swapped textures

`alloy.vary(scene, n)` makes `n` edited copies of a scene, each its own `alloy.Scene`, for visual and placement
diversity (training data, robustness checks). The same `seed` gives the same variants. It is not
`scene.randomize`, which draws values per world inside one simulation.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.3, 0.3, 0.4), pos=(0, 0, 0.2), static=True, material="wood_pine")   # stays put
scene.add_box("a", size=(0.04, 0.04, 0.04), pos=(-0.06, 0, 0.42), material="wood_beech")
scene.add_box("b", size=(0.04, 0.04, 0.04), pos=(0.06, 0, 0.42), material="abs")
variants = alloy.vary(scene, 4, pos=0.02, yaw_deg=15, seed=3, out="variants")   # variants/variant_000.json ...
sim = alloy.build(variants[2])                       # a variant is an ordinary scene
sim.run(0.5, record=[])
print(sim.state(world=0)["a.pos"], sim.check()["ok"])
```

- **Placement**: every free object's start x and y move within ±`pos` m and its yaw within ±`yaw_deg`. The
  ground, static, kinematic and held objects, robots, fixtures and liquids stay. Objects that touch at the start
  move together (`groups` in the reply). A draw that overlaps something or takes an object off what it rests on is
  drawn again, up to `max_tries` (200) times per variant, then a `SceneError`.
- **Looks**: each object's look takes another texture of its family (wood to wood, fabric to fabric) from a pool of
  CC0 textures (Poly Haven, ambientCG) in eight families: wood, metal, plastic, fabric, stone, tile, paper, floor.
  A small colour tint is added (`tint=0.05`; 0 for none). `textures="same-family"` (default), `"textured"` (only
  looks that already have a texture) or `"keep"` (positions only).
- **Checked variants**: `settle=0.5` builds each variant on the CPU and runs `alloy.check` for that long, drawing
  again until it passes. It is slower; use it when objects are stacked or close.
- **Contact sheet**: `contact_sheet=True` also writes `contact_sheet.png`, one tile per variant
  (`sheet={"width": 240, "height": 180, "spp": 4, "quality": "full" | "fast"}`). The first sheet compiles the camera,
  about 1 min. Textures not downloaded yet are left out of the sheet only: `python -m alloy.looks --fetch-textures`
  downloads them into the render cache (`$ALLOY_CACHE/render` when `ALLOY_CACHE` is set, else `~/.cache/alloy-render`;
  `$ALLOY_RENDER_ASSETS` overrides both).
- `out=` writes `variant_000.json` ... (Scene JSON that `alloy.Scene.load`, `alloy run` and the tool server take)
  and `variants.json`, which says what each variant moved and swapped.

## Command line and tool server (JSON in and out)

```bash
alloy vary scene.json --n 2 --pos 0.02 --yaw 15 --seed 3 --out variants    # any scene form: a name, a file, JSON
alloy run variants/variant_001.json --seconds 0.5 --names a,b --out v1.npz
```

The same request as JSON (`alloy batch`, or the `vary` tool of `alloy mcp`):
`{"op": "vary", "scene": "scene.json", "n": 2, "pos": 0.02, "yaw": 15, "seed": 3, "out": "variants"}`. Other
fields: `textures`, `tint`, `settle`, `max_tries`, `sheet` (true: the contact sheet; the tool server returns it as
an image) and `sheet_options`. Without `out`, the files go to a new temporary folder. The reply (paths shortened,
one variant shown):

```json
{"ok": true, "op": "vary", "result": {
  "n": 2, "seed": 3, "out": "/.../variants", "manifest": "/.../variants/variants.json",
  "files": ["/.../variants/variant_000.json", "/.../variants/variant_001.json"],
  "groups": [["a"], ["b"]],
  "families": {"fabric": 8, "floor": 8, "metal": 8, "paper": 8, "plastic": 9, "stone": 8, "tile": 8, "wood": 11},
  "variants": [{"file": "/.../variants/variant_000.json", "draws": 1,
                "moves": {"a": {"dx": -0.0166, "dy": -0.0105, "yaw_deg": 9.04},
                          "b": {"dx": 0.0033, "dy": -0.0162, "yaw_deg": -2.01}},
                "textures": {"ground": {"plywood_marine": "polyhaven:cherry_veneer"},
                             "table": {"wood_pine": "polyhaven:walnut_veneer"},
                             "a": {"wood_beech": "polyhaven:kitchen_wood"},
                             "b": {"abs": "ambientcg:Plastic016A"}}}],
  "sheet": null, "sheet_left_out": []}}
```

`moves` are metres and degrees from the original start; `textures` maps each object's look to the texture it took;
`draws` counts the tries the variant needed. Runnable: `examples/basics/13_scene_variants.py`,
`examples/cli/05_vary.sh` (with a contact sheet).
