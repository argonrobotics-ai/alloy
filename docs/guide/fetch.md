# Fetching objects from the object library

Alloy's object library is a dataset on the Hugging Face Hub: about 5,400 object cards, each with its meshes, textures
and thumbnail. `alloy fetch` downloads the objects you name into a local cache, where scenes, compose and the API find
them. Files already there with the right sha256 are not downloaded again. Alloy itself ships 118 objects.
`alloy.api.catalog.objects()` lists the objects on disk (shipped and fetched), `alloy fetch --list` the library's.

- What the cards are (5,423 in the dataset's index): 1,111 YCB and Google Scanned Objects scans; 1,989 Objaverse
  objects; 1,900 kitchen objects and fixtures made by Lightwheel (NVIDIA's
  PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF dataset, CC BY 4.0, as packaged by RoboCasa, with our
  collision hulls, estimated masses and materials; no RoboCasa kitchens or tasks); 393 cards from Gym-Anything for
  Robotics, one per rigid part of its articulated objects; and 30 of Alloy's own objects. 263 of the kitchen cards
  carry a flag for a failed mass, size or physics check (`flags` in the index).
- What the values are: every value carries a source tag, and most are estimates. The index marks 5,180 of the 5,423
  cards `estimate: true`. Mass (`mass_basis`): 122 of the YCB and Google Scanned objects use a published or measured
  weight; every Objaverse and kitchen card and the other 989 scans carry an estimate (a typical mass for the category,
  or mesh volume times a material's density: the card's note says ESTIMATE). Friction, restitution and stiffness are
  those of the library material the card names, never measured per object: the friction coefficient is an
  engineering-judgement value (`model`) for two thirds of the objects and a catalogue value for the rest, restitution
  `model` for every one. Weigh the object, or check the card, before a result depends on its mass.
- Needs the `import` extra (`install.md`; it brings `huggingface_hub`) and the network.
- The library is a public Hugging Face dataset: no access request, account or token is needed. A login
  (`hf auth login` or `HF_TOKEN`) is optional; the Hub gives logged-in downloads a higher rate limit, which helps for
  whole libraries. A private copy (`$ALLOY_ASSETS_REPO`) needs access and a login; without it the reply is a
  `HubError` whose fix says this.
- Each object's card records the licence its source declares (CC BY 4.0, MIT, CC0 1.0 or Apache-2.0); check it, and
  the credit line in the dataset's `ATTRIBUTION.md`, before commercial use.
- Files go to `~/.cache/alloy/imports/<library>/` (`$ALLOY_CACHE` moves the cache; `$ALLOY_ASSETS_REPO=<owner>/<name>`
  reads another copy of the library).
- Texture sets for looks and scene variants (`python -m alloy.looks --fetch-textures`, see `variants.md`) go to the
  render cache: `$ALLOY_CACHE/render/` when `ALLOY_CACHE` is set, else `~/.cache/alloy-render/`;
  `$ALLOY_RENDER_ASSETS` points it anywhere else. The camera reads textures from there.

```bash
alloy fetch --list --library ycb_gso                 # what there is (libraries, categories, names); nothing downloaded
alloy fetch ycb_025_mug gso_50_blocks                # objects by name (commas work too)
alloy fetch --library robocasa --category blenders   # a category; --library alone fetches a whole library
```

Libraries: `ycb_gso`, `robocasa`, `gymanything`, `objaverse`, `alloy`. A whole library is large (`ycb_gso` is 1,111
objects, 1.8 GB): list first, then fetch what the task needs.

## JSON in and out

The same request as JSON (`alloy batch`, or the `fetch` tool of `alloy mcp`): `{"op": "fetch", "names":
["ycb_025_mug"]}`; other fields `library`, `category`, `list` (true), `repo` and `workers` (8). The reply:

```json
{"ok": true, "op": "fetch", "result": {"repo": "...", "objects": 1, "downloaded": 9, "bytes": 4095870, "kept": 0,
 "roots": ["/.../.cache/alloy/imports/ycb_gso"], "names": ["ycb_025_mug"]}}
```

`downloaded` counts files fetched now, `kept` the files already in the cache (a second fetch of the same object reads
`"downloaded": 0`), and `roots` the library folders written. `--list` replies with `objects`, `bytes`, `libraries`,
`categories` (`{"ycb_gso/gso": 1033, "ycb_gso/ycb": 78}`) and up to 200 `names`.

## Using a fetched object

```python
import alloy
from alloy.assets.hub.fetch import fetch

reply = fetch(["ycb_025_mug"])                       # the same reply as `alloy fetch ycb_025_mug`
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
scene.add_object("mug", "ycb_025_mug", pos=(0, 0, 0.45))    # the object name is its card name
sim = alloy.build(scene)
sim.run(1.0, record=[])
print(sim.state(world=0)["mug.pos"], sim.check()["ok"])
```

A fetched object works like a shipped one: `scene.add_object`, a scene card, `alloy run` scene JSON
(`{"add": "object", "name": "mug", "card": "ycb_025_mug"}`) and `alloy.compose.compose(...)` all find it in the
cache, also when it was fetched later in the same process. On a fresh install (no local copy of the library) with
the `import` extra, `add_object` also takes a name that is only on the Hub and the build fetches it, so `alloy fetch`
is a way to download ahead of time and to see what there is. Without the extra, `add_object` raises `SceneError`
for any object not on disk ("object card ... is not on disk"), its fix naming the extra
(`pip install 'alloy-engine[import]'`) and `alloy fetch <name>`. With the extra,
a name that is neither on disk nor on the Hub raises "no object card ...", its fix listing close names, the Hub's
included.
Runnable: `examples/basics/14_fetch_objects.py`.
