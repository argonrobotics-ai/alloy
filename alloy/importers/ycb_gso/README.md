# YCB + Google Scanned Objects in Alloy

1,111 scanned household objects as Alloy object cards:

- 78 from the YCB Object and Model Set;
- 1,033 from Google Scanned Objects.

All are CC BY 4.0.

| Set | Objects | Mass |
|---|---|---|
| YCB | 78 `google_16k` scans | 63 measured (Calli et al. 2015, 2017), 15 ESTIMATE |
| GSO | 1,033 models (every one on Gazebo Fuel) | 59 sourced product weights; the rest ESTIMATE (documented category ranges, product descriptions or category rules) |

## Use an object (agents: start here)

```python
from pathlib import Path
from alloy.cards.roots import assets_root
from alloy.importers.ycb_gso.check import library

root = assets_root() / "imports" / "ycb_gso"
lib = library(root)                      # Alloy's starter library + the imported cards
card = lib.object("ycb_025_mug")         # validated card
mat = lib.material(card.material)        # ycb_025_mug_eff: density, friction, restitution
```

Fetch the mesh files first. Git holds only the cards, so this step is required:

```bash
python -m alloy.importers.ycb_gso fetch ycb_025_mug --store <asset store>   # or set $ALLOY_ASSET_STORE
python -m alloy.importers.ycb_gso list --lib ycb                            # names, masses, hull counts (JSON)
```

## What each object has

All files sit under `meshes/<name>/` and are checked against the sha256 in `manifest.json`.

| File | What |
|---|---|
| `physics.obj` | Watertight mesh, about 5k faces, body frame (metres, +Z up). The card's `shape.mesh`; mass = density × its volume. |
| `physics.hulls.obj` | CoACD convex hulls: at most 32 hulls of at most 64 vertices each. Collision uses these (`collision: decompose`). |
| `visual.glb` | The textured scan as glTF (+Y up, metres). glTF importers that convert to Z-up give back the body frame. |
| `thumb.png` | A 256 px preview. |

## Where the numbers come from

**Mass.** The card's material `<name>_eff` holds an effective density: object mass over `physics.obj` volume.
- With a measured or catalogue mass, the density has `source: fit`.
- Otherwise it has `source: model` and a note that starts with `ESTIMATE`.

For GSO, the estimate comes from the first rule that applies:
1. a shoe's catalogue weight in its description;
2. a packaged product's net contents plus 10 % packaging;
3. a per-category rule: a density, or a typical mass, anchored on measured YCB objects where one fits
   (`estimates.py`).

**Friction, restitution and elasticity.** Copied from a named library material. Every note says which one and why.

**Thin walls.** When the thinnest wall of `physics.obj` is under Alloy's tunnelling limit, the card declares it in
`features.min_wall`.

**Open values.** Every estimated value is a row in `queue.jsonl`, where the sourcing tool (`alloy.sourcing`) picks it up. GSO rows
carry the product GTIN and brand. The build applies the sourcing records in `alloy-assets/sources/imports/ycb_gso/`
over the estimates. A documented mass (`DOC`, `SNIPPET` or `UNVERIFIED`) retires the row. `manifest.json` gives
each object's `mass_status`: `measured`, `sourced` or `estimate`.

## Rebuild

```bash
python -m alloy.importers.ycb_gso jobs --out jobs --chunks 4
python -m alloy.importers.common.batch jobs/gso_0.json out --workers 30 --timeout 900 --mem-gb 6   # per chunk
python -m alloy.importers.ycb_gso build out
python -m alloy.importers.ycb_gso check --workers 3
python -m alloy.importers.ycb_gso publish out --store <asset store>
```

The mesh steps need `trimesh`, `fast-simplification`, `coacd` and `Pillow`. They do not need a GPU. On 32 cores the
whole set takes about 5 minutes.

## Known limits

- **076_timer** is left out. Every `google_*` archive of it upstream is a truncated 45-byte file (checked 2026-11-06).
- **Thin scans** (bowl, plate, wrench, clamps) are thinner than the real parts. The mass is still right, but the
  inertia is approximate. Their hulls are thicker than the scan: `hull_volume_ratio` in `manifest.json` shows by how
  much.
- **Sets scanned as one object** carry ESTIMATE masses: the YCB marble bag, the cube jar, the peg board and the Duplo
  pieces.
- **Collision** needs the engine's rigid emitter to read `physics.hulls.obj`. Until that lands, these cards validate and
  pass the importer's drop test, but `alloy.scene` cannot emit them yet.
