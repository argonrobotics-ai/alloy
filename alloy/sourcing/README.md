# alloy.sourcing

Fills missing or ESTIMATE values (mass, density, friction, restitution, elastic moduli) on imported object cards
with cited numbers, and reports how many are sourced versus estimated.

## Rules for a value

Each value is a number in SI, a range `[lo, hi]`, a tag and its evidence:

| tag | meaning | needs |
|---|---|---|
| `DOC` | the page or PDF was fetched and the quote was read in it | URL + exact quote |
| `SNIPPET` | quote seen only in a search-result summary | URL + quote; re-check before a gate uses it |
| `UNVERIFIED` | a cited reference (handbook, older table) not re-read | the citation |
| `ESTIMATE` | a prior: category range, proxy material, back-calculation | `lo < hi` and the reason |

Never a bare guess. A value's tag is never stronger than its best evidence. A category value applied to one object
is an ESTIMATE for that object (the category's evidence stays attached).

## Files (in the alloy-assets checkout)

- `imports/<library>/`: importer cards (read only here). Optional hints, one JSON object per line, in
  `imports/<library>/sourcing_queue.jsonl`:
  `{"object": "<card name>", "category": "ceramic_mug", "product": "<maker model>", "url": "<product page>", "fields": ["mass"]}`.
  Add `"fields": ["mass"]` when an object's density came from a generic material, so its mass gets sourced.
- `sources/imports/cache/categories/<key>.yaml`, `sources/imports/cache/materials/<key>.yaml`: per-category and
  per-material values, reused for every matching object.
- `sources/imports/cache/maps/<dataset>.json`: `{source object name: category key}` for datasets whose names do
  not say what the object is (e.g. Google Scanned Objects); an importer prefix on the card name still matches.
  A target `material:<key>` marks a single-material object: it gets that material's values and no mass prior.
- `sources/imports/<library>/<object>.yaml`: the per-object record. `fill` rewrites only its own category priors
  (reason "category ..."), so re-running it is harmless and never touches a value found for that object (`ingest`).

Fill rules worth knowing: a category mass is checked against the scan volume (mass / volume at most 1.5 x the
category material's density, 2.5 x more for open, thin-walled categories); a hand-off row without a volume gets one
back from an importer note "density x visual-mesh volume" (solid parts of one density); a category friction value
that would leave mu_dynamic above the card's mu_static (or the reverse) is left out.
- `sources/imports/coverage.json`, `COVERAGE.md`: the coverage report.

## Use

```bash
python -m alloy.sourcing queue --library ycb_gso --limit 20     # what still needs a value (JSON)
python -m alloy.sourcing lookup "coffee mug"                     # cached category values
python -m alloy.sourcing fill --library ycb_gso                  # write records from the cache
python -m alloy.sourcing coverage --write                        # shares sourced / weak / estimated / missing
python -m alloy.sourcing validate path/to/entry.yaml             # check a file before committing it
python -m alloy.sourcing verify                                  # re-fetch DOC pages, check each quote is on its page
```

An importer applies a record to its material card:

```python
from alloy.cards.library import CardLibrary
from alloy.sourcing import apply_to_material, load_records

rec = load_records("ycb_gso").get("gso_porcelain_mug")         # None until sourced
mat = CardLibrary().material("stoneware_glazed")
if rec is not None:
    mat = apply_to_material(mat, rec, volume=1.6e-4)             # mass -> density through the mesh volume (m^3)
```
