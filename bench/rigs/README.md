# Rig datasets (`alloy.bench.rigs`, SPEC §3.10, bench §4.4)

One directory per dataset: `bench/rigs/<law id>/<dataset id>/`. Datasets are **immutable**: `load_rig` recomputes
every sha256 in `MANIFEST.sha256` and refuses a dataset with an edited, missing or unlisted trial. A new recording
is a new dataset id.

| File | Content |
|---|---|
| `rig_card.json` | `RigCard`: law id, dataset id, protocol version (`synthetic/...` for generated data), objects (measured dims/masses, SI, with `*_u95`), instruments (per metric: `u95`, range, resolution, calibration date), holder (k_x, k_theta, stage profile), environment |
| `conditions.json` | `[{"id", "params"}]` (SI) |
| `trials.json` | `[{"file", "condition_id", "index", "split"}]`; split is `calib` (fitting) or `valid` (judging) |
| `trials/<cond>__<k>.npz` | one float64 array per series channel (`t` first, SI) and `scalar__<metric>` 0-d arrays |
| `MANIFEST.sha256` | `<sha256>  <path>` per file; the dataset sha256 is the hash of this file |
| `README.md` (real data) | the protocol steps as they were run |

Rules (bench §4.3): material values are fitted on `calib` trials only (`run_law` hands twins only the calib split
through `bind_calib`); `valid` trials judge. Instrument `u95` per metric feeds U_real = sqrt(u_instr^2 +
(t_0.975 s / sqrt n)^2); TUR = bound / U_real < 3 gives `inconclusive`.

Current datasets (synthetic, generated from the analytic law by `alloy.testing.bench`, `synthetic_rig`):

| Dataset | Law | Conditions | Trials |
|---|---|---|---|
| `R3.incline_friction/synthetic-v1` | slip at atan(mu), mu_k = mu, no creep; noise 0.05 deg, 0.005, 1e-7 m | PP on steel (mu 0.35), rubber on wood (0.7) | 20 per condition, 10 calib |
| `R4.drop_bounce/synthetic-v1` | e = 0.7, h1 = e^2 h0; noise 0.005 | drops 5, 10, 15 cm | 20 per condition, 10 calib |

The scoreboard keeps the real R3/R4 rows (SB-2) `not_run`/deferred while only synthetic data exist.
Regenerate (only into a new dataset id): `PYTHONPATH=. python -c "from alloy.testing.bench import
write_synthetic_rigs; write_synthetic_rigs()"`.
