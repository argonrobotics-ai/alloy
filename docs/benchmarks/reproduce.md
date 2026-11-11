# Reproduce the scorecard

DRAFT: held until the scorecard is released. Costs and wall times marked "to be measured" are filled from the
cold re-run of each row, not estimated.

Every row of the scorecard is one command. Each writes `rows.jsonl` (one record per number, with the command,
commit, device and date that produced it) and `headline.json` (the line the scorecard shows) under
`scorecard/row<id>/`.

```bash
python -m alloy.bench.scorecard list                 # the rows, which are ready, the machine each needs
python -m alloy.bench.scorecard 3 --device cpu --jobs 14   # row 3: the analytic laws against their exact answers, one law per process (a stopped run resumes)
python -m alloy.bench.scorecard 6                    # row 6: counts the coverage files in bench/scorecard/row6/ (no simulation)
python -m alloy.bench.scorecard page --root scorecard --out scorecard/SCORECARD.md
python -m alloy.bench.scorecard charts --root scorecard --out scorecard/charts   # SVG charts; needs matplotlib
```

Row 6 reads its data files from `bench/scorecard/row6/` under the folder you run in (`--data <folder>` reads another
folder). A pip install does not carry those files, so row 6 cannot be re-run from a pip install yet. `charts` needs
`pip install matplotlib`; without it the command stops with a message that names this.

`--quick` gives a short draft run; its headline is marked draft and is never the published number.

## Machines

| Rows | Machine | Image | Wall time | Cost |
|---|---|---|---|---|
| 1, 2, 4, 5 | Google Cloud `g4-standard-48`, one RTX PRO 6000 Blackwell, spot | `common-cu129-ubuntu-2204-nvidia-580` (NVIDIA driver 580) | to be measured | to be measured |
| 3, 6, 7 | Google Cloud `n2-standard-16`, spot (row 3 also runs on a GPU: `--device cuda:0`) | `ubuntu-2204-lts` | to be measured | to be measured |

```bash
gcloud compute instances create scorecard-gpu --zone <zone> --machine-type g4-standard-48 \
  --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
  --boot-disk-size 150GB --boot-disk-type hyperdisk-balanced --maintenance-policy TERMINATE \
  --provisioning-model SPOT --instance-termination-action STOP --max-run-duration 8h \
  --metadata install-nvidia-driver=True
# on the machine
python3 -m venv .venv && source .venv/bin/activate && pip install "alloy-engine[envs]"
nvidia-smi                                           # driver 580 or newer
python -m alloy.bench.scorecard <row> --device cuda:0
# when done
gcloud compute instances delete scorecard-gpu --zone <zone>
```

The first run of a row compiles kernels (minutes); rows that time anything warm up first and time only warm runs.

## What a record holds

`row, engine, task, metric, value, unit, lo, hi, bound_kind, k, n, verdict, caveat, device, command, commit, date,
source`. A rate carries a Wilson 95 % interval over independent trials: worlds that are copies of one start count
as one trial. A law's bound is its tolerance. A number that did not run is `not_run` with the reason and never
counts as passed.

## Other engines

Row 6 reads other engines' capabilities from their public code and documents; it does not run them (see
`bench/scorecard/row6/README.md`). Numbers measured on engines whose licence does not allow publishing benchmarks are not in this
repository.
