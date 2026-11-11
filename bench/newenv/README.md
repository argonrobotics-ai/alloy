# bench/newenv: the new-environment test

A cold agent gets a one-sentence environment request (`requests.json`: 24 requests in five groups: rigid,
articulated, soft, liquid, robot tasks) and must build it, from an engine's public docs only, as a running, checked,
vectorized environment behind one contract (`preamble_contract.md`). Every environment is run through the same
driver (`drive.py`: rest, the request's own action phases, random actions, the scripted policy for robot tasks, seed
checks, timing) and scored by the request's checker on the recorded probes. We report the share of requests at
level 2 ("checked") or better with a 95 % Wilson interval, wall time and tokens, per engine and per group.

Verdict levels: L0 no environment (crash, wrong shapes, non-finite values); L1 runs (all phases, resets, per-world
variation, seeds); L2 checked (the request's physics checks hold); L3 solved (robot tasks: the scripted policy succeeds
in at least 80 % of worlds); `declined` when the agent says the engine cannot do it and the engine's docs agree
(`declines` in `engines.json`), else that decline is L0.

## Check one environment yourself

    python3 -m bench.newenv task E01 mytask          # drive.py, request.json, script.py and REQUEST.md (the request)
    # write mytask/env.py (make_env(num_envs, seed) per preamble_contract.md), then in mytask:
    python3 drive.py --quick                         # self-test; `python3 drive.py` writes out/ for scoring
    python3 -m bench.newenv check E01 mytask/out     # the verdict, failed checks and metrics as JSON

`reference/alloy/<id>/env.py` is a reference environment in Alloy for each request, written with full context; it is
how each checker was validated before any cold agent ran. Checkers (`checkers/<name>.py`) are hidden from agents;
`scripts/<name>.py` holds the harness's own action laws for a request and ships with the task.

## Score and summarize runs

    python3 -m bench.newenv score RUNS_DIR --out rows.jsonl [--public]
    python3 -m bench.newenv summarize rows.jsonl [--public] [--out summary.md]

A run directory holds `task/out` (what `drive.py` wrote in the scored re-run), `meta.json`, and, from a cold-agent
run, `transcript.jsonl` (turns, tokens and cost are read from it with the cold-agent kit when that is installed).
An engine entry may carry a `notice`: that arm's results may not be published, every output with its rows starts with
the notice, and `--public` leaves them out. `$NEWENV_EXTRA_CONFIG` names a directory whose `engines.json` /
`allowlist.json` are laid over the ones here (arms or fields kept outside this package).

## Cold-agent rounds (needs the cold-agent kit in bench/agents and a cloud project)

    python3 -m bench.newenv plan --engines alloy,genesis --reps 2      # list runs, flag engines with TODO fields
    python3 -m bench.newenv prompt alloy E01                           # the TASK.md an agent would get
    python3 -m bench.newenv launch --token-file F --engines alloy,genesis --reps 2 --snapshot S --work W [--plan-only]

`harness.py` builds an overlay of the kit's harness directory (engines, allowlist, preamble, requests and the prompt
builder replaced; the kit's `run_one.sh` and `vm/prepare.sh` patched from their text with exact-match asserts) and
`launch.py` runs the round on the kit's VM, relay and polling pieces. Each arm runs in a sandbox image with its engine
pre-installed the documented way (`images/<arm>/Dockerfile`, `images/build.sh`; install time is measured once per
image and reported beside the row, never inside an agent's time). Time caps per engine come from `engines.json`
(default 45 min agent, 15 min re-run). Engines whose entry still holds `TODO`, and requests whose script or checker
is missing, refuse to launch.
