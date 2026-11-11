# SPDX-License-Identifier: Apache-2.0
"""Granular speed ladder past the configs' max_nworld caps (runner helper; the harness itself is unchanged).

    python -m bench.scenarios._granular_ladder --family granular --device cuda:0 --ladder 1,4,16,256,1024,4096 ...

Same CLI as `python -m bench.run`. The granular configs cap their ladders (max_nworld) by guessed memory and CPU time;
the GPU ladder instead stops where memory does (SCENARIOS.md: larger rungs after an error are recorded as `skipped`).
This wrapper drops the cap for the run it starts, and every row above the config's cap says so in `notes` and in
`command`. Importing the module changes nothing; only running it as __main__ patches the harness, in this process.
"""
from __future__ import annotations

import sys

from bench import harness as H
from bench import run as R

TAG = "python -m bench.scenarios._granular_ladder (max_nworld caps lifted)"


def _ladder_nocap(scn: H.Scenario, cfg: H.Config, opts: H.RunOptions) -> list[int]:
    return list(opts.ladder or scn.ladder)


def _patch() -> None:
    base = H._base_record

    def base_record(scn: H.Scenario, cfg: H.Config, nworld: int, opts: H.RunOptions) -> dict:
        rec = base(scn, cfg, nworld, opts)
        rec["command"] = rec["command"].replace("python -m bench.run", TAG, 1)
        if cfg.max_nworld is not None and nworld > cfg.max_nworld:
            rec["notes"] = f"above config max_nworld={cfg.max_nworld} (cap lifted by _granular_ladder); "
        return rec

    H._ladder_for = _ladder_nocap
    H._base_record = base_record


if __name__ == "__main__":
    _patch()
    sys.exit(R.main())
