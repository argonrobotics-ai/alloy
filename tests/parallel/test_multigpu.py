# SPDX-License-Identifier: Apache-2.0
"""Mode A launcher (SPEC §3.9 launch_mode_a, parallel §4.6), exercised with CPU child processes."""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from alloy.core.errors import LaunchError
from alloy.parallel import multigpu as mg


def test_world_blocks_cover_ids_in_order():
    blocks = [mg.world_block(r, 3, 10) for r in range(3)]
    assert [(b.offset, b.count) for b in blocks] == [(0, 4), (4, 3), (7, 3)]
    with pytest.raises(ValueError):
        mg.world_block(3, 3, 10)


def test_launch_runs_one_process_per_device(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import json, os, sys
        keys = {[mg.ENV_DEVICE, mg.ENV_OFFSET, mg.ENV_COUNT, mg.ENV_RANK, mg.ENV_SIZE]!r}
        out = {{k: os.environ[k] for k in keys}}
        open(os.path.join(sys.argv[1], "rank" + out["{mg.ENV_RANK}"] + ".json"), "w").write(json.dumps(out))
    """))
    mg.launch_mode_a(str(script), ["cpu", "cpu"], 9, args=[str(tmp_path)], timeout_s=60)
    got = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(2)]
    assert [(g[mg.ENV_OFFSET], g[mg.ENV_COUNT]) for g in got] == [("0", "5"), ("5", "4")]
    assert all(g[mg.ENV_DEVICE] == "cpu" and g[mg.ENV_SIZE] == "2" for g in got)


def test_failed_rank_is_reported(tmp_path):
    script = tmp_path / "bad.py"
    script.write_text("import os, sys\nif os.environ['ALLOY_RANK'] == '1':\n    sys.exit('boom on rank 1')\n")
    with pytest.raises(LaunchError, match="rank 1: exit 1.*boom"):
        mg.launch_mode_a(str(script), ["cpu", "cpu"], 4, timeout_s=60)


def test_launch_validates_arguments():
    with pytest.raises(ValueError):
        mg.launch_mode_a(sys.executable, [], 4)
    with pytest.raises(ValueError):
        mg.launch_mode_a(sys.executable, ["cpu", "cpu"], 1)
