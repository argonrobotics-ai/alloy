# SPDX-License-Identifier: Apache-2.0
"""Warp's no-driver line (0.1.6 from PyPI, a CPU machine with no NVIDIA driver; BATCH-0117 item 7a).

Warp's native start-up wrote "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver. GPU execution will not
be available." to stderr once per Python process, cold and warm, at the first ``alloy.build``. Laws (CPU):

- In a fresh process on a machine with no driver (skipped where one loads), ``alloy.build`` and a build stepped in
  worker processes write no such line; ``ALLOY_VERBOSE=1`` keeps Warp's own output, the line included.
- ``driver_present`` looks for the library Warp loads (none to look for on macOS); ``_stderr_muted`` mutes writes to
  file descriptor 2 (a C ``fprintf``'s too) for its block only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from alloy.api import _warp_start as WS

REPO = Path(__file__).resolve().parents[2]
LINE = "Could not find or load the NVIDIA CUDA driver"
SCRIPT = """
import alloy
s = alloy.Scene(dt=0.01)
s.add_ground()
s.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.2), mass=0.2)
sim = alloy.build(s)
sim.step(2)
print("device", sim.device)
sim.close()
if {workers}:
    with alloy.build(s, worlds=2, device="cpu", processes=2) as sim:
        sim.step(2)
    print("workers ok")
"""


def _run(verbose: bool, workers: bool = False) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ALLOY_VERBOSE"}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    if verbose:
        env["ALLOY_VERBOSE"] = "1"
    return subprocess.run([sys.executable, "-c", SCRIPT.format(workers=workers)], env=env, capture_output=True,
                          text=True, timeout=900, cwd=str(REPO))


@pytest.mark.skipif(WS.driver_present(), reason="an NVIDIA driver loads here: Warp has no no-driver line to print")
def test_a_build_on_a_machine_with_no_driver_prints_no_cuda_driver_line():
    quiet = _run(verbose=False, workers=True)
    assert quiet.returncode == 0, quiet.stderr[-2000:]
    assert "device cpu" in quiet.stdout and "workers ok" in quiet.stdout
    assert LINE not in quiet.stderr, [ln for ln in quiet.stderr.splitlines() if LINE in ln]
    loud = _run(verbose=True)
    assert loud.returncode == 0, loud.stderr[-2000:]
    assert sum(LINE in ln for ln in loud.stderr.splitlines()) == 1


def test_driver_present_looks_for_the_library_warp_loads(monkeypatch):
    assert WS.driver_present("darwin")                                    # no CUDA for Warp to look for
    monkeypatch.setitem(WS._LIBRARIES, "testos", ("libno-such-driver-here.so",))
    assert not WS.driver_present("testos")


def test_stderr_muted_mutes_descriptor_2_for_its_block_only(capfd):
    with WS._stderr_muted():
        os.write(2, b"inside the block\n")
    os.write(2, b"after the block\n")
    err = capfd.readouterr().err
    assert "inside the block" not in err and "after the block" in err
