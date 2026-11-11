# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU mode A: data-parallel packs, one process per device (SPEC §3.9, parallel §4.6).

``launch_mode_a`` splits ``world_count`` global worlds into consecutive blocks, one per device, and runs ``script``
once per device in its own Python process (the Isaac Lab / MJW pattern). World ids, seeds and RNG keys are global,
so a world's trajectory does not depend on which device holds it. The child reads its share from the environment:

    ALLOY_DEVICE            device string, e.g. "cuda:0" (CUDA_VISIBLE_DEVICES is left alone, so ordinals stay global)
    ALLOY_WORLD_ID_OFFSET   global id of the child's first world
    ALLOY_WORLD_COUNT       worlds in the child's block
    ALLOY_RANK, ALLOY_WORLD_SIZE   process index and process count

``world_block(rank, size, world_count)`` gives the same split inside the child. Mode B (solver-entry split) and
mode C (spatial decomposition) are not in v1.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from alloy.core.errors import LaunchError

log = logging.getLogger("alloy.parallel")

ENV_DEVICE = "ALLOY_DEVICE"
ENV_OFFSET = "ALLOY_WORLD_ID_OFFSET"
ENV_COUNT = "ALLOY_WORLD_COUNT"
ENV_RANK = "ALLOY_RANK"
ENV_SIZE = "ALLOY_WORLD_SIZE"
STDERR_TAIL = 2000




@dataclass(frozen=True)
class WorldBlock:
    offset: int
    count: int


def world_block(rank: int, size: int, world_count: int) -> WorldBlock:
    """Consecutive block of global world ids for process ``rank`` of ``size`` (sizes differ by at most one)."""
    if not 0 <= rank < size:
        raise ValueError(f"rank {rank} outside [0, {size})")
    base, extra = divmod(world_count, size)
    offset = rank * base + min(rank, extra)
    return WorldBlock(offset=offset, count=base + (1 if rank < extra else 0))


def child_env(rank: int, devices: Sequence[str], world_count: int,
              base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the child process of ``rank``."""
    block = world_block(rank, len(devices), world_count)
    env = dict(os.environ if base is None else base)
    env.update({ENV_DEVICE: devices[rank], ENV_OFFSET: str(block.offset), ENV_COUNT: str(block.count),
                ENV_RANK: str(rank), ENV_SIZE: str(len(devices))})
    return env


def launch_mode_a(script: str, devices: Sequence[str], world_count: int, *, args: Sequence[str] = (),
                  timeout_s: float | None = None) -> None:
    """Run ``script`` once per device concurrently and wait; raises LaunchError naming every failed rank."""
    if not devices:
        raise ValueError("launch_mode_a needs at least one device")
    if world_count < len(devices):
        raise ValueError(f"{world_count} worlds cannot cover {len(devices)} devices")
    procs = []
    for rank, dev in enumerate(devices):
        env = child_env(rank, devices, world_count)
        log.info("mode A rank %d on %s: worlds %s..+%s", rank, dev, env[ENV_OFFSET], env[ENV_COUNT])
        procs.append(subprocess.Popen([sys.executable, script, *args], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True))
    failures = [_wait(rank, p, timeout_s) for rank, p in enumerate(procs)]
    failures = [f for f in failures if f]
    if failures:
        raise LaunchError("mode A failed:\n" + "\n".join(failures))


def _wait(rank: int, proc: subprocess.Popen, timeout_s: float | None) -> str:
    try:
        _, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, err = proc.communicate()
        return f"rank {rank}: timeout after {timeout_s} s; stderr tail: {err[-STDERR_TAIL:]}"
    if proc.returncode != 0:
        return f"rank {rank}: exit {proc.returncode}; stderr tail: {err[-STDERR_TAIL:]}"
    return ""
