# SPDX-License-Identifier: Apache-2.0
"""Live view and steering for distributed Alloy training (a test layer; see docs/notes/live-test.md).

Workers publish from background threads through bounded queues and never block a training step:
per-world numbers for every world (1 Hz), and for picked worlds the 3D state (12 Hz) and the policy camera as
H.264 from NVENC (15 fps). Steering is a list of typed commands applied at step boundaries and logged for replay.

    from alloy.live import LiveConfig, LiveWorker
    live = LiveWorker(env, LiveConfig(hub="<hub-host>:47100", worker="vm1", run="cable-lift"), settings)
"""
from alloy.live.commands import OPS, CommandError, make_command
from alloy.live.settings import Settings
from alloy.live.worker import LiveConfig, LiveWorker

__all__ = ["OPS", "CommandError", "LiveConfig", "LiveWorker", "Settings", "make_command"]
