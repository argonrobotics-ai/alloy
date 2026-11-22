# SPDX-License-Identifier: Apache-2.0
"""Regression test: LeRobot observation.state columns must apply actuator-to-joint aliases."""

from __future__ import annotations

from alloy.sysid.readers import _named_columns


def test_sysid_named_columns_applies_aliases_to_state_names():
    """LeRobot observation.state actuator names must remap through aliases just like action names."""
    cols = _named_columns(
        ["shoulder", "elbow"],
        ["act_elbow.pos", "act_elbow.vel", "act_shoulder.pos"],
        3,
        aliases={"act_shoulder": "shoulder", "act_elbow": "elbow"},
    )
    assert cols == [2, 0]
