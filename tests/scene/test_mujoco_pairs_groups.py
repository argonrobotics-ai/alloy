# SPDX-License-Identifier: Apache-2.0
"""``alloy.scene.mujoco_pairs``: an explicit MuJoCo pair row bypasses contype/conaffinity, so ``_excluded`` must follow
Newton's collision-group rule exactly (``groups_collide`` = ``broad_phase_common.test_group_pair``), including two
shapes in one negative group, which never collide (WP9's finding, 2026-11-09)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alloy.scene.mujoco_pairs import _excluded, groups_collide

CASES = [(0, 0), (0, 1), (1, 1), (1, 2), (1, -1), (-1, 1), (-1, -1), (-1, -2), (-3, -3), (2, -5), (0, -1)]


@pytest.mark.parametrize("g0, g1", CASES)
def test_groups_collide_is_newtons_rule(g0: int, g1: int) -> None:
    from alloy._vendor.newton._src.geometry import broad_phase_common as bpc  # noqa: PLC0415

    rule = bpc.test_group_pair.func if hasattr(bpc.test_group_pair, "func") else None
    want = rule(g0, g1) if rule is not None else _newton(g0, g1)
    assert groups_collide(g0, g1) is bool(want) and groups_collide(g1, g0) is bool(want)


def _newton(a: int, b: int) -> bool:   # broad_phase_common.test_group_pair, line for line
    if a == 0 or b == 0:
        return False
    if a > 0:
        return a == b or b < 0
    return a != b


def test_two_shapes_in_one_negative_group_get_no_row() -> None:
    objs = [SimpleNamespace(obj=object()), SimpleNamespace(obj=object())]
    b = SimpleNamespace(_shape_collision_filter_pairs=set(), shape_collision_group=[-4, -4, -4, 1])
    assert _excluded(b, (0, None, 0), (1, None, 1), objs)          # one negative group: never collide
    b.shape_collision_group = [-4, -5]
    assert not _excluded(b, (0, None, 0), (1, None, 1), objs)      # two negative groups collide
    b.shape_collision_group = [1, 1]
    assert not _excluded(b, (0, None, 0), (1, None, 1), objs)      # the default group
