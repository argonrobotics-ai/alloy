# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON and digests (SPEC §3.1): sorted keys, 17 significant digits, stable across processes."""

from __future__ import annotations

import enum
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from alloy.core.errors import DigestError
from alloy.core.ids import canonical_json, digest_of, sha256_hex
from alloy.core.quantity import Q

REPO = Path(__file__).resolve().parents[2]


class Color(enum.Enum):
    RED = "red"


@dataclass(frozen=True)
class Card:
    name: str
    mu: Q
    tags: tuple[str, ...] = ()


SAMPLE = {"b": 0.1, "a": [1, 2.5, None, True, False], "q": Q(0.3, "catalog", note="x"), "e": Color.RED,
          "s": frozenset({"z", "y"}), "t": (1, (2, 3)), "neg0": -0.0, "big": 1e300, "nested": {"k": {2: "v"}}}


def test_sorted_keys_and_float_format() -> None:
    text = canonical_json({"b": 0.1, "a": 1}).decode()
    assert text == '{"a":1,"b":0.10000000000000001}'
    assert canonical_json(1.0) == b"1"
    assert canonical_json(1 / 3) == b"0.33333333333333331"


def test_round_trip_floats_exact() -> None:
    values = [0.1, 1 / 3, 2.1e9, 5e-5, -1.25e-3, 1e-300, 123456789.123456789]
    for v in values:
        assert float(canonical_json(v)) == v


def test_non_finite_floats_are_strings() -> None:
    assert json.loads(canonical_json([float("nan"), float("inf"), -float("inf")])) == ["NaN", "Infinity", "-Infinity"]


def test_dataclass_uses_fields_and_enum_values() -> None:
    card = Card("pp", Q(0.3, "catalog", note="x"))
    data = json.loads(canonical_json(card))
    assert data == {"mu": {"note": "x", "source": "catalog", "unc": None, "value": 0.3}, "name": "pp", "tags": []}
    assert json.loads(canonical_json(Color.RED)) == "red"


def test_sets_sorted_and_tuple_equals_list() -> None:
    assert canonical_json({"z", "a"}) == canonical_json(["a", "z"])
    assert canonical_json((1, 2)) == canonical_json([1, 2])


def test_arrays_hash_by_dtype_shape_bytes() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert digest_of(a) == digest_of(a.copy())
    assert digest_of(a) != digest_of(a.astype(np.float64))
    assert digest_of(a) != digest_of(a.reshape(3, 2))


def test_unsupported_type_raises() -> None:
    with pytest.raises(DigestError):
        canonical_json(object())
    with pytest.raises(TypeError):
        canonical_json(lambda: 0)


def test_digest_is_sha256_of_canonical_json() -> None:
    assert digest_of(SAMPLE) == sha256_hex(canonical_json(SAMPLE))


def test_digest_stable_across_processes() -> None:
    code = ("from alloy.core.quantity import Q; import enum, sys\n"
            "sys.path.insert(0, 'tests/core')\n"
            "from test_ids import SAMPLE\n"
            "from alloy.core.ids import digest_of; print(digest_of(SAMPLE))")
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == digest_of(SAMPLE)


def test_omit_if_none_keeps_old_digests() -> None:
    """A field marked OMIT_IF_NONE is not digested while None (the dataclass digests as if it had no such field), and
    is digested once set; an unmarked None field still counts."""
    from dataclasses import field

    from alloy.core.ids import OMIT_IF_NONE

    @dataclass
    class Old:
        a: int = 1

    @dataclass
    class New:
        a: int = 1
        film: dict | None = field(default=None, metadata=OMIT_IF_NONE)

    @dataclass
    class Unmarked:
        a: int = 1
        film: dict | None = None

    assert digest_of(New()) == digest_of(Old())
    assert digest_of(New(film={"h": 1e-6})) != digest_of(Old())
    assert digest_of(Unmarked()) != digest_of(Old())
