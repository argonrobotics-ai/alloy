# SPDX-License-Identifier: Apache-2.0
"""lint_atomics (SPEC §7 WP5 acceptance): flags seeded float atomics and a 2-term sum into a non-zeroed target,
accepts integer/max/min/thread-owned updates and marked lines, and passes the repo."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from alloy.tools.lint_atomics import lint, main

HEADER = "import warp as wp\n\n"


def _lint_src(tmp_path: Path, src: str, name: str = "mod.py") -> list[tuple[int, str]]:
    path = tmp_path / name
    path.write_text(HEADER + textwrap.dedent(src))
    findings, errors = lint([tmp_path])
    assert not errors
    return [(f.line - HEADER.count("\n"), f.rule) for f in findings]


def test_seeded_float_atomic_flagged(tmp_path: Path) -> None:
    rules = _lint_src(tmp_path, """
        @wp.kernel
        def k(x: wp.array(dtype=wp.float32), w: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            wp.atomic_add(out, w[i], x[i])
    """)
    assert [r for _, r in rules] == ["DET001"]


@pytest.mark.parametrize("annotation", ["wp.array[wp.vec3]", "wp.array2d(dtype=float)", "'wp.array(dtype=wp.float64)'"])
def test_float_annotation_forms(tmp_path: Path, annotation: str) -> None:
    rules = _lint_src(tmp_path, f"""
        @wp.kernel
        def k(out: {annotation}, idx: wp.array(dtype=wp.int32)):
            i = wp.tid()
            wp.atomic_sub(out, idx[i], out[i])
    """)
    assert [r for _, r in rules] == ["DET001"]


def test_augassign_on_array_is_an_atomic(tmp_path: Path) -> None:
    rules = _lint_src(tmp_path, """
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32), idx: wp.array(dtype=wp.int32)):
            i = wp.tid()
            alias = out
            alias[idx[i]] += 1.0
            v = wp.vec3(0.0)
            v[0] += 1.0
            out[i] += 2.0
    """)
    assert rules == [(6, "DET001")]  # alias flagged; local vector and thread-owned slot are not


def test_integer_max_min_and_thread_owned_are_allowed(tmp_path: Path) -> None:
    rules = _lint_src(tmp_path, """
        IntArr = wp.array(dtype=wp.int64)

        @wp.struct
        class Acc:
            exp: wp.array(dtype=wp.int32)
            acc: IntArr

        @wp.kernel
        def k(a: Acc, f: wp.array(dtype=float), n: wp.array(dtype=wp.vec3i), w: wp.array(dtype=wp.int32)):
            i, j = wp.tid()
            wp.atomic_add(a.acc, w[i], wp.int64(1))
            wp.atomic_max(a.exp, w[i], 3)
            wp.atomic_min(f, w[i], 1.0)
            wp.atomic_add(n, w[i], wp.vec3i(1, 1, 1))
            wp.atomic_add(f, i, 1.0)
            n[i] += wp.vec3i(1, 0, 0)
    """)
    # atomic_add(f, i, ...) with a 2-d tid is not thread-owned (all j threads share slot i)
    assert rules == [(16, "DET001")]


def test_unknown_dtype_and_order_dependent(tmp_path: Path) -> None:
    rules = _lint_src(tmp_path, """
        from typing import Any

        @wp.func
        def f(out: wp.array(dtype=Any), i: int, x: float):
            wp.atomic_add(out, i, x)

        @wp.kernel
        def k(out: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32)):
            i = wp.tid()
            wp.atomic_exch(out, idx[i], i)
    """)
    assert [r for _, r in rules] == ["DET002", "DET003"]


def test_det_ok_needs_a_reason(tmp_path: Path) -> None:
    rules = _lint_src(tmp_path, """
        @wp.kernel
        def k(out: wp.array(dtype=float), idx: wp.array(dtype=int)):
            i = wp.tid()
            wp.atomic_add(out, idx[i], 1.0)  # alloy: det-ok one writer per slot by construction (idx is a permutation)
            wp.atomic_add(out, idx[i], 2.0)  # alloy: det-ok
    """)
    assert rules == [(6, "DET005")]


TWO_TERM = """
    @wp.kernel
    def pair_sum(out: wp.array(dtype=float), a: wp.array(dtype=int), b: wp.array(dtype=int),
                 f: wp.array(dtype=float)):
        i = wp.tid()
        wp.atomic_add(out, a[i], f[i])  # alloy: det-2term contact i adds into its two bodies only
        wp.atomic_add(out, b[i], -f[i])  # alloy: det-2term

    @wp.kernel
    def clear(x: wp.array(dtype=float)):
        x[wp.tid()] = 0.0

    def host(out, a, b, f, n):
        {zero}
        wp.launch(pair_sum, dim=n, inputs=[out, a, b, f])
"""


@pytest.mark.parametrize(
    "zero, ok",
    [
        ("out.zero_()", True),
        ("out.fill_(0.0)", True),
        ("out = wp.zeros(n, dtype=float)", True),
        ("wp.launch(clear, dim=n, inputs=[out])", True),
        ("out.fill_(1.0)", False),
        ("pass", False),
    ],
)
def test_two_term_requires_zero_fill(tmp_path: Path, zero: str, ok: bool) -> None:
    rules = _lint_src(tmp_path, TWO_TERM.replace("{zero}", zero))
    assert (rules == []) is ok, rules
    if not ok:
        assert {r for _, r in rules} == {"DET004"}


def test_two_term_without_launch_is_unverifiable(tmp_path: Path) -> None:
    src = TWO_TERM.split("    def host")[0]
    rules = _lint_src(tmp_path, src)
    assert {r for _, r in rules} == {"DET004"}


def test_native_snippet_atomic(tmp_path: Path) -> None:
    rules = _lint_src(tmp_path, '''
        SNIP = "atomicAdd(&out[i], x);"

        @wp.func_native(SNIP)
        def add_native(out: wp.array(dtype=float), i: int, x: float): ...
    ''')
    assert [r for _, r in rules] == ["DET006"]


def test_repo_is_clean() -> None:
    root = Path(__file__).resolve().parents[2]
    findings, errors = lint([root / "alloy"])
    assert not errors
    assert findings == [], "\n".join(f.format() for f in findings)


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "bad.py").write_text(HEADER + "@wp.kernel\ndef k(o: wp.array(dtype=float), j: wp.array(dtype=int)):\n"
                                     "    wp.atomic_add(o, j[0], 1.0)\n")
    assert main([str(tmp_path)]) == 1
    assert "DET001" in capsys.readouterr().out
    (tmp_path / "bad.py").write_text(HEADER)
    assert main([str(tmp_path)]) == 0
    (tmp_path / "broken.py").write_text("def (:\n")
    assert main([str(tmp_path)]) == 2


def test_same_struct_name_in_two_modules(tmp_path: Path) -> None:
    pkg = tmp_path / "alloy" / "a"
    pkg.mkdir(parents=True)
    (tmp_path / "alloy" / "b.py").write_text(HEADER + textwrap.dedent("""
        @wp.struct
        class Accum:
            total: wp.array(dtype=wp.float32)
    """))
    (pkg / "m.py").write_text(HEADER + textwrap.dedent("""
        @wp.struct
        class Accum:
            count: wp.array(dtype=wp.int32)

        @wp.kernel
        def k(acc: Accum, idx: wp.array(dtype=wp.int32)):
            i = wp.tid()
            wp.atomic_add(acc.count, idx[i], 1)
    """))
    (pkg / "n.py").write_text(HEADER + textwrap.dedent("""
        from alloy.b import Accum

        @wp.kernel
        def k2(acc: Accum, idx: wp.array(dtype=wp.int32)):
            i = wp.tid()
            wp.atomic_add(acc.total, idx[i], 1.0)
    """))
    findings, _ = lint([tmp_path])
    assert [(Path(f.path).name, f.rule) for f in findings] == [("n.py", "DET001")]


def test_review_false_negatives(tmp_path: Path) -> None:
    """Unresolved array parameters, a rebound tid and a zero-fill outside the launch loop are all flagged."""
    rules = _lint_src(tmp_path, """
        from typing import Any

        @wp.func
        def add_any(a: Any, i: int, x: float):
            a[i] += x

        @wp.kernel
        def scatter(dest: wp.array(dtype=wp.int32), out: wp.array(dtype=float)):
            i = wp.tid()
            i = dest[i]
            wp.atomic_add(out, i, 1.0)

        @wp.kernel
        def pair_sum(out: wp.array(dtype=float), a: wp.array(dtype=int)):
            i = wp.tid()
            wp.atomic_add(out, a[i], 1.0)  # alloy: det-2term two contacts per body at most

        def host(out, a, n):
            out.zero_()
            for _ in range(10):
                wp.launch(pair_sum, dim=n, inputs=[out, a])
    """)
    assert [r for _, r in rules] == ["DET002", "DET001", "DET004"], rules


def test_zero_fill_inside_the_launch_loop_passes(tmp_path: Path) -> None:
    rules = _lint_src(tmp_path, """
        @wp.kernel
        def pair_sum(out: wp.array(dtype=float), a: wp.array(dtype=int)):
            i = wp.tid()
            wp.atomic_add(out, a[i], 1.0)  # alloy: det-2term two contacts per body at most

        def host(out, a, n):
            for _ in range(10):
                out.zero_()
                wp.launch(pair_sum, dim=n, inputs=[out, a])
    """)
    assert rules == []
