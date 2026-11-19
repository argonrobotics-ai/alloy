# SPDX-License-Identifier: Apache-2.0
"""Law: the camera's kernels keep their cache names from one process to the next.

Warp names a ``module="unique"`` kernel (the full camera's megakernel, one per material set) by hashing every function
it calls. A function whose local or argument is named like a value outside it (``e``, Warp's ``wp.e``) and used as
``e.f`` hashes that value until the function has been compiled in the process, and the local after. A second
material set's kernel (a second film look) was then named one way in a process that compiled the first set's kernel
and another way in a process that loaded it from the cache: a warm process compiled it again, about 120 s on an L4
and 10 s on a CPU. CPU; code generation only, no compiler run.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import warp as wp
from warp._src.types import is_value

import alloy.camera

REPO = Path(__file__).resolve().parents[2]


def _warp_functions(tree: ast.AST):
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and any("func" in ast.unparse(d) or "kernel" in ast.unparse(d)
                                                   for d in fn.decorator_list):
            yield fn


def _own_names(fn: ast.FunctionDef) -> set[str]:
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    return names | {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _outside_value(module, name: str) -> bool:
    """True when Warp's hasher, before the function is compiled, resolves ``name`` to a value: a module global, a
    ``warp`` attribute or a builtin (the order of ``Adjoint.resolve_path``)."""
    for space in (vars(module), vars(wp), vars(builtins)):
        if name in space:
            return is_value(space[name])
    return False


def _shadowed(module) -> list[str]:
    path = Path(module.__file__)
    found = []
    for fn in _warp_functions(ast.parse(path.read_text())):
        mine = _own_names(fn)
        for node in ast.walk(fn):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if root is not node and isinstance(root, ast.Name) and root.id in mine and _outside_value(module, root.id):
                found.append(f"{path.relative_to(REPO)}:{node.lineno} {fn.name}: {root.id}.")
    return sorted(set(found))


def test_no_camera_function_shadows_a_warp_value():
    found = []
    for info in pkgutil.walk_packages(alloy.camera.__path__, "alloy.camera."):
        if not info.name.endswith("__main__"):
            found += _shadowed(importlib.import_module(info.name))
    assert not found, "rename these locals (Warp hashes them as the outside value until compiled):\n" + "\n".join(found)


PROBE = """
import warp as wp
from warp._src.context import ModuleBuilder
from alloy.camera.pbr.bsdf import principled as pr

wp.init()


def declare():
    @wp.kernel(module="unique", enable_backward=False)
    def closures_probe(c: pr.PrincipledClosures, wi: wp.vec3, u: wp.vec3, out: wp.array(dtype=wp.vec3)):
        smp = pr.closures_sample(c, wi, u)
        val = pr.closures_eval(c, wi, smp.wo)
        out[0] = smp.weight + val.f
    return closures_probe


first = declare()
m = first.module
ModuleBuilder(m, m.resolve_options(wp.config, block_dim=256) | {"output_arch": None, "block_dim": 256})
print(first.module.name, declare().module.name)
"""


def test_a_kernel_is_named_alike_before_and_after_code_generation(tmp_path):
    """The same kernel declared again after code generation of the first (what a compile in this process runs) has
    the first one's name. In a fresh interpreter (an earlier test may have generated these functions already), from a
    file (Warp reads a kernel's source)."""
    probe = tmp_path / "kernel_names_probe.py"
    probe.write_text(PROBE)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(REPO), os.environ.get("PYTHONPATH"))))}
    out = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    before, after = out.stdout.split()[-2:]
    assert before == after, out.stdout
