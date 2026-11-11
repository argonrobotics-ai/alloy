# SPDX-License-Identifier: Apache-2.0
"""``new_plugin``: a new plug-in package from the templates in ``alloy.plugins.templates``, names filled in.

The plug-in code comes from the template module of its kind (a package resource, so it works from an installed
wheel as well as from a source checkout); the small project files (pyproject, README, package init, test) are
written from the texts below.
"""

from __future__ import annotations

import keyword
import logging
import re
import sys
from importlib import resources
from pathlib import Path

from alloy.plugins.errors import PluginError

log = logging.getLogger("alloy.plugins")

KINDS = ("force", "sensor", "object", "solver")
TEMPLATES = "alloy.plugins.templates"

PYPROJECT = '''[project]
name = "alloy-plugin-plugin_name"
version = "0.1.0"
description = "An Alloy plug-in: plugin_name"
requires-python = ">=3.11"
dependencies = ["alloy-engine"]

# Alloy finds the plug-in through this entry point after `pip install -e .`:
#   python -m alloy.plugins list            -> shows plugin_name
#   python -m alloy.plugins check plugin_name
[project.entry-points."alloy.plugins"]
plugin_name = "plugin_name.plugin:PluginName"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["plugin_name"]
'''

README = '''# plugin_name

An Alloy plug-in, made with `python -m alloy.plugins new plugin_name --kind <kind>`. The code in
`plugin_name/plugin.py` already passes the harness. Change the physics, keep the structure.

```bash
python -m alloy.plugins check plugin_name/plugin.py --quick    # crash + laws, while iterating
python -m alloy.plugins check plugin_name/plugin.py --out report.json   # every check, JSON report (exit 0 = pass)
python -m alloy.plugins check plugin_name/plugin.py --sandbox  # the same inside the OS sandbox
pip install -e . && python -m alloy.plugins list               # registered through the entry point
pytest tests/
```

The API, the rules every kernel follows and what each check means: `alloy/plugins/GUIDE.md` in the Alloy package
(`python -c "import alloy.plugins, pathlib; print(pathlib.Path(alloy.plugins.__file__).parent / 'GUIDE.md')"`).
'''

PACKAGE_INIT = '''"""plugin_name: an Alloy plug-in (made by ``python -m alloy.plugins new``)."""

from plugin_name.plugin import PluginName

__all__ = ["PluginName"]
'''

TEST = '''"""The plug-in passes Alloy's harness: laws, crash, determinism, world independence, speed."""

from pathlib import Path

import alloy.plugins as ap

PLUGIN = Path(__file__).resolve().parents[1] / "plugin_name" / "plugin.py"


def test_harness_passes():
    report = ap.check(str(PLUGIN))
    assert report.passed, report.summary()
'''


def template_source(kind: str) -> str:
    """The template module of ``kind`` as text. Example: ``"class PluginName" in template_source("force")``."""
    if kind not in KINDS:
        raise PluginError(f"kind {kind!r} is not a plug-in kind", fix=f"use one of {KINDS}")
    try:
        return resources.files(TEMPLATES).joinpath(f"{kind}.py").read_text()
    except (FileNotFoundError, ModuleNotFoundError) as err:
        raise PluginError(f"template {TEMPLATES}.{kind} is missing from this Alloy install ({err}): "
                          f"reinstall alloy-engine") from err


def camel(name: str) -> str:
    """``"my_drag"`` -> ``"MyDrag"``."""
    return "".join(part.capitalize() for part in name.split("_"))


def new_plugin(name: str, *, kind: str = "force", dest: Path = Path(".")) -> list[Path]:
    """Create ``dest/<name>/`` (pyproject, README, ``<name>/plugin.py`` from the ``kind`` template, a test) and
    return the files, the plug-in module first. Example: ``new_plugin("my_drag", kind="force", dest=Path("."))``."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or keyword.iskeyword(name):
        raise PluginError(f"plug-in name {name!r} is not snake_case",
                          fix="use lowercase letters, digits and underscores, starting with a letter: my_force")
    if name in sys.stdlib_module_names or name in ("alloy", "warp", "numpy"):
        raise PluginError(f"plug-in name {name!r} is also a Python module, which its package would hide",
                          fix=f"pick another name, e.g. my_{name}")
    code = template_source(kind)
    root = Path(dest) / name
    if root.exists():
        raise PluginError(f"{root} already exists", fix="pick another name or another --dir")
    plan = {
        root / name / "plugin.py": code,
        root / name / "__init__.py": PACKAGE_INIT,
        root / "pyproject.toml": PYPROJECT,
        root / "README.md": README,
        root / "tests" / f"test_{name}.py": TEST,
    }
    for out, text in plan.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        text = text.replace("PluginName", camel(name)).replace("plugin_name", name)
        out.write_text(text.replace("<kind>", kind))
    log.info("created plug-in %s (%s) in %s", name, kind, root)
    return list(plan)


__all__ = ["KINDS", "camel", "new_plugin", "template_source"]
