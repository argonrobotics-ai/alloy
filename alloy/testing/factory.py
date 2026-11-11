# SPDX-License-Identifier: Apache-2.0
"""Factory test helpers (WP7, D17): the scripted fake ``AgentRunner`` that copies the template plug-in.

``CopyTemplateRunner`` plays the agent: it copies ``alloy/factory/template/`` from the scratch worktree into the
output package and renames it (``PLUGIN_NAME`` and the manifest). With ``stray_write`` it also writes a file outside
its package, which the loop must revert and report; with ``commit_outside`` it edits and commits a product file
(the evaluation must not see it); with ``template_manifest`` it leaves the manifest pointing at the trusted template.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("alloy.factory")

TEMPLATE = Path("alloy/factory/template")
STRAY = Path("alloy/bully/agent_was_here.py")
TAMPERED = Path("alloy/factory/evaluate.py")
TAMPER_MARK = "# tampered by the agent"


def _commit_tamper(workdir: Path) -> None:
    """Append a marker to a product file and commit it (a sandbox escape the working-tree diff alone would miss)."""
    target = workdir / TAMPERED
    target.write_text(target.read_text() + f"\n{TAMPER_MARK}\n")
    for args in (["add", str(TAMPERED)], ["-c", "user.name=agent", "-c", "user.email=agent@example.invalid", "commit",
                                          "-qm", "agent edit"]):
        subprocess.run(["git", *args], cwd=workdir, check=True, capture_output=True)


@dataclass
class CopyTemplateRunner:
    """Scripted ``AgentRunner``: copy + rename the template; optionally misbehave once."""

    command_template: str = "copy-template {workdir} {out}"
    stray_write: bool = False
    commit_outside: bool = False
    template_manifest: bool = False
    exit_code: int = 0

    def run(self, brief: Path, workdir: Path, out_package: Path, timeout_s: float) -> int:
        del timeout_s
        name = out_package.name
        if not brief.is_file():
            raise FileNotFoundError(brief)
        shutil.copytree(workdir / TEMPLATE, out_package, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        solver = out_package / "solver.py"
        solver.write_text(re.sub(r'^PLUGIN_NAME = "template"', f'PLUGIN_NAME = "{name}"', solver.read_text(),
                                 count=1, flags=re.M))
        rel = out_package.relative_to(workdir)
        manifest = {"name": name, "module": ".".join(rel.parts), "routes": [f"contrib:{name}"],
                    "laws": [f"FAC.{name}.freefall"], "bully_scenes": [f"{name}_drop"], "author": "copy-template"}
        if self.template_manifest:
            manifest.update(name="template", module="alloy.factory.template")
        (out_package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if self.stray_write:
            (workdir / STRAY).write_text("# written outside the plug-in package\n")
        if self.commit_outside:
            _commit_tamper(workdir)
        log.info("copy-template runner: wrote %s", rel)
        return self.exit_code
