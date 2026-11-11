# SPDX-License-Identifier: Apache-2.0
"""How ``alloy.route._mjc`` compiles on first use, per platform (SCR-OSS1-8): macOS arm64 builds a ``.dylib`` that
takes MuJoCo's symbols from the libmujoco ``import mujoco`` loaded, and a Mac without Apple clang (``/usr/bin/cc`` is
then a shim that prints an ``xcrun: error``) gives no library and a reason naming the compiler. Unit level: the
compiler and the link line, no real macOS toolchain (the release CI's macOS legs build and load it for real)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import alloy.route._mjc as mjc
import alloy.route._mjc.build as mjc_build
import alloy.route.mjc_core as mc
import alloy.route.mjc_core_cabi as cabi

pytestmark = pytest.mark.cpu_route

XCRUN = ("xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun at: "
         "/Library/Developer/CommandLineTools/usr/bin/xcrun\n")


def test_macos_links_without_the_wheels_framework_install_name():
    """The mujoco wheel's dylib names itself ``@rpath/mujoco.framework/...``, which the wheel lacks: on macOS the
    library is not linked to it (no path, no rpath) and leaves MuJoCo's symbols to the load. Linux links it."""
    mac_lib = Path("/site/mujoco/libmujoco.3.16.0.dylib")
    mac = mjc_build.link_flags(mac_lib, "darwin")
    assert mac[:3] == ["-dynamiclib", "-undefined", "dynamic_lookup"] and "-shared" not in mac
    assert not any(str(mac_lib.parent) in f for f in mac), mac
    lin_lib = Path("/site/mujoco/libmujoco.so.3.16.0")
    lin = mjc_build.link_flags(lin_lib, "linux")
    assert lin[:2] == ["-shared", str(lin_lib)] and f"-Wl,-rpath,{lin_lib.parent}" in lin


def test_compile_line_ends_with_the_platform_link_flags(monkeypatch, tmp_path):
    seen: list[list[str]] = []
    monkeypatch.setattr(mjc_build, "link_flags", lambda lib: ["-dynamiclib", "-undefined", "dynamic_lookup", "-lm"])
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    mjc_build.compile_to(tmp_path / "lib.dylib", "/usr/bin/cc", tmp_path, tmp_path / "libmujoco.dylib")
    cmd = seen[0]
    assert cmd[0] == "/usr/bin/cc" and cmd[-6:] == ["-dynamiclib", "-undefined", "dynamic_lookup", "-lm", "-o",
                                                    str(tmp_path / "lib.dylib")]
    assert all(s.endswith(".c") for s in cmd[cmd.index(f"-I{mjc_build.HERE}") + 1:-6])


def test_a_build_with_linker_warnings_logs_them(monkeypatch, tmp_path, caplog):
    """A build that succeeds with messages (Apple's ld on ``-undefined dynamic_lookup`` and chained fixups) logs
    them, so release CI's macOS log shows them; a quiet build logs nothing."""
    warn = "ld: warning: -undefined dynamic_lookup may not work with chained fixups\n"
    for err in (warn, ""):
        monkeypatch.setattr(subprocess, "run", lambda cmd, err=err, **kw: subprocess.CompletedProcess(cmd, 0, "", err))
        mjc_build.compile_to(tmp_path / "lib.dylib", "/usr/bin/cc", tmp_path, tmp_path / "libmujoco.dylib")
    msgs = [r.getMessage() for r in caplog.records if r.name == "alloy.route" and r.levelname == "WARNING"]
    assert len(msgs) == 1 and "chained fixups" in msgs[0] and "/usr/bin/cc" in msgs[0], msgs


def _shim(monkeypatch, rc: int, out: str, err: str) -> None:
    """``cc`` answers ``--version`` with (rc, out, err). It is outside /usr/bin, so on a Mac too it is run (as a CC
    stub is in OSS-1's CI leg); /usr/bin/cc on a Mac is the next test's."""
    monkeypatch.setattr(mjc_build.shutil, "which", lambda name: "/usr/local/bin/cc")
    monkeypatch.setattr(mjc_build.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err))


def test_a_compiler_that_does_not_run_is_no_compiler(monkeypatch):
    """``cc`` on the PATH is not enough: ``cc --version`` must succeed (the macOS shim exits 1 with ``xcrun``)."""
    _shim(monkeypatch, 1, "", XCRUN)
    with pytest.raises(mjc_build.BuildError, match=r"no C compiler: /usr/local/bin/cc --version failed \(xcrun: error"):
        mjc_build.compiler()
    _shim(monkeypatch, 0, "Apple clang version 16.0.0 (clang-1600.0.26.6)\nTarget: arm64-apple-darwin24.1.0\n", "")
    assert mjc_build.compiler() == ("/usr/local/bin/cc", "Apple clang version 16.0.0 (clang-1600.0.26.6)")


def test_the_darwin_shim_is_not_run_without_the_tools(monkeypatch):
    """``/usr/bin/cc`` on macOS: ``xcode-select -p`` is asked first (running the shim can open the installer dialog);
    without the tools the shim never runs. A CC elsewhere (OSS-1's CI stub) is run as on Linux."""
    ran: list[list[str]] = []

    def run(cmd, **kw):
        ran.append(cmd)
        return subprocess.CompletedProcess(cmd, 2 if cmd[0] == "xcode-select" else 0, stdout="cc 1\n", stderr="")
    monkeypatch.setattr(mjc_build.sys, "platform", "darwin")
    monkeypatch.setattr(mjc_build.shutil, "which", lambda name: "/usr/bin/cc")
    monkeypatch.setattr(mjc_build.subprocess, "run", run)
    with pytest.raises(mjc_build.BuildError, match="Command Line Tools are not installed"):
        mjc_build.compiler()
    assert ran == [["xcode-select", "-p"]]
    monkeypatch.setattr(mjc_build.shutil, "which", lambda name: "/opt/homebrew/bin/gcc-14")
    assert mjc_build.compiler() == ("/opt/homebrew/bin/gcc-14", "cc 1")
    assert ran[-1] == ["/opt/homebrew/bin/gcc-14", "--version"]


def test_a_darwin_link_change_rebuilds(monkeypatch):
    """The macOS key carries the link flags: a library cached from an older link (to the framework install name,
    which never loads) is not reused. Linux keys do not change. The Linux key is built as on Linux, so the test holds on
    a Mac too (the first real-Mac run built it with darwin and it equalled the mac key)."""
    monkeypatch.setattr(mjc_build.sys, "platform", "linux")
    linux = mjc_build.build_key("3.16.0", "cc 1")
    monkeypatch.setattr(mjc_build.sys, "platform", "darwin")
    mac = mjc_build.build_key("3.16.0", "cc 1")
    monkeypatch.setattr(mjc_build, "link_flags", lambda lib, name: ["-dynamiclib"])
    assert len({linux, mac, mjc_build.build_key("3.16.0", "cc 1")}) == 3


@pytest.mark.parametrize("value, off", [("1", True), ("yes", True), ("", False), ("0", False), ("false", False),
                                        ("off", False), ("No", False)])
def test_alloy_no_ext_values(monkeypatch, value, off):
    monkeypatch.setenv(mjc.NO_EXT_ENV, value)
    assert bool(mjc.switched_off()) is off


def test_without_apple_clang_load_returns_none_with_the_reason(monkeypatch, tmp_path):
    """The loader returns None and says why; the route's warning tells a Mac user to install the tools."""
    for name, value in (("_LIB", None), ("_TRIED", False), ("reason", "")):
        monkeypatch.setattr(mjc, name, value)
    monkeypatch.setattr(cabi, "_FAILURE", "")
    monkeypatch.delenv(mjc.NO_EXT_ENV, raising=False)
    monkeypatch.setattr(mjc_build, "HERE", tmp_path)                       # no prebuilt library
    monkeypatch.setattr(mjc_build, "cache_root", lambda: tmp_path / "cache")
    _shim(monkeypatch, 1, "", XCRUN)
    assert mjc.load() is None and mjc.reason.startswith("no C compiler") and "xcrun" in mjc.reason
    assert "xcode-select --install" in mc.fallback_fix(mjc.reason, "darwin")
