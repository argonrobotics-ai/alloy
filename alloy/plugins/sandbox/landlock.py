# SPDX-License-Identifier: Apache-2.0
"""Linux Landlock through ctypes: restrict the calling process (and its children) to an allow-list of paths, and
(Landlock ABI >= 4, Linux 6.7+) forbid TCP bind and connect. Unprivileged; irreversible for the process.

Example::

    from alloy.plugins.sandbox.landlock import abi_version, restrict_self
    if abi_version() > 0:
        restrict_self(read=["/usr", sys.prefix], write=["/tmp/my-sandbox"], deny_tcp=True)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger("alloy.plugins")

_SYS_CREATE, _SYS_ADD_RULE, _SYS_RESTRICT = 444, 445, 446          # same numbers on x86_64 and aarch64
_CREATE_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REMOVE_DIR, REMOVE_FILE, MAKE_CHAR, MAKE_DIR = 1 << 4, 1 << 5, 1 << 6, 1 << 7
MAKE_REG, MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK, MAKE_SYM = 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12
REFER, TRUNCATE = 1 << 13, 1 << 14
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1

READ = EXECUTE | READ_FILE | READ_DIR
WRITE = WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG | MAKE_SOCK | MAKE_FIFO | MAKE_SYM


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def abi_version() -> int:
    """Landlock ABI of the running kernel; 0 if unavailable (not Linux, disabled, or an unsupported CPU)."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        return 0
    v = _libc().syscall(_SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(_CREATE_VERSION))
    return max(int(v), 0)


def _fs_mask(abi: int) -> int:
    mask = READ | WRITE | MAKE_CHAR | MAKE_BLOCK
    if abi >= 2:
        mask |= REFER
    if abi >= 3:
        mask |= TRUNCATE
    return mask


def _add_path(libc: ctypes.CDLL, ruleset: int, path: str, access: int) -> None:
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        if Path(path).is_file():
            access &= EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE      # file rules take file rights only
        attr = _PathBeneath(access, fd)
        if libc.syscall(_SYS_ADD_RULE, ctypes.c_int(ruleset), ctypes.c_int(_RULE_PATH_BENEATH),
                        ctypes.byref(attr), ctypes.c_uint32(0)) != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
    finally:
        os.close(fd)


def restrict_self(read: Iterable[str], write: Iterable[str], *, deny_tcp: bool = True) -> dict[str, object]:
    """Allow reading (and executing) under ``read``, reading and writing under ``write``, nothing else; with
    ``deny_tcp`` and ABI >= 4 also no TCP bind/connect. Missing paths are skipped. Returns what was enforced."""
    abi = abi_version()
    if abi < 1:
        raise OSError("Landlock is not available on this kernel")
    libc = _libc()
    fs = _fs_mask(abi)
    net = (NET_BIND_TCP | NET_CONNECT_TCP) if deny_tcp and abi >= 4 else 0
    attr = _RulesetAttr(fs, net)
    size = ctypes.sizeof(attr) if abi >= 4 else ctypes.sizeof(ctypes.c_uint64)
    ruleset = libc.syscall(_SYS_CREATE, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        rules = [(p, READ & fs) for p in read] + [(p, (READ | WRITE | TRUNCATE * (abi >= 3)) & fs) for p in write]
        added = []
        for path, access in rules:
            if os.path.exists(path):
                _add_path(libc, ruleset, path, access)
                added.append(path)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
        if libc.syscall(_SYS_RESTRICT, ctypes.c_int(ruleset), ctypes.c_uint32(0)) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset)
    log.info("landlock ABI %d enforced: %d paths, tcp %s", abi, len(added), "denied" if net else "allowed")
    return {"abi": abi, "paths": added, "tcp_denied": bool(net)}


__all__ = ["abi_version", "restrict_self"]
