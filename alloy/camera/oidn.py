# SPDX-License-Identifier: Apache-2.0
"""Intel Open Image Denoise 2.x (Apache-2.0) on CUDA, the denoiser Blender Cycles uses, over device arrays (ctypes,
no PyTorch). Cycles' settings: filter "RT", hdr, quality HIGH, prefilter ACCURATE (albedo and normal cleaned first,
then cleanAux on the colour filter). Inputs are linear (N, H, W) vec3 Warp arrays: the mean colour over samples, the
denoising albedo and the normal in camera space (x right, y up, +z forward), as ``Camera.render(output="all")`` gives.

The library is not bundled: set ``OIDN_LIB_DIR`` to a folder with ``libOpenImageDenoise.so.2`` (and its CUDA device
module), e.g. an official release from github.com/RenderKit/oidn/releases, or run
``python -m alloy.camera.oidn fetch`` to download release 2.3.3 into ``$ALLOY_CACHE/oidn``.

Example::

    from alloy.camera.oidn import OidnDenoiser
    dn = OidnDenoiser(128, 128)
    clean = dn.denoise(out.linear, out.albedo, out.normal)      # wp.array (N, H, W) vec3
"""
from __future__ import annotations

import ctypes as C
import glob
import logging
import os
from pathlib import Path

import warp as wp

from .errors import CameraError

log = logging.getLogger(__name__)
_DEVICE_TYPE_CUDA = 3
_FORMAT_FLOAT3 = 3
_QUALITY = {"fast": 4, "balanced": 5, "high": 6}
OIDN_VERSION = "2.3.3"
_LIB = None


def cache_dir() -> Path:
    """$ALLOY_CACHE/oidn (default ~/.cache/alloy/oidn)."""
    return Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy")) / "oidn"


def library_dirs() -> list[Path]:
    """Folders searched for libOpenImageDenoise.so.2, in order."""
    dirs = [os.environ.get("OIDN_LIB_DIR", "")] + glob.glob(str(cache_dir() / "*" / "lib"))
    dirs += ["/usr/local/lib", "/usr/lib/x86_64-linux-gnu"] + sorted(glob.glob("/opt/blender*/lib"), reverse=True)
    return [Path(d) for d in dirs if d and (Path(d) / "libOpenImageDenoise.so.2").exists()]


def available() -> tuple[bool, str]:
    """(True, folder) when the OIDN library is found, else (False, what to do)."""
    dirs = library_dirs()
    if dirs:
        return True, str(dirs[0])
    return False, ("Open Image Denoise not found: set OIDN_LIB_DIR to a folder with libOpenImageDenoise.so.2, or run "
                   "python -m alloy.camera.oidn fetch")


_SIG = {
    "oidnNewCUDADevice": ("vp", ["ip", "vpp", "i"]), "oidnCommitDevice": (None, ["vp"]),
    "oidnReleaseDevice": (None, ["vp"]), "oidnSyncDevice": (None, ["vp"]), "oidnGetDeviceError": ("i", ["vp", "cpp"]),
    "oidnNewSharedBuffer": ("vp", ["vp", "vp", "sz"]), "oidnReleaseBuffer": (None, ["vp"]),
    "oidnNewFilter": ("vp", ["vp", "cp"]), "oidnSetFilterImage": (None, ["vp", "cp", "vp", "i"] + ["sz"] * 5),
    "oidnSetFilterBool": (None, ["vp", "cp", "b"]), "oidnSetFilterInt": (None, ["vp", "cp", "i"]),
    "oidnCommitFilter": (None, ["vp"]), "oidnExecuteFilterAsync": (None, ["vp"]), "oidnReleaseFilter": (None, ["vp"]),
}
_CT = {"vp": C.c_void_p, "cp": C.c_char_p, "i": C.c_int, "sz": C.c_size_t, "b": C.c_bool,
       "ip": C.POINTER(C.c_int), "vpp": C.POINTER(C.c_void_p), "cpp": C.POINTER(C.c_char_p)}


def _load() -> C.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    ok, where = available()
    if not ok:
        raise CameraError("the OIDN denoiser library is missing", where.split(": ", 1)[1])
    d = Path(where)
    for dep in sorted(d.glob("libtbb.so*"))[:1]:
        C.CDLL(str(dep), mode=C.RTLD_GLOBAL)
    lib = C.CDLL(str(d / "libOpenImageDenoise.so.2"), mode=C.RTLD_GLOBAL)
    for name, (res, args) in _SIG.items():
        f = getattr(lib, name)
        f.restype, f.argtypes = (_CT[res] if res else None), [_CT[a] for a in args]
    _LIB = lib
    return lib


@wp.kernel
def _flip_rows(src: wp.array3d(dtype=wp.vec3), dst: wp.array3d(dtype=wp.vec3)):
    """dst = src upside down (Cycles hands OIDN bottom-up images; OIDN is not flip-symmetric at the borders)."""
    n, y, x = wp.tid()
    dst[n, src.shape[1] - 1 - y, x] = src[n, y, x]


class _Chain:
    """One OIDN CUDA device on one stream, with the albedo, normal and colour filters."""

    def __init__(self, quality: str, stream: int, device_id: int):
        self.lib = lib = _load()
        self.dev = lib.oidnNewCUDADevice((C.c_int * 1)(device_id), (C.c_void_p * 1)(stream), 1)
        if not self.dev:
            raise CameraError("OIDN could not open the CUDA device", "check the GPU driver and OIDN's CUDA module")
        lib.oidnCommitDevice(self.dev)
        self.check()
        self.filters = [self._filter(quality) for _ in range(3)]  # albedo, normal, colour
        main = self.filters[2]
        lib.oidnSetFilterBool(main, b"hdr", True)
        lib.oidnSetFilterBool(main, b"srgb", False)
        lib.oidnSetFilterBool(main, b"cleanAux", True)
        self.bufs: dict = {}

    def _filter(self, quality: str):
        f = self.lib.oidnNewFilter(self.dev, b"RT")
        if not f:
            self.check()
        self.lib.oidnSetFilterInt(f, b"quality", _QUALITY[quality])
        return f

    def check(self) -> None:
        msg = C.c_char_p()
        if self.lib.oidnGetDeviceError(self.dev, C.byref(msg)) != 0:
            raise CameraError(f"OIDN: {msg.value.decode() if msg.value else 'unspecified error'}",
                              "check the inputs (float32 (N, H, W, 3)) and the OIDN version (2.x)")

    def _bind(self, filt, name: bytes, ptr: int, nbytes: int, off: int, w: int, h: int) -> None:
        buf = self.bufs.get((ptr, nbytes))
        if buf is None:
            buf = self.bufs[(ptr, nbytes)] = self.lib.oidnNewSharedBuffer(self.dev, C.c_void_p(ptr), nbytes)
        self.lib.oidnSetFilterImage(filt, name, buf, _FORMAT_FLOAT3, w, h, off, 12, 12 * w)

    def run(self, w: int, h: int, ptrs: list[int], nbytes: int, off: int) -> None:
        """ptrs = (colour, albedo, normal, out, clean albedo, clean normal); one image at byte offset ``off``."""
        c, a, n, o, sa, sn = ptrs
        for f, nm, src, dst in ((self.filters[0], b"albedo", a, sa), (self.filters[1], b"normal", n, sn)):
            self._bind(f, nm, src, nbytes, off, w, h)
            self._bind(f, b"output", dst, nbytes, off, w, h)
            self.lib.oidnCommitFilter(f)
            self.lib.oidnExecuteFilterAsync(f)
        for nm, p in ((b"color", c), (b"albedo", sa), (b"normal", sn), (b"output", o)):
            self._bind(self.filters[2], nm, p, nbytes, off, w, h)
        self.lib.oidnCommitFilter(self.filters[2])
        self.lib.oidnExecuteFilterAsync(self.filters[2])

    def sync(self) -> None:
        self.lib.oidnSyncDevice(self.dev)
        self.check()

    def close(self) -> None:
        for b in self.bufs.values():
            self.lib.oidnReleaseBuffer(b)
        for f in self.filters:
            self.lib.oidnReleaseFilter(f)
        self.lib.oidnReleaseDevice(self.dev)
        self.bufs, self.filters = {}, []


class OidnDenoiser:
    """Batched OIDN on the GPU: ``n_streams`` OIDN devices, images shared out round-robin."""

    def __init__(self, width: int, height: int, quality: str = "high", device: str = "cuda:0", n_streams: int = 4):
        if quality not in _QUALITY:
            raise CameraError(f"OIDN quality {quality!r}", f"use one of {sorted(_QUALITY)}")
        self.w, self.h, self.device = int(width), int(height), wp.get_device(device)
        if not self.device.is_cuda:
            raise CameraError("OIDN runs here on CUDA only", "render on a CUDA device, or use denoise=None")
        self.streams = [wp.Stream(self.device) for _ in range(max(1, int(n_streams)))]
        self.chains = [_Chain(quality, s.cuda_stream, self.device.ordinal) for s in self.streams]
        self._scratch: dict = {}

    def _buf(self, key: str, n: int) -> wp.array:
        a = self._scratch.get(key)
        if a is None or a.shape[0] != n:
            a = self._scratch[key] = wp.empty((n, self.h, self.w), dtype=wp.vec3, device=self.device)
        return a

    def denoise(self, rgb: wp.array, albedo: wp.array, normal: wp.array, out: wp.array | None = None) -> wp.array:
        """Denoised linear colour, (N, H, W) vec3 like the inputs (rows top first)."""
        n = rgb.shape[0]
        if tuple(rgb.shape) != (n, self.h, self.w):
            raise CameraError(f"OIDN input shape {tuple(rgb.shape)}", f"pass (N, {self.h}, {self.w}) vec3 arrays")
        flipped = []
        for k, a in enumerate((rgb, albedo, normal)):
            flipped.append(self._buf(f"in{k}", n))
            wp.launch(_flip_rows, dim=(n, self.h, self.w), inputs=[a], outputs=[flipped[-1]], device=self.device)
        res, ca, cn = self._buf("res", n), self._buf("ca", n), self._buf("cn", n)
        ev = wp.get_stream(self.device).record_event()
        for s in self.streams:
            s.wait_event(ev)
        ptrs = [int(a.ptr) for a in (*flipped, res, ca, cn)]
        img = self.h * self.w * 12
        for i in range(n):
            self.chains[i % len(self.chains)].run(self.w, self.h, ptrs, n * img, i * img)
        for c in self.chains:
            c.sync()
        out = out if out is not None else wp.empty((n, self.h, self.w), dtype=wp.vec3, device=self.device)
        wp.launch(_flip_rows, dim=(n, self.h, self.w), inputs=[res], outputs=[out], device=self.device)
        return out

    def close(self) -> None:
        for c in self.chains:
            c.close()
        self.chains = []


def fetch(version: str = OIDN_VERSION) -> Path:
    """Download the official Linux release of OIDN (Apache-2.0, with its CUDA module) into the cache; returns lib dir."""
    import tarfile
    import urllib.request
    name = f"oidn-{version}.x86_64.linux"
    dest = cache_dir()
    lib = dest / name / "lib"
    if (lib / "libOpenImageDenoise.so.2").exists():
        return lib
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/RenderKit/oidn/releases/download/v{version}/{name}.tar.gz"
    tgz = dest / f"{name}.tar.gz"
    log.info("downloading %s", url)
    urllib.request.urlretrieve(url, tgz)  # noqa: S310 (fixed https URL)
    with tarfile.open(tgz) as tf:
        tf.extractall(dest, filter="data")
    tgz.unlink()
    return lib


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] == ["fetch"]:
        print(json.dumps({"oidn_lib_dir": str(fetch())}))
    else:
        ok, where = available()
        print(json.dumps({"available": ok, "detail": where}))
